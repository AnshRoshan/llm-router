"""Tests for the streaming execution path: SSE parsing, usage learning,
and the fail-before-first-token rule."""
import asyncio
import json
from contextlib import asynccontextmanager

import pytest

from llmrouter import Message, Router, StreamChunk
from llmrouter.adapters import OpenAIAdapter, AnthropicAdapter, parse_sse_line


def sse(obj):
    return f"data: {json.dumps(obj)}"


ANTHROPIC_LINES = [
    sse({"type": "message_start", "message": {
        "model": "claude-sonnet-class",
        "usage": {"input_tokens": 12, "cache_read_input_tokens": 4000,
                  "cache_creation_input_tokens": 200}}}),
    sse({"type": "content_block_start", "index": 0,
         "content_block": {"type": "text", "text": ""}}),
    sse({"type": "content_block_delta", "index": 0,
         "delta": {"type": "text_delta", "text": "Hello"}}),
    "event: ping",
    sse({"type": "content_block_delta", "index": 0,
         "delta": {"type": "text_delta", "text": " world"}}),
    sse({"type": "message_delta", "usage": {"output_tokens": 9}}),
    sse({"type": "message_stop"}),
]

OPENAI_LINES = [
    sse({"choices": [{"delta": {"role": "assistant"}}]}),
    sse({"choices": [{"delta": {"content": "Hi"}}]}),
    sse({"choices": [{"delta": {"content": " there"}}]}),
    sse({"usage": {"prompt_tokens": 20, "completion_tokens": 4,
                   "prompt_tokens_details": {"cached_tokens": 15}}}),
    "data: [DONE]",
]


class StreamResponse:
    def __init__(self, lines, status=200):
        self.lines, self.status_code = lines, status

    async def aiter_lines(self):
        for line in self.lines:
            yield line


class FakeStreamingClient:
    """Serves `lines` per URL substring; raises for substrings in `fail`."""

    def __init__(self, lines, fail=()):
        self.lines, self.fail = lines, tuple(fail)
        self.urls = []
        self.bodies = []

    async def post(self, url, **kw):  # unused on the stream path
        raise AssertionError("acomplete should not be used here")

    def stream(self, method, url, *, headers=None, json=None, timeout=None):
        self.urls.append(url)
        self.bodies.append(json)
        @asynccontextmanager
        async def cm():
            if any(f in url for f in self.fail):
                raise ConnectionError(f"reset by {url.split('//')[1][:12]}")
            yield StreamResponse(self.lines)
        return cm()


def _router(client, **kw):
    return Router.with_defaults(anthropic_key="k", openai_key="k",
                                transport=client, **kw)


async def _collect(agen):
    deltas, final = [], None
    async for chunk in agen:
        assert isinstance(chunk, StreamChunk)
        if chunk.completion is not None:
            final = chunk.completion
        else:
            deltas.append(chunk.text)
    return deltas, final


# --------------------------------------------------------------------------- #
def test_parse_sse_line_variants():
    assert parse_sse_line("data: [DONE]") is None
    assert parse_sse_line(": comment") is None
    assert parse_sse_line("") is None
    assert parse_sse_line("event: ping") is None
    assert parse_sse_line('data: {"a":1}') == {"a": 1}


def test_anthropic_stream_yields_and_learns():
    client = FakeStreamingClient(ANTHROPIC_LINES)
    # Anthropic-only fleet: the decision cannot fall to an OpenAI model, so
    # the anthropic adapter is guaranteed to be the one streaming.
    router = Router.with_defaults(anthropic_key="k", transport=client)

    async def run():
        return await _collect(router.astream(
            [Message("user", "hello friend " * 200)],
            workload="agent", session_id="s1"))

    deltas, final = asyncio.run(run())
    assert "".join(deltas) == "Hello world" == final.text
    u = final.usage
    assert (u.input_tokens, u.output_tokens, u.cache_read_tokens,
            u.cache_write_tokens) == (12, 9, 4000, 200)
    assert final.model == "claude-sonnet-class"
    # body carried stream=true and the cache breakpoints survived
    assert client.bodies[0]["stream"] is True
    # learning loop ran: session exists with a learned hit rate
    sess = router.sessions.peek("sid:s1")
    assert sess.pinned_model == router.last_decision.model_id
    assert sess.observed_cache_hit_rate == pytest.approx(4000 / 4200)


def test_openai_stream_adds_usage_and_parses():
    client = FakeStreamingClient(OPENAI_LINES)
    router = _router(client)
    deltas, final = asyncio.run(_collect(router.astream(
        [Message("user", "hi there friend")], workload="chat")))
    assert "".join(deltas) == "Hi there" == final.text
    u = final.usage
    assert (u.input_tokens, u.output_tokens, u.cache_read_tokens,
            u.cache_write_tokens) == (20, 4, 15, 5)
    assert client.bodies[0]["stream_options"] == {"include_usage": True}


def test_backend_failover_before_first_token():
    # Two backends serve the claude models; the primary's URL is broken, so
    # the stream must retry on the secondary BEFORE any token reaches the
    # caller — the streaming twin of the unary failover loop.
    from llmrouter import BackendRegistry, BackendSpec, PriceRegistry
    claude = tuple(m for m in PriceRegistry().ids() if m.startswith("claude"))
    router = Router(
        prices=PriceRegistry(),
        backends=BackendRegistry([
            BackendSpec("anthropic", models=claude,
                        base_url="https://api.anthropic.com", priority=0),
            BackendSpec("claude-proxy", models=claude,
                        base_url="https://proxy.example", priority=1),
        ]),
        transport=FakeStreamingClient(ANTHROPIC_LINES, fail=("api.anthropic",)),
    )
    deltas, final = asyncio.run(_collect(router.astream(
        [Message("user", "hello friend " * 200)], workload="chat")))
    assert final.text == "Hello world"
    assert final.backend_id == "claude-proxy"
    assert any("failed over to claude-proxy" in r
               for r in router.last_decision.reasons)


def test_no_retry_after_first_token():
    class HalfThenBreak(FakeStreamingClient):
        def stream(self, method, url, **kw):
            parent = self

            @asynccontextmanager
            async def cm():
                async def broken_lines():
                    yield ANTHROPIC_LINES[0]
                    yield ANTHROPIC_LINES[2]      # one delta delivered...
                    raise ConnectionError("mid-stream reset")
                class Resp(StreamResponse):
                    async def aiter_lines(self):
                        async for l in broken_lines():
                            yield l
                yield Resp([])
            return cm()

    client = HalfThenBreak(ANTHROPIC_LINES)
    router = _router(client)

    async def run():
        got = []
        with pytest.raises(Exception) as ei:
            async for chunk in router.astream(
                    [Message("user", "hi friend " * 200)], workload="agent"):
                if chunk.text:
                    got.append(chunk.text)
        assert "after the first token" in str(ei.value)
        assert got == ["Hello"]          # exactly one token delivered, once

    asyncio.run(run())


def test_astream_without_transport_fails_fast():
    router = Router.pure()
    with pytest.raises(Exception):  # ConfigurationError, before any iteration
        async def consume():
            async for _ in router.astream([Message("user", "hi")]):
                pass
        asyncio.run(consume())


def test_adapters_parse_stream_event_ties():
    a = AnthropicAdapter()
    ev = a.parse_stream_event(json.loads(ANTHROPIC_LINES[4][6:]))
    assert ev.text_delta == " world"
    o = OpenAIAdapter()
    assert o.parse_stream_event({"choices": [{"delta": {}}]}).text_delta == ""
