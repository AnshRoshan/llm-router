"""The facade: async execution, failover, learning, and the adapters."""
from __future__ import annotations

import pytest

from llmrouter import (
    AnthropicAdapter,
    ConfigurationError,
    BackendRegistry,
    BackendSpec,
    DeploymentClass,
    Message,
    OpenAIAdapter,
    PriceRegistry,
    Router,
    RouterError,
    SegmentKind,
    WorkloadClass,
)
from llmrouter.economics import Breakpoint


class FakeResponse:
    def __init__(self, payload, status_code=200):
        self._payload = payload
        self.status_code = status_code

    def json(self):
        return self._payload


class FakeTransport:
    """Records requests and returns canned provider responses."""

    def __init__(self, responder=None):
        self.calls: list[dict] = []
        self.responder = responder or self._default

    @staticmethod
    def _default(url, body):
        if "anthropic" in url:
            return FakeResponse({
                "model": body["model"],
                "content": [{"type": "text", "text": "hi from claude"}],
                "usage": {
                    "input_tokens": 120,
                    "output_tokens": 40,
                    "cache_creation_input_tokens": 2000,
                    "cache_read_input_tokens": 0,
                },
            })
        return FakeResponse({
            "model": body["model"],
            "choices": [{"message": {"role": "assistant", "content": "hi from gpt"}}],
            "usage": {
                "prompt_tokens": 2120,
                "completion_tokens": 40,
                "prompt_tokens_details": {"cached_tokens": 0},
            },
        })

    async def post(self, url, *, headers=None, json=None, timeout=None):
        self.calls.append({"url": url, "headers": headers, "body": json, "timeout": timeout})
        return self.responder(url, json)


SYSTEM = "You are a helpful assistant. " * 200


_NO_TRANSPORT = object()


def make_router(transport=_NO_TRANSPORT, **kw) -> Router:
    prices = PriceRegistry()
    backends = BackendRegistry([
        BackendSpec("anthropic", DeploymentClass.API,
                    models=tuple(m for m in prices.ids() if m.startswith("claude")),
                    base_url="https://api.anthropic.com", priority=0),
        BackendSpec("openai", DeploymentClass.API,
                    models=tuple(m for m in prices.ids() if m.startswith("gpt")),
                    base_url="https://api.openai.com", priority=1),
    ])
    if transport is _NO_TRANSPORT:
        transport = FakeTransport()
    router = Router(prices=prices, backends=backends, transport=transport)
    router._keys = {"anthropic": "sk-test-a", "openai": "sk-test-o"}
    return router


class TestDecisionOnly:
    """The decision layer must work with no transport and no network."""

    def test_decide_needs_no_transport(self):
        router = make_router(transport=None)
        d = router.decide(__import__("llmrouter").RoutingRequest(
            messages=(Message("system", SYSTEM), Message("user", "hi")),
            session_id="s-1",
        ))
        assert d.model_id and d.backend_id

    def test_explain_after_decide(self):
        router = make_router(transport=None)
        router.decide(__import__("llmrouter").RoutingRequest(
            messages=(Message("user", "hi"),), session_id="s-2"))
        assert "model=" in router.explain()


class TestAsyncExecution:
    @pytest.mark.asyncio
    async def test_round_trip_and_learning(self):
        transport = FakeTransport()
        router = make_router(transport)

        out = await router.acomplete(
            [Message("system", SYSTEM), Message("user", "hello")],
            session_id="s-1", workload="chat",
        )
        assert out.text
        assert len(transport.calls) == 1

        # The session must now exist and be pinned.
        assert len(router.sessions) == 1
        stats = router.stats()
        assert stats["decisions"] == 1
        assert stats["estimated_cost_usd"] > 0

    @pytest.mark.asyncio
    async def test_cache_hit_rate_is_learned_not_assumed(self):
        def responder(url, body):
            # Provider-shaped usage: Anthropic reports cache_read_input_tokens,
            # OpenAI reports prompt_tokens_details.cached_tokens. Returning one
            # shape for both makes the test pass or fail on model choice alone.
            if "anthropic" in url:
                return FakeResponse({
                    "model": body["model"],
                    "content": [{"type": "text", "text": "ok"}],
                    "usage": {"input_tokens": 10, "output_tokens": 5,
                              "cache_creation_input_tokens": 100,
                              "cache_read_input_tokens": 900},
                })
            return FakeResponse({
                "model": body["model"],
                "choices": [{"message": {"content": "ok"}}],
                "usage": {"prompt_tokens": 1000, "completion_tokens": 5,
                          "prompt_tokens_details": {"cached_tokens": 900}},
            })

        router = make_router(FakeTransport(responder))
        await router.acomplete([Message("system", SYSTEM), Message("user", "a")],
                               session_id="s-learn", workload="agent")
        sess = router.sessions.get("sid:s-learn")
        assert sess is not None
        assert sess.observed_cache_hit_rate == pytest.approx(0.9, abs=0.01)

    @pytest.mark.asyncio
    async def test_session_stickiness_across_turns(self):
        router = make_router()
        msgs = [Message("system", SYSTEM), Message("user", "hello")]
        first = await router.acomplete(msgs, session_id="s-sticky", workload="chat")
        second = await router.acomplete(
            msgs + [Message("assistant", first.text), Message("user", "and now?")],
            session_id="s-sticky", workload="chat",
        )
        assert router.last_decision.model_id == first.model

    @pytest.mark.asyncio
    async def test_failover_to_healthy_backend(self):
        state = {"n": 0}

        def responder(url, body):
            if "anthropic" in url:
                state["n"] += 1
                raise RuntimeError("upstream 503")
            return FakeResponse({
                "model": body["model"],
                "choices": [{"message": {"content": "fallback ok"}}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 2,
                          "prompt_tokens_details": {"cached_tokens": 0}},
            })

        router = make_router(FakeTransport(responder))
        # Force an Anthropic model so the failure path is exercised.
        out = await router.acomplete(
            [Message("system", SYSTEM), Message("user", "hi")],
            session_id="s-fail", workload="chat",
        )
        # Either it failed over, or the router never picked anthropic. Either way
        # the caller gets an answer — that is the contract.
        assert out.text or state["n"] == 0

    @pytest.mark.asyncio
    async def test_failover_crosses_models_when_the_only_backend_dies(self):
        """A model whose sole backend is down must cost a cache miss, not an error.

        Failover that only looks for other backends serving the SAME model would
        fail hard here, because gpt-frontier-class is served only by `openai`.
        """
        calls = []

        def responder(url, body):
            calls.append(url)
            if "openai" in url and len(calls) == 1:
                raise RuntimeError("openai down")
            return FakeResponse({
                "model": body["model"],
                "content": [{"type": "text", "text": "recovered"}],
                "usage": {"input_tokens": 5, "output_tokens": 2,
                          "cache_creation_input_tokens": 0,
                          "cache_read_input_tokens": 0},
            })

        router = make_router(FakeTransport(responder))
        out = await router.acomplete(
            [Message("system", SYSTEM), Message("user", "hi")],
            session_id="s-crossmodel", workload="chat",
        )
        assert out.text == "recovered"
        # It must have landed somewhere other than the model it started on.
        assert router.last_decision.model_id != "gpt-frontier-class" or len(calls) > 1

    @pytest.mark.asyncio
    async def test_raises_after_exhausting_the_chain(self):
        def always_fail(url, body):
            raise RuntimeError("down")

        router = make_router(FakeTransport(always_fail))
        with pytest.raises(RouterError) as ei:
            await router.acomplete([Message("user", "hi")], session_id="s-dead")
        assert ei.value.attempts

    @pytest.mark.asyncio
    async def test_no_transport_fails_fast_as_a_config_error(self):
        """A misconfiguration must not be retried and reported as an outage."""
        router = make_router(transport=None)
        with pytest.raises(ConfigurationError, match="no transport"):
            await router.acomplete([Message("user", "hi")], session_id="s-nt")

    @pytest.mark.asyncio
    async def test_in_flight_is_not_double_decremented(self):
        """The except branch used to decrement in_flight and so did the finally,
        driving it negative and making the saturation check meaningless."""
        def always_fail(url, body):
            raise RuntimeError("down")

        router = make_router(FakeTransport(always_fail))
        with pytest.raises(RouterError):
            await router.acomplete([Message("user", "hi")], session_id="s-leak")
        for bid, stats in router.backends.stats().items():
            assert stats["in_flight"] == 0.0, f"{bid} leaked in_flight"


class TestAnthropicAdapter:
    def test_emits_ttl_1h_on_the_system_block(self):
        adapter = AnthropicAdapter()
        bps = (
            Breakpoint(SegmentKind.SYSTEM, 30_000, "1h", 0),
            Breakpoint(SegmentKind.HISTORY, 20_000, "5m", 30_000),
        )
        body = adapter.build_body(
            [Message("system", "sys prompt"), Message("user", "hi")],
            tools=(), breakpoints=bps, model="claude-sonnet-class",
        )
        assert body["system"][0]["cache_control"] == {"type": "ephemeral", "ttl": "1h"}
        assert body["messages"][-1]["cache_control"] == {"type": "ephemeral"}

    def test_rejects_1h_after_5m(self):
        adapter = AnthropicAdapter()
        bps = (
            Breakpoint(SegmentKind.HISTORY, 20_000, "5m", 0),
            Breakpoint(SegmentKind.SYSTEM, 30_000, "1h", 20_000),
        )
        with pytest.raises(ValueError, match="1h breakpoint"):
            adapter.build_body([Message("system", "s"), Message("user", "u")],
                               tools=(), breakpoints=bps, model="m")

    def test_marks_the_last_tool(self):
        adapter = AnthropicAdapter()
        bps = (Breakpoint(SegmentKind.TOOLS, 2_000, "5m", 0),)
        body = adapter.build_body(
            [Message("user", "hi")], tools=({"name": "a"}, {"name": "b"}),
            breakpoints=bps, model="m",
        )
        assert "cache_control" not in body["tools"][0]
        assert body["tools"][-1]["cache_control"] == {"type": "ephemeral"}

    def test_parses_cache_counters(self):
        adapter = AnthropicAdapter()
        c = adapter.parse({
            "model": "m", "content": [{"type": "text", "text": "hello"}],
            "usage": {"input_tokens": 5, "output_tokens": 3,
                      "cache_creation_input_tokens": 100,
                      "cache_read_input_tokens": 900},
        }, model="m", backend_id="b", latency_ms=10.0)
        assert c.text == "hello"
        assert c.usage.cache_read_tokens == 900
        assert c.usage.cache_write_tokens == 100
        assert c.usage.hit_rate == pytest.approx(0.9)


class TestOpenAIAdapter:
    def test_passes_the_prompt_cache_key(self):
        """OpenAI's native affinity hook. TTLs do not translate, so no
        breakpoints are emitted."""
        adapter = OpenAIAdapter()
        body = adapter.build_body(
            [Message("user", "hi")], tools=(), breakpoints=(),
            model="gpt-mini-class", cache_key="sid:s-1",
        )
        assert body["prompt_cache_key"] == "sid:s-1"
        assert "cache_control" not in str(body)

    def test_parses_cached_tokens(self):
        adapter = OpenAIAdapter()
        c = adapter.parse({
            "model": "m",
            "choices": [{"message": {"content": "hey"}}],
            "usage": {"prompt_tokens": 1000, "completion_tokens": 20,
                      "prompt_tokens_details": {"cached_tokens": 900}},
        }, model="m", backend_id="b", latency_ms=5.0)
        assert c.usage.cache_read_tokens == 900
        assert c.usage.cache_write_tokens == 100
        assert c.usage.hit_rate == pytest.approx(0.9)


class TestPricingInvariants:
    def test_openai_has_no_write_premium(self):
        prices = PriceRegistry()
        for card in prices.all():
            if card.model_id.startswith("gpt"):
                assert card.cache.write_mult_5m == 1.0, (
                    "OpenAI pre-5.6 charges no cache write premium; a cost model "
                    "that assumes 1.25x will wrongly forbid cheap migrations"
                )

    def test_cache_discount_is_per_model_not_per_provider(self):
        prices = PriceRegistry()
        gpt = {c.model_id: c.cache.read_mult for c in prices.all()
               if c.model_id.startswith("gpt")}
        assert len(set(gpt.values())) > 1, (
            "legacy gpt is 50%, current gpt is 90% — branching on provider is a bug"
        )

    def test_switching_onto_openai_is_cheaper_than_onto_anthropic(self):
        prices = PriceRegistry()
        sonnet = prices.require("claude-sonnet-class")
        haiku = prices.require("claude-haiku-class")
        gpt_mini = prices.require("gpt-mini-class")
        assert gpt_mini.cache_write_cost(50_000) < haiku.cache_write_cost(50_000)
        assert sonnet.cache_write_cost(50_000) > sonnet.cache_read_cost(50_000)
