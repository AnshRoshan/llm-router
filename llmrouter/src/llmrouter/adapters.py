"""Backend adapters: the only part that touches the network.

`httpx` is an OPTIONAL dependency. The decision engine never imports it, so the
router is usable as a pure policy library with zero installed dependencies — you
can drive your own HTTP client and just call `Router.decide()`.

Each adapter does three jobs:
  1. build the provider-shaped request body (incl. cache_control breakpoints)
  2. send it
  3. **parse the usage counters back out** — this is how the router learns the
     real cache hit rate instead of assuming one
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any, Mapping, Protocol, Sequence

from .economics import Breakpoint
from .types import Message, SegmentKind


@dataclass(frozen=True)
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    model: str | None = None
    raw: Mapping[str, Any] | None = None

    @property
    def hit_rate(self) -> float:
        total = self.cache_read_tokens + self.cache_write_tokens
        return self.cache_read_tokens / total if total else 0.0


@dataclass(frozen=True)
class Completion:
    text: str
    usage: Usage
    model: str
    backend_id: str
    latency_ms: float
    raw: Mapping[str, Any] | None = None


class Transport(Protocol):
    """Minimal async HTTP surface. Satisfied by httpx.AsyncClient."""

    async def post(self, url: str, *, headers: Mapping[str, str] | None = None,
                   json: Mapping[str, Any] | None = None,
                   timeout: float | None = None) -> Any: ...


class ModelAdapter(Protocol):
    name: str

    def build_body(self, messages: Sequence[Message], *, tools: Sequence[Mapping[str, Any]],
                   breakpoints: Sequence[Breakpoint], model: str,
                   cache_key: str | None, **kwargs: Any) -> Mapping[str, Any]: ...

    def parse(self, payload: Mapping[str, Any], *, model: str,
              backend_id: str, latency_ms: float) -> Completion: ...


# --------------------------------------------------------------------------- #
# Anthropic
# --------------------------------------------------------------------------- #
_SEGMENT_TO_BLOCK = {
    SegmentKind.TOOLS: "tools",
    SegmentKind.SYSTEM: "system",
    SegmentKind.RETRIEVED: "system",
    SegmentKind.HISTORY: "messages",
    SegmentKind.USER: "messages",
}


class AnthropicAdapter:
    """Explicit breakpoints with per-segment TTL.

    Ordering is a HARD constraint: a 1h entry after a 5m entry is rejected or
    silently downgraded. `emit_breakpoints` already enforces this; we re-check
    here because a hand-built request can bypass it.
    """

    name = "anthropic"

    def __init__(self, path: str = "/v1/messages", version: str = "2023-06-01") -> None:
        self.path = path
        self.version = version

    def build_body(self, messages: Sequence[Message], *,
                   tools: Sequence[Mapping[str, Any]],
                   breakpoints: Sequence[Breakpoint],
                   model: str, cache_key: str | None = None,
                   max_tokens: int = 1024, **kwargs: Any) -> Mapping[str, Any]:
        _assert_ttl_ordering(breakpoints)

        ttl_for = {bp.kind: bp.ttl for bp in breakpoints}
        body: dict[str, Any] = {"model": model, "max_tokens": max_tokens}

        if tools:
            tools_out = [dict(t) for t in tools]
            if SegmentKind.TOOLS in ttl_for:
                tools_out[-1]["cache_control"] = _cc(ttl_for[SegmentKind.TOOLS])
            body["tools"] = tools_out

        system_text = "\n".join(m.text for m in messages if m.role == "system")
        if system_text:
            block: dict[str, Any] = {"type": "text", "text": system_text}
            if SegmentKind.SYSTEM in ttl_for:
                block["cache_control"] = _cc(ttl_for[SegmentKind.SYSTEM])
            body["system"] = [block]

        convo = [m for m in messages if m.role != "system"]
        msgs: list[dict[str, Any]] = [{"role": m.role, "content": m.text} for m in convo]
        # Mark the last message so the growing history is incrementally cacheable.
        if msgs and SegmentKind.HISTORY in ttl_for:
            msgs[-1]["cache_control"] = _cc(ttl_for[SegmentKind.HISTORY])
        body["messages"] = msgs
        body.update(kwargs)
        return body

    def parse(self, payload: Mapping[str, Any], *, model: str, backend_id: str,
              latency_ms: float) -> Completion:
        usage = payload.get("usage", {}) or {}
        text = ""
        for block in payload.get("content", []) or []:
            if isinstance(block, Mapping) and block.get("type") == "text":
                text += str(block.get("text", ""))
        return Completion(
            text=text,
            usage=Usage(
                input_tokens=int(usage.get("input_tokens", 0) or 0),
                output_tokens=int(usage.get("output_tokens", 0) or 0),
                cache_read_tokens=int(usage.get("cache_read_input_tokens", 0) or 0),
                cache_write_tokens=int(usage.get("cache_creation_input_tokens", 0) or 0),
                model=payload.get("model", model),
                raw=usage,
            ),
            model=payload.get("model", model),
            backend_id=backend_id,
            latency_ms=latency_ms,
            raw=payload,
        )


# --------------------------------------------------------------------------- #
# OpenAI
# --------------------------------------------------------------------------- #
class OpenAIAdapter:
    """Automatic caching. We pass `prompt_cache_key` so the provider does
    affinity routing inside its own fleet — the native hook for session stickiness.
    """

    name = "openai"

    def __init__(self, path: str = "/v1/chat/completions") -> None:
        self.path = path

    def build_body(self, messages: Sequence[Message], *,
                   tools: Sequence[Mapping[str, Any]],
                   breakpoints: Sequence[Breakpoint],
                   model: str, cache_key: str | None = None,
                   max_tokens: int = 1024, **kwargs: Any) -> Mapping[str, Any]:
        # TTLs do not translate to OpenAI — a cache_control ttl would be dropped,
        # so we deliberately do not emit breakpoints here.
        body: dict[str, Any] = {
            "model": model,
            "max_tokens": max_tokens,
            "messages": [{"role": m.role, "content": m.text} for m in messages],
        }
        if tools:
            body["tools"] = [{"type": "function", "function": dict(t)} for t in tools]
        if cache_key:
            body["prompt_cache_key"] = cache_key
        body.update(kwargs)
        return body

    def parse(self, payload: Mapping[str, Any], *, model: str, backend_id: str,
              latency_ms: float) -> Completion:
        usage = payload.get("usage", {}) or {}
        details = usage.get("prompt_tokens_details", {}) or {}
        choices = payload.get("choices", []) or []
        text = ""
        if choices:
            msg = choices[0].get("message", {}) or {}
            text = str(msg.get("content", "") or "")
        return Completion(
            text=text,
            usage=Usage(
                input_tokens=int(usage.get("prompt_tokens", 0) or 0),
                output_tokens=int(usage.get("completion_tokens", 0) or 0),
                cache_read_tokens=int(details.get("cached_tokens", 0) or 0),
                # OpenAI charges no write premium pre-GPT-5.6, so a miss is not
                # billed as a write; count the uncached remainder instead.
                cache_write_tokens=max(
                    0,
                    int(usage.get("prompt_tokens", 0) or 0)
                    - int(details.get("cached_tokens", 0) or 0),
                ),
                model=payload.get("model", model),
                raw=usage,
            ),
            model=payload.get("model", model),
            backend_id=backend_id,
            latency_ms=latency_ms,
            raw=payload,
        )


# --------------------------------------------------------------------------- #
def _cc(ttl: str) -> dict[str, str]:
    if ttl == "1h":
        return {"type": "ephemeral", "ttl": "1h"}
    return {"type": "ephemeral"}


def _assert_ttl_ordering(breakpoints: Sequence[Breakpoint]) -> None:
    seen_short = False
    for bp in breakpoints:
        if bp.ttl == "5m":
            seen_short = True
        elif seen_short:
            raise ValueError(
                f"1h breakpoint ({bp.kind.value}) after a 5m breakpoint: the "
                "provider rejects or downgrades this. Reorder longer-TTL first."
            )


def adapter_for(backend_kind: str) -> ModelAdapter:
    kind = backend_kind.lower()
    if "anthropic" in kind or "claude" in kind:
        return AnthropicAdapter()
    if "openai" in kind or "gpt" in kind:
        return OpenAIAdapter()
    # Self-hosted / OpenAI-compatible servers (vLLM, SGLang, TGI, Ollama).
    return OpenAIAdapter()


__all__ = [
    "Usage",
    "Completion",
    "Transport",
    "ModelAdapter",
    "AnthropicAdapter",
    "OpenAIAdapter",
    "adapter_for",
]
