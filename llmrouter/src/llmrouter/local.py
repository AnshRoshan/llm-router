"""Local and self-hosted models: turnkey Ollama plus honest GPU economics.

RouteLLM can only reach local models through LiteLLM; ulab LLMRouter wires
Ollama as a plain endpoint; Bifrost speaks to 23+ providers but never prices a
local GPU against a cloud invoice inside the routing score. Here the local box
is a first-class *backend* in the (model, backend) pair — which means the same
scoring machinery (budget caps, failover, circuit breaker, capacity shedding)
applies across the local/cloud boundary.

The part nobody else does properly: **what does a local token cost?** A local
model is not free — you rent the GPU by the hour. The conversion is honest
arithmetic, not a zero:

    $/MTok(output) = gpu_cost_per_hour / (tokens_per_sec * 3600) * 1e6
    $/MTok(input)  = same / prefill speedup (prefill is memory-bandwidth-cheap)

A 35 tok/s 8B on a GPU costing $0.35/hr lands at $2.78/MTok output — *more*
expensive than gpt-mini-class list price. That is exactly the comparison the
scorer needs and the marketing-free answer to "should this run locally?"

Ollama speaks an OpenAI-compatible /v1/chat/completions, so no new adapter is
needed; note its shim reports no cache-token counters, so learned hit rates
fall back to the workload priors there. vLLM/SGLang prefix caching DOES report
`prompt_tokens_details.cached_tokens` and learns properly.
"""
from __future__ import annotations

import json
import urllib.request
from typing import Mapping, Sequence

from .backends import DEFAULT_LOCAL_TIMEOUT_S, BackendSpec
from .pricing import SELF_HOSTED_CACHE, PriceCard
from .types import DeploymentClass, WorkloadClass

OLLAMA_DEFAULT_URL = "http://localhost:11434"

#: Which workload keys a local card carries. VISION is decided per card so a
#: text-only local model is *excluded* from vision traffic by the hard
#: constraints, not merely scored low.
_PRIOR_WORKLOADS = tuple(
    w for w in WorkloadClass if w not in (WorkloadClass.UNKNOWN, WorkloadClass.VISION)
)


def gpu_price_card(model_id: str, *, gpu_cost_per_hr: float = 0.35,
                   tokens_per_sec: float = 35.0, prefill_mult: float = 6.0,
                   quality: float = 0.72, context_window: int = 32_768,
                   supports_tools: bool = False,
                   supports_vision: bool = False,
                   deployment: DeploymentClass = DeploymentClass.SELF_HOSTED,
                   residency_tags: frozenset[str] = frozenset(),
                   ) -> PriceCard:
    """A price card whose $/MTok is derived from renting the GPU.

    `tokens_per_sec` is DECODE throughput (the bottleneck); prefill is priced
    `prefill_mult` times cheaper. Set `gpu_cost_per_hr=0` for hardware you
    already own outright and are not shadow-pricing.
    """
    if tokens_per_sec <= 0 or prefill_mult <= 0:
        raise ValueError("tokens_per_sec and prefill_mult must be positive")
    output_per_mtok = gpu_cost_per_hr / (tokens_per_sec * 3600.0) * 1_000_000.0
    input_per_mtok = output_per_mtok / prefill_mult
    prior = {w.value: quality for w in _PRIOR_WORKLOADS}
    # A negative prior means "explicitly unsupported" (see policy hard
    # constraints) — the honest flag for a model that cannot see.
    prior[WorkloadClass.VISION.value] = quality if supports_vision else -1.0
    return PriceCard(
        model_id=model_id,
        input_per_mtok=input_per_mtok,
        output_per_mtok=output_per_mtok,
        cache=SELF_HOSTED_CACHE,
        context_window=context_window,
        supports_tools=supports_tools,
        supports_vision=supports_vision,
        deployment=deployment,
        residency_tags=residency_tags,
        quality_prior=prior,
        p50_latency_ms=400.0,
    )


def ollama_models(base_url: str = OLLAMA_DEFAULT_URL,
                  timeout_s: float = 5.0) -> tuple[str, ...]:
    """Model names installed on an Ollama server (GET /api/tags).

    Setup-time only — the decision layer must never touch the network, so this
    runs when you BUILD the router, not when it decides. Call it, feed the
    names to `Router.with_ollama(models=...)`, and the cold path stays pure.
    """
    url = f"{base_url.rstrip('/')}/api/tags"
    with urllib.request.urlopen(url, timeout=timeout_s) as resp:
        payload = json.loads(resp.read())
    names = [str(m.get("name", "")).strip()
             for m in payload.get("models", []) if m.get("name")]
    return tuple(sorted(set(names)))


def ollama_backend(models: Sequence[str], *, backend_id: str = "ollama",
                   base_url: str = OLLAMA_DEFAULT_URL,
                   capacity_rps: float = 2.0,
                   timeout_s: float = DEFAULT_LOCAL_TIMEOUT_S,
                   deployment: DeploymentClass = DeploymentClass.SELF_HOSTED,
                   ) -> BackendSpec:
    """The local fleet as a scoring candidate: finite capacity, aggressive
    timeout (the cascade trap — a saturated GPU must shed, not queue, or every
    request tips to cloud at ~100x the cost)."""
    return BackendSpec(
        backend_id=backend_id,
        deployment=deployment,
        models=tuple(models),
        base_url=base_url,
        timeout_s=timeout_s,
        capacity_rps=capacity_rps,
        cold_start_risk=0.0 if deployment != DeploymentClass.SERVERLESS else 0.3,
    )


def ollama_price_cards(
    models: Sequence[str], *,
    gpu_cost_per_hr: float = 0.35, tokens_per_sec: float = 35.0,
    quality: float = 0.72,
    per_model: Mapping[str, Mapping[str, float]] | None = None,
) -> dict[str, PriceCard]:
    """Cards for many local models at once, with per-model overrides.

    `per_model` maps a model name to kwargs for :func:`gpu_price_card`, e.g.
    ``{"qwen2.5-coder:32b": {"quality": 0.84, "supports_tools": True,
    "tokens_per_sec": 9, "gpu_cost_per_hr": 1.80}}``.
    """
    overrides = dict(per_model or {})
    cards: dict[str, PriceCard] = {}
    for name in models:
        kwargs: dict[str, float | bool] = {
            "gpu_cost_per_hr": gpu_cost_per_hr,
            "tokens_per_sec": tokens_per_sec,
            "quality": quality,
        }
        # per-model entries win outright (no duplicate-kwarg collision)
        kwargs.update(overrides.get(name, {}))
        cards[name] = gpu_price_card(name, **kwargs)
    return cards


__all__ = [
    "OLLAMA_DEFAULT_URL",
    "gpu_price_card",
    "ollama_models",
    "ollama_backend",
    "ollama_price_cards",
]
