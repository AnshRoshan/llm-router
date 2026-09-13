"""Quality floor, pre-call context validation, budget cap/pacing, Rule F, and
the decision-latency requirement (p50 < 10ms, enforced per the research spec).
"""
from __future__ import annotations

import statistics
import time

import pytest

from llmrouter import (
    BackendRegistry,
    BackendSpec,
    Constraints,
    DeploymentClass,
    Message,
    PolicyEngine,
    PriceCard,
    PriceRegistry,
    RoutingRequest,
    Router,
    RouterConfig,
    SessionPolicy,
    Weights,
    WorkloadClass,
    WorkloadResolver,
    WorkloadSource,
)


def make_engine(**kw) -> PolicyEngine:
    prices = kw.pop("prices", None) or PriceRegistry()
    ids = prices.ids()
    backends = BackendRegistry([
        BackendSpec("anthropic", DeploymentClass.API,
                    models=tuple(m for m in ids if m.startswith("claude")),
                    priority=0),
        BackendSpec("openai", DeploymentClass.API,
                    models=tuple(m for m in ids if m.startswith("gpt")),
                    priority=1),
    ])
    return PolicyEngine(prices, backends, WorkloadResolver(), **kw)


def req(text="hello", **kw) -> RoutingRequest:
    return RoutingRequest(messages=(Message("user", text),), **kw)


# --------------------------------------------------------------------------- #
# min_quality — the operating point (was dead code before this fix)
# --------------------------------------------------------------------------- #
def test_min_quality_floor_excludes_weak_models():
    engine = make_engine(config=RouterConfig(
        weights=Weights(min_quality=0.85)))
    d = engine.decide(req())
    assert d.candidates, "expected candidates"
    assert all(c.quality >= 0.85 for c in d.candidates)
    assert d.model_id in ("claude-opus-class", "gpt-frontier-class")


def test_min_quality_floor_too_high_is_a_clear_error():
    engine = make_engine(config=RouterConfig(
        weights=Weights(min_quality=0.99)))
    with pytest.raises(RuntimeError, match="filtered out"):
        engine.decide(req())


# --------------------------------------------------------------------------- #
# Pre-call context-window validation
# --------------------------------------------------------------------------- #
def _two_window_prices(small_window: int) -> PriceRegistry:
    # An empty registry now means "start from nothing" (the falsy-trap bug that
    # used to resurrect the bundled cards is fixed), so ONLY these two exist.
    prices = PriceRegistry({})
    prices.register(PriceCard(
        "small-window", 1.0, 2.0,
        quality_prior={"chat": 0.9}, context_window=small_window,
    ))
    prices.register(PriceCard(
        "large-window", 5.0, 10.0,
        quality_prior={"chat": 0.85}, context_window=200_000,
    ))
    return prices


def test_prefix_that_fills_window_is_filtered_precall():
    prices = _two_window_prices(small_window=8_192)
    engine = make_engine(prices=prices)
    # 20_000 chars ≈ 5_000 tokens; +4096 headroom > 8192 → small window out.
    big = "x" * 20_000
    d = engine.decide(req(big))
    assert d.model_id == "large-window"


def test_prefix_beyond_every_window_is_a_clear_error():
    prices = _two_window_prices(small_window=8_192)
    engine = make_engine(prices=prices)
    with pytest.raises(RuntimeError, match="filtered out"):
        engine.decide(req("x" * 1_000_000))


# --------------------------------------------------------------------------- #
# Budget cap and pacing
# --------------------------------------------------------------------------- #
def test_budget_cap_picks_cheapest_when_all_over():
    engine = make_engine()
    d = engine.decide(RoutingRequest(
        messages=(Message("user", "hi"),),
        workload=WorkloadClass.CHAT,
        constraints=Constraints(max_cost_usd=0.0001),
    ))
    cheapest = min(d.candidates, key=lambda c: c.cost_usd)
    assert d.model_id == cheapest.model_id
    assert any("budget cap" in r for r in d.reasons)


def test_budget_pacing_prefers_cheaper_model_under_pressure():
    unconstrained = make_engine().decide(req(workload=WorkloadClass.CHAT))
    pressured = make_engine().decide(RoutingRequest(
        messages=(Message("user", "hi"),),
        workload=WorkloadClass.CHAT,
        constraints=Constraints(budget_remaining_usd=0.002),
    ))
    assert pressured.model_id != unconstrained.model_id
    assert pressured.candidates[0].cost_usd < unconstrained.candidates[0].cost_usd


# --------------------------------------------------------------------------- #
# Rule F: cache-invalidating field drift
# --------------------------------------------------------------------------- #
LONG_SYSTEM = "You are a coding agent. " * 120  # ~600 est tokens: affinity gate


def _pinned_router() -> Router:
    prices = PriceRegistry()
    ids = prices.ids()
    backends = BackendRegistry([
        BackendSpec("anthropic", DeploymentClass.API,
                    models=tuple(m for m in ids if m.startswith("claude")),
                    priority=0),
        BackendSpec("openai", DeploymentClass.API,
                    models=tuple(m for m in ids if m.startswith("gpt")),
                    priority=1),
    ])
    router = Router(prices=prices, backends=backends)
    # A live, pinned session as _learn would have created it.
    router.sessions.put(SessionPolicy(
        session_key="sid:s1", workload=WorkloadClass.AGENT,
        source=WorkloadSource.EXPLICIT_TAG, confidence=1.0,
        pinned_model="claude-sonnet-class", pinned_backend="anthropic",
        pinned_at=time.monotonic(),
    ))
    return router


def _agent_req(text: str, tools) -> RoutingRequest:
    return RoutingRequest(
        messages=(Message("system", LONG_SYSTEM), Message("user", text)),
        tools=tools, session_id="s1", workload=WorkloadClass.AGENT,
    )


def test_rule_f_stable_tools_keep_warm_affinity():
    router = _pinned_router()
    tools = ({"name": "grep"},)
    d1 = router.decide(_agent_req("turn one", tools))
    assert not any("Rule F" in r for r in d1.reasons)
    d2 = router.decide(_agent_req("turn two", tools))
    warm = next(c for c in d2.candidates
                if c.model_id == d2.model_id and c.backend_id == d2.backend_id)
    assert not any("Rule F" in r for r in d2.reasons)
    assert warm.affinity > 0.0, "pinned pair must keep its affinity bonus"


def test_rule_f_tool_drift_scores_everything_cold():
    router = _pinned_router()
    router.decide(_agent_req("turn one", ({"name": "grep"},)))

    # The tool schema changed mid-session: the provider's cache key changed,
    # so no candidate may claim a warm cache.
    d2 = router.decide(_agent_req(
        "turn two", ({"name": "grep"}, {"name": "edit"})))
    assert any("Rule F" in r for r in d2.reasons)
    assert all(c.affinity == 0.0 for c in d2.candidates)
    assert all(c.switch_cost_usd == 0.0 for c in d2.candidates)

    # The session baseline was updated, so turn 3 is not re-flagged.
    d3 = router.decide(_agent_req(
        "turn three", ({"name": "grep"}, {"name": "edit"})))
    assert not any("Rule F" in r for r in d3.reasons)


# --------------------------------------------------------------------------- #
# Decision latency — enforced, not aspirational
# --------------------------------------------------------------------------- #
def test_decision_latency_p50_under_10ms():
    engine = make_engine()
    samples = []
    for _ in range(200):
        t0 = time.perf_counter()
        engine.decide(req())
        samples.append((time.perf_counter() - t0) * 1000.0)
    p50 = statistics.median(samples)
    assert p50 < 10.0, f"decision p50 {p50:.2f}ms exceeds the 10ms budget"


# --------------------------------------------------------------------------- #
# Router.pure(): decision-only facade
# --------------------------------------------------------------------------- #
def test_router_pure_decides_without_keys_or_network():
    import asyncio
    from llmrouter import ConfigurationError

    router = Router.pure()
    d = router.decide(req())
    assert d.model_id and d.candidates
    # No transport: execution refuses clearly instead of hanging or failing
    # opaquely, while the decision surface stays fully usable.
    with pytest.raises(ConfigurationError):
        asyncio.run(router.acomplete([Message("user", "hi")]))


def test_session_stats_report_hit_rate_by_workload_source():
    from llmrouter import SessionStore, WorkloadSource

    store = SessionStore()
    s = SessionPolicy(
        session_key="sid:a", workload=WorkloadClass.AGENT,
        source=WorkloadSource.FINGERPRINT, confidence=1.0,
        pinned_model="claude-sonnet-class", pinned_backend="anthropic",
        pinned_at=time.monotonic(),
    )
    s.record_usage(cache_read_tokens=900, cache_write_tokens=100)
    store.put(s)
    stats = store.stats()
    assert stats["hit_rate_by_source"]["fingerprint"] == pytest.approx(0.9)
    assert stats["avg_hit_rate"] == pytest.approx(0.9)
