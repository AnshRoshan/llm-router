"""Affinity, backend health, and the end-to-end decision layer."""
from __future__ import annotations

import time

import pytest

from llmrouter import (
    BackendRegistry,
    BackendSpec,
    DeploymentClass,
    Message,
    PriceRegistry,
    RouterConfig,
    RoutingRequest,
    SessionPolicy,
    SessionStore,
    Weights,
    WorkloadClass,
    WorkloadResolver,
    affinity_prior,
    rendezvous_pick,
    rendezvous_rank,
)
from llmrouter.policy import PolicyEngine

SYSTEM = "You are a helpful assistant. " * 200  # ~2k tokens of stable prefix


def make_engine(**cfg) -> PolicyEngine:
    backends = BackendRegistry([
        BackendSpec("anthropic", DeploymentClass.API,
                    models=("claude-opus-class", "claude-sonnet-class", "claude-haiku-class"),
                    base_url="https://api.anthropic.com", priority=0),
        BackendSpec("openai", DeploymentClass.API,
                    models=("gpt-frontier-class", "gpt-legacy-class", "gpt-mini-class"),
                    base_url="https://api.openai.com", priority=1),
    ])
    return PolicyEngine(
        PriceRegistry(), backends, WorkloadResolver(),
        RouterConfig(weights=Weights(**cfg.pop("weights", {})), **cfg),
    )


def make_req(**kw) -> RoutingRequest:
    messages = kw.pop("messages", (Message("system", SYSTEM), Message("user", "hi")))
    return RoutingRequest(messages=messages, **kw)


def make_session(engine: PolicyEngine, model="claude-sonnet-class",
                 backend="anthropic", **kw) -> SessionPolicy:
    return SessionPolicy(
        session_key=kw.pop("session_key", "sid:s-1"),
        workload=WorkloadClass.CHAT,
        source=kw.pop("source", __import__("llmrouter").WorkloadSource.EXPLICIT_TAG),
        confidence=1.0,
        pinned_model=model,
        pinned_backend=backend,
        pinned_at=0.0,
        **kw,
    )


class TestRendezvousHashing:
    def test_deterministic(self):
        nodes = ["a", "b", "c"]
        assert rendezvous_pick("k", nodes) == rendezvous_pick("k", nodes)

    def test_stable_under_membership_change(self):
        """Adding a node must only remap that node's share.

        A hash ring rebalance is what silently turns a 90% cache hit rate into
        a 12.5% one, so this property is load-bearing.
        """
        before = ["a", "b", "c"]
        after = ["a", "b", "c", "d"]
        keys = [f"session-{i}" for i in range(200)]
        stranded = 0
        for k in keys:
            old, new = rendezvous_pick(k, before), rendezvous_pick(k, after)
            if old != new and new != "d":
                stranded += 1   # moved between two ORIGINAL nodes — the bad case
        assert stranded == 0, "rendezvous hashing must only move keys TO the new node"
        # And it must actually be doing something: some keys should move to d.
        assert any(rendezvous_pick(k, after) == "d" for k in keys)

    def test_ranking_is_a_permutation(self):
        nodes = ["a", "b", "c"]
        assert sorted(rendezvous_rank("k", nodes)) == sorted(nodes)

    def test_empty(self):
        assert rendezvous_pick("k", []) is None


class TestAffinityPrior:
    def test_batch_gets_no_affinity(self):
        assert affinity_prior(WorkloadClass.BATCH, 5.0) == 0.0

    def test_agent_gaps_are_tiny_so_hit_rate_is_high(self):
        assert affinity_prior(WorkloadClass.AGENT, 5.0) > 0.9

    def test_chat_gaps_often_exceed_the_ttl(self):
        # Humans think between turns. A 7-minute gap on a 5-minute TTL means the
        # cache is usually cold, so chat must NOT get the same bonus as an agent.
        assert affinity_prior(WorkloadClass.CHAT, 420.0) < 0.3
        assert affinity_prior(WorkloadClass.CHAT, 420.0) < affinity_prior(WorkloadClass.AGENT, 5.0)


class TestSessionStore:
    def test_learned_hit_rate_replaces_the_prior(self):
        s = make_session(make_engine())
        assert s.hit_rate(prior=0.9) == 0.9  # not enough samples yet
        for _ in range(5):
            s.record_usage(cache_read_tokens=900, cache_write_tokens=100)
        assert s.observed_cache_hit_rate == pytest.approx(0.9, abs=0.01)
        assert s.hit_rate(prior=0.5) == pytest.approx(0.9, abs=0.01)

    def test_zero_usage_does_not_pollute_the_estimate(self):
        s = make_session(make_engine())
        s.record_usage(0, 0)
        assert s.observed_cache_hit_rate is None

    def test_idle_eviction(self):
        store = SessionStore(idle_ttl_s=10.0)
        s = make_session(make_engine())
        store.put(s, now=1000.0)
        assert store.get("sid:s-1", now=1005.0) is not None
        assert store.get("sid:s-1", now=1020.0) is None


class TestBackendHealth:
    def make(self):
        reg = BackendRegistry([BackendSpec("a"), BackendSpec("b")])
        return reg

    def test_circuit_opens_after_consecutive_failures(self):
        reg = self.make()
        h = reg.health("a")
        for _ in range(5):
            h.record_failure("boom", now=100.0)
        assert h.is_available(now=100.5) is False

    def test_circuit_closes_on_success(self):
        reg = self.make()
        h = reg.health("a")
        for _ in range(5):
            h.record_failure("boom", now=100.0)
        h.record_success(120.0, now=200.0)
        assert h.is_available(now=200.0) is True
        assert h.consecutive_failures == 0

    def test_saturation_sheds_instead_of_queueing(self):
        spec = BackendSpec("a", max_concurrency=2)
        reg = BackendRegistry([spec])
        h = reg.health("a")
        h.in_flight = 2
        assert h.saturating(spec) is True

    def test_available_for_filters_by_model_and_health(self):
        reg = self.make()
        reg._specs["a"] = BackendSpec("a", models=("m1",))
        reg._specs["b"] = BackendSpec("b", models=("m2",))
        assert [b.backend_id for b in reg.available_for("m1")] == ["a"]


class TestDecision:
    def test_scores_model_backend_pairs_not_just_models(self):
        engine = make_engine()
        d = engine.decide(make_req(workload=WorkloadClass.CHAT))
        pairs = {(c.model_id, c.backend_id) for c in d.candidates}
        assert len(pairs) == len(d.candidates)
        # Anthropic models must never be paired with the openai backend.
        for model, backend in pairs:
            if model.startswith("claude"):
                assert backend == "anthropic"
            else:
                assert backend == "openai"

    def test_first_turn_pins_and_later_turns_reuse(self):
        engine = make_engine()
        d1 = engine.decide(make_req(session_id="s-1"))
        assert d1.reused_session is False

        sess = SessionPolicy(
            session_key="sid:s-1", workload=WorkloadClass.CHAT,
            source=d1.workload_source, confidence=1.0,
            pinned_model=d1.model_id, pinned_backend=d1.backend_id, pinned_at=0.0,
        )
        d2 = engine.decide(make_req(session_id="s-1"), session=sess)
        assert d2.reused_session is True
        assert d2.model_id == d1.model_id, "a live session must not drift models"

    def test_no_oscillation(self):
        """A cost-driven migration is allowed once, never twice.

        affinity=0 removes the bonus that would otherwise prop up the pinned
        model, so the engine genuinely prefers the cheaper one.
        """
        engine = make_engine(weights={"affinity": 0.0})
        req = make_req(session_id="s-1", expected_turns=100)

        # First migration: allowed.
        fresh = make_session(engine, model="claude-opus-class", backend="anthropic")
        d1 = engine.decide(req, session=fresh)
        assert d1.switched is True
        assert d1.model_id != "claude-opus-class"

        # Second migration: refused by the lock.
        locked = make_session(engine, model="claude-opus-class",
                              backend="anthropic", switched_at=1.0)
        d2 = engine.decide(req, session=locked)
        assert d2.model_id == "claude-opus-class", "second migration must be refused"
        assert any("oscillation" in r for r in d2.reasons), d2.reasons
        assert d2.switched is False

    def test_stranded_is_surfaced(self):
        """A migration we wanted but could not afford must be counted, not silent.

        One turn left: no cheaper model can pay back its cache write in time.
        """
        engine = make_engine(weights={"affinity": 0.0})
        sess = make_session(engine, model="claude-opus-class", backend="anthropic")
        before = sess.stranded_count
        d = engine.decide(make_req(session_id="s-1", expected_turns=1), session=sess)
        assert d.model_id == "claude-opus-class", "must stay pinned"
        assert sess.stranded_count == before + 1, "stranded migrations must be counted"
        assert d.stranded is True

    def test_forced_failover_beats_the_oscillation_lock(self):
        """An outage is not oscillation. If the pinned backend is down, failing
        over must NOT be blocked by the lock — or an outage becomes a hard error.
        """
        engine = make_engine()
        now = time.monotonic()
        for _ in range(5):
            engine.backends.health("anthropic").record_failure("down", now=now)
        sess = make_session(engine, model="claude-opus-class",
                            backend="anthropic", switched_at=1.0)
        d = engine.decide(make_req(session_id="s-1", expected_turns=100), session=sess)
        assert d.model_id != "claude-opus-class", "must fail over, not fail hard"
        assert any("forced failover" in r or "no viable candidate" in r
                   for r in d.reasons), d.reasons

    def test_explain_is_populated(self):
        engine = make_engine()
        d = engine.decide(make_req(workload=WorkloadClass.AGENT,
                                   tools=({"name": "search"},)))
        text = d.explain()
        assert "model=" in text and "workload=agent" in text
        assert "score=" in text

    def test_vision_requirement_is_a_hard_constraint(self):
        engine = make_engine()
        d = engine.decide(make_req(
            messages=(Message("user", [{"type": "image", "source": {}}]),),
        ))
        assert d.workload == WorkloadClass.VISION

    def test_raises_when_everything_is_filtered_out(self):
        engine = make_engine()
        engine.backends = BackendRegistry([])
        engine.backends._specs.clear()
        with pytest.raises(RuntimeError):
            engine.decide(make_req())


class TestTTLPolicyInDecisions:
    def test_agent_with_short_gaps_stays_on_5m(self):
        engine = make_engine()
        d = engine.decide(make_req(workload=WorkloadClass.AGENT,
                                   tools=({"name": "x"},),
                                   expected_gap_s=5.0,
                                   expected_generation_s=2.0))
        assert set(d.ttl_by_segment.values()) == {"5m"}

    def test_chat_with_long_gaps_upgrades_the_system_segment(self):
        engine = make_engine()
        d = engine.decide(make_req(workload=WorkloadClass.CHAT,
                                   expected_gap_s=1200.0))
        # Only when the pinned model is an Anthropic one (mixed TTL supported).
        if d.model_id.startswith("claude"):
            assert "1h" in d.ttl_by_segment.values()

    def test_generation_time_can_force_the_upgrade(self):
        engine = make_engine()
        # A 320s stream blows the 5-minute window on its own.
        d = engine.decide(make_req(workload=WorkloadClass.AGENT,
                                   tools=({"name": "x"},),
                                   expected_gap_s=10.0,
                                   expected_generation_s=320.0))
        if d.model_id.startswith("claude"):
            assert "1h" in d.ttl_by_segment.values()
