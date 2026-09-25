"""Tests for durable router state: snapshot shape, clock rebasing, restore."""
import json
import time

import pytest

from llmrouter import FileStateStore, Message, Router
from llmrouter.affinity import SessionPolicy
from llmrouter.delegation import PairRegistry
from llmrouter.signals import WorkloadSignal
from llmrouter.store import restore_snapshot, take_snapshot
from llmrouter.types import WorkloadClass, WorkloadSource


def _router(state_path=None):
    return Router.with_defaults(
        anthropic_key="k",
        state_store=FileStateStore(str(state_path)) if state_path else None,
    )


def _seed_session(router, key="sid:abc", *, pinned_at=None, expires_in=1800.0,
                  hits=4000, writes=500, switched=True, stranded=2):
    now = time.monotonic()
    sess = SessionPolicy(
        session_key=key, workload=WorkloadClass.AGENT,
        source=WorkloadSource.STRUCTURAL, confidence=0.9,
        pinned_model="claude-sonnet-class", pinned_backend="anthropic",
        pinned_at=now - 100.0,
        switched_at=now - 50.0 if switched else None,
        expires_at=0.0, turns_seen=7, prefix_tokens=4500,
        stranded_count=stranded, invalidators_hash="ff" * 8,
    )
    sess.record_usage(hits, writes)
    router.sessions.put(sess)
    # put() resets the TTL; pin an exact remaining window for the rebase test
    sess.expires_at = now + expires_in
    return sess


def test_snapshot_roundtrip_rebases_clocks(tmp_path):
    r1 = _router(tmp_path / "state.json")
    _seed_session(r1)
    r1.resolver.fingerprints.put("fp1", WorkloadSignal(
        WorkloadClass.CHAT, WorkloadSource.STRUCTURAL, 0.8, "test"))
    r1.pairs.record_outcome("lead-a", "side-b", cost=0.02, attempts=2.0)
    r1.backends.health("anthropic").record_failure("boom")
    r1.save_state()

    snap = json.loads((tmp_path / "state.json").read_text(encoding="utf-8"))
    assert snap["format"] == "llmrouter-state-v1"
    assert "sid:abc" in snap["sessions"]
    # EWMA from the neutral default: 0.7*1.0 + 0.3*2.0 = 1.3
    assert snap["pairs"]["lead-a\x1fside-b"]["sidekick_attempts"] == pytest.approx(1.3)

    r2 = _router(tmp_path / "state.json")
    s = r2.sessions.peek("sid:abc")
    assert s.pinned_model == "claude-sonnet-class"
    assert s.turns_seen == 7 and s.stranded_count == 2
    assert s.switched_at is not None            # no-oscillation lock survives
    assert 1700 < s.expires_at - time.monotonic() < 1800  # TTL rebased, not reset
    assert s.observed_cache_hit_rate == pytest.approx(4000 / 4500)
    assert r2.resolver.fingerprints.get("fp1") is not None
    prof = r2.pairs.get("lead-a", "side-b")
    assert prof.measured_cost_per_task == pytest.approx(0.02)
    assert prof.sidekick_attempts == pytest.approx(1.3)
    assert r2.backends.health("anthropic").consecutive_failures == 1


def test_in_flight_and_probes_never_resurrect(tmp_path):
    r1 = _router(tmp_path / "s.json")
    h = r1.backends.health("anthropic")
    h.in_flight = 12
    h.half_open_probes = 1
    r1.save_state()
    r2 = _router(tmp_path / "s.json")
    h2 = r2.backends.health("anthropic")
    assert h2.in_flight == 0 and h2.half_open_probes == 0
    # but the learned latency and counters survive
    assert h2.successes == h2.failures == 0


def test_expired_sessions_are_dropped_on_restore(tmp_path):
    path = tmp_path / "s.json"
    r1 = _router(path)
    _seed_session(r1, expires_in=-5.0)  # already expired when saved
    r1.save_state()
    r2 = _router(path)
    assert r2.sessions.get("sid:abc") is None


def test_autosave_every_n_decisions(tmp_path):
    path = tmp_path / "auto.json"
    store = FileStateStore(str(path), autosave_every=2)
    r = Router.with_defaults(anthropic_key="k", state_store=store)

    class Resp:
        status_code = 200

        def json(self):
            return {"model": "claude-haiku-class",
                    "content": [{"type": "text", "text": "ok"}],
                    "usage": {"input_tokens": 10, "output_tokens": 5,
                              "cache_read_input_tokens": 900,
                              "cache_creation_input_tokens": 100}}

    class T:
        async def post(self, url, **kw):
            return Resp()

    import asyncio
    r.transport = T()
    msg = [Message("user", "hi " * 300)]
    asyncio.run(r.acomplete(msg, workload="chat", session_id="a"))
    assert not path.exists()  # 1 decision < autosave_every
    asyncio.run(r.acomplete(msg, workload="chat", session_id="a"))
    assert path.exists()      # the second folded it in
    snap = json.loads(path.read_text(encoding="utf-8"))
    assert snap["counters"]["decisions"] == 2


def test_load_state_false_on_first_boot(tmp_path):
    r = _router(tmp_path / "missing.json")
    assert r.load_state() is False


def test_restore_rejects_foreign_format():
    r = _router()
    with pytest.raises(ValueError, match="not an llmrouter state snapshot"):
        restore_snapshot(r, {"format": "something-else"})


def test_sessions_for_unknown_backend_migrate_honestly(tmp_path):
    """A snapshot referencing a backend this deploy never registered must not
    crash — the session simply fails health and takes the failover path."""
    path = tmp_path / "s.json"
    r1 = _router(path)
    _seed_session(r1)
    r1.save_state()

    r2 = Router(prices=r1.prices, backends=r1.backends.__class__([]),
                state_store=None)
    restore_snapshot(r2, json.loads(path.read_text(encoding="utf-8")))
    assert r2.sessions.peek("sid:abc").pinned_backend == "anthropic"
