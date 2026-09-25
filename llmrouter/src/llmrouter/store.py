"""Durable router state: sessions, backend health, fingerprints, pair profiles.

Affinity is only as real as the store behind it. With the default in-process
stores, a restart — or a second replica — silently converts a warm cache into a
cold one the router *thinks* is cold, and the whole point of the affinity term
is lost. This module makes the state outlive the process that learned it:

    router = Router.with_defaults(..., state_store=FileStateStore("state.json"))
    # ... traffic flows; state autosaves every N decisions ...
    # restart / redeploy:
    router = Router.with_defaults(..., state_store=FileStateStore("state.json"))
    # sessions, hit-rate EWMA, circuit counters, pair economics are back.

The monotonic clock caveat: every stored timestamp is rebased on restore
(`value' = now - (saved_mono - value)`), because monotonic clocks are
process-relative. A session that had 40 minutes of TTL left when saved has 40
minutes left after restore — not "eternity" and not "expired yesterday".

Writes are atomic (tmp file + os.replace): a crash mid-save cannot corrupt the
last good snapshot. The format is plain JSON — grep it, ship it, or write a
Redis/S3 adapter against the same take_snapshot/restore_snapshot pair.
"""
from __future__ import annotations

import json
import os
import time
from typing import Any, Mapping

from .affinity import SessionPolicy, SessionStore
from .backends import BackendHealth
from .delegation import PairProfile, PairRegistry
from .signals import FingerprintRegistry
from .types import WorkloadClass, WorkloadSource

SNAPSHOT_FORMAT = "llmrouter-state-v1"
SNAPSHOT_VERSION = 1

MappingOrNone = Mapping[str, Any] | None


# --------------------------------------------------------------------------- #
# (de)serialization of one state holder at a time
# --------------------------------------------------------------------------- #
def _dump_sessions(store: SessionStore, mono: float) -> dict[str, Any]:
    def age(v: float | None) -> float | None:
        return None if v is None else mono - v

    out: dict[str, Any] = {}
    for key, s in store._sessions.items():
        out[key] = {
            "workload": s.workload.value, "source": s.source.value,
            "confidence": s.confidence,
            "pinned_model": s.pinned_model, "pinned_backend": s.pinned_backend,
            "pinned_at_age": age(s.pinned_at),
            "switched_at_age": age(s.switched_at),
            "expires_in": age(s.expires_at) if s.expires_at else 0.0,
            "turns_seen": s.turns_seen, "prefix_tokens": s.prefix_tokens,
            "observed_cache_hit_rate": s.observed_cache_hit_rate,
            "ewma_hits": s.ewma_hits, "ewma_total": s.ewma_total,
            "stranded_count": s.stranded_count,
            "invalidators_hash": s.invalidators_hash,
            "last_seen_age": age(s.last_seen),
        }
    return out


def _load_sessions(store: SessionStore, data: MappingOrNone, now: float) -> None:
    if not data:
        return
    for key, d in data.items():
        exp = d.get("expires_in")
        sess = SessionPolicy(
            session_key=key,
            workload=WorkloadClass(d["workload"]),
            source=WorkloadSource(d["source"]),
            confidence=float(d["confidence"]),
            pinned_model=d["pinned_model"],
            pinned_backend=d["pinned_backend"],
            pinned_at=now - float(d.get("pinned_at_age") or 0.0),
            switched_at=(now - float(d["switched_at_age"])
                         if d.get("switched_at_age") is not None else None),
            expires_at=(now - float(exp)) if exp else 0.0,
            turns_seen=int(d.get("turns_seen", 0)),
            prefix_tokens=int(d.get("prefix_tokens", 0)),
            observed_cache_hit_rate=d.get("observed_cache_hit_rate"),
            ewma_hits=float(d.get("ewma_hits", 0.0)),
            ewma_total=float(d.get("ewma_total", 0.0)),
            stranded_count=int(d.get("stranded_count", 0)),
            invalidators_hash=d.get("invalidators_hash"),
            last_seen=now - float(d.get("last_seen_age") or 0.0),
        )
        store._sessions[key] = sess


def _dump_health(backends: Any, mono: float) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for bid, h in backends._health.items():
        out[bid] = {
            "successes": h.successes, "failures": h.failures,
            "consecutive_failures": h.consecutive_failures,
            "ewma_latency_ms": h.ewma_latency_ms,
            "p99_latency_ms": h.p99_latency_ms,
            "open_until_age": mono - h.open_until if h.open_until else 0.0,
            "rate_limited_until_age":
                mono - h.rate_limited_until if h.rate_limited_until else 0.0,
            "last_error": h.last_error,
        }
    return out


def _load_health(backends: Any, data: MappingOrNone, now: float) -> None:
    if not data:
        return
    for bid, d in data.items():
        h = backends.health(bid)  # creates if the backend still exists
        h.successes = int(d.get("successes", 0))
        h.failures = int(d.get("failures", 0))
        h.consecutive_failures = int(d.get("consecutive_failures", 0))
        h.ewma_latency_ms = float(d.get("ewma_latency_ms", 0.0))
        h.p99_latency_ms = float(d.get("p99_latency_ms", 0.0))
        # in_flight and half_open_probes are deliberately NOT restored: the
        # requests they count died with the old process. Restoring them would
        # make a restart look like saturation.
        h.half_open_probes = 0
        h.in_flight = 0
        open_age = float(d.get("open_until_age") or 0.0)
        h.open_until = (now - open_age) if open_age > 0 else 0.0
        rl_age = float(d.get("rate_limited_until_age") or 0.0)
        h.rate_limited_until = (now - rl_age) if rl_age > 0 else 0.0
        h.last_error = d.get("last_error")


def _dump_fingerprints(fpr: FingerprintRegistry) -> dict[str, Any]:
    return {
        "map": {
            fp: {"workload": sig.workload.value, "source": sig.source.value,
                 "confidence": sig.confidence, "reason": sig.reason}
            for fp, sig in fpr._map.items()
        },
        "seen_by_client": {c: sorted(fps) for c, fps in fpr._seen_by_client.items()},
    }


def _load_fingerprints(fpr: FingerprintRegistry, data: MappingOrNone) -> None:
    if not data:
        return
    from .signals import WorkloadSignal
    for fp, d in (data.get("map") or {}).items():
        fpr._map[fp] = WorkloadSignal(
            WorkloadClass(d["workload"]), WorkloadSource(d["source"]),
            float(d["confidence"]), str(d.get("reason", "")),
        )
    for c, fps in (data.get("seen_by_client") or {}).items():
        fpr._seen_by_client.setdefault(c, set()).update(fps)


def _dump_pairs(pairs: PairRegistry) -> dict[str, Any]:
    return {
        f"{lead}\x1f{side}": {
            "lead_model": lead, "sidekick_model": side,
            "rework_factor": p.rework_factor,
            "lead_overhead_tokens": p.lead_overhead_tokens,
            "sidekick_attempts": p.sidekick_attempts,
            "lead_share": p.lead_share,
            "measured_cost_per_task": p.measured_cost_per_task,
            "measured_score": p.measured_score,
        }
        for (lead, side), p in pairs.profiles.items()
    }


def _load_pairs(pairs: PairRegistry, data: MappingOrNone) -> None:
    if not data:
        return
    for d in data.values():
        pairs.profiles[(d["lead_model"], d["sidekick_model"])] = PairProfile(
            lead_model=d["lead_model"], sidekick_model=d["sidekick_model"],
            rework_factor=float(d.get("rework_factor", 1.0)),
            lead_overhead_tokens=float(d.get("lead_overhead_tokens", 800.0)),
            sidekick_attempts=float(d.get("sidekick_attempts", 1.0)),
            lead_share=float(d.get("lead_share", 0.5)),
            measured_cost_per_task=d.get("measured_cost_per_task"),
            measured_score=d.get("measured_score"),
        )


# --------------------------------------------------------------------------- #
# The snapshot pair
# --------------------------------------------------------------------------- #
def take_snapshot(router: Any) -> dict[str, Any]:
    """Capture every mutable learning state as a JSON-safe dict."""
    now = time.monotonic()
    return {
        "format": SNAPSHOT_FORMAT,
        "version": SNAPSHOT_VERSION,
        "saved_wall": time.time(),
        "sessions": _dump_sessions(router.sessions, now),
        "backend_health": _dump_health(router.backends, now),
        "fingerprints": _dump_fingerprints(router.resolver.fingerprints),
        "pairs": _dump_pairs(router.pairs),
        "counters": {
            "decisions": router._decisions,
            "cost_usd": router._cost_usd,
            "misroutes": router._misroutes,
            "delegations": router._delegations,
        },
    }


def restore_snapshot(router: Any, snap: Mapping[str, Any]) -> None:
    """Rebase timestamps onto the CURRENT monotonic clock and put state back.

    Unknown models/backends in the snapshot are kept verbatim: registries are
    mutable and a re-deploy may register them a moment later. A session whose
    pinned backend no longer exists simply fails the health check and gets
    migrated by the normal forced-failover path — which is the honest outcome.
    """
    if snap.get("format") != SNAPSHOT_FORMAT:
        raise ValueError(f"not an llmrouter state snapshot: {snap.get('format')!r}")
    now = time.monotonic()
    _load_sessions(router.sessions, snap.get("sessions"), now)
    _load_health(router.backends, snap.get("backend_health"), now)
    _load_fingerprints(router.resolver.fingerprints, snap.get("fingerprints"))
    _load_pairs(router.pairs, snap.get("pairs"))
    counters = snap.get("counters") or {}
    router._decisions = int(counters.get("decisions", router._decisions))
    router._cost_usd = float(counters.get("cost_usd", router._cost_usd))
    router._misroutes = int(counters.get("misroutes", router._misroutes))
    router._delegations = int(counters.get("delegations", router._delegations))


# --------------------------------------------------------------------------- #
# The file-backed store
# --------------------------------------------------------------------------- #
class FileStateStore:
    """JSON snapshot on disk. Zero dependencies, atomic writes.

    `autosave_every` bounds write amplification: state is folded in per
    decision, and fsyncing per decision is madness for a hot path. Tune to how
    much learning you can afford to lose a crash (default: 25 decisions ≈ the
    EWMA horizon).
    """

    def __init__(self, path: str, autosave_every: int = 25) -> None:
        self.path = path
        self.autosave_every = max(1, autosave_every)
        self._pending = 0

    # ---- the router calls these ---------------------------------------- #
    def attach(self, router: Any) -> bool:
        """Load existing state into the router at construction. Returns whether
        a snapshot was found — the first boot legitimately has none."""
        restored = self.load(router)
        router._state_store = self  # type: ignore[attr-defined]
        return restored

    def tick(self, router: Any) -> None:
        self._pending += 1
        if self._pending >= self.autosave_every:
            self._pending = 0
            self.save(router)

    # ---- explicit I/O --------------------------------------------------- #
    def save(self, router: Any) -> None:
        snap = take_snapshot(router)
        tmp = f"{self.path}.tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(snap, f, sort_keys=True)
        os.replace(tmp, self.path)

    def load(self, router: Any) -> bool:
        if not os.path.exists(self.path):
            return False
        with open(self.path, encoding="utf-8") as f:
            restore_snapshot(router, json.load(f))
        return True


__all__ = [
    "SNAPSHOT_FORMAT",
    "FileStateStore",
    "take_snapshot",
    "restore_snapshot",
]
