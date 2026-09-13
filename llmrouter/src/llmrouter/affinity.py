"""Session affinity: sticky keys, rendezvous hashing, and the session store.

Rule A from the research: session affinity is the DEFAULT, not an optimization.
Route by a stable key to a (model, backend) pair, with overflow accepting a cache
miss rather than breaking affinity for the whole session.

Rendezvous (HRW) hashing is used rather than a hash ring because it is stable
under membership change: adding or removing one backend remaps only that
backend's share. A ring rebalance is what silently converts a 90% cache hit rate
into a 12.5% one.
"""
from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from typing import Iterable, Sequence

from .types import WorkloadClass, WorkloadSource

DEFAULT_IDLE_TTL_S = 3600.0


# --------------------------------------------------------------------------- #
# Rendezvous hashing
# --------------------------------------------------------------------------- #
def _hrw_score(key: str, node: str) -> int:
    return int.from_bytes(
        hashlib.blake2b(f"{key}\x00{node}".encode("utf-8"), digest_size=8).digest(),
        "big",
    )


def rendezvous_pick(key: str, nodes: Sequence[str]) -> str | None:
    """Pick the highest-scoring node for `key`. Stable under membership change."""
    if not nodes:
        return None
    return max(nodes, key=lambda n: _hrw_score(key, n))


def rendezvous_rank(key: str, nodes: Sequence[str]) -> list[str]:
    """Full preference order, so overflow has a deterministic next choice."""
    return sorted(nodes, key=lambda n: _hrw_score(key, n), reverse=True)


# --------------------------------------------------------------------------- #
# Session state
# --------------------------------------------------------------------------- #
@dataclass
class SessionPolicy:
    """Mutable per-session routing state."""

    session_key: str
    workload: WorkloadClass
    source: WorkloadSource
    confidence: float
    pinned_model: str
    pinned_backend: str
    pinned_at: float
    switched_at: float | None = None
    expires_at: float = 0.0
    turns_seen: int = 0
    prefix_tokens: int = 0
    #: Learned, not assumed. Seeded by a prior, then updated from the usage
    #: counters the provider returns on every response.
    observed_cache_hit_rate: float | None = None
    ewma_hits: float = 0.0
    ewma_total: float = 0.0
    #: Incremented when a better model was available but the session had too few
    #: turns left to migrate. A real, surfaced metric.
    stranded_count: int = 0
    #: Fingerprint of the request fields that are part of the provider's cache
    #: key but outside the prefix (tool schemas, images). When it drifts within a
    #: live session the cache is gone — Rule F — and the session must be scored
    #: cold until the counters say otherwise.
    invalidators_hash: str | None = None
    last_seen: float = field(default_factory=time.monotonic)

    @property
    def already_switched(self) -> bool:
        return self.switched_at is not None

    def record_usage(self, cache_read_tokens: int, cache_write_tokens: int,
                     alpha: float = 0.3) -> None:
        """Fold the provider's usage counters into an EWMA hit rate.

        The cache lives on the provider's side, so the router's affinity estimate
        is a guess unless it learns. Anthropic returns
        `cache_read_input_tokens` / `cache_creation_input_tokens`; OpenAI returns
        `usage.prompt_tokens_details.cached_tokens`.
        """
        total = cache_read_tokens + cache_write_tokens
        if total <= 0:
            return
        hit = cache_read_tokens / total
        self.ewma_hits = (1 - alpha) * self.ewma_hits + alpha * hit
        self.ewma_total = (1 - alpha) * self.ewma_total + alpha * 1.0
        self.observed_cache_hit_rate = self.ewma_hits / self.ewma_total

    def hit_rate(self, prior: float) -> float:
        """Measured rate if we have enough samples, otherwise the prior."""
        if self.observed_cache_hit_rate is None or self.ewma_total < 0.5:
            return prior
        return self.observed_cache_hit_rate


class SessionStore:
    """In-process session store with idle eviction.

    Swap this for Redis/etc. in a distributed deployment; the interface is
    deliberately tiny. Note that affinity is only *useful* if the store is shared
    across processes — a per-process store makes affinity approximate.
    """

    def __init__(self, idle_ttl_s: float = DEFAULT_IDLE_TTL_S,
                 max_sessions: int = 100_000) -> None:
        self._sessions: dict[str, SessionPolicy] = {}
        self.idle_ttl_s = idle_ttl_s
        self.max_sessions = max_sessions

    def get(self, session_key: str, now: float | None = None) -> SessionPolicy | None:
        now = now if now is not None else time.monotonic()
        sess = self._sessions.get(session_key)
        if sess is None:
            return None
        if sess.expires_at and now > sess.expires_at:
            self._sessions.pop(session_key, None)
            return None
        sess.last_seen = now
        sess.turns_seen += 1
        return sess

    def put(self, session: SessionPolicy, now: float | None = None) -> None:
        now = now if now is not None else time.monotonic()
        if len(self._sessions) >= self.max_sessions and session.session_key not in self._sessions:
            self._evict_oldest(now)
        session.last_seen = now
        session.expires_at = now + self.idle_ttl_s
        self._sessions[session.session_key] = session

    def _evict_oldest(self, now: float) -> None:
        if not self._sessions:
            return
        oldest = min(self._sessions.values(), key=lambda s: s.last_seen)
        self._sessions.pop(oldest.session_key, None)

    def extend(self, session_key: str, now: float | None = None) -> None:
        now = now if now is not None else time.monotonic()
        sess = self._sessions.get(session_key)
        if sess is not None:
            sess.expires_at = now + self.idle_ttl_s

    def evict(self, session_key: str) -> None:
        self._sessions.pop(session_key, None)

    def clear(self) -> None:
        self._sessions.clear()

    def stats(self) -> dict[str, float]:
        if not self._sessions:
            return {"sessions": 0.0, "switched": 0.0, "stranded": 0.0}
        vals = list(self._sessions.values())
        return {
            "sessions": float(len(vals)),
            "switched": float(sum(1 for s in vals if s.already_switched)),
            "stranded": float(sum(s.stranded_count for s in vals)),
            "avg_hit_rate": float(
                sum(s.observed_cache_hit_rate or 0.0 for s in vals) / len(vals)
            ),
        }

    def __len__(self) -> int:
        return len(self._sessions)

    def __contains__(self, session_key: object) -> bool:
        return session_key in self._sessions


def affinity_prior(workload: WorkloadClass, gap_s: float, ttl_s: float = 300.0) -> float:
    """Prior cache-hit probability for a workload, before any measurement.

    Chat is bursty — humans think between turns — so its real hit rate is often
    far below an agent's even with identical stickiness. Do not use a uniform
    affinity bonus across workload classes.
    """
    if workload == WorkloadClass.BATCH:
        return 0.0  # single shot per prefix; affinity is pure scoring noise
    if gap_s <= 0:
        return 0.95
    if gap_s >= ttl_s:
        return 0.15  # the cache will usually have expired between turns
    # Linear decay inside the TTL window.
    return 0.95 - 0.5 * (gap_s / ttl_s)


def pick_backend(key: str, healthy: Iterable[str], current: str | None = None) -> str | None:
    """Affinity-first backend selection with deterministic overflow.

    If the currently-pinned backend is still healthy, keep it — that is the whole
    point. Otherwise take the top-ranked healthy alternative and accept the miss.
    """
    healthy = list(healthy)
    if not healthy:
        return None
    if current is not None and current in healthy:
        return current
    return rendezvous_pick(key, healthy)


__all__ = [
    "DEFAULT_IDLE_TTL_S",
    "rendezvous_pick",
    "rendezvous_rank",
    "SessionPolicy",
    "SessionStore",
    "affinity_prior",
    "pick_backend",
]
