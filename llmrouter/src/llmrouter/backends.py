"""Backend registry, live health signals, and the circuit breaker.

Axis 2 of the design: *where* the model runs. The literature almost never models
this, which is why the router scores (model, backend) PAIRS rather than models.

Two operational facts that shape this module:
  * LLM serving pods take **5–15 minutes** to become ready (pull weights, load to
    VRAM, warm KV cache), so reactive autoscaling is useless. Route AROUND cold
    capacity rather than waiting for it.
  * A 10× traffic spike that saturates a local GPU cascades every request to the
    cloud at ~100× the cost. Local timeouts must be aggressive, not 30s.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Mapping, Sequence

from .types import DeploymentClass

#: Aggressive local timeout. The cascade trap: a saturated GPU times everything
#: out, and everything then lands on the expensive path at once.
DEFAULT_LOCAL_TIMEOUT_S = 8.0
DEFAULT_API_TIMEOUT_S = 60.0


@dataclass(frozen=True)
class BackendSpec:
    backend_id: str
    deployment: DeploymentClass = DeploymentClass.API
    models: tuple[str, ...] = ()
    base_url: str | None = None
    residency_tags: frozenset[str] = frozenset()
    priority: int = 0
    """Lower is preferred. Used to order the fallback chain."""
    timeout_s: float = DEFAULT_API_TIMEOUT_S
    max_concurrency: int = 64
    #: Serverless/cold-start backends carry extra latency risk.
    cold_start_risk: float = 0.0
    #: Self-hosted backends have a marginal cost near zero but finite capacity.
    cost_factor: float = 1.0
    """Multiplier on the model's list price. Self-hosted ≈ 0.1–0.3 of API price
    once you're above the GPU duty-cycle break-even, but see `capacity`."""
    capacity_rps: float | None = None
    """If set, the backend sheds load past this rate instead of queueing."""

    def supports(self, model_id: str) -> bool:
        return not self.models or model_id in self.models


@dataclass
class BackendHealth:
    """Live signals. Updated by the execution layer from real responses."""

    backend_id: str
    in_flight: int = 0
    successes: int = 0
    failures: int = 0
    consecutive_failures: int = 0
    ewma_latency_ms: float = 0.0
    p99_latency_ms: float = 0.0
    open_until: float = 0.0
    """Circuit open until this monotonic timestamp."""
    half_open_probes: int = 0
    rate_limited_until: float = 0.0
    last_error: str | None = None
    _latency_alpha: float = 0.2

    @property
    def error_rate(self) -> float:
        total = self.successes + self.failures
        return self.failures / total if total else 0.0

    def is_available(self, now: float | None = None) -> bool:
        now = now if now is not None else time.monotonic()
        if self.rate_limited_until and now < self.rate_limited_until:
            return False
        return now >= self.open_until

    def is_half_open(self, now: float | None = None) -> bool:
        now = now if now is not None else time.monotonic()
        return 0 < self.open_until <= now

    def record_success(self, latency_ms: float, now: float | None = None) -> None:
        now = now if now is not None else time.monotonic()
        self.successes += 1
        self.consecutive_failures = 0
        self.ewma_latency_ms = (
            (1 - self._latency_alpha) * self.ewma_latency_ms
            + self._latency_alpha * latency_ms
        )
        self.p99_latency_ms = max(self.p99_latency_ms * 0.95, latency_ms)
        self.open_until = 0.0
        self.half_open_probes = 0
        self.last_error = None

    def record_failure(self, error: str, retry_after_s: float | None = None,
                       now: float | None = None) -> None:
        now = now if now is not None else time.monotonic()
        self.failures += 1
        self.consecutive_failures += 1
        self.last_error = error
        if retry_after_s:
            self.rate_limited_until = now + retry_after_s
        if self.consecutive_failures >= CIRCUIT_FAILURE_THRESHOLD:
            backoff = min(
                CIRCUIT_BASE_BACKOFF_S * (2 ** (self.consecutive_failures
                                                - CIRCUIT_FAILURE_THRESHOLD)),
                CIRCUIT_MAX_BACKOFF_S,
            )
            self.open_until = now + backoff
            # A failure while half-open re-opens the circuit; any in-flight
            # canary allowance is revoked with it.
            self.half_open_probes = 0

    def saturating(self, spec: BackendSpec) -> bool:
        """True when the backend should be shed rather than queued."""
        if self.in_flight >= spec.max_concurrency:
            return True
        if spec.capacity_rps is not None and self.in_flight > spec.capacity_rps:
            return True
        return False


CIRCUIT_FAILURE_THRESHOLD = 5
CIRCUIT_BASE_BACKOFF_S = 1.0
CIRCUIT_MAX_BACKOFF_S = 60.0

#: How many probe requests may be in flight against a HALF_OPEN backend. Full
#: recovery is gated behind a cheap canary: one probe, and only after it
#: succeeds does the circuit close. Never probe by calling the provider for
#: health's sake — the canary is a real request that was going out anyway.
HALF_OPEN_PROBE_LIMIT = 1


class BackendRegistry:
    def __init__(self, backends: Sequence[BackendSpec] = ()) -> None:
        self._specs: dict[str, BackendSpec] = {}
        self._health: dict[str, BackendHealth] = {}
        for b in backends:
            self.register(b)

    def register(self, spec: BackendSpec) -> None:
        self._specs[spec.backend_id] = spec
        self._health.setdefault(spec.backend_id, BackendHealth(spec.backend_id))

    def unregister(self, backend_id: str) -> None:
        self._specs.pop(backend_id, None)
        self._health.pop(backend_id, None)

    def spec(self, backend_id: str) -> BackendSpec | None:
        return self._specs.get(backend_id)

    def health(self, backend_id: str) -> BackendHealth:
        h = self._health.get(backend_id)
        if h is None:
            h = BackendHealth(backend_id)
            self._health[backend_id] = h
        return h

    def candidates_for(self, model_id: str) -> tuple[BackendSpec, ...]:
        return tuple(b for b in self._specs.values() if b.supports(model_id))

    def available_for(self, model_id: str, now: float | None = None) -> tuple[BackendSpec, ...]:
        """Healthy, non-saturated backends that serve this model, by priority.

        A HALF_OPEN backend is admitted only while a canary slot is free: full
        recovery waits for one real request to succeed.
        """
        now = now if now is not None else time.monotonic()
        out = []
        for b in self.candidates_for(model_id):
            h = self.health(b.backend_id)
            if not h.is_available(now) or h.saturating(b):
                continue
            if h.is_half_open(now) and h.half_open_probes >= HALF_OPEN_PROBE_LIMIT:
                continue
            out.append(b)
        return tuple(sorted(out, key=lambda b: b.priority))

    def all_available(self, now: float | None = None) -> tuple[BackendSpec, ...]:
        now = now if now is not None else time.monotonic()
        return tuple(
            b for b in self._specs.values() if self.health(b.backend_id).is_available(now)
        )

    def ids(self) -> tuple[str, ...]:
        return tuple(self._specs)

    def stats(self) -> Mapping[str, Mapping[str, float]]:
        return {
            bid: {
                "in_flight": float(h.in_flight),
                "error_rate": h.error_rate,
                "ewma_latency_ms": h.ewma_latency_ms,
                "consecutive_failures": float(h.consecutive_failures),
                "circuit_open": 1.0 if h.open_until else 0.0,
            }
            for bid, h in self._health.items()
        }

    def __len__(self) -> int:
        return len(self._specs)


__all__ = [
    "DEFAULT_LOCAL_TIMEOUT_S",
    "DEFAULT_API_TIMEOUT_S",
    "CIRCUIT_FAILURE_THRESHOLD",
    "CIRCUIT_BASE_BACKOFF_S",
    "CIRCUIT_MAX_BACKOFF_S",
    "BackendSpec",
    "BackendHealth",
    "BackendRegistry",
]
