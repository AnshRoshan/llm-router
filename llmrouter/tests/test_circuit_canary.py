"""Half-open circuit recovery is gated behind a single canary probe."""
from __future__ import annotations

from llmrouter import (
    BackendRegistry,
    BackendSpec,
    DeploymentClass,
    HALF_OPEN_PROBE_LIMIT,
)


def make_registry() -> BackendRegistry:
    return BackendRegistry([
        BackendSpec("api", DeploymentClass.API, models=("m",), priority=0),
    ])


def test_circuit_opens_after_consecutive_failures():
    reg = make_registry()
    h = reg.health("api")
    for _ in range(5):
        h.record_failure("boom", now=100.0)
    assert not h.is_available(now=100.5)
    assert reg.available_for("m", now=100.5) == ()


def test_half_open_admits_one_canary_only():
    reg = make_registry()
    h = reg.health("api")
    for _ in range(5):
        h.record_failure("boom", now=100.0)
    # Backoff elapsed: the circuit is HALF_OPEN (open_until in the past).
    now = 100.0 + 1.0
    assert h.is_half_open(now)
    assert len(reg.available_for("m", now)) == 1

    # A probe claimed the slot: no further calls are admitted until it lands.
    h.half_open_probes = 1
    assert reg.available_for("m", now) == ()

    # The canary succeeded: the circuit closes fully.
    h.record_success(50.0, now=now)
    assert h.half_open_probes == 0
    assert len(reg.available_for("m", now)) == 1


def test_failed_canary_reopens_and_revokes_the_slot():
    reg = make_registry()
    h = reg.health("api")
    for _ in range(5):
        h.record_failure("boom", now=100.0)
    h.half_open_probes = 1
    # The probe failed: backoff extends and the canary allowance is revoked.
    h.record_failure("boom again", now=100.0 + 1.0)
    assert h.half_open_probes == 0
    assert not h.is_available(now=100.0 + 1.5)
