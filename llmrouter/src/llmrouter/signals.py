"""Workload-class inference: the two entry paths.

PATH A (declared) — the caller passed a tag or used a named agent profile.
    Cost: ~0 ms. Confidence: 1.0. Must skip the classifier entirely — not
    "run it and ignore the result", but never construct it.

PATH B (inferred) — the caller declared nothing.
    B1 structural rules   ~0 ms, free, resolves most production traffic
    B2 prompt fingerprint ~0 ms, exact, once per *application*
    B3 classifier hook    only when B1/B2 abstain; turn 1 only
                          (ready implementations in llmrouter.classifiers)
    B4 workload default   never fail to route; logged as 'default'

The economics that justify B3: the decision cost is paid ONCE and amortized over
the session, while the decision value is multiplied by it. This is the one place
an LLM-based classifier is affordable — the literature's case against LLM routing
is a case against per-*step* routing, not turn-1 routing.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping, Protocol

from .types import RoutingRequest, WorkloadClass, WorkloadSource

#: Fallback turns-per-session priors, used when the caller cannot say.
#: These are assumptions to calibrate, not measurements.
TURNS_PRIOR: Mapping[WorkloadClass, int] = {
    WorkloadClass.AGENT: 25,
    WorkloadClass.CHAT: 6,
    WorkloadClass.RAG: 2,
    WorkloadClass.BATCH: 1,
    WorkloadClass.CODE: 12,
    WorkloadClass.VISION: 3,
    WorkloadClass.UNKNOWN: 5,
}

#: Expected idle gap between turns, seconds. Agents are seconds; humans are not.
GAP_PRIOR: Mapping[WorkloadClass, float] = {
    WorkloadClass.AGENT: 5.0,
    WorkloadClass.CHAT: 90.0,
    WorkloadClass.RAG: 45.0,
    WorkloadClass.BATCH: 0.0,
    WorkloadClass.CODE: 20.0,
    WorkloadClass.VISION: 60.0,
    WorkloadClass.UNKNOWN: 60.0,
}


@dataclass(frozen=True)
class WorkloadSignal:
    workload: WorkloadClass
    source: WorkloadSource
    confidence: float
    reason: str


class WorkloadClassifier(Protocol):
    """Optional hook for a trained/prompted classifier.

    Called ONLY on turn 1 and ONLY when structural rules and the fingerprint
    registry both abstain. Implementations must be cheap relative to the session:
    budget ≈ 1% of estimated session cost.
    """

    def classify(self, request: RoutingRequest) -> WorkloadSignal | None: ...


# --------------------------------------------------------------------------- #
# Path B1 — structural rules
# --------------------------------------------------------------------------- #
_CODE_MARKERS = ("```", "def ", "class ", "function ", "import ", "const ", "SELECT ")


def infer_structural(req: RoutingRequest) -> WorkloadSignal | None:
    """Zero-cost inference from the shape of the request.

    These features are already in the request object — no model call, no extra
    tokenization. They resolve most production traffic outright.
    """
    if req.has_images():
        return WorkloadSignal(WorkloadClass.VISION, WorkloadSource.STRUCTURAL, 0.99,
                              "image content block present")

    if req.tools and req.has_tool_results():
        return WorkloadSignal(WorkloadClass.AGENT, WorkloadSource.STRUCTURAL, 0.97,
                              "tools present and prior tool_result in history")

    if req.tools:
        return WorkloadSignal(WorkloadClass.AGENT, WorkloadSource.STRUCTURAL, 0.85,
                              "tools present, no tool_result yet (first step)")

    if req.retrieved:
        return WorkloadSignal(WorkloadClass.RAG, WorkloadSource.STRUCTURAL, 0.90,
                              "retrieved chunks supplied with the request")

    if "batch" in req.tags:
        return WorkloadSignal(WorkloadClass.BATCH, WorkloadSource.STRUCTURAL, 1.0,
                              "tagged batch")

    user_text = req.first_user_text()
    code_hits = sum(1 for m in _CODE_MARKERS if m in user_text)
    if code_hits >= 2 or ("code" in req.tags):
        return WorkloadSignal(WorkloadClass.CODE, WorkloadSource.STRUCTURAL, 0.70,
                              f"{code_hits} code markers in the first user turn")

    if req.turn_index >= 2 and not req.tools:
        return WorkloadSignal(WorkloadClass.CHAT, WorkloadSource.STRUCTURAL, 0.80,
                              "multi-turn, no tools")

    return None  # abstain — let the fingerprint registry / classifier decide


# --------------------------------------------------------------------------- #
# Path B2 — prompt fingerprint registry
# --------------------------------------------------------------------------- #
class FingerprintRegistry:
    """Maps a prompt fingerprint to a workload class.

    Production traffic comes from a handful of distinct application prompts, so
    this turns "classify every request" into "classify every application" — a set
    that is typically single digits. A hit is exact, free, and confidence 1.0.

    A useful side effect: a client whose prompts embed a timestamp or request id
    produces unbounded distinct fingerprints and will never hit. Watch
    `cardinality_by_client` — that warning is simultaneously a cache-hit-rate bug
    report for the user's own prompts.
    """

    def __init__(self) -> None:
        self._map: dict[str, WorkloadSignal] = {}
        self._seen_by_client: dict[str, set[str]] = {}

    def get(self, fingerprint: str) -> WorkloadSignal | None:
        return self._map.get(fingerprint)

    def put(self, fingerprint: str, signal: WorkloadSignal) -> None:
        # Never let a low-confidence guess overwrite a known-good mapping.
        existing = self._map.get(fingerprint)
        if existing is not None and existing.confidence >= signal.confidence:
            return
        self._map[fingerprint] = signal

    def observe(self, fingerprint: str, client_id: str | None) -> None:
        self._seen_by_client.setdefault(client_id or "anon", set()).add(fingerprint)

    def cardinality_by_client(self) -> dict[str, int]:
        return {k: len(v) for k, v in self._seen_by_client.items()}

    def volatile_clients(self, threshold: int = 50) -> tuple[str, ...]:
        """Clients generating too many distinct prompts to ever cache."""
        return tuple(c for c, fps in self._seen_by_client.items() if len(fps) > threshold)

    def __len__(self) -> int:
        return len(self._map)


# --------------------------------------------------------------------------- #
# The resolver
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class AgentProfile:
    """A named agent the caller can select — the 'user chose an agent' path."""

    name: str
    workload: WorkloadClass
    allowed_models: tuple[str, ...] = ()
    sticky: bool = True
    tags: frozenset[str] = frozenset()
    expected_turns: int | None = None


class WorkloadResolver:
    """Resolves a workload class via Path A then Path B."""

    def __init__(
        self,
        profiles: Mapping[str, AgentProfile] | None = None,
        fingerprints: FingerprintRegistry | None = None,
        classifier: WorkloadClassifier | None = None,
        default_workload: WorkloadClass = WorkloadClass.CHAT,
    ) -> None:
        self.profiles: dict[str, AgentProfile] = dict(profiles or {})
        self.fingerprints = fingerprints if fingerprints is not None else FingerprintRegistry()
        self.classifier = classifier
        self.default_workload = default_workload

    def add_profile(self, profile: AgentProfile) -> None:
        self.profiles[profile.name] = profile

    def resolve(self, req: RoutingRequest) -> WorkloadSignal:
        # ---- PATH A: declared ------------------------------------------- #
        if req.agent_profile:
            profile = self.profiles.get(req.agent_profile)
            if profile is not None:
                return WorkloadSignal(
                    profile.workload, WorkloadSource.AGENT_PROFILE, 1.0,
                    f"agent profile {profile.name!r}",
                )
            # Unknown profile: fall through, but say so.
            return WorkloadSignal(
                self.default_workload, WorkloadSource.DEFAULT, 0.2,
                f"unknown agent profile {req.agent_profile!r}",
            )

        if req.workload is not None:
            return WorkloadSignal(req.workload, WorkloadSource.EXPLICIT_TAG, 1.0,
                                  "caller declared workload")

        # ---- PATH B: inferred ------------------------------------------- #
        fp = req.prompt_fingerprint()
        self.fingerprints.observe(fp, req.client_id)

        hit = self.fingerprints.get(fp)
        if hit is not None:
            return WorkloadSignal(hit.workload, WorkloadSource.FINGERPRINT, 1.0,
                                  "prompt fingerprint matched a known application")

        structural = infer_structural(req)
        if structural is not None:
            # Learn it, so the next request with this prompt is a free exact hit.
            self.fingerprints.put(fp, structural)
            return structural

        # Only abstain here — and only on turn 1 — do we pay for a classifier.
        if self.classifier is not None and req.is_first_turn:
            got = self.classifier.classify(req)
            if got is not None:
                self.fingerprints.put(fp, got)
                return got

        fallback = WorkloadSignal(
            self.default_workload, WorkloadSource.DEFAULT, 0.3,
            f"no signal; using default {self.default_workload.value}",
        )
        self.fingerprints.put(fp, fallback)
        return fallback


__all__ = [
    "TURNS_PRIOR",
    "GAP_PRIOR",
    "WorkloadSignal",
    "WorkloadClassifier",
    "infer_structural",
    "FingerprintRegistry",
    "AgentProfile",
    "WorkloadResolver",
]
