"""Core types. Zero dependencies — stdlib only.

The vocabulary of the router. Everything here is a frozen dataclass or an enum so
that decisions are reproducible and hashable.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping, Sequence


# --------------------------------------------------------------------------- #
# Workload classes (Axis 1)
# --------------------------------------------------------------------------- #
class WorkloadClass(str, Enum):
    """What the request *is*. Drives which policy gets selected."""

    AGENT = "agent"
    CHAT = "chat"
    RAG = "rag"
    BATCH = "batch"
    CODE = "code"
    VISION = "vision"
    UNKNOWN = "unknown"


class WorkloadSource(str, Enum):
    """How we learned the workload class. Emitted on every decision so you can
    measure per-source quality — without this you cannot tell whether the
    classifier is helping."""

    EXPLICIT_TAG = "explicit_tag"      # caller passed workload=
    AGENT_PROFILE = "agent_profile"    # caller used a named agent/profile
    FINGERPRINT = "fingerprint"        # known application prompt (registry hit)
    STRUCTURAL = "structural"          # inferred from request shape
    CLASSIFIER = "classifier"          # model call, turn 1 only
    DEFAULT = "default"                # fallback; logged and auditable


class DeploymentClass(str, Enum):
    """Where the model runs (Axis 2)."""

    API = "api"
    SELF_HOSTED = "self_hosted"
    SERVERLESS = "serverless"
    EDGE = "edge"


class SegmentKind(str, Enum):
    """A slice of the prompt prefix, for per-breakpoint TTL policy."""

    TOOLS = "tools"
    SYSTEM = "system"
    RETRIEVED = "retrieved"
    HISTORY = "history"
    USER = "user"


# --------------------------------------------------------------------------- #
# Messages
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Message:
    role: str
    content: str | Sequence[Any] = ""

    @property
    def text(self) -> str:
        if isinstance(self.content, str):
            return self.content
        # content blocks: pick out the text ones
        out = []
        for block in self.content:
            if isinstance(block, Mapping):
                if block.get("type") == "text":
                    out.append(str(block.get("text", "")))
                elif block.get("type") == "tool_result":
                    out.append(str(block.get("content", "")))
            else:
                out.append(str(block))
        return "\n".join(out)

    def has_block_type(self, block_type: str) -> bool:
        if isinstance(self.content, str):
            return False
        return any(
            isinstance(b, Mapping) and b.get("type") == block_type for b in self.content
        )


# --------------------------------------------------------------------------- #
# The request
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Constraints:
    max_cost_usd: float | None = None
    max_latency_ms: float | None = None
    budget_remaining_usd: float | None = None
    residency_tags: frozenset[str] = frozenset()
    require_tools: bool = False


@dataclass(frozen=True)
class RoutingRequest:
    """Everything the router is allowed to look at.

    `workload` and `session_id` are OPTIONAL by design: the router must work when
    the caller declares nothing (see :mod:`llmrouter.signals`).
    """

    messages: tuple[Message, ...]
    workload: WorkloadClass | None = None
    session_id: str | None = None
    agent_profile: str | None = None
    tools: tuple[Mapping[str, Any], ...] = ()
    retrieved: tuple[str, ...] = ()
    tags: frozenset[str] = frozenset()
    constraints: Constraints = field(default_factory=Constraints)

    # Session-shape hints. Optional; the router falls back to per-workload priors.
    expected_turns: int | None = None
    expected_gap_s: float | None = None
    expected_generation_s: float | None = None

    client_id: str | None = None

    # ---- derived helpers ------------------------------------------------- #
    @property
    def turn_index(self) -> int:
        """Number of prior user turns. 1 == first turn of a conversation."""
        return sum(1 for m in self.messages if m.role == "user")

    @property
    def is_first_turn(self) -> bool:
        return self.turn_index <= 1

    def first_user_text(self) -> str:
        for m in self.messages:
            if m.role == "user":
                return m.text
        return ""

    def system_text(self) -> str:
        return "\n".join(m.text for m in self.messages if m.role == "system")

    def has_tool_results(self) -> bool:
        return any(
            m.role in ("tool", "user") and m.has_block_type("tool_result")
            for m in self.messages
        )

    def has_images(self) -> bool:
        return any(
            m.has_block_type("image") or m.has_block_type("image_url")
            for m in self.messages
        )

    def prompt_fingerprint(self) -> str:
        """Hash of the stable part of the prompt: system text + tool schemas.

        This is what turns "classify every request" into "classify every
        *application*" — production traffic comes from a handful of distinct
        system prompts.
        """
        tool_schemas = json.dumps(
            [dict(t) for t in self.tools], sort_keys=True, separators=(",", ":")
        )
        material = f"{self.system_text()}\x00{tool_schemas}".encode("utf-8")
        return hashlib.blake2b(material, digest_size=16).hexdigest()

    def sticky_key(self) -> str:
        """Derive a session key WITHOUT requiring the caller to send one.

        Priority:
          1. caller-declared session_id
          2. hash of the first user turn — two requests in the same conversation
             share their turn-1 prefix byte-for-byte, which is precisely what
             makes prompt caching work, so it doubles as a session id.
          3. per-application key from the prompt fingerprint
        """
        if self.session_id:
            return f"sid:{self.session_id}"
        first = self.first_user_text()
        if first:
            h = hashlib.blake2b(first.encode("utf-8"), digest_size=16).hexdigest()
            return f"ctx:{h}"
        return f"fp:{self.prompt_fingerprint()}:{self.client_id or 'anon'}"


# --------------------------------------------------------------------------- #
# Decisions
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Candidate:
    model_id: str
    backend_id: str
    score: float
    quality: float
    cost_usd: float
    latency_ms: float
    affinity: float
    switch_cost_usd: float
    reasons: tuple[str, ...] = ()


@dataclass(frozen=True)
class Decision:
    """The output. `explain()` is not optional — every decision is auditable."""

    model_id: str
    backend_id: str
    workload: WorkloadClass
    workload_source: WorkloadSource
    workload_confidence: float
    sticky_key: str
    session_key: str
    is_first_turn: bool
    reused_session: bool
    switched: bool
    stranded: bool
    ttl_by_segment: Mapping[SegmentKind, str]
    candidates: tuple[Candidate, ...]
    reasons: tuple[str, ...]

    # filled in after execution
    cache_write_tokens: int = 0
    cache_read_tokens: int = 0
    cost_usd: float = 0.0
    latency_ms: float = 0.0
    error: str | None = None

    def explain(self) -> str:
        head = (
            f"model={self.model_id} backend={self.backend_id} "
            f"workload={self.workload.value}(src={self.workload_source.value}, "
            f"conf={self.workload_confidence:.2f}) "
            f"sticky={self.reused_session} switched={self.switched} "
            f"stranded={self.stranded}"
        )
        ttl = ", ".join(f"{k.value}={v}" for k, v in self.ttl_by_segment.items())
        lines = [head, f"  ttl: {ttl or 'n/a'}"]
        for c in self.candidates[:5]:
            lines.append(
                f"  {c.model_id}@{c.backend_id} score={c.score:+.4f} "
                f"q={c.quality:.3f} cost=${c.cost_usd:.5f} "
                f"lat={c.latency_ms:.0f}ms aff={c.affinity:.2f} "
                f"switch=${c.switch_cost_usd:.5f}"
            )
        for r in self.reasons:
            lines.append(f"  · {r}")
        return "\n".join(lines)


__all__ = [
    "WorkloadClass",
    "WorkloadSource",
    "DeploymentClass",
    "SegmentKind",
    "Message",
    "Constraints",
    "RoutingRequest",
    "Candidate",
    "Decision",
]
