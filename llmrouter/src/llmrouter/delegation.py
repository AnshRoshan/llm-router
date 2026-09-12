"""Task-phase routing and lead/sidekick delegation.

Motivated by Cognition's Fusion architecture (Thread 10 of the research dossier).
Two ideas, both of which our earlier design only had half of:

**1. Route on task PHASE, not difficulty.**
Difficulty is not a property of the prompt — it is a property of the
investigation. "Fix xyz bug" may be a one-liner or a rearchitecture, and you
cannot know until you have read the code. Phase, by contrast, IS observable from
the tool-call history. FrontierCode turn distribution: Plan 32%, Setup 34%,
Implementation 5%, Debug 17%, Validate 9%.

**2. Delegate instead of switching.**
Mid-session model switching destroys the prompt cache. Delegation does not: the
lead and the sidekick are two agents with two separate contexts and two separate
warm caches, exchanging only *briefs*. Neither prefix ever changes for the
other's sake, so both stay hot for the whole session.

Where delegation applies it strictly dominates switching: no cache write, no
oscillation risk, no stranded migrations.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Mapping, Sequence

from .types import Message, RoutingRequest


# --------------------------------------------------------------------------- #
# Task phases
# --------------------------------------------------------------------------- #
class TaskPhase(str, Enum):
    PLAN = "plan"
    SETUP = "setup"
    IMPLEMENT = "implement"
    DEBUG = "debug"
    VALIDATE = "validate"
    CLOSEOUT = "closeout"


#: Which phases are safe to delegate to a cheaper model.
#:
#: The non-obvious constraint, from Cognition: **exploration that shapes the plan
#: must stay with the lead.** A weaker model may not judge what information
#: matters, and delegating it starves the lead's context — which the lead needs in
#: its own cache to plan well. So DEBUG is delegable but PLAN is not.
DELEGABLE_PHASES = frozenset({
    TaskPhase.SETUP,
    TaskPhase.IMPLEMENT,
    TaskPhase.VALIDATE,
})

#: Phases that must stay with the lead regardless of cost.
LEAD_ONLY_PHASES = frozenset({TaskPhase.PLAN, TaskPhase.CLOSEOUT})

#: FrontierCode turn distribution, for weighting. Implementation is only 5% of
#: turns — the sidekick's win is absorbing high-token, low-judgement work, not
#: taking over the session.
PHASE_TURN_SHARE: Mapping[TaskPhase, float] = {
    TaskPhase.PLAN: 0.32,
    TaskPhase.SETUP: 0.34,
    TaskPhase.IMPLEMENT: 0.05,
    TaskPhase.DEBUG: 0.17,
    TaskPhase.VALIDATE: 0.09,
    TaskPhase.CLOSEOUT: 0.03,
}


_TOOL_PHASE_HINTS: tuple[tuple[frozenset[str], TaskPhase], ...] = (
    (frozenset({"test", "pytest", "run_tests", "unittest"}), TaskPhase.VALIDATE),
    (frozenset({"edit", "write", "apply_patch", "str_replace", "write_file"}),
     TaskPhase.IMPLEMENT),
    (frozenset({"read", "grep", "glob", "search", "list_dir", "cat"}), TaskPhase.SETUP),
    (frozenset({"debug", "trace", "profile", "gdb"}), TaskPhase.DEBUG),
)

_ERROR_MARKERS = ("error", "failed", "traceback", "assertionerror", "exception")


def infer_phase(
    messages: Sequence[Message],
    tools: Sequence[Mapping[str, object]] = (),
) -> TaskPhase:
    """Infer the current phase from tool-call history.

    Observable, unlike difficulty. Falls back to PLAN when there is no history,
    because that is the correct default for a fresh session — and planning is
    lead-only, so the fallback is the conservative one.
    """
    tool_names: list[str] = []
    saw_error = False
    for m in messages:
        text = m.text.lower()
        if m.role in ("tool", "user") and any(k in text for k in _ERROR_MARKERS):
            saw_error = True
        if isinstance(m.content, str):
            continue
        for block in m.content:
            if isinstance(block, Mapping):
                name = block.get("name") or block.get("tool_name")
                if name:
                    tool_names.append(str(name).lower())

    if not tool_names:
        return TaskPhase.PLAN

    if saw_error and len(tool_names) >= 3:
        return TaskPhase.DEBUG

    counts: dict[TaskPhase, int] = {}
    for name in tool_names:
        for hints, phase in _TOOL_PHASE_HINTS:
            if any(h in name for h in hints):
                counts[phase] = counts.get(phase, 0) + 1
                break
    if not counts:
        return TaskPhase.PLAN

    # Most recent phase wins ties: the tail of the history is what matters now.
    best = max(counts.items(), key=lambda kv: kv[1])
    return best[0]


# --------------------------------------------------------------------------- #
# Briefs — the only thing that crosses the boundary
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Brief:
    """A bounded unit of work handed from lead to sidekick.

    Deliberately NOT a conversation. The whole point is that neither agent's
    prefix grows because of the other, so both caches stay warm.
    """

    task: str
    constraints: tuple[str, ...] = ()
    success_criteria: tuple[str, ...] = ()
    phase: TaskPhase = TaskPhase.IMPLEMENT
    #: A stronger sidekick can be given less detail and allowed to push back; a
    #: weaker one needs prescriptive briefs and should not be opinionated.
    prescriptive: bool = True
    allow_pushback: bool = False

    def token_estimate(self, chars_per_token: float = 4.0) -> int:
        body = "\n".join((self.task, *self.constraints, *self.success_criteria))
        return max(1, int(len(body) / chars_per_token))


@dataclass(frozen=True)
class DelegationResult:
    brief: Brief
    output: str
    tokens_used: int = 0
    attempts: int = 1
    pushback: str | None = None
    needs_review: bool = True


# --------------------------------------------------------------------------- #
# Pair economics — cost belongs to the PAIR, not the model
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class PairProfile:
    """Measured behaviour of a (lead, sidekick) pair.

    Cognition's finding: "the models' cost and intelligence are coupled. What
    matters is how efficiently the pair completes work together." A 275% pricier
    sidekick was 2% *cheaper* end to end, because stronger sidekicks need fewer
    attempts and create fewer review rounds.

    A price registry keyed by model cannot express this. These are the per-pair
    factors that sit on top of it.
    """

    lead_model: str
    sidekick_model: str
    #: Multiplier on the sidekick's nominal cost. >1 means the pair generates
    #: extra review/rework that erodes the sidekick's savings.
    rework_factor: float = 1.0
    #: How many tokens the lead spends per delegated task on briefing + review.
    lead_overhead_tokens: float = 800.0
    #: Mean attempts the sidekick needs to satisfy a brief.
    sidekick_attempts: float = 1.0
    #: Fraction of work the lead keeps for itself.
    lead_share: float = 0.5
    measured_cost_per_task: float | None = None
    measured_score: float | None = None

    @property
    def price_per_task(self) -> float | None:
        """The headline metric. Price per token is only part of the equation."""
        return self.measured_cost_per_task

    def effective_sidekick_multiplier(self) -> float:
        """Cost of the sidekick once rework and attempts are priced in."""
        return self.rework_factor * self.sidekick_attempts


@dataclass
class PairRegistry:
    """Per-pair profiles. Falls back to a neutral profile for unseen pairs."""

    profiles: dict[tuple[str, str], PairProfile] = field(default_factory=dict)

    def register(self, profile: PairProfile) -> None:
        self.profiles[(profile.lead_model, profile.sidekick_model)] = profile

    def get(self, lead: str, sidekick: str) -> PairProfile:
        key = (lead, sidekick)
        if key not in self.profiles:
            self.profiles[key] = PairProfile(lead_model=lead, sidekick_model=sidekick)
        return self.profiles[key]

    def best_by_price_per_task(self) -> PairProfile | None:
        measured = [p for p in self.profiles.values() if p.measured_cost_per_task]
        if not measured:
            return None
        return min(measured, key=lambda p: p.measured_cost_per_task)

    def record_outcome(self, lead: str, sidekick: str, *, cost: float,
                       score: float | None = None, alpha: float = 0.3,
                       attempts: float | None = None,
                       rework_factor: float | None = None) -> None:
        """Fold an observed session cost back into the pair profile.

        EWMA, not replacement: one unusually cheap session must not erase the
        history. `PairProfile` is frozen, so we swap the whole entry — which also
        means a reader never sees a half-updated profile.

        `attempts` and `rework_factor` are what `effective_sidekick_multiplier()`
        actually reads, and therefore what the scorer's pair term sees. Without
        folding them here, measured behaviour never reaches a routing decision —
        the profile would be pure reporting. Pass them (or let `adelegate` do it)
        so a sidekick that needs more correction rounds costs more to pick.
        """
        p = self.get(lead, sidekick)

        def ewma(prev: float | None, obs: float) -> float:
            return obs if prev is None else (1 - alpha) * prev + alpha * obs

        blended = ewma(p.measured_cost_per_task, cost)
        new_score = (p.measured_score if score is None
                     else ewma(p.measured_score, score))
        new_attempts = (p.sidekick_attempts if attempts is None
                        else ewma(p.sidekick_attempts, attempts))
        new_rework = (p.rework_factor if rework_factor is None
                      else ewma(p.rework_factor, rework_factor))
        self.profiles[(lead, sidekick)] = replace(
            p, measured_cost_per_task=blended, measured_score=new_score,
            sidekick_attempts=new_attempts, rework_factor=new_rework,
        )

    def __len__(self) -> int:
        return len(self.profiles)


# --------------------------------------------------------------------------- #
# The delegation decision
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class DelegationDecision:
    should_delegate: bool
    phase: TaskPhase
    reason: str
    brief: Brief | None = None


class DelegationPolicy:
    """Decides what the lead keeps and what it hands off."""

    def __init__(
        self,
        sidekick_is_strong: bool = False,
        delegate_debug: bool = True,
    ) -> None:
        # A stronger sidekick earns more latitude: terser briefs, pushback
        # allowed, and it may help with initial exploration. A weaker one gets
        # prescriptive briefs and no opinion.
        self.sidekick_is_strong = sidekick_is_strong
        self.delegate_debug = delegate_debug

    def delegable(self, phase: TaskPhase) -> bool:
        if phase in LEAD_ONLY_PHASES:
            return False
        if phase == TaskPhase.DEBUG:
            return self.delegate_debug
        return phase in DELEGABLE_PHASES

    def decide(self, req: RoutingRequest, task: str) -> DelegationDecision:
        phase = infer_phase(req.messages, req.tools)

        if not self.delegable(phase):
            why = (
                "exploration that shapes the plan stays with the lead — a weaker "
                "model may not judge what information matters, and delegating it "
                "starves the lead's context"
                if phase == TaskPhase.PLAN
                else f"{phase.value} is lead-only"
            )
            return DelegationDecision(False, phase, why)

        brief = Brief(
            task=task,
            phase=phase,
            # Weaker sidekick: spend lead tokens upfront to avoid review rounds.
            prescriptive=not self.sidekick_is_strong,
            # Stronger sidekick: pushback catches mistakes in the lead's plan.
            allow_pushback=self.sidekick_is_strong,
        )
        return DelegationDecision(
            True, phase,
            f"{phase.value} is delegable"
            + (" (strong sidekick: terse brief, pushback allowed)"
               if self.sidekick_is_strong else " (weak sidekick: prescriptive brief)"),
            brief=brief,
        )


__all__ = [
    "TaskPhase",
    "DELEGABLE_PHASES",
    "LEAD_ONLY_PHASES",
    "PHASE_TURN_SHARE",
    "infer_phase",
    "Brief",
    "DelegationResult",
    "PairProfile",
    "PairRegistry",
    "DelegationDecision",
    "DelegationPolicy",
]
