"""The decision layer: pure, no I/O, deterministic.

    score(model, backend) = w_q · quality(workload, features)
                          − w_c · cost(model, backend, cache_hit=affinity)
                          − w_l · latency(backend, queue_depth, cold_start_risk)
                          − w_s · switch_cost          ← the term nobody has
                          + w_a · affinity_bonus       ← sticky-session reward
    subject to: residency ∧ budget ∧ context_window ∧ tool_support

Scoring is per (model, backend) PAIR, not per model. That is what makes the
workload axis and the deployment axis composable instead of sequential.

Nothing in here touches the network, the clock (except where injected), or
global state, so a decision is reproducible from its inputs.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Mapping, Sequence

from . import economics
from .affinity import SessionPolicy, affinity_prior, pick_backend, rendezvous_pick
from .backends import BackendRegistry, BackendSpec
from .learning import QualityModel, extract_features
from .pricing import PriceCard, PriceRegistry
from .signals import GAP_PRIOR, TURNS_PRIOR, WorkloadResolver

if False:  # pragma: no cover
    from .delegation import PairRegistry  # typing only; avoids an import cycle
from .types import (
    Candidate,
    Decision,
    RoutingRequest,
    SegmentKind,
    WorkloadClass,
    WorkloadSource,
)

DEFAULT_EST_PREFIX_TOKENS = 4000


def _tool_chars(tool: Mapping[str, Any]) -> int:
    """Tool-schema length, serialized exactly as the fingerprint does.

    Must match the JS engine's `JSON.stringify(tool)` byte-for-byte so both
    runtimes estimate the same token counts and make the same decisions —
    the cross-runtime parity suite asserts this.
    """
    return len(json.dumps(tool, separators=(",", ":")))


@dataclass(frozen=True)
class Weights:
    """Scoring weights. Exposed so you can tune cost-vs-quality without a fork."""

    quality: float = 1.0
    cost: float = 1.0
    latency: float = 0.0002
    switch_cost: float = 1.0
    affinity: float = 0.10

    #: Hard floor: never route below this estimated quality for the workload.
    min_quality: float = 0.0
    #: Soft preference: accept up to this much extra cost for a quality gain.
    quality_cost_tradeoff: float = 1.0


@dataclass(frozen=True)
class RouterConfig:
    weights: Weights = field(default_factory=Weights)
    #: Gate cache-affinity logic below this prefix length — routing overhead can
    #: exceed the savings on short prefixes.
    min_prefix_tokens_for_affinity: int = 500
    est_tokens_per_char: float = 0.25
    default_prefix_tokens: int = DEFAULT_EST_PREFIX_TOKENS
    enable_step_escalation: bool = True
    #: Keepalive pinger is opt-in: it needs a background timer per live session.
    enable_keepalive: bool = False
    #: Reserved output tokens when checking a candidate's context window before
    #: calling (the pre-call check). A prefix that fills the window leaves no
    #: room for the response; better to filter the candidate than to have the
    #: provider reject the call mid-session.
    context_headroom_tokens: int = 4096


@dataclass(frozen=True)
class ScoredCandidate:
    candidate: Candidate
    card: PriceCard
    backend: BackendSpec


class PolicyEngine:
    """Turns a request into a Decision."""

    def __init__(
        self,
        prices: PriceRegistry,
        backends: BackendRegistry,
        resolver: WorkloadResolver,
        config: RouterConfig | None = None,
        pairs: Any = None,
        lead_model: str | None = None,
        quality_model: QualityModel | None = None,
    ) -> None:
        self.prices = prices
        self.backends = backends
        self.resolver = resolver
        self.config = config if config is not None else RouterConfig()
        #: Cost belongs to the (lead, sidekick) PAIR, not the model. When a lead
        #: is known, a measured pair profile scales the sidekick's cost.
        self.pairs = pairs
        self.lead_model = lead_model
        #: Offline-trained quality heads (llmrouter.learning). When present, the
        #: quality term is prior + shrunk learned delta instead of a guess.
        self.quality_model = quality_model

    # ------------------------------------------------------------------ #
    # quality: prior, or prior + learned delta
    # ------------------------------------------------------------------ #
    def _quality(self, card: PriceCard, workload: WorkloadClass,
                 features: tuple[float, ...] | None) -> float:
        if self.quality_model is None or features is None:
            return card.quality(workload.value)
        return self.quality_model.quality(card, workload, features)

    def _pair_multiplier(self, card: PriceCard) -> float:
        """Measured pair rework, if we have a profile for (lead, this model)."""
        if self.pairs is None or self.lead_model is None:
            return 1.0
        if card.model_id == self.lead_model:
            return 1.0
        profile = self.pairs.profiles.get((self.lead_model, card.model_id))
        if profile is None:
            return 1.0
        return profile.effective_sidekick_multiplier()

    # ------------------------------------------------------------------ #
    # feature estimation
    # ------------------------------------------------------------------ #
    def estimate_prefix_tokens(self, req: RoutingRequest) -> int:
        """Estimate the cacheable prefix: system + tools + prior history."""
        chars = 0
        for m in req.messages:
            if m.role == "system":
                chars += len(m.text)
            elif m.role != "user" or not req.is_first_turn:
                chars += len(m.text)
        for t in req.tools:
            chars += _tool_chars(t)
        for r in req.retrieved:
            chars += len(r)
        est = int(chars * self.config.est_tokens_per_char)
        return est if est > 0 else 0

    def estimate_total_tokens(self, req: RoutingRequest) -> int:
        """Estimate the FULL prompt, including the current user turn.

        The cacheable prefix excludes the turn being sent (it is not cacheable
        yet), but the provider's context window does not — the pre-call window
        check must count everything.
        """
        chars = sum(len(m.text) for m in req.messages)
        chars += sum(_tool_chars(t) for t in req.tools)
        chars += sum(len(r) for r in req.retrieved)
        est = int(chars * self.config.est_tokens_per_char)
        # Never smaller than the prefix estimate: they measure overlapping
        # material and a floor of zero would defeat the window check.
        return max(est, self.estimate_prefix_tokens(req))

    def estimate_turn_cost(self, card: PriceCard, prefix_tokens: int,
                           hit_rate: float, output_tokens: int = 500) -> float:
        """Expected $ for one turn, modelling the cache honestly.

        Scaled by `card.task_cost_multiplier`: sticker price per token is only
        part of the real cost. A model that needs fewer tokens per task and
        generates less rework is cheaper per TASK even at a higher $/token — which
        is exactly what Cognition measured for Fusion sidekicks.
        """
        cached = int(prefix_tokens * hit_rate)
        uncached = prefix_tokens - cached
        sticker = (
            card.cache_read_cost(cached)
            + card.input_cost(uncached)
            + card.output_cost(output_tokens)
        )
        return sticker * card.task_cost_multiplier

    # ------------------------------------------------------------------ #
    # the decision
    # ------------------------------------------------------------------ #
    def decide(
        self,
        req: RoutingRequest,
        session: SessionPolicy | None = None,
        now: float | None = None,
    ) -> Decision:
        now = now if now is not None else time.monotonic()
        reasons: list[str] = []
        w = self.config.weights

        # ---- 1. workload class (Path A then Path B) -------------------- #
        signal = self.resolver.resolve(req)
        workload = signal.workload
        reasons.append(f"workload={workload.value} via {signal.source.value}: {signal.reason}")

        allowed = self._allowed_models(req, signal.source)
        if not allowed:
            raise RuntimeError(
                "no models available — check the price registry and agent profile"
            )

        # Learned quality: one feature vector per decision, reused per card.
        features = (extract_features(req, workload)
                    if self.quality_model is not None else None)
        if self.quality_model is not None:
            covered, total = self.quality_model.coverage(sorted(allowed))
            if covered:
                reasons.append(
                    f"learned-quality: {covered}/{total} models have trained "
                    f"heads (shrinkage k={self.quality_model.shrinkage_k:g})"
                )

        # ---- 2. session shape ------------------------------------------ #
        prefix_tokens = self.estimate_prefix_tokens(req)
        total_tokens = self.estimate_total_tokens(req)
        gap_s = (
            req.expected_gap_s
            if req.expected_gap_s is not None
            else GAP_PRIOR.get(workload, 60.0)
        )
        gen_s = req.expected_generation_s or 0.0
        turns = req.expected_turns or TURNS_PRIOR.get(workload, 5)
        turns_remaining = max(1, turns - (session.turns_seen if session else 0))

        # ---- 3. affinity ---------------------------------------------- #
        # Gate affinity on prefix length: below ~500 tokens the routing overhead
        # can exceed the savings, and below the provider's cache minimum no cache
        # entry is created at all.
        affinity_enabled = prefix_tokens >= self.config.min_prefix_tokens_for_affinity
        prior = affinity_prior(workload, gap_s) if affinity_enabled else 0.0

        sticky_key = req.sticky_key()
        pinned_model = session.pinned_model if session else None
        pinned_backend = session.pinned_backend if session else None

        # ---- 3b. Rule F: cache-invalidating fields ---------------------- #
        # The provider's cache key covers more than the prefix text: tool
        # schemas, image usage, thinking/effort settings. If those drift within
        # a live session, the cache is gone and EVERY candidate is cold — the
        # warm pair must not be rewarded for a cache that no longer exists.
        inv_fp = req.invalidators_fingerprint()
        rule_f_invalidated = False
        if session is not None:
            if session.invalidators_hash is None:
                session.invalidators_hash = inv_fp
            elif session.invalidators_hash != inv_fp:
                rule_f_invalidated = True
                session.invalidators_hash = inv_fp
                reasons.append(
                    "Rule F: cache-invalidating request fields changed (tool "
                    "schema / image usage) — all candidates scored cold"
                )

        # ---- 4. score every (model, backend) pair ---------------------- #
        scored: list[ScoredCandidate] = []
        over_budget: list[ScoredCandidate] = []
        for card in self.prices.all():
            if card.model_id not in allowed:
                continue
            if not self._passes_hard_constraints(card, req, workload,
                                                 total_tokens, features):
                continue

            for backend in self.backends.available_for(card.model_id, now):
                hit_rate = prior
                if session is not None and card.model_id == pinned_model:
                    # Learned, not assumed: the provider tells us the real rate.
                    hit_rate = session.hit_rate(prior)
                if rule_f_invalidated:
                    hit_rate = 0.0

                turn_cost = self.estimate_turn_cost(card, prefix_tokens, hit_rate)
                cost = turn_cost * backend.cost_factor * self._pair_multiplier(card)

                latency = self._latency(card, backend)

                # Affinity bonus only for the pair we are already warm on.
                is_warm = (
                    session is not None
                    and card.model_id == pinned_model
                    and backend.backend_id == pinned_backend
                )
                affinity_bonus = 0.0 if rule_f_invalidated else (
                    hit_rate if is_warm else 0.0
                )

                # Switch cost: what it costs to cold-write the prefix elsewhere.
                # Under Rule F invalidation everyone pays a write anyway, so the
                # relative term would double-count.
                switch_cost = 0.0
                if (session is not None and not is_warm and affinity_enabled
                        and not rule_f_invalidated):
                    switch_cost = self._switch_cost(
                        session, card, prefix_tokens, workload
                    )

                quality = self._quality(card, workload, features)

                cap = req.constraints.max_cost_usd
                over_cap = cap is not None and cost > cap

                # Closed-loop budget pacing: a request whose projected session
                # spend exceeds the remaining budget is penalized in proportion
                # to the overshoot. Open-ended pacing without this overshoots.
                budget_penalty = 0.0
                projected = 0.0
                remaining = req.constraints.budget_remaining_usd
                if remaining is not None and not over_cap:
                    projected = cost * max(1, turns_remaining)
                    if projected > remaining:
                        budget_penalty = projected - remaining

                cand_reasons = [f"hit_rate={hit_rate:.2f}", f"warm={is_warm}"]
                if budget_penalty > 0:
                    cand_reasons.append(
                        f"budget pacing: projected ${projected:.4f} exceeds "
                        f"${remaining:.4f} remaining"
                    )

                score = (
                    w.quality * quality
                    - w.cost * cost * w.quality_cost_tradeoff
                    - w.cost * budget_penalty
                    - w.latency * latency
                    - w.switch_cost * switch_cost
                    + w.affinity * affinity_bonus
                )

                cand = Candidate(
                    model_id=card.model_id,
                    backend_id=backend.backend_id,
                    score=score,
                    quality=quality,
                    cost_usd=cost,
                    latency_ms=latency,
                    affinity=affinity_bonus,
                    switch_cost_usd=switch_cost,
                    reasons=tuple(cand_reasons),
                )
                entry = ScoredCandidate(cand, card, backend)
                if over_cap:
                    over_budget.append(entry)
                else:
                    scored.append(entry)

        budget_fallback = False
        if not scored and over_budget:
            # A cap no candidate can meet is a preference, not an outage: pick
            # the cheapest violator rather than erroring, and say so. (If the
            # list is empty for hard-constraint reasons the error below stands —
            # there is no honest way to serve the request.)
            scored = over_budget
            budget_fallback = True
            cheapest = min(scored, key=lambda s: s.candidate.cost_usd)
            reasons.append(
                f"budget cap ${req.constraints.max_cost_usd:.4f} excludes every "
                f"candidate; chose the cheapest ({cheapest.candidate.model_id})"
            )

        if not scored:
            raise RuntimeError(
                f"every candidate was filtered out for workload={workload.value}; "
                "check hard constraints (residency, context window, tool support) "
                "and backend health"
            )

        if budget_fallback:
            # Honor the cap's intent: order by cost, so best = cheapest.
            scored.sort(key=lambda s: s.candidate.cost_usd)
        else:
            scored.sort(key=lambda s: s.candidate.score, reverse=True)
        best = scored[0]

        # ---- 5. stickiness / migration -------------------------------- #
        switched = False
        stranded = False
        chosen_model = best.candidate.model_id
        chosen_backend = best.candidate.backend_id

        if session is not None:
            # Distinguish a COST-DRIVEN migration from a FORCED failover. The
            # no-oscillation lock exists to stop cost-driven ping-pong; it must
            # never block a failover away from a backend that is actually down,
            # or an outage becomes a hard failure.
            pinned_available = bool(
                self.backends.available_for(session.pinned_model, now)
            )
            forced = False
            wants_switch = chosen_model != session.pinned_model
            if wants_switch and not pinned_available:
                forced = True
                switched = True
                session.switched_at = now
                reasons.append(
                    f"forced failover: pinned {session.pinned_model} has no healthy "
                    "backend (not a cost-driven switch)"
                )
            if wants_switch and not forced:
                cur = self.prices.get(session.pinned_model)
                tgt = best.card
                if cur is not None and tgt is not None:
                    verdict = economics.evaluate_switch(
                        cur, tgt, prefix_tokens, turns_remaining,
                        already_switched=session.already_switched,
                    )
                    if verdict.should_switch:
                        switched = True
                        session.switched_at = now
                        reasons.append(f"migrating: {verdict.reason}")
                    else:
                        stranded = not verdict.reason.startswith("no-oscillation")
                        if stranded:
                            session.stranded_count += 1
                        # Stay pinned — the migration cannot pay back.
                        chosen_model = session.pinned_model
                        chosen_backend = pick_backend(
                            sticky_key,
                            [b.backend_id for b in
                             self.backends.available_for(chosen_model, now)],
                            current=session.pinned_backend,
                        ) or session.pinned_backend
                        reasons.append(f"staying pinned: {verdict.reason}")
            else:
                # Same model: keep the backend sticky, overflow accepts the miss.
                chosen_backend = pick_backend(
                    sticky_key,
                    [b.backend_id for b in
                     self.backends.available_for(chosen_model, now)],
                    current=session.pinned_backend,
                ) or chosen_backend

        # ---- 6. TTL policy per breakpoint ------------------------------ #
        segment_tokens = self.segment_tokens(req, prefix_tokens)
        sem = best.card.cache
        ttl_by_segment = economics.choose_ttl_by_segment(
            sem, segment_tokens, gap_s, gen_s
        )
        if sem.supports_mixed_ttl and any(v == "1h" for v in ttl_by_segment.values()):
            reasons.append(
                f"mixed TTL: {', '.join(k.value + '=' + v for k, v in ttl_by_segment.items())}"
            )

        # ---- 7. build the decision ------------------------------------ #
        ordered = tuple(s.candidate for s in scored)
        final = next(
            (c for c in ordered
             if c.model_id == chosen_model and c.backend_id == chosen_backend),
            None,
        )
        if final is None:
            # The pinned pair produced no candidate. Fall back on the SAME model
            # with any healthy backend first; only then accept a different model,
            # and record that we did so rather than failing silently.
            final = next(
                (c for c in ordered if c.model_id == chosen_model), None
            )
            if final is None:
                final = ordered[0]
                chosen_model = final.model_id
                if session is not None and session.pinned_model != chosen_model:
                    switched = True
                    session.switched_at = now
                    reasons.append(
                        f"pinned {session.pinned_model} produced no viable candidate; "
                        f"fell back to {chosen_model}"
                    )
            chosen_backend = final.backend_id

        return Decision(
            model_id=final.model_id,
            backend_id=final.backend_id,
            workload=workload,
            workload_source=signal.source,
            workload_confidence=signal.confidence,
            sticky_key=sticky_key,
            session_key=sticky_key,
            is_first_turn=req.is_first_turn,
            reused_session=session is not None,
            switched=switched,
            stranded=stranded,
            ttl_by_segment=ttl_by_segment,
            candidates=ordered,
            reasons=tuple(reasons),
        )

    # ------------------------------------------------------------------ #
    # helpers
    # ------------------------------------------------------------------ #
    def _allowed_models(self, req: RoutingRequest,
                        source: WorkloadSource) -> set[str]:
        allowed: set[str] | None = None
        if req.agent_profile:
            profile = self.resolver.profiles.get(req.agent_profile)
            if profile is not None and profile.allowed_models:
                allowed = set(profile.allowed_models) & set(self.prices.ids())
        if req.constraints.allowed_models:
            pool = set(req.constraints.allowed_models) & set(self.prices.ids())
            allowed = pool if allowed is None else (allowed & pool)
        return allowed if allowed is not None else set(self.prices.ids())

    def _passes_hard_constraints(self, card: PriceCard, req: RoutingRequest,
                                 workload: WorkloadClass,
                                 prompt_tokens: int,
                                 features: tuple[float, ...] | None = None,
                                 ) -> bool:
        quality = self._quality(card, workload, features)
        if quality < 0:
            # A negative prior means "explicitly unsupported for this workload".
            return False
        if self.config.weights.min_quality > 0 and quality < self.config.weights.min_quality:
            # The quality floor is the router's operating point: it is what
            # `calibrate` tunes, and what the eval CI gate guards.
            return False
        if req.constraints.require_tools and not card.supports_tools:
            return False
        if card.supports_vision is False and req.has_images():
            return False
        if req.constraints.residency_tags and not (
            req.constraints.residency_tags <= card.residency_tags
        ):
            return False
        # Pre-call context validation: a prompt that fills the window leaves no
        # room for the response, and the provider would reject the call.
        # Measured against the FULL prompt (see estimate_total_tokens).
        if (card.context_window
                and prompt_tokens + self.config.context_headroom_tokens
                > card.context_window):
            return False
        return True

    def _latency(self, card: PriceCard, backend: BackendSpec) -> float:
        h = self.backends.health(backend.backend_id)
        base = h.ewma_latency_ms or card.p50_latency_ms
        queue = h.in_flight * 5.0
        cold = backend.cold_start_risk * 2000.0
        return base + queue + cold

    def _switch_cost(self, session: SessionPolicy, tgt: PriceCard,
                     prefix_tokens: int, workload: WorkloadClass) -> float:
        """Cost of abandoning the warm cache: a cold write on the target.

        Note this is provider-specific in a way that surprises people: OpenAI
        charges no write premium, so switching onto an OpenAI-backed model is
        nearly free, while Anthropic charges 1.25×.
        """
        if workload == WorkloadClass.BATCH:
            return 0.0
        return tgt.cache_write_cost(prefix_tokens, "5m")

    def segment_tokens(self, req: RoutingRequest,
                        prefix_tokens: int) -> dict[SegmentKind, int]:
        """Split the prefix into TTL-policy segments."""
        def toks(s: str) -> int:
            return int(len(s) * self.config.est_tokens_per_char)

        sys_t = toks(req.system_text())
        tools_t = sum(int(_tool_chars(t) * self.config.est_tokens_per_char)
                      for t in req.tools)
        retr_t = sum(toks(r) for r in req.retrieved)
        hist_t = max(0, prefix_tokens - sys_t - tools_t - retr_t)

        out: dict[SegmentKind, int] = {}
        if tools_t:
            out[SegmentKind.TOOLS] = tools_t
        if sys_t:
            out[SegmentKind.SYSTEM] = sys_t
        if retr_t:
            out[SegmentKind.RETRIEVED] = retr_t
        if hist_t:
            out[SegmentKind.HISTORY] = hist_t
        return out


__all__ = [
    "Weights",
    "RouterConfig",
    "ScoredCandidate",
    "PolicyEngine",
    "DEFAULT_EST_PREFIX_TOKENS",
]
