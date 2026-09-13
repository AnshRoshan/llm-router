"""The `Router` facade — the public entry point.

Three layers, kept separate on purpose:

    DECISION   pure, no I/O          PolicyEngine.decide()
    EXECUTION  async, fallible       Router.acomplete()
    LEARNING   feedback loop         usage counters -> SessionPolicy.record_usage

You can use the decision layer alone (zero dependencies, no network) or the full
async path with the optional httpx extra.
"""
from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, replace
from typing import Any, Awaitable, Callable, Mapping, Sequence

from .adapters import Completion, ModelAdapter, Usage, adapter_for
from .affinity import SessionPolicy, SessionStore
from .delegation import (
    Brief,
    DelegationDecision,
    DelegationPolicy,
    DelegationResult,
    PairRegistry,
    TaskPhase,
    infer_phase,
)
from .backends import BackendRegistry, BackendSpec
from .economics import emit_breakpoints
from .policy import PolicyEngine, RouterConfig
from .pricing import PriceRegistry
from .signals import AgentProfile, FingerprintRegistry, WorkloadClassifier, WorkloadResolver
from .types import (
    Decision,
    DeploymentClass,
    Message,
    RoutingRequest,
    WorkloadClass,
    WorkloadSource,
)

DEFAULT_MAX_ATTEMPTS = 3


@dataclass(frozen=True)
class RetryPolicy:
    max_attempts: int = DEFAULT_MAX_ATTEMPTS
    base_backoff_s: float = 0.25
    max_backoff_s: float = 4.0
    retry_on_status: frozenset[int] = frozenset({408, 409, 429, 500, 502, 503, 504})


class RouterError(RuntimeError):
    def __init__(self, message: str, *, attempts: list[str] | None = None) -> None:
        super().__init__(message)
        self.attempts = attempts or []


class ConfigurationError(RouterError):
    """A misconfiguration, not an upstream failure.

    Must NOT be retried or failed over. Retrying "you forgot to configure a
    transport" three times and then reporting "all attempts failed" hides the
    actual cause — the caller ends up debugging their provider instead of their
    setup. Raised before the attempt loop ever starts.
    """


class Router:
    """Cache-aware LLM router.

    >>> router = Router.with_defaults(api_key="...")          # doctest: +SKIP
    >>> out = await router.acomplete([Message("user", "hi")])  # doctest: +SKIP
    """

    def __init__(
        self,
        prices: PriceRegistry,
        backends: BackendRegistry,
        resolver: WorkloadResolver | None = None,
        config: RouterConfig | None = None,
        sessions: SessionStore | None = None,
        retry: RetryPolicy | None = None,
        transport: Any | None = None,
        adapters: Mapping[str, ModelAdapter] | None = None,
        delegation: DelegationPolicy | None = None,
        pairs: PairRegistry | None = None,
        sidekick_model: str | None = None,
        lead_model: str | None = None,
    ) -> None:
        self.prices = prices
        self.backends = backends
        self.resolver = resolver if resolver is not None else WorkloadResolver()
        self.config = config if config is not None else RouterConfig()
        self.sessions = sessions if sessions is not None else SessionStore()
        self.retry = retry if retry is not None else RetryPolicy()
        self.transport = transport
        self._adapters: dict[str, ModelAdapter] = dict(adapters or {})
        # Delegation: a separate routing surface with its own session keys, so a
        # delegated task never touches the lead's warm prefix cache.
        # NOTE: these must be set BEFORE the engine is built — the engine reads
        # them to scale cost by measured pair economics.
        self.delegation = delegation if delegation is not None else DelegationPolicy()
        self.pairs = pairs if pairs is not None else PairRegistry()
        self.lead_model = lead_model
        # A delegated task must run on the configured sidekick, not on whatever
        # scores highest — otherwise `adelegate` silently routes to the frontier
        # model and the whole point (a cheaper partner with its own warm cache)
        # is lost. We express the pin through a restricted agent profile, the
        # same mechanism a caller-selected agent uses, so it stays inside the
        # normal decision path (hard constraints, failover, health) instead of
        # bypassing it. The profile is (re)registered by the `sidekick_model`
        # setter below, so the pin tracks the attribute however it is assigned —
        # in the constructor OR later (which is how examples/demo.py uses it).
        # `_sidekick_profile` must exist before the first assignment triggers it.
        self._sidekick_profile = "__sidekick__"
        self._sidekick_model: str | None = None
        self.sidekick_model = sidekick_model
        self.engine = PolicyEngine(
            prices, self.backends, self.resolver, self.config,
            pairs=self.pairs, lead_model=self.lead_model,
        )
        self._cost_usd = 0.0
        self._decisions = 0
        self._misroutes = 0
        self._delegations = 0
        self._last_decision: Decision | None = None
        #: Attempts the last acomplete() needed (1 = first try). Delegated
        #: tasks fold this into pair economics: more attempts => a sidekick
        #: that needs more correction rounds => costlier as a partner.
        self._last_attempts: int = 1

    # ------------------------------------------------------------------ #
    # sidekick pin — derived state, kept in sync via a property so the pin
    # holds whether `sidekick_model` is set in the constructor or later.
    # ------------------------------------------------------------------ #
    @property
    def sidekick_model(self) -> str | None:
        return self._sidekick_model

    @sidekick_model.setter
    def sidekick_model(self, value: str | None) -> None:
        self._sidekick_model = value
        if value is None:
            self.resolver.profiles.pop(self._sidekick_profile, None)
            return
        if value not in self.prices.ids():
            raise ConfigurationError(
                f"sidekick_model {value!r} has no price card — register one "
                "first (PriceRegistry.register) or pick a bundled model"
            )
        self.resolver.add_profile(AgentProfile(
            name=self._sidekick_profile,
            workload=WorkloadClass.AGENT,
            allowed_models=(value,),
            sticky=True,
        ))

    # ------------------------------------------------------------------ #
    # construction helpers
    # ------------------------------------------------------------------ #
    @classmethod
    def with_defaults(
        cls,
        *,
        anthropic_key: str | None = None,
        openai_key: str | None = None,
        profiles: Mapping[str, AgentProfile] | None = None,
        classifier: WorkloadClassifier | None = None,
        default_workload: WorkloadClass = WorkloadClass.CHAT,
        extra_backends: Sequence[BackendSpec] = (),
        config: RouterConfig | None = None,
        transport: Any | None = None,
        sidekick_model: str | None = None,
        lead_model: str | None = None,
        delegation: DelegationPolicy | None = None,
        pairs: PairRegistry | None = None,
    ) -> "Router":
        """Sensible starting point: the bundled price cards plus one backend per
        provider you supplied a key for.

        Register your own backends for self-hosted/serverless deployments — the
        bundled defaults are API-only, because deployment class is the axis you
        know and we do not.
        """
        prices = PriceRegistry()
        backends: list[BackendSpec] = []
        if anthropic_key:
            backends.append(BackendSpec(
                backend_id="anthropic",
                deployment=DeploymentClass.API,
                models=tuple(m for m in prices.ids() if m.startswith("claude")),
                base_url="https://api.anthropic.com",
                priority=0,
            ))
        if openai_key:
            backends.append(BackendSpec(
                backend_id="openai",
                deployment=DeploymentClass.API,
                models=tuple(m for m in prices.ids() if m.startswith("gpt")),
                base_url="https://api.openai.com",
                priority=1,
            ))
        backends.extend(extra_backends)
        if not backends:
            raise ValueError(
                "no backends configured — pass anthropic_key/openai_key or extra_backends"
            )

        router = cls(
            prices=prices,
            backends=BackendRegistry(backends),
            resolver=WorkloadResolver(
                profiles=profiles,
                classifier=classifier,
                default_workload=default_workload,
            ),
            config=config,
            transport=transport,
            sidekick_model=sidekick_model,
            lead_model=lead_model,
            delegation=delegation,
            pairs=pairs,
        )
        router._keys = {"anthropic": anthropic_key, "openai": openai_key}  # type: ignore[attr-defined]
        return router

    # ------------------------------------------------------------------ #
    # DECISION (pure, no I/O)
    # ------------------------------------------------------------------ #
    def decide(self, req: RoutingRequest) -> Decision:
        """Pick a (model, backend) pair without calling anything.

        Safe to call in a hot path: no network, no I/O. Use it to inspect routing
        behaviour, or to drive your own HTTP client.
        """
        session = self.sessions.get(req.sticky_key())
        decision = self.engine_decide(req, session)
        self._last_decision = decision
        return decision

    def engine_decide(self, req: RoutingRequest,
                      session: SessionPolicy | None) -> Decision:
        return self.engine.decide(req, session=session)

    def explain(self, last: bool = True) -> str:
        if last and self._last_decision is not None:
            return self._last_decision.explain()
        return "no decision recorded yet"

    # ------------------------------------------------------------------ #
    # EXECUTION (async)
    # ------------------------------------------------------------------ #
    async def acomplete(
        self,
        messages: Sequence[Message] | Sequence[Mapping[str, Any]],
        *,
        workload: WorkloadClass | str | None = None,
        session_id: str | None = None,
        agent_profile: str | None = None,
        tools: Sequence[Mapping[str, Any]] = (),
        retrieved: Sequence[str] = (),
        tags: Sequence[str] = (),
        expected_turns: int | None = None,
        expected_gap_s: float | None = None,
        expected_generation_s: float | None = None,
        max_tokens: int = 1024,
        **kwargs: Any,
    ) -> Completion:
        req = self._to_request(
            messages,
            workload=workload, session_id=session_id, agent_profile=agent_profile,
            tools=tools, retrieved=retrieved, tags=tags,
            expected_turns=expected_turns, expected_gap_s=expected_gap_s,
            expected_generation_s=expected_generation_s,
        )
        decision = self.decide(req)
        card = self.prices.require(decision.model_id)
        sem = card.cache

        prefix_tokens = self.engine.estimate_prefix_tokens(req)
        segment_tokens = self.engine.segment_tokens(req, prefix_tokens)
        breakpoints = emit_breakpoints(sem, segment_tokens, decision.ttl_by_segment)

        attempts: list[str] = []
        last_error: Exception | None = None

        # Fail fast on misconfiguration. This must be checked BEFORE the attempt
        # loop: retrying it three times and reporting "all attempts failed"
        # sends the caller debugging their provider instead of their setup.
        if self.transport is None and not self._adapters:
            raise ConfigurationError(
                "no transport configured — install llmrouter[http] and pass an "
                "httpx.AsyncClient, or use Router.decide() for pure routing"
            )

        for attempt in range(self.retry.max_attempts):
            backend = self.backends.spec(decision.backend_id)
            if backend is None:
                attempts.append(f"{decision.backend_id}: no such backend")
                break

            health = self.backends.health(backend.backend_id)
            # Claim a canary slot if this backend is HALF_OPEN: exactly one
            # request may probe it while the circuit recovers.
            if health.is_half_open():
                health.half_open_probes += 1
            health.in_flight += 1
            started = time.monotonic()
            try:
                completion = await self._call(
                    backend, decision, req, breakpoints, max_tokens, kwargs
                )
            except ConfigurationError:
                health.in_flight -= 1
                raise
            except Exception as exc:  # noqa: BLE001 — we re-raise after the chain
                last_error = exc
                retry_after = _retry_after(exc)
                health.record_failure(str(exc), retry_after_s=retry_after)
                attempts.append(f"{backend.backend_id}: {exc}")
                # Fall through to the next healthy backend rather than giving up.
                nxt = self._next_backend(decision, req, exclude=backend.backend_id)
                if nxt is None:
                    # No other backend serves this model. Re-decide against the
                    # remaining healthy capacity instead of failing: a model whose
                    # only backend is down should cost a cache miss, not an error.
                    redecided = self._redecide_excluding(req, decision.model_id)
                    if redecided is None:
                        break
                    decision = redecided
                    attempts.append(
                        f"re-decided to {decision.model_id}@{decision.backend_id}"
                    )
                    await asyncio.sleep(
                        min(self.retry.base_backoff_s * (2 ** attempt),
                            self.retry.max_backoff_s)
                    )
                    continue
                decision = replace(
                    decision, backend_id=nxt,
                    reasons=decision.reasons + (f"failed over to {nxt}",),
                )
                await asyncio.sleep(
                    min(self.retry.base_backoff_s * (2 ** attempt),
                        self.retry.max_backoff_s)
                )
                continue
            finally:
                # Exactly one decrement per increment. The `except` branch used
                # to decrement as well, which drove in_flight negative and made
                # the saturation check meaningless under load.
                health.in_flight = max(0, health.in_flight - 1)

            latency_ms = (time.monotonic() - started) * 1000.0
            health.record_success(latency_ms)
            self._learn(decision, completion.usage, req)
            self._last_attempts = attempt + 1
            self._last_decision = replace(
                decision,
                cache_read_tokens=completion.usage.cache_read_tokens,
                cache_write_tokens=completion.usage.cache_write_tokens,
                latency_ms=latency_ms,
            )
            return completion

        raise RouterError(
            f"all {len(attempts)} attempt(s) failed for workload="
            f"{decision.workload.value}", attempts=attempts
        ) from last_error

    # ------------------------------------------------------------------ #
    def _to_request(self, messages, **kw) -> RoutingRequest:
        norm: list[Message] = []
        for m in messages:
            if isinstance(m, Message):
                norm.append(m)
            elif isinstance(m, Mapping):
                norm.append(Message(str(m.get("role", "user")), m.get("content", "")))
            else:
                raise TypeError(f"unsupported message: {m!r}")
        workload = kw.pop("workload", None)
        if isinstance(workload, str):
            workload = WorkloadClass(workload)
        tags = kw.pop("tags", ())
        return RoutingRequest(
            messages=tuple(norm),
            workload=workload,
            tags=frozenset(tags),
            tools=tuple(kw.pop("tools", ()) or ()),
            retrieved=tuple(kw.pop("retrieved", ()) or ()),
            **kw,
        )

    async def _call(self, backend: BackendSpec, decision: Decision,
                    req: RoutingRequest, breakpoints, max_tokens: int,
                    kwargs: Mapping[str, Any]) -> Completion:
        if self.transport is None:
            raise ConfigurationError(
                "no transport configured — install llmrouter[http] and pass an "
                "httpx.AsyncClient, or use Router.decide() for pure routing"
            )
        adapter = self._adapters.get(backend.backend_id) or adapter_for(backend.backend_id)
        card = self.prices.require(decision.model_id)
        key_param = card.cache.explicit_cache_key_param

        body = adapter.build_body(
            req.messages,
            tools=req.tools,
            breakpoints=breakpoints,
            model=decision.model_id,
            cache_key=decision.sticky_key if key_param else None,
            max_tokens=max_tokens,
            **dict(kwargs),
        )
        headers = self._headers(backend)
        url = f"{(backend.base_url or '').rstrip('/')}{getattr(adapter, 'path', '')}"
        started = time.monotonic()
        resp = await self.transport.post(
            url, headers=headers, json=dict(body), timeout=backend.timeout_s
        )
        status = getattr(resp, "status_code", 200)
        if status >= 400:
            raise RouterError(f"HTTP {status} from {backend.backend_id}")
        payload = resp.json() if callable(getattr(resp, "json", None)) else dict(resp)
        latency_ms = (time.monotonic() - started) * 1000.0
        return adapter.parse(payload, model=decision.model_id,
                             backend_id=backend.backend_id, latency_ms=latency_ms)

    def _headers(self, backend: BackendSpec) -> dict[str, str]:
        keys: Mapping[str, str | None] = getattr(self, "_keys", {})
        key = keys.get(backend.backend_id)
        if backend.backend_id == "anthropic":
            return {
                "x-api-key": key or "",
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            }
        return {"Authorization": f"Bearer {key or ''}", "content-type": "application/json"}

    def _redecide_excluding(self, req: RoutingRequest,
                            dead_model: str) -> Decision | None:
        """Re-run the decision with the failed model ruled out.

        This is what makes failover cross *models*, not just cross backends. It
        costs a cache miss (the new model has its own, model-scoped cache) but
        that is far cheaper than an error, and the session re-pins afterwards.
        """
        # Temporarily hide every backend that serves only the dead model, so the
        # engine scores the remaining capacity honestly.
        hidden: dict[str, BackendSpec] = {}
        for bid, spec in list(self.backends._specs.items()):
            if spec.models and dead_model in spec.models:
                hidden[bid] = self.backends._specs.pop(bid)
        try:
            decision = self.engine.decide(req, session=None)
        except RuntimeError:
            return None
        finally:
            self.backends._specs.update(hidden)
        if decision.model_id == dead_model:
            return None
        return replace(
            decision,
            switched=True,
            reasons=decision.reasons + (
                f"forced failover: {dead_model} has no healthy backend",
            ),
        )

    def _next_backend(self, decision: Decision, req: RoutingRequest,
                      exclude: str) -> str | None:
        for b in self.backends.available_for(decision.model_id):
            if b.backend_id != exclude:
                return b.backend_id
        return None

    # ------------------------------------------------------------------ #
    # DELEGATION (lead/sidekick)
    # ------------------------------------------------------------------ #
    def should_delegate(self, req: RoutingRequest, task: str) -> DelegationDecision:
        """Ask whether `task` belongs to the sidekick, given the current phase.

        Pure — no I/O. Phase is inferred from the tool-call history, which is
        observable; difficulty is not, so we never try to score it.
        """
        return self.delegation.decide(req, task)

    async def adelegate(
        self,
        brief: Brief,
        *,
        lead_session_id: str,
        tools: Sequence[Mapping[str, Any]] = (),
        max_tokens: int = 2048,
        **kwargs: Any,
    ) -> DelegationResult:
        """Run a delegated task on the sidekick.

        The mechanism that makes this cheaper than switching models: the sidekick
        gets **its own session key**, so it builds its own persistent context and
        its own warm prompt cache. The lead's prefix is never touched, so the
        lead's cache never goes cold.

        Only the brief crosses the boundary — never the lead's conversation.
        """
        if self.sidekick_model is None:
            raise ConfigurationError(
                "no sidekick_model configured — pass one to Router(...) to use "
                "delegation, or call should_delegate() to route manually"
            )
        if not self.delegation.delegable(brief.phase):
            raise ValueError(
                f"{brief.phase.value} is not delegable; this work belongs to the "
                "lead (exploration that shapes the plan must stay with the lead)"
            )

        # Separate session key => separate prefix => separate cache. This is the
        # whole point of the architecture.
        sidekick_session = f"{lead_session_id}::sidekick::{brief.phase.value}"
        messages = [Message("system", _brief_system_prompt(brief)),
                    Message("user", brief.task)]

        completion = await self.acomplete(
            messages,
            session_id=sidekick_session,
            agent_profile=self._sidekick_profile,
            tools=tools,
            max_tokens=max_tokens,
            **kwargs,
        )
        self._delegations += 1

        # Close the learning loop: fold this run's measured cost and attempt
        # count back into the pair profile, so the NEXT decision is informed by
        # how this sidekick actually performed. Cost is derived from the
        # provider's usage counters exactly as `_learn` does — no estimate.
        card = self.prices.get(completion.model)
        if self.lead_model is not None and card is not None:
            measured = (
                card.input_cost(completion.usage.input_tokens)
                + card.cache_read_cost(completion.usage.cache_read_tokens)
                + card.cache_write_cost(completion.usage.cache_write_tokens)
                + card.output_cost(completion.usage.output_tokens)
            )
            self.pairs.record_outcome(
                self.lead_model, completion.model,
                cost=measured, attempts=float(self._last_attempts),
            )

        return DelegationResult(
            brief=brief,
            output=completion.text,
            tokens_used=(completion.usage.input_tokens
                         + completion.usage.output_tokens),
            attempts=self._last_attempts,
            needs_review=True,   # the lead always reviews
        )

    # ------------------------------------------------------------------ #
    # LEARNING
    # ------------------------------------------------------------------ #
    def _learn(self, decision: Decision, usage: Usage, req: RoutingRequest) -> None:
        """Fold the provider's usage counters back into session state.

        The cache lives on the provider's side. Without this the router's
        affinity bonus is a guess and it will systematically overvalue the pair
        it already pinned.
        """
        session = self.sessions.get(decision.session_key)
        if session is None:
            session = SessionPolicy(
                session_key=decision.session_key,
                workload=decision.workload,
                source=decision.workload_source,
                confidence=decision.workload_confidence,
                pinned_model=decision.model_id,
                pinned_backend=decision.backend_id,
                pinned_at=time.monotonic(),
                # Baseline for Rule F drift detection: the invalidating fields
                # of the request that created the session.
                invalidators_hash=req.invalidators_fingerprint(),
            )
        else:
            # Re-pin after a migration so subsequent turns are sticky again.
            if decision.switched:
                session.pinned_model = decision.model_id
                session.pinned_backend = decision.backend_id
        session.record_usage(usage.cache_read_tokens, usage.cache_write_tokens)
        self.sessions.put(session)

        card = self.prices.get(decision.model_id)
        if card is not None:
            self._cost_usd += (
                card.input_cost(usage.input_tokens)
                + card.cache_read_cost(usage.cache_read_tokens)
                + card.cache_write_cost(usage.cache_write_tokens)
                + card.output_cost(usage.output_tokens)
            )
        self._decisions += 1

    def report_feedback(self, session_id: str, *, misrouted: bool) -> None:
        """Record a human/eval judgement. Feeds the misroute→training-example
        pipeline; `misrouted` sessions are the ones worth learning from."""
        if misrouted:
            self._misroutes += 1

    # ------------------------------------------------------------------ #
    def stats(self) -> dict[str, Any]:
        return {
            "decisions": self._decisions,
            "delegations": self._delegations,
            "pairs": {
                f"{p.lead_model}+{p.sidekick_model}": {
                    "price_per_task": p.measured_cost_per_task,
                    "score": p.measured_score,
                    "effective_sidekick_mult": p.effective_sidekick_multiplier(),
                }
                for p in self.pairs.profiles.values()
            },
            "estimated_cost_usd": round(self._cost_usd, 6),
            "misroutes": self._misroutes,
            "sessions": self.sessions.stats(),
            "backends": self.backends.stats(),
            "fingerprints": len(self.resolver.fingerprints),
            "volatile_clients": list(self.resolver.fingerprints.volatile_clients()),
        }

    @property
    def last_decision(self) -> Decision | None:
        return self._last_decision


def _brief_system_prompt(brief: Brief) -> str:
    """Build the sidekick's system prompt from a brief.

    Stable across delegations of the same phase so the sidekick's own prefix
    stays cacheable. Pushback is granted only when the policy says the sidekick
    is strong enough for it to help rather than hurt.
    """
    parts = [
        f"You are executing a delegated {brief.phase.value} task.",
        "Report results concisely; the lead reviews your work.",
    ]
    if brief.prescriptive:
        parts.append("Follow the brief exactly. Do not reinterpret the task.")
    if brief.allow_pushback:
        parts.append(
            "If the brief seems wrong or incomplete, say so before proceeding."
        )
    if brief.constraints:
        parts.append("Constraints: " + "; ".join(brief.constraints))
    if brief.success_criteria:
        parts.append("Success criteria: " + "; ".join(brief.success_criteria))
    return "\n".join(parts)


def _retry_after(exc: Exception) -> float | None:
    for attr in ("retry_after", "retry_after_s"):
        val = getattr(exc, attr, None)
        if isinstance(val, (int, float)):
            return float(val)
    return None


__all__ = ["Router", "RetryPolicy", "RouterError", "ConfigurationError",
           "DEFAULT_MAX_ATTEMPTS"]
