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
import json
import time
from collections import deque
from dataclasses import dataclass, replace
from typing import Any, AsyncIterator, Awaitable, Callable, Mapping, Sequence

from .adapters import (
    Completion,
    ModelAdapter,
    Usage,
    _merge_usage,
    adapter_for,
    parse_sse_line,
)
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
from .learning import QualityModel
from .policy import PolicyEngine, RouterConfig
from .pricing import PriceRegistry
from .signals import AgentProfile, FingerprintRegistry, WorkloadClassifier, WorkloadResolver
from .store import FileStateStore
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


@dataclass(frozen=True)
class StreamChunk:
    """One item from `Router.astream()`.

    Text chunks carry a `text` delta; the FINAL chunk carries the completed
    `Completion` with the provider's real usage counters.
    """

    text: str = ""
    completion: Completion | None = None


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
        quality_model: QualityModel | None = None,
        state_store: FileStateStore | None = None,
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
            quality_model=quality_model,
        )
        self._cost_usd = 0.0
        self._decisions = 0
        self._misroutes = 0
        self._delegations = 0
        self._last_decision: Decision | None = None
        self._last_user_text = ""
        #: Feedback records awaiting training (the learning loop's outbox).
        #: Bounded so a long-running router cannot grow it without limit; the
        #: oldest records are the least relevant when the traffic mix shifts.
        self._feedback: deque[dict[str, object]] = deque(maxlen=10_000)
        #: Attempts the last acomplete() needed (1 = first try). Delegated
        #: tasks fold this into pair economics: more attempts => a sidekick
        #: that needs more correction rounds => costlier as a partner.
        self._last_attempts: int = 1
        self._state_store: FileStateStore | None = None
        if state_store is not None:
            state_store.attach(self)

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
    def pure(
        cls,
        *,
        config: RouterConfig | None = None,
        profiles: Mapping[str, AgentProfile] | None = None,
        classifier: WorkloadClassifier | None = None,
        default_workload: WorkloadClass = WorkloadClass.CHAT,
        quality_model: QualityModel | None = None,
        state_store: FileStateStore | None = None,
    ) -> "Router":
        """Decision-only router over the bundled defaults: no keys, no network.

        `acomplete()` will refuse (no transport), but `decide()`, `explain()`
        and `should_delegate()` all work. Useful for tests, the CLI, cold paths,
        and for driving your own HTTP client from the Decision.
        """
        prices = PriceRegistry()
        ids = prices.ids()
        backends = BackendRegistry([
            BackendSpec(
                backend_id="anthropic", deployment=DeploymentClass.API,
                models=tuple(m for m in ids if m.startswith("claude")),
                priority=0,
            ),
            BackendSpec(
                backend_id="openai", deployment=DeploymentClass.API,
                models=tuple(m for m in ids if m.startswith("gpt")),
                priority=1,
            ),
        ])
        return cls(
            prices=prices,
            backends=backends,
            resolver=WorkloadResolver(
                profiles=profiles,
                classifier=classifier,
                default_workload=default_workload,
            ),
            config=config,
            quality_model=quality_model,
            state_store=state_store,
        )

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
        quality_model: QualityModel | None = None,
        state_store: FileStateStore | None = None,
    ) -> "Router":
        """Sensible starting point: the bundled price cards plus one backend per
        provider you supplied a key for.

        Register your own backends for self-hosted/serverless deployments — the
        bundled defaults are API-only, because deployment class is the axis you
        know and we do not.
        """
        prices = PriceRegistry()
        backends: list[BackendSpec] = cls._cloud_backends(prices, anthropic_key, openai_key)
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
            quality_model=quality_model,
            state_store=state_store,
        )
        router._keys = {"anthropic": anthropic_key, "openai": openai_key}  # type: ignore[attr-defined]
        return router

    @staticmethod
    def _cloud_backends(prices: PriceRegistry,
                        anthropic_key: str | None,
                        openai_key: str | None) -> list[BackendSpec]:
        out: list[BackendSpec] = []
        if anthropic_key:
            out.append(BackendSpec(
                backend_id="anthropic",
                deployment=DeploymentClass.API,
                models=tuple(m for m in prices.ids() if m.startswith("claude")),
                base_url="https://api.anthropic.com",
                priority=0,
            ))
        if openai_key:
            out.append(BackendSpec(
                backend_id="openai",
                deployment=DeploymentClass.API,
                models=tuple(m for m in prices.ids() if m.startswith("gpt")),
                base_url="https://api.openai.com",
                priority=1,
            ))
        return out

    @classmethod
    def with_ollama(
        cls,
        *,
        base_url: str = "http://localhost:11434",
        models: Sequence[str] | None = None,
        gpu_cost_per_hr: float = 0.35,
        tokens_per_sec: float = 35.0,
        local_quality: float = 0.72,
        per_model: Mapping[str, Mapping[str, float]] | None = None,
        capacity_rps: float = 2.0,
        anthropic_key: str | None = None,
        openai_key: str | None = None,
        profiles: Mapping[str, AgentProfile] | None = None,
        classifier: WorkloadClassifier | None = None,
        default_workload: WorkloadClass = WorkloadClass.CHAT,
        config: RouterConfig | None = None,
        transport: Any | None = None,
        sidekick_model: str | None = None,
        lead_model: str | None = None,
        delegation: DelegationPolicy | None = None,
        pairs: PairRegistry | None = None,
        quality_model: QualityModel | None = None,
        state_store: FileStateStore | None = None,
    ) -> "Router":
        """A local-first fleet: your Ollama models AND the cloud, one score.

        `models=None` discovers what is installed (GET /api/tags) at build
        time — the decision layer itself stays pure and offline. Each local
        model gets a price card whose $/MTok is derived from the GPU hourly
        rent and decode throughput (see llmrouter.local), so cheap-and-close
        beats expensive-and-frontier exactly when the arithmetic says so, and
        fails over the other way when the GPU saturates.

        Pass `anthropic_key`/`openai_key` to mix cloud capacity into the same
        candidate pool — the routing decision then spans llama-on-my-laptop and
        the frontier API on one score line.
        """
        from .local import (
            ollama_backend, ollama_models as _discover, ollama_price_cards,
        )
        names = tuple(models) if models is not None else _discover(base_url)
        if not names:
            raise ValueError(
                f"no local models found at {base_url!r} — start Ollama, pull a "
                "model, or pass models=[...] explicitly"
            )
        prices = PriceRegistry()  # bundled cloud cards stay registered; a card
        # with no healthy backend is simply never a candidate.
        prices.register_many(ollama_price_cards(
            names, gpu_cost_per_hr=gpu_cost_per_hr,
            tokens_per_sec=tokens_per_sec, quality=local_quality,
            per_model=per_model,
        ))
        backends = [ollama_backend(names, base_url=base_url,
                                   capacity_rps=capacity_rps)]
        backends.extend(cls._cloud_backends(prices, anthropic_key, openai_key))

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
            quality_model=quality_model,
            state_store=state_store,
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
        self._last_user_text = req.first_user_text()
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
    async def astream(
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
    ) -> "AsyncIterator[StreamChunk]":
        """Streaming variant of `acomplete()`: yields `StreamChunk`s.

        Each chunk carries a `text` delta; the LAST chunk carries the finished
        `completion` with real usage counters, so the learning loop and session
        affinity work exactly as they do on the unary path.

        Failover rule: an upstream failure is retried on the next backend only
        while nothing has reached the caller. Once the first token has been
        yielded, a break is raised — silently restarting mid-stream would
        duplicate text the caller already rendered.
        """
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

        if self.transport is None or not hasattr(self.transport, "stream"):
            raise ConfigurationError(
                "no streaming transport — pass an httpx.AsyncClient "
                "(client.stream is used), or use acomplete()/decide()"
            )

        attempts: list[str] = []
        last_error: Exception | None = None

        for attempt in range(self.retry.max_attempts):
            backend = self.backends.spec(decision.backend_id)
            if backend is None:
                attempts.append(f"{decision.backend_id}: no such backend")
                break
            health = self.backends.health(backend.backend_id)
            if health.is_half_open():
                health.half_open_probes += 1
            health.in_flight += 1
            yielded = False
            parts: list[str] = []
            usage: Usage | None = None
            started = time.monotonic()
            try:
                adapter, url, headers, body = self._prepare_call(
                    backend, decision, req, breakpoints, max_tokens, kwargs,
                    stream=True,
                )
                completion: Completion | None = None
                try:
                    ctx = self.transport.stream(
                        "POST", url, headers=headers, json=dict(body),
                        timeout=backend.timeout_s,
                    )
                    async with ctx as resp:
                        status = getattr(resp, "status_code", 200)
                        if status >= 400:
                            raise RouterError(
                                f"HTTP {status} from {backend.backend_id}")
                        async for line in resp.aiter_lines():
                            event = parse_sse_line(line)
                            if event is None:
                                continue
                            se = adapter.parse_stream_event(event)
                            usage = _merge_usage(usage, se.usage)
                            if se.text_delta:
                                yielded = True
                                parts.append(se.text_delta)
                                yield StreamChunk(text=se.text_delta)
                finally:
                    health.in_flight = max(0, health.in_flight - 1)
                latency_ms = (time.monotonic() - started) * 1000.0
                usage = usage or Usage()
                completion = Completion(
                    text="".join(parts), usage=usage,
                    model=usage.model or decision.model_id,
                    backend_id=backend.backend_id, latency_ms=latency_ms,
                )
            except ConfigurationError:
                raise
            except Exception as exc:  # noqa: BLE001 — re-raised after the chain
                last_error = exc
                retry_after = _retry_after(exc)
                health.record_failure(str(exc), retry_after_s=retry_after)
                attempts.append(f"{backend.backend_id}: {exc}")
                if yielded:
                    raise RouterError(
                        f"stream broke after the first token from "
                        f"{backend.backend_id}", attempts=attempts
                    ) from exc
                nxt = self._next_backend(decision, req, exclude=backend.backend_id)
                if nxt is None:
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
            assert completion is not None
            health.record_success(latency_ms)
            self._learn(decision, completion.usage, req)
            self._last_attempts = attempt + 1
            self._last_decision = replace(
                decision,
                cache_read_tokens=completion.usage.cache_read_tokens,
                cache_write_tokens=completion.usage.cache_write_tokens,
                latency_ms=completion.latency_ms,
            )
            yield StreamChunk(completion=completion)
            return

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

    def _prepare_call(self, backend: BackendSpec, decision: Decision,
                      req: RoutingRequest, breakpoints, max_tokens: int,
                      kwargs: Mapping[str, Any], *, stream: bool = False
                      ) -> tuple[ModelAdapter, str, dict[str, str], Mapping[str, Any]]:
        if self.transport is None:
            raise ConfigurationError(
                "no transport configured — install llmrouter[http] and pass an "
                "httpx.AsyncClient, or use Router.decide() for pure routing"
            )
        adapter = self._adapters.get(backend.backend_id) or adapter_for(backend.backend_id)
        card = self.prices.require(decision.model_id)
        key_param = card.cache.explicit_cache_key_param

        extra: dict[str, Any] = dict(kwargs)
        if stream:
            extra["stream"] = True
        body = adapter.build_body(
            req.messages,
            tools=req.tools,
            breakpoints=breakpoints,
            model=decision.model_id,
            cache_key=decision.sticky_key if key_param else None,
            max_tokens=max_tokens,
            **extra,
        )
        headers = self._headers(backend)
        if stream:
            headers["accept"] = "text/event-stream"
        url = f"{(backend.base_url or '').rstrip('/')}{getattr(adapter, 'path', '')}"
        return adapter, url, headers, body

    async def _call(self, backend: BackendSpec, decision: Decision,
                    req: RoutingRequest, breakpoints, max_tokens: int,
                    kwargs: Mapping[str, Any]) -> Completion:
        adapter, url, headers, body = self._prepare_call(
            backend, decision, req, breakpoints, max_tokens, kwargs)
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
    def note_usage(self, req: RoutingRequest, usage: Usage) -> None:
        """Fold provider usage counters from a call YOU made into session state.

        The sidecar (`llmrouter serve`) and callers driving their own HTTP
        client use this to keep the learned hit rate alive without the
        `acomplete()` execution layer: decide with us, execute wherever, report
        the counters back on the next turn.
        """
        session = self.sessions.peek(req.sticky_key())
        if session is None:
            return  # no session yet: nothing to attribute the counters to
        session.record_usage(usage.cache_read_tokens, usage.cache_write_tokens)
        self.sessions.put(session)

    def note_decision(self, req: RoutingRequest, decision: Decision) -> None:
        """Pin the session for a decision the CALLER executed.

        The decide-side counterpart of `_learn`: without it, a pure-decision
        consumer (HTTP sidecar, LiteLLM plugin) never builds session state and
        every turn scores cold. Usage counters arrive separately via
        `note_usage` on the following turn.
        """
        self._learn(decision, Usage(), req)

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
        if self._state_store is not None:
            self._state_store.tick(self)

    # ------------------------------------------------------------------ #
    # DURABLE STATE
    # ------------------------------------------------------------------ #
    def save_state(self) -> None:
        """Explicit snapshot write (the autosave covers steady traffic; this
        covers shutdown hooks and graceful deploys)."""
        if self._state_store is not None:
            self._state_store.save(self)

    def load_state(self) -> bool:
        """Restore a snapshot, rebasing every timestamp onto this process's
        monotonic clock. Returns False when no snapshot exists yet."""
        return self._state_store.load(self) if self._state_store else False

    def report_feedback(self, session_id: str, *, misrouted: bool,
                        expected_model: str | None = None) -> None:
        """Record a human/eval judgement.

        This closes the loop: every report becomes a training sample for the
        learned quality heads (`export_feedback` -> `llmrouter train` ->
        `QualityModel.load` -> pass it back to the Router). `expected_model`
        turns the pair (expected, chosen) into a preference row automatically.
        """
        if misrouted:
            self._misroutes += 1
        session = self.sessions.peek(f"sid:{session_id}")
        chosen = session.pinned_model if session else (
            self._last_decision.model_id if self._last_decision else None
        )
        if chosen is None:
            return  # nothing to attribute the judgement to
        workload = session.workload if session else (
            self._last_decision.workload if self._last_decision else None
        )
        row: dict[str, object] = {
            "text": self._last_user_text,
            "session_id": session_id,
        }
        if workload is not None:
            row["workload"] = workload.value
        if expected_model and misrouted:
            row.update({"model_a": expected_model, "model_b": chosen,
                        "winner": "a"})
        else:
            row.update({"model": chosen, "outcome": 0 if misrouted else 1})
        self._feedback.append(row)

    def export_feedback(self, path: str | None = None) -> str:
        """Serialize collected feedback as training JSONL (to `path` if given).
        Feed the result to `llmrouter train --data <file>`."""
        text = "\n".join(json.dumps(r, sort_keys=True) for r in self._feedback)
        if path:
            with open(path, "w", encoding="utf-8") as f:
                f.write(text + ("\n" if text else ""))
        return text

    def feedback_count(self) -> int:
        return len(self._feedback)

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
           "StreamChunk", "DEFAULT_MAX_ATTEMPTS"]
