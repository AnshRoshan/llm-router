"""llmrouter — cache-aware LLM routing.

Picks the **(model, backend) pair**, keeps conversational sessions sticky so the
prompt cache survives, and chooses prompt-cache TTL **per breakpoint**.

Quick start::

    from llmrouter import Message, Router, WorkloadClass

    router = Router.with_defaults(anthropic_key="sk-...", openai_key="sk-...")

    # Path A — declared intent (cheapest, skips classification entirely)
    out = await router.acomplete(
        [Message("user", "hello")],
        workload="chat",
        session_id="user-42",
    )

    # Path B — nothing declared; the router infers the workload class from the
    # first request and pins the session from turn 2 onward.
    out = await router.acomplete([Message("user", "refactor this function")])

    print(router.explain())      # every decision is auditable

Pure decision layer (zero dependencies, no network)::

    decision = router.decide(request)
    decision.model_id, decision.backend_id, decision.ttl_by_segment

The things this package does that most routers do not:
  * scores **(model, backend) pairs**, so workload class and deployment class
    compose instead of being decided sequentially
  * treats **KV-cache state as a first-class routing signal**, learned from the
    provider's own usage counters rather than assumed
  * picks **TTL per prompt segment** (system/tools at 1h, history at 5m)
  * learns **quality heads** from your feedback (`llmrouter train`) and blends
    them with the priors via shrinkage — zero-dependency logistic inference,
    identical in both runtimes from one JSON checkpoint
  * streams (`Router.astream`), persists its learning across restarts
    (`FileStateStore`), and serves the decision over HTTP for existing
    gateways (`llmrouter serve`)
"""
from __future__ import annotations

from .adapters import (
    AnthropicAdapter,
    Completion,
    ModelAdapter,
    OpenAIAdapter,
    StreamEvent,
    Usage,
    adapter_for,
    parse_sse_line,
)
from .affinity import (
    SessionPolicy,
    SessionStore,
    affinity_prior,
    rendezvous_pick,
    rendezvous_rank,
)
from .backends import BackendHealth, BackendRegistry, BackendSpec, HALF_OPEN_PROBE_LIMIT
from .delegation import (
    DELEGABLE_PHASES,
    LEAD_ONLY_PHASES,
    PHASE_TURN_SHARE,
    Brief,
    DelegationDecision,
    DelegationPolicy,
    DelegationResult,
    PairProfile,
    PairRegistry,
    TaskPhase,
    infer_phase,
)
from .economics import (
    EXPIRY_THRESHOLD_E_STAR,
    KEEPALIVE_BREAK_EVEN_MIN,
    Breakpoint,
    SwitchVerdict,
    break_even_turns,
    choose_ttl_by_segment,
    emit_breakpoints,
    evaluate_switch,
    keepalive_beats_upgrade,
    prefix_cost,
    session_cost_with_switch,
)
from .learning import (
    CHECKPOINT_FORMAT,
    FEATURE_NAMES,
    FeedbackSample,
    ModelHead,
    QualityModel,
    extract_features,
    load_feedback,
    train_heads,
)
from .policy import PolicyEngine, RouterConfig, Weights
from .pricing import (
    ANTHROPIC_CACHE,
    OPENAI_CACHE_CURRENT,
    OPENAI_CACHE_LEGACY,
    SELF_HOSTED_CACHE,
    CacheSemantics,
    PriceCard,
    PriceRegistry,
    default_price_cards,
)
from .router import ConfigurationError, RetryPolicy, Router, RouterError, StreamChunk
from .signals import (
    AgentProfile,
    FingerprintRegistry,
    WorkloadClassifier,
    WorkloadResolver,
    WorkloadSignal,
    infer_structural,
)
from .store import FileStateStore, restore_snapshot, take_snapshot
from .types import (
    Candidate,
    Constraints,
    Decision,
    DeploymentClass,
    Message,
    RoutingRequest,
    SegmentKind,
    WorkloadClass,
    WorkloadSource,
)

__version__ = "0.4.0"

__all__ = [
    "__version__",
    # facade
    "Router",
    "RetryPolicy",
    "RouterError",
    "ConfigurationError",
    # types
    "Message",
    "RoutingRequest",
    "Constraints",
    "Decision",
    "Candidate",
    "WorkloadClass",
    "WorkloadSource",
    "DeploymentClass",
    "SegmentKind",
    # pricing
    "PriceCard",
    "PriceRegistry",
    "CacheSemantics",
    "default_price_cards",
    "ANTHROPIC_CACHE",
    "OPENAI_CACHE_LEGACY",
    "OPENAI_CACHE_CURRENT",
    "SELF_HOSTED_CACHE",
    # delegation
    "TaskPhase",
    "DELEGABLE_PHASES",
    "LEAD_ONLY_PHASES",
    "PHASE_TURN_SHARE",
    "infer_phase",
    "Brief",
    "DelegationResult",
    "DelegationDecision",
    "DelegationPolicy",
    "PairProfile",
    "PairRegistry",
    # backends
    "BackendSpec",
    "BackendHealth",
    "BackendRegistry",
    "HALF_OPEN_PROBE_LIMIT",
    # policy
    "PolicyEngine",
    "RouterConfig",
    "Weights",
    # signals
    "WorkloadResolver",
    "WorkloadSignal",
    "WorkloadClassifier",
    "AgentProfile",
    "FingerprintRegistry",
    "infer_structural",
    # affinity
    "SessionPolicy",
    "SessionStore",
    "affinity_prior",
    "rendezvous_pick",
    "rendezvous_rank",
    # economics
    "prefix_cost",
    "session_cost_with_switch",
    "break_even_turns",
    "evaluate_switch",
    "SwitchVerdict",
    "choose_ttl_by_segment",
    "emit_breakpoints",
    "Breakpoint",
    "keepalive_beats_upgrade",
    "EXPIRY_THRESHOLD_E_STAR",
    "KEEPALIVE_BREAK_EVEN_MIN",
    # adapters
    "AnthropicAdapter",
    "OpenAIAdapter",
    "ModelAdapter",
    "Completion",
    "Usage",
    "StreamEvent",
    "parse_sse_line",
    "StreamChunk",
    "adapter_for",
    # learning
    "CHECKPOINT_FORMAT",
    "FEATURE_NAMES",
    "QualityModel",
    "ModelHead",
    "FeedbackSample",
    "extract_features",
    "load_feedback",
    "train_heads",
    # state
    "FileStateStore",
    "take_snapshot",
    "restore_snapshot",
]
