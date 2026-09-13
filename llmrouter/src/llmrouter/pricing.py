"""Pricing and cache semantics.

The multipliers below were verified on 2026-09-12 against primary documentation
(see the research dossier, SOURCES.md §G/§H). **Absolute $/token figures go stale
fast** — treat the bundled cards as defaults and override them.

Two facts that the price registry must encode and most routers get wrong:

1. Cache discount is **per model, not per provider**. OpenAI's discount is 50% on
   gpt-4o/4.1 and 90% on GPT-5.x. Branching on `provider` is a bug.
2. Anthropic charges a cache **write** premium (1.25× for 5m, 2.0× for 1h);
   OpenAI pre-GPT-5.6 charges **no write premium at all**. This single field is
   what decides whether a mid-session model switch is cheap or expensive.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Mapping

from .types import DeploymentClass


@dataclass(frozen=True)
class CacheSemantics:
    """How one model's prompt cache behaves. Per model, never per provider."""

    read_mult: float = 0.10
    """Cache read as a multiple of base input price. 0.10 = 90% off."""

    write_mult_5m: float = 1.25
    """5-minute cache write multiplier. 1.0 means no write premium (OpenAI)."""

    write_mult_1h: float = 2.00
    """1-hour cache write multiplier."""

    min_cacheable_tokens: int = 1024
    """Below this prefix length no cache entry is created, so affinity is moot."""

    ttl_options: tuple[str, ...] = ("5m",)
    """Which TTL tiers exist. Do not model what the provider lacks."""

    ttl_refreshes_on_hit: bool = True
    """Anthropic: True. An active session never expires on the 5m tier."""

    supports_mixed_ttl: bool = False
    """Anthropic: True, with the constraint that longer TTL must precede shorter."""

    max_breakpoints: int = 4

    invalidating_fields: frozenset[str] = frozenset()
    """Request fields that are part of the cache key. Changing them mid-session
    costs a full prefix rewrite — see Rule F."""

    explicit_cache_key_param: str | None = None
    """OpenAI exposes `prompt_cache_key` so the provider does affinity routing
    for you. Anthropic has no equivalent."""

    def write_mult(self, ttl: str) -> float:
        if ttl == "1h":
            return self.write_mult_1h
        return self.write_mult_5m

    def can_cache(self, prefix_tokens: int) -> bool:
        return prefix_tokens >= self.min_cacheable_tokens


@dataclass(frozen=True)
class PriceCard:
    model_id: str
    input_per_mtok: float
    output_per_mtok: float
    cache: CacheSemantics = field(default_factory=CacheSemantics)
    context_window: int = 200_000
    supports_tools: bool = True
    supports_vision: bool = False
    deployment: DeploymentClass = DeploymentClass.API
    residency_tags: frozenset[str] = frozenset()
    # Quality prior per workload class, 0..1. Calibrate these on your own traffic;
    # the bundled values are placeholders, NOT benchmarks.
    quality_prior: Mapping[str, float] = field(default_factory=dict)
    p50_latency_ms: float = 800.0
    #: Price-per-task factors. Cognition's Fusion data shows a model costing 275%
    #: more per token can be 2% CHEAPER per task, because stronger models are more
    #: token-efficient and generate fewer rework rounds. `tokens x $/token` alone
    #: cannot express that, so these scale the effective cost.
    token_efficiency: float = 1.0
    """Tokens needed per unit of work, relative to a 1.0 baseline. <1 = more
    efficient, so fewer tokens per task and lower real cost."""
    rework_factor: float = 1.0
    """>1 means output often needs correction, multiplying downstream cost."""

    @property
    def task_cost_multiplier(self) -> float:
        """Effective cost per TASK relative to sticker price per token."""
        return self.token_efficiency * self.rework_factor

    def quality(self, workload: str) -> float:
        return float(self.quality_prior.get(workload, 0.5))

    # ---- money ---------------------------------------------------------- #
    def input_cost(self, tokens: int) -> float:
        return tokens / 1_000_000 * self.input_per_mtok

    def output_cost(self, tokens: int) -> float:
        return tokens / 1_000_000 * self.output_per_mtok

    def cache_read_cost(self, tokens: int) -> float:
        return self.input_cost(tokens) * self.cache.read_mult

    def cache_write_cost(self, tokens: int, ttl: str = "5m") -> float:
        return self.input_cost(tokens) * self.cache.write_mult(ttl)


# --------------------------------------------------------------------------- #
# Bundled defaults
# --------------------------------------------------------------------------- #
# Cache semantics verified 2026-09-12 against primary docs.
ANTHROPIC_CACHE = CacheSemantics(
    read_mult=0.10,
    write_mult_5m=1.25,
    write_mult_1h=2.00,
    min_cacheable_tokens=1024,
    ttl_options=("5m", "1h"),
    ttl_refreshes_on_hit=True,
    supports_mixed_ttl=True,
    max_breakpoints=4,
    invalidating_fields=frozenset(
        {"model", "thinking", "tool_choice", "images", "output_config.effort"}
    ),
    explicit_cache_key_param=None,
)

# Haiku-class models have a higher cache floor.
ANTHROPIC_CACHE_SMALL = replace(ANTHROPIC_CACHE, min_cacheable_tokens=4096)

# Legacy OpenAI (gpt-4o / 4.1): automatic, 50% discount, NO write premium.
OPENAI_CACHE_LEGACY = CacheSemantics(
    read_mult=0.50,
    write_mult_5m=1.00,
    write_mult_1h=1.00,
    min_cacheable_tokens=1024,
    ttl_options=("5m",),
    ttl_refreshes_on_hit=True,
    supports_mixed_ttl=False,
    max_breakpoints=0,
    invalidating_fields=frozenset({"model"}),
    explicit_cache_key_param="prompt_cache_key",
)

# Current OpenAI (GPT-5.x): 90% discount, still no write premium pre-5.6.
OPENAI_CACHE_CURRENT = replace(OPENAI_CACHE_LEGACY, read_mult=0.10)

# Self-hosted (vLLM APC): compute optimization, not a billed tier. No write
# premium, and reads are not separately priced — the saving is throughput.
SELF_HOSTED_CACHE = CacheSemantics(
    read_mult=1.0,
    write_mult_5m=1.0,
    write_mult_1h=1.0,
    min_cacheable_tokens=512,
    ttl_options=(),
    ttl_refreshes_on_hit=False,
    supports_mixed_ttl=False,
    max_breakpoints=0,
    invalidating_fields=frozenset({"model"}),
)


def default_price_cards() -> dict[str, PriceCard]:
    """Bundled defaults.

    ⚠ The $/token figures are illustrative starting points. Override with your
    contracted rates — the *cache multipliers* are the part that was verified.
    Quality priors are placeholders to be replaced by calibration on your traffic.
    """
    # "vision" must appear in every prior: a vision request scored against a card
    # with no vision entry would silently fall back to the 0.5 neutral default.
    strong = {"agent": 0.90, "chat": 0.92, "rag": 0.88, "code": 0.90,
              "batch": 0.85, "vision": 0.90}
    mid = {"agent": 0.75, "chat": 0.85, "rag": 0.78, "code": 0.76,
           "batch": 0.75, "vision": 0.80}
    weak = {"agent": 0.55, "chat": 0.75, "rag": 0.62, "code": 0.58,
            "batch": 0.60, "vision": 0.45}

    return {
        "claude-opus-class": PriceCard(
            "claude-opus-class", 15.0, 75.0, ANTHROPIC_CACHE,
            quality_prior=strong, p50_latency_ms=1200, supports_vision=True,
        ),
        "claude-sonnet-class": PriceCard(
            "claude-sonnet-class", 3.0, 15.0, ANTHROPIC_CACHE,
            quality_prior=mid, p50_latency_ms=900, supports_vision=True,
        ),
        "claude-haiku-class": PriceCard(
            "claude-haiku-class", 1.0, 5.0, ANTHROPIC_CACHE_SMALL,
            quality_prior=weak, p50_latency_ms=500,
        ),
        "gpt-frontier-class": PriceCard(
            "gpt-frontier-class", 5.0, 20.0, OPENAI_CACHE_CURRENT,
            quality_prior=strong, p50_latency_ms=1000, supports_vision=True,
        ),
        "gpt-legacy-class": PriceCard(
            "gpt-legacy-class", 2.5, 10.0, OPENAI_CACHE_LEGACY,
            quality_prior=mid, p50_latency_ms=800,
        ),
        "gpt-mini-class": PriceCard(
            "gpt-mini-class", 0.75, 3.0, OPENAI_CACHE_CURRENT,
            quality_prior=weak, p50_latency_ms=450,
        ),
    }


class PriceRegistry:
    """Mutable registry so prices can be hot-swapped without a restart.

    Thread-safety note: dict assignment is atomic under the GIL, and we only ever
    replace whole entries, so reads never see a half-written card.
    """

    def __init__(self, cards: Mapping[str, PriceCard] | None = None) -> None:
        # Explicit `is None`: an EMPTY mapping means "start from nothing" —
        # the falsy-or-default idiom here silently resurrected the bundled
        # cards and was exactly the registry bug class the research log warns
        # about.
        self._cards: dict[str, PriceCard] = dict(
            default_price_cards() if cards is None else cards
        )

    def register(self, card: PriceCard) -> None:
        self._cards[card.model_id] = card

    def unregister(self, model_id: str) -> None:
        self._cards.pop(model_id, None)

    def get(self, model_id: str) -> PriceCard | None:
        return self._cards.get(model_id)

    def require(self, model_id: str) -> PriceCard:
        card = self._cards.get(model_id)
        if card is None:
            raise KeyError(f"no price card for model {model_id!r}")
        return card

    def all(self) -> tuple[PriceCard, ...]:
        return tuple(self._cards.values())

    def ids(self) -> tuple[str, ...]:
        return tuple(self._cards)

    def __len__(self) -> int:
        return len(self._cards)

    def __contains__(self, model_id: object) -> bool:
        return model_id in self._cards


__all__ = [
    "CacheSemantics",
    "PriceCard",
    "PriceRegistry",
    "default_price_cards",
    "ANTHROPIC_CACHE",
    "ANTHROPIC_CACHE_SMALL",
    "OPENAI_CACHE_LEGACY",
    "OPENAI_CACHE_CURRENT",
    "SELF_HOSTED_CACHE",
]
