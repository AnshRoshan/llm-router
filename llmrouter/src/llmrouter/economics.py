"""Cache-aware cost algebra.

Every formula here is ported from the verified research scripts
(`verify_08_switch_cost.py`, `verify_09_ttl_policy.py`) and the tests in
`tests/test_economics.py` assert the same numbers those scripts assert. If you
change a formula here, change the test — the numbers are the spec.

Notation used throughout:
    c, c'   base input $/token, current model / target model
    w, w'   cache-write multiplier (tier-dependent)
    r       cache-read multiplier
    S       tokens in the shared prefix
    N/T     total turns, R = turns remaining after a switch
"""
from __future__ import annotations

from dataclasses import dataclass

from .pricing import CacheSemantics, PriceCard
from .types import SegmentKind

# Verified constants (2026-09-12, primary docs).
W_5M = 1.25
W_1H = 2.00
R_10 = 0.10

#: Expiry threshold for the 1-hour upgrade: E* = 0.75 / 1.15 = 0.652.
#: Any single expected expiry during the session makes 1h cheaper than 5m.
EXPIRY_THRESHOLD_E_STAR = (W_1H - W_5M) / (W_5M - R_10)

#: Gap (minutes) at which keepalive pinging costs the same as the 1h upgrade.
KEEPALIVE_BREAK_EVEN_MIN = (W_1H - W_5M) / R_10 * 5  # 37.5

#: Default 5-minute TTL in seconds.
TTL_5M_S = 300.0


# --------------------------------------------------------------------------- #
# Session prefix cost
# --------------------------------------------------------------------------- #
def prefix_cost(card: PriceCard, prefix_tokens: int, turns: int, ttl: str = "5m") -> float:
    """Cost of a shared prefix over `turns` turns, staying on one model.

    Turn 1 writes at the tier's write multiplier; turns 2..N read at read_mult.
    """
    if turns <= 0:
        return 0.0
    sem = card.cache
    w = sem.write_mult(ttl)
    per_tok = card.input_per_mtok / 1_000_000
    return prefix_tokens * per_tok * (w + (turns - 1) * sem.read_mult)


def session_cost_with_switch(
    cur: PriceCard,
    tgt: PriceCard,
    prefix_tokens: int,
    total_turns: int,
    switch_at: int,
) -> float:
    """Direct summation: `switch_at` turns on `cur`, the remainder on `tgt`.

    Convention: `switch_at = k` means turns 1..k run on `cur` and k+1..N on `tgt`,
    so R = total_turns - k. The assertion below exists because the research
    draft got exactly this wrong and overstated the saving.
    """
    stay = switch_at
    remaining = total_turns - switch_at
    if stay < 0 or remaining < 0 or stay + remaining != total_turns:
        raise ValueError(
            f"switch_at={switch_at} inconsistent with total_turns={total_turns}"
        )
    return prefix_cost(cur, prefix_tokens, stay) + prefix_cost(tgt, prefix_tokens, remaining)


def break_even_turns(cur: PriceCard, tgt: PriceCard) -> float:
    """R* — turns needed for a one-way migration to pay for itself.

        R* = (c'·w' − c·r) / ((c − c')·r)

    Prices the migration against a *read* on the old model. Returns inf when the
    target is not cheaper per turn (migration can never pay back).
    """
    r = cur.cache.read_mult
    denom = (cur.input_per_mtok - tgt.input_per_mtok) * r
    if denom <= 0:
        return float("inf")
    num = tgt.input_per_mtok * tgt.cache.write_mult_5m - cur.input_per_mtok * r
    return num / denom


@dataclass(frozen=True)
class SwitchVerdict:
    should_switch: bool
    break_even_turns: float
    turns_remaining: int
    one_time_cost_usd: float
    per_turn_saving_usd: float
    reason: str


def evaluate_switch(
    cur: PriceCard,
    tgt: PriceCard,
    prefix_tokens: int,
    turns_remaining: int,
    already_switched: bool = False,
) -> SwitchVerdict:
    """Should we migrate a live session from `cur` to `tgt`?

    Three rules, in order:
      1. **No oscillation.** A session that has already migrated once does not
         migrate again. Four legs of A→B→A→B on a 50k prefix costs more than
         never routing at all.
      2. **Amortization.** The migration must pay back inside the turns left.
      3. Otherwise **stranded** — we wanted to save money and could not. That
         counter is a real, surfaced metric, not a silent no-op.
    """
    if already_switched:
        return SwitchVerdict(False, 0.0, turns_remaining, 0.0, 0.0,
                             "no-oscillation lock: session already migrated once")

    r_star = break_even_turns(cur, tgt)
    if r_star == float("inf"):
        return SwitchVerdict(False, r_star, turns_remaining, 0.0, 0.0,
                             "target is not cheaper per turn")

    per_tok = cur.input_per_mtok / 1_000_000
    one_time = prefix_tokens * (
        tgt.input_per_mtok / 1_000_000 * tgt.cache.write_mult_5m
        - cur.input_per_mtok / 1_000_000 * cur.cache.read_mult
    )
    per_turn = (cur.input_per_mtok - tgt.input_per_mtok) * cur.cache.read_mult / 1_000_000
    per_turn *= prefix_tokens

    if turns_remaining >= r_star:
        return SwitchVerdict(True, r_star, turns_remaining, one_time, per_turn,
                             f"amortizes in {r_star:.2f} turns, {turns_remaining} left")
    return SwitchVerdict(False, r_star, turns_remaining, one_time, per_turn,
                         f"stranded: needs {r_star:.2f} turns, only {turns_remaining} left")


# --------------------------------------------------------------------------- #
# TTL tier policy
# --------------------------------------------------------------------------- #
def expected_expiries(
    session_duration_s: float,
    gap_s: float,
    ttl_s: float = TTL_5M_S,
) -> float:
    """How many times a 5m cache is expected to expire.

    `gap_s` should be `expected_gap_s + expected_generation_s`: Anthropic measures
    the lifetime from the START of the request, so generation time is charged
    against the window. A 240s stream leaves only ~60s for the follow-up.
    """
    if gap_s <= ttl_s or session_duration_s <= 0:
        return 0.0
    return max(0.0, session_duration_s / gap_s - 1)


def choose_ttl_for_prefix(
    sem: CacheSemantics,
    prefix_tokens: int,
    expected_exposures_s: float,
    ttl_s: float = TTL_5M_S,
) -> str:
    """Pick a TTL for a single prefix segment.

    `expected_exposures_s` = expected idle gap + expected generation time.
    """
    if "1h" not in sem.ttl_options:
        return "5m"
    if not sem.can_cache(prefix_tokens):
        return "5m"  # nothing will be cached either way; take the cheaper write
    # One expected expiry is enough to justify the upgrade (E* = 0.652).
    return "1h" if expected_exposures_s > ttl_s * EXPIRY_THRESHOLD_E_STAR * 2 else "5m"


#: Which prompt segments are worth protecting across a human-sized gap.
LONG_TTL_SEGMENTS = frozenset(
    {SegmentKind.TOOLS, SegmentKind.SYSTEM, SegmentKind.RETRIEVED}
)


def choose_ttl_by_segment(
    sem: CacheSemantics,
    segment_tokens: dict[SegmentKind, int],
    expected_gap_s: float,
    expected_generation_s: float = 0.0,
    ttl_s: float = TTL_5M_S,
) -> dict[SegmentKind, str]:
    """Per-breakpoint TTL: system/tools/retrieved at 1h, conversation history at 5m.

    This is the highest-value cache lever available and almost nobody ships it.
    Conversation history is append-only and read once before it grows, so paying
    2× to cache text that changes every turn is pure waste.
    """
    if not sem.supports_mixed_ttl or "1h" not in sem.ttl_options:
        return {k: "5m" for k in segment_tokens}

    exposures = expected_gap_s + expected_generation_s
    out: dict[SegmentKind, str] = {}
    for kind, tokens in segment_tokens.items():
        if kind in LONG_TTL_SEGMENTS and sem.can_cache(tokens):
            out[kind] = "1h" if exposures > ttl_s else "5m"
        else:
            out[kind] = "5m"
    return out


@dataclass(frozen=True)
class Breakpoint:
    kind: SegmentKind
    tokens: int
    ttl: str
    offset: int


def emit_breakpoints(
    sem: CacheSemantics,
    segment_tokens: dict[SegmentKind, int],
    ttl_by_segment: dict[SegmentKind, str],
) -> tuple[Breakpoint, ...]:
    """Build the cache_control breakpoint list, enforcing the hard constraints.

    Invariants (violating any of these silently breaks caching or gets the
    request rejected):
      * at most `max_breakpoints` breakpoints
      * **longer TTL must precede shorter TTL** in prefix order — this is the bug
        that shipped in a real proxy: injected default 5m blocks caused later
        `ttl: "1h"` blocks to be stripped
      * segments under `min_cacheable_tokens` are skipped entirely
      * prefix order is tools → system → retrieved → history → user
    """
    order = [
        SegmentKind.TOOLS,
        SegmentKind.SYSTEM,
        SegmentKind.RETRIEVED,
        SegmentKind.HISTORY,
        SegmentKind.USER,
    ]
    ranked = sorted(
        (k for k in order if k in segment_tokens),
        # longer TTL first, then canonical prefix order
        key=lambda k: (0 if ttl_by_segment.get(k, "5m") == "1h" else 1, order.index(k)),
    )

    out: list[Breakpoint] = []
    offset = 0
    seen_short = False
    for kind in ranked:
        tokens = segment_tokens[kind]
        ttl = ttl_by_segment.get(kind, "5m")
        if not sem.can_cache(tokens):
            offset += tokens
            continue
        if len(out) >= sem.max_breakpoints:
            offset += tokens
            continue
        if ttl == "5m":
            seen_short = True
        elif seen_short:
            # A 1h entry after a 5m entry is rejected/downgraded — downgrade it.
            ttl = "5m"
        out.append(Breakpoint(kind=kind, tokens=tokens, ttl=ttl, offset=offset))
        offset += tokens

    _assert_ttl_ordering(out)
    return tuple(out)


def _assert_ttl_ordering(breakpoints: tuple[Breakpoint, ...]) -> None:
    """Property test baked into the emitter: no 1h after a 5m."""
    seen_short = False
    for bp in breakpoints:
        if bp.ttl == "5m":
            seen_short = True
        elif seen_short:
            raise AssertionError(
                f"1h breakpoint for {bp.kind.value} appears after a 5m breakpoint; "
                "the provider will reject or downgrade this request"
            )


# --------------------------------------------------------------------------- #
# Keepalive
# --------------------------------------------------------------------------- #
def keepalive_cost_multiplier(gap_minutes: float) -> float:
    """Cost of pinging every 5 min across a gap, in units of base input price.

    Cheaper than the 1h upgrade below ~37.5 minutes. Off by default: a ping is a
    real API call and needs a background timer per live session.
    """
    return (gap_minutes / 5.0) * R_10


def keepalive_beats_upgrade(gap_minutes: float) -> bool:
    return keepalive_cost_multiplier(gap_minutes) < (W_1H - W_5M)


__all__ = [
    "W_5M",
    "W_1H",
    "R_10",
    "EXPIRY_THRESHOLD_E_STAR",
    "KEEPALIVE_BREAK_EVEN_MIN",
    "TTL_5M_S",
    "LONG_TTL_SEGMENTS",
    "prefix_cost",
    "session_cost_with_switch",
    "break_even_turns",
    "evaluate_switch",
    "SwitchVerdict",
    "expected_expiries",
    "choose_ttl_for_prefix",
    "choose_ttl_by_segment",
    "emit_breakpoints",
    "Breakpoint",
    "keepalive_cost_multiplier",
    "keepalive_beats_upgrade",
]
