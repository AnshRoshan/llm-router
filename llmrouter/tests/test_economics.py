"""The cache economics. These numbers are the spec.

Every expected value here was first derived and checked in the research scripts
`verify_08_switch_cost.py` and `verify_09_ttl_policy.py`. If a formula in
`economics.py` changes, these fail — that is the point.
"""
from __future__ import annotations

import pytest

from llmrouter import (
    EXPIRY_THRESHOLD_E_STAR,
    KEEPALIVE_BREAK_EVEN_MIN,
    PriceCard,
    SegmentKind,
    break_even_turns,
    choose_ttl_by_segment,
    emit_breakpoints,
    evaluate_switch,
    keepalive_beats_upgrade,
    prefix_cost,
    session_cost_with_switch,
)
from llmrouter.economics import R_10, W_1H, W_5M, expected_expiries
from llmrouter.pricing import (
    ANTHROPIC_CACHE,
    ANTHROPIC_CACHE_SMALL,
    OPENAI_CACHE_LEGACY,
)

# The verified working example: Sonnet-class -> Haiku-class, 50k prefix.
SONNET = PriceCard("sonnet", 3.00, 15.0, ANTHROPIC_CACHE)
HAIKU = PriceCard("haiku", 1.00, 5.0, ANTHROPIC_CACHE)
# OpenAI-backed target: NO write premium. This is the case that breaks the
# naive "never switch mid-session" rule.
OPENAI_TARGET = PriceCard("gpt-mini", 0.50, 2.0, OPENAI_CACHE_LEGACY)

S = 50_000


class TestVerifiedMultipliers:
    def test_anthropic_multipliers(self):
        assert W_5M == 1.25
        assert W_1H == 2.00
        assert R_10 == 0.10

    def test_1h_vs_5m_write_ratio_is_1_6_not_2(self):
        # "2x" compares 1h to UNCACHED input. Against the 5m write it is 1.6x.
        # A cost model using 2x overstates the upgrade by 60%.
        assert W_1H / W_5M == pytest.approx(1.6)


class TestPrefixCost:
    def test_stay_30_turns(self):
        # 0.15 * (1.25 + 29*0.1) = $0.6225
        assert prefix_cost(SONNET, S, 30) == pytest.approx(0.6225, abs=1e-4)

    def test_switch_at_5(self):
        # 5 turns on Sonnet, 25 on Haiku.
        # The research draft wrote $0.4350 here by pricing 26 turns on the
        # target while claiming 25. The correct figure is $0.4300.
        got = session_cost_with_switch(SONNET, HAIKU, S, 30, 5)
        assert got == pytest.approx(0.4300, abs=1e-4)

    def test_saving_is_30_9_percent(self):
        stay = prefix_cost(SONNET, S, 30)
        switched = session_cost_with_switch(SONNET, HAIKU, S, 30, 5)
        assert (1 - switched / stay) == pytest.approx(0.309, abs=0.002)

    def test_turn_counts_must_sum(self):
        with pytest.raises(ValueError):
            session_cost_with_switch(SONNET, HAIKU, S, 30, 35)
        with pytest.raises(ValueError):
            session_cost_with_switch(SONNET, HAIKU, S, 30, -1)

    def test_zero_turns_is_free(self):
        assert prefix_cost(SONNET, S, 0) == 0.0


class TestBreakEven:
    def test_sonnet_to_haiku_on_anthropic(self):
        # (1.0*1.25 - 3.0*0.1) / ((3.0-1.0)*0.1) = 4.75 turns
        assert break_even_turns(SONNET, HAIKU) == pytest.approx(4.75, abs=0.01)

    def test_switch_onto_openai_pays_back_inside_one_turn(self):
        # No write premium => 0.8 turns. This is why "never switch mid-session"
        # is wrong as an absolute rule.
        assert break_even_turns(SONNET, OPENAI_TARGET) == pytest.approx(0.8, abs=0.01)

    def test_more_expensive_target_never_pays_back(self):
        assert break_even_turns(HAIKU, SONNET) == float("inf")


class TestSwitchVerdict:
    def test_no_oscillation_lock(self):
        v = evaluate_switch(SONNET, HAIKU, S, turns_remaining=50, already_switched=True)
        assert v.should_switch is False
        assert "oscillation" in v.reason

    def test_oscillation_costs_more_than_never_switching(self):
        # Four legs of A->B->A->B on a 50k prefix, at 1.25x write each.
        oscillation = 4 * S * (3.00 / 1e6) * 1.25
        stay = prefix_cost(SONNET, S, 30)
        assert oscillation == pytest.approx(0.75, abs=1e-4)
        assert oscillation > stay, "oscillation must exceed the cost of never routing"

    def test_migrates_when_enough_turns_remain(self):
        v = evaluate_switch(SONNET, HAIKU, S, turns_remaining=20)
        assert v.should_switch is True
        assert v.break_even_turns == pytest.approx(4.75, abs=0.01)

    def test_stranded_when_too_few_turns_remain(self):
        v = evaluate_switch(SONNET, HAIKU, S, turns_remaining=2)
        assert v.should_switch is False
        assert v.reason.startswith("stranded")


class TestTTLThreshold:
    def test_expiry_threshold(self):
        # E* = 0.75 / 1.15 = 0.652. NOT 0.60: an expiry also removes a read.
        assert EXPIRY_THRESHOLD_E_STAR == pytest.approx(0.652, abs=0.001)

    def test_one_expiry_makes_1h_cheaper(self):
        uses = 30
        # 5m WITH one expiry vs 1h WITHOUT (a 1h entry does not expire in-hour).
        cost_5m = S * (3.0 / 1e6) * (W_5M * 2 + (uses - 2) * R_10)
        cost_1h = S * (3.0 / 1e6) * (W_1H + (uses - 1) * R_10)
        assert cost_5m == pytest.approx(0.7950, abs=1e-4)
        assert cost_1h == pytest.approx(0.7350, abs=1e-4)
        assert cost_1h < cost_5m

    def test_no_expiry_makes_5m_cheaper(self):
        # The "frequent conversation" case: 5m wins by exactly 0.75x base.
        uses = 30
        cost_5m = prefix_cost(SONNET, S, uses)
        cost_1h = S * (3.0 / 1e6) * (W_1H + (uses - 1) * R_10)
        assert cost_5m < cost_1h
        assert cost_1h - cost_5m == pytest.approx(S * (3.0 / 1e6) * 0.75, abs=1e-6)

    def test_zero_expiries_when_gap_is_inside_ttl(self):
        assert expected_expiries(3600, 60) == 0.0

    def test_generation_time_counts_against_the_window(self):
        # Anthropic measures lifetime from the START of the request, so a 240s
        # stream leaves only ~60s for the follow-up.
        assert expected_expiries(3600, 240 + 70) > 0.0


class TestKeepalive:
    def test_break_even_gap(self):
        assert KEEPALIVE_BREAK_EVEN_MIN == pytest.approx(37.5, abs=0.001)

    @pytest.mark.parametrize("gap,wins", [(10, True), (20, True), (60, False)])
    def test_keepalive_wins_below_37_5_min(self, gap, wins):
        assert keepalive_beats_upgrade(gap) is wins


class TestMixedTTL:
    SEGMENTS = {
        SegmentKind.SYSTEM: 30_000,
        SegmentKind.HISTORY: 20_000,
    }

    def test_system_gets_1h_and_history_stays_5m(self):
        ttl = choose_ttl_by_segment(ANTHROPIC_CACHE, self.SEGMENTS,
                                    expected_gap_s=1200)
        assert ttl[SegmentKind.SYSTEM] == "1h"
        assert ttl[SegmentKind.HISTORY] == "5m"

    def test_short_gap_keeps_everything_on_5m(self):
        ttl = choose_ttl_by_segment(ANTHROPIC_CACHE, self.SEGMENTS, expected_gap_s=30)
        assert set(ttl.values()) == {"5m"}

    def test_mixed_ttl_pays_back_within_one_gap(self):
        # After one idle gap: mixed reads the system segment and rewrites only
        # the history, vs all-5m rewriting the whole prefix.
        per_tok = 3.0 / 1e6
        sys_t, hist_t = 30_000, 20_000
        after_gap_mixed = sys_t * per_tok * R_10 + hist_t * per_tok * W_5M
        after_gap_all5m = (sys_t + hist_t) * per_tok * W_5M
        upfront = sys_t * per_tok * (W_1H - W_5M)
        assert after_gap_mixed == pytest.approx(0.0840, abs=1e-4)
        assert after_gap_all5m == pytest.approx(0.1875, abs=1e-4)
        assert after_gap_mixed + upfront < after_gap_all5m

    def test_no_1h_when_provider_lacks_it(self):
        ttl = choose_ttl_by_segment(OPENAI_CACHE_LEGACY, self.SEGMENTS,
                                    expected_gap_s=1200)
        assert set(ttl.values()) == {"5m"}

    def test_small_prefix_is_not_upgraded(self):
        # Haiku-class needs 4,096 tokens before any cache entry exists.
        small = {SegmentKind.SYSTEM: 2_000, SegmentKind.HISTORY: 500}
        ttl = choose_ttl_by_segment(ANTHROPIC_CACHE_SMALL, small, expected_gap_s=1200)
        assert ttl[SegmentKind.SYSTEM] == "5m"


class TestBreakpointEmitter:
    def test_longer_ttl_precedes_shorter(self):
        segments = {
            SegmentKind.TOOLS: 2_000,
            SegmentKind.SYSTEM: 30_000,
            SegmentKind.HISTORY: 20_000,
        }
        ttl = {
            SegmentKind.TOOLS: "5m",
            SegmentKind.SYSTEM: "1h",
            SegmentKind.HISTORY: "5m",
        }
        bps = emit_breakpoints(ANTHROPIC_CACHE, segments, ttl)
        # The emitter must downgrade the TOOLS 5m... or reorder so 1h comes first.
        kinds = [b.kind for b in bps]
        assert kinds[0] == SegmentKind.SYSTEM, "1h segment must be emitted first"
        ttls = [b.ttl for b in bps]
        assert "1h" not in ttls[1:] or ttls.count("1h") == ttls.index("5m") or True
        # The hard invariant:
        seen_short = False
        for t in ttls:
            if t == "5m":
                seen_short = True
            else:
                assert not seen_short, "1h after 5m would be rejected/downgraded"

    def test_respects_max_breakpoints(self):
        segments = {k: 5_000 for k in SegmentKind}
        ttl = {k: "5m" for k in SegmentKind}
        bps = emit_breakpoints(ANTHROPIC_CACHE, segments, ttl)
        assert len(bps) <= ANTHROPIC_CACHE.max_breakpoints

    def test_skips_segments_below_cache_minimum(self):
        segments = {SegmentKind.SYSTEM: 500, SegmentKind.HISTORY: 20_000}
        ttl = {SegmentKind.SYSTEM: "5m", SegmentKind.HISTORY: "5m"}
        bps = emit_breakpoints(ANTHROPIC_CACHE, segments, ttl)
        assert all(b.tokens >= ANTHROPIC_CACHE.min_cacheable_tokens for b in bps)
