"""Verify the switch-cost algebra in 08-session-stickiness-and-entry-intent.md §8.3.

Run: python3 verify_08_switch_cost.py
Pure functions, no I/O. Numbers cross-checked against the worked examples in the doc.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class CachePricing:
    base_input_per_tok: float   # $/token
    write_mult: float           # cache-write multiplier (Anthropic 5m=1.25, 1h=2.0, OpenAI pre-5.6=1.0)
    read_mult: float            # cache-read multiplier (0.1 = 90% off, 0.5 = 50% off)


def prefix_cost(S: int, p: CachePricing, turns: int) -> float:
    """Cost of the S-token shared prefix over `turns` turns, staying on one model.

    Turn 1 writes (write_mult), turns 2..turns read (read_mult).
    """
    if turns <= 0:
        return 0.0
    return S * p.base_input_per_tok * (p.write_mult + (turns - 1) * p.read_mult)


def switch_cost_formula(S: int, cur: CachePricing, tgt: CachePricing) -> float:
    """One-time cost of migrating: cold-write on target minus the read we gave up."""
    return S * (tgt.base_input_per_tok * tgt.write_mult - cur.base_input_per_tok * cur.read_mult)


def per_turn_saving(S: int, cur: CachePricing, tgt: CachePricing) -> float:
    return (cur.base_input_per_tok - tgt.base_input_per_tok) * cur.read_mult * S


def break_even_turns(cur: CachePricing, tgt: CachePricing) -> float:
    """R* from 08 §8.3. Note: prices the migration against a READ on the old model."""
    denom = (cur.base_input_per_tok - tgt.base_input_per_tok) * cur.read_mult
    if denom <= 0:
        return float("inf")
    return (tgt.base_input_per_tok * tgt.write_mult - cur.base_input_per_tok * cur.read_mult) / denom


def session_cost_with_switch(S: int, cur: CachePricing, tgt: CachePricing,
                             total_turns: int, switch_at: int) -> float:
    """Direct summation.

    Convention (matches 08 §8.3): `switch_at = k` means turns 1..k run on `cur`
    and turns k+1..total_turns run on `tgt`, so R = total_turns - k.
    Turn counts must sum to total_turns -- asserted, because the doc's original
    worked example got exactly this wrong.
    """
    stay_turns = switch_at
    remaining = total_turns - switch_at
    assert stay_turns + remaining == total_turns
    assert stay_turns >= 0 and remaining >= 0
    return prefix_cost(S, cur, stay_turns) + prefix_cost(S, tgt, remaining)


# ---- the doc's worked example: Sonnet -> Haiku, 50k prefix, Anthropic pricing ----
S = 50_000
sonnet = CachePricing(3.00 / 1e6, 1.25, 0.10)
haiku = CachePricing(1.00 / 1e6, 1.25, 0.10)

results = []

r_star = break_even_turns(sonnet, haiku)
results.append(("R* Sonnet->Haiku (Anthropic)", f"{r_star:.2f} turns", "doc says 4.75"))

stay30 = prefix_cost(S, sonnet, 30)
sw5 = session_cost_with_switch(S, sonnet, haiku, 30, 5)
results.append(("stay 30 turns", f"${stay30:.4f}", "doc says $0.6225"))
results.append(("switch@5 (5 turns Sonnet, 25 Haiku)", f"${sw5:.4f}", "corrected: $0.4300"))
results.append(("saving", f"{100 * (1 - sw5 / stay30):.1f}%", "corrected: 30.9%"))

# OpenAI-backed target: no write premium
oai_target = CachePricing(0.50 / 1e6, 1.00, 0.10)
results.append(("R* Sonnet->OpenAI-backed (no write premium)",
                f"{break_even_turns(sonnet, oai_target):.2f} turns", "doc says 0.8"))

# Oscillation: A->B->A->B on a 50k prefix = 4 cold writes
osc = 4 * S * sonnet.base_input_per_tok * sonnet.write_mult
results.append(("oscillation 4 legs", f"${osc:.4f}", "doc says $0.75"))
results.append(("oscillation > stay-30-turns?", str(osc > stay30), "doc says True"))

# 1-hour TTL: 9 expiries over 45 min vs 1 write (blog.sandbase.ai example, 50k Sonnet)
five_min = 9 * (S * 3.00 / 1e6 * 1.25) + 21 * (S * 3.00 / 1e6 * 0.10)
one_hour = 1 * (S * 3.00 / 1e6 * 1.50) + 29 * (S * 3.00 / 1e6 * 0.10)
results.append(("5-min TTL, 30 calls / 45 min", f"${five_min:.4f}", "source says $2.0025"))
results.append(("1-hour TTL, same", f"${one_hour:.4f}", "source says $0.66"))

# Minimum cacheable prefix: 2,000-token chat prefix on Haiku (min 4,096) earns nothing
results.append(("2,000-tok prefix cacheable on Haiku 4.5 (min 4096)?", str(2000 >= 4096), "expect False"))

print(f"{'check':<46}{'computed':<16}{'doc/source'}")
print("-" * 92)
for name, got, want in results:
    print(f"{name:<46}{got:<16}{want}")

# Hard assertions so a silent change to the algebra fails loudly.
assert abs(r_star - 4.75) < 0.01, r_star
assert abs(stay30 - 0.6225) < 0.0001, stay30
assert abs(sw5 - 0.4300) < 0.0001, sw5
assert abs(break_even_turns(sonnet, oai_target) - 0.8) < 0.01
assert abs(osc - 0.75) < 0.0001, osc
assert osc > stay30, "oscillation should exceed the cost of never switching"
assert abs(five_min - 2.0025) < 0.0001, five_min
assert abs(one_hour - 0.66) < 0.0001, one_hour
assert not (2000 >= 4096)
print("\nAll assertions passed.")
