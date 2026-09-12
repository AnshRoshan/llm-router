"""Verify the TTL-tier algebra for 09-ttl-tier-policy.md.

Run: python3 verify_09_ttl_policy.py
Pure functions, no I/O. Multipliers verified 2026-09-12 against
docs.anthropic.com/en/docs/build-with-claude/prompt-caching (primary).
"""
from dataclasses import dataclass

BASE = 3.00 / 1e6          # Sonnet-class $/input token
W5, W1H, R = 1.25, 2.00, 0.10


@dataclass(frozen=True)
class Tier:
    name: str
    write: float
    def cost(self, S: int, uses: int) -> float:
        """One write, then `uses` total requests served from that entry."""
        assert uses >= 1
        return S * BASE * (self.write + (uses - 1) * R)

t5, t1h = Tier("5m", W5), Tier("1h", W1H)
S = 50_000
out = []


def show(label, got, want):
    out.append((label, got, want))


# --- 1. break-even per tier, vs paying uncached input every time ---
def breakeven_uses(tier: Tier) -> int:
    n = 1
    while tier.cost(S, n) >= S * BASE * n:
        n += 1
        assert n < 100
    return n


show("5m break-even uses", breakeven_uses(t5), "2 (source: thegdsks 'pays for itself after the second')")
show("1h break-even uses", breakeven_uses(t1h), "3 (source: thegdsks 'after the third')")
show("1h-vs-5m write ratio", f"{W1H / W5:.2f}x", "1.60x (brandonwie: NOT 2x; 2x is vs uncached)")

# --- 2. when does the 1h upgrade pay for itself? ---
# Extra cost of 1h over 5m on one write = (2.00-1.25) = 0.75x base.
# Cost of a 5m expiry-rewrite we would have paid = 1.25x base.
def tier_cost_with_expiries(S, tier, uses, expiries):
    """`expiries` cold rewrites at the tier's own write multiplier.

    Each expiry replaces one read with one write, so reads = uses - 1 - expiries.
    """
    return S * BASE * (tier.write * (1 + expiries) + (uses - 1 - expiries) * R)


# The comparison that matters: 5m WITH expiries vs 1h WITHOUT them.
# A 1h entry does not expire inside the hour, so its expiry count is 0 -- my first
# draft applied the same expiry count to both tiers, which made 1h look worse
# everywhere and contradicted the algebra. Fixed here.
uses = 30
e5_1 = tier_cost_with_expiries(S, t5, uses, 1)
e1h_1 = tier_cost_with_expiries(S, t1h, uses, 0)
show("1 expiry on 5m vs 0 on 1h: 5m cost", f"${e5_1:.4f}", "")
show("1 expiry on 5m vs 0 on 1h: 1h cost", f"${e1h_1:.4f}", "")
show("1h cheaper when there is 1 expiry?", str(e1h_1 < e5_1), "True")

# Exact threshold. Cost5m(E) - Cost1h(0) = S*c*[1.15E - 0.75], so 1h wins for E > 0.75/1.15.
E_star = (W1H - W5) / (W5 - R)
show("expiry threshold E* for the 1h upgrade", f"{E_star:.3f}",
     "0.652 (not 0.60 -- an expiry also removes a read)")
show("=> any single expected expiry justifies 1h", str(E_star < 1.0), "True")

e5_0 = tier_cost_with_expiries(S, t5, uses, 0)
e1h_0 = tier_cost_with_expiries(S, t1h, uses, 0)
show("0 expiries: 5m cheaper?", str(e5_0 < e1h_0), "True  <- the user's 'frequent chat' case")

# --- 3. keepalive ping vs the 1h upgrade ---
# A ping every 5 min reads the prefix at 0.1x. Over a G-minute gap that is G/5 pings.
# Break-even: (G/5) * 0.10 == (2.00 - 1.25)  ->  G = 37.5 min
G_star_keepalive = (W1H - W5) / R * 5
show("gap where keepalive == 1h upgrade", f"{G_star_keepalive:.1f} min", "37.5 min (derived, no source)")
for gap in (10, 20, 37.5, 60):
    keep = (gap / 5) * R
    show(f"  gap={gap}min: keepalive cost (x base)", f"{keep:.3f}", f"1h upgrade = {W1H - W5:.2f}")

# --- 4. generation time eats the TTL window (Anthropic docs, primary) ---
# "lifetime is measured from the START of the request... if a response takes 4 minutes
#  to stream, a follow-up must start within about 1 minute of that response completing"
TTL_S = 300
for gen in (2, 30, 120, 240):
    show(f"effective follow-up window, gen={gen}s", f"{TTL_S - gen}s", f"{TTL_S - gen}s")

# --- 5. measured Bedrock evidence (anthropics/claude-code#32671, proxy logs) ---
# NOTE: the issue's "$5.57" counts the 99 READS only and omits the 1h write entirely;
# its implied base price is also inconsistent (reads at $0.9375/M = 0.1x of $9.375/M,
# but $9.375/M is itself the 1.25x write rate). So the exact figure is not reproducible.
# What IS reproducible is the direction and rough magnitude. Flagged, not asserted.
TOK, OPUS_W, OPUS_R = 60_000, 9.375 / 1e6, 0.9375 / 1e6
five_m = 20 * TOK * OPUS_W
one_h_reads_only = 99 * TOK * OPUS_R
one_h_with_write = TOK * OPUS_W * (W1H / W5) + one_h_reads_only
show("Bedrock 5m: 20 rewrites x 60k @ $9.375/M", f"${five_m:.2f}", "issue says $11.25  MATCHES")
show("Bedrock 1h: 99 reads only", f"${one_h_reads_only:.2f}", "issue says $5.57  MATCHES (reads only)")
show("Bedrock 1h incl. the 1h write @1.6x", f"${one_h_with_write:.2f}", "issue omits this")
show("reduction, reads-only basis", f"{100 * (1 - one_h_reads_only / five_m):.0f}%", "issue says ~50%")
show("reduction, incl. write", f"{100 * (1 - one_h_with_write / five_m):.0f}%", "42% -- the honest number")

# --- 6. mixed-TTL billing partition, modelled ACROSS A GAP (where it actually pays) ---
# Anthropic docs: charging is read for A, 1h write for (B-A), 5m write for (C-B).
# A snapshot with no gap always favours all-5m on the write, so the honest test is
# "what happens after a long idle gap?" -- the 1h segment survives, the 5m one does not.
SYS, HIST = 30_000, 20_000          # system/tools segment at 1h, growing history at 5m
after_gap_mixed = SYS * BASE * R + HIST * BASE * W5          # sys read, history rewritten
after_gap_all5m = (SYS + HIST) * BASE * W5                   # whole prefix rewritten
upfront_penalty = SYS * BASE * (W1H - W5)                    # extra cost of writing sys at 1h
show("after a gap: mixed (sys@1h, hist@5m)", f"${after_gap_mixed:.4f}", "")
show("after a gap: all-5m", f"${after_gap_all5m:.4f}", "")
show("upfront 1h penalty on the sys segment", f"${upfront_penalty:.4f}", "")
G_star = upfront_penalty / (after_gap_all5m - after_gap_mixed)
show("gaps needed to pay back the mixed policy", f"{G_star:.3f}", "<1 => one gap justifies it")
show("mixed wins after 1 gap?", str(after_gap_mixed + upfront_penalty < after_gap_all5m), "True")

print(f"{'check':<50}{'computed':<14}expected / source")
print("-" * 118)
for label, got, want in out:
    print(f"{label:<50}{got:<14}{want}")

assert breakeven_uses(t5) == 2
assert breakeven_uses(t1h) == 3
assert abs(W1H / W5 - 1.6) < 0.001
assert abs(E_star - 0.75 / 1.15) < 0.001, E_star
assert e1h_1 < e5_1, "1h must win when 5m suffers at least one expiry"
assert e5_0 < e1h_0, "5m must win when there are no expiries"
assert abs(G_star_keepalive - 37.5) < 0.001
assert abs(five_m - 11.25) < 0.01, five_m
assert abs(one_h_reads_only - 5.57) < 0.05, one_h_reads_only
assert one_h_with_write > one_h_reads_only, "the 1h write is a real cost the issue omits"
assert after_gap_mixed + upfront_penalty < after_gap_all5m, "mixed TTL must pay back within one gap"
assert G_star < 1.0, G_star
assert TTL_S - 240 == 60
print("\nAll assertions passed.")
