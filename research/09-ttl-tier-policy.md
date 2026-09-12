# Thread 9 — TTL tier selection as a routing decision

Research date: 2026-09-12. **Read after `08`. This file answers `08` §8.11 open question 4**
("should the router choose Anthropic's TTL tier?") and **partially corrects `08` §8.7(3)**.

Triggering insight (user, 2026-09-12):

> "That should be the LLM router's choice, whether it has to choose the 5 minute or one hour. If the
> conversation is going on frequently I think 5 minute is enough, because very quickly it will switch
> and the same thing won't be hit multiple times. For a chat interface where the system prompt is
> very big and user messages are very small, even one hour is good... Otherwise most agentic
> scenarios won't require more than 5 minute, because the new message will have its new TTL — if I
> am correct."

**You are correct on the mechanism, and it is confirmed by the primary source.** Two refinements
change the design, and one of them turns out to be better than either option you named.

---

## 9.1 Verdict on your three claims

| Your claim | Verdict | Evidence |
|---|---|---|
| "The new message will have its new TTL" | ✅ **Confirmed by primary docs.** "By default, the cache has a 5-minute lifetime. **The cache is refreshed for no additional cost each time the cached content is used.**" (docs.anthropic.com) | A practitioner quoting the same doc: "if you're sending a prompt every 4:55, then the cache never expires. It only expires if you go 5 full minutes without a single request. **For any active session, whether it lasts 10 minutes or 3 hours, you pay the cache write once.**" |
| "Frequent conversation ⇒ 5 minute is enough" | ✅ **Confirmed, and it is the cheaper option.** With zero expiries, 5m costs `1.25 + (N−1)·0.1` vs 1h's `2.0 + (N−1)·0.1`. 5m wins by exactly **0.75× base input on the prefix**, every time. Verified: at 30 uses / 50k prefix, 5m = **$0.6225**, 1h = **$0.8225**. | `verify_09_ttl_policy.py` |
| "Agentic scenarios won't need more than 5 minute" | ⚠️ **Mostly right — two exceptions in §9.4.** Long *generations* eat the TTL window, and the Batch API specifically wants 1h. | docs.anthropic.com FAQ |
| "Big system prompt + small user messages ⇒ 1 hour is good" | ⚠️ **Right conclusion, wrong reason — and the fix is better than either choice.** Prefix *size* does not decide the tier; the **gap** does. See §9.3. | algebra in §9.2 |

---

## 9.2 The decision rule, derived

Let `E` = expected number of cache expiries during the session, `N` = total uses, `S` = prefix tokens.

```
Cost(5m)  = S·c·[ 1.25·(1 + E) + (N − 1 − E)·0.1 ]      # each expiry replaces a read with a write
Cost(1h)  = S·c·[ 2.00        + (N − 1)·0.1 ]            # does not expire inside the hour

Cost(5m) − Cost(1h) = S·c·[ 1.15·E − 0.75 ]
```

```
                     0.75
  E* = ────────────────────  =  0.652 expiries
              1.15
```

> **The rule: if you expect even ONE cache expiry during the session, the 1-hour tier is cheaper.**
> If you expect none, 5-minute is cheaper by `0.75·S·c`.

Note `S` **cancels out**. That is the correction to your chat reasoning:

> **Prefix size does not choose the tier — the expected idle gap does.** A 100k system prompt and a
> 2k system prompt have the *same* break-even. Size only scales the dollar amount at stake, which
> matters for *whether to bother deciding*, not for *which way to decide*.

What actually predicts `E`:

| Signal | Effect on `E` |
|---|---|
| Expected inter-arrival gap `< TTL` | `E = 0` ⇒ **5m** |
| Expected gap `> 5 min`, session `< 1 h` | `E ≥ 1` ⇒ **1h** |
| Session duration / gap length | `E ≈ floor(duration / gap)` when gap > TTL |
| Long generation times | raises `E` — see §9.4(1) |
| Batch / concurrent requests | effectively `E = ∞` for 5m ⇒ **1h** — see §9.4(2) |

Two supporting facts, verified:

- **The 1h-vs-5m write ratio is 1.6×, not 2×.** "The '2×' claim compares 1h write to *uncached base
  input* (2.0× vs 1.0×). The ratio of 1h write to 5m write is **1.6×** (2.0/1.25). Both are correct
  depending on the comparison frame." (brandonwie.dev, 2026-05-31). Any cost model that treats the
  upgrade as "2× more expensive" overstates it by 60%.
- **Per-tier break-even vs no caching at all:** 5m pays for itself on the **2nd** use, 1h on the
  **3rd**. Verified by direct computation; matches dev.to/thegdsks (2026-04-29).

---

## 9.3 The better answer: mixed TTL per breakpoint

You framed it as a binary — 5m *or* 1h, chosen per conversation. **Anthropic lets you do both in one
request**, and that is strictly better than either.

> "You can use both 1-hour and 5-minute cache controls in the same request, but with an important
> constraint: **cache entries with longer TTL must appear before shorter TTLs**." (docs.anthropic.com)
>
> "When mixing TTLs, the API determines three billing locations: **A** (highest cache hit), **B**
> (highest 1h breakpoint after A), **C** (last cache breakpoint). Charged: **read for A, 1h write for
> (B−A), 5m write for (C−B)**."

The policy that falls out — and it is exactly the shape you were reaching for, just applied per
*segment* rather than per *session*:

```
┌─────────────────────────────┬──────────┬──────────────────────────────────────┐
│ Segment                     │ TTL      │ Why                                  │
├─────────────────────────────┼──────────┼──────────────────────────────────────┤
│ tools + system prompt       │ 1h       │ Stable, large, reused every turn AND │
│ (+ CLAUDE.md-style context) │          │ potentially across sessions. Survives│
│                             │          │ a human thinking for 20 minutes.     │
├─────────────────────────────┼──────────┼──────────────────────────────────────┤
│ RAG documents (if reused)   │ 1h       │ Same reuse profile as system prompt. │
├─────────────────────────────┼──────────┼──────────────────────────────────────┤
│ conversation history        │ 5m       │ Append-only, read once then grown.   │
│                             │          │ Paying 2× to cache text that changes │
│                             │          │ every turn is pure waste.            │
└─────────────────────────────┴──────────┴──────────────────────────────────────┘
```

An independent implementation proposal reaches the identical conclusion — spring-projects/spring-ai
issue #4325 (2025-09-05) asks for "per-message-type TTL (e.g., **SYSTEM & TOOLS at 1h; USER &
ASSISTANT at 5m**)". Nobody has shipped it as a *router* decision.

**Verified economics** (30k system segment at 1h, 20k history at 5m, one long gap mid-session):

```
after gap, mixed : sys read (30k × 0.1) + history rewrite (20k × 1.25) = $0.0840
after gap, all-5m: whole prefix rewrite (50k × 1.25)                   = $0.1875
upfront 1h penalty on the system segment (30k × 0.75)                 = $0.0675
gaps to pay back                                                     = 0.652   → ONE gap justifies it
```

So: **mixed TTL pays for itself the first time the user pauses.** That is your chat case, handled
correctly — and it does not force you to pay 2× on the conversation history.

**Ordering is a hard constraint, so the router must emit breakpoints in prefix order:** longer TTL
first. A real proxy got this wrong and silently downgraded 1h blocks — router-for-me/CLIProxyAPI
#3398: injected default `{type: ephemeral}` blocks (5m) caused `normalizeCacheControlTTL()` to strip
later `ttl: "1h"` values, because Anthropic evaluates in **tools → system → messages** order. Any
implementation needs an ordering-normalization pass *after* breakpoint injection, and a max-4
breakpoint budget.

---

## 9.4 Two exceptions to "agents never need 1h"

### (1) Generation time is charged against the TTL — this is the one that bites agents

From the primary docs, and easy to miss:

> "The lifetime is measured **from the start of the request** that writes or reads the cache entry,
> **not from the end of its response**. Time spent generating a response counts against the lifetime:
> if a response takes 4 minutes to stream, a follow-up request that reuses the same cached prefix
> must start **within about 1 minute** of that response completing."

So the effective window is `300s − generation_time`:

| Generation time | Effective follow-up window |
|---|---|
| 2 s | 298 s |
| 30 s | 270 s |
| 120 s | 180 s |
| 240 s | **60 s** |

**This inverts your expectation for one important agent class.** An agent step that streams a long
output (large code generation, extended reasoning, long tool arguments) followed by a slow tool call
can blow a 5-minute TTL *even though the gap between requests is short*. The signal to model is not
`expected_gap` alone but **`expected_gap + expected_generation_time`**.

Design consequence: `expected_generation_time` belongs in `RoutingRequest` alongside
`expected_gap_s` (`08` §8.8), and the TTL decision should use their sum.

### (2) The Batch API actively wants 1h

> "Because asynchronous batch requests can be processed **concurrently and in any order**, cache hits
> are provided on a **best-effort** basis. **The 1-hour cache can help improve your cache hits.**"
> (docs.anthropic.com) — and OpenRouter: "To get reliable cache hits, use `ttl: "1h"` breakpoints on a
> shared prefix and **reuse that prefix across successive batches** (or warm the cache with a sync
> request first)."

This **partially corrects `08` §8.7(3)**, which said batch should be *exempt* from affinity logic.
The corrected rule: batch is exempt from **session affinity** (there is no session), but it is *not*
exempt from **cache policy** — a batch job with a large shared instruction prefix should pin that
prefix at 1h and reuse it across batches, or pre-warm it with one sync request.

---

## 9.5 A third lever you did not mention: keepalive pinging

There are two ways to survive a gap — pay 2× once (1h), or keep the 5m entry alive with periodic
reads at 0.1×. The docs describe the mechanism: "For the default 5-minute cache, **send a new
pre-warm request at least every 5 minutes** to keep the cache warm."

```
keepalive cost over a G-minute gap = (G / 5) × 0.10 × S·c
1h upgrade cost                    =        0.75 × S·c

break-even:  G* = 0.75 / 0.10 × 5 = 37.5 minutes
```

| Gap | Keepalive cost (× base) | vs 1h upgrade (0.75) | Cheaper |
|---|---|---|---|
| 10 min | 0.20 | 0.75 | **keepalive** |
| 20 min | 0.40 | 0.75 | **keepalive** |
| 37.5 min | 0.75 | 0.75 | tie |
| 60 min | 1.20 | 0.75 | **1h** |

**So for gaps under ~37 minutes, pinging beats the upgrade.** Caveats that probably decide it in
practice rather than the arithmetic: a ping is a real API call (rate limits, and `max_tokens: 0`
pre-warm still bills the cache read), it needs a background timer per live session, and it only works
if you know the session will resume. **Recommended default: use 1h; offer keepalive as an opt-in for
deployments with strict cost ceilings and predictable resume patterns.** Do not ship the timer by
default.

---

## 9.6 ⚠ A collision with the step-escalation design

The primary docs list cache invalidators that are **not** in `07` §7.1's four rules:

> "Verify that **`tool_choice`, image usage, the `thinking` configuration, and `output_config.effort`**
> remain consistent between calls." (docs.anthropic.com, troubleshooting)

`08` §8.6's escalation policy escalates on tool failure or verification drift. If escalation is
implemented by **raising `thinking.effort`** on the same model — the cheapest-looking option, since it
avoids a model switch — **it invalidates the cache just as thoroughly as switching models does.**

> **Design rule F — escalation must not silently change cache-invalidating request fields.**
> `thinking.effort`, `tool_choice`, image usage, and `output_config.effort` are part of the cache key.
> Changing them mid-session costs a full prefix rewrite, so they must be scored with the same
> `switch_cost` term as a model change — not treated as free knobs. If you are going to pay a rewrite
> anyway, prefer escalating to a *stronger model* (which at least buys capability) over raising
> effort on the same one.

Corroborating measurement from the same failure family (brandonwie.dev, 2026-05-31): "Add/remove MCP
server mid-session → tool defs change → full invalidation. **Claude Code's design locks the tool list
at startup to prevent this.**" And: "**Pin model per session** — don't switch Opus ↔ Sonnet inside one
task." Independent implementations converge on `07` Rule A from the operational side.

---

## 9.7 What this adds to the price registry and policy

```python
@dataclass(frozen=True)
class CacheSemantics:                 # per MODEL, not per provider (08 §8.1)
    read_mult: float                  # 0.10, or 0.025 on Claude Fable 5.1 / Mythos 5.1
    write_mult_5m: float              # 1.25
    write_mult_1h: float              # 2.00
    min_cacheable_tokens: int         # 1024 / 2048 / 4096 — below this, no cache exists
    ttl_options: tuple[str, ...]      # ("5m",) | ("5m", "1h") | ("30m",) ...
    ttl_refreshes_on_hit: bool        # Anthropic True; OpenAI "5-10 min of inactivity"
    supports_mixed_ttl: bool          # Anthropic True (longer TTL must precede shorter)
    max_breakpoints: int              # 4
    invalidating_fields: frozenset[str]   # thinking, tool_choice, images, output_config.effort
    explicit_cache_key_param: str | None  # OpenAI "prompt_cache_key"; Anthropic None
```

TTL policy function (pure, testable, no I/O):

```python
def choose_ttl(seg: Segment, ctx: SessionContext, sem: CacheSemantics) -> str:
    if "1h" not in sem.ttl_options:
        return "5m"                                    # don't model what the provider lacks
    if seg.kind in (SegmentKind.SYSTEM, SegmentKind.TOOLS, SegmentKind.RETRIEVED):
        expected_exposures = ctx.expected_gap_s + ctx.expected_generation_s
        return "1h" if expected_exposures > 300 else "5m"
    return "5m"                                        # conversation history: always 5m
```

Then a **breakpoint emitter** that (a) enforces longer-TTL-first ordering, (b) respects
`max_breakpoints`, (c) skips segments under `min_cacheable_tokens`, and (d) runs *after* any
breakpoint injection — the exact ordering that broke CLIProxyAPI (#3398).

---

## 9.8 Provider portability — TTLs do not translate

> "**TTLs are not translated** — a `cache_control` ttl is **dropped toward OpenAI**, and the
> request-level `prompt_cache_options` stays OpenAI-only." (openrouter.ai/docs, 2026-07-12)

OpenAI's shape is different again: `prompt_cache_breakpoint` on a block, plus request-level
`prompt_cache_options` with `mode: "explicit"` and a `ttl` such as `"30m"` — **a third TTL value that
has no Anthropic equivalent.** So `ttl_options` must be per-model, and a routing decision that picks
a TTL is **backend-specific**: it can move a session between two Anthropic backends freely, but not
between Anthropic and OpenAI without re-deciding the cache policy.

Also corrected: `08` §8.1 recorded (via Helicone) that "Vertex AI and Bedrock only support 5-minute
caching." **That is now outdated.** AWS's own docs list 1h TTL support for Claude Fable 5, Opus 5,
Opus 4.8/4.7/4.6/4.5, Sonnet 5, Sonnet 4.6/4.5 and Haiku 4.5, via `"ttl": "1h"` on `cachePoint`
(Converse) or `cache_control` (InvokeModel). docs.anthropic.com confirms 1h on "the Claude API,
Amazon Bedrock, Amazon Bedrock (Opus 4.6 and earlier), Claude Platform on AWS, Google Cloud, and
Microsoft Foundry."

---

## 9.9 Measured evidence

**anthropics/claude-code issue #32671** (2026-03-10) — proxy logs from a 100+ request Bedrock
session where Claude Code hardcoded 5m and users hit >5 min gaps:

| Configuration | Tokens | Cost |
|---|---|---|
| 5m TTL: ~20 cache rewrites × 60k @ $9.375/M | 1.2M | **$11.25** |
| 1h TTL: 99 reads × 60k @ $0.9375/M | 5.9M | **$5.57** |
| Issue's claimed reduction | | **~50%** |

> ⚠️ **The issue's $5.57 counts the 99 reads only and omits the 1h write entirely**; its implied base
> price is also internally inconsistent (reads at $0.9375/M = 0.1× of $9.375/M, but $9.375/M is itself
> the 1.25× write rate). Adding the 1h write at 1.6× the 5m rate gives **$6.47**, i.e. a **42%**
> reduction rather than 50%. Both figures are in `verify_09_ttl_policy.py`; the *direction and rough
> magnitude* are solid, the exact number is not. **Cite ~40–50%, not 50%.**

Also relevant: Claude Code's TTL was **silently reduced from 1h to 5m around early March 2026**
(issue #46829), which is what produced this cost spike for users. Claude Code 2.1.108+ added
`ENABLE_PROMPT_CACHING_1H=1` (plus `FORCE_PROMPT_CACHING_5M`) across API, Bedrock, Vertex and Foundry.
**Lesson for your package: TTL default is a cost-critical setting that users will not notice changing.
Make it explicit, log it, and emit it in `explain()`.**

---

## 9.10 Priority changes (delta on `08` §8.9)

Promoted to **P0**:

| Item | Why |
|---|---|
| `CacheSemantics` per **model** with `ttl_options`, `write_mult_5m/1h`, `min_cacheable_tokens`, `invalidating_fields` | the TTL decision is unimplementable without it |
| Mixed-TTL breakpoint emitter with **longer-TTL-first ordering** and max-4 budget | §9.3 — the single highest-value cache lever, and the ordering bug is easy to hit silently |
| `choose_ttl()` using `expected_gap_s + expected_generation_s` | §9.4(1) — generation time is charged against the window |
| **Rule F**: cache-invalidating request fields scored with `switch_cost` | §9.6 — otherwise step-escalation silently destroys the cache |

Promoted to **P1**:

| Item | Why |
|---|---|
| Batch: shared prefix pinned at 1h + cross-batch reuse, or sync pre-warm | §9.4(2) — corrects `08` §8.7(3) |
| Per-segment TTL policy by `SegmentKind` | §9.3 |
| Keepalive pinger (opt-in, off by default) | §9.5 — cheaper below ~37.5 min gaps, but operationally heavy |
| TTL choice surfaced in `explain()` and in metrics | §9.9 lesson |

**Corrected, not new:** `08` §8.7(3) "batch is exempt from cache logic" → exempt from *session
affinity* only.

---

## 9.11 Verification additions (delta on `08` §8.10)

7. **Cost by TTL decision.** Group `cache_creation_input_tokens` by `(segment_kind, ttl_chosen)`.
   A 1h segment with a near-zero expiry rate is money thrown away; a 5m segment with frequent
   rewrites is the §9.9 failure mode.
8. **Expiry counter.** Count cold writes that follow an idle period. This is `E` measured rather than
   predicted — feed it back into `choose_ttl` so the prior converges.
9. **Ordering-invariant test.** Property test: in every emitted request, no 1h breakpoint appears
   after a 5m one, and breakpoint count ≤ 4. This is exactly the bug CLIProxyAPI shipped.
10. **Invalidator-drift test.** Assert that within a pinned session, `thinking`, `tool_choice`,
    image usage and `output_config.effort` never change — or if they do, that the change was scored
    with `switch_cost`.

---

## 9.12 Open questions

1. **`expected_generation_s` is hard to predict.** Best available proxy: the model's recent p50 output
   length for this `workload_class`, tracked online. **My assumption, not measured.**
2. **Does the 1h TTL also refresh on hit?** brandonwie.dev says "same sliding-window behavior, longer
   dead-clock" — plausible and consistent with the docs' general phrasing, but **I did not find an
   explicit primary statement.** If it does *not* refresh, a session longer than one hour needs
   periodic 1h rewrites and `E*` changes. **Verify before relying on it.**
3. **Gemini's explicit cache bills token-hours of storage**, a different cost shape entirely
   (not a write multiplier). `choose_ttl` as written does not model it. Out of scope for P0.
4. **Keepalive under rate limits.** A ping per live session per 5 minutes may be infeasible at scale
   — needs a bound on concurrent pinned sessions before it is safe to offer.
5. **Does a `max_tokens: 0` pre-warm actually refresh the TTL, and is it billed as a read?** Widely
   assumed (aioutlooks lists it as the pre-warm technique), **not confirmed against primary docs.**
   This is the load-bearing assumption behind §9.5.

---

## Cross-references

- `07` §7.1 Rules A–E (**Rule F added in §9.6**; invalidator list extended)
- `08` §8.7(2) TTL vs inter-arrival (**extended with generation time, §9.4(1)**)
- `08` §8.7(3) batch exemption (**partially corrected, §9.4(2)**)
- `08` §8.8 `RoutingRequest` (**`expected_generation_s` added, §9.4(1)**)
- `08` §8.11 Q4 (**answered: yes, and per-breakpoint, §9.3**)
- `verify_09_ttl_policy.py` — 29 computed checks, 13 assertions, all pass
