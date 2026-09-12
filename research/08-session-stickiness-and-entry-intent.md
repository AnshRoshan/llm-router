# Thread 8 — Session stickiness and the two entry paths

Research date: 2026-09-12. **Read after `07`. This file closes open questions Q1 and Q2 of `07` §7.6
and CORRECTS Rule B of `07` §7.1.**

Triggering insight (user, 2026-09-12):

> "For conversational work you have to keep using the same LLM, otherwise the cache drops and it
> starts to cost more instead of costing less. So you have to understand the intent from point A —
> or if the user chooses some agent, it can go directly through the LLM router from there too."

Two design consequences fall out of this, and both are missing from the literature:

1. **Stickiness is a session-level constraint, so the *entry point* is the highest-leverage routing
   decision the router makes.** A wrong turn-1 decision is paid for on every subsequent turn.
2. **There are two entry paths, and they are not symmetric.** One is *declared* (the user picked an
   agent / passed a tag) and one is *inferred* (the router reads intent from the first request). They
   need different code, different budgets, and different failure modes.

---

## 8.1 Verification pass — the cache numbers, checked today

I re-verified the cache pricing rather than reusing the search snippets behind `07`, because the
switch-cost algebra in §8.3 depends on one specific multiplier. **Verified 2026-09-12** across five
independent sources:

| Fact | Value | Sources (all 2026, dates in `SOURCES.md` §G) |
|---|---|---|
| Anthropic cache **read** | **0.1×** base input (90% off); **0.025×** on a subset of newest models | intuitionlabs (2026-09-05), technspire (2026-09-06), ofox (2026-06-10) |
| Anthropic cache **write**, 5-min TTL | **1.25×** base input | intuitionlabs, technspire, buzzai (2026-05-24), dev.to/rikuq (2026-06-14) |
| Anthropic cache **write**, 1-hour TTL | **2.0×** base input | intuitionlabs, technspire, ofox |
| Anthropic breakpoints | up to **4** explicit markers per request | buzzai, ofox |
| Anthropic cache key | byte-exact prefix **+ model + account**; model change ⇒ guaranteed miss | buzzai FAQ, ofox |
| **OpenAI cache write premium** | **NONE** (pre-GPT-5.6). First call billed at standard input rate | ofox, intuitionlabs, gingerlabs, aioutlooks |
| OpenAI GPT-5.6+ | manual breakpoints **and** a 1.25× write / 0.1× read split, mirroring Anthropic | intuitionlabs (2026-09-05) |
| OpenAI `prompt_cache_key` | optional caller-supplied key **"for routing optimization"** | ofox spec table |
| Minimum cacheable prefix | OpenAI 1,024; Anthropic 1,024 (Sonnet/Opus-class) to 4,096 (Haiku 4.5 / older Claude); Gemini 4,096 (2.5/3) to 32,768 (2.0) | intuitionlabs, ofox, ai-tldr, sgryt |
| Default TTL | 5 min both providers, refreshed on hit; OpenAI 5–10 min of *inactivity*, up to 24 h on GPT-5.4/5.5 | buzzai, ofox, ai-tldr |
| Self-hosted (vLLM) write premium | **none** — APC is a compute optimization, not a billed tier | inference from `07` §7.1 sources; **no explicit source, marked [inferred]** |

### ⚠️ Correction to `07` — the "OpenAI is 50%" shorthand is stale

`07` §7.1 quotes gmicloud.ai: "OpenAI GPT-4o $2.50→$1.25/M (50%)". That figure is **correct for
gpt-4o / gpt-4.1 and wrong for current models**. Per ofox (2026-06-10) and dev.to/rikuq (2026-06-10),
**GPT-5.4 and GPT-5.5 are at $0.50/M against $5/M — a 90% discount, matching Anthropic.** ofox:
> "The '50% vs 90%' framing in older comparison posts and tutorials is outdated."

**Design consequence:** `discount_pct` in the price registry must be **per-model, not per-provider**.
Any code that branches on `provider == "openai"` to pick a cache discount will be wrong for half the
OpenAI catalogue.

### ⚠️ Correction to Rule B of `07` — stated as an absolute, it is false

Rule B says: *"Never switch model mid-session for cost reasons alone."* That is the right **instinct**
and the wrong **rule**. `07` §7.1 even contains the counter-evidence: **OpenAI charges no write
premium at all**, so switching onto an OpenAI-backed model costs *nothing* in cache-write terms.
The correct rule is quantitative and is derived in §8.3. Short version:

- **A one-way migration usually pays back in 1–2 turns** on Anthropic pricing, and immediately on
  OpenAI pre-5.6 pricing.
- **What actually destroys you is oscillation** (A→B→A), TTL expiry, and switching on a prefix you
  were about to abandon anyway.

Rule B should read: *"Never **oscillate**. Evaluate a one-way migration against the amortization
test in `08` §8.3; forbid migration when the session has fewer turns remaining than the break-even."*

---

## 8.2 The asymmetry that drives everything

| | Per-step routing (agent inner loop) | Turn-1 routing (session entry) |
|---|---|---|
| How often the decision cost is paid | **every step** — hundreds per session | **once** |
| Measured cost of an LLM-based decision | 50–200 ms per step, "compounds across hundreds of agent calls" (augmentcode, 2026-06-18) | 50–200 ms **once**, amortized over the whole session |
| Measured cost of a *wrong* decision | one bad step, correctable next step | **every remaining turn**, because the session is pinned |
| Verdict from the literature | **don't** (arXiv:2608.14641; arXiv:2606.22902) | **do** — this is the one place a strong classifier is affordable |

**This is the single cleanest design conclusion in the whole dossier.** The evidence against LLM-based
routing is evidence against LLM-based routing *per step*. It says nothing about turn 1, where the
decision cost is divided by the session length and the decision value is multiplied by it.

Corollary: **budget the classifier by session length, not by request.**

```
classifier_budget(session) = α · estimated_session_cost · P(sticky)
                           ≈ α · (turns · avg_turn_cost)
```

With `turns = 10`, `avg_turn_cost = $0.05`, `α = 1%` ⇒ **$0.005 for the turn-1 decision**. That is
~5,000 tokens of a small model, or ~1,000 tokens of a frontier model. An LLM classifier is comfortably
inside budget at turn 1 — and wildly outside it at step 40.

---

## 8.3 The switch-cost algebra (the missing `switch_cost` term, made concrete)

Notation, per session with `S` tokens of shared prefix and `T` turns:

| Symbol | Meaning |
|---|---|
| `c`, `c'` | base **input** $/token, current model / target model |
| `w'` | cache-**write** multiplier of the target model's provider + TTL tier |
| `r` | cache-**read** multiplier (0.1 Anthropic, 0.1 or 0.5 OpenAI by model) |
| `R` | turns remaining after the switch |

Cost of the cached portion of the prefix:

```
stay:          S·c·( w + (T−1)·r )
switch at k:   S·c·( w + (k−1)·r ) + S·c'·( w' + R·r )        R = T − k
```

One-time cost of switching = `S·(c'·w' − c·r)`; per-turn saving = `(c − c')·r·S`. Therefore:

```
                    c'·w' − c·r
  R* = ─────────────────────────────────────         ← turns needed to pay back a migration
              (c − c')·r
```

### Worked: Sonnet → Haiku, 50k-token prefix, Anthropic pricing

`c = $3/M`, `c' = $1/M`, `w' = 1.25`, `r = 0.1` (values verified in §8.1; token counts are my
illustrative choice, not measured):

```
R* = (1·1.25 − 3·0.1) / ((3 − 1)·0.1) = 0.95 / 0.2 = 4.75 turns
```

Cross-check by direct summation over `T = 30` turns (switch at `k = 5` ⇒ 5 turns on Sonnet, 25 on
Haiku; verified by `verify_08_switch_cost.py`):

```
stay    : 50,000/1e6 · 3.00 · (1.25 + 29·0.1) = 0.15 · 4.15 = $0.6225
switch@5: 0.15·(1.25 + 4·0.1) + 0.05·(1.25 + 24·0.1)
        = 0.15·1.65 + 0.05·3.65 = $0.2475 + $0.1825 = $0.4300
saving  : 30.9% of the cached-input line
```

> **Correction caught by running the algebra.** My first draft of this worked example wrote
> `0.05·(1.25 + 25·0.1) = $0.4350` — that prices **26** turns on the target model while claiming 25,
> an off-by-one that inflates the saving. The verified figures are **$0.4300 / 30.9%**. The
> convention is now asserted in the script (`stay_turns + remaining == total_turns`) so it cannot
> drift back.

(The two disagree slightly because `R*` prices the migration against a *read* on the old model, not
against an uncached write; the direct sum is the trustworthy one. Ship the direct sum.)

### Now the same switch onto an OpenAI-backed target

`w' = 1.0` (no write premium, pre-GPT-5.6), `c' = $0.5/M` hypothetical:

```
R* = (0.5·1.0 − 3·0.1) / ((3 − 0.5)·0.1) = 0.2 / 0.25 = 0.8 turns
```

**The migration pays back inside a single turn.** Rule B's absolute would have forbidden the single
cheapest move available.

### And the case that genuinely loses money — oscillation

A→B→A→B with a 50k prefix at `w = 1.25`: every return leg is a cold write, `S·c·1.25 = $0.1875`
per leg on Sonnet. **Four legs = $0.75**, which exceeds the *entire* 30-turn cost of just staying on
Sonnet ($0.6225). Oscillation is not a small inefficiency; on long-prefix sessions it is more
expensive than never routing at all. This is the real content of Rule B, and it is why the router
needs an explicit **no-oscillation lock**, not just a threshold.

### Three second-order effects that must be in the model

1. **Output tokens are never cached.** `switch_cost` covers only the input prefix. If the cost saving
   comes from a cheaper *output* rate, it accrues every turn regardless of cache state and should be
   scored separately — it is not subject to `R*`.
2. **TTL expiry silently zeroes the "stay" term.** If the gap between turns exceeds the TTL, the next
   turn is a fresh write at `w`, not a read at `r`. For a **chat** workload with human typing, a
   7-minute gap on a 5-minute TTL means every turn is a write. See §8.7(2).
3. **A sliding conversation window defeats caching before routing does.** ofox names it as one of the
   three canonical cache-miss patterns: "a sliding conversation window that rotates the oldest
   message every turn." If the caller trims history, `affinity_bonus` must decay with history
   rotation, not with turn count.

---

## 8.4 The two entry paths

```
                          ┌──────────────────────────────────────────┐
   request ──────────────►│ PATH A: DECLARED                         │
   (workload / agent      │  workload_class present, or             │  cost: ~0 ms
    tag supplied)         │  prompt_fingerprint matches registry    │  confidence: 1.0
                          └───────────────┬──────────────────────────┘
                                          │ miss
                          ┌───────────────▼──────────────────────────┐
                          │ PATH B: INFERRED                         │
                          │  B1 structural rules   (~0 ms, high conf)│
                          │  B2 prompt fingerprint (~0 ms, exact)    │
                          │  B3 classifier model   (only if B1/B2    │
                          │     abstain; budget = α·session_cost)    │
                          │  B4 workload default   (log as 'default')│
                          └───────────────┬──────────────────────────┘
                                          ▼
                              SessionPolicy {class, source, conf,
                                             pinned_model, expires_at}
                                          ▼
                     turn ≥ 2 ⇒ sticky lookup, NO re-classification
```

### Path A — declared intent ("the user chose an agent")

This is the cheap path and it must be first. Three ways intent arrives declared:

1. **Explicit tag** — `router.complete(..., workload="agent")` or a `X-Workload-Class` header.
2. **Named agent / profile** — the caller instantiates a configured agent
   (`router.agent("support-triage")`), whose profile carries `workload_class`, allowed models,
   budget, and stickiness policy. *This is the "user chooses an agent" case, and it should be the
   recommended integration path in the README.*
3. **Prompt fingerprint hit** — see B2; it makes Path A apply even when the caller declared nothing.

**Design rule: Path A must skip the classifier entirely.** Not "run it and ignore the result" — the
classifier must not be constructed, because the latency budget and the dependency footprint differ.

**Native provider hook (verified, ofox 2026-06-10):** OpenAI exposes **`prompt_cache_key`**, described
as an optional routing key "for routing optimization". When Path A supplies a session/agent id, the
router should pass it straight through as `prompt_cache_key` on OpenAI backends — the provider then
does affinity routing *for you*, inside its own fleet. Anthropic has no equivalent (its cache is
scoped to org + model), so affinity there is purely your problem. This is a real backend-capability
difference and belongs in `BackendSpec.capabilities`.

### Path B — inferred intent ("understand the intent from point A")

**B1 — Structural rules. Zero cost, and they resolve most production traffic.**
These features are free because they are already in the request object; no model call, no tokenizer
round-trip beyond what you need anyway:

| Signal in the request | Inferred class | Confidence |
|---|---|---|
| `tools` present **and** prior `tool_result` in history | `agent` (mid-session) | high — and already sticky |
| `tools` present, no tool results yet | `agent` (first step) | high |
| ≥2 prior turns, no tools | `chat` | high |
| single turn, retrieved chunks in the user block | `rag` | high |
| batch endpoint / array of independent requests | `batch` | certain |
| `image_url` or image content blocks | `vision` | certain |
| code fence or ≥N% code tokens | `code` | medium |
| single short turn, nothing else | ambiguous → B2/B3 | — |

**B2 — Prompt-fingerprint registry. The trick that makes Path B nearly free.**
Hash the **system prompt** (and the tool-schema block) and look it up. Most production traffic comes
from a *small number of distinct application prompts* — a support bot has one system prompt, a code
agent has one, a RAG template has one. So:

```python
fp = blake2b(system_prompt_bytes + tools_schema_bytes, digest_size=16).hexdigest()
hit = fingerprint_registry.get(fp)      # fp → (workload_class, model_pref, stickiness, ttl)
```

- First sighting of a fingerprint: classify it (B1, or B3 if B1 abstains) and **store the mapping**.
- Every subsequent sighting: **exact match, zero cost, zero latency, confidence 1.0.**
- This turns "classify every request" into "classify every *application*" — a set that is typically
  single digits.

It also solves the session-id problem for free (§8.5): the fingerprint is a stable identity for the
*application*, and `hash(fingerprint, user_or_client_id)` is a perfectly good sticky key even when
the caller never sends `session_id`.

**Caveat to design for:** a fingerprint that embeds a timestamp or request id (one of ofox's three
canonical cache-miss patterns) will never repeat, so B2 silently degrades to "always miss." The router
should **measure fingerprint cardinality per client** and warn when one client produces unbounded
distinct fingerprints — that warning is simultaneously a cache-hit-rate bug report for the user's own
prompts. Two birds, one metric.

**B3 — Classifier model, but only on turn 1, and only when B1/B2 abstain.**
Affordable per §8.2. Three implementation notes:

- **Do not serialise it.** Speculatively issue turn 1 to the workload default *at the same time* as
  the classifier, then pin from turn 2 onward. You pay one turn of possibly-suboptimal routing to
  avoid adding 100–300 ms of TTFT to the user's first response.
- **Or use the turn-1 model itself.** Ask the routed model to return a `workload_class` field
  alongside its answer (structured output). Zero extra round-trip, zero extra cost beyond a few
  output tokens, and the pin applies from turn 2 — which is exactly where stickiness starts to matter.
- **Prefer a trained small classifier over a prompted LLM** once you have labels, per the calibration
  findings in `04`.

**B4 — Workload default.** Never fail to route. Default per deployment (a coding product defaults to
`code`), and record `source="default"` so misclassification is auditable.

---

## 8.5 Session identity — answering `07` Q1

Q1 asked whether the router gets a `session_id`. **Answer: it must work without one.** Priority order
for deriving the sticky key:

```python
def sticky_key(req) -> str:
    if req.session_id:                     return f"sid:{req.session_id}"           # best: caller-declared
    if req.messages and has_prior_turns:   return f"ctx:{hash(turn_1_prefix)}"      # first user turn is a natural session id
    return f"fp:{req.prompt_fingerprint}:{req.client_id}"                            # per-application, not per-session
```

Notes:

- **The first user turn is already a session id.** Two requests in the same conversation share their
  turn-1 prefix byte-for-byte (that is precisely what makes caching work), so hashing it gives a
  stable key with no cooperation from the caller. This is the key insight that unblocks Q1.
- **Never key on the full current prefix** — it changes every turn, so it is not a session key.
- **Key granularity determines affinity scope.** `sid:` = one conversation; `fp:client:` = all of a
  user's traffic through one app (higher hit rate, but cross-conversation cache sharing may be a
  compliance question — gate on `tags`).
- **Consistent hashing, with overflow that accepts the miss.** `07` Rule A already says this; the
  addition is that the sticky key must be **stable under rehashing** (rendezvous/HRW or a ring with
  bounded disruption), because rebalancing the ring is what silently converts a 90% hit rate into a
  12.5% one (ranvier.systems, `07` §7.1).

---

## 8.6 Reclassification policy — what to do when the intent changes

The user's framing is "understand the intent from point A". The corollary nobody states: **intent can
change mid-session**, and the router must not thrash when it does.

| Trigger | Action | Why |
|---|---|---|
| `workload_class` unchanged | stay pinned | default |
| classifier now disagrees, confidence < τ_high | **stay pinned**, log disagreement | a cheap correction signal is not worth a cache write |
| classifier disagrees, confidence ≥ τ_high, **and** `R_remaining ≥ R*` | migrate **once**, set `switched_at`, forbid return | amortization test passes |
| classifier disagrees, `R_remaining < R*` | stay pinned, log as `stranded` | migration cannot pay back — this is a real, measurable loss you should surface |
| explicit trigger (tool failure, verification drift — arXiv:2607.11399) | prefer a **sub-agent** on a stronger model over a mid-session switch | `07` §7.2(a); the sub-agent gets its own fresh prefix, so no write penalty on the parent session |
| task boundary / new sub-agent | re-decide freely | no cache to lose |
| **any** migration request when `switched_at` is set | **refuse** (no-oscillation lock) | §8.3 — oscillation costs more than never routing |

The `stranded` counter is worth exposing. It is the metric that tells a user "your router wanted to
save you money 4,000 times this week and could not, because sessions were too short to migrate" —
which is an argument for cheaper *entry* classification, not for looser stickiness.

---

## 8.7 Three findings from this pass that belong in the architecture

### (1) The router must *learn* cache state — it cannot assume it

The cache lives on the provider's side. The router's `affinity_bonus` is an **estimate**, and if it
assumes a hit it will systematically overvalue the pinned pair. But every response already carries the
ground truth:

| Provider | Fields in the response |
|---|---|
| Anthropic | `cache_creation_input_tokens`, `cache_read_input_tokens`, `input_tokens` |
| OpenAI | `usage.prompt_tokens_details.cached_tokens` |

So the adapter must **parse these into a per-`(session_key, model)` empirical hit rate**, and the
scorer must use the measured rate, seeded by a prior, rather than a constant. This makes
`07` Rule C implementable instead of aspirational — and it is also the only way to detect that a
user's own prompt engineering is destroying the cache.

```python
# learned, not assumed
affinity = ewma_hit_rate(session_key, model, default=prior_for(workload_class, ttl, gap))
```

### (2) TTL vs inter-arrival time — affinity priors are workload-class dependent

The 5-minute TTL is refreshed on hit, but **chat is bursty**: humans think between turns. If the
median gap exceeds the TTL, a chat session's real hit rate is near zero while an agent's (seconds
between steps) is near 100%. So the affinity prior should be a function of *expected* gap:

| Workload | Typical inter-arrival | Expected hit rate on a 5-min TTL | Affinity bonus |
|---|---|---|---|
| agent (inner loop) | 0.5–10 s | high | large |
| chat (human) | 10 s – many min | **variable, often low** | moderate, and *measured not assumed* |
| RAG (follow-up refinement) | seconds–minutes | moderate | moderate |
| batch | n/a (single shot) | n/a | **zero — don't pay for affinity** |
| code (IDE agent) | seconds | high | large |

`07` Rule E gated affinity on prefix length (`> ~500` tokens). It also needs gating on
**`expected_gap < ttl`** and on the provider's **minimum cacheable prefix** — 1,024 on OpenAI and
Sonnet/Opus-class Claude, 4,096 on Haiku 4.5 / older Claude, 4,096 on Gemini 2.5/3, **32,768 on
Gemini 2.0**. A 2,000-token chat prefix earns nothing on Haiku or Gemini 2.0 no matter how sticky the
routing is. That table belongs in the price registry next to `cache_read`.

### (3) Batch workloads should be *exempt* from affinity logic

> ⚠️ **PARTIALLY CORRECTED in `09-ttl-tier-policy.md` §9.4(2).** Batch is exempt from **session
> affinity** (there is no session), but **not** from cache policy: Anthropic's docs say batch
> requests "can be processed concurrently and in any order, cache hits are best-effort" and that
> "**the 1-hour cache can help improve your cache hits**," with OpenRouter recommending a 1h
> breakpoint on the shared prefix reused across successive batches. So batch still needs a cache
> decision — just a different one.

A batch job is one shot per prefix: `R = 0`, so `R*` is never satisfied and affinity can only add
scoring noise. Batch should route purely on price (and provider batch APIs, ~50% off per `07` §7.2(d))
and skip the whole cache-affinity branch. This is a small rule that removes a whole class of
misrouting.

---

## 8.8 Schema and API changes (delta on `07` §7.4)

```python
class WorkloadSource(str, Enum):
    explicit_tag = "explicit_tag"      # Path A — caller declared
    agent_profile = "agent_profile"    # Path A — named agent/profile
    fingerprint = "fingerprint"        # Path B2 — known application prompt
    structural = "structural"          # Path B1 — request shape
    classifier = "classifier"          # Path B3 — model call, turn 1 only
    default = "default"                # Path B4 — logged, auditable

@dataclass(frozen=True)
class SessionPolicy:
    session_key: str
    workload: WorkloadClass
    source: WorkloadSource             # ← so you can measure per-source quality
    confidence: float
    pinned_model: str
    pinned_backend: str
    pinned_at: float
    switched_at: float | None          # ← no-oscillation lock
    expires_at: float                  # idle eviction
    observed_cache_hit_rate: float | None   # ← learned (§8.7-1), not assumed

@dataclass(frozen=True)
class RoutingRequest:                 # fields ADDED to 07's version
    workload: WorkloadClass | None    # ← now OPTIONAL; None triggers Path B
    session_id: str | None            # ← still optional; §8.5 derives a key without it
    agent_profile: str | None         # Path A
    prompt_fingerprint: str | None    # Path B2 (computed by the router if absent)
    expected_turns: int | None        # → classifier budget + R* feasibility
    expected_gap_s: float | None      # → TTL viability (§8.7-2)
    ...                               # messages, tags, constraints as in 07
```

Public surface:

```python
router = Router(registry, policy="session-sticky")

# Path A — the recommended integration for products
support = router.agent("support-triage")        # workload="chat", stickiness=STRICT
await support.complete(messages)

# Path A — explicit tag, no profile
await router.complete(messages, workload="agent", session_id="s-42")

# Path B — nothing declared; router infers at turn 1 and pins thereafter
await router.complete(messages)

# Introspection (non-negotiable per 06): every decision explainable
d = router.explain(last=True)
# → workload='chat' source='fingerprint' conf=1.00 pinned='sonnet-4.6'
#   affinity=0.93 observed_hit=0.91 switch_cost=$0.1875 R*=4.75 turns_remaining=12
```

---

## 8.9 Priority changes (delta on `07` §7.5)

Promoted to **P0** (was implicit / absent):

| Item | Why P0 |
|---|---|
| `WorkloadSource` enum + per-source metrics | without it you cannot tell whether the classifier is helping; it is 6 lines of code |
| Prompt-fingerprint registry | makes Path B nearly free and supplies a sticky key when `session_id` is absent |
| Adapter parsing of `cache_read_input_tokens` / `cached_tokens` into a learned hit rate | turns `affinity_bonus` from a guess into a measurement |
| No-oscillation lock (`switched_at`) | §8.3 — oscillation costs more than not routing |
| `R*` amortization test with the direct-sum formulation | replaces Rule B's absolute with a computable rule |
| Minimum-cacheable-prefix + TTL fields in the price registry | §8.7-2; without them affinity is claimed where it cannot exist |

Promoted to **P1**:

| Item | Why |
|---|---|
| Turn-1 speculative routing + concurrent classification | avoids adding TTFT to the first response |
| Workload-class-dependent affinity priors | §8.7-2 |
| `stranded` counter (wanted to migrate, couldn't) | the metric that justifies better entry classification |
| OpenAI `prompt_cache_key` pass-through | provider does affinity routing for you |
| Batch exemption from affinity logic | §8.7-3 |

**Dropped:** "per-step difficulty classifier" stays a non-goal (`07` §7.5). Nothing in this pass
revives it; §8.2 explains why turn 1 is different and step 40 is not.

---

## 8.10 How to verify this design (eval additions to `06`'s harness)

1. **Cache hit rate by `WorkloadSource`.** If `fingerprint` and `explicit_tag` sessions have
   materially higher hit rates than `classifier` sessions, Path B3 is costing you real money and
   should be tightened or dropped.
2. **Misclassification audit.** Sample N sessions, label the true class by hand, compare to
   `SessionPolicy.workload`. Report per-source accuracy. This is the *only* way to catch a turn-1
   error, because it is invisible at request granularity.
3. **Oscillation rate.** Count sessions with ≥2 migrations. Target: **exactly 0** (the lock enforces
   it, so a nonzero value is a bug).
4. **`stranded` rate.** Fraction of sessions where `R_remaining < R*` at the moment a better model
   became available.
5. **Fingerprint cardinality per client.** Unbounded growth ⇒ the client's prompts contain volatile
   content ⇒ their cache is broken regardless of what the router does. Surface as a warning.
6. **A/B: sticky vs per-request routing on the same traffic.** This is the number that justifies the
   whole design, and per gmicloud it should be large (routing −59%, caching −35%, combined −75%).

---

## 8.11 Open questions this pass does *not* answer

1. **Turn-1 misclassification cost is unquantified.** `R*` tells you the cost of migrating; nothing
   here tells you the expected cost of a *wrong pin* across a distribution of session lengths. Needs
   a simulation, not a formula.
2. **Should the fingerprint registry be local or shared across deployments?** Shared gives faster
   warm-up and cross-tenant learning; it also leaks "customer X runs this system prompt." Default to
   local, opt-in to shared.
3. **Does `expected_turns` exist in practice?** Most callers don't know. Fallback: infer from
   `workload_class` (agent ≈ 10–50, chat ≈ 3–8, RAG ≈ 1–3) — **these priors are my assumption, not
   measured; flag for calibration.**
4. **Anthropic's 1-hour TTL (`w' = 2.0`) interacts with `R*` non-linearly.** A long chat session
   should arguably pay 2.0× once rather than 1.25× nine times (blog.sandbase.ai's worked example:
   $2.0025 vs $0.66 over 45 minutes). The router could **choose the TTL tier**, not just the model —
   that is a routing decision nobody in the literature makes.
5. **Unverified:** whether any provider bills cache writes on a *failed* or aborted request. If they
   do, a speculative turn-1 route (§8.4 B3) has a cost I have not priced.

---

## Cross-references

- `07` §7.1 — KV-cache finding, Rules A–E (**Rule B corrected here; Rule E extended in §8.7-2**)
- `07` §7.4 — unified decision model (**schema delta in §8.8**)
- `07` §7.6 Q1, Q2 — **answered in §8.5 and §8.4 respectively**
- `06` — API surface, eval harness spec (**harness additions in §8.10**)
- `04` — calibration methodology (applies to the Path B3 classifier)
