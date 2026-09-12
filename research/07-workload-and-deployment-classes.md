# Thread 7 — Workload-Class × Deployment-Class Routing (the revised scope)

Research date: 2026-09-12. **This file supersedes parts of `06-design-recommendations.md`.**

Revised product scope: a Python router package that routes across **any workload class**
(agent / chat / RAG / batch / code / vision) **and any deployment class** (self-hosted / API /
serverless GPU), choosing both *which model* and *which backend*.

---

## 7.0 The core reframe

Routing is not one decision. It is a decision over **two orthogonal axes plus a cache constraint**:

```
Axis 1: WORKLOAD CLASS   → what the request is (agent step, chat turn, RAG query, batch job)
Axis 2: DEPLOYMENT CLASS → where it can run (self-hosted GPU, API, serverless, edge)
Constraint: KV-CACHE AFFINITY → what you already paid to warm up
```

Every technique in `03-technique-taxonomy.md` is a *scoring function* for Axis 1. Almost nothing in the
2025 literature models Axis 2 or the cache constraint. **That is where a new package wins.**

---

## 7.1 THE KV-CACHE FINDING — this should drive your entire agent-routing design

### The mechanism
Prompt caching reuses the key/value tensors computed for a prompt prefix. But those tensors are
produced by **that model's own weights** — so **the cache is model-scoped and non-transferable**.

Verified from `dev.to/sumedhbala/claude-code-costs-act-ii` (2026-06-27, *measured* data):
> "The biggest swing in a multi-model bill comes from one move — **switching models** — because the
> prompt cache belongs to a single model. The instant you switch, the cache you already paid for is
> thrown away."
>
> Concrete measurement: Haiku **cold-wrote a 57,739-token duplicate** of a shared prefix it could never
> read from Sonnet — **~85,113 total cache-creation tokens of pure duplication** that a single-model
> session would never pay.

They also measured the thinking-block interaction: Haiku strips thinking before it reaches the cache (so
stripping is free), while **Sonnet keeps thinking in the prefix** — removing it cold-rewrites ~9.3K
tokens. Cache behaviour differs *per model family*, not just per model.

### The economics
| Source | Finding |
|---|---|
| dreaming.press (2026-07-28) | Claude Sonnet cached input ≈ **$0.30/M vs ~$3.00/M uncached — a 10× gap** on identical content. "**One changed token** invalidates the cache from that point to the end." Manus calls KV-cache hit rate "the single most important production metric for an agent." |
| spheron.network (2026-06-17) | At 100:1 input:output ratios, **prefill is 85–95% of total GPU time**. 90% hit rate ⇒ `0.90 × 0.92 = 82.8%` compute saving. "The difference between 0% and 90% hit rate is the difference between a **$20,000/month GPU bill and a $2,000/month bill**." |
| gmicloud.ai (2026-08-10) | **Cache-hit pricing:** Anthropic $3.00→$0.30/M (**90%** off), OpenAI GPT-4o $2.50→$1.25/M (**50%**), Kimi K3 $3.00→$0.30/M (90%). Provider discounts are **not uniform** — a 50% discount changes the routing math vs a 90% discount. |
| ranvier.systems (2026-04-30) | Measured: 8 backends + random routing = **12.5% cache hit** (chance), p99 TTFT 6,800 ms, 36.3 req/s. Prefix-aware routing = **97.5% hit**, p99 TTFT **1,000 ms**, 44.4 req/s. Same GPUs, same model, same workload. |
| CacheRoute, arXiv:2608.19677 | On Llama-3.3-70B fp8 / 60×H100: Flat-LB **64.1%** KV hit, Sticky **87.3%**, CacheRoute **93.2%**. Capacity@3.5s SLO: Flat-LB 42 QPS, **Sticky 30 QPS**, CacheRoute **176 QPS**. → **pure sticky routing has the best hit rate but the worst load balance.** You need affinity *plus* rebalancing. |
| GKE Inference Gateway (spheron, 2026-06-22) | Round-robin wastes **~88%** of prefill compute vs **~12%** with KV-aware routing; effective throughput **3.1×**. |
| gmicloud.ai (2026-08-20) | **Routing alone −59%, caching alone −35%, combined −75%** vs frontier-only on a 10k-request/day workload. They are *complementary and multiplicative*, not alternatives. |

### The four rules that protect the cache (provider-agnostic, from dreaming.press)
1. Keep the prompt prefix **byte-stable** (no timestamps with second precision in the system prompt).
2. Make context **append-only** (never mutate history in place).
3. **Serialize deterministically** (non-deterministic JSON key order kills it).
4. **Mask tool logits instead of adding/removing tool definitions** (editing tool defs mid-conversation invalidates the prefix).

### What this means for your package — the design rules

> **Rule A — Session affinity is the default, not an optimization.**
> Route by `hash(session_id)` to a *(model, backend)* pair. gmicloud.ai: "Load balancers should use
> **consistent hashing on session identifiers** rather than round-robin for conversational workloads,
> **with overflow traffic routed to secondary instances that accept the cache miss rather than breaking
> affinity for the whole session**."

> **Rule B — Never switch model mid-session for cost reasons alone.**
> The saving from a cheaper model must exceed the cost of re-writing the entire prefix at cache-write
> prices. For a 57k-token prefix on Claude, that's real money on every switch. **Model switches should
> happen at task boundaries (new session, new sub-agent), not turn boundaries.** This is exactly what
> the Entelligence "per-turn" router and Hermes `/model` do — they switch at *task* granularity.

> ⚠️ **CORRECTED in `08-session-stickiness-and-entry-intent.md` §8.1 / §8.3.** Stated as an
> absolute, Rule B is false: OpenAI charges **no cache-write premium** (pre-GPT-5.6), so migrating
> onto an OpenAI-backed model costs *nothing* in cache-write terms — the verified break-even is
> **0.8 turns**, inside a single turn. The correct rule is *"never **oscillate**; evaluate a one-way
> migration against the `R*` amortization test in `08` §8.3."* The instinct (task boundaries, not
> turn boundaries) survives; the absolute does not.

> **Rule C — Cache state is a first-class routing signal.**
> Emit `prefix_cache_hit_estimate` and `cache_write_cost_if_switching` as signals the policy can read.
> gmicloud.ai's verification metric: "the cache hit rate **broken down by routing decision** — if the
> hit rate for requests routed to their cache-affinity endpoint is significantly higher than for
> requests routed elsewhere, the routing layer is correctly incorporating cache state."

> **Rule D — Model the provider's cache discount, not just its sticker price.**
> A model with a 90% cache discount and a sticky session can be cheaper than a 5×-cheaper model with a
> 50% discount that you switch to constantly. Your price table needs
> `(input, output, cache_read, cache_write_5m, cache_write_1h, discount_pct)` per model.

> **Rule E — Short prefixes don't care.**
> ranvier.systems: for prefixes <500 tokens, "the routing overhead (~3 ms minimum) can exceed the
> savings." And for fully unique conversations, "there's nothing to cache." **Gate cache-affinity logic
> on `shared_prefix_tokens > ~500`.**

---

## 7.2 Workload classes — what each one actually needs

### (a) AGENT workloads

**Granularity is the design decision.** Not Diamond (2026-04-23) names three levels:
- **Task level** — planning vs code-gen vs summarization are different tasks; users switch tasks mid-session.
- **Step level** — inside a task, interpreting a tool result differs from proposing a fix.
- **Session level** — one model for the whole run.

**Evidence on which to choose:**

| Finding | Source | Implication |
|---|---|---|
| "Agent execution is multi-step, stateful, and **recoverable**: a cheap decision at one step may force costly correction later, while a strong model may be unnecessary for routine control but essential after verification detects drift" | arXiv:2607.11399, *Agentic Routing: The Harness-Native Data Flywheel* (2026-07) | Step-level routing needs a **drift/verification trigger**, not just a difficulty score |
| "Dynamic routing adds a classification step before every request, introducing latency (typically **50–200 ms per routing decision) that compounds across hundreds of agent calls per session**" | augmentcode.com (2026-06-18) | Per-step routing must cost <10 ms or it eats the saving |
| A **zero-shot LLM-as-a-Router**, even with Claude Sonnet 4.6, "falls short of the per-task oracle **by a wide margin**"; its selections were "nearly uniform, failing to exploit the strong dimensional structure" | arXiv:2606.22902, *Agent-as-a-Router* (2026-06) | Confirms the anti-pattern independently, in the agentic setting specifically |
| Existing routers "evaluate routers only on one-shot prompts. They never expose the router-visible prefix at an intermediate agent step, never test whether a cheaper replacement preserves downstream task success" | OmniRouter critique (KDD 2025) | Your eval harness must support **mid-trajectory** evaluation |
| Four OSS routers (RouteLLM, LiteLLM, vLLM-SR, Aurelio) tested on BFCL v4, tau2-bench, WebArena: effects **inconclusive on 3 of 4, equivalent on 1**. All four "decide once per task or session" — none does per-step | arXiv:2608.14641 (2026-07-28) | **Per-step agent routing is unproven territory.** Treat it as experimental, measure hard |
| Same paper: LiteLLM and RouteLLM made **identical selections on every task**, but RouteLLM paid $0.01/request + 2.1–2.3 s decision latency vs LiteLLM's 0.7–1.4 ms → **LiteLLM strictly dominates** under a cost-and-latency-aware utility | arXiv:2608.14641 §6.9 | Decision latency is decisive in agent loops. RouteLLM's 2.1 s is disqualifying |
| Coding agents: "Reading a file, planning a cross-repository change, reacting to a test failure, and writing a summary **do not require the same model**" | entelligence.ai (2026-08-03) | Per-turn routing *within* a coding session is a real, monetized product |
| Hermes: three independent mechanisms — `delegate_task` (parallel subagents, capped at 3), `auxiliary:` task slots (background), `/model` (mid-session switch) | techjacksolutions.com (2026-09-07) | **Sub-agent delegation is a separate routing surface from model routing.** Model it explicitly |
| "Model routing is **per-session, not per-agent**. Tool routing is already unlimited" — and "model routing changes how well your agent thinks; tool routing changes what your agent can do, and the latter is far more powerful" | workos.com (2026-03-18) | Don't conflate the two; your package is model routing, but the config should not fight tool routing |
| Tool-call reliability, not benchmark score, "is the single biggest differentiator between models in production" for agents | docs.promptise.com (2026-07-16) | Ship a **tool-call reliability signal** in the model registry |

**Agent routing policy to ship:**
```
session start → pick (model, backend) once, using task-level difficulty + tool-call reliability
              → pin via consistent hash on session_id
per step      → cheap deterministic check only (<1 ms): did a tool fail? did verification drift?
              → escalate ONLY on an explicit trigger, and prefer a *sub-agent* over a mid-session switch
sub-agent     → own session, own cache, own model — this is where model switching is free
task boundary → re-decide
```

### (b) CHAT workloads
- Dominant cost axis: **TTFT** and cache hit rate (multi-turn history is a growing stable prefix).
- Per-turn routing *is* viable here — but only with session affinity, so the "routing" is really
  *tier selection at session start* + escalation on difficulty spikes.
- channel.tel (2026-07-04): CX agents responding in 300–600 ms make 15 ms of routing overhead invisible;
  **voice agents** are the exception — measure before committing.
- **Hard exclusions matter most here**: financial, medical, regulatory, account-compromise, already-
  escalated sessions, and *any turn following a tool failure* stay on the strongest model.

### (c) RAG workloads — **needs its own router, this is a real finding**

| Finding | Source |
|---|---|
| **RAGRouter** is "the first RAG-aware routing method." Non-RAG-aware routers are measurably worse: RAGRouter beats GraphRouter by **+3.29%**, MF by **+4.21%**, KNN by **+4.02%**, RouterDC by **+7.75%**; beats the best single RAG-enabled LLM by **+3.61%** (avg 64.46% vs Llama-3.3-70B's 60.85%) | arXiv:2505.23052 (v2 2025-10-17) |
| Why: RAGRouter uses **contrastive learning to capture "knowledge representation shifts induced by external documents"** — the retrieved context changes which model is best, and query-only routers can't see it | arXiv:2505.23052 |
| Robust to retrieval noise: on TriviaQA with injected Golden/Relevant/Irrelevant/Counterfactual noise, RAGRouter averaged **90.83%** vs Oracle Single Best **87.92%** | arXiv:2505.23052 App. G |
| **Routing *retrieval depth* is a separate, high-value axis.** RAGRouter-Bench: 7,727 queries, 4 domains, 3 query types (factual/reasoning/summarization), 5 RAG paradigms. Best config = **TF-IDF + SVM: macro-F1 0.928, 93.2% accuracy, 28.1% token savings** vs always-most-expensive | arXiv:2604.03455 (2026-04) |
| Critical caveat: "paradigm applicability is shaped by **query-corpus interactions, not query type alone**" | arXiv:2604.03455 |
| CA-RAG: routing retrieval depth gives **26% fewer billed tokens** vs always-heavy and **34% lower mean latency** vs always-direct, at equivalent quality. "Always-heavy retrieval inflates prompt tokens by **36%** for no quality gain on the majority of queries" | arXiv:2606.02581 (2026-03) |
| But: CA-RAG was evaluated on **28 queries**. Tiny. | arXiv:2606.02581 — flag |
| RouteRAG: end-to-end **RL** over text + graph retrieval, two-stage training on task outcome + retrieval efficiency | arXiv:2512.09487 (2025-12) |
| Counter-evidence on retrieval complexity: **DOS RAG** (simple, structure-preserving) matches or beats ReadAgent/RAPTOR at all budgets ≥5K tokens; at 30K on ∞Bench, 93.1% vs Vanilla 87.8% | arXiv:2506.03989 |

> **Design implication: RAG needs (1) the retrieved context in the routing feature vector, not just the
> query, and (2) a separate retrieval-depth decision. A query-only router is provably worse in RAG.**

### (d) BATCH workloads
- No user waiting ⇒ latency is free, throughput is everything. Route for $/token and GPU utilization.
- Provider batch APIs are ~**50% off** on-demand (Bedrock Batch/Flex; others similar) — a batch-aware
  router should prefer them by default.
- "Overnight batch scoring of 100K+ documents → vLLM. Continuous batching gets more tokens per GPU-hour
  than per-token API pricing" (markaicode, 2026-07-22).

### (e) CODE workloads
- Best-documented per-role routing in the whole corpus (augmentcode.com, 2026-06-18):
  Opus for planning, Sonnet for implementation, Haiku 4.5 for file navigation/linting/sub-agent execution.
- Concrete arithmetic: 3 implementation tasks at 12K in / 8K out each → **Opus $0.78 vs Sonnet $0.468**.
  Haiku 4.5 at $1.00/M input is **3× cheaper than Sonnet, 5× cheaper than Opus**, cache reads $0.10/M —
  "those savings compound across hundreds of repeated tool invocations per session."
- Reported **40–60% cost reduction** vs Opus-for-everything on a typical multi-agent coding session.

### (f) VISION / MULTIMODAL
- TSRouter (arXiv:2607.08940): routes each query to the best **(modality, model)** pair — text LLM vs
  visual/mix VLM — via a 4-partite heterogeneous graph over task/query/modality/model nodes, with
  cost-aware routing and **zero-shot generalization to unseen models**.
- LLMRouter already ships multimodal routing on Geometry3K, MathVista, Charades-Ego.
- Cheap structural rule: presence of image/audio ⇒ modality gate first (a rule, not a classifier).

---

## 7.3 Deployment classes — the signals nobody models

### The break-even economics (use these to set defaults)

| Source | Break-even |
|---|---|
| markaicode (2026-08-07), vLLM vs Anthropic API | **API wins under ~30–40M tokens/month.** vLLM wins above ~35M (owned GPUs) to ~100M (rented) |
| tensoria.fr (2026-05-15) | API below ~50M tokens/month. At $3/M you need **500–833M tokens/month** to justify a GPU |
| digitalapplied (2026-05-27) | Past **~1B tokens/month at sustained 60–70%+ load**, owning capacity wins; savings up to ~5× at industrial scale. "**A 10× penalty waits for any deployment that runs idle**" |
| markaicode, vLLM vs RunPod | Self-host wins above **~50–65% GPU duty cycle**; below that, serverless |
| The proven hybrid pattern | "**self-hosted smaller model handles 80–90% of requests** (routine queries, extraction, classification), and an API call to a frontier model handles the complex cases" (tensoria.fr); localaimaster reports 85–95% local / 5–15% cloud in three production deployments |

### The signals you need per backend

| Signal | Why | Source |
|---|---|---|
| `gpu_utilization` (0–100) | Overflow trigger | sitepoint hybrid guide uses `GPU_COMPLEX_THRESHOLD = 90`, `GPU_SIMPLE_THRESHOLD = 85` — **simple tasks overflow sooner to preserve local capacity for hard ones** |
| `queue_depth` | Backpressure | LiteLLM/SGLang |
| `kv_cache_utilization` | The binding constraint on concurrency | PagedAttention; "a single 128K session needs ~40 GB of KV cache, half an H100 80GB before weights" |
| `prefix_cache_hit_rate` | See §7.1 | `vllm:gpu_prefix_cache_hit_rate` |
| `model_loaded` / `cold_start_s` | **LLM serving pods take 5–15 MINUTES to become ready** (container start, pull 10–30 GB weights, load to VRAM, warm KV cache) ⇒ reactive autoscaling is useless | tensoria.fr FAQ |
| `data_residency_tags` | Hard constraint, not a preference | sitepoint: "**fail closed** if local unavailable" for sensitive data |
| `rate_limit_remaining` | Cloud overflow trigger | `CLOUD_RATE_LIMIT_THRESHOLD = 10` |
| `idle_cost_per_hour` | The 10× idle penalty | digitalapplied |
| `cache_discount_pct` | See §7.1 Rule D | gmicloud |

### The hybrid routing pattern to ship (validated shape)

sitepoint.com's three-pillar model (2026-04-22) is the cleanest formulation found, with real code:
```
Pillar 1  SENSITIVITY  → sensitive ⇒ local, FAIL CLOSED if local down (throw, don't fall back to cloud)
Pillar 2  COMPLEXITY   → complex ⇒ cloud, unless cloud degraded
Pillar 3  AVAILABILITY → local saturated ⇒ cloud overflow; cloud down ⇒ local fallback
```
Note the **asymmetry that everyone gets wrong**: cloud-down falls back to local, but
*sensitive-request-with-local-down must error*, not fall back. Your `fallbacks` config needs a
`fail_closed: true` option per policy.

The production trap, verbatim: "if you load-test locally and think you're ready, a **10× traffic spike
will saturate your GPU and *all* requests will timeout and cascade to cloud, which now costs you 100×
more**. Measure and pre-warm your local instance capacity, and set **aggressive local timeouts (not
30 s)** so you fail-fast to cloud instead of blocking users."

### The architectural principle that justifies this whole package
Vercel (2026-08-31):
> "**The routing layer and the model layer are separate decisions, and settling routing first keeps
> every model choice reversible.** Treating 'self-hosted LLM' as one yes-or-no decision is what
> produces GPU purchases that a routing change would have solved."

That is your pitch. The router is what makes the self-host-vs-API decision *reversible and per-request*
instead of a one-time capital commitment.

---

## 7.4 The unified decision model

```python
@dataclass(frozen=True)
class RoutingRequest:
    messages: list[Message]
    workload: WorkloadClass          # agent | chat | rag | batch | code | vision
    session_id: str | None           # → cache affinity key
    step: StepContext | None         # agent only: tool history, prior failures, verification state
    retrieved: list[Chunk] | None    # RAG only: goes INTO the feature vector
    tags: set[str]                   # sensitivity, tenant, compliance
    constraints: Constraints         # max_cost, max_latency, budget_remaining, residency

@dataclass(frozen=True)
class Candidate:
    model: ModelSpec                 # pricing incl. cache_read/cache_write/discount
    backend: BackendSpec             # self_hosted | api | serverless; live health + cache state
    affinity: float                  # estimated prefix-cache hit if this pair is chosen
    switch_cost: float               # $ to cold-write the prefix if switching away from the warm pair
```

**Scoring is per (model, backend) *pair*, not per model.** This is the single most important schema
decision in the package — it's what makes Axis 1 and Axis 2 composable instead of sequential.

```
score(model, backend) = w_q · quality(workload, features)
                      − w_c · cost(model, backend, cache_hit=affinity)
                      − w_l · latency(backend, queue_depth, cold_start_risk)
                      − w_s · switch_cost          # ← the term nobody has
                      + w_a · affinity_bonus       # ← sticky-session reward
subject to: residency(tags) ∧ budget_remaining ∧ context_window ∧ tool_support
```

---

## 7.5 Revised strategy priorities for this scope

| P | Capability | Why it's now P0 |
|---|---|---|
| **P0** | **Model registry with cache-aware pricing** (`input, output, cache_read, cache_write_5m/1h, discount_pct`) | Without it the routing math is wrong by up to 10× on cached traffic |
| **P0** | **Backend registry with live health signals** (util, queue depth, KV util, cold-start state, residency) | Axis 2 is the differentiator |
| **P0** | **Session affinity via consistent hashing**, with overflow that accepts a cache miss rather than breaking the session | §7.1 Rules A & B |
| **P0** | **Workload-class tag** driving policy selection (agent/chat/RAG/batch/code/vision) | The user's core requirement |
| **P0** | Rules gate with `fail_closed` per policy | Sensitive data must error, not fall back |
| **P0** | Cost-aware + clustering router (from `06-design-recommendations.md`) | Unchanged |
| **P1** | **RAG-aware features** (retrieved context in the vector) + **retrieval-depth routing** as a second decision | RAGRouter proves query-only routing is 3–8% worse |
| **P1** | **Agent step-level escalation on explicit triggers** (tool failure, verification drift) — not difficulty score | arXiv:2607.11399; and per-step routing is unproven (arXiv:2608.14641) |
| ~~P1~~ → **P0** | **Sub-agent delegation as a first-class routing surface** | Hermes pattern; switching is free at sub-agent boundaries. **Promoted to P0 and BUILT after `10`: Cognition Fusion shipped exactly this and reports −23% to −46% cost. See `10-fusion-delegation-architecture.md`.** |
| **P1** | Hybrid local↔cloud overflow with **aggressive local timeouts** and pre-warm | The 100× cascade trap |
| **P2** | Cache-affinity-aware load balancing (CacheRoute-style: affinity + replication + LPT) | Sticky alone gives 87% hit but 30 QPS; you need both |
| **P2** | Batch-mode routing (prefer provider batch APIs, ~50% off) | Free money for offline workloads |
| **P3** | Multimodal / (modality, model) routing | TSRouter, LLMRouter already ship it |

### Revised non-goals
- **Don't build a tool router.** workos.com: tool routing is more powerful than model routing, but it's
  a different product. Interoperate with MCP; don't reimplement it.
- **Don't do per-step model switching by default.** The cache math says no, and the benchmarks say
  unproven. Make it opt-in with a measured warning.
- **Don't try to autoscale GPUs.** Serving pods take 5–15 minutes; that's a serving-layer problem. Your
  job is to *route around* cold capacity, not to warm it.

---

## 7.6 New open questions

1. **What is your session model?** Do you get a `session_id`, or must the router infer session
   boundaries from message history? Cache affinity depends entirely on this.
2. **Do you control the prompt prefix?** If the user builds their own system prompt with timestamps in
   it, you can't protect the cache from inside the router — you can only measure the damage and expose it.
3. **Which providers' cache semantics must you model first?** Anthropic (90%, explicit breakpoints,
   per-model thinking-block behaviour) and OpenAI (50%, automatic) differ enough that a single
   abstraction will leak.
4. **RAG: do you route the retriever, the reader, or both?** CA-RAG and RAGRouter-Bench both route
   retrieval *strategy*; RAGRouter routes the *reader*. Different decisions, different features.
5. **Agent frameworks: integrate or intercept?** LangGraph / CrewAI / OpenClaw all have their own
   model-config surfaces. A drop-in `RouterClient` plus an OpenAI-compatible server covers all three
   without integrating with any.
