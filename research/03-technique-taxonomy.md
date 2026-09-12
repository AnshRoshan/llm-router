# Thread 3 — Technique Taxonomy: every routing technique, at three depths

Research date: 2026-09-12. Organized by technique family, each with: definition/mechanism → strengths/limits →
cost/quality tradeoff → reference implementations → real-world use.

The two dominant taxonomies in the 2026 literature agree, so I use their union:

**A. When the decision is made** (arXiv:2603.04445; arXiv:2502.00409)
- *Pre-generation* — decide from the query alone, before any model runs.
- *Post-generation / cascade* — run a model, judge the output, escalate if inadequate.
- *Multi-stage* — both, composed.
- *Online/adaptive* — policy updates from feedback.

**B. What signals feed it** — Query features · Model metadata · Response-level · Feedback.

**C. How it is computed** — Heuristic (rules) · Supervised (classifier) · Bandit · RL policy · Optimization.

---

## 3.1 RULE-BASED ROUTING

### Definition & mechanism
Deterministic mapping from request attributes to a model. Signals: keyword/regex match, prompt length,
context-window size, language detection, presence of tool calls / structured output / images, user role,
tenant, task label, explicit metadata. Sub-millisecond, no training, fully auditable.

### Evidence

| Source | Finding | Confidence |
|---|---|---|
| vLLM Semantic Router, arXiv:2603.04444 (2026) | Rule signals (keyword patterns, regex, language detection, context length, role-based authz) are **sub-millisecond and deterministic**, composed with neural signals via Boolean rules; "only signal types referenced by configured decisions are computed" (demand-driven) | High — primary |
| LiteLLM docs (read 2026-09-12) | `enable_pre_call_checks` validates context window *before* the call — a pure rule that prevents paying for a doomed request | High — primary |
| morphllm.com (2026-03-31) | Keyword/regex: <1 ms, accuracy **50–65%**. Prompt-length heuristics: <1 ms, accuracy **45–60%** ("long prompts can be easy, short prompts can be hard") | Medium — vendor |
| channel.tel (2026-07-04) | Rule-based overhead **<1 ms**; weakness is *coverage* — unseen intents fall to the default tier, so "set that fallback to standard rather than fast" | Medium |
| ai-tldr.dev (2026-06-12) | Rule-based 10–50 ms; embedding similarity 50–200 ms; lightweight classifier 50–150 ms; LLM-as-judge 500–2000 ms | Medium |

### Strengths
- Zero added latency, zero added cost, zero model dependency.
- Fully deterministic → auditable, testable in CI, explainable to compliance.
- The only class that can enforce **hard exclusions** ("never send PII/financial/safety-critical queries to a cheap model").
- Always works at cold start — no training data needed.

### Limitations
- Poor semantic discrimination (50–65% accuracy per vendor benchmark).
- Requires anticipating every intent category; **routing drift** when the product ships new features
  (channel.tel names this as a top-4 gotcha).
- Keyword matching is adversarially trivial to defeat and does not measure *difficulty*.

### Cost/quality tradeoff
Free. Quality ceiling is the ceiling of your rule author. Best used as a **gate and a fast path**, never as the whole router.

### Real-world use cases
Tenant/policy pinning, compliance exclusion lists, context-length fallback, image/audio modality dispatch,
rate-limit avoidance, cache-eligibility gating, explicit user-requested model.

---

## 3.2 SLM-BASED / LIGHTWEIGHT CLASSIFIER ROUTING

### Definition & mechanism
A small model (BERT-class, ModernBERT, a fine-tuned encoder, a random forest on embeddings, or a matrix
factorization model) scores the query and predicts difficulty / best model / win-rate. This is the
workhorse of production routers in 2026.

### Evidence (per technique)

| Technique | Reference impl | Reported result | Latency | Confidence |
|---|---|---|---|---|
| **Matrix factorization** on preference data | RouteLLM `mf` | 95% GPT-4 quality on MT-Bench at 14% strong-model calls (with augmentation) | low | High (paper) |
| **BERT classifier** on preference data | RouteLLM `bert` | 45% cost savings on MMLU at comparable quality | mid | High |
| **Similarity-weighted Elo ranking** | RouteLLM `sw_ranking` | Best raw quality; heaviest | highest | High |
| **Random forest on embeddings** | Not-Diamond/RoRF | 12 pre-trained pairwise routers; outperforms RouteLLM at lower cost on MMLU/BBH (their claim); 12.5% cost cut vs Claude 3.5 Sonnet | ~100–150 ms hosted | High (repo), Medium (claims) |
| **ModernBERT + LoRA intent classifier** | vLLM Semantic Router | +10.2 pp accuracy, −47.1% latency, −48.5% tokens on MMLU-Pro; 98.53% routing accuracy with 805 examples / 2h compute | sub-10 ms signal extraction | High (arXiv:2510.08731), Medium (98.53%) |
| **Per-LLM regression (RF + MLP)** on quality+cost | MixLLM (NAACL 2025) | **97.25% of GPT-4 quality at 24.18% of cost**; with Llama 3.1 added: 98.55% at 16.79% | not reported | High (peer-reviewed) |
| **KNN router** | RouterBench baseline; pulzeai-oss/knn-router (Go) | RouterBench predictive baseline; RouterArena: KNN 69% acc but **$300.30/1K queries** — expensive | low | High |
| **MLP router** | RouterBench baseline | RouterArena: MLP 65% acc, **$286.70/1K** | low | High |
| **IRT (Item Response Theory) + BERT** | IRT-Router (ACL 2025) | Models *model ability* and *query difficulty+discrimination* separately → interpretable. RouterArena: MIRT-BERT rank #3 (0.6731) but RouterArena also notes MIRT-BERT can hit **5× the optimal cost** | ~27 ms | High |
| **Dual contrastive learning** | RouterDC (SUSTech) | Best *cost-ratio* score in RouterArena, but worst accuracy (32–33%) | ~10 ms | High |
| **Clustering (training-free)** | Avengers / **Avengers-Pro** (AAAI 2026 / CIKM) | See §3.6 — the standout 2026 result | embedding pass only | High |
| **Heterogeneous GNN edge prediction** | GraphRouter (ICLR 2025) | ≥12.28% Reward improvement over strongest baseline; ≥88.89% of oracle; generalizes to **new LLMs without retraining**; −99% time cost for new-LLM onboarding | graph forward pass | High |
| **Hidden-state probe** | ProbeDirichlet (arXiv:2602.11877, Feb 2026) | Internal hidden states beat output probabilities (which suffer softmax overconfidence); cross-layer aggregation via learned Dirichlet; +16.68% router ability, +18.86% in high-accuracy scenarios | requires model internals | High (primary) |

### Strengths
- **Best quality-per-millisecond of any learned approach.** 5–100 ms against 500–2,000 ms of inference.
- Trainable on your own eval data; calibratable to a target strong-model share.
- Deterministic given inputs → cacheable, testable.

### Limitations
- **Checkpoint rot.** RouteLLM's 2024 weights rank #25/26 on the 2026 RouterArena leaderboard.
- Distribution dependence: performs best when eval ≈ training distribution.
- RouterArena measured **low robustness** to surface perturbations for BERT-based academic routers.
- Needs labels (preference pairs, or outcome data from all candidate models — expensive).

### Cost/quality tradeoff
Router cost is ~$0.001–0.005/request locally, or ~100–200 ms hosted. Rule of thumb from
dreaming.press: routing pays only when **(a)** the strong/weak price gap is large and **(b)** a
meaningful share of traffic is genuinely routable. Below that, you pay latency for nothing.

### Real-world use
High-volume classification/extraction/summarization where a 70/30 easy/hard split exists; coding agents
(Morph, Not Diamond); self-hosted Kubernetes fleets (vLLM-SR).

---

## 3.3 LLM-BASED ROUTING (an LLM decides)

### Definition & mechanism
A cheap/fast LLM is prompted (or fine-tuned) to classify the query or pick the model. Variants:
zero-shot JSON classification, few-shot, or RouteLLM's `causal_llm` router (an LLM fine-tuned on
preference data).

### Evidence

| Source | Finding |
|---|---|
| morphllm.com (2026-03-31) | LLM-as-judge classifier: **1–5 s** latency, 85–90% accuracy. "If the average request takes 3 seconds, you have doubled the latency... The cost of the classification call itself also reduces net savings." |
| ai-tldr.dev (2026-06-12) | "The router latency paradox... An LLM-based classifier that takes 1,500 ms to decide routing adds a full second to every request — at that point you'd be better off routing nothing. **Keep the classification step under 200 ms.** LLM-as-judge classifiers don't." |
| channel.tel (2026-07-04) | Explicit anti-pattern: "**Don't use another LLM to classify.** This adds a third model call on escalations and negates some of your savings... The classification doesn't need language model reasoning; it needs a fast, deterministic mapping." |
| blog.dailydoseofds.com (2026-09-07) | "You pay for two inference calls... The savings survive only when the classifier costs far less than the gap between model tiers and routes accurately enough." Also: "**General models are mediocre routers** — 'fix this' or 'make it faster' contains almost no useful signal." |
| mindstudio.ai (2026-02-10) | LLM-assisted routing adds 500–2,000 ms + classifier inference cost + "another point of failure in your pipeline" + requires prompt engineering. Best for high-value requests where accuracy > latency. |
| ailog.fr (2026-08-03) | Counter-position: a 50-token GPT-4o-mini classifier adds only 30–80 ms; recommends "rule-based classifier for trivial cases, and an LLM classifier only for ambiguous cases." |
| arXiv:2603.04445 (2026 survey) | Notes "Prompt LLM" as a multi-round baseline that "directly prompts an LLM to select LLMs without explicit routing modules" |

**Verdict from the corpus:** LLM-based routing is **not** the production default in 2026. Its defensible
uses are (1) as the *last tier* of a cascade for genuinely ambiguous queries, (2) offline label
generation (RouteLLM's `D_judge` GPT-4-judge augmentation is exactly this — an LLM as *labeler*, not
as *router*), and (3) low-QPS/high-value workloads where a 1 s decision is amortized.

### Real-world use
Offline dataset labeling, agentic planners where routing is already an LLM call, low-volume enterprise
workloads, "Prompt LLM" baselines in papers.

---

## 3.4 CASCADE / POST-GENERATION ROUTING

### Definition & mechanism
Run the cheap model → estimate the quality/confidence of its answer → accept or escalate. No
pre-classification needed; the decision uses the *response*, which is strictly more information than
the query alone.

### Evidence

| Method | Mechanism | Result | Confidence |
|---|---|---|---|
| **FrugalGPT** (Stanford, TMLR 2024, arXiv:2305.05176) | Prompt adaptation + LLM approximation + **LLM cascade** with a learned scoring function and per-model thresholds τ; cascade order + thresholds solved as constrained optimization | "up to **98% cost reduction** at matched performance, or +4% accuracy at same cost"; **actual reported range 50–98% across datasets** (Table 3); on HEADLINES 98% cost cut with accuracy gain; on OVERRULING +1% accuracy at −73% cost | High (paper text read) |
| **Cascade Routing** (ETH, arXiv:2410.10347) | Provably optimal *unified* routing+cascading; picks the best model at each step, may skip/reorder | "consistently outperforms the individual approaches by a large margin"; **identifies good quality estimators as the critical factor** | High |
| **AutoMix** (NeurIPS 2024) | Self-verification of the small model's output as the escalation trigger | Foundational verification-routing baseline | High |
| **EcoAssistant** (2023) | Hierarchy of LLM assistants, cheapest first | Code-driven query answering | Medium |
| **BEST-Route** (Microsoft, arXiv:2506.22716, ICML 2025) | Sample the cheap model N times at varied temperatures; escalate only on genuine **disagreement** | Reduces unnecessary escalations a further **15–30%** | Medium (secondary source only) |
| **SATER** (EMNLP 2025) | Self-aware, token-efficient routing + cascading | — | Medium |
| **X-Router** (Findings ACL 2026) | Decouples knowledge from reasoning for cost-effective inference | — | Medium |
| **RASER** (2026) | Recoverability-aware selective escalation for multi-hop QA | — | Medium |
| **UCCI** (arXiv:2605.18796, 2026) | Isotonic-regression calibration of token-margin uncertainty → error probability; threshold by constrained cost minimization | **−31% cost (95% CI [27,35]) at micro-F1 0.91** on 75,000 real production NER queries on H100s; ECE 0.12 → 0.03; beats entropy (2.31), conformal (2.18), FrugalGPT-style (2.24) at cost 2.08 | High — *real production workload, measured latency, end-to-end* |
| **Conformal Cascade** (arXiv:2607.25018, 2026) | Split conformal prediction per tier; defer when prediction set size > 1 | Distribution-free coverage guarantee at each tier; **α reads directly as the user's error budget**; expected cost is closed-form in α. Caveat: at α ≤ 0.20 on some benchmarks sets saturate → 100% deferral | High |
| **CP-Router** (AAAI 2026, arXiv:2505.19970) | Conformal prediction set size as the uncertainty proxy; FBE metric auto-selects α | Training-free, model-agnostic; reduces tokens vs LRM-only while matching/exceeding accuracy; LLM↔**LRM** routing | High |

### Strengths
- No pre-training needed for the *routing* decision (only for the quality estimator).
- Uses response information → strictly better signal than query-only.
- Naturally handles distribution shift: the cheap model's *own* confidence adapts.
- **Cascade routing provably dominates pure routing and pure cascading** when the quality estimator is calibrated (ETH).

### Limitations
- **Latency doubles on escalation** — the query pays for both models. channel.tel: at 25% escalation,
  escalated calls pay ~1.5× latency.
- **LLM confidence is miscalibrated** (UCCI, Conformal Cascade both open with this). Raw entropy
  thresholding is the weakest baseline measured.
- Needs a held-out labelled split for calibration, per model pair and per domain.
- FrugalGPT's 98% required a **domain-trained estimator on a specific enterprise workload**; the same
  source notes untrained implementations typically get 60–80%.

### Cost/quality tradeoff
The highest ceiling of any family (up to 98%) and the widest variance. The quality estimator *is* the system.

### Real-world use
Support/CX agents, RAG pipelines, NER/extraction at scale, translation QE-deferral, reasoning-model
gating (LLM vs LRM).

---

## 3.5 HYBRID / COMPOSITIONAL ROUTING ← the 2026 production consensus

### Definition & mechanism
Layer cheap deterministic signals first, escalate through progressively more expensive decision makers,
and compose multiple signals with an explicit policy. The 2026 survey (arXiv:2603.04445) states it
directly: *"practical systems are often compositional, integrating multiple paradigms under operational
constraints."*

### The canonical production pattern (repeated independently across ≥4 sources)

```
Layer 1  Rule gate          <1 ms    hard exclusions, modality, tenant pin, cache eligibility
Layer 2  Semantic/embedding 5–20 ms  nearest-route by cosine similarity + threshold + margin
Layer 3  Small classifier   10–100ms difficulty / best-model score
Layer 4  LLM classifier     200–800ms only for the genuinely ambiguous residual
Layer 5  Cascade/escalation +1 model turn  post-generation quality check → escalate
```
Sources: 123ofai.com agent-router guide (2026-02-07, with a working `CascadingRouter` implementation:
rules → semantic → LLM → fallback, with circuit-breaker guidance); channel.tel (2026-07-04);
ai-tldr.dev (2026-06-12); sureprompts.com (2026-04-22).

### vLLM Semantic Router = the reference hybrid architecture (arXiv:2603.04444)
Three layers: **signal extraction** (13 signal types, heuristic + neural, computed on demand and in
parallel) → **Boolean decision evaluation** (`d* = argmax conf(d, S(r))`) → **per-decision plugin
chains** (safety, caching, memory). Different deployment scenarios (multi-cloud, privacy-regulated,
cost-optimized, latency-sensitive) are **different configurations over the same architecture, no code
change**. They call this "hybrid embedding": extraction spans sub-millisecond heuristics to LoRA
classifiers, "yet all project onto the same interpretable coordinate system."

### Strengths
- Pays for the decision it needs: most traffic resolves at Layer 1–2.
- Handles novel queries (they fall through to the expensive layer rather than being misrouted cheaply).
- Auditable *and* adaptive.

### Limitations
- Most complex to build, tune, and debug (mindstudio names "debugging becomes more difficult with
  multiple layers" and "monitoring needs to track multiple routing paths").
- Requires thresholds at every boundary → more calibration surface, more drift.

### Real-world use
This is what LiteLLM+RouteLLM, Portkey, and vLLM-SR deployments actually look like in production.

---

## 3.6 EMERGING / SPECIALIZED METHODS

> **Terminology bridge.** The user's brief used the labels *classification-based* and *similarity-based*
> routing. Those are not separate families in the 2026 literature — they are aliases for methods covered
> above, and I map them explicitly so nothing is lost:
>
> | Brief's label | Where it lives here | Methods |
> |---|---|---|
> | **Classification-based routing** | §3.2 (SLM/lightweight classifier) | BERT / ModernBERT+LoRA classifiers, MLP router, SVM router, matrix-factorization win-rate model, RouteLLM `bert` + `causal_llm`, RoRF random forest, difficulty classifiers (Morph-style easy/medium/hard) |
> | **Similarity-based routing** | §3.2 and §3.6(a) | Embedding-similarity / semantic routing (Aurelio `semantic-router`, SemRoute, cosine-to-utterance), KNN router (RouterBench baseline, pulzeai `knn-router`), RouteLLM `sw_ranking` (similarity-weighted Elo), 1NN router (arXiv:2502.00409 — reported to perform *worse than random*), and clustering-based routing (Avengers-Pro), which is similarity at cluster granularity |
> | **Cost-optimization techniques** | §4 (`04-cost-benchmarks-eval.md`) + §3.4 + §3.6(b) | FrugalGPT cascade, cost-aware selection, CARROT (minimax cost-aware rate-optimal), budget-constrained bandits (PILOT/WISERouter/PROTEUS), prompt + semantic caching, and the honest savings ladder |
>
> One negative result worth recording on the similarity side: the extended survey arXiv:2502.00409
> reports that a plain **1NN router "fails to capture complex relationships between user queries and
> expert answers, and performs worse than randomly selecting from available models"** — naive
> similarity is not enough; clustering or learned scoring is.

### (a) Clustering-based, training-free routing — **the 2026 sleeper hit**
**Avengers / Avengers-Pro** (Zhang et al.; AAAI 2026 and CIKM '25, code `ZhangYiqun018/AvengersPro`):
embed queries → k-means cluster (k=64) → build per-cluster **capability profiles** of each model on a
validation set → route to the top-scoring model for the nearest cluster. Optional top-n + voting.
- **No neural network training. One hyperparameter (k).** Robust across embedding models, clustering
  algorithms, and ensemble strategies.
- Avengers-Pro: **+7.1% accuracy over GPT-5-medium at comparable cost**; **−27% cost at matched
  accuracy**; −63% cost at 90% of GPT-5-medium; −81% vs Gemini-2.5-pro; −92% vs Claude-4.1-opus.
  Achieves a **Pareto frontier** (ParetoDist ≈ 0).
- **New model onboarding is incremental**: evaluate the new model on the existing validation set and
  compute its cluster profile — no re-clustering, no retraining.
- LLMRouterBench (arXiv:2601.07206, Jan 2026) independently confirms: *"Top routing methods are
  comparable, but can be free of neural network training"*; **up to 4% accuracy gain over Best Single
  and up to 31.7% cost reduction at matched performance**; "Avengers-Pro nearly dominates the frontier";
  and critically — **"embedding models have little influence on routing performance"** and adding more
  models yields **diminishing returns**.
- **Design implication: a training-free clustering router should be a first-class shipped strategy, not
  an afterthought.** It removes the checkpoint-rot failure mode entirely.

### (b) Bandit / online-learning routing
| Method | Venue | Mechanism | Result |
|---|---|---|---|
| **PILOT** | EMNLP 2025 Findings, arXiv:2508.21141 (Fujitsu) | LinUCB + preference-prior on a shared query↔LLM embedding space; budget as an **online multi-choice knapsack** | **93% of GPT-4 at 25% of cost on RouterBench**; 86% at 27% on single-task MMLU. Proves preference-prior gives a tighter regret bound. **Code not public** (per WISERouter) |
| **MixLLM** | NAACL 2025 | Contextual bandit + per-LLM quality/cost regressors + latency penalty + policy gradient on thumbs up/down | 97.25% GPT-4 quality @ 24.18% cost; 98.55% @ 16.79% with Llama 3.1; OOD drop mitigated by 3.35% via online learning; **"Top-3" policy surpasses GPT-4 at ~20% cost** |
| **MetaLLM** | 2024 | Single bandit, quality-cost reward | Limited scalability when adding/removing LLMs |
| **PROTEUS** | arXiv 2026 | Lagrangian RL: minimize cost s.t. accuracy floor τ | Freezes dual variables at deployment (identified gap) |
| **WISERouter** | arXiv:2607.23765 (2026) | UCB with a cost scaling factor under a **workload** (not per-query) budget | Notes per-query constraints are "myopic" and require retraining to change budget |
| **arXiv:2604.00136** (2026-03) | — | Closed-loop budget pacing over an open-ended stream + hot-swap model registry + forced-exploration onboarding | **9.8 ms end-to-end routing on CPU; the decision itself 22.5 µs**; never exceeds ceiling by >4%; under silent quality degradation baselines without closed-loop cost control overshoot by **up to 6.9×** |
| **BaRP / LLM-Bandit** | — | Bandit feedback routers | Optimize preference-conditioned reward, not explicit dollar constraints |

> **The single most under-served production requirement in the whole literature is closed-loop budget
> pacing.** arXiv:2604.00136 names it precisely: PILOT assumes a known horizon, PROTEUS freezes duals,
> and nothing ships a pacer that adapts to price changes and silent quality regression.

### (c) Uncertainty / calibrated-confidence routing
See §3.4 table (UCCI, Conformal Cascade, CP-Router) plus:
- **Self-REF** (ICML 2025) — trains LLMs to emit **confidence tokens** that trigger escalation.
- **Confident or Seek Stronger** (NeurIPS 2025 Workshop) — benchmarks uncertainty-driven routing from on-device SLMs to stronger LLMs.
- **Uncertainty-Based Two-Tier Selection** (COLM 2024, arXiv:2405.02134).
- **Key insight:** raw confidence is miscalibrated; you must calibrate (isotonic regression or conformal)
  before thresholding. UCCI reduced ECE 0.12 → 0.03 and that alone bought the win.

### (d) Graph-based routing
- **GraphRouter** (ICLR 2025): heterogeneous graph of task/query/LLM nodes; routing as **edge
  prediction**; ≥12.28% Reward gain, ≥88.89% of oracle, **generalizes to new LLMs zero-shot**.
- **PersonalizedRouter** (TMLR, Nov 2025): adds **user nodes** → personalized model selection.
- **GraphPlanner** (OpenReview): graph memory-augmented multi-round routing.
- **GraphRAG-Router** (2026): two-stage RL over GraphRAG variants + LLMs.

### (e) Reasoning-mode routing (the newest axis)
Route not *which model* but *whether to think*. vLLM-SR (arXiv:2510.08731): +10.2 pp accuracy,
−47.1% latency, −48.5% tokens on MMLU-Pro. Also **RADAR** (reasoning-ability & difficulty-aware,
NeurIPS 2025 WS), **X-Router** (Findings ACL 2026). ai-tldr.dev frames it as a new routing dimension:
**thinking budget**.

### (f) Specialized / niche
| Method | Note |
|---|---|
| **IRT-Router** (ACL 2025) | Item Response Theory — interpretable ability/difficulty/discrimination parameters |
| **RouterDC** | Dual contrastive learning; query & LLM embeddings in a shared space |
| **EmbedLLM** (2024) | Matrix factorization producing "model embeddings" capturing LLM characteristics |
| **CARROT** (ICLR 2025 WS, arXiv:2502.03261) | Cost-Aware Rate-Optimal Router; ~35% cost cut at <2% accuracy loss on RouterArena |
| **Zooter** (NAACL 2024) | Reward-guided ensemble routing via distilled rewards |
| **MODEL-SAT** (2025) | Encodes model capability profiles; a lightweight LLM predicts the best candidate |
| **Causal LLM Routing** (NeurIPS 2025, arXiv:2505.16037) | End-to-end regret minimization from **observational** data |
| **R2-Router** (ICML 2026, arXiv:2602.02823) | Routing *with* reasoning; −84.46% API cost (secondary source) |
| **MasRouter** (ACL 2025) | Routing for multi-agent systems |
| **OmniRouter** (KDD 2025) | Budget- and performance-controllable multi-LLM routing; **+6.30% accuracy and ≥10.15% lower cost** vs router baselines, on *long-horizon agent* tasks |
| **DiSRouter** (ICLR 2026) | Distributed self-routing |
| **ACAR** (2026) | Adaptive complexity routing with **auditable decision traces** |
| **GreenServ** (2026) | Energy-efficient context-aware dynamic routing |
| **Ant Colony Optimization routing** (2026) | Interpretable multi-agent routing |
| **Federate the Router** (2026) | Learning routers from sparse, decentralized evaluations |
| **Task/Session-Level Routing** (arXiv:2608.14641, 2026) | Evaluates 4 OSS routers across RouterBench, BFCL v4, tau2-bench, WebArena; **routing effect was statistically inconclusive or equivalent on 3 of 4** agent benchmarks |

> **Sobering agent-routing finding (arXiv:2608.14641, 2026-07-28):** on agentic benchmarks, router-vs-
> blind-tier differences were "inconclusive" on BFCL v4, RouterBench, and tau2-bench, and "equivalent"
> on WebArena (TOST equivalence). OmniRouter exists precisely because "existing router benchmarks
> evaluate routers only on one-shot prompts... never expose the router-visible prefix at an
> intermediate agent step." **If your target users run agents, one-shot benchmarks will mislead you.**
