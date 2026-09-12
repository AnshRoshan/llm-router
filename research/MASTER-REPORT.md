# LLM Routing: Comprehensive Research Dossier
### Design input for a production-grade LLM router Python package

**Research date:** 2026-09-12 (Asia/Calcutta)
**Scope:** ~34 web investigations + 14 primary documents read directly this session
**Companion files:** `01-routellm-deep-dive.md` · `02-frameworks-landscape.md` · `03-technique-taxonomy.md` · `04-cost-benchmarks-eval.md` · `05-architecture-patterns.md` · `06-design-recommendations.md` · **`07-workload-and-deployment-classes.md`** · **`08-session-stickiness-and-entry-intent.md`** · **`09-ttl-tier-policy.md`** · **`10-fusion-delegation-architecture.md`** · `00-RESEARCH-LOG.md` · `summary-table.csv`

**Every URL and citation is in `SOURCES.md`** — 212+ unique URLs and 70+ arXiv IDs, sorted into
primary / secondary / vendor / unverified tiers with dates.

---

## ⚠ READING ORDER: `07` → `08` → `09` → `10`

These four threads were written after the initial pass and each revises what came before. Read them in
order; where they conflict with `01`–`06`, **they win**.

| Thread | What it settles |
|---|---|
| `07` | The two axes (workload × deployment) and the KV-cache finding. **Rule B corrected by `08`.** |
| `08` | Session stickiness, the two entry paths, the `R*` switch algebra. **§8.7(3) corrected by `09`.** |
| `09` | TTL tier per breakpoint. Answers `08` Q4. |
| `10` | Cognition Fusion: **delegation beats routing**, and cost is per *pair*, not per model. Promotes delegation P1 → P0 and partially falsifies the cost model in `06`. |

Scope was revised after the initial pass. The product routes across **any workload class**
(agent / chat / RAG / batch / code / vision) **and any deployment class** (self-hosted / API /
serverless). Thread 7 covers those two axes and contains the session's most consequential finding:

> **Prompt caches are model-scoped. Switching model mid-session throws away the entire cache you paid
> to build** — measured at ~85,113 duplicated cache-creation tokens in one Claude Code session, against
> a **10× price gap** between cached ($0.30/M) and uncached ($3.00/M) input on Claude Sonnet.
> **Naive per-step agent routing is therefore often net-negative on cost.** Session affinity via
> consistent hashing must be the default, and `switch_cost` must be a term in the scoring function.
> This is also why scoring must be per **(model, backend) pair**, not per model.

---

## Executive summary — the eleven findings that should change your design

1. **"Gateway" and "router" are different products, and the market needs both.** A gateway routes by
   rules you declare (deterministic, auditable). A router *predicts* which model is good enough
   (learned, has a quality tail). RouteLLM is explicitly "routing decision only, not a full gateway."
   **Your wedge: the decision + learning layers, delegating execution.**

2. **The honest savings number is ~30–40%, not 85%.** RouteLLM's 85% and FrugalGPT's 98% are real,
   peer-reviewed, and *benchmark-specific ceilings*. The only neutral third-party benchmark (RouterArena,
   ICLR 2026) found the best routers achieve **~35% cost cut at <2% accuracy loss**. The most
   production-realistic measurement in the entire corpus — UCCI on 75,000 real production NER queries on
   H100s — got **31%**. Microsoft's own first-party managed router, measured on real prompts, delivered
   **4.5–14.2%**.

3. **A training-free clustering router is the highest value-per-effort strategy available.**
   Avengers-Pro (AAAI 2026 / CIKM) does embed → k-means → per-cluster capability profiles. No neural
   training, **one hyperparameter**, incremental new-model onboarding, **Pareto-optimal** in the neutral
   LLMRouterBench, **+7.1% accuracy over GPT-5-medium at comparable cost** and **−27% cost at matched
   accuracy**. LLMRouterBench independently found "embedding models have little influence on routing
   performance." **This removes the checkpoint-rot failure mode entirely.**

4. **Checkpoint rot is the #1 silent killer.** RouteLLM's repo has had no commit since 2024-08-10. Its
   2024-era checkpoints now rank **#25 of 26** on the live RouterArena leaderboard (accuracy 47%) despite
   having best-in-class robustness (100.0) and near-lowest cost ($0.27/1K). **Ship a `fit` command that
   retrains in minutes; never ship frozen weights as the default.**

5. **LLM-as-router is an anti-pattern in production.** Four independent 2026 sources converge:
   1–5 s latency, a second inference bill, "another point of failure," and "general models are mediocre
   routers." Its defensible uses are as the *last* cascade tier and as an **offline label generator**
   (which is exactly how RouteLLM uses its GPT-4 judge).

6. **The winning production architecture is Signal → Policy → Plugin chain, not "a smarter classifier."**
   vLLM Semantic Router (arXiv:2603.04444, 32 authors) composes **13 signal types** — sub-millisecond
   heuristics and LoRA neural classifiers — through Boolean decision rules, evaluated **on demand and in
   parallel**. Different deployments are different *configs*, not different code.

7. **Calibration is the actual product.** Uncalibrated confidence is the weakest baseline measured
   everywhere. UCCI's isotonic regression cut ECE from 0.12 → 0.03 and that alone bought the 31%.
   Conformal prediction turns α into a literal **user-facing error budget**. Ship isotonic + conformal.

8. **Design for 3–10 candidate models.** RouterEval (EMNLP 2025, 8,500+ LLMs, 200M records):
   "cost-effectiveness of this paradigm is highest when there are approximately 3 to 10 LLM candidates."
   RouteLLM's binary-only design leaves money on the table. LLMRouterBench: adding models past that
   yields diminishing returns.

9. **Nobody ships closed-loop budget pacing — the single biggest unmet production need.**
   arXiv:2604.00136 (2026) names it precisely: PILOT assumes a known horizon, PROTEUS freezes its dual
   variables at deployment, and **baselines without closed-loop cost control overshoot by up to 6.9×**
   under silent quality degradation. That same system hits **9.8 ms end-to-end routing on CPU, 22.5 µs
   for the decision itself** — the latency bar to aim at.

10. **There is a direct competitor, and it is strong — but it has a clear gap.**
    **`llmrouter-lib` 0.4.0** (ulab-uiuc, 2026-08-14, MIT, 1.3k+ stars) already ships 16+ routers, a
    unified CLI, Gradio/ComfyUI UIs, a plugin system, and the xRouteBench benchmark. What it does *not*
    ship is the production execution layer: fallback chains, circuit breakers, per-route quality
    monitoring, budget pacing, Redis-shared state, OTel. **That gap is the product.**

11. **Delegation beats routing — and it invalidates our per-token cost model.** *(added after
    `10-fusion-delegation-architecture.md`)* Cognition's **Fusion** architecture runs a frontier
    **lead** and a cheap **sidekick** as two agents with **two separate warm caches**, exchanging only
    *briefs*. Because neither prefix ever changes for the other's sake, it captures the savings of
    routing **without ever paying a cache write** — the exact cost that makes ordinary mid-session
    switching self-defeating. Reported **−23% to −46% cost at roughly equal score**.
    They reject routing for two reasons we had independently reached: *"the initial prompt isn't enough
    to know the difficulty of the task"*, and *"you break prompt caches by switching models mid-task."*
    Two consequences that change the design:
    - **Route on task *phase*, not difficulty.** Phase (plan/setup/implement/debug/validate) is
      observable from tool-call history; difficulty is not observable at all. Their FrontierCode split:
      Plan 32%, Setup 34%, **Implementation only 5%**, Debug 17%, Validate 9%.
    - **Cost is not a per-model property.** A sidekick costing **275% more per token** made the system
      **2% cheaper** — fewer attempts, fewer review rounds. *"Models should be evaluated on price per
      task rather than price per token."* Cost belongs to the **(lead, sidekick) pair**, which a
      model-keyed price registry cannot express.

    **Caveat:** vendor-reported, coding-benchmarks only, and the comparison is harness-vs-harness, so
    architecture and harness are confounded. Against Astra, Fusion was **materially worse** on
    Terminal-Bench 4 (55.6 → 50.0). Do not generalise the 39% headline.

---

## Master comparative summary table — every technique

| Technique | How it decides | Latency | Reported cost reduction | Quality | Training data needed | N models | Prod-readiness | Best reference |
|---|---|---|---|---|---|---|---|---|
| **Rule / metadata** | keyword, regex, length, modality, tenant, role | **<1 ms** | n/a (enabler) | n/a — ceiling = rule quality | None | any | ★★★★★ | vLLM-SR signal layer; LiteLLM pre-call checks |
| **Cost-aware selection** | cheapest model meeting constraints | <1 ms | 30–60% (naive) | Unmodeled ⚠ | None (needs price table) | any | ★★★★★ | LiteLLM `cost-based-routing` |
| **Embedding similarity** | cosine to route utterances + threshold | **5–20 ms** | varies by domain split | Good (34.85%→89.39% after `fit`) | 5–15 utterances/route | any | ★★★★☆ | aurelio-labs/semantic-router |
| **Clustering (training-free)** | k-means cluster → per-cluster capability profile | embed pass | **27–31.7% at matched acc; +7.1% acc** | Matches or beats best single model | Validation set only (no training) | any | ★★★★☆ | **Avengers-Pro** (AAAI 2026) |
| **Matrix factorization** | MF on preference data → win rate | low | **85% MT-Bench @95% GPT-4** (2–3.66× defensible) | 95% GPT-4 on MT-Bench; 47% on RouterArena '26 | Preference pairs | 2 (orig) | ★★★☆☆ | RouteLLM `mf` |
| **BERT / ModernBERT classifier** | fine-tuned encoder → difficulty/model score | **10–100 ms** | 45% MMLU; −48.5% tokens (reasoning gate) | 98.53% routing acc w/ 805 examples | Labeled outcomes | any | ★★★★☆ | vLLM-SR (LoRA, ONNX) |
| **Random forest on embeddings** | RF over query embedding | ~100–150 ms hosted | 12.5% vs Sonnet; 20–40% (vendor) | Beats RouteLLM on MMLU/BBH (their claim) | Scored pairs (≥15) | 2 | ★★★★☆ | Not-Diamond/RoRF |
| **KNN router** | best model among k nearest training queries | low | RouterArena: $300/1K queries ⚠ | 69% acc, worst cost | Outcome matrix | any | ★★☆☆☆ | RouterBench baseline |
| **MLP router** | regression on query features | low | RouterArena: $287/1K ⚠ | 65% acc | Outcome matrix | any | ★★☆☆☆ | RouterBench baseline |
| **IRT-Router** | Item Response Theory: ability × difficulty × discrimination | ~27 ms | Can hit **5× optimal cost** ⚠ | RouterArena rank #3 | Outcome matrix | any | ★★★☆☆ | IRT-Router (ACL 2025) |
| **Graph (GNN) router** | heterogeneous graph edge prediction | graph pass | −99% onboarding time cost | ≥88.89% of oracle; **zero-shot to new LLMs** | Interaction data | any | ★★★☆☆ | GraphRouter (ICLR 2025) |
| **Dual contrastive (RouterDC)** | shared query↔LLM embedding space | ~10 ms | Best cost-ratio score | Worst accuracy (32%) | Pairs | any | ★★☆☆☆ | RouterDC |
| **Cascade + confidence** | cheap model → score output → escalate | **+1 model turn on escalation** | **50–98%** (FrugalGPT); **31%** real production | ECE 0.03 after calibration | Labelled calibration split | 2–K | ★★★★☆ | FrugalGPT; UCCI; Conformal Cascade |
| **Cascade routing (unified)** | optimal mix of route + cascade per step | variable | Provably dominates both | Requires calibrated quality estimator | Same as cascade | 2–K | ★★★☆☆ | ETH arXiv:2410.10347 |
| **Bandit / online (PILOT)** | LinUCB + preference prior + knapsack budget | ~ms | **75%** (25% of GPT-4 cost @93%) | Adapts to drift; +3.35% OOD | Online feedback | any | ★★☆☆☆ (code not public) | PILOT (EMNLP 2025) |
| **Bandit + latency (MixLLM)** | per-LLM quality/cost regressors + latency penalty + PG | ~ms | **76–83%** (24.18%→16.79% of cost) | 97.25–98.55% of GPT-4 | Online feedback | any | ★★☆☆☆ | MixLLM (NAACL 2025) |
| **Uncertainty / conformal** | prediction-set size or calibrated error prob | +calibration pass | 31% | Distribution-free coverage guarantee | Calibration set | 2–K | ★★★★☆ | CP-Router (AAAI 2026); UCCI |
| **Hidden-state probe** | internal activations, cross-layer Dirichlet | needs model internals | — | +16.68% router ability | Multi-domain data | 2 | ★★☆☆☆ | ProbeDirichlet (2026-02) |
| **LLM-as-classifier** | prompt a cheap LLM to pick | **200–2000 ms** ⚠ | Often negative after overhead | 85–90% acc | Prompt eng. | any | ★★☆☆☆ | Anti-pattern; last-tier only |
| **Hybrid / compositional** | layered rules → embedding → classifier → LLM → cascade | 1 ms – 800 ms | 40–85% (mix-dependent) | Best of each layer | Per-layer | any | ★★★★★ | **vLLM Semantic Router** |
| **Reasoning-mode routing** | think vs. don't-think | ~ms | **−48.5% tokens, −47.1% latency** | **+10.2 pp accuracy** | Small labeled set | 2 modes | ★★★★☆ | vLLM-SR (arXiv:2510.08731) |

★ = production-readiness judgment based on the evidence gathered, not a published rating.

> **Machine-readable version:** `summary-table.csv` — 21 techniques × 10 fields, including an
> `implementation_difficulty` column (Very low → High) that is my assessment derived from the training-data
> and infrastructure requirements documented per technique in `03-technique-taxonomy.md`.
> `production_readiness` is a 1–5 judgment on the same basis; neither is a published rating.

---

## Answers to the five research questions

### Q1 — Existing frameworks

**Learned routers (OSS):** RouteLLM (Apache-2.0, stale), **LLMRouter/llmrouter-lib** (MIT, 0.4.0,
2026-08, the strongest), semantic-router (MIT), RoRF (Not Diamond), Avengers-Pro, CARROT, GraphRouter,
IRT-Router, local-llm-router (zero-dep), llm-router (ypollak2).

**Gateways:** LiteLLM (MIT, 57k★, v1.98.0), Portkey (OSS core + cloud), OpenRouter (hosted, 5.5% fee),
Cloudflare AI Gateway, Kong AI Gateway, Helicone, Vercel AI Gateway, **SGLang Model Gateway** (the
production-quality reference: 40+ Prometheus metrics, OTel, circuit breakers, token-bucket rate limiting).

**Managed routers:** Not Diamond, Martian (closed), Unify, **AWS Bedrock Intelligent Prompt Routing**
(~85 ms P90 overhead, same-family only, 30–56% savings), **Microsoft Foundry Model Router**
(4.5–14.2% measured), Morph Router ($0.001/classification, ~430 ms), Inworld Router, Neutrino, Requesty.

**Benchmarks:** RouterBench (405k outcomes), **RouterArena** (ICLR 2026 + live leaderboard),
RouterEval (200M records), LLMRouterBench, RouterXBench, xRouteBench, OmniRouter.

Full detail with sources: `02-frameworks-landscape.md`, `06-design-recommendations.md` §6.1.

### Q2 — Technique details
Full treatment with per-technique accuracy, latency, cost, difficulty and use cases: `03-technique-taxonomy.md`.

### Q3 — Architectural patterns
- **Backends:** abstract `ModelBackend` protocol + OpenAI-compatible default covers ~90%. RouteLLM
  delegates to LiteLLM; vLLM-SR uses Envoy `ext_proc` for transparent interception.
- **Fallback:** three separate channels (general / content-policy / context-window) + `max_fallbacks`
  cap + per-exception retry policy + **per-deployment** circuit breaker + pre-call context validation +
  **parameter stripping on model switch** (the bug everyone hits).
- **Distributed:** Redis for shared cooldowns/rate-limits/latency-windows (**required above 1 replica**),
  Postgres for keys/spend/audit, stateless ASGI replicas, HPA on queue depth + CPU.
- **Quality-while-cutting-cost:** per-route (not aggregate) quality monitoring, quality floors the
  router may not cross, routing-mix drift alarms, CI eval gate, calibration refresh on model-version change.
- Full detail: `05-architecture-patterns.md`.

### Q4 — Python packages as foundations or alternatives
**Do not fork RouteLLM** (stale, binary-only, numpy<2 pinned). **Study LLMRouter** — it has already
solved the multi-strategy plugin problem well; differentiate on the production execution layer and the
calibration/monitoring loop. **Consider semantic-router as a dependency** for the embedding-similarity
layer rather than reimplementing it. **Do not compete with SGLang/LiteLLM on gateway throughput** —
LiteLLM measures ~40–50 ms and ~200 req/s in Python vs <1 ms and 10K QPS for Rust gateways; a Python
package wins on decision quality, not on proxy speed. Detail: `06-design-recommendations.md`.

### Q5 — Quality benchmarks and evaluation methodology
Adopt: **AIQ** (RouterBench), **Arena Score** (weighted harmonic mean of accuracy and log2 cost),
**AUROC of the routing decision** (threshold-independent, RouterXBench), **deferral curve**,
**ParetoDist**, **oracle gap**, **difficulty-stratified accuracy**, **surface-perturbation robustness**,
**ID/OOD split**. Plus the operational set: routing mix, per-route quality, escalation rate,
cost-vs-frontier-only, ECE, routing latency p50/p95, classifier overhead, cache hit rate.
Detail and metric definitions: `04-cost-benchmarks-eval.md`.

---

## Recommendations — what to build

### The product, in one sentence
A Python library that combines a **pluggable multi-strategy decision layer** (N models, signal→policy→select),
a **production execution layer** (fallback chains, circuit breakers, budget pacing), and an
**evaluation + calibration loop** (fit/calibrate/eval CLI, per-route monitoring, CI gate) — with
**zero mandatory heavy dependencies**.

### Strategy priority (evidence-backed)
| P | Strategy | Why |
|---|---|---|
| **P0** | Rules/metadata gate | Free, deterministic, the only way to do hard exclusions |
| **P0** | Cost-aware selection | Table stakes; needs a priced model registry |
| **P0** | **Training-free clustering** (Avengers-Pro) | Best value/effort; no checkpoint rot; Pareto-optimal |
| **P0** | **Cascade + isotonic/conformal calibration** | Highest ceiling; 31% on real production traffic |
| **P1** | ONNX small classifier (ModernBERT+LoRA) | Best quality-per-ms; optional extra, CPU-friendly |
| **P1** | Embedding-similarity | Zero-training cold start; consider depending on semantic-router |
| **P1** | Matrix factorization on preference data | Your own, trained on current data — not RouteLLM's 2024 weights |
| **P2** | Bandit / online adaptation, graph router | Differentiation; hardest problems |
| **P3** | LLM-as-classifier | Last-tier + offline labeling only |

### Non-negotiables
- **Never ship frozen checkpoints as the default.** `fit` must work in minutes on the user's prompts.
- **Never claim 85%.** Ship the tool that measures *their* number; quote the neutral ~35% / production ~31%.
- **Zero mandatory dependencies.** torch, LiteLLM, GPU, vector DBs are all optional extras.
- **Router decision budget: <10 ms p50.** Enforce it in tests. (arXiv:2604.00136 hits 22.5 µs.)
- **`explain` on every decision.** Every signal, every candidate score, the threshold, the reason.
- **async-first.** Sync wrappers only.
- **Apache-2.0.**
- **Ship the eval harness before the fancy routers.** It is the moat and the only defence against
  benchmark-washing.

### The three differentiators no one has
1. **Closed-loop budget pacing** with hot-swap model registry and forced-exploration onboarding.
2. **Calibration as a first-class CLI** (isotonic + conformal, with α as a user-facing error budget).
3. **Misroute → training-example pipeline**: every routing decision logged in a schema that feeds the
   next `fit`. This is the loop that stops checkpoint rot, which is what killed RouteLLM.

---

## Confidence and source-quality notes

**High confidence (primary sources read directly this session):** RouteLLM README + repo metadata;
arXiv:2406.18665 (RouteLLM); arXiv:2603.04445 (routing/cascading survey, v2 2026-04-21);
arXiv:2603.04444 (vLLM Semantic Router, v4 2026-06-03); arXiv:2510.08731 (When to Reason);
arXiv:2510.00202 / ICLR 2026 proceedings + `RouteWorks/RouterArena` README (RouterArena);
arXiv:2502.18482 / ACL Anthology (MixLLM, NAACL 2025); arXiv:2508.21141 (PILOT, EMNLP 2025 Findings);
arXiv:2305.05176 PDF (FrugalGPT); arXiv:2505.19970 + AAAI (CP-Router); arXiv:2605.18796 (UCCI);
arXiv:2607.25018 (Conformal Cascade); arXiv:2502.03261 (CARROT); arXiv:2602.11877 (RouterXBench);
arXiv:2601.07206 (LLMRouterBench); arXiv:2508.12631 + ACM DL (Avengers-Pro); arXiv:2505.19797 + AAAI (Avengers);
arXiv:2410.03834 (GraphRouter); `docs.litellm.ai/docs/routing` + `/config_settings`;
`raw.githubusercontent.com/lm-sys/RouteLLM/main/README.md`; `ulab-uiuc/LLMRouter` README + PyPI JSON;
`Not-Diamond/awesome-ai-model-routing`; `MilkThink-Lab/Awesome-Routing-LLMs`; sglang docs.

**Medium confidence (single reputable secondary source, primary not read):** BEST-Route
(arXiv:2506.22716 — cited only via getnadir.com tutorial); Microsoft Foundry Model Router savings
(Microsoft TechCommunity blog, first-party but a 10-prompt demo); AWS Bedrock IPR savings figures
(AWS-published but relayed, and **the pricing is contradictory across sources**: $1/1,000 requests vs
no separate fee — unresolved); gateway latency/throughput comparison (getmaxim.ai, vendor-authored);
Morph Router figures (vendor).

**Low confidence / flagged:** `local-llm-router` and `ai-llm-router` (PyPI pages only, no repo audit);
Martian's "up to 97%" (closed, no primary docs found); the "60–75% typical savings" figures circulating
in marketing content (no neutral basis found).

**Explicitly unresolved:** (1) Bedrock IPR pricing model. (2) Whether LLMRouterBench's claim that "most
routing methods collapse to similar performance" under fair evaluation generalizes beyond its own
benchmark suite — I read the finding but not a replication. (3) Whether agentic-workload routing works
at all: arXiv:2608.14641 found effects "inconclusive" on 3 of 4 agent benchmarks, which directly
conflicts with OmniRouter's +6.30%/−10.15% claim. **Both are 2025–2026 papers; treat agentic routing as
unproven and measure it yourself.**
