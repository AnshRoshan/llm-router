# Thread 4 — Cost Economics, Benchmarks & Evaluation Methodology

Research date: 2026-09-12.

## 4.1 The honest savings ladder (most important table in this dossier)

Every number below is traceable. Read top (best case) to bottom (worst case) — the spread *is* the finding.

| Reported saving | Method | Benchmark / workload | Who measured | Realistic? |
|---|---|---|---|---|
| **98%** | FrugalGPT cascade | HEADLINES (financial news), domain-trained scorer | Stanford, TMLR 2024 | **Ceiling, not expectation.** Paper's own Table 3 range is **50–98%**. Requires a domain-trained quality estimator. |
| **92%** | Avengers-Pro | vs Claude-4.1-opus at matched accuracy | Authors (CIKM '25) | Benchmark-specific model pairing |
| **85%** | RouteLLM `mf` | MT-Bench, 95% GPT-4 quality, 14% strong-model calls (with GPT-4-judge augmentation) | LMSYS, ICLR 2025 | **Benchmark-specific.** Without augmentation it's 26% strong calls (~74%). |
| **81% / 63%** | Avengers-Pro | vs Gemini-2.5-pro / at 90% of GPT-5-medium | Authors | Benchmark-specific |
| **76% / 83%** | MixLLM | 24.18% / 16.79% of GPT-4 cost at 97.25% / 98.55% quality | NAACL 2025, RouterBench | Peer-reviewed, on RouterBench |
| **75%** | PILOT | 25% of GPT-4 cost at 93% quality | EMNLP 2025, RouterBench | Peer-reviewed |
| **65%** | Bedrock Intelligent Prompt Routing | Real RAG workload, ~70/30 easy/hard split | Reddit r/aws hands-on, 2026-06 | Independent, single anecdote |
| **56% / 35% / 16%** | Bedrock IPR | Anthropic family / Nova / Meta — **within-family only** | AWS-published (via beri.net) | Vendor |
| **45%** | RouteLLM `bert` | MMLU at 92% GPT-4 quality | LMSYS | Benchmark-specific |
| **37% / 42%** | UCCI | At cost ratios 5× / 10× between models | arXiv:2605.18796 | Real production NER, measured |
| **35%** | RouteLLM `mf` | GSM8K at 87% GPT-4 quality | LMSYS | Benchmark-specific |
| **~35% at <2% accuracy loss** | vLLM-SR, CARROT | RouterArena, 12 routers, common footing | **Neutral third party**, ICLR 2026 | **Best neutral estimate available** |
| **31.7%** | Avengers-Pro | Matched Best-Single accuracy | LLMRouterBench (arXiv:2601.07206) | Neutral benchmark |
| **31%** | UCCI | 75,000 real production queries, H100, micro-F1 0.91 | arXiv:2605.18796 | **Most production-realistic number in the corpus** |
| **27%** | Avengers-Pro | Matched GPT-5-medium accuracy, 8 models, 6 benchmarks | Authors | Neutral-ish |
| **20–40%** | Not Diamond | Their own published figure | Vendor | Conservative for a vendor |
| **4.5–14.2%** | Microsoft Foundry Model Router | 10 real prompts vs fixed GPT-5-nano | Microsoft TechCommunity, 2026-02 | **First-party, honest, low** |

### The defensible planning number
> **Budget ~30–40% cost reduction at ≤2% quality loss. Treat 60–85% as achievable only if (a) your
> traffic is genuinely skewed easy, (b) the strong/weak price gap is ≥10×, and (c) you invest in a
> domain-calibrated quality estimator.** This matches the neutral RouterArena finding, LLMRouterBench's
> 31.7%, UCCI's 31% on real production traffic, and beri.net's production anecdote (~35%).

### Why the research numbers are systematically inflated
1. **MT-Bench is easy.** dreaming.press: "MT-Bench [is] a benchmark of the kind of broad, chatty
   questions where a cheap model often *is* good enough. The further your traffic sits from that
   distribution — narrow domain, structured extraction, agent tool-calls where one wrong route breaks a
   chain — the less the win-rate the router learned transfers."
2. **Cost is computed on input tokens only.** digitalapplied.com: "these figures use input-token pricing
   for a clean apples-to-apples comparison; your real bill blends input and output, and output is where
   the spread is widest."
3. **Every vendor figure is self-measured.** RouterArena is the only neutral scorekeeper found.
4. **Support costs eat the saving.** beri.net reports a production account where a team that cut
   inference by 60% "lost an inferred four to five times that saving to support costs and churn within
   three months."

### The arithmetic that actually drives savings
The gap is enormous and that's the whole game:
- Bedrock Nova Micro $0.035/M input vs Claude Opus $15/M input = **~430×**
- GPT-4o is ~16× GPT-4o-mini (neuraltrust)
- Blended-rate example (getnadir.com): 80% at $0.10/M + 20% at $15/M = **$3.08/M vs $15/M = −80%**

Corollary: **if your strong/weak price gap is under ~3×, routing cannot pay for itself.**

## 4.2 Benchmarks — the full landscape

| Benchmark | Scale | What it measures | Venue/date | Code |
|---|---|---|---|---|
| **RouterBench** | **405,467** inference outcomes, **11 models**, 8 datasets, **64 tasks** | Cost-quality frontier offline (pre-generated outputs → no live inference). Metric: **AIQ** (Average Improvement in Quality) + **NDCH** (non-decreasing convex hull). Baselines: **Zero Router** (lower bound) and **Oracle Router** (upper bound) | arXiv:2403.12031, Martian + UC Berkeley (Keutzer lab) + UCSD, Mar 2024. Submitted to ICLR 2024 as "MARS" | `withmartian/routerbench` |
| **RouterArena** | 8,400 queries, 12 routers, **oracle acc 90.89%** | 5 dimensions: Arena Score (weighted harmonic mean of accuracy & log2 cost), Cost-ratio, Optimal-acc, Latency, Robustness. Difficulty stratified empirically by "how many models can answer correctly"; domain coverage via Dewey Decimal Classification, cognitive levels via Bloom's taxonomy | **ICLR 2026** (arXiv:2510.00202, Sep 2025) + **live leaderboard** | `RouteWorks/RouterArena` |
| **RouterEval** | **200M+** performance records, **8,500+ LLMs**, 12 evaluations | **Model-level scaling up** — performance rises rapidly as candidate count grows | Findings of EMNLP 2025, pp. 3860–3887 | — |
| **LLMRouterBench** | Massive, multi-benchmark | Unified framework; introduces **ParetoDist**; findings: "embedding models have little influence on routing performance"; "adding more models yields diminishing returns" | arXiv:2601.07206, Jan 2026 | — |
| **RouterXBench** | 6 benchmarks (Alpaca, MMLU, Big-Math + OOD) | **Triple-perspective**: (i) Router Ability = **AUROC** (threshold-independent!), (ii) Scenario Alignment = **LPM / MPM / HCR** (low-cost / mid / high-accuracy regimes), (iii) Cross-Domain Robustness (ID vs OOD) | arXiv:2602.11877, Feb 2026 | `zhuchichi56/RouterXBench` |
| **MixInstruct** | 11 models, diverse prompts | Instruction-following routing | (survey-cited) | — |
| **SPROUT** | Chat-style prompts, more models | Smart Price-aware ROUTing; extends RouterBench diversity | (survey-cited) | — |
| **OmniRouter dataset** | Long-horizon agent tasks | The only benchmark exposing "the router-visible prefix at an intermediate agent step" | 2025 | `dongyuanjushi/OmniRouter` |

### RouterEval's most actionable finding for package design
> *"The cost-effectiveness of this paradigm is highest when there are approximately **3 to 10 LLM
> candidates**."* Beyond that, performance growth rate falls off while deployment complexity rises.
> **Design your package for a 3–10 model pool, not for 2 and not for 100.**

### RouterArena's most actionable findings
- **Most routers cluster near (100% cost, 100% accuracy)** — they over-rely on the strongest model and
  fail to defer. A router that simply never defers is the default failure mode.
- **Accuracy >89% on easy queries, <10% on hard queries** for most routers. Stratify your eval by difficulty or you will not see this.
- **Robustness is generally low**; BERT-based routers break under surface perturbations (paraphrase, typos).
- **Latency matters and varies 50×**: vLLM-SR 546.8 ms (OpenAI embedding calls) vs most others <100 ms, RouteLLM 259.0 ms.
- **Commercial ≠ better**: GPT-5 ranked #7, NotDiamond #12 (paper snapshot).

## 4.3 Metric definitions to implement (copy these)

**Routing-quality metrics**
- `PGR@X` / "Percent of GPT-4 performance Retained" — RouteLLM's primary metric.
- `CPT(α)` — Cost Per Threshold; cost-savings ratio vs the random router at strong-model share α.
- **AIQ** — Average Improvement in Quality (RouterBench); integrates quality gain over the cost axis.
- **Arena Score** — weighted harmonic mean of accuracy and log2-transformed cost (RouterArena).
  The log2 makes each doubling of price cost one unit — a good, legible choice.
- **AUROC of the routing decision** (RouterXBench) — threshold-independent; the right metric for
  comparing routers *before* you pick an operating point. **This is under-used and should be shipped.**
- **Best Selection Rate / Optimal-acc** — how often the router picks the model the oracle would.
- **ParetoDist** (LLMRouterBench) — distance to the Pareto frontier.
- **Deferral curve** — accuracy vs inference cost as you raise the budget.

**Robustness metrics**
- Surface-perturbation accuracy retention (RouterArena: paraphrase, typo, format, style — 420 sampled
  examples perturbed by GPT-4o).
- ID vs OOD AUROC gap (RouterXBench).

**Production operational metrics** (channel.tel, tmls.nyc, ailog.fr, mindstudio — all 2026)
| Metric | What it tells you | Alarm condition |
|---|---|---|
| **Routing mix** (fraction per model) | Drift toward the cheap model | Sudden shift week-over-week |
| **Quality per route** (not aggregate!) | The cheap model's adequacy *on its own traffic* | Any route below its quality floor |
| **Escalation rate** | Is the cheap model keeping up? | Spike |
| **Cost vs frontier-only baseline** | Is the trade explicit? | Savings bought with quality |
| **Confidence calibration (ECE)** | Does the score predict correctness? | Drift after any model update |
| **Routing latency p50/p95** | The tax you're paying | p50 > 200 ms |
| **Classifier cost overhead** | Net-of-router savings | Overhead > 10% of gross saving |
| **Cache hit rate** | Compounds with routing | <70% where applicable |

### The monitoring rule that matters most
> *"Instrument quality **per route**, not just aggregate cost... set a quality floor that the router may
> not cross regardless of cost savings; derive escalation thresholds from evaluation data rather than by
> feel; and monitor the routing mix and per-route quality continuously so a drift toward the cheap model
> registers as a quality regression immediately."* — tmls.nyc, 2026-06

### The CI gate
digitalapplied.com (2026-06-14): *"a pre-merge CI gate running 50–500 representative cases —
groundedness, context adherence, and an LLM-as-judge check — that blocks any routing change which drops
quality below threshold."* Teams without one "discover regressions from customer tickets days after they
hit production." **Ship a `router eval` command that can run in CI.**

## 4.4 Calibration methodology (the part everyone under-documents)

1. **Threshold calibration to a target strong-model share.** RouteLLM ships this as a CLI:
   `calibrate_threshold --strong-model-pct 0.5` → prints the threshold. It runs on the public
   `lmsys/lmsys-arena-human-preference-55k` dataset. **Their own warning:** "because we calibrate the
   thresholds based on an existing dataset, the % of calls routed to each model will differ based on the
   actual queries received... we recommend calibrating on a dataset that closely resembles the types of
   queries you receive."
2. **Per-route threshold optimization.** Aurelio's `SemanticRouter.fit(X, y)` optimizes each route's
   `score_threshold` — their docs show 34.85% → 89.39% accuracy. `evaluate(X, y)` reports accuracy.
   **Threshold AND margin are both required** (sureprompts.com): threshold alone passes close-calls as
   confident; margin alone passes low-absolute-similarity queries as "best of a bad lot." Check both.
3. **Confidence calibration before thresholding.** UCCI: isotonic regression from token-margin
   uncertainty → error probability; ECE 0.12 → 0.03. Without this, entropy thresholding is the worst
   baseline measured.
4. **Conformal calibration as an error-budget knob.** Conformal Cascade: α *is* the user's error budget,
   and expected cost is closed-form in α. Per-tier level α/K keeps cascade error ≤ α.
   **Caveat found in their own results:** at α ≤ 0.20 prediction sets saturate to the full answer set →
   100% deferral → no saving. Guard against degenerate calibration.
5. **Constrained-optimization threshold selection.** UCCI/Theorem 1: threshold policies on *calibrated*
   error probabilities are cost-optimal under three explicit assumptions. This is the principled version
   of "pick a threshold by feel."

## 4.5 Evaluation pitfalls catalogued across the corpus

- **Discrete scoring loses nuance.** RouterBench uses {0, 0.25, 0.5, 0.75, 1} — a Stanford CS224N report
  flags this as oversimplifying model-output complexity.
- **Single-turn benchmarks don't test agents.** RouterBench "is single-turn and tests neither tool use
  nor long-horizon agent behavior."
- **Router-visible prefix is never exposed** at intermediate agent steps in existing benchmarks.
- **Routers do best when eval ≈ training distribution** (LLMRouterBench / RouterXBench). Always report
  an OOD split.
- **Online LLM judges at eval time** make results non-reproducible and expensive (OmniRouter's complaint).
- **Cold-model latency** (ai-tldr.dev): self-hosted tiers that are cold stall 3–15 s loading weights.
  "Keep at least one warm instance of each tier you route to in production, or use provider APIs that
  guarantee hot serving. Cold-start latency turns a routing win into a user-experience regression."
