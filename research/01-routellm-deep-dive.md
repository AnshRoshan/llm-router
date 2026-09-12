# Thread 1 — RouteLLM Deep Dive (lm-sys/RouteLLM)

Research date: 2026-09-12. Everything below is from primary sources fetched this session unless flagged.

## 1. Project facts (verified from GitHub repo README + repo metadata)

| Attribute | Value | Source |
|---|---|---|
| Repo | `lm-sys/RouteLLM` | https://github.com/lm-sys/RouteLLM |
| Stars / Forks | ~5.5k / 430 | repo page, fetched 2026-09-12 |
| License | Apache-2.0 | repo LICENSE + README |
| PyPI | `routellm` 0.2.0; install extras `routellm[serve,eval]` | https://pypi.org/project/routellm/ |
| Last commit on `main` | **2024-08-10** ("create /health endpoint for health checks"), 175 commits total, 0 tags, 3 branches | repo file listing, fetched 2026-09-12 |
| PyPI last release | 0.2.0, dated 2024-07-08 | PyPI page |
| Paper | arXiv:2406.18665, "RouteLLM: Learning to Route LLMs with Preference Data" (Ong, Almahairi, Wu, Chiang, Wu, Gonzalez, Kadous, Stoica) | README citation block |
| Collaborator | Anyscale (commercial router comparison partner) | README "Motivation" |

> **Critical recency flag:** the repository has not had a commit since Aug 2024 (~25 months stale as of Sep 2026). It pins `numpy<2`. The pre-trained router checkpoints are therefore trained against the 2024 model landscape (GPT-4-1106-preview / Mixtral-8x7B era). This is the single most important caveat for anyone reusing it.

## 2. Core design (what it actually is)

RouteLLM is a **binary** router: exactly two models, a *strong* and a *weak*. A router emits a scalar
`calculate_strong_win_rate(prompt) -> float` = predicted probability the strong model wins on that prompt.
If win_rate > user-specified **cost threshold** → strong model, else weak model.

Key architectural decisions worth stealing:

1. **Single-method extension contract.** "There is only a single method to implement:
   `calculate_strong_win_rate`, which takes in the user prompt and returns the win rate for the strong
   model conditioned on that given prompt." → trivially pluggable router interface.
2. **Threshold is a first-class, per-request parameter**, not a global config. Encoded in the model
   string on the wire: `model="router-mf-0.11593"` → router name + threshold. This is a genuinely
   elegant trick: an OpenAI-compatible protocol carries router+threshold with **zero API changes**.
3. **Drop-in client AND server.** `routellm.controller.Controller` subclasses the OpenAI client;
   `python -m routellm.openai_server` runs a uvicorn OpenAI-compatible endpoint on :6060.
4. **Delegates provider adapters to LiteLLM.** RouteLLM does not write provider SDKs — it uses LiteLLM
   under the hood, plus `openai/` prefix + `--base-url`/`--api-key` for any OpenAI-compatible endpoint.
5. **Threshold calibration is a shipped CLI**, not an afterthought:
   `python -m routellm.calibrate_threshold --routers mf --strong-model-pct 0.5` →
   prints `threshold = 0.11593`. Calibration runs on public Chatbot Arena data.
6. **Eval harness is a first-class CLI:** `python -m routellm.evals.evaluate --routers random sw_ranking bert --benchmark gsm8k`.
   Supported benchmarks: `mmlu`, `gsm8k`, `mt-bench`.
7. **Hidden coupling worth knowing:** regardless of model pair, an `OPENAI_API_KEY` is required because
   the `mf` and `sw_ranking` routers generate embeddings with OpenAI. (README, "Model Support".)

## 3. The shipped routers (implementation detail)

README "Routers" section lists **5** routers, all trained on the `gpt-4-1106-preview` /
`mixtral-8x7b-instruct-v0.1` pair (checkpoints on HF under the `routellm` and `lmsys` orgs):

| Router | Mechanism (README wording) | Cost/latency | Notes |
|---|---|---|---|
| `mf` | "Uses a matrix factorization model trained on the preference data (**recommended**)" | low | LMSYS' recommended default: "very strong and lightweight" |
| `sw_ranking` | "Uses a weighted Elo calculation for routing, where each vote is weighted according to how similar it is to the user's prompt" | highest | best raw quality, needs OpenAI embeddings |
| `bert` | "Uses a BERT classifier trained on the preference data" | mid | |
| `causal_llm` | "Uses a LLM-based classifier tuned on the preference data" | highest | training notebook lives in `anyscale/llm-router` |
| `random` | coin flip | ~0 | baseline only |

Extension contract (README "Adding a new router"): implement the abstract `Router` class in
`routers.py`, register it in the `ROUTER_CLS` dict — then it is immediately usable in both the server
and the eval framework. Adding a benchmark: implement the abstract `Benchmark` class in
`benchmarks.py` and wire it into `evaluate.py`; README recommends **precomputing** benchmark results.
Eval results are **cached per (router, benchmark)** with `--overwrite-cache` to bust.
For MMLU/GSM8K, results were generated with **SGLang**; for MT Bench they use precomputed judgements.

## 4. Headline numbers — and the precise caveats

From README + arXiv:2406.18665v4 (verified text):

- "reduce costs by up to **85%** while maintaining **95% GPT-4 performance** on widely-used benchmarks like MT Bench"
- "achieve the same performance as commercial offerings while being **>40% cheaper**" (vs Martian / Unify)
- Paper's own more conservative framing: "reduce costs by **over 2 times** without sacrificing response quality"
- Paper cost-analysis table (CPT = Cost Per Threshold, i.e. cost-savings ratio vs random router):

| Benchmark | CPT(50%) | CPT(80%) |
|---|---|---|
| MT Bench | **3.66** (at 95% GPT-4 quality) | 2.49 |
| MMLU | 1.41 (92% GPT-4 quality) | 1.14 |
| GSM8K | 1.49 (87% GPT-4 quality) | 1.27 |

**The 14% vs 26% discrepancy resolved:** on MT-Bench the MF router reaches 95% of GPT-4 quality while
calling GPT-4 on **26%** of prompts trained on Arena data; with the GPT-4-judge augmentation dataset
(`D_judge`) that drops to **14%** — the source of "up to 85%" (1/0.14 ≈ 7.1× on the GPT-4-dominant cost).
Secondary source (dreaming.press, 2026-06-22) documents this explicitly; the 3.66× CPT figure in the
paper is the *defensible* number, 85% is the *headline* number. **Use 2–3.7× in your own claims, not 85%.**

- Data augmentation matters more than router architecture: the paper reports dataset-vs-benchmark
  similarity scores (MT Bench 0.6078 → 0.6525 with judge augmentation; GSM8K 0.4926 → 0.5335).
- **Generalization claim:** routers "maintain performance even when routing between LLMs not included
  in training" — reported to transfer to Claude 3 Opus + Llama 3 8B without retraining.

## 5. Independent verification of RouteLLM's numbers (the important part)

**RouterArena** (arXiv:2510.00202, published at ICLR 2026, code `RouteWorks/RouterArena`) benchmarked 12
routers on a common footing. Results that directly bear on RouteLLM:

RouterArena leaderboard (paper Table 2, β=0.01 / β=0.1 / β=1 cost weights):

| Rank @β=0.01 | Router | Arena score | Accuracy | Norm. cost |
|---|---|---|---|---|
| 1 | GPT-5 (single model) | 0.7282 | 0.7428 | 0.2453 |
| 2 | Azure-Router | 0.6780 | 0.6798 | 0.5386 |
| 3 | MIRT-BERT | 0.6731 | 0.6731 | 0.6705 |
| 4 | CARROT | 0.6682 | 0.6720 | 0.4266 |
| 5 | vLLM-SR | 0.6633 | 0.6665 | 0.4494 |
| 6 | NotDiamond | 0.6565 | 0.6651 | 0.2858 |
| **7** | **RouteLLM** | **0.6175** | **0.6224** | **0.3451** |
| 8 | NIRT-BERT | 0.6151 | 0.6159 | 0.5416 |
| 10 | GraphRouter | 0.6070 | 0.6072 | 0.5884 |
| 12 | RouterDC | 0.3362 | 0.3344 | 0.7507 |

Other RouterArena findings:
- Oracle accuracy ceiling = **90.89%**; most routers cluster near (100% cost, 100% accuracy) — i.e. they
  over-rely on the strongest model and fail to defer.
- **Best efficiency routers: vLLM-SR and CARROT ≈ 35% lower cost at <2% accuracy degradation.**
- NIRT-BERT reached baseline accuracy at **378–400%** of the best-available-model cost.
- Latency: vLLM-SR **546.8 ms** (because it calls OpenAI embeddings), RouteLLM **259.0 ms**, others mostly <100 ms.
- Robustness (surface perturbations): generally LOW; BERT-based academic routers are very sensitive.
- Difficulty stratification: >89% accuracy on easy queries, **<10% on hard** queries for most routers.
- Commercial routers do NOT necessarily beat open source; NotDiamond ranked #12 on the live leaderboard
  at one snapshot for over-selecting expensive models.

**Current live RouterArena leaderboard (GitHub README, fetched via search 2026-09-12)** — a *different*
and more recent snapshot than the paper. RouteLLM ranks **#25 of 26** (Arena 48.07, accuracy 47.04,
cost $0.27/1K queries, optimal-selection 99.72, robustness 100.0). Leaders: Cross-Router (76.12),
vLLM-SR (75.30), Sqwish Router (75.27), Nadir-Tumbler (75.17), AgentForge Router (74.13).

> **Interpretation:** RouteLLM's *robustness* is best-in-class (100.0) and its cost is near-lowest
> ($0.27/1K), but its accuracy on this harder, broader benchmark is poor (47%). Its 2024 checkpoints are
> the likely cause. **Design lesson: router quality is checkpoint-freshness-dependent. A production
> package must treat routers as retrainable artifacts with a refresh cadence, not as frozen weights.**

## 6. Fairness caveat — router benchmarks disagree

2026 unified-evaluation work cited by secondary sources:
- "Towards Fair and Comprehensive Evaluation of Routers" (arXiv:2602.11877)
- "LLMRouterBench" (arXiv:2601.07206)

Reported conclusion (as relayed): router performance is highly benchmark-dependent; routers do best when
eval distribution matches training distribution, and under fair unified evaluation "most routing methods
collapse to similar performance," with several recent approaches failing to reliably beat simple baselines.
**STATUS: NOT INDEPENDENTLY VERIFIED — arXiv IDs come from a secondary blog (agentropic.ai). Flagged as
low-confidence until the abstracts are read directly.**

## 7. What to steal / what to avoid for a new package

**Steal:**
- The one-method router contract (extend by implementing one scoring function).
- Threshold-in-the-model-string protocol → zero-change OpenAI compatibility.
- Shipping calibration as a CLI command that answers "what threshold gives me X% strong-model calls".
- Shipping an eval harness alongside the router — this is why RouteLLM stayed relevant while going stale.
- Delegate provider adapters to an existing unified client (LiteLLM or the OpenAI SDK) instead of writing them.

**Avoid:**
- Binary strong/weak only. RouterEval found cost-effectiveness peaks at **3–10 candidate models**
  (EMNLP 2025 Findings), so a 2-model design leaves money on the table.
- Frozen pre-trained checkpoints with no retraining story.
- Mandatory OpenAI API key for embeddings.
- No async story / no observability / no caching (wavect.io comparison: RouteLLM has neither).
- Repo maintenance: 0 tags, no releases since 2024.
