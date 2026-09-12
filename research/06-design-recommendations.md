# Thread 6 — Python Packages Landscape + Design Recommendations

Research date: 2026-09-12.

## 6.1 Existing Python packages — the competitive set (all verified on PyPI/GitHub this session)

| Package | PyPI name | Version / date | What it is | Gap it leaves |
|---|---|---|---|---|
| **LLMRouter** (ulab-uiuc, UIUC Jiaxuan You lab) | `llmrouter-lib` | **0.4.0, 2026-08-14**; 1.3k+ stars; MIT; Python ≥3.10 | **The closest direct competitor.** "A unified framework for LLM routing and evaluation." **16+ routers** in 5 categories (single-round, multi-round, multimodal, agentic, personalized): knn, svm, mlp, **mf**, elo, **routerdc**, automix, hybrid_llm, **graphrouter**, causallm_router, smallest_llm, largest_llm. Unified `llmrouter` CLI (train/infer/chat), Gradio UI, ComfyUI visual interface, plugin system for custom routers, data-gen pipeline for **11 benchmarks**, **xRouteBench** benchmark, **OpenClaw Router** = OpenAI-compatible server w/ multimodal + retrieval-augmented routing memory + streaming. Paper arXiv:2608.06867 (HF Daily Papers, 2026-08). Key result: **learned routers beat the strongest fixed-model baseline by 14.6% relatively**; lightweight + user-conditioned routers win under tight budgets. | Research-oriented: training-first, GPU-optional extras pin `vllm==0.6.3`/`torch==2.4.0`; no production ops surface (no fallback chains, circuit breakers, budget pacing, per-route quality monitoring) |
| **RouteLLM** (lm-sys) | `routellm` | 0.2.0, 2024-07-08 (repo stale since 2024-08) | Binary strong/weak learned router + OpenAI-compatible server + eval harness | Binary only; stale checkpoints; no async; no observability; no caching |
| **semantic-router** (Aurelio) | `semantic-router` | ~3.7k stars, MIT | Embedding-similarity decision layer, pluggable encoders, Pinecone/Qdrant indexes, `fit()`/`evaluate()` threshold optimization, dynamic + multi-modal routes | Intent routing, not model selection; no cost model; no execution layer |
| **RoRF** (Not Diamond) | `rorf` | — | Pairwise random-forest router, 12 pre-trained routers | Pairwise only; no serving; no eval harness |
| **local-llm-router** | `local-llm-router` | 0.4.1, 2026-06-06 | **Zero runtime dependencies.** Local-first complexity-tier router for agent loops: `configure(vram_gb, quant)`, `route(prompt, hint=...)` → (tier, model). Hints: lookup/explain/design/code/reason. CLI: `llm-router compare`, `stack benchmark`. | Local/Ollama only; heuristic only; no learned routing |
| **llm-router** (ypollak2) | `llm-routing` | 2026-07 | Free-first fallback chain for coding tools (Claude Code, Cursor, Codex, Gemini CLI). Claims 35–80% cost cut. Listed on RouterArena leaderboard (#11, Arena 71.26). | Coding-tool specific |
| **llm-api-router** | `llm-api-router` | 0.1.1, 2026-01-19 | OpenAI-SDK-shaped multi-provider client, zero-code provider switching, unified SSE streaming | Unified client, not a router |
| **ai-llm-router** | `ai-llm-router` | 0.1.0, MIT | Multi-provider routing with fallbacks, cost optimization, monitoring | Early |
| **sglang-router** | `sglang-router` | 2026-01 | **SGLang Model Gateway** — the production-quality reference: 40+ Prometheus metrics, OTel OTLP tracing, request-ID propagation, retries w/ jitter, **worker-scoped circuit breakers**, token-bucket rate limiting with queuing, background health checks, cache-aware load monitoring, PD-disaggregation service discovery, conversation/history state, native MCP client | SGLang-ecosystem; worker routing, not model-capability routing |
| **LiteLLM** | `litellm` | v1.98.0, 57k+ stars, MIT | The execution layer. See `02-frameworks-landscape.md` | `cost-based-routing` is **cost-only** — no quality model |

### Maturity and extensibility assessment (the "could this be my foundation?" question)

| Candidate | Maturity signals | Extensibility mechanism | Verdict as a foundation |
|---|---|---|---|
| **RouteLLM** | PyPI 0.2.0 (2024-07-08); **repo dead since 2024-08-10**; 0 tags, no releases; pins `numpy<2`; ~5.5k★; Apache-2.0 | Implement one method (`calculate_strong_win_rate`) + register in `ROUTER_CLS`. Genuinely elegant. | **Do not fork.** Mine the two ideas (one-method contract; threshold-in-model-string) and rebuild |
| **LLMRouter / llmrouter-lib** | v0.4.0 2026-08-14; **actively released** (0.1.0 → 0.4.0 in 8 months); 1.3k+★; MIT; paper arXiv:2608.06867 on HF Daily Papers; 5 router categories, 16+ strategies; CLI + Gradio + ComfyUI | Subclass `MetaRouter`, implement `route_single(query_input) -> dict` and `route_batch`. Clean plugin workflow; example `RandomRouter` shipped. | **The real competitor. Study closely; consider depending on it for research strategies and building the production layer around it** — but note `Development Status :: 3 - Alpha` and the `router-r1` extra pins `vllm==0.6.3` / `torch==2.4.0` |
| **semantic-router** | ~3.7k★; MIT; multi-encoder (OpenAI/Cohere/HF/FastEmbed/LiteLLM); Pinecone + Qdrant indexes; `fit`/`evaluate` API | Pluggable `Encoder` protocol; `Route` objects; dynamic routes | **Depend on it** for the embedding-similarity layer rather than reimplementing |
| **LiteLLM** | 57k★; v1.98.0; MIT; the de facto execution layer | `Custom Routing Strategy` documented; per-key/team overrides | **Optional extra** (`[litellm]`) for provider breadth — do not hard-depend |
| **SGLang Model Gateway** | Production-grade; 40+ Prometheus metrics; OTel | Not a Python library — a gateway service | **Reference architecture only** for the ops surface |
| **RoRF** | ~200★; 12 pre-trained pairwise routers | Random-forest classifier over embeddings | Take the random-forest-on-embeddings recipe; the package itself is too narrow |
| **Avengers-Pro** | AAAI 2026 / CIKM '25; code on GitHub; Pareto-optimal in a neutral benchmark | 3 functions: embed, cluster, score | **Port the algorithm** — it is ~200 lines and removes checkpoint rot |
| **local-llm-router** | 0.4.1 (2026-06-06); **zero runtime dependencies**; works offline | Step hints (lookup/explain/design/code/reason) | **Copy the packaging discipline**: zero mandatory deps, optional `[ollama]` extra |

### The gap, stated precisely
> **Nobody ships a package that combines (a) a pluggable multi-strategy *decision* layer with N>2
> models, (b) the production *execution* layer (fallback chains, circuit breakers, budget pacing), and
> (c) the *evaluation + monitoring* loop (calibration CLI, AUROC/per-route quality, CI gate, retraining
> trigger) in one installable Python library with zero mandatory heavy dependencies.**
>
> LLMRouter has (a) and (c-partial) but not (b). LiteLLM has (b) but not (a) or (c). RouteLLM has
> (a-binary) and (c-partial). semantic-router has a piece of (a). sglang-router has (b) for workers.
> **That's the product.**

## 6.2 Recommended architecture

### Core abstraction: three separable layers

```
┌─ DECISION LAYER (pure, testable, no I/O) ───────────────────────────┐
│  SignalExtractor → PolicyEngine → ModelSelector                     │
│  returns a RoutingDecision(model, scores, signals, reason, latency) │
└─────────────────────────────────────────────────────────────────────┘
┌─ EXECUTION LAYER (async, stateful, pluggable) ──────────────────────┐
│  ModelBackend protocol → RetryPolicy → CircuitBreaker → FallbackChain│
└─────────────────────────────────────────────────────────────────────┘
┌─ LEARNING LAYER (offline + online) ─────────────────────────────────┐
│  Calibrator → Evaluator → Metrics/Telemetry → Retrainer             │
└─────────────────────────────────────────────────────────────────────┘
```

**Why this split:** it is exactly what the 2026 survey found real systems converge on
(arXiv:2603.04445: pre-router → post-generation verifier → escalation policy), it is vLLM-SR's
architecture (signal extraction → decision engine → plugin chains), and it makes the decision layer
unit-testable without any network.

### Signal → Policy → Select (steal from vLLM-SR, simplify it)

```python
@dataclass(frozen=True)
class Signal:
    name: str
    value: Any
    cost_ms: float          # so you can budget the decision itself
    kind: Literal["heuristic", "learned"]

class SignalExtractor(Protocol):
    name: str
    cost_tier: int                              # 0=sub-ms, 1=ms, 2=10ms, 3=100ms+
    def applies(self, req: RoutingRequest, needed: set[str]) -> bool: ...
    async def extract(self, req: RoutingRequest) -> Signal: ...
```
- **Demand-driven evaluation:** only compute signals a configured policy references (vLLM-SR's key trick).
- **Parallel execution of same-tier signals** (vLLM-SR: wall-clock = slowest evaluator, not the sum).
- **Cost-tiered ordering:** heuristics first, learned signals lazily.

Policy = declarative rules over signals. Ship a small DSL/dict form, not a full language:
```yaml
policies:
  - name: never-cheap
    when: {any_of: [{signal: pii, eq: true}, {signal: intent, in: [billing, legal, safety]}]}
    then: {model: strong}
  - name: cheap-path
    when: {all_of: [{signal: difficulty, lt: 0.35}, {signal: context_tokens, lt: 32000}]}
    then: {model: cheap}
  - name: default
    then: {selector: calibrated_classifier, threshold: 0.42}
```

### The router protocol (steal RouteLLM's one-method contract, generalize it)

```python
class Router(Protocol):
    name: str
    async def score(self, req: RoutingRequest, candidates: Sequence[Model]) -> dict[str, float]:
        """Return a score per candidate model. Higher = better."""
```
RouteLLM's insight was that a single method makes extension trivial — but it was
`calculate_strong_win_rate(prompt) -> float`, which hard-codes 2 models. Generalizing to
`score(request, candidates) -> {model_id: float}` supports N models with the same simplicity.

## 6.3 Which strategies to ship, in order of priority

| Priority | Strategy | Why (evidence) | Effort |
|---|---|---|---|
| **P0** | **Rules / metadata gate** | Free, deterministic, the only way to do hard exclusions. Every hybrid system starts here. | S |
| **P0** | **Cost-aware selection** (cheapest model meeting constraints) | What LiteLLM does; table stakes. Needs a model registry with pricing. | S |
| **P0** | **Training-free clustering router** (Avengers-Pro) | **Removes checkpoint-rot entirely.** 3 lightweight ops (embed, k-means, per-cluster profile), 1 hyperparameter, incremental new-model onboarding, Pareto-optimal in LLMRouterBench, −27% at matched GPT-5-medium, +7.1% accuracy. This is the highest value-per-effort learned strategy available. | M |
| **P0** | **Cascade w/ calibrated confidence** | Highest savings ceiling; UCCI's 31% on 75k real production queries is the most credible number in the corpus. Ship **isotonic calibration** and **conformal** as the two calibration modes. | M |
| **P1** | **Fine-tuned small classifier** (ModernBERT/BERT + LoRA, ONNX runtime) | 10–100 ms, best quality-per-ms of learned approaches. vLLM-SR: 98.53% with 805 examples / 2h compute. Ship as an **optional extra**, ONNX by default so it runs on CPU with no torch. | L |
| **P1** | **Embedding-similarity router** | 5–20 ms, zero training, `fit()`-optimizable thresholds (Aurelio's 34.85%→89.39%). Great cold-start default. | S |
| **P1** | **Preference-data matrix factorization** | RouteLLM's best cost/latency tradeoff; but ship it as *your* implementation trained on *current* data, not their 2024 checkpoint. | M |
| **P2** | **Bandit / online adaptation** (LinUCB + preference prior, PILOT-style) | The only family that adapts post-deployment. MixLLM: +3.35% OOD mitigation via online learning. Hard to get right; ship behind a flag. | L |
| **P2** | **Graph router** | Generalizes to new models zero-shot (the hardest problem in this space). GraphRouter ≥88.89% of oracle. | L |
| **P2** | **LLM-as-classifier (last-resort tier only)** | 200–800 ms. Only for the ambiguous residual. Also: your **offline label generator**. | S |
| **P3** | IRT-Router, RouterDC, PersonalizedRouter, reasoning-mode routing | Interesting, niche, or unproven at production scale. | — |

### Explicitly do NOT do
- **Do not ship pre-trained frozen checkpoints as the default.** RouteLLM's #25/26 ranking on the 2026
  leaderboard is the cautionary tale. Ship a `router fit --data your_prompts.jsonl` that works in minutes.
- **Do not require torch or an API key to install.** `local-llm-router`'s "zero runtime dependencies,
  works offline" is a genuinely good design constraint. Make torch/GPU/LiteLLM optional extras.
- **Do not build a gateway.** Compete on the decision + learning layers; delegate execution to LiteLLM
  or your own thin async backend layer.
- **Do not claim 85% savings.** Claim the neutral number and ship the tool that measures *their* number.

## 6.4 API surface to design

```python
# 1. In-process (the 90% case)
router = Router.from_yaml("router.yaml")
decision = await router.decide(messages=..., metadata={...})
resp = await router.complete(messages=...)          # decide + execute + fallback

# 2. Drop-in OpenAI-compatible client (RouteLLM's proven trick)
client = RouterClient(router=router)                # subclasses/mirrors openai.AsyncOpenAI
client.chat.completions.create(model="router:cost-aware@0.42", messages=[...])

# 3. OpenAI-compatible server (for non-Python consumers)
python -m yourrouter serve --config router.yaml --port 6060
```
**Threshold-in-the-model-string** (`router:<strategy>@<threshold>`) is RouteLLM's best idea — it carries
router+threshold over an unmodified OpenAI protocol. Keep it, and extend it to name the *policy*.

### CLI (RouteLLM and LLMRouter both prove a CLI is what makes a router adoptable)
```
yourrouter fit        --data prompts.jsonl --strategy cluster --k 64
yourrouter calibrate  --strategy classifier --target-strong-pct 0.30
yourrouter eval       --data holdout.jsonl --strategies cluster,classifier,cascade   # CI-able
yourrouter profile    --models models.yaml            # build per-cluster capability profiles
yourrouter serve      --config router.yaml
yourrouter benchmark  --replay routerbench            # offline, no live inference
yourrouter explain    --prompt "..."                  # show every signal + why this model
```
`explain` is not a nice-to-have: RouterArena found interpretability is where academic routers fail, and
ACAR (2026) exists specifically to add "auditable decision traces." **Ship explainability as a first-class output.**

## 6.5 Production checklist (each item maps to a verified source)

- [ ] **async-first**, sync wrappers only (`acompletion`/`completion`)
- [ ] **Redis-backed shared state** for cooldowns, rate limits, latency windows (required >1 replica — LiteLLM docs)
- [ ] **Per-deployment circuit breaker** with CLOSED/OPEN/HALF_OPEN + canary probe on half-open (markaicode)
- [ ] **Three separate fallback channels** (general / content-policy / context-window) + `max_fallbacks` (LiteLLM)
- [ ] **Per-exception retry policy** (429 ≠ 401 ≠ 500) (LiteLLM)
- [ ] **Pre-call context-window validation** (LiteLLM `enable_pre_call_checks`)
- [ ] **Parameter stripping on model switch** (LiteLLM ≥1.44 lesson)
- [ ] **Health probe must not call the LLM provider** (markaicode)
- [ ] **OpenTelemetry + Prometheus** with per-stage latency and every candidate score (sglang-router: 40+ metrics)
- [ ] **Budget pacing with closed-loop enforcement** — the biggest unmet need (arXiv:2604.00136: baselines overshoot by up to 6.9× without it)
- [ ] **Per-route quality monitoring + routing-mix drift alarm** (tmls.nyc)
- [ ] **Semantic cache** (HNSW) as an optional layer — compounds with routing (mindstudio: 30–50% of redundant calls)
- [ ] **Router's own latency budget enforced**: target <10 ms p50 for the decision (arXiv:2604.00136 achieves 22.5 µs for the decision, 9.8 ms end-to-end on CPU)
- [ ] **Security posture**: dependency pinning, no `eval` of config, SBOM (LiteLLM's 2026 CVE history)
- [ ] **License: Apache-2.0** (matches RouteLLM and vLLM-SR; more permissive than MIT on patents)

## 6.6 Evaluation harness — what to build (this is the moat)

1. **Replay mode** on RouterBench (405k pre-generated outcomes → no live inference, no cost). This is
   how you benchmark without burning money.
2. **Oracle-gap reporting**: always show the oracle ceiling next to the router score (RouterArena's
   oracle = 90.89%; most routers sit at 100% cost / 100% accuracy by failing to defer).
3. **AUROC of the routing decision** — threshold-independent, from RouterXBench. Report this *before*
   picking an operating point.
4. **Difficulty-stratified accuracy** (easy/medium/hard) — RouterArena shows >89% on easy, <10% on hard.
   Aggregate accuracy hides the failure.
5. **Robustness suite**: paraphrase / typo / format / style perturbations (RouterArena's method).
6. **ID + OOD split** always reported separately.
7. **Deferral curve** (accuracy vs cost as budget rises) + **ParetoDist**.
8. **A `--ci` mode** that exits non-zero if quality drops below a floor. This is the pre-merge gate
   digitalapplied.com recommends (50–500 cases).
9. **Misroute → training-example pipeline.** Braintrust's "production traces become evaluation cases"
   is the pattern; apply it to routing decisions.

## 6.7 Phased build plan

> ⚠ **COST MODEL PARTIALLY FALSIFIED — see `10-fusion-delegation-architecture.md` §10.4.**
> Phase 0's "cost-aware selector + model registry w/ pricing" assumes **cost is a per-model property
> derived from `tokens × $/token`**. Cognition's Fusion data contradicts this: a sidekick costing
> **275% more per token** made the system **2% cheaper**, because stronger models are more
> token-efficient and generate fewer rework/review rounds. Their conclusion: *"models should be
> evaluated on **price per task** rather than price per token."*
>
> **Correction:** the registry must carry, per model, a token-efficiency factor and a rework rate;
> and cost must additionally be recorded **per (lead, sidekick) pair**, because *"the models' cost and
> intelligence are coupled."* A registry keyed only by model cannot express pair cost. Implemented as
> `PairProfile` / `PairRegistry` in `llmrouter`.
>
> Also **promoted P1 → P0**: sub-agent delegation (`10`). Where delegation applies it strictly
> dominates mid-session switching — no cache write, no oscillation, no stranded migrations.

| Phase | Ship | Rationale |
|---|---|---|
| **0** | `Router` protocol, `RoutingRequest`/`RoutingDecision` dataclasses, YAML config, `explain` output, rules engine, cost-aware selector, model registry w/ pricing | Foundation + immediate utility (matches LiteLLM's cost routing but as a library) |
| **1** | Clustering router (Avengers-Pro style) + embedding-similarity router + `fit`/`calibrate`/`eval` CLI + RouterBench replay | Working learned routing with **zero model training**, plus the eval moat |
| **2** | Cascade with isotonic + conformal calibration, fallback chains, circuit breaker, Redis state, OTel/Prometheus | Production-ready; captures the highest savings ceiling |
| **3** | ONNX small-classifier router (optional extra), OpenAI-compatible server + drop-in client, semantic cache | Scale + integration surface |
| **4** | Bandit/online adaptation, budget pacer with closed-loop enforcement, graph router | Differentiation; the unmet needs |

## 6.8 Open questions to resolve before building

1. **Target workload: chat/RAG or agents?** arXiv:2608.14641 found routing effects statistically
   inconclusive on BFCL v4, RouterBench and tau2-bench, and equivalent on WebArena. If agents are the
   target, one-shot benchmarks will mislead and you need session-level routing from day one.
2. **Pool size.** RouterEval says 3–10 candidates is the sweet spot; LLMRouterBench says adding models
   yields diminishing returns. Design for 3–10, support up to ~50.
3. **Self-hosted models or API-only?** Self-hosted adds cold-start (3–15 s) and queue-depth signals;
   API-only adds rate-limit and price-drift signals. Different signal sets.
4. **Do you need multimodal?** LLMRouter and TSRouter already ship it; it's a real cost axis (text LLM
   vs VLM) but a big surface area.
5. **Is "router" or "gateway" the wedge?** The market is converging on both-in-one. A pure router library
   is a smaller, cleaner, more defensible wedge.
