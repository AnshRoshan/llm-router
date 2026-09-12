# Thread 2 — Framework & Gateway Landscape (2026 state)

Research date: 2026-09-12.

## 2.0 The distinction that matters most for design

A 2026 secondary analysis (dreaming.press, 2026-06-22) and the wavect.io comparison (2026-07-07)
converge on the same taxonomy, and it is the right frame for a new package:

- **Gateway** (LiteLLM, Portkey, OpenRouter, Kong, Cloudflare AI Gateway): routes by **rules you
  declare** — virtual key → model, fall back on 529, cap this budget. Deterministic and auditable.
- **Router** (RouteLLM, Not Diamond, Martian, vLLM-SR, Bedrock IPR): **predicts**, per prompt, which
  model is good enough. Adds a learned decision, a quality tail, and extra latency.

> **Design implication:** a production-grade Python package should be a **router that lives inside a
> gateway**, or expose both layers separately. Nobody in 2026 is shipping a bare classifier; the
> market has converged on "decision layer + execution layer." RouteLLM explicitly is *not* a gateway
> (wavect.io: "routing decision only, not a full gateway", no caching, no observability).

## 2.1 Framework index (verified from multiple 2026 comparisons; per-item confidence noted)

| Tool | Type | Hosting | License | Routing intelligence | Confidence |
|---|---|---|---|---|---|
| **LiteLLM** | Proxy gateway + SDK | Self-host or managed | MIT | Deterministic: 6 load-balancing strategies, fallback chains, budgets, routing groups | High — official docs read this session |
| **Portkey AI Gateway** | Gateway + observability + guardrails | Self-host core + cloud | OSS core | Conditional routing, circuit breakers, guardrails, semantic caching | High |
| **OpenRouter** | Hosted aggregator/marketplace | Managed | Proprietary | Price/latency/availability + "Auto" mode (powered by Not Diamond) | High |
| **RouteLLM** | Research routing framework | Self-host, embedded | Apache-2.0 | Learned (MF/BERT/causal-LLM/SW-ranking) | High — README read this session |
| **vLLM Semantic Router** | Envoy ExtProc router | Self-host (K8s-native) | Apache-2.0 | **Hybrid**: 13 signal types (heuristics + LoRA neural classifiers) | High — arXiv read this session |
| **Not Diamond / RoRF** | Managed router + OSS pairwise | Managed | Mixed | Learned (random forest on embeddings) | High — GitHub read this session |
| **Martian** | Managed router | Managed | Proprietary | Closed "model mapping" | Medium — no primary docs read |
| **AWS Bedrock Intelligent Prompt Routing** | Managed, in-family | Managed | Proprietary | Learned quality prediction, same-family only | High |
| **Microsoft Foundry Model Router** | Managed, cross-pool | Managed | Proprietary | Trained LM, 3 modes (Balanced/Cost/Quality) | High — MS TechCommunity read |
| **Cloudflare AI Gateway** | Edge proxy | Managed | Proprietary | Caching, rate limiting, analytics | Medium |
| **Kong AI Gateway** | API gateway plugin | Both | OSS | Governance/rate-limit oriented | Medium |
| **Helicone** | Observability proxy | Both | OSS | Minimal routing | Medium |
| **semantic-router (Aurelio)** | Python decision layer | Embedded library | MIT | Embedding similarity + thresholds | High — docs read |
| **DigitalOcean / Together / Fireworks** | Provider routers | Managed | Proprietary | Semantic task routing / endpoint routing | Low-Medium |

Adoption signals (as reported 2026, pkgpulse.com at access time): LiteLLM 57,242 stars / 10,886 forks
(v1.98.0); Portkey gateway 12,821 stars / 1,267 forks (v1.15.2); RouteLLM ~5.5k stars (verified directly).

## 2.2 LiteLLM — the de facto execution layer (deep detail)

Verified from `docs.litellm.ai/docs/routing` and `/docs/proxy/config_settings` this session.

### Routing strategies (`router_settings.routing_strategy`)

| Strategy | Decision rule | Caveat from official docs |
|---|---|---|
| `simple-shuffle` (**default**) | Random pick, weighted by `rpm`/`weight` | "RECOMMENDED for best performance" |
| `least-busy` | Fewest active connections | |
| `latency-based-routing` | Lowest cached latency; args `ttl`, `lowest_latency_buffer` | Needs warm-up calls |
| `usage-based-routing` / `-v2` | Furthest from TPM/RPM limit | Docs explicitly warn: adds Redis latency on **every** request, "not recommended for production" |
| `cost-based-routing` | Cheapest `(in_tok × in_cost) + (out_tok × out_cost)` | **Async only**; unknown models get cost `$1` and are deprioritized |
| `weighted-pick` | RPM/TPM-aware weights | Manual weight management |

**Routing Groups** (newer feature): per-model-group strategies + callable virtual models:
```yaml
router_settings:
  routing_strategy: simple-shuffle      # fallback for ungrouped models
  routing_groups:
    - group_name: hot-path
      models: [gpt-5.6-terra, claude-sonnet]
      routing_strategy: latency-based-routing
      routing_strategy_args: {ttl: 60}
    - group_name: batch
      models: [gpt-5.6-luna, llama-70b]
      routing_strategy: usage-based-routing-v2
      routing_strategy_args: {rpm: 10000}
```
Also available **per key / per team** (docs `/docs/proxy/keys_teams_router_settings`): different routing
strategy, different fallback chains, different timeouts and retry policies per tenant.

### Fallback / resilience surface (this is the production pattern set to copy)

- `fallbacks: [{model: [alt1, alt2]}]` — general errors
- `content_policy_fallbacks` — ContentPolicyViolationError specifically
- `context_window_fallbacks` — token-limit errors specifically
- `default_fallbacks: [str]` — unhandled errors
- `max_fallbacks: 5` — cap chain length
- `retry_policy: {AuthenticationErrorRetries, TimeoutErrorRetries, RateLimitErrorRetries, ContentPolicyViolationErrorRetries, InternalServerErrorRetries}`
- `allowed_fails` / `cooldown_time` / `disable_cooldowns` — **circuit breaker per deployment**, not per group ("healthy peers keep serving while a failing one recovers")
- `allowed_fails_policy` — per-exception-type failure budgets
- `enable_pre_call_checks: true` — **pre-call context-window validation** to fail fast instead of paying for a doomed call
- `num_retries: 3`, `timeout: 600`, `stream_timeout`
- **Multi-instance coordination:** `redis_host/port/password/url` to share cooldowns + TPM/RPM counters across replicas
- `caching_groups` — models sharing caches
- `model_group_alias` — alias mapping
- `default_max_parallel_requests`, `default_priority`, `polling_interval: 0.003` (scheduler queue)

> These are the exact primitives a new package needs. Note that LiteLLM's `cost-based-routing` is
> **cost-only** — it does not model quality at all. That is precisely the gap a learned router fills.

## 2.3 Cloud-managed routers — real measured numbers

### AWS Bedrock Intelligent Prompt Routing
- Single serverless endpoint, **same model family only** (e.g. Claude Sonnet + Haiku, Nova Pro + Lite, Llama 3.3 70B + 3.1 8B).
- **Adds ~85 ms (P90) latency** per request. (AWS-published; relayed via medium/aws-tip.)
- Pricing conflict in sources — **unresolved**: `$1.00 per 1,000 routed requests` (cloudburn.io, DigitalOcean comparison) vs "no separate fee per routing call, pay per underlying tokens" (llmreference.com). Flag as unverified.
- AWS-reported savings: **up to 30%** (AWS docs, widely cited) vs **56% Anthropic family / 35% Nova / 16% Meta** (beri.net buyer's guide) vs **60%** vs uniformly using Claude Sonnet 3.5 v2 (internal AWS test per llmreference.com). All within-family.
- Independent hands-on (Reddit r/aws, 2026-06): ~**65%** savings at a ~70/30 easy/hard split on a RAG workload.
- Caveats from practitioners: English-only optimization; routing decisions **cannot** be tuned with app-specific performance data; must choose exactly **two** models from the same provider family.

### Microsoft Foundry Model Router
- **A trained language model deployed as a single Azure endpoint.** Modes: Balanced (default) / Cost / Quality, switched in the portal with no code change.
- Microsoft TechCommunity hands-on (published 2026-02-27, modified 2026-02-13), 10 prompts vs fixed GPT-5-nano:

| Metric | Balanced | Cost-Optimised | Quality-Optimised |
|---|---|---|---|
| Cost savings | ~4.5% | ~4.7% | ~14.2% |
| Avg latency (router) | ~7,800 ms | ~7,800 ms | ~6,800 ms |
| Models used across 10 prompts | 4 | prefers cheap | prefers premium |

> **This is the single most sobering data point in the whole research corpus.** A first-party, managed,
> trained router on real prompts delivered **4.5–14.2%** savings — an order of magnitude below the
> 85%/98% research headlines. Their own analysis: "savings scale with workload diversity"; the test set
> was small (10 prompts) and skewed. It confirms the RouterArena finding that real-world routable share
> is much smaller than MT-Bench implies.

## 2.4 vLLM Semantic Router — the most advanced open architecture

Two papers; both read this session.

**(a) arXiv:2510.08731 "When to Reason: Semantic Router for vLLM"** (Wang, Liu, Liu, Zhu, Mo, Jiang, Chen;
IBM Research / Tencent / U. Chicago). Submitted 2025-10-09. NeurIPS 2025 ML-for-Systems Workshop.
- Task: decide **reasoning vs non-reasoning** pathway (a routing decision, not model choice).
- Results on MMLU-Pro (14 domains, Qwen3-30B-A3B on an NVIDIA L4): **+10.2 pp accuracy, −47.1% latency, −48.5% tokens** vs direct vLLM inference.
- Improves in **11 of 14** domains; struggles in technical/reasoning-heavy ones.
- Implementation: **ModernBERT fine-tuned** intent classifier + **Rust** classification core + **Golang/Rust CGO** bindings into **Envoy `ext_proc`**.

**(b) arXiv:2603.04444 "vLLM Semantic Router: Signal Driven Decision Routing for Mixture-of-Modality Models"**
(Liu, Chen, Lu, Ovadia et al., 32 authors). v1 2026-02-23, v4 2026-06-03. Repo: `vllm-project/semantic-router`.
This is the **reference architecture for a production router** and should be studied closely:

- **Three-layer pipeline:** (1) signal extraction, (2) Boolean decision evaluation, (3) per-decision plugin chains.
- **Composable signal orchestration — 13 heterogeneous signal types**, spanning *sub-millisecond
  heuristics* (keyword patterns, regex, language detection, context length, role-based authorization)
  and *learned neural classifiers* (domain, complexity, embedding similarity, factual grounding, modality).
- **Demand-driven evaluation:** only signal types referenced by configured decisions are computed.
- **Decision engine:** `d* = argmax_{d ∈ D_match} conf(d, S(r))`; selection strategies include embedding
  similarity and classifier confidence. **13 semantic model selection algorithms** behind one interface.
- **LoRA multi-task architecture:** one base model (ModernBERT or mmBERT-32K) + n ~0.2 MB LoRA adapters.
  At n=6 → **~6× less model memory** vs 6 full model copies. Latency win comes from *parallel execution*
  of classifiers, not from LoRA itself.
- **Four inference runtimes:** Candle (GPU/CPU classification, LoRA, MLP), Linfa (CPU KNN/KMeans/SVM),
  ONNX Runtime (embeddings), NLP binding (BM25, n-gram). GPU vs CPU-only selection strategy documented.
- **mmBERT-32K + YaRN RoPE** extends 8,192 → 32K context while keeping backward compatibility with
  BERT-base LoRA adapters (≤512 tokens: no degradation).
- **Safety in the routing path:** jailbreak + PII classifiers launched as concurrent goroutines inside
  signal extraction; wall-clock = slowest evaluator, not the sum. Plus **HaluGate** three-stage
  hallucination detection (~50% cost reduction via sentinel filtering) and episodic memory with ReflectionGate.
- **Deployment:** Envoy External Processor — transparent interception, no client changes.
  Multi-provider backends: vLLM, OpenAI, Anthropic, Azure, Bedrock, Gemini, Vertex AI.
  `pip install vllm-sr && vllm-sr serve` bootstraps router + Envoy + dashboard.
- **Typed neural-symbolic DSL** (referred to as Athena in secondary sources) compiles routing policies
  to multiple deployment targets → "configuration-first adaptation without code changes."
- Reported: sub-10 ms signal extraction; routing accuracy **98.53%** with **805 training examples** and
  **2 hours of compute** (secondary source, gingerlabs.ai — flag as medium confidence).
- Semantic caching layer uses **HNSW** vector similarity (catches paraphrases, not just exact matches).

> **Key design lesson:** the winner in 2026 is not "a smarter classifier" — it is **a signal bus + a
> policy DSL + a plugin chain**, where cheap deterministic signals are always evaluated first and
> neural signals are evaluated lazily and in parallel.

## 2.5 Not Diamond / RoRF (verified from GitHub + their blog)

- `Not-Diamond/RoRF`, ~200 stars, PyPI `rorf`. **Random Forest classifier on embeddings** (jina-embeddings-v3 or voyage-large-2-instruct).
- Ships **12 pre-trained routers** across 6 model pairs × 2 embedding models.
- Two regimes: strong/weak pair → cost reduction while maintaining strong-model performance;
  **two-strong pair → beats both individual models** while reducing cost.
- Their reported peak: **12.5% cost reduction vs Claude 3.5 Sonnet** (blog, 2024-09-25).
- Explicit threshold parameter to tune cost/performance; "outperforms RouteLLM at a lower cost" on MMLU and BigBenchHard (their claim).
- Hosted product: custom routers trained on **your** eval data (needs ≥15 samples per morphllm.com),
  `tradeoff: "cost"` mode, LangChain integration, routes between **custom RAG agents** (not just models),
  powers **OpenRouter "Auto"** mode.
- Published added latency **100–150 ms** average; published savings **"at least 20–40%"** — notably below
  the marketing figures circulating. Pricing: fixed fee per M tokens routed, **rate not public**
  (beri.net cites ~$0.05/1M routed tokens; NotDiamond pricing page does not publish it).

## 2.6 Aurelio semantic-router (the embedding decision layer)

- MIT, ~3.7k stars. Routes by **embedding similarity to per-route example utterances** — no training, no LLM call.
- Encoders: OpenAI, Cohere, HuggingFace, FastEmbed, LiteLLM-backed (verified in `cohere.py` source).
- Vector indexes: Pinecone, Qdrant, or in-memory. Scales to thousands of routes.
- **Threshold is the load-bearing hyperparameter.** Ships `fit()` / `evaluate()`: pass (utterance, route)
  pairs and it optimizes per-route `score_threshold` in seconds. Their own docs show accuracy jumping
  **34.85% → 89.39%** after `fit()` on 500 examples (default 0.5 thresholds → optimized 0.05–0.32).
- Documented latency band: 5–20 ms typical; deepchecks reports "5000ms → 100ms" vs LLM-based decisions.
- Supports dynamic routes (function-call/parameter generation) and multi-modal routes.
- `semroute` (PyPI) is a smaller alternative with `static` vs `dynamic` thresholding and `centroid` scoring.

## 2.7 What no framework does well yet (the gaps = the opportunity)

From triangulating all sources above:

1. **N>2 model routing with quality models.** RouteLLM is binary. RouterEval says 3–10 candidates is the
   cost-optimal sweet spot. Most gateways do cost-only, not quality-aware, selection.
2. **Routers as retrainable artifacts.** RouteLLM went stale in 24 months. Nothing ships a refresh loop.
3. **Online/adaptive learning in production.** PILOT, BaRP, LLM-Bandit, PROTEUS are all research; WISERouter
   notes "PILOT is not publicly available."
4. **Budget pacing over an open-ended stream.** arXiv:2604.00136 (2026-03) identifies precisely this gap:
   "PILOT assumes a known horizon; PROTEUS freezes its dual variables at deployment."
5. **Router-side observability.** No open router ships per-route quality monitoring, routing-mix drift
   detection, or escalation-rate alerting out of the box.
6. **Honest evaluation on your own traffic.** Every savings number in this space is self-measured.
