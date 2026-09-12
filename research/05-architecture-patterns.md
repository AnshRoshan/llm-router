# Thread 5 — Production Architecture Patterns

Research date: 2026-09-12. What production routers actually do, extracted from official docs and
architecture write-ups read this session.

## 5.1 The reference request path

Consolidated from LiteLLM's documented architecture, SGLang Model Gateway docs, and three independent
production-architecture write-ups (markaicode.com, 2026-07):

```
Client
  └─ Gateway        TLS, auth, virtual keys, rate limit (Redis sliding window), tenant resolution
      └─ Policy     budget check, per-key/team routing strategy override, hard exclusions
          └─ Router DECISION: signals → policy → model choice     ← the novel part
              └─ Execution   retries, circuit breaker, cooldown, fallback chain
                  └─ Provider pool (OpenAI / Anthropic / Azure / Bedrock / local vLLM / Ollama)
      State: Redis (rate limits, cooldowns, latency windows) + Postgres (keys, spend, audit)
      Observability: OpenTelemetry traces + Prometheus metrics at every hop
```

**LiteLLM's production shape (verified):** "a stateless proxy tier (ASGI workers behind a load balancer)
that calls out to a built-in router for provider selection, retries, and fallbacks, backed by Postgres
for virtual keys/spend data and Redis for cross-instance rate-limit and cooldown state. You scale it by
adding replicas — not by adding a separate task-queue service."
- Match `--num_workers` to CPU cores per pod.
- **Redis is optional for a single instance but required once you run >1 replica** and need shared
  rate-limit/cooldown state. This is *the* distributed-systems requirement for routers.
- Known weakness: "router-level retries keep the proxy dependency-free, but a slow provider still
  occupies the worker handling that request until a fallback or timeout fires."

## 5.2 Multiple model backends — the adapter question

| Approach | Who does it | Assessment |
|---|---|---|
| **Delegate to LiteLLM** | RouteLLM does exactly this ("We leverage LiteLLM to support chat completions from a wide-range of open-source and closed models") | Lowest effort, inherits 100+ providers, but inherits LiteLLM's dependency weight and release cadence |
| **OpenAI-compatible + base_url** | RouteLLM: prefix model with `openai/`, set `--base-url` + `--api-key` | Covers vLLM, SGLang, Ollama, Together, Groq, most of the market in ~20 lines |
| **Per-provider native adapters** | LiteLLM, Portkey | Only needed for provider-specific features (Bedrock guardrails, Anthropic prompt caching, Gemini system instructions) |
| **Envoy `ext_proc`** | vLLM Semantic Router | Transparent interception, zero client change, Kubernetes/service-mesh native. Heaviest to operate. |

> **Recommendation-shaped finding:** an **abstract `ModelBackend` protocol with an OpenAI-compatible
> default implementation** covers ~90% of the market. Do not write 20 provider adapters; do not hard
> -depend on LiteLLM either — make it an optional extra (`pip install yourrouter[litellm]`).

## 5.3 Fallback logic — the complete primitive set

From LiteLLM official docs (`router_settings`), the primitives a production router needs:

**Three distinct fallback channels, not one:**
- `fallbacks` — general errors
- `content_policy_fallbacks` — ContentPolicyViolationError
- `context_window_fallbacks` — token-limit exceeded
- `default_fallbacks` — anything unhandled
- `max_fallbacks: 5` — **cap the chain** (prevents fallback loops)

**Per-exception retry budgets** (`retry_policy`): separate counts for Authentication / Timeout /
RateLimit / ContentPolicyViolation / InternalServerError. A blanket `num_retries: 3` is wrong — you
should retry a 429 many times and a 401 once.

**Circuit breaker, per deployment not per model group:** `allowed_fails`, `cooldown_time`,
`allowed_fails_policy` (per-exception-type failure budgets). "Cooldowns isolate individual deployments,
not entire model groups — healthy peers keep serving while a failing one recovers."

**Pre-call validation:** `enable_pre_call_checks: true` catches context-window errors *before* paying
for a doomed call, and can trigger `context_window_fallbacks` automatically.

**Context-aware parameter stripping:** LiteLLM ≥ v1.44 strips unsupported params (e.g. `response_format`)
when switching models so the fallback doesn't itself fail. **This is a subtle production bug that
everyone hits.** Without it: `litellm.BadRequestError after fallback`.

**Documented failure modes to design against (markaicode, 2026-05-25):**
| Symptom | Cause | Fix |
|---|---|---|
| Fallback never triggers | `num_retries: 0`, or error type not covered | set retries; verify error class |
| Fallback model returns poor output | different capability | pick a similar-capability fallback |
| Latency spikes on fallback | extra hop + cold start | `cooldown_time`; streaming |
| `BadRequestError` after fallback | incompatible params not stripped | upgrade; enable context_window_fallbacks |
| **Fallback loops** | multiple failed deployments, no cooldown | `allowed_fails` + `cooldown_time` |

**Circuit breaker at scale (markaicode, 2026-07-24):** "Centralize state in Redis once you run more than
one replica — an in-process breaker per pod means each pod trips independently, which delays a clean
fast-fail across the fleet." Recovery should be gated behind a **cheap canary probe on HALF_OPEN**, not
blind retries.

**Health-check probe hygiene:** "LLM proxy liveness probe should NOT call the actual LLM provider —
check that the circuit breaker library and queue consumer are running instead, or you'll fail health
checks during a provider outage that your circuit breaker is already handling correctly."

## 5.4 Scaling across distributed systems

| Concern | Pattern found | Source |
|---|---|---|
| Shared rate limits / cooldowns | **Redis** (required >1 replica) | LiteLLM docs |
| Shared latency windows | `routing_strategy_args.ttl` (e.g. 3600s) in Redis | LiteLLM docs |
| Stateless proxy tier | ASGI workers behind LB, scale by replicas, HPA on CPU / queue depth | LiteLLM arch |
| Spend / virtual keys / audit | **Postgres** | LiteLLM arch |
| Request durability + backpressure | **Redis Streams with consumer groups**, stateless workers | LangChain inference arch |
| Two-tier cache | Redis (hot, TTL minutes) + Postgres (cold, TTL hours) | LangChain inference arch |
| Autoscaling signal | queue depth + CPU (not just CPU) | multiple |
| Router-level throughput | SGLang Model Gateway: **40+ Prometheus metrics** across HTTP/router/worker/circuit-breaker/retry/discovery/MCP/DB; OTel OTLP export; request-ID propagation | sglang docs |
| Router decision cost at scale | arXiv:2604.00136: **9.8 ms end-to-end on CPU, 22.5 µs for the decision itself** | 2026 |
| Gateway overhead comparison | Bifrost 11 µs @5K RPS (Go) · Helicone ~1–5 ms P95 (Rust) · **LiteLLM ~40–50 ms, ~200 req/s, 372 MB (Python)** · TensorZero <1 ms P99 (Rust) | getmaxim.ai 2026-05 (vendor comparison — medium confidence) |

> **The Python-language overhead is real and measured: ~40–50 ms and ~200 req/s for LiteLLM vs <1 ms
> and 10K QPS for Rust gateways.** For a Python router package this means: keep the router's *own*
> overhead in the low single-digit ms, use async I/O throughout, and don't try to out-gateway the Rust
> gateways — out-decide them.

## 5.5 Observability — what to ship

**Standards:** OpenTelemetry (OTLP) + Prometheus is the 2026 table stakes. Arize Phoenix, OpenLLMetry
(7.2k+ stars), Langfuse (MIT), MLflow, Opik (Apache 2.0) all consume OTLP.

**Per-request data points worth capturing** (Portkey records **40+ per request**; use that as the bar):
routing decision + all candidate scores + which signal fired + threshold used + model actually called +
fallback path taken + tokens in/out + cost + latency per stage + cache hit + guardrail result + tenant/key.

**Router-specific metrics (the gap no one fills):** see `04-cost-benchmarks-eval.md` §4.3.

**The evaluation loop:** Braintrust's differentiator is "production traces become evaluation cases with
one click, then run through scoring and CI/CD gates." **A router that can't turn a misroute into a
training example is a router you can't improve.** Design the logging schema for this from day one.

## 5.6 Semantic caching (compounds with routing)

- **Exact-match caching**: LiteLLM (Redis-backed). Cheapest, lowest hit rate.
- **Semantic caching (HNSW)**: Portkey, Bifrost, Helicone, vLLM-SR. Catches paraphrases, not just
  exact matches. mindstudio: "eliminates 30–50% of redundant API calls."
- **Prompt caching (provider-side)**: Bedrock cache read is **0.1× the standard input price** (cache
  write 1.25× at 5-min TTL, 2× at 1-hour TTL). Anthropic/OpenAI equivalents similar.
  effloow's worked example: a repeated 40k-token prefix goes from ~$15k/mo-class to ~$1.5k/mo at 0.1×
  with a high hit rate — **zero model changes, zero quality risk.**

> **Ordering insight from effloow (2026-08-21):** the three cost layers, in the order you should apply
> them, are (1) server-side context pruning, (2) prompt caching, (3) hybrid routing. Layers 1–2
> "require no change to which model you use" and carry no quality risk. Routing is layer 3 and changes
> the *slope* of the cost curve. **A router package that also does semantic caching and prefix-stability
> advice captures more value than a router alone.**

## 5.7 Safety in the routing path (vLLM-SR's contribution)

vLLM-SR puts jailbreak and PII classifiers **inside** signal extraction, launched as concurrent
goroutines, so wall-clock = slowest evaluator rather than the sum. Plus **HaluGate** three-stage
hallucination detection (~50% cost reduction via sentinel-based filtering) and per-decision thresholds.

**Why this matters for a Python package:** guardrail results should be *signals that can veto a cheap
route*, not a separate pre-processing step. If a request trips PII, the routing policy must be able to
say "force to the on-prem model" without a second pass.

## 5.8 Known production gotchas catalogue (compiled from 2026 practitioner sources)

1. **Safety-critical queries that look simple** — hard-exclude from cheap routing (financial, medical,
   regulatory, account compromise, already-escalated sessions, turns after a tool failure).
2. **Routing drift** as the product changes — new features create query types the classifier never saw.
   Watch routing-mix weekly.
3. **Calibration decays after any model update** — provider silently swaps a model version and your
   thresholds are wrong. Re-calibrate on model-version change events.
4. **The temptation to classify with an LLM** — see §3.3.
5. **Cold-model latency** — 3–15 s to load weights on a cold self-hosted tier. Keep every routed tier warm.
6. **Silent fallback failure** — "a misconfigured fallback that fails silently costs more than no
   fallback at all" (developersdigest). Test fallbacks before you need them; watch traces.
7. **Aggregate metrics hide per-route regression** — always segment quality by route.
8. **Routing doesn't fix a badly prompted agent** — "a cheaper 35% transfer rate, not a better experience."
9. **`usage-based-routing` in LiteLLM adds Redis latency on every request** — official docs warn against it in production.
10. **LiteLLM's 2026 CVE history** — devtoollab (2026-07-28) advises staying current on patch versions. **Security posture is a differentiator for a new package.**
