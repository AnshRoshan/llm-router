# llmrouter

Cache-aware LLM routing for Python. Picks the **(model, backend) pair**, keeps
conversational sessions sticky so the prompt cache survives, and chooses
prompt-cache TTL **per breakpoint**.

```bash
pip install llmrouter          # decision engine, zero dependencies
pip install "llmrouter[http]"  # + httpx, to actually call providers
```

## Why another router

Most routers answer one question — *which model?* — by scoring difficulty. This
one answers three, because the cost of a request is not decided by the model
alone:

| Question | What it depends on |
|---|---|
| Which **model**? | workload class, quality bar, budget |
| Which **backend**? | self-hosted vs API vs serverless, live health, residency |
| Which **cache TTL**? | expected idle gap, generation time, prompt segment |

The third one is where the money is. Prompt caches are **model-scoped and
non-transferable**, so a router that switches models mid-conversation throws away
a cache it already paid for — and can easily cost *more* than not routing at all.

## Quick start

```python
import asyncio
from llmrouter import Message, Router

router = Router.with_defaults(anthropic_key="sk-...", openai_key="sk-...")

async def main():
    out = await router.acomplete(
        [Message("system", "You are a support bot."), Message("user", "hi")],
        session_id="user-42",
        workload="chat",
    )
    print(out.text)
    print(router.explain())

asyncio.run(main())
```

`explain()` is not optional. Every decision reports the model, backend, workload
class, how it was inferred, the TTL chosen per segment, and the full scored
candidate list:

```
model=claude-sonnet-class backend=anthropic workload=chat(src=explicit_tag, conf=1.00) sticky=True switched=False stranded=False
  ttl: system=1h, history=5m
  claude-sonnet-class@anthropic score=+0.7164 q=0.850 cost=$0.00431 lat=900ms aff=0.80 switch=$0.00000
  gpt-mini-class@openai score=+0.7007 q=0.600 cost=$0.00121 lat=450ms aff=0.00 switch=$0.00250
  · workload=chat via explicit_tag: caller declared workload
  · mixed TTL: system=1h, history=5m
```

## The two entry paths

**Path A — declared intent.** You pass `workload="agent"`, or select a named
agent profile. Costs nothing and skips classification entirely.

```python
router = Router.with_defaults(
    anthropic_key="sk-...",
    profiles={"triage": AgentProfile("triage", WorkloadClass.AGENT)},
)
await router.acomplete(msgs, agent_profile="triage")
```

**Path B — inferred intent.** Declare nothing and the router works it out:

1. **structural rules** (~0 ms) — `tools` + a prior `tool_result` ⇒ agent;
   retrieved chunks ⇒ RAG; image block ⇒ vision; code fences ⇒ code
2. **prompt fingerprint** (~0 ms) — hashes your system prompt + tool schemas.
   Production traffic comes from a handful of distinct application prompts, so
   this turns "classify every request" into "classify every *application*". The
   second request through a given prompt is an exact, free hit.
3. **classifier hook** — only when 1 and 2 abstain, and **only on turn 1**. The
   cost is paid once and amortized over the session; per-*step* classification is
   deliberately not supported.
4. **default** — never fails to route; logged as `default` so it stays auditable.

## What makes it different

### 1. It scores (model, backend) pairs

Workload class and deployment class compose instead of being decided
sequentially:

```
score(model, backend) = w_q · quality
                      − w_c · cost
                      − w_l · latency
                      − w_s · switch_cost      ← cache rewrite if you leave
                      + w_a · affinity_bonus   ← reward for staying warm
```

### 2. The cache is a first-class signal — and it is *measured*

The cache lives on the provider's side, so the router parses the usage counters
out of every response (`cache_read_input_tokens` / `cached_tokens`) and folds
them into a per-session EWMA. The affinity bonus is learned, not assumed.

### 3. Session stickiness with an amortization test

Never oscillate: four legs of A→B→A→B on a 50k prefix costs **more than never
routing at all**. A one-way migration is evaluated against

```
R* = (c'·w' − c·r) / ((c − c')·r)      # turns needed to pay back
```

which is why the policy is provider-specific: Anthropic charges a 1.25× cache
write, **OpenAI charges none** — so with the bundled cards a migration from
sonnet-class to haiku-class needs **4.75 turns** to pay back, while the same
migration onto `gpt-mini-class` needs only **2.00**. On a cheaper OpenAI target
the pay-back can fall inside a single turn.

Note the asymmetry: a migration onto a no-write-premium provider is nearly free,
while oscillating between two Anthropic models is not. Four legs of A→B→A→B on a
50k prefix costs **$0.75** against **$0.62** for never routing at all.

If a migration cannot pay back in the turns remaining, the router stays put and
increments a **`stranded`** counter, so you can see the money it wanted to save
and could not.

### 4. TTL chosen per prompt segment

```
tools + system + retrieved  →  1h    (stable, reused every turn, survives a pause)
conversation history        →  5m    (append-only; paying 2× on it is waste)
```

The decision uses `expected_gap_s + expected_generation_s`, because Anthropic
measures the cache lifetime **from the start of the request** — a 4-minute stream
leaves only ~1 minute for the follow-up.

## Delegation: better than switching models

Mid-session model switching destroys the prompt cache. **Delegation does not.**

Following Cognition's Fusion architecture, the lead and a cheaper sidekick run as
two agents with two separate contexts and two separate warm caches. Only a
*brief* crosses the boundary, so neither prefix ever changes for the other's sake
— both stay hot for the whole session.

```python
router = Router.with_defaults(anthropic_key="sk-...", sidekick_model="gpt-mini-class")

decision = router.should_delegate(request, task="apply the patch")
if decision.should_delegate:
    result = await router.adelegate(decision.brief, lead_session_id="user-42")
```

### Route on task phase, not difficulty

Difficulty is not a property of the prompt — it's a property of the
investigation. "Fix xyz bug" may be a one-liner or a rearchitecture, and you
cannot know until you've read the code. **Phase is observable** from the
tool-call history:

| Phase | Delegable? | Turn share (FrontierCode) |
|---|---|---|
| Plan | **no** — lead only | 32% |
| Setup | yes | 34% |
| Implement | yes | 5% |
| Debug | configurable | 17% |
| Validate | yes | 9% |

The non-obvious rule, from Cognition: **exploration that shapes the plan must
stay with the lead.** A weaker model may not judge what information matters, and
delegating it starves the lead's context — which the lead needs in its own cache
to plan well. So `PLAN` is never delegable, and `infer_phase` falls back to
`PLAN`, making the default the conservative one.

A stronger sidekick earns more latitude — terser briefs and permission to push
back, which catches mistakes in the lead's plan. A weaker one gets prescriptive
briefs and no opinion.

### Cost belongs to the pair, not the model

Cognition measured that a sidekick costing **275% more per token** made the
system **2% cheaper**, because stronger sidekicks need fewer attempts and
generate fewer review rounds. A price registry keyed by model cannot express
this, so `PairRegistry` sits on top of it:

```python
router.pairs.record_outcome("claude-sonnet-class", "gpt-mini-class",
                            cost=1.20, score=60.0)
router.pairs.best_by_price_per_task()   # → the cheapest pair, by measurement
```

**Evaluate on price per task, not price per token.** This is not just reporting —
measured pair rework feeds back into the scorer, so a model that needs three
attempts stops getting picked.

Two things make that loop actually close:

- **`adelegate` runs on the model you configured.** A delegated task is pinned to
  `sidekick_model` (via a restricted internal agent profile, so it still passes
  through hard constraints, health checks, and failover). It does *not* silently
  route to whatever scores highest — that would defeat the entire point of
  delegation. The pin is a property, so it holds whether you pass `sidekick_model`
  to `Router(...)` or set `router.sidekick_model = ...` afterwards; setting it to
  `None` removes it, and an unknown model raises `ConfigurationError`.
- **`adelegate` auto-records the outcome.** After each delegated run it folds the
  provider's real usage counters and the attempt count back into the pair
  profile. A sidekick that needed a failover (2 attempts) raises its
  `effective_sidekick_multiplier`, and the *next* decision sees it. No manual
  `record_outcome` call required — though you can still call it to inject your own
  measurements (`cost`, `score`, `attempts`, `rework_factor`).

You can also set the factors directly on a price card when you have your own
measurements:

```python
from dataclasses import replace
from llmrouter import PriceCard

router.prices.register(replace(
    router.prices.require("gpt-mini-class"),
    token_efficiency=1.4,   # needs 40% more tokens per unit of work
    rework_factor=1.8,      # output usually needs a correction pass
))
# effective cost per task = sticker × token_efficiency × rework_factor  → 2.52×
```

`token_efficiency` and `rework_factor` default to `1.0`, so nothing changes until
you calibrate them on your own traffic. Pass `lead_model=` to `Router` to have
pair profiles scale the sidekick's cost during scoring.

## Pure decision mode

The decision layer has **zero dependencies** and does no I/O, so you can use it
to drive your own HTTP client, or in a cold-start-sensitive path:

```python
from llmrouter import Message, RoutingRequest

decision = router.decide(RoutingRequest(
    messages=(Message("user", "hi"),), session_id="user-42",
))
decision.model_id, decision.backend_id, decision.ttl_by_segment
```

## The learning layer: `explain` / `calibrate` / `eval` / `train` / `serve`

A router you cannot interrogate is a router you cannot trust. The CLI runs the
pure decision layer — no keys, no network:

```bash
python -m llmrouter explain --system "You are a support bot" --prompt "hi"
python -m llmrouter eval --data cases.jsonl --ci --quality-floor 0.75
python -m llmrouter calibrate --data cases.jsonl --target-strong-pct 0.5
python -m llmrouter train --data feedback.jsonl --out quality.json
python -m llmrouter serve --port 8787   # HTTP decide-sidecar for your gateway
```

- **`explain`** routes one prompt and prints the full auditable decision —
  candidates, scores, TTL choices, and the reason the workload class was chosen.
- **`eval`** runs a JSONL case file (`{"text": ..., "workload": ...}` per line,
  optionally `expected_model` for a misroute audit) and reports routing mix,
  estimated cost, average quality, and decision latency. With `--ci` it **exits
  non-zero** when quality drops below the floor, cases become unroutable, or the
  decision p50 exceeds 10 ms — the pre-merge gate for any routing change.
- **`calibrate`** sweeps the quality floor (`Weights.min_quality`, the router's
  operating point) and reports the value that achieves your target strong-model
  share — the RouteLLM `calibrate_threshold` pattern, generalized past two
  models.

## Hard constraints and budget pacing

The decision layer enforces what the caller declares, before any call is made:

```python
from llmrouter import Constraints, Message, RoutingRequest

req = RoutingRequest(
    messages=[Message("user", "hi")],
    constraints=Constraints(
        max_cost_usd=0.01,            # per-turn cap: cheapest violator wins if nothing fits
        budget_remaining_usd=5.00,    # pacing: projected session spend over budget is penalized
        residency_tags=frozenset({"eu"}),   # fail closed — no compliant model, no call
    ),
)
```

Prompts are also validated against each candidate's **context window before
calling** (with headroom reserved for the response), so a too-big request routes
to a model that can actually serve it instead of failing at the provider.

## Rule F: cache-invalidating drift

A provider's cache key covers more than the prefix text — tool schemas, image
usage, thinking/effort settings. If those change **within a live session**, the
cache is gone and raising `thinking.effort` on the same model invalidates it as
thoroughly as switching models. The router fingerprints those fields per
session; when they drift, every candidate is scored cold for that turn (no
affinity bonus, no switch-cost asymmetry), and a reason says so. Escalating by
changing an invalidating field is therefore priced honestly.

## Circuit breaker: recovery through a canary

After `allowed` consecutive failures the circuit opens with exponential
backoff. When the backoff elapses the backend is **half-open**: exactly one
probe request is admitted, and full recovery waits for it to succeed. A failed
probe re-opens the circuit and revokes the slot. The canary is a real request
that was going out anyway — there is no health-check call to the provider.

## Self-hosted and mixed deployments

Register your own backends. `cost_factor` is your marginal cost relative to list
price; `capacity_rps` makes the router shed load rather than queue into a
cascade.

```python
from llmrouter import BackendSpec, DeploymentClass, SELF_HOSTED_CACHE, PriceCard

router.prices.register(PriceCard(
    "llama-70b-local", 0.0, 0.0, SELF_HOSTED_CACHE,
    quality_prior={"chat": 0.72, "code": 0.68},
))
router.backends.register(BackendSpec(
    "vllm-gpu-0", DeploymentClass.SELF_HOSTED,
    models=("llama-70b-local",), base_url="http://gpu-0:8000",
    priority=0, timeout_s=8.0, capacity_rps=40, cost_factor=0.0,
))
```

Keep local timeouts aggressive. A 10× spike that saturates a GPU times
everything out and cascades the whole lot to the cloud at ~100× the cost.

## Observability

```python
router.stats()
# {'decisions': 1204, 'estimated_cost_usd': 3.41, 'misroutes': 2,
#  'sessions': {'sessions': 88.0, 'switched': 3.0, 'stranded': 11.0,
#               'avg_hit_rate': 0.87},
#  'backends': {...}, 'fingerprints': 7, 'volatile_clients': []}
```

`volatile_clients` flags callers whose prompts embed a timestamp or request id —
they generate unbounded distinct fingerprints, so their cache never hits. That
warning is a cache-hit-rate bug report for *their* prompts.

The metric that matters most: **cache hit rate broken down by routing decision**.
If the hit rate for pinned sessions is not materially higher than for the rest,
the affinity logic is not working. `stats()["sessions"]["hit_rate_by_source"]`
breaks it down by how the workload class was learned.

## Learned quality: `train` closes the loop

The price-card `quality_prior` is a placeholder until you measure your own
traffic. `llmrouter train` fits one logistic head per model from feedback
JSONL (pointwise `{"text","model","outcome"}` or pairwise
`{"text","model_a","model_b","winner"}` — including the rows
`router.export_feedback()` writes from `report_feedback`) and emits a frozen
checkpoint:

```python
from llmrouter import Router, QualityModel
router = Router.with_defaults(anthropic_key=..., quality_model=QualityModel.load("quality.json"))
```

The blend is `quality = clamp(prior + λ·(b + w·x))` with `λ = n/(n+k)`:
a model with no feedback behaves exactly like before; the learned delta only
talks as loudly as its sample count earns. The feature vector is 17 bounded
numbers derived from the request shape — the inference is 17 multiply-adds,
so it stays inside the zero-dependency cold-path rule. The Node engine loads
the same checkpoint and decides identically (asserted by `scripts/parity.mjs`).

## Streaming

`router.astream(...)` is the twin of `acomplete()`: it yields `StreamChunk`s —
text deltas, then a final chunk carrying the `Completion` with the provider's
real usage counters, so session affinity and the learning loop work on the
stream path. Failover is honest about the boundary: an upstream failure is
retried on the next backend **only while nothing has reached the caller**; a
break after the first token raises instead of duplicating rendered text.

## Durable state

All learned state — session pins and hit-rate EWMAs, circuit health, prompt
fingerprints, (lead, sidekick) pair economics — fits in one JSON snapshot:

```python
from llmrouter import Router, FileStateStore
router = Router.with_defaults(..., state_store=FileStateStore("state.json"))
# autosaves every 25 decisions; rebases TTLs on restore; router.save_state()
# on shutdown
```

Without it, a restart converts a warm cache into a cold one the router thinks
is cold. The format (`llmrouter-state-v1`) is shared with the Node engine, so
replicas in either language can follow the same file.

## The decide-sidecar (`serve`)

Running your own gateway (LiteLLM, Bifrost, Envoy, your proxy)? Point it at
`llmrouter serve`: `POST /decide` with a routing-request JSON body answers the
full Decision (pair, TTLs, candidate scores, reasons); `GET /stats` and
`GET /healthz` complete the surface. Binds localhost by default, `--token` for
bearer auth, `--state` to keep affinity durable, and a following request's
`observed_usage` feeds the provider counters back into the learned hit rate —
decide with us, execute wherever.

## Other runtimes

The decision engine is also available for Node.js: [`npm/llmrouter`](../npm/llmrouter)
(`npm install llmrouter`) is a zero-dependency TypeScript port of this package's
decision layer — same price cards, same formulas, same decisions, asserted by a
paired test suite. The documentation site in [`website/`](../website) covers
both packages side by side.

## Development

```bash
pip install -e ".[dev]"
pytest            # 170 tests
```

## Caveats

- **Prices go stale.** The bundled `$/token` figures are illustrative starting
  points — override them with your contracted rates. The *cache multipliers*
  (1.25× / 2.0× / 0.1×, and OpenAI's no-write-premium) were verified against
  primary docs on 2026-09-12.
- **Quality priors are placeholders.** Calibrate them on your own traffic; the
  bundled numbers are not benchmarks.
- **The session store is in-process.** For multi-process deployments, back it
  with Redis — affinity is only useful if the state is shared.
- **TTLs do not translate across providers.** A `cache_control` ttl is dropped
  toward OpenAI, which has its own `prompt_cache_options`. The TTL decision is
  backend-specific.

## License

Apache-2.0
