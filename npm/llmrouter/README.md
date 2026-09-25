# llmrouter (Node.js)

Cache-aware LLM routing for JavaScript/TypeScript. Picks the **(model, backend)
pair**, keeps sessions sticky so the prompt cache survives, and chooses
prompt-cache TTL **per breakpoint**. **Zero runtime dependencies.**

This is the decision engine of the Python
[`llmrouter`](../../llmrouter) package, ported 1:1: same price cards, same
formulas (`R*` amortization, `E*` expiry threshold, TTL policy), same scoring.
The two runtimes make the **same decision** for the same request — including
when a **learned quality checkpoint** (`llmrouter train` on the Python side)
is loaded: `QualityModel.fromJson(...)` consumes it here and the parity suite
asserts identical output. State snapshots (`llmrouter-state-v1`) are also
cross-runtime: `loadSnapshotFile` / `saveSnapshotFile` restore sessions, hit
rates and circuit health from either language. `npx llmrouter serve` runs the
same HTTP decide-sidecar as Python (`POST /decide`).

```bash
npm install llmrouter
```

## Quick start

```js
import {
  Message, RoutingRequest, WorkloadClass, PolicyEngine, PriceRegistry, BackendRegistry,
} from "llmrouter";

const engine = new PolicyEngine(new PriceRegistry(), new BackendRegistry());

const decision = engine.decide(new RoutingRequest(
  [new Message("user", "refactor this function")],
  WorkloadClass.Code,
  "user-42",                                  // session id — keeps the cache warm
));

decision.modelId;      // "gpt-frontier-class"
decision.backendId;    // "openai"
decision.ttlBySegment; // Map { "system" => "1h", "history" => "5m" }
console.log(decision.explain());   // the full auditable decision
```

No network, no keys — this is the pure decision layer. Drive your own HTTP
client from the decision, or use the Python package for the full
execute/failover/learn loop.

## What it computes

- **(model, backend) pair scoring** with quality, cost, latency, a
  `switch_cost` term (the cache you throw away by leaving) and an
  `affinity_bonus` (the measured hit rate of staying).
- **Session stickiness** via rendezvous hashing with a no-oscillation lock:
  a migration must pay back in `R* = (c'·w' − c·r) / ((c − c')·r)` turns.
- **TTL per prompt segment**: system/tools/retrieved at 1h, conversation
  history at 5m — the highest-value cache lever almost nobody ships.
- **Rule F**: tool-schema or image drift inside a session scores every
  candidate cold, because it invalidates the provider's cache key.
- **Hard constraints**: quality floor, context window (pre-call), residency
  (fail closed), budget cap + closed-loop budget pacing.
- **Workload-class inference** in two entry paths: declared (free) or inferred
  via structural rules → prompt fingerprint → classifier hook → default.
- **A circuit breaker** per backend with canary-gated half-open recovery.

## CLI

```bash
npx llmrouter explain --prompt "hi" --workload chat
npx llmrouter eval --data cases.jsonl --ci --quality-floor 0.75
npx llmrouter calibrate --data cases.jsonl --target-strong-pct 0.5
```

`eval --ci` exits non-zero when quality drops below the floor, cases become
unroutable, `expected_model` disagrees, or decision p50 exceeds 10 ms — the
pre-merge gate for routing changes.

## Not in the JS port (yet)

Delegation (lead/sidekick briefs), the async execution layer (retries,
failover, adapters), and provider usage-counter parsing live in the Python
package today. The decision engines are kept numerically identical by paired
test suites.

## Development

```bash
npm install
npm test        # builds, then runs the node:test suite
```

## License

Apache-2.0
