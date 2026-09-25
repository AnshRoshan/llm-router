# LLM Router

Cache-aware LLM routing: pick the **(model, backend) pair**, keep sessions
sticky so the prompt cache survives, and choose prompt-cache TTL **per
breakpoint**.

| Part | What it is | Install |
|---|---|---|
| [`llmrouter/`](llmrouter/) | Python package: full decision + execution + learning layers | `pip install llmrouter[http]` |
| [`npm/llmrouter/`](npm/llmrouter/) | Node.js package: the decision engine, 1:1 port, zero deps | `npm install llmrouter` |
| [`website/`](website/) | Docs site: what it is, how both packages work | open `website/index.html` |
| [`research/`](research/) | The research corpus the implementation is grounded in | start at `MASTER-REPORT.md` |

Both packages make the **same decision** for the same request: the price
cards, the scoring formulas (`R*` amortization, `E*` expiry threshold, mixed
TTL policy), the learned-quality checkpoints (`llmrouter-quality-v1`) and the
durable-state snapshots (`llmrouter-state-v1`) are shared artifacts, and the
cross-runtime parity suite asserts the decisions match byte-for-byte.

```bash
# Python — decisions, async execution, CLI
python -m llmrouter explain --prompt "hi" --workload chat

# Node.js — decisions, CLI
npx llmrouter explain --prompt "hi" --workload chat

# the loop: collect feedback -> train quality heads -> route with the checkpoint
python -m llmrouter train --data feedback.jsonl --out quality.json
python -m llmrouter eval --data cases.jsonl --ci --quality-model quality.json

# decision sidecar for LiteLLM / Bifrost / your own proxy (both runtimes)
python -m llmrouter serve --port 8787        # POST /decide
npx llmrouter serve --port 8787              # same wire format

# local GPU + cloud scored together, with honest $/MTok GPU economics
python -c "from llmrouter import Router; Router.with_ollama(openai_key='...')"
```

The Python engine also **streams** (`router.astream`, failover safe before the
first token) and can **persist every piece of learned state** (sessions, hit
rates, circuit health, pair economics) across restarts with
`Router.with_defaults(state_store=FileStateStore("state.json"))`.

Apache-2.0. The honest savings expectation is 30-40% at no more than 2%
quality loss; ship the `eval --ci` gate and measure your own traffic.
