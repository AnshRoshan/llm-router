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
TTL policy) and the tests asserting them are kept identical across runtimes.

```bash
# Python — decisions, async execution, CLI
python -m llmrouter explain --prompt "hi" --workload chat

# Node.js — decisions, CLI
npx llmrouter explain --prompt "hi" --workload chat
```

Apache-2.0. The honest savings expectation is 30-40% at no more than 2%
quality loss; ship the `eval --ci` gate and measure your own traffic.
