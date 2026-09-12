# Thread 10 — Cognition Fusion: the delegation architecture, and what it changes for us

Source: **"Introducing Fusion in Devin Desktop & CLI"**, cognition.com/blog/local-fusion.
Read 2026-09-12 (full text, both chunks). Vendor primary source [V] — Cognition describing their own
product, with third-party eval partners (Artificial Analysis, Vals AI). Treat numbers as
vendor-reported.

**This is the most important external validation in the dossier**, because Cognition reached our KV-cache
conclusion independently, from production data, and then went one step further than we did.

---

## 10.1 What Fusion is

Not a router. A **two-agent delegation architecture**:

| Role | Model | Owns |
|---|---|---|
| **Lead** | frontier (Fable 5.1, Astra) | the plan, ambiguity interpretation, review, **the user-facing session** |
| **Sidekick** | cost-effective (SWE-2) | exploring code, implementing, running tests, reporting back |

They run **in parallel, each with its own persistent context and tools**. The lead hands the sidekick a
*brief* — task, constraints, success criteria — and gets back results and feedback.

> "Instead of passing entire conversations between models, the lead and sidekick only exchange **briefs,
> results, and feedback**. The sidekick doesn't need the lead's entire history to implement a change, and
> the lead doesn't need every intermediate tool result to review the work. **Each agent builds its own
> persistent context, taking full advantage of prompt caching.**"

That last sentence is the whole design. Read §10.3.

---

## 10.2 They explicitly rejected routing — for our exact reason

The post has a section titled **"Model routing is not enough."** Their two arguments:

> 1. "**The initial prompt isn't enough to know the difficulty of the task.** 'Fix xyz bug' could be a
>    one-line edge case or could require rearchitecting your entire product; you can't know until you've
>    actually investigated the code."
> 2. "**You break prompt caches by switching models mid-task**, incurring $$$ for frontier models and
>    defeating the purpose of routing."

Argument 2 is **our Rule B**, verbatim in substance. Argument 1 is a direct hit on difficulty-based
routing generally — and it's the same finding as arXiv:2606.22902 (LLM-as-router "falls short of the
per-task oracle by a wide margin") reached from the opposite direction: not "the classifier is bad"
but "**the information doesn't exist yet at decision time**."

That's a stronger claim than anything in our dossier, and it's worth internalising:

> **Difficulty is not a property of the prompt. It is a property of the investigation.** Any router that
> scores difficulty from the request text is scoring a proxy that is only weakly correlated with the
> thing it needs.

### ⚠ But note what they did NOT reject

They rejected **mid-task model switching** and **upfront difficulty classification**. They did *not*
reject workload-class inference. Their own architecture routes on **task phase**, which is exactly our
Axis 1:

> "This works because we can separate most coding tasks into well-defined phases."

And they publish the phase distribution:

**Where an agent spends its turns on FrontierCode:** Plan 32% · Setup 34% · Implementation 5% ·
Debug 17% · Validate 9% · Closeout (remainder).

Two things to take from that table:

1. **Implementation is only 5% of turns.** The sidekick is doing the *cheapest* slice by turn count.
   The savings come from the sidekick absorbing high-token, low-judgement work (file reads, test runs),
   not from taking over most of the session.
2. **Plan + Setup is 66%.** That's the frontier model's real job, and it's where the tokens are.

**Consequence for our package:** the phase distribution is a *better* routing signal than difficulty,
and it's observable mid-session (you can tell whether you're debugging or implementing from the tool
call history). This is implementable; difficulty scoring is not.

---

## 10.3 Why the cache survives — and why this beats switching

This is the insight we had **partially**. `07` §7.2(a) said:

> "sub-agent → own session, own cache, own model — **this is where model switching is free**"

Correct, but we framed it as an *escape hatch* from a switch. Cognition frames it as **the primary
architecture**. The difference matters:

| | Mid-session switch (what we optimised) | Delegation (what Fusion does) |
|---|---|---|
| Prefix cache | **destroyed** — cold rewrite at 1.25× | untouched; each agent keeps its own warm cache |
| Context transfer | whole conversation | a **brief** — bounded, small |
| Cost of being wrong | paid every remaining turn | lead reviews and can take control back |
| Decision reversibility | low (cache is gone) | high |

The brief is the key mechanism. Because only a brief crosses the boundary, **neither agent's prefix
ever changes for the other's sake.** Both caches stay hot for the whole session. That's how they get
39% cost reduction *without* the cache penalty that makes ordinary routing self-defeating.

**Our `R*` amortization test optimises the cost of switching. Fusion removes the need to switch.**
Where delegation is available it strictly dominates: no write penalty, no oscillation risk, no
`stranded` cases. `R*` is still correct — it's the right tool when you genuinely must migrate a single
session — but delegation should be tried first.

---

## 10.4 The finding that breaks our cost model

> "**Using more expensive models can make the entire system cheaper.** ... Price per token is only part
> of the equation... In 2026, **models (and model-harness combos) should be evaluated on price per task
> rather than price per token.**"

Their evidence:

| Change | List price | Result |
|---|---|---|
| Opus 4.8 → Fable 5 as lead | Fable **2× more per token** | Fable-led sessions **9% cheaper**, higher FrontierCode score. "Fable delegated earlier and gave better briefs, while Opus micromanaged the sidekick and redid much of its work." |
| GPT-5.6 Luna → SWE-2 as sidekick | **$0.20 → $0.75/Mtok (+275%)** | 62.0 @ $2.39 → **63.4 @ $2.34 (−2%)** |

> "Thus, the models' cost and intelligence are **coupled**. What matters is how efficiently the pair
> completes work together."

**This falsifies a core assumption in `llmrouter`.** Our `PriceCard` models `input_per_mtok` /
`output_per_mtok` and a scalar `quality_prior`. Our scorer computes `cost = tokens × price`. Fusion
shows that a 275% price increase can produce a 2% cost *decrease*, because token efficiency and rework
are not in the model at all.

What's missing from our cost model, per Cognition's data:

1. **Token efficiency** — how many tokens a model needs per unit of work.
2. **Rework rate** — how often the output needs correction, which multiplies downstream cost.
3. **Brief quality / delegation efficiency** — how much back-and-forth a lead creates. Opus
   "micromanaged"; Fable didn't. That's a *model property*, not a harness property.
4. **Pair interaction** — cost is a property of the *(lead, sidekick)* pair, not either model alone.
   "Instructions that help one pair work efficiently can make another perform worse."

Item 4 is the deepest problem: **there is no per-model cost. There is only per-pair cost.** A price
registry keyed by model cannot express it.

---

## 10.5 Harness tuning — the part nobody else talks about

> "Picking a lead and a sidekick is not enough to get the best out of either model. Instructions that
> help one pair work efficiently can make another perform worse. **We continuously tune the harness
> around how models actually work together**, rather than hoping that any combination will work out of
> the box."

Three dials they tune **per model pair**:

| Dial | Weaker sidekick | Stronger sidekick |
|---|---|---|
| Brief detail | More prescriptive (spend lead tokens upfront, avoid review rounds later) | Leave implementation details to the sidekick |
| Pushback | Discourage — "allowing weaker sidekicks to be opinionated ends up hurting performance and cost" | Encourage — catches mistakes in the lead's plan |
| Delegated exploration | **Don't delegate exploration needed for planning** — "a weaker sidekick may not have the ability to properly decide what information matters" | Encourage supporting initial exploration |

That third row is a genuinely non-obvious constraint, and it's a routing rule: **exploration that
shapes the plan must stay with the lead**, because the lead needs the context in its own cache to plan
well. Delegating it doesn't just risk a wrong answer — it starves the lead's context.

---

## 10.6 Reported numbers

Headline: **up to 39% more efficient** than other harnesses, at roughly equal benchmark score.

| Benchmark | Lead alone | Fusion (lead + SWE-2) | Cost change |
|---|---|---|---|
| DeepSWE 1.1 | Fable 5.1: 64.3 @ $14.63 | 63.1 @ $7.88 | **−46%** |
| Terminal-Bench 4 | Fable 5.1: 57.6 @ $17.46 | 56.1 @ $13.37 | −23% |
| SWE-Atlas QnA | Fable 5.1: 64.8 @ $7.57 | **65.9** @ $5.00 | −34% |
| Vals Code Migration | Fable 5.1: 54.6 @ $70.97 | **57.3** @ $42.00 | −41% |
| FrontierCode 1.1 (Ext) | Fable 5.1: 63.6 @ $2.68 | 63.5 @ $1.67 | −38% |
| DeepSWE 1.1 | Astra: 67.6 @ $7.88 | 67.3 @ $4.69 | −40% |
| Terminal-Bench 4 | Astra: 55.6 @ $10.08 | 50.0 @ $6.06 | −40% |
| Vals Code Migration | Astra: 67.7 @ $44.36 | 61.3 @ $35.51 | −20% |

**Read the score column honestly.** Fusion is *cheaper at similar score*, not better. Against Fable it
is within ~1 point on four of five, and *higher* on two. Against Astra it is **materially worse on
Terminal-Bench 4 (55.6 → 50.0, −5.6)** and Vals Code Migration (67.7 → 61.3, −6.4) while saving 20–40%.

So the honest framing: **Fusion is a cost lever with a quality tax that varies a lot by benchmark and
by lead.** −23% to −46% cost, with quality between +2.7 and −6.4. That is a *much* better trade than
the "30–40% savings at ≤2% quality loss" we set as the defensible planning number in `MASTER-REPORT` —
but only for the Fable pairing, and only on some benchmarks. **Do not generalise the 39% headline.**

Also note the harness comparison in the chart is harness-vs-harness (Claude Code, Codex, Muse Code,
Opencode, Kimi Code CLI, Grok Build), so part of the win is Cognition's harness, not only the
delegation architecture. The two are not separable from this post.

---

## 10.7 What we should change in `llmrouter`

### Adopt

1. **Delegation as a first-class routing surface (P0, was P1).** A `lead`/`sidekick` pair with
   separate session keys and separate caches. This is strictly better than mid-session switching
   wherever it applies, because it never pays a cache write.
2. **Route on task *phase*, not difficulty.** Plan / setup / implement / debug / validate are
   observable from tool-call history; difficulty is not observable at all. Replaces the difficulty
   premise entirely for agent workloads.
3. **Exploration that shapes the plan stays with the lead.** A hard rule, not a tuning knob.
4. **Brief-mediated handoff.** Bounded context transfer, so neither prefix grows because of the other.
5. **Per-pair cost accounting.** Cost belongs to the `(lead, sidekick)` pair, not the model.
6. **Price-per-task as the headline metric**, with tokens-per-task and rework rate as the inputs.

### Keep (Fusion does not contradict these)

- Session stickiness and the no-oscillation lock — still correct for the single-session case, which is
  what Fusion's *lead* is.
- Mixed TTL per breakpoint — each agent still has its own prefix to protect.
- Workload-class inference at entry — Fusion routes on phase, which presupposes you know it's an agent.
- `R*` — still the right test when delegation isn't available.

### Explicitly reject

- **Upfront difficulty classification.** Their argument 1 is decisive and we should stop treating
  difficulty as scorable from the prompt.
- **Mid-task model switching as a cost lever.** We already restricted this; Fusion removes the last
  reason to want it in agent workloads.

---

## 10.8 Open questions

1. **Does delegation generalise beyond coding agents?** Every benchmark cited is SWE. The phase
   structure (plan → setup → implement → debug → validate) is coding-specific. Chat and RAG have no
   equivalent clean split. **Unverified outside coding.**
2. **What does the lead's review actually cost?** The post doesn't break out review tokens. If review
   is 30% of lead spend, the 39% is less impressive than it looks.
3. **Is the win harness or architecture?** Chart compares against other *harnesses*, so the two are
   confounded.
4. **How is "turn" defined in the phase distribution?** Turns, tokens, and wall-clock would give very
   different pictures, and the post doesn't say.
5. **Vendor-reported numbers.** Artificial Analysis and Vals AI are named as partners, which is better
   than pure self-evaluation, but this is still a launch post for their own product. **Independent
   replication: none.**

---

## Cross-references

- `07` §7.2(a) — we predicted "sub-agent → own session, own cache, own model — switching is free here."
  **Fusion is production validation of that line.**
- `07` §7.5 P1 "Sub-agent delegation as a first-class routing surface" → **promoted to P0**
- `08` §8.3 — the `R*` algebra. Still valid; now the *fallback* when delegation isn't available.
- `08` §8.6 — "prefer a sub-agent over a mid-session switch." **Fusion makes this the default, not a
  preference.**
- `06` — the `PriceCard` cost model is **falsified by §10.4**; needs per-pair accounting.
