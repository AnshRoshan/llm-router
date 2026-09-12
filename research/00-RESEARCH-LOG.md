# LLM Router Research — Working Log & Source Ledger

**Started / completed:** 2026-09-12 (Asia/Calcutta)
**Purpose:** design input for a production-grade LLM router Python package.

## Deliverables

| File | Contents |
|---|---|
| `MASTER-REPORT.md` | Executive summary (10 findings), master comparative table, answers to the 5 research questions, build recommendations, confidence notes |
| `01-routellm-deep-dive.md` | RouteLLM: repo facts, architecture, 5 routers, headline numbers + caveats, RouterArena verification, steal/avoid list |
| `02-frameworks-landscape.md` | Gateway-vs-router taxonomy, 14-tool index, LiteLLM deep detail, cloud managed routers, vLLM Semantic Router architecture, Not Diamond/RoRF, semantic-router, the 7 unsolved gaps |
| `03-technique-taxonomy.md` | Every technique at 3 depths: rule / SLM-classifier / LLM-based / cascade / hybrid / 16 emerging methods |
| `04-cost-benchmarks-eval.md` | The honest savings ladder (17 data points), 8 benchmarks, metric definitions, calibration methodology, eval pitfalls |
| `05-architecture-patterns.md` | Reference request path, backend adapters, complete fallback primitive set, distributed scaling, observability, semantic caching, safety-in-path, 10 production gotchas |
| `06-design-recommendations.md` | Python package competitive set, recommended 3-layer architecture, strategy priority matrix, API surface + CLI, production checklist, eval harness spec, phased build plan, open questions |
| `SOURCES.md` | **Complete source register, sorted into [P]rimary / [S]econdary / [V]endor / [X]unverified tiers. Section F holds everything added in the revised-scope pass** |
| **`07-workload-and-deployment-classes.md`** | **READ FIRST.** The two new axes (workload class × deployment class), the KV-cache/model-scoped-cache finding, per-workload requirements (agent/chat/RAG/batch/code/vision), deployment signals, unified decision model, revised priorities. **Rule B corrected by `08`.** |
| **`08-session-stickiness-and-entry-intent.md`** | **READ SECOND.** Session stickiness, the two entry paths (declared vs inferred intent), the `R*` switch-cost algebra, session identity without `session_id`, reclassification/no-oscillation policy, learned cache hit rates. **Closes `07` Q1/Q2, corrects `07` Rule B. §8.7(3) partially corrected by `09`.** |
| **`09-ttl-tier-policy.md`** | **READ THIRD.** TTL tier (5m vs 1h) as a routing decision: the `E* = 0.652` expiry threshold, **mixed TTL per breakpoint** (system/tools@1h, history@5m), generation-time-eats-the-window, keepalive-vs-upgrade break-even at 37.5 min, **Rule F** on cache-invalidating request fields. **Answers `08` Q4; corrects `08` §8.7(3).** |
| **`10-fusion-delegation-architecture.md`** | **READ FOURTH.** Cognition Fusion: lead/sidekick **delegation** instead of routing. Validates our KV-cache finding independently, **falsifies the per-token cost model** (a 275% pricier sidekick was 2% cheaper), and says difficulty is unscorable from the prompt. **Promotes sub-agent delegation P1→P0.** |
| **`verify_08_switch_cost.py`** | Executable check of the §8.3 algebra. 10 checks, all assertions pass. Caught an off-by-one in the doc's worked example. |
| **`verify_09_ttl_policy.py`** | Executable check of the §9 TTL algebra. 29 checks, 13 assertions, all pass. Caught two modeling errors (same expiry count applied to both tiers; snapshot-only mixed-TTL model). |
| `summary-table.csv` | Machine-readable version of the master comparative table (21 techniques × 9 fields) |

## Investigation threads (all completed)

| # | Thread | Key output |
|---|---|---|
| 1 | RouteLLM deep dive | Stale since 2024-08-10; ranks #25/26 on RouterArena '26 despite 85% MT-Bench claim |
| 2 | Framework/gateway landscape | Gateway ≠ router; LiteLLM = execution layer, RouteLLM = decision layer |
| 3 | Technique taxonomy | 6 families + 16 specialized methods; hybrid signal-bus is the 2026 consensus |
| 4 | Cost economics + benchmarks | Honest number ~30–40%, not 85%; RouterEval says 3–10 candidates optimal |
| 5 | Production architecture | 3 fallback channels, per-deployment circuit breakers, Redis required >1 replica |
| 6 | Python packages + design | **llmrouter-lib 0.4.0 is the direct competitor**; the gap is the execution + monitoring layer |

## Corrections made during research (audit trail)

1. RouteLLM ships **5** routers (`mf`, `sw_ranking`, `bert`, `causal_llm`, `random`), not 4 — my first
   draft conflated `bert` and `causal_llm`. Corrected in `01-routellm-deep-dive.md` §3 from the README.
2. The claim "most routing methods collapse to similar performance under fair evaluation" was initially
   logged as unverified from a blog citing arXiv 2602.11877 + 2601.07206. **Verified:** 2602.11877 is
   "Towards Fair and Comprehensive Evaluation of Routers in Collaborative LLM Systems" (RouterXBench,
   ProbeDirichlet, 2026-02-12) and 2601.07206 is "LLMRouterBench" (2026-01). Both real. The
   "collapse to similar performance" phrasing is LLMRouterBench's and is now attributed correctly.
3. **`07` Rule B ("never switch model mid-session for cost reasons alone") is false as an absolute.**
   Re-verifying cache pricing on 2026-09-12 established that **OpenAI charges no cache-write premium**
   (pre-GPT-5.6), so a migration onto an OpenAI-backed model costs nothing in write terms — break-even
   **0.8 turns**. Rule B is annotated in place; the corrected rule (never *oscillate*; use the `R*`
   amortization test) is in `08` §8.1/§8.3.
4. **`07` §7.1's "OpenAI 50% cache discount" is stale.** It is right for gpt-4o / gpt-4.1 and wrong for
   GPT-5.4 / 5.5, which are at 90% — matching Anthropic. `discount_pct` must be per-model, not
   per-provider. Corrected in `08` §8.1.
5. **Off-by-one in `08`'s own worked example**, caught by running `verify_08_switch_cost.py`: the draft
   priced 26 turns on the target model while claiming 25, inflating the saving to $0.4350. Correct
   figures are **$0.4300 / 30.9%**. The script now asserts `stay_turns + remaining == total_turns`.
6. **Two modeling errors in `09`'s TTL algebra**, caught by `verify_09_ttl_policy.py`: (a) the first
   draft applied the *same* expiry count to both the 5m and 1h tiers, which made 1h look worse
   everywhere and contradicted the derived `E* = 0.652`; the correct comparison is 5m-with-expiries
   vs 1h-without. (b) The mixed-TTL model was a no-gap snapshot, where all-5m always wins on the
   write — the honest test is *after an idle gap*, where mixed pays back within one gap.
7. **`08` §8.1's "Vertex/Bedrock are 5-min only" is outdated** — AWS docs list 1h TTL for 10 Claude
   models. Corrected in `09` §9.8.
8. **`08` §8.7(3)'s "batch is exempt from cache logic" is too strong** — exempt from session affinity,
   not cache policy. Annotated in place; corrected in `09` §9.4(2).
9. **claude-code #32671's "~50% TTL saving" counts reads only.** Including the 1h write gives **42%**.
   The issue's implied base price is also internally inconsistent. Cite ~40–50%.

## Thread 10 implementation record (Fusion → `llmrouter`)

Analysis is in `10-fusion-delegation-architecture.md`. The following were **built** into
`/home/user/llmrouter` as a result:

| Adopted | Where | Status |
|---|---|---|
| Task-phase routing (route on phase, not difficulty) | `delegation.py` `TaskPhase`, `infer_phase()` | implemented + tested |
| Lead/sidekick delegation with **separate session keys** | `router.py` `adelegate()`, `should_delegate()` | implemented + tested |
| Brief-mediated handoff (bounded, not a conversation) | `delegation.py` `Brief`, `_brief_system_prompt()` | implemented + tested |
| Plan-phase is lead-only; exploration that shapes the plan never delegates | `LEAD_ONLY_PHASES`, `DelegationPolicy.delegable()` | implemented + tested |
| Strong vs weak sidekick dials (brief detail, pushback) | `DelegationPolicy(sidekick_is_strong=...)` | implemented + tested |
| Per-pair cost accounting (price per task) | `delegation.py` `PairProfile`, `PairRegistry` | implemented + tested |

Two real bugs found while wiring this up:
1. `with_defaults()` did not accept `sidekick_model` — the README documented an API that did not
   exist. Fixed, plus `TestDocumentedAPI` now guards the exact README call shapes.
2. **Falsy-trap bug**: `PairRegistry`, `FingerprintRegistry` and `SessionStore` all define `__len__`,
   so an *empty* registry is falsy and `self.pairs = pairs or PairRegistry()` silently **discarded the
   caller's object**. Their later `record_outcome()` calls would have gone nowhere. All eight
   `x or SomeClass()` sites in `router.py` / `policy.py` / `signals.py` replaced with explicit
   `is None` checks; three regression tests added.

Suite: **148 tests pass** (was 99), verified against the built wheel installed in a clean venv (zero runtime deps); `examples/demo.py` now shows delegation (§7) and pair economics (§8).
The scorer applies `task_cost_multiplier` and measured pair rework, so price-per-task replaces price-per-token in the decision path rather than sitting beside it as reporting.
**Two delegation bugs found and fixed while verifying the loop end-to-end:**
1. `adelegate` ignored `sidekick_model` and routed to whatever scored highest (the frontier model) — defeating delegation entirely. Now pinned via a restricted internal agent profile, so it still passes through hard constraints, health checks, and failover. The pin is a `sidekick_model` property, so it also holds when set after construction (the demo path) — the __init__-only registration silently lost it. Regression-tested.
2. `record_outcome` only moved `measured_cost_per_task`/`measured_score`, never the `sidekick_attempts`/`rework_factor` that `effective_sidekick_multiplier()` (and thus the scorer) actually reads — so observed behaviour never reached a decision. Fixed, and `adelegate` now auto-records real usage + attempt count. Verified: a forced failover moved the scorer's multiplier 1.0 → 1.3 while the pin held.

## Unresolved / needs further work before design sign-off

- [ ] **Bedrock Intelligent Prompt Routing pricing** — $1.00 per 1,000 routed requests (cloudburn.io,
      DigitalOcean comparison) vs "no separate fee, pay per underlying tokens" (llmreference.com).
      Check the AWS pricing page directly.
- [ ] **Agentic routing: does it work?** arXiv:2608.14641 found effects inconclusive on BFCL v4,
      RouterBench, tau2-bench and equivalent on WebArena; OmniRouter claims +6.30% accuracy / −10.15%
      cost. Read both in full before committing to agent support.
- [ ] **BEST-Route** (arXiv:2506.22716, Microsoft, ICML 2025) — only seen via a secondary tutorial.
      Verify the 15–30% escalation reduction directly.
- [ ] **LLMRouter's full router list** — only chunk 0 of the README was read (single-round routers +
      part of multi-round). The multi-round, multimodal, agentic and personalized categories were not
      enumerated. Worth reading before finalizing the strategy matrix.
- [ ] **Martian** — no primary documentation found; all figures are self-reported marketing.

## Source-quality key used throughout

- **High** = primary source (paper PDF/HTML, official docs, repo README, PyPI JSON) read this session.
- **Medium** = one reputable secondary source; primary not read.
- **Low** = vendor marketing or single unattributed claim.
- Every quantitative claim in the deliverables is tagged with its source and, where the source is
  secondary or contradictory, flagged as such.
