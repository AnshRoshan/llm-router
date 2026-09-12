"""End-to-end demo. Runs offline against a fake provider — no API keys needed.

    python examples/demo.py

Shows, in order:
  1. Path A vs Path B intent inference
  2. Session stickiness across turns (the cache survives)
  3. The learned cache hit rate replacing the assumed one
  4. Mixed TTL per prompt segment
  5. A forced failover when a backend dies
  6. Why the no-oscillation lock exists, in dollars
"""
from __future__ import annotations

import asyncio

from llmrouter import (
    AgentProfile,
    Brief,
    DelegationPolicy,
    TaskPhase,
    BackendRegistry,
    BackendSpec,
    DeploymentClass,
    Message,
    PriceRegistry,
    Router,
    RoutingRequest,
    WorkloadClass,
    WorkloadResolver,
    break_even_turns,
    prefix_cost,
)

SYSTEM = "You are a helpful coding assistant. " * 120   # a realistic ~1.2k-token prefix


class FakeResponse:
    def __init__(self, payload, status_code=200):
        self._payload, self.status_code = payload, status_code

    def json(self):
        return self._payload


class FakeProvider:
    """Simulates a warming cache: turn 1 writes, later turns hit."""

    def __init__(self, dead: set[str] | None = None):
        self.seen: dict[str, int] = {}
        self.dead = dead or set()

    async def post(self, url, *, headers=None, json=None, timeout=None):
        backend = "anthropic" if "anthropic" in url else "openai"
        if backend in self.dead:
            raise RuntimeError(f"{backend} is down")
        model = json["model"]
        turn = self.seen.get(model, 0)
        self.seen[model] = turn + 1
        # Turn 1 writes the cache; every later turn reads it.
        write = 1200 if turn == 0 else 40
        read = 0 if turn == 0 else 1160
        if backend == "anthropic":
            return FakeResponse({
                "model": model,
                "content": [{"type": "text", "text": f"[{model}] turn {turn + 1}"}],
                "usage": {"input_tokens": 30, "output_tokens": 25,
                          "cache_creation_input_tokens": write,
                          "cache_read_input_tokens": read},
            })
        return FakeResponse({
            "model": model,
            "choices": [{"message": {"content": f"[{model}] turn {turn + 1}"}}],
            "usage": {"prompt_tokens": write + read + 30, "completion_tokens": 25,
                      "prompt_tokens_details": {"cached_tokens": read}},
        })


def make_router(transport) -> Router:
    prices = PriceRegistry()
    backends = BackendRegistry([
        BackendSpec("anthropic", DeploymentClass.API,
                    models=tuple(m for m in prices.ids() if m.startswith("claude")),
                    base_url="https://api.anthropic.com", priority=0),
        BackendSpec("openai", DeploymentClass.API,
                    models=tuple(m for m in prices.ids() if m.startswith("gpt")),
                    base_url="https://api.openai.com", priority=1),
    ])
    router = Router(
        prices=prices, backends=backends, transport=transport,
        resolver=WorkloadResolver(profiles={
            "code-agent": AgentProfile("code-agent", WorkloadClass.AGENT),
        }),
    )
    router._keys = {"anthropic": "sk-demo", "openai": "sk-demo"}
    return router


def rule(title: str) -> None:
    print(f"\n{'=' * 74}\n{title}\n{'=' * 74}")


async def main() -> None:
    # ------------------------------------------------------------------ #
    rule("1. Path A (declared) vs Path B (inferred)")
    router = make_router(FakeProvider())
    plain = [Message("system", SYSTEM), Message("user", "hello")]

    a = await router.acomplete(plain, workload="chat", session_id="a")
    da = router.last_decision
    print(f"Path A: workload={da.workload.value} src={da.workload_source.value} "
          f"-> {a.model}")

    b = await router.acomplete(
        [Message("system", SYSTEM), Message("user", "refactor this")],
        tools=({"name": "read_file"},), session_id="b",
    )
    d = router.last_decision
    print(f"Path B: workload={d.workload.value:5} src={d.workload_source.value:11} "
          f"conf={d.workload_confidence:.2f} -> {d.model_id}")

    # Same prompt again: now a free exact fingerprint hit.
    await router.acomplete(
        [Message("system", SYSTEM), Message("user", "and now this")],
        tools=({"name": "read_file"},), session_id="c",
    )
    d = router.last_decision
    print(f"Path B again: src={d.workload_source.value} (registry hit, 0 cost)")

    # ------------------------------------------------------------------ #
    rule("2. Session stickiness across turns")
    router = make_router(FakeProvider())
    msgs = [Message("system", SYSTEM), Message("user", "start a task")]
    chosen = []
    for i in range(4):
        await router.acomplete(msgs, session_id="sticky", workload="agent")
        msgs = msgs + [Message("assistant", "working..."), Message("user", f"step {i}")]
        chosen.append(router.last_decision.model_id)
    print("model per turn:", chosen)
    print("all identical:", len(set(chosen)) == 1,
          "<- the cache survives because we never switched")

    # ------------------------------------------------------------------ #
    rule("3. The cache hit rate is LEARNED, not assumed")
    sess = router.sessions.get("sid:sticky")
    print(f"observed hit rate: {sess.observed_cache_hit_rate:.3f}")
    print("(turn 1 wrote the cache, turns 2-4 read it)")

    # ------------------------------------------------------------------ #
    rule("4. Mixed TTL per prompt segment")
    # Forced onto an Anthropic model: OpenAI has no mixed TTL, so it could not
    # demonstrate the feature.
    router = make_router(FakeProvider())
    prices = PriceRegistry()
    engine_router = Router(
        prices=prices,
        backends=BackendRegistry([BackendSpec(
            "anthropic", DeploymentClass.API,
            models=tuple(m for m in prices.ids() if m.startswith("claude")),
            base_url="https://api.anthropic.com")]),
    )
    msgs = [Message("system", SYSTEM), Message("user", "hi"),
            Message("assistant", "hello"), Message("user", "more")]
    d = engine_router.decide(RoutingRequest(
        messages=tuple(msgs), workload=WorkloadClass.CHAT,
        session_id="ttl", expected_gap_s=1200.0,
    ))
    print(f"model={d.model_id}")
    for kind, ttl in d.ttl_by_segment.items():
        print(f"  {kind.value:10} -> {ttl}")
    print("(system/tools/retrieved survive a 20-minute pause; history does not need to)")
    router = engine_router

    # ------------------------------------------------------------------ #
    rule("5. Forced failover when a backend dies")
    provider = FakeProvider()
    router = make_router(provider)
    await router.acomplete(plain, session_id="fail", workload="chat")
    first = router.last_decision.model_id
    provider.dead.add(router.last_decision.backend_id)
    print(f"pinning to {first}; killing {router.last_decision.backend_id}")
    try:
        out = await router.acomplete(plain, session_id="fail", workload="chat")
        d = router.last_decision
        print(f"failed over to {out.model} (switched={d.switched})")
        for r_ in d.reasons:
            if "failover" in r_ or "re-decid" in r_:
                print(f"  · {r_}")
    except Exception as exc:
        print(f"raised {type(exc).__name__}: {exc}")

    # ------------------------------------------------------------------ #
    rule("6. Why the no-oscillation lock exists (dollars)")
    prices = PriceRegistry()
    sonnet = prices.require("claude-sonnet-class")
    S = 50_000
    stay = prefix_cost(sonnet, S, 30)
    oscillate = 4 * sonnet.cache_write_cost(S)
    print(f"30 turns, never switching : ${stay:.4f}")
    print(f"4 legs of A->B->A->B      : ${oscillate:.4f}")
    print(f"oscillating costs {oscillate / stay:.2f}x more than never routing at all")

    print("\nswitch pay-back (R*) by target:")
    haiku = prices.require("claude-haiku-class")
    gpt_mini = prices.require("gpt-mini-class")
    print(f"  sonnet -> haiku    (Anthropic, 1.25x write): "
          f"{break_even_turns(sonnet, haiku):.2f} turns")
    print(f"  sonnet -> gpt-mini (OpenAI, no write fee) : "
          f"{break_even_turns(sonnet, gpt_mini):.2f} turns")

    # ------------------------------------------------------------------ #
    rule("7. Delegation: the sidekick gets its own cache")
    router = make_router(FakeProvider())
    router.sidekick_model = "gpt-mini-class"
    router.delegation = DelegationPolicy(sidekick_is_strong=True)

    lead_req = RoutingRequest(messages=(
        Message("system", SYSTEM), Message("user", "fix the parser"),
        Message("user", [{"type": "tool_use", "name": "str_replace"}]),
    ), session_id="lead")
    decision = router.should_delegate(lead_req, task="apply the patch")
    print(f"phase={decision.phase.value}  delegate={decision.should_delegate}")
    print(f"  {decision.reason}")

    result = await router.adelegate(decision.brief, lead_session_id="lead")
    print(f"  sidekick returned {result.tokens_used} tokens, needs_review={result.needs_review}")
    keys = sorted(router.sessions._sessions)
    print("  sessions now:", keys)
    print("  -> lead's prefix untouched; the sidekick warmed its OWN cache")

    # Plan-phase work must never leave the lead.
    plan_req = RoutingRequest(messages=(Message("user", "fix the parser"),),
                              session_id="lead")
    pd = router.should_delegate(plan_req, task="decide the approach")
    print(f"\nphase={pd.phase.value}  delegate={pd.should_delegate}")
    print(f"  {pd.reason}")

    rule("8. Cost belongs to the pair, not the model")
    router.pairs.record_outcome("claude-sonnet-class", "gpt-mini-class", cost=2.39, score=62.0)
    router.pairs.record_outcome("claude-sonnet-class", "claude-haiku-class", cost=2.34, score=63.4)
    best = router.pairs.best_by_price_per_task()
    print(f"cheapest pair by measurement: {best.lead_model} + {best.sidekick_model}"
          f" @ ${best.price_per_task}")
    print("(a pricier sidekick can win: fewer attempts, fewer review rounds)")

    rule("stats (from the last router built above)")
    for k, v in router.stats().items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    asyncio.run(main())
