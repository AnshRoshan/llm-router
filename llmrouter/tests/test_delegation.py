"""Task-phase routing and lead/sidekick delegation (Thread 10)."""
from __future__ import annotations

import pytest

from llmrouter import (
    DELEGABLE_PHASES,
    LEAD_ONLY_PHASES,
    PHASE_TURN_SHARE,
    Brief,
    DelegationPolicy,
    Message,
    PairProfile,
    PairRegistry,
    RoutingRequest,
    TaskPhase,
    infer_phase,
)


def req(messages=(), tools=()) -> RoutingRequest:
    return RoutingRequest(messages=tuple(messages), tools=tuple(tools))


class TestPhaseInference:
    def test_no_history_is_plan(self):
        """A fresh session plans. PLAN is lead-only, so this is the conservative
        default — the fallback never delegates."""
        assert infer_phase(()) == TaskPhase.PLAN
        assert infer_phase((Message("user", "fix the bug"),)) == TaskPhase.PLAN

    def test_reads_are_setup(self):
        msgs = (
            Message("user", [{"type": "tool_use", "name": "read_file"}]),
            Message("user", [{"type": "tool_use", "name": "grep"}]),
        )
        assert infer_phase(msgs) == TaskPhase.SETUP

    def test_edits_are_implementation(self):
        msgs = (
            Message("user", [{"type": "tool_use", "name": "str_replace"}]),
            Message("user", [{"type": "tool_use", "name": "write_file"}]),
        )
        assert infer_phase(msgs) == TaskPhase.IMPLEMENT

    def test_tests_are_validation(self):
        msgs = (Message("user", [{"type": "tool_use", "name": "run_tests"}]),) * 2
        assert infer_phase(msgs) == TaskPhase.VALIDATE

    def test_errors_after_tool_use_are_debug(self):
        msgs = (
            Message("user", [{"type": "tool_use", "name": "read_file"}]),
            Message("user", [{"type": "tool_use", "name": "str_replace"}]),
            Message("user", [{"type": "tool_use", "name": "run_tests"}]),
            Message("user", "Traceback: AssertionError in test_x"),
        )
        assert infer_phase(msgs) == TaskPhase.DEBUG

    def test_phase_is_observable_where_difficulty_is_not(self):
        """The whole premise: the same prompt can be in different phases, but its
        difficulty is unknowable until the investigation is done."""
        prompt = Message("user", "fix the xyz bug")
        early = req((prompt,))
        later = req((prompt, Message("user", [{"type": "tool_use", "name": "str_replace"}])))
        assert infer_phase(early.messages) == TaskPhase.PLAN
        assert infer_phase(later.messages) == TaskPhase.IMPLEMENT


class TestDelegationPolicy:
    def test_plan_is_never_delegated(self):
        policy = DelegationPolicy()
        d = policy.decide(req((Message("user", "fix it"),)), task="investigate")
        assert d.should_delegate is False
        assert d.phase == TaskPhase.PLAN
        assert "shapes the plan" in d.reason

    def test_closeout_is_lead_only(self):
        assert TaskPhase.CLOSEOUT in LEAD_ONLY_PHASES
        assert not DelegationPolicy().delegable(TaskPhase.CLOSEOUT)

    def test_implementation_delegates(self):
        msgs = (Message("user", [{"type": "tool_use", "name": "str_replace"}]),) * 2
        d = DelegationPolicy().decide(req(msgs), task="apply the patch")
        assert d.should_delegate is True
        assert d.phase == TaskPhase.IMPLEMENT
        assert d.brief is not None

    def test_weak_sidekick_gets_prescriptive_briefs_and_no_pushback(self):
        msgs = (Message("user", [{"type": "tool_use", "name": "run_tests"}]),) * 2
        d = DelegationPolicy(sidekick_is_strong=False).decide(req(msgs), task="t")
        assert d.brief.prescriptive is True
        assert d.brief.allow_pushback is False

    def test_strong_sidekick_gets_terse_briefs_and_pushback(self):
        msgs = (Message("user", [{"type": "tool_use", "name": "run_tests"}]),) * 2
        d = DelegationPolicy(sidekick_is_strong=True).decide(req(msgs), task="t")
        assert d.brief.prescriptive is False
        assert d.brief.allow_pushback is True

    def test_debug_delegation_is_configurable(self):
        assert DelegationPolicy(delegate_debug=True).delegable(TaskPhase.DEBUG)
        assert not DelegationPolicy(delegate_debug=False).delegable(TaskPhase.DEBUG)

    def test_delegable_and_lead_only_do_not_overlap(self):
        assert not (DELEGABLE_PHASES & LEAD_ONLY_PHASES)

    def test_phase_shares_are_a_distribution(self):
        assert sum(PHASE_TURN_SHARE.values()) == pytest.approx(1.0, abs=0.001)
        # The non-obvious fact worth encoding: implementation is a small slice.
        assert PHASE_TURN_SHARE[TaskPhase.IMPLEMENT] < 0.10
        assert (PHASE_TURN_SHARE[TaskPhase.PLAN]
                + PHASE_TURN_SHARE[TaskPhase.SETUP]) > 0.6


class TestBrief:
    def test_a_brief_is_bounded_not_a_conversation(self):
        """Neither agent's prefix may grow because of the other, or both caches
        go cold — which is the entire mechanism that makes delegation cheaper
        than switching."""
        b = Brief(task="implement the parser",
                  constraints=("no new deps",), success_criteria=("tests pass",))
        assert b.token_estimate() < 50
        assert len(b.task) < 200

    def test_brief_carries_success_criteria(self):
        b = Brief(task="t", success_criteria=("tests pass",))
        assert b.success_criteria == ("tests pass",)


class TestPairEconomics:
    def test_cost_belongs_to_the_pair_not_the_model(self):
        reg = PairRegistry()
        reg.register(PairProfile("fable", "swe-2", measured_cost_per_task=1.67,
                                 measured_score=63.5))
        reg.register(PairProfile("fable", "luna", measured_cost_per_task=2.39,
                                 measured_score=62.0))
        best = reg.best_by_price_per_task()
        assert best.sidekick_model == "swe-2"

    def test_pricier_sidekick_can_be_cheaper_overall(self):
        """Cognition's finding: SWE-2 is +275% per token yet the pair was 2%
        cheaper, because fewer attempts and fewer review rounds."""
        luna_price, swe2_price = 0.20, 0.75
        assert swe2_price / luna_price > 2.0          # 275% more expensive per token
        reg = PairRegistry()
        reg.register(PairProfile("astra", "luna", measured_cost_per_task=2.39))
        reg.register(PairProfile("astra", "swe-2", measured_cost_per_task=2.34))
        assert reg.get("astra", "swe-2").price_per_task < reg.get("astra", "luna").price_per_task

    def test_unknown_pair_gets_a_neutral_profile(self):
        reg = PairRegistry()
        p = reg.get("x", "y")
        assert p.rework_factor == 1.0
        assert p.measured_cost_per_task is None

    def test_effective_multiplier_includes_rework_and_attempts(self):
        p = PairProfile("lead", "side", rework_factor=1.4, sidekick_attempts=1.5)
        assert p.effective_sidekick_multiplier() == pytest.approx(2.1)

    def test_outcomes_are_folded_in_not_overwritten(self):
        reg = PairRegistry()
        reg.record_outcome("l", "s", cost=10.0)
        assert reg.get("l", "s").measured_cost_per_task == 10.0
        reg.record_outcome("l", "s", cost=0.0)
        # EWMA, not replacement: one cheap session must not erase history.
        assert reg.get("l", "s").measured_cost_per_task == pytest.approx(7.0)

    def test_best_is_none_without_measurements(self):
        reg = PairRegistry()
        reg.get("a", "b")
        assert reg.best_by_price_per_task() is None


# --------------------------------------------------------------------------- #
# Router integration
# --------------------------------------------------------------------------- #
from llmrouter import (  # noqa: E402
    BackendRegistry,
    BackendSpec,
    ConfigurationError,
    DeploymentClass,
    PriceRegistry,
    Router,
    WorkloadResolver,
)
from tests.test_router import SYSTEM, FakeTransport, make_router  # noqa: E402


class TestDelegationIntegration:
    def test_sidekick_gets_its_own_session_key(self):
        """The whole mechanism: a separate session key means a separate prefix
        means a separate warm cache, so the lead's cache is never disturbed."""
        router = make_router()
        router.sidekick_model = "gpt-mini-class"
        msgs = (Message("user", [{"type": "tool_use", "name": "str_replace"}]),) * 2
        d = router.should_delegate(req(msgs), task="apply patch")
        assert d.should_delegate is True

        lead_key = "lead-1"
        sidekick_key = f"{lead_key}::sidekick::{d.phase.value}"
        assert sidekick_key != lead_key
        assert lead_key in sidekick_key

    @pytest.mark.asyncio
    async def test_adelegate_runs_and_keeps_sessions_separate(self):
        router = make_router()
        router.sidekick_model = "gpt-mini-class"
        brief = Brief(task="apply the patch", phase=TaskPhase.IMPLEMENT,
                      constraints=("no new deps",),
                      success_criteria=("tests pass",))
        result = await router.adelegate(brief, lead_session_id="lead-1")
        assert result.output
        assert result.needs_review is True

        # sticky_key() prefixes declared ids with "sid:", so the sidekick lands on
        # its own key, distinct from the lead's "sid:lead-1".
        keys = set(router.sessions._sessions)
        assert any("::sidekick::" in k for k in keys), keys
        assert "sid:lead-1" not in keys, "the lead's own session must be untouched"
        assert router.stats()["delegations"] == 1

    @pytest.mark.asyncio
    async def test_lead_only_phase_cannot_be_delegated(self):
        router = make_router()
        router.sidekick_model = "gpt-mini-class"
        brief = Brief(task="decide the architecture", phase=TaskPhase.PLAN)
        with pytest.raises(ValueError, match="not delegable"):
            await router.adelegate(brief, lead_session_id="lead-1")

    @pytest.mark.asyncio
    async def test_no_sidekick_is_a_config_error_not_a_silent_noop(self):
        router = make_router()
        assert router.sidekick_model is None
        with pytest.raises(ConfigurationError, match="sidekick_model"):
            await router.adelegate(Brief(task="t", phase=TaskPhase.IMPLEMENT),
                                   lead_session_id="lead-1")

    @pytest.mark.asyncio
    async def test_pair_economics_are_recorded(self):
        router = make_router()
        router.sidekick_model = "gpt-mini-class"
        router.pairs.record_outcome("claude-sonnet-class", "gpt-mini-class",
                                    cost=1.20, score=60.0)
        stats = router.stats()
        pair = stats["pairs"]["claude-sonnet-class+gpt-mini-class"]
        assert pair["price_per_task"] == pytest.approx(1.20)
        assert pair["score"] == pytest.approx(60.0)

    def test_sidekick_system_prompt_is_stable_per_phase(self):
        """If the sidekick's system prompt changed every delegation its own cache
        would never hit — defeating the purpose."""
        from llmrouter.router import _brief_system_prompt

        a = _brief_system_prompt(Brief(task="task one", phase=TaskPhase.IMPLEMENT))
        b = _brief_system_prompt(Brief(task="task two", phase=TaskPhase.IMPLEMENT))
        assert a == b, "system prompt must not depend on the task text"

    def test_pushback_appears_only_for_strong_sidekicks(self):
        from llmrouter.router import _brief_system_prompt

        weak = _brief_system_prompt(Brief(task="t", prescriptive=True,
                                          allow_pushback=False))
        strong = _brief_system_prompt(Brief(task="t", prescriptive=False,
                                            allow_pushback=True))
        assert "Follow the brief exactly" in weak
        assert "Follow the brief exactly" not in strong
        assert "say so before proceeding" in strong
        assert "say so before proceeding" not in weak


class TestDocumentedAPI:
    """Guard the exact call shapes shown in the README."""

    def test_with_defaults_accepts_sidekick_model(self):
        router = Router.with_defaults(anthropic_key="k",
                                      sidekick_model="claude-haiku-class")
        assert router.sidekick_model == "claude-haiku-class"
        assert isinstance(router.delegation, DelegationPolicy)
        assert isinstance(router.pairs, PairRegistry)

    def test_with_defaults_accepts_delegation_and_pairs(self):
        pairs = PairRegistry()
        router = Router.with_defaults(
            anthropic_key="k",
            sidekick_model="claude-haiku-class",
            delegation=DelegationPolicy(sidekick_is_strong=True, delegate_debug=False),
            pairs=pairs,
        )
        assert router.delegation.sidekick_is_strong is True
        assert router.delegation.delegate_debug is False
        assert router.pairs is pairs

    def test_empty_registry_is_not_silently_replaced(self):
        """Regression: PairRegistry defines __len__, so an empty registry is
        falsy and `pairs or PairRegistry()` used to discard the caller's object.
        Their later record_outcome calls would then go nowhere.
        """
        pairs = PairRegistry()
        assert bool(pairs) is False, "precondition: empty registry is falsy"
        router = Router.with_defaults(anthropic_key="k", pairs=pairs)
        assert router.pairs is pairs
        router.pairs.record_outcome("a", "b", cost=1.0)
        assert pairs.get("a", "b").measured_cost_per_task == 1.0

    def test_empty_fingerprint_registry_is_not_silently_replaced(self):
        """Same falsy trap: FingerprintRegistry also defines __len__."""
        from llmrouter import FingerprintRegistry, WorkloadResolver

        reg = FingerprintRegistry()
        assert bool(reg) is False
        assert WorkloadResolver(fingerprints=reg).fingerprints is reg

    def test_empty_session_store_is_not_silently_replaced(self):
        """Same falsy trap: SessionStore also defines __len__."""
        from llmrouter import SessionStore

        store = SessionStore()
        assert bool(store) is False
        assert Router.with_defaults(anthropic_key="k",
                                    transport=None).sessions is not None
        router = Router(
            prices=PriceRegistry(),
            backends=BackendRegistry([BackendSpec("anthropic", DeploymentClass.API,
                                                  base_url="x")]),
            sessions=store,
        )
        assert router.sessions is store

    @pytest.mark.asyncio
    async def test_readme_quickstart_shape_runs(self):
        router = make_router()
        router.sidekick_model = "gpt-mini-class"
        request = req((Message("user", [{"type": "tool_use", "name": "str_replace"}]),) * 2)
        decision = router.should_delegate(request, task="apply the patch")
        assert decision.should_delegate is True
        result = await router.adelegate(decision.brief, lead_session_id="user-42")
        assert result.output


class TestPairEconomicsReachTheScorer:
    """Regression: PairRegistry existed but the scorer never read it, so the
    accounting was dead. These prove measured pair cost actually moves a decision.
    """

    def _engine(self, pairs=None, lead=None):
        from llmrouter.policy import PolicyEngine, RouterConfig, Weights

        backends = BackendRegistry([
            BackendSpec("anthropic", DeploymentClass.API,
                        models=("claude-opus-class", "claude-sonnet-class",
                                "claude-haiku-class"),
                        base_url="x", priority=0),
            BackendSpec("openai", DeploymentClass.API,
                        models=("gpt-frontier-class", "gpt-legacy-class",
                                "gpt-mini-class"),
                        base_url="y", priority=1),
        ])
        return PolicyEngine(PriceRegistry(), backends, WorkloadResolver(),
                            RouterConfig(weights=Weights(affinity=0.0)),
                            pairs=pairs, lead_model=lead)

    def _req(self):
        return RoutingRequest(
            messages=(Message("system", SYSTEM), Message("user", "hi")),
            session_id="s", expected_turns=50,
        )

    def test_heavy_pair_rework_changes_the_chosen_model(self):
        base = self._engine().decide(self._req())
        pairs = PairRegistry()
        pairs.register(PairProfile("claude-sonnet-class", base.model_id,
                                   rework_factor=6.0, sidekick_attempts=3.0))
        poisoned = self._engine(pairs=pairs,
                                lead="claude-sonnet-class").decide(self._req())
        assert poisoned.model_id != base.model_id, (
            "an 18x measured rework multiplier must change the decision"
        )

    def test_unmeasured_pair_is_neutral(self):
        """No profile => multiplier 1.0 => decision identical to no registry."""
        plain = self._engine().decide(self._req())
        with_empty = self._engine(pairs=PairRegistry(),
                                  lead="claude-sonnet-class").decide(self._req())
        assert plain.model_id == with_empty.model_id

    def test_lead_is_never_scaled_by_its_own_pair_profile(self):
        """The lead's cost must not be inflated by the sidekick's rework."""
        eng = self._engine()
        eng.pairs = PairRegistry()
        eng.lead_model = "claude-sonnet-class"
        eng.pairs.register(PairProfile("claude-sonnet-class", "claude-sonnet-class",
                                       rework_factor=99.0))
        assert eng._pair_multiplier(eng.prices.require("claude-sonnet-class")) == 1.0

    def test_task_cost_multiplier_composes(self):
        from dataclasses import replace

        card = PriceRegistry().require("claude-sonnet-class")
        assert card.task_cost_multiplier == 1.0
        scaled = replace(card, token_efficiency=1.5, rework_factor=2.0)
        assert scaled.task_cost_multiplier == pytest.approx(3.0)

    def test_pricier_per_token_can_win_per_task(self):
        """The Fusion finding, expressed in our cost model: a model with a higher
        $/token but better token efficiency and less rework costs less per task."""
        from dataclasses import replace

        prices = PriceRegistry()
        cheap = prices.require("gpt-mini-class")            # $0.75/M
        pricey = replace(prices.require("claude-sonnet-class"),  # $3.00/M
                         token_efficiency=0.3, rework_factor=0.5)
        assert pricey.input_per_mtok > cheap.input_per_mtok
        eng = self._engine()
        t_cheap = eng.estimate_turn_cost(cheap, 4000, 0.9)
        t_pricey = eng.estimate_turn_cost(pricey, 4000, 0.9)
        assert t_pricey < t_cheap, "price per task, not price per token"


class TestSidekickPinIsHonored:
    """Regression: adelegate routed to whatever scored highest, ignoring the
    configured sidekick — defeating the entire point of delegation."""

    def _router(self, transport, sidekick="gpt-mini-class"):
        prices = PriceRegistry()
        backends = BackendRegistry([
            BackendSpec("anthropic", DeploymentClass.API,
                        models=tuple(m for m in prices.ids()
                                     if m.startswith("claude")),
                        base_url="https://api.anthropic.com", priority=0),
            BackendSpec("openai", DeploymentClass.API,
                        models=tuple(m for m in prices.ids()
                                     if m.startswith("gpt")),
                        base_url="https://api.openai.com", priority=1),
        ])
        r = Router(prices=prices, backends=backends, transport=transport,
                   lead_model="claude-sonnet-class", sidekick_model=sidekick)
        r._keys = {"anthropic": "sk", "openai": "sk"}
        return r

    @pytest.mark.asyncio
    async def test_delegation_runs_on_the_configured_sidekick(self):
        from demo import FakeProvider

        r = self._router(FakeProvider())
        await r.adelegate(Brief(task="fix", phase=TaskPhase.IMPLEMENT),
                          lead_session_id="lead")
        assert r.last_decision.model_id == "gpt-mini-class", (
            "adelegate must run on the configured sidekick, not the top scorer"
        )

    @pytest.mark.asyncio
    async def test_pair_is_learned_for_the_right_sidekick(self):
        from demo import FakeProvider

        r = self._router(FakeProvider())
        await r.adelegate(Brief(task="fix", phase=TaskPhase.IMPLEMENT),
                          lead_session_id="lead")
        assert ("claude-sonnet-class", "gpt-mini-class") in r.pairs.profiles


class TestLearningReachesTheMultiplier:
    """Regression: record_outcome only moved measured_cost/score, never the
    sidekick_attempts/rework_factor the scorer's multiplier actually reads — so
    observed behaviour never influenced a decision."""

    def test_record_outcome_moves_the_multiplier_the_scorer_reads(self):
        reg = PairRegistry()
        before = reg.get("l", "s").effective_sidekick_multiplier()
        reg.record_outcome("l", "s", cost=1.0, attempts=3.0)
        after = reg.get("l", "s").effective_sidekick_multiplier()
        assert after > before, "observed attempts must raise the pair multiplier"

    def test_record_outcome_folds_rework_factor(self):
        reg = PairRegistry()
        reg.record_outcome("l", "s", cost=1.0, rework_factor=2.0)
        # EWMA from 1.0 toward 2.0 at alpha 0.3 -> 1.3
        assert reg.get("l", "s").rework_factor == pytest.approx(1.3)

    @pytest.mark.asyncio
    async def test_adelegate_autorecords_real_cost_and_attempts(self):
        from demo import FakeProvider, FakeResponse

        class FlakyFirst(FakeProvider):
            def __init__(self):
                super().__init__()
                self._failed = False

            async def post(self, url, *, headers=None, json=None, timeout=None):
                if not self._failed:
                    self._failed = True
                    raise RuntimeError("transient 500")
                return await super().post(url, headers=headers, json=json,
                                          timeout=timeout)

        prices = PriceRegistry()
        backends = BackendRegistry([
            BackendSpec("anthropic", DeploymentClass.API,
                        models=tuple(m for m in prices.ids()
                                     if m.startswith("claude")),
                        base_url="https://api.anthropic.com", priority=0),
            BackendSpec("openai-a", DeploymentClass.API,
                        models=("gpt-mini-class",),
                        base_url="https://a.openai.com", priority=1),
            BackendSpec("openai-b", DeploymentClass.API,
                        models=("gpt-mini-class",),
                        base_url="https://b.openai.com", priority=2),
        ])
        r = Router(prices=prices, backends=backends, transport=FlakyFirst(),
                   lead_model="claude-sonnet-class",
                   sidekick_model="gpt-mini-class")
        r._keys = {"anthropic": "sk", "openai-a": "sk", "openai-b": "sk"}

        before = r.engine._pair_multiplier(r.prices.require("gpt-mini-class"))
        res = await r.adelegate(Brief(task="fix", phase=TaskPhase.IMPLEMENT),
                                lead_session_id="lead")
        after = r.engine._pair_multiplier(r.prices.require("gpt-mini-class"))

        assert res.attempts == 2, "failover to the 2nd backend is a 2nd attempt"
        assert r.last_decision.model_id == "gpt-mini-class", "pin held through failover"
        assert after > before, "the failover must raise the multiplier the scorer reads"


class TestSidekickPinTracksAssignment:
    """Regression: the pin was registered only in __init__, so setting
    `router.sidekick_model` after construction (exactly what examples/demo.py
    does) silently lost it and delegation routed to the frontier model."""

    def _router(self):
        from demo import FakeProvider

        prices = PriceRegistry()
        backends = BackendRegistry([
            BackendSpec("anthropic", DeploymentClass.API,
                        models=tuple(m for m in prices.ids()
                                     if m.startswith("claude")),
                        base_url="https://api.anthropic.com", priority=0),
            BackendSpec("openai", DeploymentClass.API,
                        models=tuple(m for m in prices.ids()
                                     if m.startswith("gpt")),
                        base_url="https://api.openai.com", priority=1),
        ])
        r = Router(prices=prices, backends=backends, transport=FakeProvider())
        r._keys = {"anthropic": "sk", "openai": "sk"}
        return r

    def test_pin_registered_when_set_after_construction(self):
        r = self._router()
        assert "__sidekick__" not in r.resolver.profiles
        r.sidekick_model = "gpt-mini-class"
        assert "__sidekick__" in r.resolver.profiles

    @pytest.mark.asyncio
    async def test_delegation_honors_pin_set_after_construction(self):
        r = self._router()
        r.sidekick_model = "gpt-mini-class"
        await r.adelegate(Brief(task="fix", phase=TaskPhase.IMPLEMENT),
                          lead_session_id="lead")
        assert r.last_decision.model_id == "gpt-mini-class"

    def test_setting_none_removes_the_pin(self):
        r = self._router()
        r.sidekick_model = "gpt-mini-class"
        r.sidekick_model = None
        assert "__sidekick__" not in r.resolver.profiles
        assert r.sidekick_model is None

    def test_unknown_sidekick_model_raises(self):
        r = self._router()
        with pytest.raises(ConfigurationError):
            r.sidekick_model = "no-such-model"
        # a rejected assignment must not leave a stale pin behind
        assert "__sidekick__" not in r.resolver.profiles
