"""Workload-class inference: Path A (declared) and Path B (inferred)."""
from __future__ import annotations

from llmrouter import (
    AgentProfile,
    FingerprintRegistry,
    Message,
    RoutingRequest,
    WorkloadClass,
    WorkloadResolver,
    WorkloadSignal,
    WorkloadSource,
    infer_structural,
)


def req(**kw) -> RoutingRequest:
    messages = kw.pop("messages", (Message("user", "hello there"),))
    return RoutingRequest(messages=messages, **kw)


class TestPathB1Structural:
    def test_image_block_is_vision(self):
        r = req(messages=(Message("user", [{"type": "image", "source": {}}]),))
        got = infer_structural(r)
        assert got is not None and got.workload == WorkloadClass.VISION

    def test_tools_plus_tool_result_is_agent(self):
        r = req(
            messages=(
                Message("user", "go"),
                Message("user", [{"type": "tool_result", "content": "ok"}]),
            ),
            tools=({"name": "search"},),
        )
        got = infer_structural(r)
        assert got is not None and got.workload == WorkloadClass.AGENT
        assert got.confidence > 0.9

    def test_tools_without_result_is_still_agent(self):
        r = req(tools=({"name": "search"},))
        got = infer_structural(r)
        assert got is not None and got.workload == WorkloadClass.AGENT

    def test_retrieved_chunks_are_rag(self):
        r = req(retrieved=("chunk one", "chunk two"))
        got = infer_structural(r)
        assert got is not None and got.workload == WorkloadClass.RAG

    def test_code_markers(self):
        r = req(messages=(Message("user", "fix this:\n```py\ndef f():\n    pass\n```"),))
        got = infer_structural(r)
        assert got is not None and got.workload == WorkloadClass.CODE

    def test_multi_turn_no_tools_is_chat(self):
        r = req(messages=(Message("user", "a"), Message("assistant", "b"),
                          Message("user", "c")))
        got = infer_structural(r)
        assert got is not None and got.workload == WorkloadClass.CHAT

    def test_abstains_on_ambiguous_single_turn(self):
        assert infer_structural(req()) is None


class TestPathADeclared:
    def test_explicit_tag_skips_everything_else(self):
        resolver = WorkloadResolver()
        got = resolver.resolve(req(workload=WorkloadClass.BATCH,
                                   retrieved=("x",)))
        assert got.workload == WorkloadClass.BATCH
        assert got.source == WorkloadSource.EXPLICIT_TAG
        assert got.confidence == 1.0

    def test_agent_profile(self):
        resolver = WorkloadResolver(profiles={
            "triage": AgentProfile("triage", WorkloadClass.AGENT),
        })
        got = resolver.resolve(req(agent_profile="triage"))
        assert got.workload == WorkloadClass.AGENT
        assert got.source == WorkloadSource.AGENT_PROFILE

    def test_unknown_profile_falls_back_and_says_so(self):
        resolver = WorkloadResolver()
        got = resolver.resolve(req(agent_profile="nope"))
        assert got.source == WorkloadSource.DEFAULT
        assert "unknown agent profile" in got.reason


class TestPathB2Fingerprint:
    def test_second_request_is_an_exact_free_hit(self):
        resolver = WorkloadResolver()
        sys_prompt = "You are a support bot. " * 40
        # A tool-bearing request so the structural rules resolve it on turn 1.
        messages = (Message("system", sys_prompt), Message("user", "hi"))

        first = resolver.resolve(req(messages=messages, tools=({"name": "search"},)))
        assert first.source == WorkloadSource.STRUCTURAL

        second = resolver.resolve(req(messages=messages, tools=({"name": "search"},)))
        assert second.source == WorkloadSource.FINGERPRINT
        assert second.confidence == 1.0

    def test_ambiguous_first_turn_is_default_then_fingerprint(self):
        """A bare single turn has no structural signal: DEFAULT first, then the
        registry makes every later sighting an exact hit."""
        resolver = WorkloadResolver()
        messages = (Message("system", "sys " * 40), Message("user", "hi"))
        assert resolver.resolve(req(messages=messages)).source == WorkloadSource.DEFAULT
        assert resolver.resolve(req(messages=messages)).source == WorkloadSource.FINGERPRINT

    def test_fingerprint_ignores_the_user_turn(self):
        sys_prompt = "You are a support bot. " * 40
        a = req(messages=(Message("system", sys_prompt), Message("user", "first")))
        b = req(messages=(Message("system", sys_prompt), Message("user", "totally different")))
        assert a.prompt_fingerprint() == b.prompt_fingerprint()

    def test_fingerprint_changes_with_tool_schema(self):
        a = req(tools=({"name": "search"},))
        b = req(tools=({"name": "search", "extra": 1},))
        assert a.prompt_fingerprint() != b.prompt_fingerprint()

    def test_volatile_client_detection(self):
        reg = FingerprintRegistry()
        for i in range(60):
            reg.observe(f"fp-{i}", "client-with-timestamps")
        reg.observe("stable", "good-client")
        assert "client-with-timestamps" in reg.volatile_clients(threshold=50)
        assert "good-client" not in reg.volatile_clients(threshold=50)

    def test_high_confidence_is_not_overwritten_by_low(self):
        reg = FingerprintRegistry()
        strong = WorkloadSignal(WorkloadClass.AGENT, WorkloadSource.STRUCTURAL, 0.97, "x")
        weak = WorkloadSignal(WorkloadClass.CHAT, WorkloadSource.DEFAULT, 0.3, "y")
        reg.put("fp", strong)
        reg.put("fp", weak)
        assert reg.get("fp").workload == WorkloadClass.AGENT


class TestPathB3AndB4:
    def test_classifier_only_runs_when_b1_and_b2_abstain(self):
        calls = []

        class Spy:
            def classify(self, request):
                calls.append(request)
                return WorkloadSignal(WorkloadClass.RAG, WorkloadSource.CLASSIFIER,
                                      0.8, "classifier said so")

        resolver = WorkloadResolver(classifier=Spy())
        # Ambiguous single turn, nothing declared -> abstains -> classifier runs.
        got = resolver.resolve(req())
        assert len(calls) == 1
        assert got.workload == WorkloadClass.RAG
        assert got.source == WorkloadSource.CLASSIFIER

    def test_classifier_is_not_called_on_later_turns(self):
        calls = []

        class Spy:
            def classify(self, request):
                calls.append(request)
                return None

        resolver = WorkloadResolver(classifier=Spy())
        resolver.resolve(req(messages=(Message("user", "a"), Message("assistant", "b"),
                                       Message("user", "c"))))
        assert calls == [], "per-step classification is a non-goal"

    def test_never_fails_to_route(self):
        resolver = WorkloadResolver()
        got = resolver.resolve(req())
        assert got.source == WorkloadSource.DEFAULT
        assert got.workload is not None


class TestStickyKey:
    def test_declared_session_id_wins(self):
        assert req(session_id="s-42").sticky_key() == "sid:s-42"

    def test_first_user_turn_doubles_as_a_session_id(self):
        a = req(messages=(Message("user", "same opener"), Message("assistant", "x"),
                          Message("user", "follow up")))
        b = req(messages=(Message("user", "same opener"),))
        assert a.sticky_key() == b.sticky_key()
        assert a.sticky_key().startswith("ctx:")

    def test_different_openers_are_different_sessions(self):
        a = req(messages=(Message("user", "opener one"),))
        b = req(messages=(Message("user", "opener two"),))
        assert a.sticky_key() != b.sticky_key()

    def test_falls_back_to_fingerprint_when_no_user_text(self):
        r = req(messages=(Message("system", "sys"),))
        assert r.sticky_key().startswith("fp:")

    def test_full_prefix_is_not_a_session_key(self):
        # Keying on the whole prefix would change every turn.
        a = req(messages=(Message("user", "x"),))
        b = req(messages=(Message("user", "x"), Message("assistant", "y"),
                          Message("user", "z")))
        assert a.sticky_key() == b.sticky_key()
