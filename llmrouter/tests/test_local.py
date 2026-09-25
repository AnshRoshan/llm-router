"""Tests for the local/self-hosted fleet: GPU economics, discovery,
with_ollama routing, constraint and saturation behavior."""
import json

import pytest

from llmrouter import Message, Router, WorkloadClass
from llmrouter.local import (
    gpu_price_card,
    ollama_backend,
    ollama_models,
    ollama_price_cards,
)
from llmrouter.types import Constraints, RoutingRequest, DeploymentClass


# --------------------------------------------------------------------------- #
# honest GPU economics
# --------------------------------------------------------------------------- #
def test_gpu_price_card_math():
    card = gpu_price_card("llama-test", gpu_cost_per_hr=0.35,
                          tokens_per_sec=35.0, prefill_mult=6.0)
    # 0.35 / (35 * 3600) * 1e6 = $2.7778/MTok output
    assert card.output_per_mtok == pytest.approx(2.777777, abs=1e-4)
    assert card.input_per_mtok == pytest.approx(2.777777 / 6, abs=1e-4)
    assert card.deployment == DeploymentClass.SELF_HOSTED
    assert card.cache is not None
    assert card.task_cost_multiplier == 1.0


def test_gpu_price_card_vision_is_excluded_not_downgraded():
    card = gpu_price_card("text-only", supports_vision=False)
    assert card.quality("vision") < 0  # hard-constraint exclusion signal
    seen = gpu_price_card("vl-model", supports_vision=True)
    assert seen.quality("vision") > 0


def test_free_when_gpu_cost_zero():
    card = gpu_price_card("owned-box", gpu_cost_per_hr=0.0)
    assert card.input_per_mtok == 0.0 and card.output_per_mtok == 0.0


def test_bad_throughput_raises():
    with pytest.raises(ValueError):
        gpu_price_card("m", tokens_per_sec=0)


def test_price_card_is_honest_vs_cloud():
    # A slow-but-cheap box ($0.35/hr @ 35 tok/s = $2.78/MTok out) BEATS
    # gpt-mini list price ($3/MTok); a rented GPU at 140 tok/s on a $1.80/hr
    # box ($3.57/MTok) does not. Local wins on cost only when it's actually
    # cheap — this test pins that arithmetic.
    cheap_local = gpu_price_card("llama-8b", gpu_cost_per_hr=0.35,
                                 tokens_per_sec=35)
    cloud = Router.pure().prices.require("gpt-mini-class")
    assert cheap_local.output_per_mtok < cloud.output_per_mtok
    fast_but_pricey = gpu_price_card("llama-70b", gpu_cost_per_hr=1.80,
                                     tokens_per_sec=140)
    assert fast_but_pricey.output_per_mtok > cloud.output_per_mtok


# --------------------------------------------------------------------------- #
# discovery (setup-time only — the decision layer stays offline)
# --------------------------------------------------------------------------- #
def test_ollama_models_discovery(monkeypatch):
    payload = {"models": [{"name": "llama3.1:8b"}, {"name": "qwen2.5-coder:7b"},
                          {"name": "llama3.1:8b"}, {"digest": "no-name"}]}

    class Resp:
        def read(self):
            return json.dumps(payload).encode()

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    seen = {}

    def fake_urlopen(url, timeout=None):
        seen["url"] = url
        return Resp()

    monkeypatch.setattr("llmrouter.local.urllib.request.urlopen", fake_urlopen)
    names = ollama_models("http://localhost:11434/")
    assert names == ("llama3.1:8b", "qwen2.5-coder:7b")
    assert seen["url"] == "http://localhost:11434/api/tags"


# --------------------------------------------------------------------------- #
# with_ollama: one score across laptop and cloud
# --------------------------------------------------------------------------- #
def _local_router(**kw):
    kw.setdefault("models", ("llama3.1:8b", "qwen2.5-coder:32b"))
    return Router.with_ollama(
        per_model={"qwen2.5-coder:32b": {"quality": 0.86,
                                         "supports_tools": True,
                                         "tokens_per_sec": 9.0,
                                         "gpu_cost_per_hr": 1.80}},
        **kw,
    )


def test_local_only_fleet_routes_local():
    router = _local_router()
    d = router.decide(RoutingRequest(
        messages=(Message("user", "summarize this document please"),),
        workload=WorkloadClass.CHAT))
    assert d.backend_id == "ollama"
    assert d.model_id in ("llama3.1:8b", "qwen2.5-coder:32b")


def test_no_models_and_no_server_is_a_clear_error(monkeypatch):
    def empty(base_url="http://localhost:11434", timeout_s=5.0):
        return ()
    monkeypatch.setattr("llmrouter.local.ollama_models", empty)
    with pytest.raises(ValueError, match="no local models found"):
        Router.with_ollama()


def test_hybrid_tool_requests_go_to_a_tool_capable_candidate():
    # The 8b card is tools=False; a require_tools request must pick the 32b
    # local model (tools=True) or the cloud — never the laptop 8b.
    router = _local_router(openai_key="sk-test")
    d = router.decide(RoutingRequest(
        messages=(Message("user", "use the weather tool"),),
        workload=WorkloadClass.AGENT,
        constraints=Constraints(require_tools=True),
        tools=({"name": "weather"},)))
    winner = router.engine.prices.require(d.model_id)
    assert winner.supports_tools


def test_saturated_gpu_cascades_to_cloud():
    router = _local_router(openai_key="sk-test")
    h = router.backends.health("ollama")
    h.in_flight = 10  # capacity_rps default 2.0 -> shedding
    d = router.decide(RoutingRequest(
        messages=(Message("user", "summarize this document please"),),
        workload=WorkloadClass.CHAT))
    assert d.backend_id == "openai"
    # and once it drains, local capacity is a candidate again
    h.in_flight = 0
    d2 = router.decide(RoutingRequest(
        messages=(Message("user", "summarize that other document"),),
        workload=WorkloadClass.CHAT))
    assert d2.backend_id == "ollama"


def test_local_backend_spec_is_operationally_sane():
    spec = ollama_backend(["a", "b"])
    assert spec.timeout_s == 8.0          # aggressive: beat the cascade trap
    assert spec.capacity_rps == 2.0
    assert spec.deployment == DeploymentClass.SELF_HOSTED
    assert spec.supports("a") and not spec.supports("z")


def test_price_cards_carry_per_model_overrides():
    cards = ollama_price_cards(("m-a", "m-b"), quality=0.5,
                               per_model={"m-b": {"quality": 0.9}})
    assert cards["m-a"].quality("chat") == 0.5
    assert cards["m-b"].quality("chat") == 0.9


def test_local_appears_in_stats_and_explain():
    router = _local_router()
    d = router.decide(RoutingRequest(
        messages=(Message("user", "hi"),), workload=WorkloadClass.CHAT))
    assert "ollama" in router.backends.ids()
    text = d.explain()
    assert "ollama" in text
