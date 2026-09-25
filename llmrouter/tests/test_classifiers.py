"""Tests for the Laya B3 classifier adapter: mapping, abstention, and
end-to-end resolver wiring (including fingerprint learning)."""
import json

import pytest

from llmrouter import (
    LayaWorkloadClassifier,
    Message,
    Router,
    RoutingRequest,
    WorkloadClass,
    WorkloadResolver,
)
from llmrouter.types import WorkloadSource


def _body(choice="code", **extra):
    ans = {"choice": choice}
    ans.update(extra)
    return {"answers": {"workload": ans}, "routing": {"model": "english"}}


class FakeUrlopen:
    """Context-manager fake of urllib.request.urlopen."""

    def __init__(self, payload=None, error=None):
        self.payload, self.error = payload, error
        self.seen = {}

    def __call__(self, req, timeout=None):
        self.seen = {"url": req.full_url,
                     "body": json.loads(req.data), "timeout": timeout}
        if self.error:
            raise self.error

        class Resp:
            def read(inner):
                return json.dumps(self.payload).encode()

            def __enter__(inner):
                return inner

            def __exit__(inner, *a):
                return False
        return Resp()


@pytest.fixture()
def classifier(monkeypatch):
    def install(payload=None, error=None, **kw):
        fake = FakeUrlopen(payload, error)
        monkeypatch.setattr("llmrouter.classifiers.urllib.request.urlopen",
                            fake)
        return LayaWorkloadClassifier(**kw), fake
    return install


# --------------------------------------------------------------------------- #
def test_choice_maps_to_workload_with_probability(classifier):
    c, fake = classifier(_body("agent", probability=0.91))
    sig = c.classify(RoutingRequest(messages=(Message("user", "do the thing"),)))
    assert sig is not None
    assert sig.workload == WorkloadClass.AGENT
    assert sig.source == WorkloadSource.CLASSIFIER
    assert sig.confidence == pytest.approx(0.91)
    # request shape follows Laya's documented /predict contract
    q = fake.seen["body"]["questions"]["workload"]
    assert q["type"] == "choice"
    assert set(q["criteria"]) == {"agent", "chat", "rag", "batch", "code",
                                  "vision"}
    assert fake.seen["body"]["state"]["body"] == "do the thing"


def test_scores_dict_is_used_when_no_probability_field(classifier):
    c, _ = classifier(_body("rag", scores={"rag": 0.77, "chat": 0.2}))
    sig = c.classify(RoutingRequest(messages=(Message("user", "x"),)))
    assert sig.confidence == pytest.approx(0.77)


def test_confidence_floor_abstains(classifier):
    c, _ = classifier(_body("code", probability=0.2))
    assert c.classify(RoutingRequest(messages=(Message("user", "x"),))) is None


def test_unknown_label_and_failures_abstain(classifier):
    c, _ = classifier(_body("sentiment-analysis"))
    assert c.classify(RoutingRequest(messages=(Message("user", "x"),))) is None

    c2, _ = classifier(error=OSError("connection refused"))
    assert c2.classify(RoutingRequest(messages=(Message("user", "x"),))) is None

    c3, _ = classifier({"garbage": True})
    assert c3.classify(RoutingRequest(messages=(Message("user", "x"),))) is None

    c4, _ = classifier(_body("code"))
    assert c4.classify(RoutingRequest(messages=())) is None  # empty request


def test_timeout_bound_is_passed_to_the_http_call(classifier):
    c, fake = classifier(_body("chat"))
    c.classify(RoutingRequest(messages=(Message("user", "hi"),)))
    assert fake.seen["timeout"] == 2.0


# --------------------------------------------------------------------------- #
# end-to-end: this is what the B3 hook is FOR
# --------------------------------------------------------------------------- #
def test_resolver_uses_laya_only_when_b1_b2_abstain_then_learns(classifier):
    c, fake = classifier(_body("code", probability=0.88))
    resolver = WorkloadResolver(classifier=c)
    # "hello there" makes structural rules abstain (no tools/markers/turn-2)
    req = RoutingRequest(messages=(Message("user", "hello there buddy"),))
    sig = resolver.resolve(req)
    assert sig.workload == WorkloadClass.CODE
    assert sig.source == WorkloadSource.CLASSIFIER
    assert fake.seen  # the HTTP call actually happened

    # turn 2+: the fingerprint registry learned it — the classifier must NOT
    # be consulted again (proved by making any further call explode)
    fake.error = AssertionError("classifier called twice — fingerprint "
                                "registry should have answered")
    sig2 = resolver.resolve(req)
    assert sig2.source == WorkloadSource.FINGERPRINT
    assert sig2.workload == WorkloadClass.CODE


def test_router_construction_accepts_the_classifier():
    c = LayaWorkloadClassifier()
    router = Router.pure(classifier=c)
    assert router.resolver.classifier is c


# --------------------------------------------------------------------------- #
# in-process family: callable wrapper, GLiClass, chain
# --------------------------------------------------------------------------- #
def _req(text="hello there buddy"):
    return RoutingRequest(messages=(Message("user", text),))


def test_callable_classifier_label_tuple_and_none():
    from llmrouter import CallableWorkloadClassifier
    clf = CallableWorkloadClassifier(lambda t: "code", name="head")
    sig = clf.classify(_req())
    assert sig.workload == WorkloadClass.CODE
    assert "head: code" in sig.reason

    clf2 = CallableWorkloadClassifier(lambda t: ("rag", 0.4))
    assert clf2.classify(_req()).confidence == pytest.approx(0.4)
    assert CallableWorkloadClassifier(lambda t: ("rag", 0.1)).classify(_req()) is None
    assert CallableWorkloadClassifier(lambda t: None).classify(_req()) is None
    assert CallableWorkloadClassifier(lambda t: "nonsense").classify(_req()) is None

    def boom(_):
        raise RuntimeError("model on fire")
    assert CallableWorkloadClassifier(boom).classify(_req()) is None


def test_gliclass_abstains_cleanly_when_package_missing():
    # gliclass is an OPTIONAL user-side install; with it absent the adapter
    # must abstain, not ImportError, so routing keeps working.
    from llmrouter import GLiClassWorkloadClassifier
    clf = GLiClassWorkloadClassifier()
    try:
        import gliclass  # noqa: F401
        pytest.skip("gliclass installed — abstention path not reachable")
    except ImportError:
        pass
    assert clf.classify(_req()) is None
    assert clf._import_failed is True


def test_gliclass_parses_pipeline_scores(monkeypatch):
    from llmrouter import GLiClassWorkloadClassifier

    class FakePipe:
        def run(self, text, labels):
            return [{"agent": 0.05, "code": 0.72, "chat": 0.1,
                     "rag": 0.03, "batch": 0.05, "vision": 0.05}]

    clf = GLiClassWorkloadClassifier()
    monkeypatch.setattr(clf, "_get_pipeline", lambda: FakePipe())
    sig = clf.classify(_req())
    assert sig.workload == WorkloadClass.CODE
    assert sig.confidence == pytest.approx(0.72)
    assert "gliclass" in sig.reason


def test_chain_first_non_abstention_wins():
    from llmrouter import CallableWorkloadClassifier, ChainWorkloadClassifier

    order = []
    first = CallableWorkloadClassifier(
        lambda t: order.append("first") or None, name="a")
    second = CallableWorkloadClassifier(
        lambda t: order.append("second") or ("batch", 0.9), name="b")
    third = CallableWorkloadClassifier(
        lambda t: order.append("third") or ("chat", 0.9), name="c")
    chain = ChainWorkloadClassifier([first, second, third])
    sig = chain.classify(_req())
    assert sig.workload == WorkloadClass.BATCH
    assert order == ["first", "second"]   # third never consulted

    # a crashing link does not kill the chain
    bad = CallableWorkloadClassifier(
        lambda t: (_ for _ in ()).throw(ValueError("nope")))
    good = CallableWorkloadClassifier(lambda t: ("agent", 0.8))
    sig2 = ChainWorkloadClassifier([bad, good]).classify(_req())
    assert sig2.workload == WorkloadClass.AGENT
    assert ChainWorkloadClassifier([bad]).classify(_req()) is None


def test_chain_with_laya_http_link_and_local_fallback(classifier):
    from llmrouter import CallableWorkloadClassifier, ChainWorkloadClassifier
    laya, fake = classifier(_body("vision", probability=0.8))
    fake.error = OSError("server down")  # link 1 abstains
    fallback = CallableWorkloadClassifier(lambda t: ("code", 0.6), name="fn")
    sig = ChainWorkloadClassifier([laya, fallback]).classify(_req())
    assert sig.workload == WorkloadClass.CODE
    assert "fn: code" in sig.reason


def test_chain_wired_through_the_resolver(classifier):
    from llmrouter import CallableWorkloadClassifier, ChainWorkloadClassifier
    c, _ = classifier(_body("agent", probability=0.9))
    resolver = WorkloadResolver(classifier=ChainWorkloadClassifier([
        CallableWorkloadClassifier(lambda t: None), c]))
    sig = resolver.resolve(_req())
    assert sig.source == WorkloadSource.CLASSIFIER
    assert sig.workload == WorkloadClass.AGENT
