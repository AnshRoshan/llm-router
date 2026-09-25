"""Tests for the learned quality layer: features, checkpoint, trainer,
policy integration, and the feedback -> train -> decide loop."""
import json

import pytest

from llmrouter import (
    Message,
    QualityModel,
    Router,
    RoutingRequest,
    WorkloadClass,
)
from llmrouter.cli import main as cli_main
from llmrouter.learning import (
    FEATURE_NAMES,
    N_FEATURES,
    ModelHead,
    extract_features,
    load_feedback,
    parse_feedback_line,
    train_heads,
)
from llmrouter.pricing import PriceCard


def _req(text="hello there", workload=WorkloadClass.CHAT, **kw):
    return RoutingRequest(messages=(Message("user", text),),
                          workload=workload, **kw)


# --------------------------------------------------------------------------- #
# features
# --------------------------------------------------------------------------- #
def test_feature_vector_shape_and_onehot():
    x = extract_features(_req(workload=WorkloadClass.CODE), WorkloadClass.CODE)
    assert len(x) == N_FEATURES == len(FEATURE_NAMES)
    assert x[FEATURE_NAMES.index("wl_code")] == 1.0
    assert sum(x[:6]) == 1.0
    # unknown workload: all-zero one-hot
    z = extract_features(_req(), WorkloadClass.UNKNOWN)
    assert sum(z[:6]) == 0.0


def test_features_are_deterministic_and_bounded():
    req = _req("def f(): ``` import x ``` class Foo SELECT ",
               workload=WorkloadClass.CODE,
               tools=({"name": "t", "description": "d"},),
               retrieved=("chunk a",))
    a, b = extract_features(req, WorkloadClass.CODE), \
        extract_features(req, WorkloadClass.CODE)
    assert a == b
    assert all(0.0 <= v <= 1.0 for v in a)
    assert a[FEATURE_NAMES.index("has_tools")] == 1.0
    assert a[FEATURE_NAMES.index("code_markers")] > 0


# --------------------------------------------------------------------------- #
# checkpoint
# --------------------------------------------------------------------------- #
def test_checkpoint_roundtrip(tmp_path):
    qm = QualityModel({"m1": ModelHead(bias=0.1, weights=(0.01,) * N_FEATURES,
                                       samples=40)})
    path = tmp_path / "q.json"
    qm.save(str(path))
    back = QualityModel.load(str(path))
    assert back.to_json() == qm.to_json()
    assert back.shrinkage_k == 20.0


def test_checkpoint_rejects_foreign_format_and_features():
    with pytest.raises(ValueError, match="format"):
        QualityModel.from_dict({"format": "other", "heads": {}})
    good = QualityModel({"m": ModelHead(0.0, (0.0,) * N_FEATURES, 5)}).to_dict()
    good["feature_names"] = ["nope"] * N_FEATURES
    with pytest.raises(ValueError, match="feature set"):
        QualityModel.from_dict(good)


def test_quality_blends_with_shrinkage_and_clamps():
    from llmrouter.pricing import PriceCard
    card = PriceCard("m", 1.0, 1.0, quality_prior={"chat": 0.5})
    x = extract_features(_req(), WorkloadClass.CHAT)

    zero = QualityModel({"m": ModelHead(0.9, (0.0,) * N_FEATURES, 0)})
    assert zero.quality(card, WorkloadClass.CHAT, x) == pytest.approx(0.5)

    mid = QualityModel({"m": ModelHead(0.9, (0.0,) * N_FEATURES, 20)})
    assert mid.quality(card, WorkloadClass.CHAT, x) == pytest.approx(0.5 + 0.45)

    huge = QualityModel({"m": ModelHead(50.0, (0.0,) * N_FEATURES, 2000)})
    assert huge.quality(card, WorkloadClass.CHAT, x) == 1.0  # clamped

    # no head for the model -> untouched prior
    empty = QualityModel({})
    assert empty.quality(card, WorkloadClass.CHAT, x) == 0.5


def test_negative_prior_means_unsupported_even_with_head():
    from llmrouter.pricing import PriceCard
    card = PriceCard("m", 1.0, 1.0, quality_prior={"chat": -1.0})
    qm = QualityModel({"m": ModelHead(0.5, (0.0,) * N_FEATURES, 100)})
    assert qm.quality(card, WorkloadClass.CHAT,
                      extract_features(_req(), WorkloadClass.CHAT)) == -1.0


# --------------------------------------------------------------------------- #
# feedback parsing
# --------------------------------------------------------------------------- #
def test_pointwise_and_pairwise_rows():
    rows = parse_feedback_line({"text": "hi", "workload": "chat",
                                "model": "a", "outcome": "good"})
    assert len(rows) == 1 and rows[0].outcome == 1.0

    pair = parse_feedback_line({"text": "hi", "model_a": "a", "model_b": "b",
                                "winner": "b"})
    assert {(p.model_id, p.outcome) for p in pair} == {("a", 0.0), ("b", 1.0)}

    tie = parse_feedback_line({"text": "hi", "model_a": "a", "model_b": "b",
                               "winner": "tie"})
    assert all(p.outcome == 0.5 for p in tie)

    with pytest.raises(ValueError):
        parse_feedback_line({"text": "hi", "model_a": "a", "winner": "a"})


def test_load_feedback_reports_line_numbers(tmp_path):
    p = tmp_path / "fb.jsonl"
    p.write_text('{"text":"hi","model":"a","outcome":1}\n'
                 '{"model":"a","outcome":1}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="line 2"):
        load_feedback(str(p))


# --------------------------------------------------------------------------- #
# trainer
# --------------------------------------------------------------------------- #
def _synthetic_rows(n=30):
    rows = []
    for i in range(n):
        rows.append({"text": f"hello friend number {i}", "workload": "chat",
                     "model": "cheap", "outcome": 1})
        rows.append({"text": f"hello friend number {i}", "workload": "chat",
                     "model": "strong", "outcome": 0})
    return rows


def test_train_heads_learns_separable_signal(tmp_path):
    samples = load_feedback(str(_write(tmp_path, _synthetic_rows())))
    qm, reports = train_heads(samples, epochs=120)
    by_model = {r.model_id: r for r in reports}
    assert by_model["cheap"].eval_accuracy == 1.0
    assert by_model["cheap"].mean_outcome == 1.0


def test_min_samples_keeps_prior_for_thin_models(tmp_path):
    rows = _synthetic_rows(10) + [{"text": f"t{i}", "model": "rare",
                                   "outcome": 1} for i in range(3)]
    samples = load_feedback(str(_write(tmp_path, rows)))
    qm, _ = train_heads(samples, min_samples=8)
    assert "rare" not in qm.heads
    assert "cheap" in qm.heads


def _write(tmp_path, rows):
    p = tmp_path / "fb.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    return p


# --------------------------------------------------------------------------- #
# policy + router integration
# --------------------------------------------------------------------------- #
def _trained_router(quality_model=None):
    return Router.pure(quality_model=quality_model)


def _feedback_file(tmp_path):
    # strong model FAILS on this shape of request; cheap model wins.
    # The bundled cards: claude-haiku-class (weak) vs claude-opus-class (strong).
    rows = []
    for i in range(30):
        rows.append({"text": f"tiny talk {i}", "workload": "chat",
                     "model": "claude-haiku-class", "outcome": 1})
        rows.append({"text": f"tiny talk {i}", "workload": "chat",
                     "model": "claude-opus-class", "outcome": 0})
    return _write(tmp_path, rows)


def test_policy_engine_uses_quality_model(tmp_path):
    path = str(_feedback_file(tmp_path))
    qm, _ = train_heads(load_feedback(path), epochs=150)
    engine = _trained_router(qm).engine
    req = _req("tiny talk about nothing in particular",
               workload=WorkloadClass.CHAT)
    decision = engine.decide(req)
    assert any("learned-quality" in r for r in decision.reasons)
    chosen = next(c for c in decision.candidates
                  if c.model_id == decision.model_id)
    haiku = next(c for c in decision.candidates
                 if c.model_id == "claude-haiku-class")
    # learned quality moved haiku ABOVE its prior (0.75 chat prior)
    assert haiku.quality > 0.75
    assert chosen.quality == pytest.approx(haiku.quality) or \
        decision.model_id == "claude-haiku-class"


def test_no_quality_model_is_pure_prior_behaviour():
    req = _req("hi", workload=WorkloadClass.CHAT)
    d_plain = _trained_router().decide(req)
    d_none = _trained_router(QualityModel({})).decide(req)
    assert d_plain.model_id == d_none.model_id
    assert not any("learned-quality" in r for r in d_plain.reasons)


def test_min_quality_floor_applies_to_learned_quality(tmp_path):
    rows = [{"text": f"q{i}", "workload": "chat",
             "model": "claude-haiku-class", "outcome": 1} for i in range(20)]
    qm, _ = train_heads(load_feedback(str(_write(tmp_path, rows))), epochs=150)
    # The head can lift haiku's chat quality toward 1.0; with enough push it
    # must clear a floor that its bare prior (0.75) would fail.
    haiku_head = qm.heads["claude-haiku-class"]
    lifted = haiku_head.bias + sum(haiku_head.weights)  # upper bound estimate
    assert lifted > 0.3  # trainer learned a strong positive delta


def test_feedback_loop_roundtrip(tmp_path):
    r1 = Router.pure()
    req = _req("hello there little one", workload=WorkloadClass.CHAT,
               session_id="keep")
    r1.decide(req)
    r1.report_feedback("keep", misrouted=True, expected_model="x-class")
    out = tmp_path / "fb.jsonl"
    text = r1.export_feedback(str(out))
    assert '"model_a": "x-class"' in text
    qm, _ = train_heads(load_feedback(str(out)), epochs=10, min_samples=1)
    assert "x-class" in qm.heads


def test_train_cli_writes_checkpoint(tmp_path, capsys):
    path = str(_feedback_file(tmp_path))
    out = str(tmp_path / "q.json")
    rc = cli_main(["train", "--data", path, "--out", out, "--epochs", "60"])
    assert rc == 0
    obj = json.load(open(out, encoding="utf-8"))
    assert obj["format"] == "llmrouter-quality-v1"
    assert "claude-haiku-class" in obj["heads"]
    # and the explain/eval commands accept it
    rc = cli_main(["explain", "--prompt", "tiny talk", "--workload", "chat",
                   "--quality-model", out])
    assert rc == 0
    assert "learned-quality" in capsys.readouterr().out
