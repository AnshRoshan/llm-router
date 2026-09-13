"""The CLI: explain, calibrate, and the eval CI gate."""
from __future__ import annotations

import json

import pytest

from llmrouter.cli import main


CASES = (
    '{"text": "hello there", "workload": "chat", "session_id": "s1"}\n'
    '{"text": "write a quicksort in python", "workload": "code"}\n'
)


@pytest.fixture()
def case_file(tmp_path):
    p = tmp_path / "cases.jsonl"
    p.write_text(CASES, encoding="utf-8")
    return str(p)


def test_explain_prints_an_auditable_decision(capsys):
    rc = main(["explain", "--prompt", "hi", "--workload", "chat"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "model=" in out and "workload=chat" in out and "ttl:" in out


def test_eval_reports_mix_and_quality(case_file, capsys):
    rc = main(["eval", "--data", case_file])
    out = capsys.readouterr().out
    assert rc == 0
    assert "routing mix" in out and "avg quality" in out


def test_eval_ci_gate_passes_above_floor(case_file, capsys):
    rc = main(["eval", "--data", case_file, "--ci", "--quality-floor", "0.75"])
    assert rc == 0
    assert "CI gate passed" in capsys.readouterr().out


def test_eval_ci_gate_fails_below_floor(case_file, capsys):
    rc = main(["eval", "--data", case_file, "--ci",
               "--quality-floor", "0.999", "--min-quality", "0.0"])
    assert rc == 1
    assert "CI gate FAILED" in capsys.readouterr().out


def test_eval_ci_gate_fails_on_min_quality_excluding_everything(
        case_file, capsys):
    rc = main(["eval", "--data", case_file, "--ci",
               "--min-quality", "0.999"])
    assert rc == 1
    assert "unroutable" in capsys.readouterr().out


def test_eval_reports_misroutes_vs_expected_model(tmp_path, capsys):
    p = tmp_path / "cases.jsonl"
    p.write_text(
        json.dumps({"text": "hi", "workload": "chat",
                    "expected_model": "claude-haiku-class"}) + "\n",
        encoding="utf-8")
    rc = main(["eval", "--data", str(p), "--fail-on-misroute", "--ci"])
    out = capsys.readouterr().out
    assert rc == 1
    assert "misroute" in out


def test_calibrate_prints_an_operating_point(case_file, capsys):
    rc = main(["calibrate", "--data", case_file,
               "--target-strong-pct", "0.5", "--quiet"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "calibrated min_quality" in out
    assert "Weights(min_quality=" in out


def test_calibrate_full_table_without_quiet(case_file, capsys):
    rc = main(["calibrate", "--data", case_file, "--target-strong-pct", "0.9"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "min_quality" in out and "strong_share" in out
