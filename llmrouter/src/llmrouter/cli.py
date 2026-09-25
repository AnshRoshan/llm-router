"""Command-line surface: the learning layer's minimum viable loop.

Five commands, all running the PURE decision layer (no network, no keys):

  explain     route one prompt and print the full auditable decision
  calibrate   sweep the quality floor to hit a target strong-model share
              (the RouteLLM calibrate_threshold pattern, per-workload)
  eval        run a labeled case file, report quality/cost/mix/latency,
              and — with --ci — exit non-zero when quality drops below
              the floor (the pre-merge gate)
  train       fit logistic quality heads from feedback JSONL and write a
              checkpoint the engines load (the offline half of the loop)
  serve       expose the same decision over HTTP for LiteLLM/Bifrost/custom
              gateways (POST /decide -> Decision JSON; the serve module owns
              this one)

Case file format: one JSON object per line. Shorthand is `{"text": "..."}`;
full form is `{"messages": [...], "workload": "chat", "session_id": "s1",
"expected_model": "claude-haiku-class"}`. `expected_model` turns eval into a
misroute audit. Lines starting with `#` are comments.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from dataclasses import replace
from typing import Any, Mapping, Sequence

from .learning import QualityModel, load_feedback, train_heads
from .policy import PolicyEngine, RouterConfig, Weights
from .router import Router
from .types import Message, RoutingRequest, WorkloadClass

#: Decision-latency alarm (ms). The decision layer is pure arithmetic; if the
#: p50 approaches this the router is costing more than it saves.
DECISION_LATENCY_ALARM_MS = 10.0


# --------------------------------------------------------------------------- #
# A pure decision engine over the bundled defaults
# --------------------------------------------------------------------------- #
def pure_engine(config: RouterConfig | None = None,
                quality_model: QualityModel | None = None) -> PolicyEngine:
    """Decision-only engine: bundled price cards, one logical backend per
    provider, no keys and no network. This is what the CLI runs on."""
    return Router.pure(config=config, quality_model=quality_model).engine


def _quality_model_from(args: argparse.Namespace) -> QualityModel | None:
    path = getattr(args, "quality_model", None)
    return QualityModel.load(path) if path else None


def build_request(args: argparse.Namespace) -> RoutingRequest:
    tools: tuple[Mapping[str, Any], ...] = ()
    if getattr(args, "tools", None):
        with open(args.tools, encoding="utf-8") as f:
            loaded = json.load(f)
        if not isinstance(loaded, list):
            loaded = [loaded]
        tools = tuple(loaded)

    messages: list[Message] = []
    if getattr(args, "system", None):
        messages.append(Message("system", args.system))
    messages.append(Message("user", args.prompt))

    workload = getattr(args, "workload", None)
    return RoutingRequest(
        messages=tuple(messages),
        workload=WorkloadClass(workload) if workload else None,
        session_id=getattr(args, "session_id", None),
        tools=tools,
    )


# --------------------------------------------------------------------------- #
# Case files
# --------------------------------------------------------------------------- #
def load_cases(path: str) -> list[tuple[int, Mapping[str, Any], tuple[Message, ...]]]:
    """Parse a JSONL case file into (line_number, meta, messages)."""
    if path == "-":
        raw = sys.stdin.read()
    else:
        with open(path, encoding="utf-8") as f:
            raw = f.read()

    cases: list[tuple[int, Mapping[str, Any], tuple[Message, ...]]] = []
    for lineno, line in enumerate(raw.splitlines(), start=1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        obj = json.loads(line)
        if "messages" in obj:
            msgs = tuple(
                Message(str(m.get("role", "user")), str(m.get("content", "")))
                for m in obj["messages"]
            )
        else:
            text = obj.get("text", obj.get("prompt"))
            if text is None:
                raise ValueError(
                    f"line {lineno}: case needs 'messages' or 'text'"
                )
            msgs = (Message("user", str(text)),)
        cases.append((lineno, obj, msgs))
    return cases


def _case_request(meta: Mapping[str, Any], msgs: tuple[Message, ...],
                  default_workload: str) -> RoutingRequest:
    workload = meta.get("workload", default_workload)
    return RoutingRequest(
        messages=msgs,
        workload=WorkloadClass(str(workload)),
        session_id=meta.get("session_id"),
        tools=tuple(meta.get("tools", ()) or ()),
        retrieved=tuple(str(r) for r in (meta.get("retrieved", ()) or ())),
        tags=frozenset(str(t) for t in (meta.get("tags", ()) or ())),
    )


def _strong_models(engine: PolicyEngine, workload: WorkloadClass) -> set[str]:
    """A model is 'strong' for a workload when it carries the best quality prior
    available. Mirrors RouteLLM's strong/weak split for N>2 candidates."""
    qualities = {
        c.model_id: c.quality(workload.value) for c in engine.prices.all()
    }
    if not qualities:
        return set()
    top = max(qualities.values())
    return {m for m, q in qualities.items() if q == top}


# --------------------------------------------------------------------------- #
# explain
# --------------------------------------------------------------------------- #
def cmd_explain(args: argparse.Namespace) -> int:
    engine = pure_engine(quality_model=_quality_model_from(args))
    decision = engine.decide(build_request(args))
    print(decision.explain())
    return 0


# --------------------------------------------------------------------------- #
# calibrate
# --------------------------------------------------------------------------- #
def cmd_calibrate(args: argparse.Namespace) -> int:
    cases = load_cases(args.data)
    if not cases:
        print("no cases found in", args.data, file=sys.stderr)
        return 2

    base = pure_engine().config
    grid = [round(0.05 * i, 2) for i in range(20)]  # 0.00 .. 0.95
    target = args.target_strong_pct

    rows: list[tuple[float, float, int]] = []  # (threshold, strong_share, unroutable)
    for threshold in grid:
        engine = pure_engine(RouterConfig(
            weights=replace(base.weights, min_quality=threshold)))
        strong_hits = 0
        unroutable = 0
        for _, meta, msgs in cases:
            req = _case_request(meta, msgs, args.workload)
            strong = _strong_models(engine, req.workload)
            try:
                d = engine.decide(req)
            except RuntimeError:
                unroutable += 1
                continue
            if d.model_id in strong:
                strong_hits += 1
        share = strong_hits / len(cases)
        rows.append((threshold, share, unroutable))

    valid = [r for r in rows if r[2] < len(cases)]
    if not valid:
        print("every threshold unrouted every case — the quality floor exceeds "
              "all bundled quality priors", file=sys.stderr)
        return 2

    # Closest share to the target; ties resolve toward the HIGHER floor, which
    # is the more conservative operating point.
    best = min(valid, key=lambda r: (abs(r[1] - target), -r[0]))

    if not args.quiet:
        print(f"{'min_quality':>12} {'strong_share':>13} {'unroutable':>11}")
        for t, share, unroutable in rows:
            marker = "  <-- calibrated" if t == best[0] else ""
            print(f"{t:>12.2f} {share:>13.3f} {unroutable:>11}{marker}")

    print(f"\ncalibrated min_quality = {best[0]:.2f} "
          f"(target strong-model share {target:.2f}, achieved {best[1]:.3f} "
          f"on {len(cases)} cases)")
    print("set it with: RouterConfig(weights=Weights(min_quality="
          f"{best[0]}))")
    return 0


# --------------------------------------------------------------------------- #
# eval — the CI gate
# --------------------------------------------------------------------------- #
def cmd_eval(args: argparse.Namespace) -> int:
    base = pure_engine().config
    engine = pure_engine(RouterConfig(
        weights=replace(base.weights, min_quality=args.min_quality)),
        quality_model=_quality_model_from(args))
    cases = load_cases(args.data)
    if not cases:
        print("no cases found in", args.data, file=sys.stderr)
        return 2

    mix: dict[str, int] = {}
    qualities: list[float] = []
    costs = 0.0
    latencies: list[float] = []
    misroutes: list[str] = []
    unroutable: list[int] = []

    for lineno, meta, msgs in cases:
        req = _case_request(meta, msgs, args.workload)
        try:
            d = engine.decide(req)
        except RuntimeError as exc:
            unroutable.append(lineno)
            if args.verbose:
                print(f"line {lineno}: UNROUTABLE: {exc}")
            continue
        # Decision latency: measured on a second decide() because the decision
        # is pure and deterministic — the reported number is real wall-clock.
        latencies.append(_measure_decide(engine, req))
        card = engine.prices.require(d.model_id)
        chosen = next(
            (c for c in d.candidates
             if c.model_id == d.model_id and c.backend_id == d.backend_id),
            None,
        )
        q = chosen.quality if chosen else card.quality(d.workload.value)
        qualities.append(q)
        costs += d.candidates[0].cost_usd if d.candidates else 0.0
        mix[d.model_id] = mix.get(d.model_id, 0) + 1
        expected = meta.get("expected_model")
        if expected and d.model_id != expected:
            misroutes.append(f"line {lineno}: expected {expected}, got {d.model_id}")
        if args.verbose:
            print(f"line {lineno}: {d.model_id}@{d.backend_id} q={q:.2f}")

    p50 = statistics.median(latencies) if latencies else float("nan")
    avg_q = statistics.mean(qualities) if qualities else 0.0

    report = {
        "cases": len(cases),
        "routed": len(qualities),
        "unroutable": unroutable,
        "routing_mix": mix,
        "avg_quality": round(avg_q, 4),
        "est_cost_usd": round(costs, 6),
        "decision_p50_ms": round(p50, 3) if p50 == p50 else None,
        "misroutes": misroutes,
    }
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print(f"cases: {len(cases)}  routed: {len(qualities)}  "
              f"unroutable: {len(unroutable)}")
        print(f"routing mix: {json.dumps(mix, sort_keys=True)}")
        print(f"avg quality: {avg_q:.3f}  est cost: ${costs:.5f}  "
              f"decision p50: {p50:.2f}ms")
        if misroutes:
            print(f"misroutes ({len(misroutes)}):")
            for m in misroutes[:10]:
                print(f"  {m}")

    if args.ci:
        failures: list[str] = []
        if avg_q < args.quality_floor:
            failures.append(
                f"avg quality {avg_q:.3f} < floor {args.quality_floor:.3f}"
            )
        if unroutable:
            failures.append(f"{len(unroutable)} case(s) unroutable")
        if misroutes and args.fail_on_misroute:
            failures.append(f"{len(misroutes)} misroute(s) vs expected_model")
        if latencies and p50 > DECISION_LATENCY_ALARM_MS:
            failures.append(
                f"decision p50 {p50:.2f}ms > {DECISION_LATENCY_ALARM_MS:.0f}ms"
            )
        if args.json and failures:
            print(json.dumps({"gate": "failed", "failures": failures},
                             indent=2))
        elif args.json:
            print(json.dumps({"gate": "passed"}, indent=2))
        elif failures:
            print("CI gate FAILED: " + "; ".join(failures))
            return 1
        else:
            print("CI gate passed "
                  f"(floor {args.quality_floor:.2f}, "
                  f"latency <{DECISION_LATENCY_ALARM_MS:.0f}ms)")
        if failures:
            return 1
    return 0


def _measure_decide(engine: PolicyEngine, req: RoutingRequest) -> float:
    import time
    t0 = time.perf_counter()
    engine.decide(req)
    return (time.perf_counter() - t0) * 1000.0


# --------------------------------------------------------------------------- #
# train — the feedback loop's offline half
# --------------------------------------------------------------------------- #
def cmd_train(args: argparse.Namespace) -> int:
    samples = load_feedback(args.data)
    if not samples:
        print("no feedback samples found in", args.data, file=sys.stderr)
        return 2
    model, reports = train_heads(
        samples, epochs=args.epochs, lr=args.lr, l2=args.l2,
        holdout=args.holdout, min_samples=args.min_samples, seed=args.seed,
        shrinkage_k=args.shrinkage_k,
    )
    if not reports:
        print(f"no model reached min_samples={args.min_samples} — the "
              "checkpoint is empty and every decision keeps its prior",
              file=sys.stderr)
    if not args.quiet:
        print(f"samples: {len(samples)}  heads trained: {len(reports)}\n")
        print(f"{'model':>24} {'n':>5} {'mean y':>7} "
              f"{'train acc':>9} {'eval acc':>9}")
        for r in sorted(reports, key=lambda r: r.model_id):
            ev = f"{r.eval_accuracy:.3f}" if r.eval_accuracy is not None else "  n/a"
            print(f"{r.model_id:>24} {r.samples:>5} {r.mean_outcome:>7.3f} "
                  f"{r.train_accuracy:>9.3f} {ev:>9}")
        print(f"\nmax shrinkage lambda (trust in the learned delta at the "
              f"best-sampled head): "
              f"{max((h.samples / (h.samples + model.shrinkage_k) for h in model.heads.values()), default=0.0):.2f}")
    model.save(args.out)
    print(f"wrote {len(model.heads)} quality head(s) -> {args.out}")
    print('use it: Router.with_defaults(quality_model=QualityModel.load('
          f'"{args.out}"))')
    return 0


# --------------------------------------------------------------------------- #
def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m llmrouter",
        description="Cache-aware LLM routing — decision-layer CLI "
                    "(no network, no keys)",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("explain", help="route one prompt and explain it")
    p.add_argument("--prompt", required=True)
    p.add_argument("--system", default=None)
    p.add_argument("--workload", default=None,
                   choices=[w.value for w in WorkloadClass])
    p.add_argument("--session-id", default=None)
    p.add_argument("--tools", default=None,
                   help="path to a JSON array of tool schemas")
    p.add_argument("--quality-model", default=None,
                   help="quality checkpoint from `llmrouter train`")
    p.set_defaults(func=cmd_explain)

    p = sub.add_parser("calibrate",
                       help="pick the quality floor hitting a target "
                            "strong-model share")
    p.add_argument("--data", required=True,
                   help="JSONL case file ('-' for stdin)")
    p.add_argument("--target-strong-pct", type=float, default=0.5)
    p.add_argument("--workload", default="chat",
                   help="default workload for cases without one")
    p.add_argument("--quiet", action="store_true")
    p.set_defaults(func=cmd_calibrate)

    p = sub.add_parser("eval",
                       help="run labeled cases; --ci gates on the quality floor")
    p.add_argument("--data", required=True)
    p.add_argument("--workload", default="chat")
    p.add_argument("--min-quality", type=float, default=0.0,
                   help="the operating point under test")
    p.add_argument("--quality-floor", type=float, default=0.75)
    p.add_argument("--ci", action="store_true",
                   help="exit 1 when quality/latency gates fail")
    p.add_argument("--fail-on-misroute", action="store_true",
                   help="also fail when expected_model disagrees")
    p.add_argument("--json", action="store_true",
                   help="machine-readable report (CI artifact)")
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--quality-model", default=None,
                   help="score with trained quality heads from a checkpoint")
    p.set_defaults(func=cmd_eval)

    p = sub.add_parser("train",
                       help="fit per-model logistic quality heads from "
                            "feedback JSONL and write a checkpoint")
    p.add_argument("--data", required=True,
                   help="feedback JSONL ('-' for stdin); rows are pointwise "
                        "(model+outcome) or pairwise (model_a+model_b+winner)")
    p.add_argument("--out", default="quality.json")
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--lr", type=float, default=0.25)
    p.add_argument("--l2", type=float, default=1e-4)
    p.add_argument("--holdout", type=float, default=0.2)
    p.add_argument("--min-samples", type=int, default=8,
                   help="models below this sample count keep the prior")
    p.add_argument("--shrinkage-k", type=float, default=20.0)
    p.add_argument("--seed", type=int, default=13)
    p.add_argument("--quiet", action="store_true")
    p.set_defaults(func=cmd_train)

    p = sub.add_parser("serve",
                       help="run the decision engine as an HTTP decide-sidecar "
                            "(POST /decide, GET /stats, GET /healthz)")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8787)
    p.add_argument("--token", default=None,
                   help="require Authorization: Bearer <token>")
    p.add_argument("--quality-model", default=None,
                   help="score with trained quality heads from a checkpoint")
    p.add_argument("--state", default=None,
                   help="JSON file for durable session/health state")
    p.add_argument("--verbose", action="store_true")
    p.set_defaults(func=_cmd_serve)

    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


def _cmd_serve(args: argparse.Namespace) -> int:
    from .serve import cmd_serve
    return cmd_serve(args)


__all__ = ["main", "pure_engine", "load_cases", "build_request"]
