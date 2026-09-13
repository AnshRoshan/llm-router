/**
 * CLI: `npx llmrouter explain|eval|calibrate` — the pure decision layer,
 * no keys, no network. Mirrors `python -m llmrouter`.
 */
import { readFileSync } from "node:fs";
import { PolicyEngine, type RouterConfig, type Weights } from "./policy.js";
import { BackendRegistry, BackendSpec } from "./backends.js";
import { PriceRegistry } from "./pricing.js";
import { WorkloadResolver } from "./signals.js";
import {
  DeploymentClass,
  Message,
  RoutingRequest,
  WorkloadClass,
} from "./types.js";

const DECISION_LATENCY_ALARM_MS = 10.0;

export function pureEngine(config?: RouterConfig): PolicyEngine {
  const prices = new PriceRegistry();
  const ids = prices.ids();
  const backends = new BackendRegistry([
    new BackendSpec("anthropic", {
      deployment: DeploymentClass.Api,
      models: ids.filter((m) => m.startsWith("claude")),
      priority: 0,
    }),
    new BackendSpec("openai", {
      deployment: DeploymentClass.Api,
      models: ids.filter((m) => m.startsWith("gpt")),
      priority: 1,
    }),
  ]);
  return new PolicyEngine(prices, backends, new WorkloadResolver(), config);
}

interface CaseMeta {
  messages?: { role: string; content: string }[];
  text?: string;
  prompt?: string;
  workload?: string;
  sessionId?: string;
  expectedModel?: string;
}

function loadCases(path: string): { lineno: number; meta: CaseMeta; messages: Message[] }[] {
  const raw = readFileSync(path, "utf8");
  const cases: { lineno: number; meta: CaseMeta; messages: Message[] }[] = [];
  for (const [i, line] of raw.split("\n").entries()) {
    const trimmed = line.trim();
    if (!trimmed || trimmed.startsWith("#")) continue;
    const obj = JSON.parse(trimmed) as CaseMeta;
    let messages: Message[];
    if (obj.messages) {
      messages = obj.messages.map((m) => new Message(m.role, m.content));
    } else {
      const text = obj.text ?? obj.prompt;
      if (text === undefined) {
        throw new Error(`line ${i + 1}: case needs 'messages' or 'text'`);
      }
      messages = [new Message("user", text)];
    }
    cases.push({ lineno: i + 1, meta: obj, messages });
  }
  return cases;
}

function caseRequest(meta: CaseMeta, messages: Message[], defaultWorkload: string): RoutingRequest {
  return new RoutingRequest(
    messages,
    WorkloadClass[capitalize(meta.workload ?? defaultWorkload) as keyof typeof WorkloadClass] ?? WorkloadClass.Chat,
    meta.sessionId ?? null,
  );
}

function capitalize(s: string): string {
  return s.length === 0 ? s : s[0].toUpperCase() + s.slice(1);
}

function strongModels(engine: PolicyEngine, workload: WorkloadClass): Set<string> {
  const byModel = new Map<string, number>();
  for (const c of engine.prices.all()) byModel.set(c.modelId, c.quality(workload));
  let top = -Infinity;
  for (const q of byModel.values()) if (q > top) top = q;
  return new Set([...byModel.entries()].filter(([, q]) => q === top).map(([m]) => m));
}

function cmdExplain(args: Map<string, string>): number {
  const messages: Message[] = [];
  if (args.get("system")) messages.push(new Message("system", args.get("system")!));
  messages.push(new Message("user", args.get("prompt") ?? ""));
  const workload = args.get("workload");
  const req = new RoutingRequest(
    messages,
    workload ? WorkloadClass[capitalize(workload) as keyof typeof WorkloadClass] : null,
    args.get("session-id") ?? null,
  );
  console.log(pureEngine().decide(req).explain());
  return 0;
}

function cmdEval(args: Map<string, string>, flags: Set<string>): number {
  const minQuality = Number(args.get("min-quality") ?? 0);
  const base = pureEngine().config;
  const engine = pureEngine({
    ...base,
    weights: { ...base.weights, minQuality } as Weights,
  });
  const cases = loadCases(args.get("data")!);
  const qualityFloor = Number(args.get("quality-floor") ?? 0.75);
  const ci = flags.has("ci");
  const failOnMisroute = flags.has("fail-on-misroute");
  const json = flags.has("json");

  const mix: Record<string, number> = {};
  const qualities: number[] = [];
  let costs = 0;
  const latencies: number[] = [];
  const misroutes: string[] = [];
  const unroutable: number[] = [];

  for (const { lineno, meta, messages } of cases) {
    const req = caseRequest(meta, messages, args.get("workload") ?? "chat");
    let decision;
    try {
      decision = engine.decide(req);
    } catch {
      unroutable.push(lineno);
      continue;
    }
    const t0 = process.hrtime.bigint();
    engine.decide(req);
    latencies.push(Number(process.hrtime.bigint() - t0) / 1e6);
    const card = engine.prices.require(decision.modelId);
    const q = card.quality(decision.workload);
    qualities.push(q);
    if (decision.candidates.length > 0) costs += decision.candidates[0].costUsd;
    mix[decision.modelId] = (mix[decision.modelId] ?? 0) + 1;
    if (meta.expectedModel && decision.modelId !== meta.expectedModel) {
      misroutes.push(`line ${lineno}: expected ${meta.expectedModel}, got ${decision.modelId}`);
    }
  }

  const p50 = latencies.length > 0 ? median(latencies) : NaN;
  const avgQ = qualities.length > 0 ? qualities.reduce((a, b) => a + b, 0) / qualities.length : 0;

  const failures: string[] = [];
  if (avgQ < qualityFloor) failures.push(`avg quality ${avgQ.toFixed(3)} < floor ${qualityFloor}`);
  if (unroutable.length > 0) failures.push(`${unroutable.length} case(s) unroutable`);
  if (misroutes.length > 0 && failOnMisroute) failures.push(`${misroutes.length} misroute(s)`);
  if (latencies.length > 0 && p50 > DECISION_LATENCY_ALARM_MS) {
    failures.push(`decision p50 ${p50.toFixed(2)}ms > ${DECISION_LATENCY_ALARM_MS}ms`);
  }

  if (json) {
    console.log(JSON.stringify({
      cases: cases.length, routed: qualities.length, unroutable,
      routingMix: mix, avgQuality: round(avgQ, 4), estCostUsd: round(costs, 6),
      decisionP50Ms: Number.isNaN(p50) ? null : round(p50, 3),
      misroutes, gate: ci ? (failures.length > 0 ? "failed" : "passed") : null,
      failures,
    }, null, 2));
  } else {
    console.log(
      `cases: ${cases.length}  routed: ${qualities.length}  unroutable: ${unroutable.length}`,
    );
    console.log(`routing mix: ${JSON.stringify(mix)}`);
    console.log(`avg quality: ${avgQ.toFixed(3)}  est cost: $${costs.toFixed(5)}  decision p50: ${p50.toFixed(2)}ms`);
    for (const m of misroutes.slice(0, 10)) console.log(`  ${m}`);
    if (ci) {
      console.log(
        failures.length > 0
          ? `CI gate FAILED: ${failures.join("; ")}`
          : "CI gate passed",
      );
    }
  }
  if (ci && failures.length > 0) return 1;
  return 0;
}

function cmdCalibrate(args: Map<string, string>): number {
  const cases = loadCases(args.get("data")!);
  const target = Number(args.get("target-strong-pct") ?? 0.5);
  const base = pureEngine().config;
  const rows: { threshold: number; share: number; unroutable: number }[] = [];

  for (let i = 0; i < 20; i++) {
    const threshold = Math.round(0.05 * i * 100) / 100;
    const engine = pureEngine({
      ...base,
      weights: { ...base.weights, minQuality: threshold } as Weights,
    });
    let strongHits = 0;
    let unroutable = 0;
    for (const { meta, messages } of cases) {
      const req = caseRequest(meta, messages, args.get("workload") ?? "chat");
      try {
        const d = engine.decide(req);
        if (strongModels(engine, d.workload).has(d.modelId)) strongHits += 1;
      } catch {
        unroutable += 1;
      }
    }
    rows.push({ threshold, share: strongHits / cases.length, unroutable });
  }
  const valid = rows.filter((r) => r.unroutable < cases.length);
  if (valid.length === 0) {
    console.error("every threshold unrouted every case");
    return 2;
  }
  const best = valid.reduce((a, b) =>
    Math.abs(a.share - target) < Math.abs(b.share - target) ? a : b);
  console.log(`calibrated min_quality = ${best.threshold.toFixed(2)} ` +
    `(target ${target.toFixed(2)}, achieved ${best.share.toFixed(3)} on ${cases.length} cases)`);
  return 0;
}

function median(xs: number[]): number {
  const s = [...xs].sort((a, b) => a - b);
  const mid = Math.floor(s.length / 2);
  return s.length % 2 ? s[mid] : (s[mid - 1] + s[mid]) / 2;
}

function round(x: number, digits: number): number {
  const f = 10 ** digits;
  return Math.round(x * f) / f;
}

export function runCli(argv: string[]): number {
  const [command, ...rest] = argv;
  const args = new Map<string, string>();
  const flags = new Set<string>();
  for (let i = 0; i < rest.length; i++) {
    const a = rest[i];
    if (a.startsWith("--")) {
      if (i + 1 < rest.length && !rest[i + 1].startsWith("--")) {
        args.set(a.slice(2), rest[i + 1]);
        i++;
      } else {
        flags.add(a.slice(2));
      }
    }
  }
  try {
    switch (command) {
      case "explain":
        if (!args.has("prompt")) {
          console.error("usage: llmrouter explain --prompt \"...\" [--system S] [--workload W]");
          return 2;
        }
        return cmdExplain(args);
      case "eval":
        if (!args.has("data")) {
          console.error("usage: llmrouter eval --data cases.jsonl [--ci] [--json] [--quality-floor 0.75]");
          return 2;
        }
        return cmdEval(args, flags);
      case "calibrate":
        if (!args.has("data")) {
          console.error("usage: llmrouter calibrate --data cases.jsonl [--target-strong-pct 0.5]");
          return 2;
        }
        return cmdCalibrate(args);
      default:
        console.error(
          "llmrouter — cache-aware LLM routing (decision engine)\n" +
            "usage: llmrouter <explain|eval|calibrate> [options]",
        );
        return command ? 2 : 0;
    }
  } catch (err) {
    console.error(`error: ${err instanceof Error ? err.message : err}`);
    return 2;
  }
}

// Direct execution: node dist/cli.js ...
if (process.argv[1] && process.argv[1].endsWith("cli.js")) {
  process.exit(runCli(process.argv.slice(2)));
}
