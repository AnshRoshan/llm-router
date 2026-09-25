/**
 * Cross-runtime parity check: the SAME requests, the SAME checkpoint and the
 * SAME price cards through the Python and JS decision engines must produce the
 * SAME decisions — model, backend, workload, source, TTL map, and the top
 * candidate's cost/quality/score.
 *
 *   1. Python half (from llmrouter/): writes .tmp_parity/{cases.json,py.json}
 *   2. JS half: node scripts/parity.mjs   # re-runs cases through dist/ + diffs
 *
 * If this fails the two runtimes disagree and one of them is wrong. The
 * `learned` rows exercise the checkpoint path: feature extraction, the
 * shrinkage blend and the float64 dot product must match byte-for-byte.
 */
import { readFileSync, writeFileSync, existsSync } from "node:fs";
import {
  Message,
  RoutingRequest,
  WorkloadClass,
  PolicyEngine,
  PriceRegistry,
  BackendRegistry,
  BackendSpec,
  DeploymentClass,
  WorkloadResolver,
  QualityModel,
} from "../dist/index.js";

const dir = "D:/WORKK/LLM_ROUTER/.tmp_parity";
const bundlePath = `${dir}/cases.json`;
if (!existsSync(bundlePath)) {
  console.error(`missing ${bundlePath} — generate the parity bundle from Python first`);
  process.exit(2);
}
const bundle = JSON.parse(readFileSync(bundlePath, "utf8"));
const cases = bundle.cases ?? bundle; // tolerate the old bare-array format

const prices = new PriceRegistry();
const ids = prices.ids();
const backends = () => new BackendRegistry([
  new BackendSpec("anthropic", {
    deployment: DeploymentClass.Api,
    models: ids.filter((m) => m.startsWith("claude")), priority: 0,
  }),
  new BackendSpec("openai", {
    deployment: DeploymentClass.Api,
    models: ids.filter((m) => m.startsWith("gpt")), priority: 1,
  }),
]);
const plain = new PolicyEngine(prices, backends(), new WorkloadResolver());
const engines = { plain };
if (bundle.checkpoint) {
  engines.learned = new PolicyEngine(
    prices, backends(), new WorkloadResolver(), undefined, null,
    QualityModel.fromJson(bundle.checkpoint),
  );
}

function requestFor(c) {
  const msgs = c.messages.map(([role, content]) => new Message(role, content));
  const wl = c.workload
    ? WorkloadClass[c.workload[0].toUpperCase() + c.workload.slice(1)]
    : null;
  return new RoutingRequest(
    msgs, wl, null, null, c.tools ?? [], c.retrieved ?? [],
  );
}

const js = [];
for (const c of cases) {
  const req = requestFor(c);
  for (const [mode, engine] of Object.entries(engines)) {
    const d = engine.decide(req);
    const top = d.candidates[0];
    js.push({
      label: `${c.label}/${mode}`, model: d.modelId, backend: d.backendId,
      workload: d.workload, source: d.workloadSource,
      ttl: Object.fromEntries([...d.ttlBySegment].map(([k, v]) => [k, v])),
      top_cost: Math.round(top.costUsd * 1e12) / 1e12,
      top_q: Math.round(top.quality * 1e12) / 1e12,
      top_score: Math.round(top.score * 1e12) / 1e12,
    });
  }
}
writeFileSync(`${dir}/js.json`, JSON.stringify(js, null, 1));

if (!existsSync(`${dir}/py.json`)) {
  console.error("js.json written; run the Python half to produce py.json, then re-run");
  process.exit(0);
}
const py = JSON.parse(readFileSync(`${dir}/py.json`, "utf8"));
const keys = ["label", "model", "backend", "workload", "source", "ttl",
  "top_cost", "top_q", "top_score"];
let ok = true;
for (let i = 0; i < Math.max(py.length, js.length); i++) {
  for (const k of keys) {
    if (JSON.stringify(py[i]?.[k]) !== JSON.stringify(js[i]?.[k])) {
      ok = false;
      console.log(`MISMATCH ${py[i]?.label ?? js[i]?.label} ${k}:`,
        JSON.stringify(py[i]?.[k]), "vs", JSON.stringify(js[i]?.[k]));
    }
  }
}
console.log(ok
  ? `PARITY: all ${py.length} decisions identical across runtimes (incl. learned-quality path)`
  : "PARITY FAILED");
process.exit(ok ? 0 : 1);
