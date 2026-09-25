/**
 * Live routing demo — runs the REAL llmrouter decision engine (the same
 * TypeScript modules published to npm) in your browser. No network: the
 * engine is pure decision logic, so the demo works offline.
 */
import { Message, RoutingRequest, WorkloadClass } from "./assets/engine/types.js";
import { PriceRegistry } from "./assets/engine/pricing.js";
import { BackendRegistry, BackendSpec, DeploymentClass } from "./assets/engine/backends.js";
import { PolicyEngine } from "./assets/engine/policy.js";

const prices = new PriceRegistry();
const ids = prices.ids();
const engine = new PolicyEngine(
  prices,
  new BackendRegistry([
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
  ]),
);

const $ = (id) => document.getElementById(id);

const DEFAULT_SYSTEM =
  "You are a senior support engineer for a payments platform. " +
  "Answer with the relevant policy, the exact next steps, and the API calls " +
  "required. Cite the policy id for every claim. Escalate disputes over $500 " +
  "to a human reviewer. Be concise: no more than 150 words unless the caller " +
  "asks for a full walkthrough. ";

function route() {
  const prompt = $("demo-prompt").value.trim() || "hello";
  const system = $("demo-system").value;
  const session = $("demo-session").value.trim() || null;
  const workloadSel = $("demo-workload").value;
  const gap = parseFloat($("demo-gap").value);

  const messages = [];
  if (system.trim()) messages.push(new Message("system", system));
  messages.push(new Message("user", prompt));

  const req = new RoutingRequest(
    messages,
    workloadSel === "auto" ? null : WorkloadClass[workloadSel],
    session,
    null, [], [], new Set(), {},
    null,
    Number.isFinite(gap) ? gap : null,
  );

  const t0 = performance.now();
  let decision;
  try {
    decision = engine.decide(req);
  } catch (err) {
    $("demo-out").textContent = `unroutable: ${err.message}`;
    return;
  }
  const decideMs = performance.now() - t0;
  render(decision, decideMs);
}

function render(d, decideMs) {
  const chosen = d.candidates.find(
    (c) => c.modelId === d.modelId && c.backendId === d.backendId,
  );

  const rows = d.candidates
    .map((c) => {
      const pick = c === chosen;
      return `<tr class="${pick ? "picked" : ""}">
        <td>${c.modelId}<span class="dim">@${c.backendId}</span></td>
        <td class="num">${fmt(c.score, 4, true)}</td>
        <td class="num">${c.quality.toFixed(2)}</td>
        <td class="num">$${c.costUsd.toFixed(5)}</td>
        <td class="num">${c.latencyMs.toFixed(0)}ms</td>
        <td class="num">${c.affinity.toFixed(2)}</td>
        <td class="num">$${c.switchCostUsd.toFixed(5)}</td>
      </tr>`;
    })
    .join("");

  const ttl = [...d.ttlBySegment.entries()]
    .map(([k, v]) => `${k}=${v}`)
    .join(", ") || "n/a";

  const reasons = d.reasons.map((r) => `<li>${esc(r)}</li>`).join("");

  $("demo-out").innerHTML = `
    <div class="demo-chosen">${esc(d.modelId)}<span class="dim">@${esc(d.backendId)}</span></div>
    <div class="demo-meta">
      <span>workload <strong>${esc(d.workload)}</strong> <span class="dim">via ${esc(d.workloadSource)}, conf ${d.workloadConfidence.toFixed(2)}</span></span>
      <span>session <strong>${d.reusedSession ? "reused" : "new"}</strong>${d.switched ? ", switched" : ""}${d.stranded ? ", stranded" : ""}</span>
      <span>ttl <strong>${esc(ttl)}</strong></span>
      <span>decided in <strong>${decideMs.toFixed(2)}ms</strong></span>
    </div>
    <div class="table-wrap demo-table"><table>
      <thead><tr><th>candidate</th><th>score</th><th>quality</th><th>est $/turn</th><th>latency</th><th>affinity</th><th>switch</th></tr></thead>
      <tbody>${rows}</tbody>
    </table></div>
    <ul class="demo-reasons">${reasons}</ul>
    <p class="demo-hint">Route again with the same session id to see the session reuse and the
    affinity bonus appear; raise the idle gap past 300s to watch the 1-hour TTL kick in.</p>
  `;
}

const fmt = (n, digits, signed) =>
  (signed && n >= 0 ? "+" : "") + n.toFixed(digits);

const esc = (s) =>
  String(s).replace(/[&<>"]/g, (c) =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));

$("demo-route").addEventListener("click", route);
route();
