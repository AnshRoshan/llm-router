/**
 * The numbers ARE the spec — these assertions mirror the Python suite and the
 * verified research scripts. If one of these breaks, the two runtimes disagree.
 */
import { test } from "node:test";
import assert from "node:assert/strict";
import {
  EXPIRY_THRESHOLD_E_STAR,
  KEEPALIVE_BREAK_EVEN_MIN,
  assertTtlOrdering,
  breakEvenTurns,
  chooseTtlBySegment,
  emitBreakpoints,
  evaluateSwitch,
  keepaliveBeatsUpgrade,
  prefixCost,
  sessionCostWithSwitch,
} from "../dist/economics.js";
import {
  BackendRegistry,
  BackendSpec,
  HALF_OPEN_PROBE_LIMIT,
} from "../dist/backends.js";
import {
  Message,
  RoutingRequest,
  SegmentKind,
  WorkloadClass,
  WorkloadSource,
  DeploymentClass,
} from "../dist/types.js";
import {
  PriceCard,
  PriceRegistry,
} from "../dist/pricing.js";
import { PolicyEngine, DEFAULT_CONFIG } from "../dist/policy.js";
import { SessionStore, SessionPolicy, affinityPrior } from "../dist/affinity.js";

// Bundled cards: sonnet-class $3/M, haiku-class $1/M, both Anthropic cache.
function cards() {
  const prices = new PriceRegistry();
  return {
    cur: prices.require("claude-sonnet-class"),
    tgt: prices.require("claude-haiku-class"),
  };
}

test("R*(sonnet-class -> haiku-class) = 4.75 turns", () => {
  const { cur, tgt } = cards();
  assert.ok(Math.abs(breakEvenTurns(cur, tgt) - 4.75) < 1e-9);
});

test("stay-30-turns on sonnet-class = $0.6225", () => {
  const { cur } = cards();
  assert.ok(Math.abs(prefixCost(cur, 50_000, 30) - 0.6225) < 1e-9);
});

test("switch at 5 of 30 turns = $0.4300 (the corrected figure)", () => {
  const { cur, tgt } = cards();
  const c = sessionCostWithSwitch(cur, tgt, 50_000, 30, 5);
  assert.ok(Math.abs(c - 0.4300) < 1e-9);
});

test("no-oscillation lock forbids a second migration", () => {
  const { cur, tgt } = cards();
  const v = evaluateSwitch(cur, tgt, 50_000, 25, true);
  assert.equal(v.shouldSwitch, false);
  assert.ok(v.reason.startsWith("no-oscillation"));
});

test("E* = 0.652 and keepalive break-even = 37.5 min", () => {
  assert.ok(Math.abs(EXPIRY_THRESHOLD_E_STAR - 0.75 / 1.15) < 1e-12);
  assert.ok(Math.abs(KEEPALIVE_BREAK_EVEN_MIN - 37.5) < 1e-12);
  assert.ok(keepaliveBeatsUpgrade(30));
  assert.ok(!keepaliveBeatsUpgrade(45));
});

test("breakpoint ordering: longer TTL first, max 4", () => {
  const sem = { canCache: (t) => t >= 1024, maxBreakpoints: 4 };
  const segs = new Map([
    [SegmentKind.History, 8000],
    [SegmentKind.System, 4000],
    [SegmentKind.Tools, 2000],
  ]);
  const ttl = new Map([
    [SegmentKind.System, "1h"],
    [SegmentKind.Tools, "1h"],
    [SegmentKind.History, "5m"],
  ]);
  const bps = emitBreakpoints(sem, segs, ttl);
  assertTtlOrdering(bps);
  assert.equal(bps[0].ttl, "1h");
  assert.equal(bps[bps.length - 1].ttl, "5m");
});

test("mixed TTL: system at 1h, history at 5m, on anthropic semantics", () => {
  const prices = new PriceRegistry();
  const sem = prices.require("claude-sonnet-class").cache;
  const segs = new Map([
    [SegmentKind.System, 4000],
    [SegmentKind.History, 8000],
  ]);
  const ttl = chooseTtlBySegment(sem, segs, 900.0); // gap > 5 min
  assert.equal(ttl.get(SegmentKind.System), "1h");
  assert.equal(ttl.get(SegmentKind.History), "5m");
});

// --------------------------------------------------------------------------- //
// Decision layer
// --------------------------------------------------------------------------- //
function makeEngine(prices = new PriceRegistry()) {
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
  return new PolicyEngine(prices, backends);
}

test("decides without network and explains itself", () => {
  const d = makeEngine().decide(new RoutingRequest([new Message("user", "hi")], WorkloadClass.Chat));
  assert.ok(d.modelId.length > 0);
  assert.ok(d.explain().includes("model="));
});

test("min-quality floor excludes weak models", () => {
  const base = makeEngine();
  const engine = new PolicyEngine(
    base.prices, base.backends, base.resolver,
    { ...DEFAULT_CONFIG, weights: { ...DEFAULT_CONFIG.weights, minQuality: 0.85 } },
  );
  const d = engine.decide(new RoutingRequest([new Message("user", "hi")], WorkloadClass.Chat));
  for (const c of d.candidates) assert.ok(c.quality >= 0.85);
});

test("empty registry means empty, not defaults (falsy-trap)", () => {
  assert.equal(new PriceRegistry(new Map()).size, 0);
  assert.equal(new PriceRegistry().size, 6);
});

test("Rule F: tool-schema drift scores everything cold", () => {
  const engine = makeEngine();
  const store = engine.sessions;
  const mk = (tools) =>
    new RoutingRequest(
      [new Message("system", "You are a coding agent. ".repeat(120)), new Message("user", "turn")],
      WorkloadClass.Agent, "s1", null, tools,
    );
  store.put(new SessionPolicy("sid:s1", WorkloadClass.Agent, WorkloadSource.ExplicitTag,
    1.0, "claude-sonnet-class", "anthropic", 0));
  const d1 = engine.decide(mk([{ name: "grep" }]));
  assert.ok(!d1.reasons.some((r) => r.includes("Rule F")));
  const d2 = engine.decide(mk([{ name: "grep" }, { name: "edit" }]));
  assert.ok(d2.reasons.some((r) => r.includes("Rule F")));
  for (const c of d2.candidates) assert.equal(c.affinity, 0.0);
});

test("half-open circuit admits exactly one canary", () => {
  const reg = new BackendRegistry([
    new BackendSpec("api", { models: ["m"] }),
  ]);
  const h = reg.health("api");
  for (let i = 0; i < 5; i++) h.recordFailure("boom", null, 100.0);
  assert.equal(reg.availableFor("m", 100.5).length, 0);
  const now = 100.0 + 1.0; // backoff elapsed: HALF_OPEN
  assert.ok(h.isHalfOpen(now));
  assert.equal(reg.availableFor("m", now).length, 1);
  h.halfOpenProbes = HALF_OPEN_PROBE_LIMIT;
  assert.equal(reg.availableFor("m", now).length, 0);
  h.recordSuccess(50.0, now);
  assert.equal(reg.availableFor("m", now).length, 1);
});

test("session stats report hit rate by workload source", () => {
  const store = new SessionStore();
  const s = new SessionPolicy("sid:a", WorkloadClass.Agent,
    WorkloadSource.Fingerprint, 1.0, "claude-sonnet-class", "anthropic", 0);
  s.recordUsage(900, 100);
  store.put(s);
  const stats = store.stats();
  assert.ok(Math.abs(stats.hitRateBySource.fingerprint - 0.9) < 1e-9);
});

test("affinity prior decays inside the TTL window", () => {
  assert.ok(affinityPrior(WorkloadClass.Agent, 5) > affinityPrior(WorkloadClass.Chat, 90));
});

test("budget cap picks the cheapest violator when nothing fits", () => {
  const engine = makeEngine();
  const d = engine.decide(new RoutingRequest(
    [new Message("user", "hi")], WorkloadClass.Chat, null, null, [], [],
    new Set(), { maxCostUsd: 0.0001 },
  ));
  const cheapest = [...d.candidates].sort((a, b) => a.costUsd - b.costUsd)[0];
  assert.equal(d.modelId, cheapest.modelId);
  assert.ok(d.reasons.some((r) => r.includes("budget cap")));
});

test("context-window pre-call check routes to a model that fits", () => {
  const prices = new PriceRegistry(new Map());
  prices.register(new PriceCard("small-window", 1.0, 2.0, undefined, 8192, true, false,
    undefined, new Set(), { chat: 0.9 }));
  prices.register(new PriceCard("large-window", 5.0, 10.0, undefined, 200_000, true, false,
    undefined, new Set(), { chat: 0.85 }));
  const engine = makeEngine(prices);
  const d = engine.decide(new RoutingRequest([new Message("user", "x".repeat(20_000))], WorkloadClass.Chat));
  assert.equal(d.modelId, "large-window");
});

// --------------------------------------------------------------------------- //
// sha256 — known-answer vectors (the hash must be right in EVERY runtime)
// --------------------------------------------------------------------------- //
import { sha256, hash16, hash8Big } from "../dist/hash.js";

test("sha256 matches FIPS known-answer vectors", () => {
  const hex = (s) => Buffer.from(sha256(new TextEncoder().encode(s))).toString("hex");
  assert.equal(hex("abc"),
    "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
  assert.equal(hex(""),
    "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855");
  // > 55 bytes: exercises multi-block padding
  assert.equal(hex("a".repeat(100)),
    "2816597888e4a0d3a36b82b83316ab32680eb8f00f8cd3b904d681246d285a0e");
});

test("hash16 returns 32 hex chars, hash8Big a bigint", () => {
  assert.equal(hash16("x").length, 32);
  assert.equal(typeof hash8Big("x"), "bigint");
});

test("recordDecision builds session state for decision-only engines", () => {
  const engine = makeEngine();
  const mk = (text) => new RoutingRequest(
    [new Message("system", "You are a support bot. ".repeat(120)), new Message("user", text)],
    WorkloadClass.Chat, "s9",
  );
  const d1 = engine.decide(mk("turn one"));
  engine.recordDecision(mk("turn one"), d1);
  const d2 = engine.decide(mk("turn two"));
  assert.equal(d2.reusedSession, true);
  const warm = d2.candidates.find(
    (c) => c.modelId === d2.modelId && c.backendId === d2.backendId,
  );
  assert.ok(warm.affinity > 0, "pinned pair should carry an affinity bonus");
});

test("Constraints.allowedModels restricts the candidate pool", () => {
  const engine = makeEngine();
  const d = engine.decide(new RoutingRequest(
    [new Message("user", "hi")], WorkloadClass.Chat, null, null, [], [],
    new Set(), { allowedModels: new Set(["claude-haiku-class", "claude-sonnet-class"]) },
  ));
  assert.ok(["claude-haiku-class", "claude-sonnet-class"].includes(d.modelId));
  for (const c of d.candidates) assert.ok(c.modelId.startsWith("claude"));
});
