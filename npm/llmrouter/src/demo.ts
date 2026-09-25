/**
 * Browser demo surface: the REAL decision engine, bundled for the docs site.
 * No mocks — the site's interactive demo runs this exact code path.
 *
 * NOTE: imports the engine modules directly, NOT index.js — the barrel also
 * exports the CLI, which needs node:fs and must stay out of the browser bundle.
 */
import {
  Message,
  RoutingRequest,
  WorkloadClass,
  DeploymentClass,
} from "./types.js";
import { PolicyEngine } from "./policy.js";
import { PriceRegistry } from "./pricing.js";
import { BackendRegistry, BackendSpec } from "./backends.js";
import { WorkloadResolver } from "./signals.js";
import { VERSION } from "./version.js";

export interface DemoRouteInput {
  prompt: string;
  system?: string;
  /** Workload tag, or null for automatic inference (Path B). */
  workload?: string | null;
  /** Tool schemas to include; their presence drives agent inference + TTL. */
  tools?: { name: string }[];
  retrieved?: string[];
  sessionId?: string | null;
  /** Expected idle gap between turns; drives the TTL policy. */
  expectedGapS?: number;
  /** Time spent generating, which the provider charges against the TTL. */
  expectedGenerationS?: number;
  /** Restrict the candidate pool: "anthropic" exposes the 1h TTL tier. */
  modelPool?: "all" | "anthropic" | "openai";
}

function workloadClass(name: string | null | undefined): WorkloadClass | null {
  if (!name) return null;
  const key = name.charAt(0).toUpperCase() + name.slice(1).toLowerCase();
  return WorkloadClass[key as keyof typeof WorkloadClass] ?? null;
}

export function createDemoEngine() {
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
  const engine = new PolicyEngine(prices, backends, new WorkloadResolver());
  const pool = (prefix: string) =>
    new Set(ids.filter((m) => m.startsWith(prefix)));

  return {
    route(input: DemoRouteInput) {
      const messages: Message[] = [];
      if (input.system) messages.push(new Message("system", input.system));
      messages.push(new Message("user", input.prompt));
      const constraints: { allowedModels?: ReadonlySet<string> } = {};
      if (input.modelPool === "anthropic") constraints.allowedModels = pool("claude");
      if (input.modelPool === "openai") constraints.allowedModels = pool("gpt");
      const req = new RoutingRequest(
        messages,
        workloadClass(input.workload),
        input.sessionId ?? null,
        null,
        input.tools ?? [],
        input.retrieved ?? [],
        new Set<string>(),
        constraints,
        null,
        input.expectedGapS ?? null,
        input.expectedGenerationS ?? null,
      );
      const d = engine.decide(req);
      // The execution layer's learning step, so turn 2 in the demo sees the
      // session it pinned on turn 1 (warm cache, sticky backend).
      engine.recordDecision(req, d);
      const card = prices.require(d.modelId);
      return {
        modelId: d.modelId,
        backendId: d.backendId,
        workload: d.workload,
        source: d.workloadSource,
        confidence: d.workloadConfidence,
        reusedSession: d.reusedSession,
        switched: d.switched,
        stranded: d.stranded,
        quality: card.quality(d.workload),
        ttl: Object.fromEntries(
          [...d.ttlBySegment].map(([k, v]) => [String(k), v]),
        ),
        candidates: d.candidates.map((c) => ({
          modelId: c.modelId,
          backendId: c.backendId,
          score: c.score,
          quality: c.quality,
          costUsd: c.costUsd,
          affinity: c.affinity,
          switchCostUsd: c.switchCostUsd,
          warm: c.reasons.some((r) => r === "warm=true"),
        })),
        reasons: [...d.reasons],
        explain: d.explain(),
      };
    },
    newSession() {
      engine.sessions.clear();
    },
  };
}

export const DEMO_VERSION = VERSION;
