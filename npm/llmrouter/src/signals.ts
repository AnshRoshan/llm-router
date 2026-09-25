/**
 * Workload-class inference: the two entry paths.
 *   PATH A (declared) — tag or named agent profile. ~0 ms, skips inference.
 *   PATH B (inferred) — B1 structural rules → B2 prompt fingerprint →
 *                       B3 classifier hook (turn 1 only) → B4 default.
 */
import {
  RoutingRequest,
  WorkloadClass,
  WorkloadSource,
  type ToolSchema,
  type WorkloadSignal,
} from "./types.js";

/** Fallback turns-per-session priors. Assumptions to calibrate, not measurements. */
export const TURNS_PRIOR: Readonly<Record<WorkloadClass, number>> = {
  [WorkloadClass.Agent]: 25,
  [WorkloadClass.Chat]: 6,
  [WorkloadClass.Rag]: 2,
  [WorkloadClass.Batch]: 1,
  [WorkloadClass.Code]: 12,
  [WorkloadClass.Vision]: 3,
  [WorkloadClass.Unknown]: 5,
};

/** Expected idle gap between turns, seconds. Agents are seconds; humans are not. */
export const GAP_PRIOR: Readonly<Record<WorkloadClass, number>> = {
  [WorkloadClass.Agent]: 5.0,
  [WorkloadClass.Chat]: 90.0,
  [WorkloadClass.Rag]: 45.0,
  [WorkloadClass.Batch]: 0.0,
  [WorkloadClass.Code]: 20.0,
  [WorkloadClass.Vision]: 60.0,
  [WorkloadClass.Unknown]: 60.0,
};

export interface WorkloadClassifier {
  classify(request: RoutingRequest): WorkloadSignal | null;
}

export const CODE_MARKERS = [
  "```", "def ", "class ", "function ", "import ", "const ", "SELECT ",
] as const;

/** B1 — zero-cost inference from the shape of the request. */
export function inferStructural(req: RoutingRequest): WorkloadSignal | null {
  if (req.hasImages()) {
    return sig(WorkloadClass.Vision, WorkloadSource.Structural, 0.99,
      "image content block present");
  }
  if (req.tools.length > 0 && req.hasToolResults()) {
    return sig(WorkloadClass.Agent, WorkloadSource.Structural, 0.97,
      "tools present and prior tool_result in history");
  }
  if (req.tools.length > 0) {
    return sig(WorkloadClass.Agent, WorkloadSource.Structural, 0.85,
      "tools present, no tool_result yet (first step)");
  }
  if (req.retrieved.length > 0) {
    return sig(WorkloadClass.Rag, WorkloadSource.Structural, 0.90,
      "retrieved chunks supplied with the request");
  }
  if (req.tags.has("batch")) {
    return sig(WorkloadClass.Batch, WorkloadSource.Structural, 1.0, "tagged batch");
  }
  const userText = req.firstUserText();
  const codeHits = CODE_MARKERS.filter((m) => userText.includes(m)).length;
  if (codeHits >= 2 || req.tags.has("code")) {
    return sig(WorkloadClass.Code, WorkloadSource.Structural, 0.70,
      `${codeHits} code markers in the first user turn`);
  }
  if (req.turnIndex >= 2 && req.tools.length === 0) {
    return sig(WorkloadClass.Chat, WorkloadSource.Structural, 0.80,
      "multi-turn, no tools");
  }
  return null;
}

function sig(
  workload: WorkloadClass, source: WorkloadSource, confidence: number, reason: string,
): WorkloadSignal {
  return { workload, source, confidence, reason };
}

/**
 * B2 — prompt fingerprint registry. Production traffic comes from a handful of
 * distinct application prompts; this turns "classify every request" into
 * "classify every application".
 */
export class FingerprintRegistry {
  private readonly map = new Map<string, WorkloadSignal>();
  private readonly seenByClient = new Map<string, Set<string>>();

  get(fingerprint: string): WorkloadSignal | null {
    return this.map.get(fingerprint) ?? null;
  }

  put(fingerprint: string, signal: WorkloadSignal): void {
    const existing = this.map.get(fingerprint);
    if (existing && existing.confidence >= signal.confidence) return;
    this.map.set(fingerprint, signal);
  }

  observe(fingerprint: string, clientId: string | null): void {
    const key = clientId ?? "anon";
    const set = this.seenByClient.get(key) ?? new Set<string>();
    set.add(fingerprint);
    this.seenByClient.set(key, set);
  }

  /** Snapshot accessors for state persistence (see store.ts). */
  snapshotMap(): Array<[string, WorkloadSignal]> {
    return [...this.map.entries()];
  }

  hydrate(fingerprint: string, signal: WorkloadSignal): void {
    this.map.set(fingerprint, signal);
  }

  snapshotSeenByClient(): Record<string, string[]> {
    const out: Record<string, string[]> = {};
    for (const [k, v] of this.seenByClient) out[k] = [...v];
    return out;
  }

  cardinalityByClient(): Record<string, number> {
    const out: Record<string, number> = {};
    for (const [k, v] of this.seenByClient) out[k] = v.size;
    return out;
  }

  /** Clients generating too many distinct prompts to ever cache. */
  volatileClients(threshold = 50): string[] {
    const out: string[] = [];
    for (const [k, v] of this.seenByClient) {
      if (v.size > threshold) out.push(k);
    }
    return out;
  }

  get size(): number {
    return this.map.size;
  }
}

/** A named agent the caller can select — the "user chose an agent" path. */
export interface AgentProfile {
  name: string;
  workload: WorkloadClass;
  allowedModels?: readonly string[];
  sticky?: boolean;
  tags?: ReadonlySet<string>;
  expectedTurns?: number | null;
}

/** Resolves a workload class via Path A then Path B. */
export class WorkloadResolver {
  readonly profiles: Map<string, AgentProfile>;
  readonly fingerprints: FingerprintRegistry;

  constructor(
    profiles: ReadonlyMap<string, AgentProfile> | Record<string, AgentProfile> | null = null,
    public classifier: WorkloadClassifier | null = null,
    public defaultWorkload: WorkloadClass = WorkloadClass.Chat,
  ) {
    if (profiles instanceof Map) {
      this.profiles = new Map(profiles);
    } else if (profiles) {
      this.profiles = new Map(Object.entries(profiles));
    } else {
      this.profiles = new Map();
    }
    this.fingerprints = new FingerprintRegistry();
  }

  addProfile(profile: AgentProfile): void {
    this.profiles.set(profile.name, profile);
  }

  resolve(req: RoutingRequest): WorkloadSignal {
    // ---- PATH A: declared ---------------------------------------------
    if (req.agentProfile) {
      const profile = this.profiles.get(req.agentProfile);
      if (profile) {
        return {
          workload: profile.workload, source: WorkloadSource.AgentProfile,
          confidence: 1.0, reason: `agent profile '${profile.name}'`,
        };
      }
      return {
        workload: this.defaultWorkload, source: WorkloadSource.Default,
        confidence: 0.2, reason: `unknown agent profile '${req.agentProfile}'`,
      };
    }
    if (req.workload !== null) {
      return {
        workload: req.workload, source: WorkloadSource.ExplicitTag,
        confidence: 1.0, reason: "caller declared workload",
      };
    }

    // ---- PATH B: inferred ----------------------------------------------
    const fp = req.promptFingerprint();
    this.fingerprints.observe(fp, req.clientId);

    const hit = this.fingerprints.get(fp);
    if (hit) {
      return {
        workload: hit.workload, source: WorkloadSource.Fingerprint,
        confidence: 1.0, reason: "prompt fingerprint matched a known application",
      };
    }

    const structural = inferStructural(req);
    if (structural) {
      this.fingerprints.put(fp, structural);
      return structural;
    }

    if (this.classifier && req.isFirstTurn) {
      const got = this.classifier.classify(req);
      if (got) {
        this.fingerprints.put(fp, got);
        return got;
      }
    }

    const fallback: WorkloadSignal = {
      workload: this.defaultWorkload, source: WorkloadSource.Default,
      confidence: 0.3,
      reason: `no signal; using default ${this.defaultWorkload}`,
    };
    this.fingerprints.put(fp, fallback);
    return fallback;
  }
}
