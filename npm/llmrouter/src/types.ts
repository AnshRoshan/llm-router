/**
 * Core types — the vocabulary of the router.
 *
 * This is the TypeScript port of the Python decision layer
 * (`llmrouter/src/llmrouter`). Formulas, thresholds and defaults are kept
 * identical on purpose: the two packages must make the SAME decision for the
 * SAME request and price cards. Where a hash is involved (fingerprints, sticky
 * keys) the digest differs between runtimes by design — a sticky key is only
 * ever compared within one process.
 */
import { createHash } from "node:crypto";

// --------------------------------------------------------------------------- //
// Enums
// --------------------------------------------------------------------------- //
export enum WorkloadClass {
  Agent = "agent",
  Chat = "chat",
  Rag = "rag",
  Batch = "batch",
  Code = "code",
  Vision = "vision",
  Unknown = "unknown",
}

export enum WorkloadSource {
  ExplicitTag = "explicit_tag",
  AgentProfile = "agent_profile",
  Fingerprint = "fingerprint",
  Structural = "structural",
  Classifier = "classifier",
  Default = "default",
}

export enum DeploymentClass {
  Api = "api",
  SelfHosted = "self_hosted",
  Serverless = "serverless",
  Edge = "edge",
}

export enum SegmentKind {
  Tools = "tools",
  System = "system",
  Retrieved = "retrieved",
  History = "history",
  User = "user",
}

// --------------------------------------------------------------------------- //
// Messages
// --------------------------------------------------------------------------- //
export interface ContentBlock {
  type: string;
  text?: string;
  content?: string;
  [key: string]: unknown;
}

export type Content = string | readonly ContentBlock[];

export class Message {
  constructor(public role: string, public content: Content = "") {}

  /** Text of the message: string content, or the text/tool_result blocks. */
  get text(): string {
    if (typeof this.content === "string") return this.content;
    const out: string[] = [];
    for (const block of this.content) {
      if (block.type === "text") out.push(String(block.text ?? ""));
      else if (block.type === "tool_result") out.push(String(block.content ?? ""));
      else out.push(String(block));
    }
    return out.join("\n");
  }

  hasBlockType(blockType: string): boolean {
    if (typeof this.content === "string") return false;
    return this.content.some((b) => b.type === blockType);
  }
}

// --------------------------------------------------------------------------- //
// Request
// --------------------------------------------------------------------------- //
export interface Constraints {
  maxCostUsd?: number | null;
  maxLatencyMs?: number | null;
  budgetRemainingUsd?: number | null;
  residencyTags?: ReadonlySet<string>;
  requireTools?: boolean;
}

export const DEFAULT_CONSTRAINTS: Constraints = {};

export interface ToolSchema {
  name?: string;
  [key: string]: unknown;
}

export class RoutingRequest {
  readonly constraints: Constraints;
  readonly tags: ReadonlySet<string>;

  constructor(
    public readonly messages: readonly Message[],
    public readonly workload: WorkloadClass | null = null,
    public readonly sessionId: string | null = null,
    public readonly agentProfile: string | null = null,
    public readonly tools: readonly ToolSchema[] = [],
    public readonly retrieved: readonly string[] = [],
    tags: ReadonlySet<string> | Iterable<string> = new Set<string>(),
    constraints: Constraints = DEFAULT_CONSTRAINTS,
    public readonly expectedTurns: number | null = null,
    public readonly expectedGapS: number | null = null,
    public readonly expectedGenerationS: number | null = null,
    public readonly clientId: string | null = null,
  ) {
    this.tags =
      tags instanceof Set ? tags : new Set<string>(tags as Iterable<string>);
    this.constraints = constraints;
  }

  /** Number of prior user turns. 1 == first turn of a conversation. */
  get turnIndex(): number {
    return this.messages.filter((m) => m.role === "user").length;
  }

  get isFirstTurn(): boolean {
    return this.turnIndex <= 1;
  }

  firstUserText(): string {
    for (const m of this.messages) {
      if (m.role === "user") return m.text;
    }
    return "";
  }

  systemText(): string {
    return this.messages
      .filter((m) => m.role === "system")
      .map((m) => m.text)
      .join("\n");
  }

  hasToolResults(): boolean {
    return this.messages.some(
      (m) => (m.role === "tool" || m.role === "user") && m.hasBlockType("tool_result"),
    );
  }

  hasImages(): boolean {
    return this.messages.some(
      (m) => m.hasBlockType("image") || m.hasBlockType("image_url"),
    );
  }

  /** Hash of the stable part of the prompt: system text + tool schemas. */
  promptFingerprint(): string {
    return hash16(`${this.systemText()}\x00${stableJson(this.tools)}`);
  }

  /**
   * Hash of the request fields that are part of a provider cache key but
   * OUTSIDE the prefix text: tool schemas and image presence (Rule F).
   */
  invalidatorsFingerprint(): string {
    return hash16(`${stableJson(this.tools)}\x00${this.hasImages() ? 1 : 0}`);
  }

  /** Session key WITHOUT requiring the caller to send a session id. */
  stickyKey(): string {
    if (this.sessionId) return `sid:${this.sessionId}`;
    const first = this.firstUserText();
    if (first) return `ctx:${hash16(first)}`;
    return `fp:${this.promptFingerprint()}:${this.clientId ?? "anon"}`;
  }
}

function stableJson(value: unknown): string {
  return JSON.stringify(value, (_k, v) => {
    if (v && typeof v === "object" && !Array.isArray(v)) {
      return Object.fromEntries(
        Object.entries(v as Record<string, unknown>).sort(([a], [b]) =>
          a < b ? -1 : a > b ? 1 : 0,
        ),
      );
    }
    return v;
  });
}

function hash16(material: string): string {
  return createHash("sha256").update(material, "utf8").digest().subarray(0, 16).toString("hex");
}

// --------------------------------------------------------------------------- //
// Decisions
// --------------------------------------------------------------------------- //
export interface Candidate {
  modelId: string;
  backendId: string;
  score: number;
  quality: number;
  costUsd: number;
  latencyMs: number;
  affinity: number;
  switchCostUsd: number;
  reasons: readonly string[];
}

export interface WorkloadSignal {
  workload: WorkloadClass;
  source: WorkloadSource;
  confidence: number;
  reason: string;
}

export class Decision {
  constructor(
    public modelId: string,
    public backendId: string,
    public workload: WorkloadClass,
    public workloadSource: WorkloadSource,
    public workloadConfidence: number,
    public stickyKey: string,
    public sessionKey: string,
    public isFirstTurn: boolean,
    public reusedSession: boolean,
    public switched: boolean,
    public stranded: boolean,
    public ttlBySegment: ReadonlyMap<SegmentKind, string>,
    public candidates: readonly Candidate[],
    public reasons: readonly string[],
  ) {}

  /** The full auditable decision — every decision reports why. */
  explain(): string {
    const head =
      `model=${this.modelId} backend=${this.backendId} ` +
      `workload=${this.workload}(src=${this.workloadSource}, ` +
      `conf=${this.workloadConfidence.toFixed(2)}) ` +
      `sticky=${this.reusedSession} switched=${this.switched} ` +
      `stranded=${this.stranded}`;
    const ttl =
      [...this.ttlBySegment.entries()].map(([k, v]) => `${k}=${v}`).join(", ") ||
      "n/a";
    const lines = [head, `  ttl: ${ttl}`];
    for (const c of this.candidates.slice(0, 5)) {
      lines.push(
        `  ${c.modelId}@${c.backendId} score=${c.score >= 0 ? "+" : ""}${c.score.toFixed(4)} ` +
          `q=${c.quality.toFixed(3)} cost=$${c.costUsd.toFixed(5)} ` +
          `lat=${c.latencyMs.toFixed(0)}ms aff=${c.affinity.toFixed(2)} ` +
          `switch=$${c.switchCostUsd.toFixed(5)}`,
      );
    }
    for (const r of this.reasons) lines.push(`  · ${r}`);
    return lines.join("\n");
  }
}
