/**
 * Backend registry, live health signals, the circuit breaker.
 * Axis 2: WHERE the model runs. The router scores (model, backend) PAIRS.
 * Local timeouts are aggressive on purpose: a saturated GPU that cascades to
 * cloud at 100x cost is the trap.
 */
import { nowMonotonic } from "./clock.js";
import { DeploymentClass } from "./types.js";

export const DEFAULT_LOCAL_TIMEOUT_S = 8.0;
export const DEFAULT_API_TIMEOUT_S = 60.0;
export const CIRCUIT_FAILURE_THRESHOLD = 5;
export const CIRCUIT_BASE_BACKOFF_S = 1.0;
export const CIRCUIT_MAX_BACKOFF_S = 60.0;
/** Probe requests allowed in flight against a HALF_OPEN backend. */
export const HALF_OPEN_PROBE_LIMIT = 1;

export interface BackendSpecInput {
  backendId?: string;
  deployment?: DeploymentClass;
  models?: readonly string[];
  baseUrl?: string | null;
  priority?: number;
  timeoutS?: number;
  maxConcurrency?: number;
  coldStartRisk?: number;
  costFactor?: number;
  capacityRps?: number | null;
}

export class BackendSpec {
  readonly backendId: string;
  readonly deployment: DeploymentClass;
  readonly models: readonly string[];
  readonly baseUrl: string | null;
  readonly priority: number;
  readonly timeoutS: number;
  readonly maxConcurrency: number;
  readonly coldStartRisk: number;
  readonly costFactor: number;
  readonly capacityRps: number | null;

  constructor(
    backendIdOrOpts: string | BackendSpecInput,
    maybeOpts: BackendSpecInput = {},
  ) {
    const opts =
      typeof backendIdOrOpts === "string" ? maybeOpts : backendIdOrOpts;
    const backendId =
      typeof backendIdOrOpts === "string"
        ? backendIdOrOpts
        : backendIdOrOpts.backendId;
    if (!backendId) throw new Error("BackendSpec requires a backendId");
    this.backendId = backendId;
    this.deployment = opts.deployment ?? DeploymentClass.Api;
    this.models = opts.models ?? [];
    this.baseUrl = opts.baseUrl ?? null;
    this.priority = opts.priority ?? 0;
    this.timeoutS = opts.timeoutS ?? DEFAULT_API_TIMEOUT_S;
    this.maxConcurrency = opts.maxConcurrency ?? 64;
    this.coldStartRisk = opts.coldStartRisk ?? 0.0;
    this.costFactor = opts.costFactor ?? 1.0;
    this.capacityRps = opts.capacityRps ?? null;
  }

  supports(modelId: string): boolean {
    return this.models.length === 0 || this.models.includes(modelId);
  }
}

export class BackendHealth {
  inFlight = 0;
  successes = 0;
  failures = 0;
  consecutiveFailures = 0;
  ewmaLatencyMs = 0.0;
  p99LatencyMs = 0.0;
  /** Circuit open until this monotonic timestamp. */
  openUntil = 0.0;
  halfOpenProbes = 0;
  rateLimitedUntil = 0.0;
  lastError: string | null = null;
  private readonly latencyAlpha = 0.2;

  constructor(public readonly backendId: string) {}

  get errorRate(): number {
    const total = this.successes + this.failures;
    return total === 0 ? 0.0 : this.failures / total;
  }

  isAvailable(now: number = nowMonotonic()): boolean {
    if (this.rateLimitedUntil && now < this.rateLimitedUntil) return false;
    return now >= this.openUntil;
  }

  isHalfOpen(now: number = nowMonotonic()): boolean {
    return this.openUntil > 0 && this.openUntil <= now;
  }

  recordSuccess(latencyMs: number, now: number = nowMonotonic()): void {
    this.successes += 1;
    this.consecutiveFailures = 0;
    this.ewmaLatencyMs =
      (1 - this.latencyAlpha) * this.ewmaLatencyMs + this.latencyAlpha * latencyMs;
    this.p99LatencyMs = Math.max(this.p99LatencyMs * 0.95, latencyMs);
    this.openUntil = 0.0;
    this.halfOpenProbes = 0;
    this.lastError = null;
  }

  recordFailure(
    error: string, retryAfterS: number | null = null, now: number = nowMonotonic(),
  ): void {
    this.failures += 1;
    this.consecutiveFailures += 1;
    this.lastError = error;
    if (retryAfterS) this.rateLimitedUntil = now + retryAfterS;
    if (this.consecutiveFailures >= CIRCUIT_FAILURE_THRESHOLD) {
      const backoff = Math.min(
        CIRCUIT_BASE_BACKOFF_S *
          2 ** (this.consecutiveFailures - CIRCUIT_FAILURE_THRESHOLD),
        CIRCUIT_MAX_BACKOFF_S,
      );
      this.openUntil = now + backoff;
      // A failure while half-open re-opens the circuit and revokes any
      // in-flight canary allowance.
      this.halfOpenProbes = 0;
    }
  }

  saturating(spec: BackendSpec): boolean {
    if (this.inFlight >= spec.maxConcurrency) return true;
    if (spec.capacityRps !== null && this.inFlight > spec.capacityRps) return true;
    return false;
  }
}

export class BackendRegistry {
  private readonly specs = new Map<string, BackendSpec>();
  private readonly healthMap = new Map<string, BackendHealth>();

  constructor(backends: readonly BackendSpec[] = []) {
    for (const b of backends) this.register(b);
  }

  register(spec: BackendSpec): void {
    this.specs.set(spec.backendId, spec);
    if (!this.healthMap.has(spec.backendId)) {
      this.healthMap.set(spec.backendId, new BackendHealth(spec.backendId));
    }
  }

  unregister(backendId: string): void {
    this.specs.delete(backendId);
    this.healthMap.delete(backendId);
  }

  spec(backendId: string): BackendSpec | undefined {
    return this.specs.get(backendId);
  }

  health(backendId: string): BackendHealth {
    let h = this.healthMap.get(backendId);
    if (!h) {
      h = new BackendHealth(backendId);
      this.healthMap.set(backendId, h);
    }
    return h;
  }

  /** Read-only view for state snapshots (see store.ts). */
  allHealth(): IterableIterator<BackendHealth> {
    return this.healthMap.values();
  }

  candidatesFor(modelId: string): BackendSpec[] {
    return [...this.specs.values()].filter((b) => b.supports(modelId));
  }

  /** Healthy, non-saturated backends serving this model, by priority. */
  availableFor(modelId: string, now: number = nowMonotonic()): BackendSpec[] {
    const out = this.candidatesFor(modelId).filter((b) => {
      const h = this.health(b.backendId);
      if (!h.isAvailable(now) || h.saturating(b)) return false;
      // HALF_OPEN: admit only while a canary slot is free.
      if (h.isHalfOpen(now) && h.halfOpenProbes >= HALF_OPEN_PROBE_LIMIT) return false;
      return true;
    });
    return out.sort((a, b) => a.priority - b.priority);
  }

  ids(): string[] {
    return [...this.specs.keys()];
  }

  get size(): number {
    return this.specs.size;
  }
}

