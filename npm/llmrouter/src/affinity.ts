/**
 * Session affinity: sticky keys, rendezvous hashing, the session store.
 * Rule A: session affinity is the DEFAULT, not an optimization. Rendezvous
 * (HRW) hashing because it is stable under membership change — a hash-ring
 * rebalance is what silently converts a 90% hit rate into 12.5%.
 */
import { createHash } from "node:crypto";
import { WorkloadClass, WorkloadSource } from "./types.js";

export const DEFAULT_IDLE_TTL_S = 3600.0;

/** Deterministic, orderable rendezvous score for (key, node). */
function hrw(key: string, node: string): bigint {
  return createHash("sha256")
    .update(key + "\x00" + node, "utf8")
    .digest()
    .readBigUInt64BE(0);
}

export function rendezvousPick(key: string, nodes: readonly string[]): string | null {
  if (nodes.length === 0) return null;
  let best = nodes[0];
  let bestScore = hrw(key, nodes[0]);
  for (const n of nodes.slice(1)) {
    const s = hrw(key, n);
    if (s > bestScore) {
      best = n;
      bestScore = s;
    }
  }
  return best;
}

export function rendezvousRank(key: string, nodes: readonly string[]): string[] {
  return [...nodes].sort((a, b) => (hrw(key, a) < hrw(key, b) ? 1 : -1));
}

/** Mutable per-session routing state. */
export class SessionPolicy {
  switchedAt: number | null = null;
  expiresAt = 0.0;
  turnsSeen = 0;
  prefixTokens = 0;
  /** Learned, not assumed: seeded by a prior, updated from usage counters. */
  observedCacheHitRate: number | null = null;
  private ewmaHits = 0.0;
  private ewmaTotal = 0.0;
  /** A better model existed but the session had too few turns left to migrate. */
  strandedCount = 0;
  /** Rule F baseline: invalidating-field fingerprint at pin time. */
  invalidatorsHash: string | null = null;
  lastSeen: number;

  constructor(
    public readonly sessionKey: string,
    public workload: WorkloadClass,
    public source: WorkloadSource,
    public confidence: number,
    public pinnedModel: string,
    public pinnedBackend: string,
    public pinnedAt: number,
    public lastSeenNow?: number,
  ) {
    this.lastSeen = lastSeenNow ?? pinnedAt;
  }

  get alreadySwitched(): boolean {
    return this.switchedAt !== null;
  }

  /** Fold the provider's usage counters into an EWMA hit rate. */
  recordUsage(cacheReadTokens: number, cacheWriteTokens: number, alpha = 0.3): void {
    const total = cacheReadTokens + cacheWriteTokens;
    if (total <= 0) return;
    const hit = cacheReadTokens / total;
    this.ewmaHits = (1 - alpha) * this.ewmaHits + alpha * hit;
    this.ewmaTotal = (1 - alpha) * this.ewmaTotal + alpha;
    this.observedCacheHitRate = this.ewmaHits / this.ewmaTotal;
  }

  /** Measured rate once we have samples, otherwise the prior. */
  hitRate(prior: number): number {
    if (this.observedCacheHitRate === null || this.ewmaTotal < 0.5) return prior;
    return this.observedCacheHitRate;
  }
}

/** In-process session store with idle eviction. Swap for Redis >1 replica. */
export class SessionStore {
  private readonly sessions = new Map<string, SessionPolicy>();

  constructor(
    public idleTtlS: number = DEFAULT_IDLE_TTL_S,
    public maxSessions: number = 100_000,
  ) {}

  get(sessionKey: string, now: number = nowMonotonic()): SessionPolicy | null {
    const sess = this.sessions.get(sessionKey);
    if (!sess) return null;
    if (sess.expiresAt && now > sess.expiresAt) {
      this.sessions.delete(sessionKey);
      return null;
    }
    sess.lastSeen = now;
    sess.turnsSeen += 1;
    return sess;
  }

  put(session: SessionPolicy, now: number = nowMonotonic()): void {
    if (this.sessions.size >= this.maxSessions && !this.sessions.has(session.sessionKey)) {
      this.evictOldest();
    }
    session.lastSeen = now;
    session.expiresAt = now + this.idleTtlS;
    this.sessions.set(session.sessionKey, session);
  }

  private evictOldest(): void {
    let oldest: SessionPolicy | null = null;
    for (const s of this.sessions.values()) {
      if (!oldest || s.lastSeen < oldest.lastSeen) oldest = s;
    }
    if (oldest) this.sessions.delete(oldest.sessionKey);
  }

  evict(sessionKey: string): void {
    this.sessions.delete(sessionKey);
  }

  clear(): void {
    this.sessions.clear();
  }

  get size(): number {
    return this.sessions.size;
  }

  stats(): Record<string, unknown> {
    const vals = [...this.sessions.values()];
    if (vals.length === 0) {
      return {
        sessions: 0, switched: 0, stranded: 0,
        avgHitRate: 0, hitRateBySource: {},
      };
    }
    const bySource = new Map<string, number[]>();
    for (const s of vals) {
      if (s.observedCacheHitRate !== null) {
        const list = bySource.get(s.source) ?? [];
        list.push(s.observedCacheHitRate);
        bySource.set(s.source, list);
      }
    }
    const hitRateBySource: Record<string, number> = {};
    for (const [k, v] of bySource) {
      hitRateBySource[k] = v.reduce((a, b) => a + b, 0) / v.length;
    }
    return {
      sessions: vals.length,
      switched: vals.filter((s) => s.alreadySwitched).length,
      stranded: vals.reduce((a, s) => a + s.strandedCount, 0),
      avgHitRate:
        vals.reduce((a, s) => a + (s.observedCacheHitRate ?? 0), 0) / vals.length,
      hitRateBySource,
    };
  }
}

/** Prior cache-hit probability for a workload, before any measurement. */
export function affinityPrior(
  workload: WorkloadClass, gapS: number, ttlS: number = 300.0,
): number {
  if (workload === WorkloadClass.Batch) return 0.0;
  if (gapS <= 0) return 0.95;
  if (gapS >= ttlS) return 0.15;
  return 0.95 - 0.5 * (gapS / ttlS);
}

/** Affinity-first backend selection with deterministic overflow. */
export function pickBackend(
  key: string, healthy: readonly string[], current: string | null = null,
): string | null {
  if (healthy.length === 0) return null;
  if (current !== null && healthy.includes(current)) return current;
  return rendezvousPick(key, healthy);
}

function nowMonotonic(): number {
  return Number(process.hrtime.bigint()) / 1e9;
}
