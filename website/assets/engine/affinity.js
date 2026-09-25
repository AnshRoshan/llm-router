/**
 * Session affinity: sticky keys, rendezvous hashing, the session store.
 * Rule A: session affinity is the DEFAULT, not an optimization. Rendezvous
 * (HRW) hashing because it is stable under membership change — a hash-ring
 * rebalance is what silently converts a 90% hit rate into 12.5%.
 */
import { nowMonotonic } from "./clock.js";
import { WorkloadClass } from "./types.js";
import { hash8Big } from "./hash.js";
export const DEFAULT_IDLE_TTL_S = 3600.0;
/** Deterministic, orderable rendezvous score for (key, node). */
function hrw(key, node) {
    return hash8Big(key + "\x00" + node);
}
export function rendezvousPick(key, nodes) {
    if (nodes.length === 0)
        return null;
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
export function rendezvousRank(key, nodes) {
    return [...nodes].sort((a, b) => (hrw(key, a) < hrw(key, b) ? 1 : -1));
}
/** Mutable per-session routing state. */
export class SessionPolicy {
    sessionKey;
    workload;
    source;
    confidence;
    pinnedModel;
    pinnedBackend;
    pinnedAt;
    lastSeenNow;
    switchedAt = null;
    expiresAt = 0.0;
    turnsSeen = 0;
    prefixTokens = 0;
    /** Learned, not assumed: seeded by a prior, updated from usage counters. */
    observedCacheHitRate = null;
    ewmaHits = 0.0;
    ewmaTotal = 0.0;
    /** A better model existed but the session had too few turns left to migrate. */
    strandedCount = 0;
    /** Rule F baseline: invalidating-field fingerprint at pin time. */
    invalidatorsHash = null;
    lastSeen;
    constructor(sessionKey, workload, source, confidence, pinnedModel, pinnedBackend, pinnedAt, lastSeenNow) {
        this.sessionKey = sessionKey;
        this.workload = workload;
        this.source = source;
        this.confidence = confidence;
        this.pinnedModel = pinnedModel;
        this.pinnedBackend = pinnedBackend;
        this.pinnedAt = pinnedAt;
        this.lastSeenNow = lastSeenNow;
        this.lastSeen = lastSeenNow ?? pinnedAt;
    }
    get alreadySwitched() {
        return this.switchedAt !== null;
    }
    /** Fold the provider's usage counters into an EWMA hit rate. */
    recordUsage(cacheReadTokens, cacheWriteTokens, alpha = 0.3) {
        const total = cacheReadTokens + cacheWriteTokens;
        if (total <= 0)
            return;
        const hit = cacheReadTokens / total;
        this.ewmaHits = (1 - alpha) * this.ewmaHits + alpha * hit;
        this.ewmaTotal = (1 - alpha) * this.ewmaTotal + alpha;
        this.observedCacheHitRate = this.ewmaHits / this.ewmaTotal;
    }
    /** Measured rate once we have samples, otherwise the prior. */
    hitRate(prior) {
        if (this.observedCacheHitRate === null || this.ewmaTotal < 0.5)
            return prior;
        return this.observedCacheHitRate;
    }
    /** Read the EWMA accumulators (for state snapshots). */
    get ewma() {
        return [this.ewmaHits, this.ewmaTotal];
    }
    /** Hydrate the EWMA from a snapshot (see store.ts). */
    hydrateEwma(hits, total, observed) {
        this.ewmaHits = hits;
        this.ewmaTotal = total;
        this.observedCacheHitRate = observed;
    }
}
/** In-process session store with idle eviction. Swap for Redis >1 replica. */
export class SessionStore {
    idleTtlS;
    maxSessions;
    sessions = new Map();
    constructor(idleTtlS = DEFAULT_IDLE_TTL_S, maxSessions = 100_000) {
        this.idleTtlS = idleTtlS;
        this.maxSessions = maxSessions;
    }
    get(sessionKey, now = nowMonotonic()) {
        const sess = this.sessions.get(sessionKey);
        if (!sess)
            return null;
        if (sess.expiresAt && now > sess.expiresAt) {
            this.sessions.delete(sessionKey);
            return null;
        }
        sess.lastSeen = now;
        sess.turnsSeen += 1;
        return sess;
    }
    put(session, now = nowMonotonic()) {
        if (this.sessions.size >= this.maxSessions && !this.sessions.has(session.sessionKey)) {
            this.evictOldest();
        }
        session.lastSeen = now;
        session.expiresAt = now + this.idleTtlS;
        this.sessions.set(session.sessionKey, session);
    }
    evictOldest() {
        let oldest = null;
        for (const s of this.sessions.values()) {
            if (!oldest || s.lastSeen < oldest.lastSeen)
                oldest = s;
        }
        if (oldest)
            this.sessions.delete(oldest.sessionKey);
    }
    evict(sessionKey) {
        this.sessions.delete(sessionKey);
    }
    /** Read without side effects (no turn counting, no TTL sweep). */
    peek(sessionKey) {
        return this.sessions.get(sessionKey) ?? null;
    }
    /** Read-only view for state snapshots (see store.ts). */
    all() {
        return this.sessions.values();
    }
    /** Insert a restored session verbatim, bypassing TTL/max-size rewriting. */
    restore(session) {
        this.sessions.set(session.sessionKey, session);
    }
    clear() {
        this.sessions.clear();
    }
    get size() {
        return this.sessions.size;
    }
    stats() {
        const vals = [...this.sessions.values()];
        if (vals.length === 0) {
            return {
                sessions: 0, switched: 0, stranded: 0,
                avgHitRate: 0, hitRateBySource: {},
            };
        }
        const bySource = new Map();
        for (const s of vals) {
            if (s.observedCacheHitRate !== null) {
                const list = bySource.get(s.source) ?? [];
                list.push(s.observedCacheHitRate);
                bySource.set(s.source, list);
            }
        }
        const hitRateBySource = {};
        for (const [k, v] of bySource) {
            hitRateBySource[k] = v.reduce((a, b) => a + b, 0) / v.length;
        }
        return {
            sessions: vals.length,
            switched: vals.filter((s) => s.alreadySwitched).length,
            stranded: vals.reduce((a, s) => a + s.strandedCount, 0),
            avgHitRate: vals.reduce((a, s) => a + (s.observedCacheHitRate ?? 0), 0) / vals.length,
            hitRateBySource,
        };
    }
}
/** Prior cache-hit probability for a workload, before any measurement. */
export function affinityPrior(workload, gapS, ttlS = 300.0) {
    if (workload === WorkloadClass.Batch)
        return 0.0;
    if (gapS <= 0)
        return 0.95;
    if (gapS >= ttlS)
        return 0.15;
    return 0.95 - 0.5 * (gapS / ttlS);
}
/** Affinity-first backend selection with deterministic overflow. */
export function pickBackend(key, healthy, current = null) {
    if (healthy.length === 0)
        return null;
    if (current !== null && healthy.includes(current))
        return current;
    return rendezvousPick(key, healthy);
}
