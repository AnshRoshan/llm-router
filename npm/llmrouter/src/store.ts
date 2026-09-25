/**
 * Durable router state: sessions, backend health, fingerprints.
 *
 * The TypeScript counterpart of llmrouter/store.py, sharing the SAME snapshot
 * format (`llmrouter-state-v1`) — a Python router and a Node router can pick
 * up each other's learning state after a restart.
 *
 * Every stored timestamp is rebased onto the CURRENT monotonic clock at
 * restore, because monotonic clocks are process-relative: a session with
 * 40 minutes of TTL left when saved has 40 minutes left after restore.
 *
 * This module is pure JSON in / state out; file I/O belongs to the caller
 * (cli.ts does an atomic write: tmp + rename).
 */
import { nowMonotonic } from "./clock.js";
import { SessionPolicy, SessionStore } from "./affinity.js";
import { BackendRegistry } from "./backends.js";
import { FingerprintRegistry } from "./signals.js";
import { WorkloadClass, WorkloadSource } from "./types.js";
import type { PolicyEngine } from "./policy.js";

export const SNAPSHOT_FORMAT = "llmrouter-state-v1";
export const SNAPSHOT_VERSION = 1

export interface Snapshot {
  format: string;
  version: number;
  saved_wall: number;
  sessions: Record<string, Record<string, unknown>>;
  backend_health: Record<string, Record<string, unknown>>;
  fingerprints: {
    map: Record<string, { workload: string; source: string; confidence: number; reason: string }>;
    seen_by_client: Record<string, string[]>;
  };
  // pairs/counters are Python-side (delegation + billing live there);
  // they are preserved untouched if present.
  [key: string]: unknown;
}

export function takeSnapshot(engine: PolicyEngine): Snapshot {
  const mono = nowMonotonic();
  const age = (v: number | null | undefined): number | null =>
    v === null || v === undefined ? null : mono - v;

  const sessions: Snapshot["sessions"] = {};
  for (const s of engine.sessions.all()) {
    const [hits, total] = s.ewma;
    sessions[s.sessionKey] = {
      workload: s.workload, source: s.source, confidence: s.confidence,
      pinned_model: s.pinnedModel, pinned_backend: s.pinnedBackend,
      pinned_at_age: age(s.pinnedAt),
      switched_at_age: age(s.switchedAt),
      expires_in: s.expiresAt ? age(s.expiresAt) : 0.0,
      turns_seen: s.turnsSeen, prefix_tokens: s.prefixTokens,
      observed_cache_hit_rate: s.observedCacheHitRate,
      ewma_hits: hits, ewma_total: total,
      stranded_count: s.strandedCount,
      invalidators_hash: s.invalidatorsHash,
      last_seen_age: age(s.lastSeen),
    };
  }

  const health: Snapshot["backend_health"] = {};
  for (const h of engine.backends.allHealth()) {
    health[h.backendId] = {
      successes: h.successes, failures: h.failures,
      consecutive_failures: h.consecutiveFailures,
      ewma_latency_ms: h.ewmaLatencyMs, p99_latency_ms: h.p99LatencyMs,
      open_until_age: h.openUntil ? mono - h.openUntil : 0.0,
      rate_limited_until_age: h.rateLimitedUntil ? mono - h.rateLimitedUntil : 0.0,
      last_error: h.lastError,
    };
  }

  return {
    format: SNAPSHOT_FORMAT,
    version: SNAPSHOT_VERSION,
    saved_wall: Date.now() / 1000,
    sessions,
    backend_health: health,
    fingerprints: dumpFingerprints(engine.resolver.fingerprints),
  };
}

export function restoreSnapshot(engine: PolicyEngine, snap: Snapshot): void {
  if (snap.format !== SNAPSHOT_FORMAT) {
    throw new Error(`not an llmrouter state snapshot: ${String(snap.format)}`);
  }
  const now = nowMonotonic();

  for (const [key, d] of Object.entries(snap.sessions ?? {})) {
    const exp = d.expires_in as number;
    const switchedAge = d.switched_at_age as number | null;
    const sess = new SessionPolicy(
      key,
      d.workload as WorkloadClass,
      d.source as WorkloadSource,
      Number(d.confidence),
      String(d.pinned_model),
      String(d.pinned_backend),
      now - Number(d.pinned_at_age ?? 0),
    );
    sess.switchedAt = switchedAge === null ? null : now - switchedAge;
    sess.expiresAt = exp ? now - exp : 0.0;
    sess.turnsSeen = Number(d.turns_seen ?? 0);
    sess.prefixTokens = Number(d.prefix_tokens ?? 0);
    sess.observedCacheHitRate =
      (d.observed_cache_hit_rate as number | null | undefined) ?? null;
    sess.hydrateEwma(
      Number(d.ewma_hits ?? 0), Number(d.ewma_total ?? 0),
      sess.observedCacheHitRate,
    );
    sess.strandedCount = Number(d.stranded_count ?? 0);
    sess.invalidatorsHash = (d.invalidators_hash as string | null) ?? null;
    sess.lastSeen = now - Number(d.last_seen_age ?? 0);
    engine.sessions.restore(sess);
  }

  for (const [bid, d] of Object.entries(snap.backend_health ?? {})) {
    const h = engine.backends.health(bid);
    h.successes = Number(d.successes ?? 0);
    h.failures = Number(d.failures ?? 0);
    h.consecutiveFailures = Number(d.consecutive_failures ?? 0);
    h.ewmaLatencyMs = Number(d.ewma_latency_ms ?? 0);
    h.p99LatencyMs = Number(d.p99_latency_ms ?? 0);
    // in_flight and half_open_probes are deliberately NOT restored: the
    // requests they count died with the old process.
    h.inFlight = 0;
    h.halfOpenProbes = 0;
    const openAge = Number(d.open_until_age ?? 0);
    h.openUntil = openAge > 0 ? now - openAge : 0.0;
    const rlAge = Number(d.rate_limited_until_age ?? 0);
    h.rateLimitedUntil = rlAge > 0 ? now - rlAge : 0.0;
    h.lastError = (d.last_error as string | null) ?? null;
  }

  loadFingerprints(engine.resolver.fingerprints, snap.fingerprints);
}

// --------------------------------------------------------------------------- //
function dumpFingerprints(fpr: FingerprintRegistry): Snapshot["fingerprints"] {
  const map: Snapshot["fingerprints"]["map"] = {};
  for (const [fp, sig] of fpr.snapshotMap()) {
    map[fp] = {
      workload: sig.workload, source: sig.source,
      confidence: sig.confidence, reason: sig.reason,
    };
  }
  return { map, seen_by_client: fpr.snapshotSeenByClient() };
}

function loadFingerprints(
  fpr: FingerprintRegistry,
  data: Snapshot["fingerprints"] | undefined,
): void {
  if (!data) return;
  for (const [fp, d] of Object.entries(data.map ?? {})) {
    fpr.hydrate(fp, {
      workload: d.workload as WorkloadClass,
      source: d.source as WorkloadSource,
      confidence: Number(d.confidence),
      reason: String(d.reason ?? ""),
    });
  }
  for (const [client, fps] of Object.entries(data.seen_by_client ?? {})) {
    for (const fp of fps) fpr.observe(fp, client === "anon" ? null : client);
  }
}
