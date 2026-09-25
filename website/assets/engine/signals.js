/**
 * Workload-class inference: the two entry paths.
 *   PATH A (declared) — tag or named agent profile. ~0 ms, skips inference.
 *   PATH B (inferred) — B1 structural rules → B2 prompt fingerprint →
 *                       B3 classifier hook (turn 1 only) → B4 default.
 */
import { WorkloadClass, WorkloadSource, } from "./types.js";
/** Fallback turns-per-session priors. Assumptions to calibrate, not measurements. */
export const TURNS_PRIOR = {
    [WorkloadClass.Agent]: 25,
    [WorkloadClass.Chat]: 6,
    [WorkloadClass.Rag]: 2,
    [WorkloadClass.Batch]: 1,
    [WorkloadClass.Code]: 12,
    [WorkloadClass.Vision]: 3,
    [WorkloadClass.Unknown]: 5,
};
/** Expected idle gap between turns, seconds. Agents are seconds; humans are not. */
export const GAP_PRIOR = {
    [WorkloadClass.Agent]: 5.0,
    [WorkloadClass.Chat]: 90.0,
    [WorkloadClass.Rag]: 45.0,
    [WorkloadClass.Batch]: 0.0,
    [WorkloadClass.Code]: 20.0,
    [WorkloadClass.Vision]: 60.0,
    [WorkloadClass.Unknown]: 60.0,
};
export const CODE_MARKERS = [
    "```", "def ", "class ", "function ", "import ", "const ", "SELECT ",
];
/** B1 — zero-cost inference from the shape of the request. */
export function inferStructural(req) {
    if (req.hasImages()) {
        return sig(WorkloadClass.Vision, WorkloadSource.Structural, 0.99, "image content block present");
    }
    if (req.tools.length > 0 && req.hasToolResults()) {
        return sig(WorkloadClass.Agent, WorkloadSource.Structural, 0.97, "tools present and prior tool_result in history");
    }
    if (req.tools.length > 0) {
        return sig(WorkloadClass.Agent, WorkloadSource.Structural, 0.85, "tools present, no tool_result yet (first step)");
    }
    if (req.retrieved.length > 0) {
        return sig(WorkloadClass.Rag, WorkloadSource.Structural, 0.90, "retrieved chunks supplied with the request");
    }
    if (req.tags.has("batch")) {
        return sig(WorkloadClass.Batch, WorkloadSource.Structural, 1.0, "tagged batch");
    }
    const userText = req.firstUserText();
    const codeHits = CODE_MARKERS.filter((m) => userText.includes(m)).length;
    if (codeHits >= 2 || req.tags.has("code")) {
        return sig(WorkloadClass.Code, WorkloadSource.Structural, 0.70, `${codeHits} code markers in the first user turn`);
    }
    if (req.turnIndex >= 2 && req.tools.length === 0) {
        return sig(WorkloadClass.Chat, WorkloadSource.Structural, 0.80, "multi-turn, no tools");
    }
    return null;
}
function sig(workload, source, confidence, reason) {
    return { workload, source, confidence, reason };
}
/**
 * B2 — prompt fingerprint registry. Production traffic comes from a handful of
 * distinct application prompts; this turns "classify every request" into
 * "classify every application".
 */
export class FingerprintRegistry {
    map = new Map();
    seenByClient = new Map();
    get(fingerprint) {
        return this.map.get(fingerprint) ?? null;
    }
    put(fingerprint, signal) {
        const existing = this.map.get(fingerprint);
        if (existing && existing.confidence >= signal.confidence)
            return;
        this.map.set(fingerprint, signal);
    }
    observe(fingerprint, clientId) {
        const key = clientId ?? "anon";
        const set = this.seenByClient.get(key) ?? new Set();
        set.add(fingerprint);
        this.seenByClient.set(key, set);
    }
    /** Snapshot accessors for state persistence (see store.ts). */
    snapshotMap() {
        return [...this.map.entries()];
    }
    hydrate(fingerprint, signal) {
        this.map.set(fingerprint, signal);
    }
    snapshotSeenByClient() {
        const out = {};
        for (const [k, v] of this.seenByClient)
            out[k] = [...v];
        return out;
    }
    cardinalityByClient() {
        const out = {};
        for (const [k, v] of this.seenByClient)
            out[k] = v.size;
        return out;
    }
    /** Clients generating too many distinct prompts to ever cache. */
    volatileClients(threshold = 50) {
        const out = [];
        for (const [k, v] of this.seenByClient) {
            if (v.size > threshold)
                out.push(k);
        }
        return out;
    }
    get size() {
        return this.map.size;
    }
}
/** Resolves a workload class via Path A then Path B. */
export class WorkloadResolver {
    classifier;
    defaultWorkload;
    profiles;
    fingerprints;
    constructor(profiles = null, classifier = null, defaultWorkload = WorkloadClass.Chat) {
        this.classifier = classifier;
        this.defaultWorkload = defaultWorkload;
        if (profiles instanceof Map) {
            this.profiles = new Map(profiles);
        }
        else if (profiles) {
            this.profiles = new Map(Object.entries(profiles));
        }
        else {
            this.profiles = new Map();
        }
        this.fingerprints = new FingerprintRegistry();
    }
    addProfile(profile) {
        this.profiles.set(profile.name, profile);
    }
    resolve(req) {
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
        const fallback = {
            workload: this.defaultWorkload, source: WorkloadSource.Default,
            confidence: 0.3,
            reason: `no signal; using default ${this.defaultWorkload}`,
        };
        this.fingerprints.put(fp, fallback);
        return fallback;
    }
}
