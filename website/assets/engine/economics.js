import { SegmentKind } from "./types.js";
export const W_5M = 1.25;
export const W_1H = 2.0;
export const R_10 = 0.10;
export const EXPIRY_THRESHOLD_E_STAR = (W_1H - W_5M) / (W_5M - R_10); // 0.652...
export const KEEPALIVE_BREAK_EVEN_MIN = ((W_1H - W_5M) / R_10) * 5; // 37.5
export const TTL_5M_S = 300.0;
/** Cost of a shared prefix over `turns` turns on one model. */
export function prefixCost(card, prefixTokens, turns, ttl = "5m") {
    if (turns <= 0)
        return 0.0;
    const w = card.cache.writeMult(ttl);
    const perTok = card.inputPerMtok / 1_000_000;
    return prefixTokens * perTok * (w + (turns - 1) * card.cache.readMult);
}
/** Direct summation: `switchAt` turns on `cur`, the remainder on `tgt`. */
export function sessionCostWithSwitch(cur, tgt, prefixTokens, totalTurns, switchAt) {
    const stay = switchAt;
    const remaining = totalTurns - switchAt;
    if (stay < 0 || remaining < 0 || stay + remaining !== totalTurns) {
        throw new Error(`switchAt=${switchAt} inconsistent with totalTurns=${totalTurns}`);
    }
    return prefixCost(cur, prefixTokens, stay) + prefixCost(tgt, prefixTokens, remaining);
}
/** R* — turns needed for a one-way migration to pay for itself. */
export function breakEvenTurns(cur, tgt) {
    const r = cur.cache.readMult;
    const denom = (cur.inputPerMtok - tgt.inputPerMtok) * r;
    if (denom <= 0)
        return Number.POSITIVE_INFINITY;
    const num = tgt.inputPerMtok * tgt.cache.writeMult5m - cur.inputPerMtok * r;
    return num / denom;
}
/**
 * Should we migrate a live session from `cur` to `tgt`?
 *   1. No oscillation — a session that already migrated once never does again.
 *   2. Amortization — must pay back inside the turns left.
 *   3. Otherwise stranded (a surfaced metric, not a silent no-op).
 */
export function evaluateSwitch(cur, tgt, prefixTokens, turnsRemaining, alreadySwitched = false) {
    if (alreadySwitched) {
        return {
            shouldSwitch: false, breakEvenTurns: 0, turnsRemaining,
            oneTimeCostUsd: 0, perTurnSavingUsd: 0,
            reason: "no-oscillation lock: session already migrated once",
        };
    }
    const rStar = breakEvenTurns(cur, tgt);
    if (!Number.isFinite(rStar)) {
        return {
            shouldSwitch: false, breakEvenTurns: rStar, turnsRemaining,
            oneTimeCostUsd: 0, perTurnSavingUsd: 0,
            reason: "target is not cheaper per turn",
        };
    }
    const oneTime = prefixTokens *
        ((tgt.inputPerMtok / 1e6) * tgt.cache.writeMult5m -
            (cur.inputPerMtok / 1e6) * cur.cache.readMult);
    let perTurn = ((cur.inputPerMtok - tgt.inputPerMtok) * cur.cache.readMult) / 1e6;
    perTurn *= prefixTokens;
    if (turnsRemaining >= rStar) {
        return {
            shouldSwitch: true, breakEvenTurns: rStar, turnsRemaining,
            oneTimeCostUsd: oneTime, perTurnSavingUsd: perTurn,
            reason: `amortizes in ${rStar.toFixed(2)} turns, ${turnsRemaining} left`,
        };
    }
    return {
        shouldSwitch: false, breakEvenTurns: rStar, turnsRemaining,
        oneTimeCostUsd: oneTime, perTurnSavingUsd: perTurn,
        reason: `stranded: needs ${rStar.toFixed(2)} turns, only ${turnsRemaining} left`,
    };
}
/** How many times a 5m cache is expected to expire over the session. */
export function expectedExpiries(sessionDurationS, gapS, ttlS = TTL_5M_S) {
    if (gapS <= ttlS || sessionDurationS <= 0)
        return 0.0;
    return Math.max(0, sessionDurationS / gapS - 1);
}
/** Pick a TTL for one prefix segment. `exposuresS` = gap + generation time. */
export function chooseTtlForPrefix(sem, prefixTokens, exposuresS, ttlS = TTL_5M_S) {
    if (!sem.ttlOptions.includes("1h"))
        return "5m";
    if (!sem.canCache(prefixTokens))
        return "5m";
    return exposuresS > ttlS * EXPIRY_THRESHOLD_E_STAR * 2 ? "1h" : "5m";
}
/** Segments worth protecting across a human-sized gap. */
export const LONG_TTL_SEGMENTS = new Set([
    SegmentKind.Tools, SegmentKind.System, SegmentKind.Retrieved,
]);
/** Per-breakpoint TTL: system/tools/retrieved at 1h, history at 5m. */
export function chooseTtlBySegment(sem, segmentTokens, expectedGapS, expectedGenerationS = 0, ttlS = TTL_5M_S) {
    const out = new Map();
    if (!sem.supportsMixedTtl || !sem.ttlOptions.includes("1h")) {
        for (const k of segmentTokens.keys())
            out.set(k, "5m");
        return out;
    }
    const exposures = expectedGapS + expectedGenerationS;
    for (const [kind, tokens] of segmentTokens) {
        if (LONG_TTL_SEGMENTS.has(kind) && sem.canCache(tokens)) {
            out.set(kind, exposures > ttlS ? "1h" : "5m");
        }
        else {
            out.set(kind, "5m");
        }
    }
    return out;
}
const PREFIX_ORDER = [
    SegmentKind.Tools, SegmentKind.System, SegmentKind.Retrieved,
    SegmentKind.History, SegmentKind.User,
];
/**
 * Build the cache_control breakpoint list. Invariants:
 *   * at most `maxBreakpoints`
 *   * longer TTL must PRECEDE shorter (a real proxy shipped this bug)
 *   * segments under minCacheableTokens are skipped
 */
export function emitBreakpoints(sem, segmentTokens, ttlBySegment) {
    const present = PREFIX_ORDER.filter((k) => segmentTokens.has(k));
    const rank = (k) => (ttlBySegment.get(k) === "1h" ? 0 : 1);
    const ranked = [...present].sort((a, b) => rank(a) - rank(b) || PREFIX_ORDER.indexOf(a) - PREFIX_ORDER.indexOf(b));
    const out = [];
    let offset = 0;
    let seenShort = false;
    for (const kind of ranked) {
        const tokens = segmentTokens.get(kind);
        let ttl = ttlBySegment.get(kind) ?? "5m";
        if (!sem.canCache(tokens)) {
            offset += tokens;
            continue;
        }
        if (out.length >= sem.maxBreakpoints) {
            offset += tokens;
            continue;
        }
        if (ttl === "5m")
            seenShort = true;
        else if (seenShort)
            ttl = "5m"; // 1h after 5m is rejected/downgraded
        out.push({ kind, tokens, ttl, offset });
        offset += tokens;
    }
    assertTtlOrdering(out);
    return out;
}
/** Property baked into the emitter: no 1h after a 5m. */
export function assertTtlOrdering(breakpoints) {
    let seenShort = false;
    for (const bp of breakpoints) {
        if (bp.ttl === "5m")
            seenShort = true;
        else if (seenShort) {
            throw new Error(`1h breakpoint for ${bp.kind} appears after a 5m breakpoint; ` +
                "the provider will reject or downgrade this request");
        }
    }
}
export function keepaliveCostMultiplier(gapMinutes) {
    return (gapMinutes / 5.0) * R_10;
}
export function keepaliveBeatsUpgrade(gapMinutes) {
    return keepaliveCostMultiplier(gapMinutes) < W_1H - W_5M;
}
