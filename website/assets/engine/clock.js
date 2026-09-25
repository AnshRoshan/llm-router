/**
 * Monotonic seconds, runtime-agnostic: performance.now in browsers, workers
 * and Deno; process.hrtime in Node. The decision layer takes `now` as a
 * parameter everywhere it matters, so this is only the default.
 */
export function nowMonotonic() {
    const g = globalThis;
    if (typeof g.performance?.now === "function") {
        return g.performance.now() / 1000;
    }
    const p = globalThis
        .process;
    if (p?.hrtime?.bigint) {
        return Number(p.hrtime.bigint()) / 1e9;
    }
    return Date.now() / 1000; // last resort: not monotonic, still consistent
}
