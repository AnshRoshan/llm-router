/**
 * Monotonic seconds, runtime-agnostic: performance.now in browsers, workers
 * and Deno; process.hrtime in Node. The decision layer takes `now` as a
 * parameter everywhere it matters, so this is only the default.
 */
export function nowMonotonic(): number {
  const g = globalThis as { performance?: { now(): number } };
  if (typeof g.performance?.now === "function") {
    return g.performance.now() / 1000;
  }
  const p = (globalThis as { process?: { hrtime?: { bigint(): bigint } } })
    .process;
  if (p?.hrtime?.bigint) {
    return Number(p.hrtime.bigint()) / 1e9;
  }
  return Date.now() / 1000; // last resort: not monotonic, still consistent
}
