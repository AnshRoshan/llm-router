/**
 * llmrouter — cache-aware LLM routing for Node.js (decision engine).
 *
 * Picks the (model, backend) pair, keeps sessions sticky so the prompt cache
 * survives, and chooses prompt-cache TTL per breakpoint. Zero dependencies.
 */
export * from "./types.js";
export * from "./pricing.js";
export * from "./economics.js";
export * from "./affinity.js";
export * from "./backends.js";
export * from "./signals.js";
export * from "./policy.js";
export { runCli } from "./cli.js";

export const VERSION = "0.1.0";
