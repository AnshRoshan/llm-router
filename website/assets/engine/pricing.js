/**
 * Pricing and cache semantics — ported 1:1 from the Python package.
 *
 * The two facts the price registry must encode and most routers get wrong:
 *   1. Cache discount is PER MODEL, not per provider.
 *   2. Anthropic charges a cache WRITE premium (1.25x / 2.0x); OpenAI
 *      pre-GPT-5.6 charges none. This single field decides whether a
 *      mid-session model switch is cheap or expensive.
 */
import { DeploymentClass } from "./types.js";
export class CacheSemantics {
    readMult;
    writeMult5m;
    writeMult1h;
    minCacheableTokens;
    ttlOptions;
    ttlRefreshesOnHit;
    supportsMixedTtl;
    maxBreakpoints;
    explicitCacheKeyParam;
    constructor(
    /** Cache read as a multiple of base input price. */
    readMult = 0.10, 
    /** 5-minute cache write multiplier. 1.0 = no write premium (OpenAI). */
    writeMult5m = 1.25, 
    /** 1-hour cache write multiplier. */
    writeMult1h = 2.0, 
    /** Below this prefix length no cache entry is created at all. */
    minCacheableTokens = 1024, 
    /** Which TTL tiers exist. Do not model what the provider lacks. */
    ttlOptions = ["5m"], ttlRefreshesOnHit = true, supportsMixedTtl = false, maxBreakpoints = 4, explicitCacheKeyParam = null) {
        this.readMult = readMult;
        this.writeMult5m = writeMult5m;
        this.writeMult1h = writeMult1h;
        this.minCacheableTokens = minCacheableTokens;
        this.ttlOptions = ttlOptions;
        this.ttlRefreshesOnHit = ttlRefreshesOnHit;
        this.supportsMixedTtl = supportsMixedTtl;
        this.maxBreakpoints = maxBreakpoints;
        this.explicitCacheKeyParam = explicitCacheKeyParam;
    }
    writeMult(ttl) {
        return ttl === "1h" ? this.writeMult1h : this.writeMult5m;
    }
    canCache(prefixTokens) {
        return prefixTokens >= this.minCacheableTokens;
    }
}
export class PriceCard {
    modelId;
    inputPerMtok;
    outputPerMtok;
    cache;
    contextWindow;
    supportsTools;
    supportsVision;
    deployment;
    residencyTags;
    qualityPrior;
    p50LatencyMs;
    tokenEfficiency;
    reworkFactor;
    constructor(modelId, inputPerMtok, outputPerMtok, cache = new CacheSemantics(), contextWindow = 200_000, supportsTools = true, supportsVision = false, deployment = DeploymentClass.Api, residencyTags = new Set(), 
    /** Quality prior per workload class. Placeholders — calibrate on YOUR traffic. */
    qualityPrior = {}, p50LatencyMs = 800.0, 
    /** Price-per-task factors (Cognition's Fusion finding). */
    tokenEfficiency = 1.0, reworkFactor = 1.0) {
        this.modelId = modelId;
        this.inputPerMtok = inputPerMtok;
        this.outputPerMtok = outputPerMtok;
        this.cache = cache;
        this.contextWindow = contextWindow;
        this.supportsTools = supportsTools;
        this.supportsVision = supportsVision;
        this.deployment = deployment;
        this.residencyTags = residencyTags;
        this.qualityPrior = qualityPrior;
        this.p50LatencyMs = p50LatencyMs;
        this.tokenEfficiency = tokenEfficiency;
        this.reworkFactor = reworkFactor;
    }
    /** Effective cost per TASK relative to sticker price per token. */
    get taskCostMultiplier() {
        return this.tokenEfficiency * this.reworkFactor;
    }
    quality(workload) {
        return this.qualityPrior[workload] ?? 0.5;
    }
    inputCost(tokens) {
        return (tokens / 1_000_000) * this.inputPerMtok;
    }
    outputCost(tokens) {
        return (tokens / 1_000_000) * this.outputPerMtok;
    }
    cacheReadCost(tokens) {
        return this.inputCost(tokens) * this.cache.readMult;
    }
    cacheWriteCost(tokens, ttl = "5m") {
        return this.inputCost(tokens) * this.cache.writeMult(ttl);
    }
}
// Verified 2026-09-12 against primary docs.
export const ANTHROPIC_CACHE = new CacheSemantics(0.10, 1.25, 2.0, 1024, ["5m", "1h"], true, true, 4, null);
export const ANTHROPIC_CACHE_SMALL = new CacheSemantics(0.10, 1.25, 2.0, 4096, ["5m", "1h"], true, true, 4, null);
export const OPENAI_CACHE_LEGACY = new CacheSemantics(0.50, 1.0, 1.0, 1024, ["5m"], true, false, 0, "prompt_cache_key");
export const OPENAI_CACHE_CURRENT = new CacheSemantics(0.10, 1.0, 1.0, 1024, ["5m"], true, false, 0, "prompt_cache_key");
export const SELF_HOSTED_CACHE = new CacheSemantics(1.0, 1.0, 1.0, 512, [], false, false, 0, null);
/** Bundled defaults. $ figures are illustrative; multipliers are verified. */
export function defaultPriceCards() {
    const strong = { agent: 0.9, chat: 0.92, rag: 0.88, code: 0.9, batch: 0.85, vision: 0.9 };
    const mid = { agent: 0.75, chat: 0.85, rag: 0.78, code: 0.76, batch: 0.75, vision: 0.8 };
    const weak = { agent: 0.55, chat: 0.75, rag: 0.62, code: 0.58, batch: 0.6, vision: 0.45 };
    const cards = [
        new PriceCard("claude-opus-class", 15.0, 75.0, ANTHROPIC_CACHE, 200_000, true, true, DeploymentClass.Api, new Set(), strong, 1200),
        new PriceCard("claude-sonnet-class", 3.0, 15.0, ANTHROPIC_CACHE, 200_000, true, true, DeploymentClass.Api, new Set(), mid, 900),
        new PriceCard("claude-haiku-class", 1.0, 5.0, ANTHROPIC_CACHE_SMALL, 200_000, true, false, DeploymentClass.Api, new Set(), weak, 500),
        new PriceCard("gpt-frontier-class", 5.0, 20.0, OPENAI_CACHE_CURRENT, 200_000, true, true, DeploymentClass.Api, new Set(), strong, 1000),
        new PriceCard("gpt-legacy-class", 2.5, 10.0, OPENAI_CACHE_LEGACY, 200_000, true, false, DeploymentClass.Api, new Set(), mid, 800),
        new PriceCard("gpt-mini-class", 0.75, 3.0, OPENAI_CACHE_CURRENT, 200_000, true, false, DeploymentClass.Api, new Set(), weak, 450),
    ];
    return new Map(cards.map((c) => [c.modelId, c]));
}
/**
 * Mutable registry so prices can be hot-swapped.
 * NOTE the explicit `=== undefined`: an EMPTY map means "start from nothing" —
 * the `cards || defaults` idiom silently resurrected the bundled cards in the
 * Python package once already (the falsy-trap bug in the research log).
 */
export class PriceRegistry {
    cards;
    constructor(cards) {
        this.cards = cards === undefined ? defaultPriceCards() : cards;
    }
    register(card) {
        this.cards.set(card.modelId, card);
    }
    unregister(modelId) {
        this.cards.delete(modelId);
    }
    get(modelId) {
        return this.cards.get(modelId);
    }
    require(modelId) {
        const card = this.cards.get(modelId);
        if (!card)
            throw new Error(`no price card for model ${modelId}`);
        return card;
    }
    all() {
        return [...this.cards.values()];
    }
    ids() {
        return [...this.cards.keys()];
    }
    get size() {
        return this.cards.size;
    }
}
