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
  constructor(
    /** Cache read as a multiple of base input price. */
    public readonly readMult: number = 0.10,
    /** 5-minute cache write multiplier. 1.0 = no write premium (OpenAI). */
    public readonly writeMult5m: number = 1.25,
    /** 1-hour cache write multiplier. */
    public readonly writeMult1h: number = 2.0,
    /** Below this prefix length no cache entry is created at all. */
    public readonly minCacheableTokens: number = 1024,
    /** Which TTL tiers exist. Do not model what the provider lacks. */
    public readonly ttlOptions: readonly string[] = ["5m"],
    public readonly ttlRefreshesOnHit: boolean = true,
    public readonly supportsMixedTtl: boolean = false,
    public readonly maxBreakpoints: number = 4,
    public readonly explicitCacheKeyParam: string | null = null,
  ) {}

  writeMult(ttl: string): number {
    return ttl === "1h" ? this.writeMult1h : this.writeMult5m;
  }

  canCache(prefixTokens: number): boolean {
    return prefixTokens >= this.minCacheableTokens;
  }
}

export class PriceCard {
  constructor(
    public readonly modelId: string,
    public readonly inputPerMtok: number,
    public readonly outputPerMtok: number,
    public readonly cache: CacheSemantics = new CacheSemantics(),
    public readonly contextWindow: number = 200_000,
    public readonly supportsTools: boolean = true,
    public readonly supportsVision: boolean = false,
    public readonly deployment: DeploymentClass = DeploymentClass.Api,
    public readonly residencyTags: ReadonlySet<string> = new Set<string>(),
    /** Quality prior per workload class. Placeholders — calibrate on YOUR traffic. */
    public readonly qualityPrior: Readonly<Record<string, number>> = {},
    public readonly p50LatencyMs: number = 800.0,
    /** Price-per-task factors (Cognition's Fusion finding). */
    public readonly tokenEfficiency: number = 1.0,
    public readonly reworkFactor: number = 1.0,
  ) {}

  /** Effective cost per TASK relative to sticker price per token. */
  get taskCostMultiplier(): number {
    return this.tokenEfficiency * this.reworkFactor;
  }

  quality(workload: string): number {
    return this.qualityPrior[workload] ?? 0.5;
  }

  inputCost(tokens: number): number {
    return (tokens / 1_000_000) * this.inputPerMtok;
  }

  outputCost(tokens: number): number {
    return (tokens / 1_000_000) * this.outputPerMtok;
  }

  cacheReadCost(tokens: number): number {
    return this.inputCost(tokens) * this.cache.readMult;
  }

  cacheWriteCost(tokens: number, ttl: string = "5m"): number {
    return this.inputCost(tokens) * this.cache.writeMult(ttl);
  }
}

// Verified 2026-09-12 against primary docs.
export const ANTHROPIC_CACHE = new CacheSemantics(
  0.10, 1.25, 2.0, 1024, ["5m", "1h"], true, true, 4, null,
);
export const ANTHROPIC_CACHE_SMALL = new CacheSemantics(
  0.10, 1.25, 2.0, 4096, ["5m", "1h"], true, true, 4, null,
);
export const OPENAI_CACHE_LEGACY = new CacheSemantics(
  0.50, 1.0, 1.0, 1024, ["5m"], true, false, 0, "prompt_cache_key",
);
export const OPENAI_CACHE_CURRENT = new CacheSemantics(
  0.10, 1.0, 1.0, 1024, ["5m"], true, false, 0, "prompt_cache_key",
);
export const SELF_HOSTED_CACHE = new CacheSemantics(
  1.0, 1.0, 1.0, 512, [], false, false, 0, null,
);

/** Bundled defaults. $ figures are illustrative; multipliers are verified. */
export function defaultPriceCards(): Map<string, PriceCard> {
  const strong = { agent: 0.9, chat: 0.92, rag: 0.88, code: 0.9, batch: 0.85, vision: 0.9 };
  const mid = { agent: 0.75, chat: 0.85, rag: 0.78, code: 0.76, batch: 0.75, vision: 0.8 };
  const weak = { agent: 0.55, chat: 0.75, rag: 0.62, code: 0.58, batch: 0.6, vision: 0.45 };
  const cards: PriceCard[] = [
    new PriceCard("claude-opus-class", 15.0, 75.0, ANTHROPIC_CACHE, 200_000, true, true,
      DeploymentClass.Api, new Set(), strong, 1200),
    new PriceCard("claude-sonnet-class", 3.0, 15.0, ANTHROPIC_CACHE, 200_000, true, true,
      DeploymentClass.Api, new Set(), mid, 900),
    new PriceCard("claude-haiku-class", 1.0, 5.0, ANTHROPIC_CACHE_SMALL, 200_000, true, false,
      DeploymentClass.Api, new Set(), weak, 500),
    new PriceCard("gpt-frontier-class", 5.0, 20.0, OPENAI_CACHE_CURRENT, 200_000, true, true,
      DeploymentClass.Api, new Set(), strong, 1000),
    new PriceCard("gpt-legacy-class", 2.5, 10.0, OPENAI_CACHE_LEGACY, 200_000, true, false,
      DeploymentClass.Api, new Set(), mid, 800),
    new PriceCard("gpt-mini-class", 0.75, 3.0, OPENAI_CACHE_CURRENT, 200_000, true, false,
      DeploymentClass.Api, new Set(), weak, 450),
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
  private readonly cards: Map<string, PriceCard>;

  constructor(cards?: Map<string, PriceCard>) {
    this.cards = cards === undefined ? defaultPriceCards() : cards;
  }

  register(card: PriceCard): void {
    this.cards.set(card.modelId, card);
  }

  unregister(modelId: string): void {
    this.cards.delete(modelId);
  }

  get(modelId: string): PriceCard | undefined {
    return this.cards.get(modelId);
  }

  require(modelId: string): PriceCard {
    const card = this.cards.get(modelId);
    if (!card) throw new Error(`no price card for model ${modelId}`);
    return card;
  }

  all(): PriceCard[] {
    return [...this.cards.values()];
  }

  ids(): string[] {
    return [...this.cards.keys()];
  }

  get size(): number {
    return this.cards.size;
  }
}
