/**
 * The decision layer: pure, no I/O, deterministic.
 *
 *   score(model, backend) = w_q · quality
 *                         − w_c · cost      (× pair multiplier × pacing)
 *                         − w_l · latency
 *                         − w_s · switch_cost   ← the term nobody has
 *                         + w_a · affinity_bonus
 *   subject to: residency ∧ budget ∧ context_window ∧ tool_support
 *
 * Port of the Python PolicyEngine; the two runtimes must agree on decisions.
 */
import {
  affinityPrior,
  pickBackend,
  SessionPolicy,
  SessionStore,
} from "./affinity.js";
import { BackendRegistry } from "./backends.js";
import {
  chooseTtlBySegment,
  evaluateSwitch,
} from "./economics.js";
import { PriceCard, PriceRegistry } from "./pricing.js";
import {
  GAP_PRIOR,
  TURNS_PRIOR,
  WorkloadResolver,
} from "./signals.js";
import {
  Candidate,
  Decision,
  RoutingRequest,
  SegmentKind,
  WorkloadClass,
} from "./types.js";

export interface Weights {
  quality: number;
  cost: number;
  latency: number;
  switchCost: number;
  affinity: number;
  /** Hard floor: never route below this estimated quality for the workload. */
  minQuality: number;
  /** Soft preference: accept up to this much extra cost for a quality gain. */
  qualityCostTradeoff: number;
}

export const DEFAULT_WEIGHTS: Weights = {
  quality: 1.0,
  cost: 1.0,
  latency: 0.0002,
  switchCost: 1.0,
  affinity: 0.10,
  minQuality: 0.0,
  qualityCostTradeoff: 1.0,
};

export interface RouterConfig {
  weights: Weights;
  /** Gate cache-affinity logic below this prefix length. */
  minPrefixTokensForAffinity: number;
  estTokensPerChar: number;
  /** Reserved output tokens for the pre-call context-window check. */
  contextHeadroomTokens: number;
}

export const DEFAULT_CONFIG: RouterConfig = {
  weights: DEFAULT_WEIGHTS,
  minPrefixTokensForAffinity: 500,
  estTokensPerChar: 0.25,
  contextHeadroomTokens: 4096,
};

export interface ScoredCandidate {
  candidate: Candidate;
  card: PriceCard;
  backendId: string;
  backendCostFactor: number;
}

export class PolicyEngine {
  readonly sessions: SessionStore;

  constructor(
    public readonly prices: PriceRegistry,
    public readonly backends: BackendRegistry,
    public readonly resolver: WorkloadResolver = new WorkloadResolver(),
    public readonly config: RouterConfig = DEFAULT_CONFIG,
    sessions: SessionStore | null = null,
  ) {
    this.sessions = sessions ?? new SessionStore();
  }

  // ------------------------------------------------------------------ //
  // feature estimation
  // ------------------------------------------------------------------ //
  estimatePrefixTokens(req: RoutingRequest): number {
    let chars = 0;
    for (const m of req.messages) {
      if (m.role === "system") chars += m.text.length;
      else if (m.role !== "user" || !req.isFirstTurn) chars += m.text.length;
    }
    for (const t of req.tools) chars += JSON.stringify(t).length;
    for (const r of req.retrieved) chars += r.length;
    const est = Math.trunc(chars * this.config.estTokensPerChar);
    return est > 0 ? est : 0;
  }

  /** Full prompt INCLUDING the current user turn — the window check uses this. */
  estimateTotalTokens(req: RoutingRequest): number {
    let chars = 0;
    for (const m of req.messages) chars += m.text.length;
    for (const t of req.tools) chars += JSON.stringify(t).length;
    for (const r of req.retrieved) chars += r.length;
    const est = Math.trunc(chars * this.config.estTokensPerChar);
    return Math.max(est, this.estimatePrefixTokens(req));
  }

  estimateTurnCost(
    card: PriceCard, prefixTokens: number, hitRate: number, outputTokens = 500,
  ): number {
    const cached = Math.trunc(prefixTokens * hitRate);
    const uncached = prefixTokens - cached;
    const sticker =
      card.cacheReadCost(cached) + card.inputCost(uncached) + card.outputCost(outputTokens);
    return sticker * card.taskCostMultiplier;
  }

  // ------------------------------------------------------------------ //
  // the decision
  // ------------------------------------------------------------------ //
  decide(req: RoutingRequest, now: number = nowMonotonic()): Decision {
    const reasons: string[] = [];
    const w = this.config.weights;

    // ---- 1. workload class (Path A then Path B) -------------------------
    const signal = this.resolver.resolve(req);
    const workload = signal.workload;
    reasons.push(`workload=${workload} via ${signal.source}: ${signal.reason}`);

    const allowed = this.allowedModels(req);

    // ---- 2. session shape ------------------------------------------------
    const prefixTokens = this.estimatePrefixTokens(req);
    const totalTokens = this.estimateTotalTokens(req);
    const gapS = req.expectedGapS ?? GAP_PRIOR[workload] ?? 60.0;
    const genS = req.expectedGenerationS ?? 0.0;
    const turns = req.expectedTurns ?? TURNS_PRIOR[workload] ?? 5;
    const stickyKey = req.stickyKey();

    // ---- 3. affinity ------------------------------------------------------
    const affinityEnabled = prefixTokens >= this.config.minPrefixTokensForAffinity;
    let prior = affinityEnabled ? affinityPrior(workload, gapS) : 0.0;
    const session = this.sessions.get(stickyKey, now);
    const pinnedModel = session?.pinnedModel ?? null;
    const pinnedBackend = session?.pinnedBackend ?? null;
    const turnsRemaining = Math.max(1, turns - (session?.turnsSeen ?? 0));

    // ---- 3b. Rule F: cache-invalidating fields ----------------------------
    const invFp = req.invalidatorsFingerprint();
    let ruleFInvalidated = false;
    if (session) {
      if (session.invalidatorsHash === null) {
        session.invalidatorsHash = invFp;
      } else if (session.invalidatorsHash !== invFp) {
        ruleFInvalidated = true;
        session.invalidatorsHash = invFp;
        reasons.push(
          "Rule F: cache-invalidating request fields changed (tool " +
            "schema / image usage) — all candidates scored cold",
        );
      }
    }

    // ---- 4. score every (model, backend) pair ------------------------------
    const scored: ScoredCandidate[] = [];
    const overBudget: ScoredCandidate[] = [];
    const cap = req.constraints.maxCostUsd ?? null;

    for (const card of this.prices.all()) {
      if (!allowed.has(card.modelId)) continue;
      if (!this.passesHardConstraints(card, req, workload, totalTokens)) continue;

      for (const backend of this.backends.availableFor(card.modelId, now)) {
        let hitRate = prior;
        if (session && card.modelId === pinnedModel) {
          hitRate = session.hitRate(prior);
        }
        if (ruleFInvalidated) hitRate = 0.0;

        const turnCost = this.estimateTurnCost(card, prefixTokens, hitRate);
        const cost = turnCost * backend.costFactor;

        const health = this.backends.health(backend.backendId);
        const latency =
          (health.ewmaLatencyMs || card.p50LatencyMs) +
          health.inFlight * 5.0 +
          backend.coldStartRisk * 2000.0;

        const isWarm =
          !!session &&
          card.modelId === pinnedModel &&
          backend.backendId === pinnedBackend;
        const affinityBonus = ruleFInvalidated ? 0.0 : isWarm ? hitRate : 0.0;

        let switchCost = 0.0;
        if (session && !isWarm && affinityEnabled && !ruleFInvalidated) {
          if (workload !== WorkloadClass.Batch) {
            switchCost = card.cacheWriteCost(prefixTokens, "5m");
          }
        }

        const quality = card.quality(workload);
        const overCap = cap !== null && cost > cap;

        // Closed-loop budget pacing on projected session spend.
        let budgetPenalty = 0.0;
        let projected = 0.0;
        const remaining = req.constraints.budgetRemainingUsd ?? null;
        if (remaining !== null && !overCap) {
          projected = cost * Math.max(1, turnsRemaining);
          if (projected > remaining) budgetPenalty = projected - remaining;
        }

        const candReasons = [`hit_rate=${hitRate.toFixed(2)}`, `warm=${isWarm}`];
        if (budgetPenalty > 0) {
          candReasons.push(
            `budget pacing: projected $${projected.toFixed(4)} exceeds ` +
              `$${remaining!.toFixed(4)} remaining`,
          );
        }

        const score =
          w.quality * quality -
          w.cost * cost * w.qualityCostTradeoff -
          w.cost * budgetPenalty -
          w.latency * latency -
          w.switchCost * switchCost +
          w.affinity * affinityBonus;

        const candidate: Candidate = {
          modelId: card.modelId,
          backendId: backend.backendId,
          score,
          quality,
          costUsd: cost,
          latencyMs: latency,
          affinity: affinityBonus,
          switchCostUsd: switchCost,
          reasons: candReasons,
        };
        const entry: ScoredCandidate = {
          candidate, card, backendId: backend.backendId,
          backendCostFactor: backend.costFactor,
        };
        if (overCap) overBudget.push(entry);
        else scored.push(entry);
      }
    }

    let budgetFallback = false;
    if (scored.length === 0 && overBudget.length > 0) {
      // A cap nothing meets is a preference, not an outage: cheapest wins.
      scored.push(...overBudget);
      budgetFallback = true;
      const cheapest = scored.reduce((a, b) =>
        a.candidate.costUsd <= b.candidate.costUsd ? a : b);
      reasons.push(
        `budget cap $${cap!.toFixed(4)} excludes every candidate; ` +
          `chose the cheapest (${cheapest.candidate.modelId})`,
      );
    }
    if (scored.length === 0) {
      throw new Error(
        `every candidate was filtered out for workload=${workload}; ` +
          "check hard constraints (residency, context window, tool support) " +
          "and backend health",
      );
    }
    if (budgetFallback) {
      scored.sort((a, b) => a.candidate.costUsd - b.candidate.costUsd);
    } else {
      scored.sort((a, b) => b.candidate.score - a.candidate.score);
    }
    const best = scored[0];

    // ---- 5. stickiness / migration ----------------------------------------
    let switched = false;
    let stranded = false;
    let chosenModel = best.candidate.modelId;
    let chosenBackend = best.candidate.backendId;

    if (session) {
      const pinnedAvailable =
        this.backends.availableFor(session.pinnedModel, now).length > 0;
      const wantsSwitch = chosenModel !== session.pinnedModel;
      let forced = false;
      if (wantsSwitch && !pinnedAvailable) {
        forced = true;
        switched = true;
        session.switchedAt = now;
        reasons.push(
          `forced failover: pinned ${session.pinnedModel} has no healthy ` +
            "backend (not a cost-driven switch)",
        );
      }
      if (wantsSwitch && !forced) {
        const cur = this.prices.get(session.pinnedModel);
        const tgt = best.card;
        if (cur && tgt) {
          const verdict = evaluateSwitch(
            cur, tgt, prefixTokens, turnsRemaining, session.alreadySwitched,
          );
          if (verdict.shouldSwitch) {
            switched = true;
            session.switchedAt = now;
            reasons.push(`migrating: ${verdict.reason}`);
          } else {
            stranded = !verdict.reason.startsWith("no-oscillation");
            if (stranded) session.strandedCount += 1;
            chosenModel = session.pinnedModel;
            const healthyIds = this.backends
              .availableFor(chosenModel, now)
              .map((b) => b.backendId);
            chosenBackend =
              pickBackend(stickyKey, healthyIds, session.pinnedBackend) ??
              session.pinnedBackend;
            reasons.push(`staying pinned: ${verdict.reason}`);
          }
        }
      } else if (!wantsSwitch || forced) {
        // Same model: keep the backend sticky, overflow accepts the miss.
        const healthyIds = this.backends
          .availableFor(chosenModel, now)
          .map((b) => b.backendId);
        chosenBackend =
          pickBackend(stickyKey, healthyIds, session.pinnedBackend) ??
          chosenBackend;
      }
    }

    // ---- 6. TTL policy per breakpoint --------------------------------------
    const segmentTokens = this.segmentTokens(req, prefixTokens);
    const sem = best.card.cache;
    const ttlBySegment = chooseTtlBySegment(sem, segmentTokens, gapS, genS);
    if (
      sem.supportsMixedTtl &&
      [...ttlBySegment.values()].includes("1h")
    ) {
      reasons.push(
        `mixed TTL: ${[...ttlBySegment.entries()].map(([k, v]) => `${k}=${v}`).join(", ")}`,
      );
    }

    // ---- 7. build the decision ----------------------------------------------
    const ordered = scored.map((s) => s.candidate);
    let final =
      ordered.find(
        (c) => c.modelId === chosenModel && c.backendId === chosenBackend,
      ) ?? null;
    if (!final) {
      final = ordered.find((c) => c.modelId === chosenModel) ?? null;
      if (!final) {
        final = ordered[0];
        chosenModel = final.modelId;
        if (session && session.pinnedModel !== chosenModel) {
          switched = true;
          session.switchedAt = now;
          reasons.push(
            `pinned ${session.pinnedModel} produced no viable candidate; ` +
              `fell back to ${chosenModel}`,
          );
        }
      }
      chosenBackend = final.backendId;
    }

    return new Decision(
      final.modelId,
      final.backendId,
      workload,
      signal.source,
      signal.confidence,
      stickyKey,
      stickyKey,
      req.isFirstTurn,
      session !== null,
      switched,
      stranded,
      ttlBySegment,
      ordered,
      reasons,
    );
  }

  // ------------------------------------------------------------------ //
  private allowedModels(req: RoutingRequest): Set<string> {
    if (req.agentProfile) {
      const profile = this.resolver.profiles.get(req.agentProfile);
      if (profile?.allowedModels && profile.allowedModels.length > 0) {
        return new Set(
          profile.allowedModels.filter((m) => this.prices.get(m) !== undefined),
        );
      }
    }
    return new Set(this.prices.ids());
  }

  private passesHardConstraints(
    card: PriceCard, req: RoutingRequest, workload: WorkloadClass, promptTokens: number,
  ): boolean {
    const quality = card.quality(workload);
    if (quality < 0) return false;
    if (this.config.weights.minQuality > 0 && quality < this.config.weights.minQuality) {
      return false;
    }
    if (req.constraints.requireTools && !card.supportsTools) return false;
    if (!card.supportsVision && req.hasImages()) return false;
    const tags = req.constraints.residencyTags;
    if (tags && tags.size > 0) {
      for (const t of tags) {
        if (!card.residencyTags.has(t)) return false;
      }
    }
    // Pre-call context validation (full prompt + headroom for the response).
    if (
      card.contextWindow &&
      promptTokens + this.config.contextHeadroomTokens > card.contextWindow
    ) {
      return false;
    }
    return true;
  }

  segmentTokens(
    req: RoutingRequest, prefixTokens: number,
  ): Map<SegmentKind, number> {
    const toks = (s: string) => Math.trunc(s.length * this.config.estTokensPerChar);
    const sysT = toks(req.systemText());
    const toolsT = req.tools.reduce((a, t) => a + toks(JSON.stringify(t)), 0);
    const retrT = req.retrieved.reduce((a, r) => a + toks(r), 0);
    const histT = Math.max(0, prefixTokens - sysT - toolsT - retrT);

    const out = new Map<SegmentKind, number>();
    if (toolsT) out.set(SegmentKind.Tools, toolsT);
    if (sysT) out.set(SegmentKind.System, sysT);
    if (retrT) out.set(SegmentKind.Retrieved, retrT);
    if (histT) out.set(SegmentKind.History, histT);
    return out;
  }
}

function nowMonotonic(): number {
  return Number(process.hrtime.bigint()) / 1e9;
}
