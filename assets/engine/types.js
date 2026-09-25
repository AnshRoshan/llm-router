/**
 * Core types — the vocabulary of the router.
 *
 * This is the TypeScript port of the Python decision layer
 * (`llmrouter/src/llmrouter`). Formulas, thresholds and defaults are kept
 * identical on purpose: the two packages must make the SAME decision for the
 * SAME request and price cards. Where a hash is involved (fingerprints, sticky
 * keys) the digest differs between runtimes by design — a sticky key is only
 * ever compared within one process.
 */
import { hash16 } from "./hash.js";
// --------------------------------------------------------------------------- //
// Enums
// --------------------------------------------------------------------------- //
export var WorkloadClass;
(function (WorkloadClass) {
    WorkloadClass["Agent"] = "agent";
    WorkloadClass["Chat"] = "chat";
    WorkloadClass["Rag"] = "rag";
    WorkloadClass["Batch"] = "batch";
    WorkloadClass["Code"] = "code";
    WorkloadClass["Vision"] = "vision";
    WorkloadClass["Unknown"] = "unknown";
})(WorkloadClass || (WorkloadClass = {}));
export var WorkloadSource;
(function (WorkloadSource) {
    WorkloadSource["ExplicitTag"] = "explicit_tag";
    WorkloadSource["AgentProfile"] = "agent_profile";
    WorkloadSource["Fingerprint"] = "fingerprint";
    WorkloadSource["Structural"] = "structural";
    WorkloadSource["Classifier"] = "classifier";
    WorkloadSource["Default"] = "default";
})(WorkloadSource || (WorkloadSource = {}));
export var DeploymentClass;
(function (DeploymentClass) {
    DeploymentClass["Api"] = "api";
    DeploymentClass["SelfHosted"] = "self_hosted";
    DeploymentClass["Serverless"] = "serverless";
    DeploymentClass["Edge"] = "edge";
})(DeploymentClass || (DeploymentClass = {}));
export var SegmentKind;
(function (SegmentKind) {
    SegmentKind["Tools"] = "tools";
    SegmentKind["System"] = "system";
    SegmentKind["Retrieved"] = "retrieved";
    SegmentKind["History"] = "history";
    SegmentKind["User"] = "user";
})(SegmentKind || (SegmentKind = {}));
export class Message {
    role;
    content;
    constructor(role, content = "") {
        this.role = role;
        this.content = content;
    }
    /** Text of the message: string content, or the text/tool_result blocks. */
    get text() {
        if (typeof this.content === "string")
            return this.content;
        const out = [];
        for (const block of this.content) {
            if (block.type === "text")
                out.push(String(block.text ?? ""));
            else if (block.type === "tool_result")
                out.push(String(block.content ?? ""));
            else
                out.push(String(block));
        }
        return out.join("\n");
    }
    hasBlockType(blockType) {
        if (typeof this.content === "string")
            return false;
        return this.content.some((b) => b.type === blockType);
    }
}
export const DEFAULT_CONSTRAINTS = {};
export class RoutingRequest {
    messages;
    workload;
    sessionId;
    agentProfile;
    tools;
    retrieved;
    expectedTurns;
    expectedGapS;
    expectedGenerationS;
    clientId;
    constraints;
    tags;
    constructor(messages, workload = null, sessionId = null, agentProfile = null, tools = [], retrieved = [], tags = new Set(), constraints = DEFAULT_CONSTRAINTS, expectedTurns = null, expectedGapS = null, expectedGenerationS = null, clientId = null) {
        this.messages = messages;
        this.workload = workload;
        this.sessionId = sessionId;
        this.agentProfile = agentProfile;
        this.tools = tools;
        this.retrieved = retrieved;
        this.expectedTurns = expectedTurns;
        this.expectedGapS = expectedGapS;
        this.expectedGenerationS = expectedGenerationS;
        this.clientId = clientId;
        this.tags =
            tags instanceof Set ? tags : new Set(tags);
        this.constraints = constraints;
    }
    /** Number of prior user turns. 1 == first turn of a conversation. */
    get turnIndex() {
        return this.messages.filter((m) => m.role === "user").length;
    }
    get isFirstTurn() {
        return this.turnIndex <= 1;
    }
    firstUserText() {
        for (const m of this.messages) {
            if (m.role === "user")
                return m.text;
        }
        return "";
    }
    systemText() {
        return this.messages
            .filter((m) => m.role === "system")
            .map((m) => m.text)
            .join("\n");
    }
    hasToolResults() {
        return this.messages.some((m) => (m.role === "tool" || m.role === "user") && m.hasBlockType("tool_result"));
    }
    hasImages() {
        return this.messages.some((m) => m.hasBlockType("image") || m.hasBlockType("image_url"));
    }
    /** Hash of the stable part of the prompt: system text + tool schemas. */
    promptFingerprint() {
        return hash16(`${this.systemText()}\x00${stableJson(this.tools)}`);
    }
    /**
     * Hash of the request fields that are part of a provider cache key but
     * OUTSIDE the prefix text: tool schemas and image presence (Rule F).
     */
    invalidatorsFingerprint() {
        return hash16(`${stableJson(this.tools)}\x00${this.hasImages() ? 1 : 0}`);
    }
    /** Session key WITHOUT requiring the caller to send a session id. */
    stickyKey() {
        if (this.sessionId)
            return `sid:${this.sessionId}`;
        const first = this.firstUserText();
        if (first)
            return `ctx:${hash16(first)}`;
        return `fp:${this.promptFingerprint()}:${this.clientId ?? "anon"}`;
    }
    /** Wire form for the HTTP decide endpoint — snake_case, mirrors Python
     *  RoutingRequest.from_payload exactly. */
    static fromPayload(obj) {
        const rawMessages = obj.messages;
        if (!Array.isArray(rawMessages) || rawMessages.length === 0) {
            throw new Error("payload needs a non-empty 'messages' array");
        }
        const messages = rawMessages.map((m) => new Message(String(m.role ?? "user"), m.content ?? ""));
        const workload = workloadFromName(obj.workload);
        const c = obj.constraints ?? {};
        const constraints = {
            maxCostUsd: c.max_cost_usd ?? null,
            maxLatencyMs: c.max_latency_ms ?? null,
            budgetRemainingUsd: c.budget_remaining_usd ?? null,
            residencyTags: new Set((c.residency_tags ?? []).map(String)),
            requireTools: !!c.require_tools,
            allowedModels: new Set((c.allowed_models ?? []).map(String)),
        };
        return new RoutingRequest(messages, workload, obj.session_id ?? null, obj.agent_profile ?? null, (obj.tools ?? []).map((t) => t), (obj.retrieved ?? []).map(String), new Set((obj.tags ?? []).map(String)), constraints, obj.expected_turns ?? null, obj.expected_gap_s ?? null, obj.expected_generation_s ?? null, obj.client_id ?? null);
    }
}
function workloadFromName(value) {
    if (typeof value !== "string" || value.length === 0)
        return null;
    const hit = Object.values(WorkloadClass).includes(value)
        ? value
        : null;
    if (value.length > 0 && hit === null) {
        throw new Error(`unknown workload ${JSON.stringify(value)}`);
    }
    return hit;
}
function stableJson(value) {
    return JSON.stringify(value, (_k, v) => {
        if (v && typeof v === "object" && !Array.isArray(v)) {
            return Object.fromEntries(Object.entries(v).sort(([a], [b]) => a < b ? -1 : a > b ? 1 : 0));
        }
        return v;
    });
}
export class Decision {
    modelId;
    backendId;
    workload;
    workloadSource;
    workloadConfidence;
    stickyKey;
    sessionKey;
    isFirstTurn;
    reusedSession;
    switched;
    stranded;
    ttlBySegment;
    candidates;
    reasons;
    constructor(modelId, backendId, workload, workloadSource, workloadConfidence, stickyKey, sessionKey, isFirstTurn, reusedSession, switched, stranded, ttlBySegment, candidates, reasons) {
        this.modelId = modelId;
        this.backendId = backendId;
        this.workload = workload;
        this.workloadSource = workloadSource;
        this.workloadConfidence = workloadConfidence;
        this.stickyKey = stickyKey;
        this.sessionKey = sessionKey;
        this.isFirstTurn = isFirstTurn;
        this.reusedSession = reusedSession;
        this.switched = switched;
        this.stranded = stranded;
        this.ttlBySegment = ttlBySegment;
        this.candidates = candidates;
        this.reasons = reasons;
    }
    /** The full auditable decision — every decision reports why. */
    explain() {
        const head = `model=${this.modelId} backend=${this.backendId} ` +
            `workload=${this.workload}(src=${this.workloadSource}, ` +
            `conf=${this.workloadConfidence.toFixed(2)}) ` +
            `sticky=${this.reusedSession} switched=${this.switched} ` +
            `stranded=${this.stranded}`;
        const ttl = [...this.ttlBySegment.entries()].map(([k, v]) => `${k}=${v}`).join(", ") ||
            "n/a";
        const lines = [head, `  ttl: ${ttl}`];
        for (const c of this.candidates.slice(0, 5)) {
            lines.push(`  ${c.modelId}@${c.backendId} score=${c.score >= 0 ? "+" : ""}${c.score.toFixed(4)} ` +
                `q=${c.quality.toFixed(3)} cost=$${c.costUsd.toFixed(5)} ` +
                `lat=${c.latencyMs.toFixed(0)}ms aff=${c.affinity.toFixed(2)} ` +
                `switch=$${c.switchCostUsd.toFixed(5)}`);
        }
        for (const r of this.reasons)
            lines.push(`  · ${r}`);
        return lines.join("\n");
    }
    /** Wire form for the HTTP decide endpoint — mirrors Python
     *  Decision.to_payload (snake_case, same candidate fields). */
    toPayload() {
        return {
            model_id: this.modelId,
            backend_id: this.backendId,
            workload: this.workload,
            workload_source: this.workloadSource,
            workload_confidence: this.workloadConfidence,
            sticky_key: this.stickyKey,
            session_key: this.sessionKey,
            is_first_turn: this.isFirstTurn,
            reused_session: this.reusedSession,
            switched: this.switched,
            stranded: this.stranded,
            ttl_by_segment: Object.fromEntries(this.ttlBySegment),
            candidates: this.candidates.map((c) => ({
                model_id: c.modelId, backend_id: c.backendId,
                score: c.score, quality: c.quality,
                cost_usd: c.costUsd, latency_ms: c.latencyMs,
                affinity: c.affinity, switch_cost_usd: c.switchCostUsd,
                reasons: [...c.reasons],
            })),
            reasons: [...this.reasons],
        };
    }
}
