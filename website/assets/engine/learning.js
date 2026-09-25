import { CODE_MARKERS } from "./signals.js";
import { WorkloadClass } from "./types.js";
export const CHECKPOINT_FORMAT = "llmrouter-quality-v1";
/** FIXED ORDER — mirrors Python FEATURE_NAMES. Checked against the checkpoint. */
export const FEATURE_NAMES = [
    "wl_agent", "wl_chat", "wl_rag", "wl_batch", "wl_code", "wl_vision",
    "has_tools", "has_tool_results", "has_images",
    "n_tools", "tool_schema_size", "prefix_size",
    "turn_depth", "is_first_turn", "code_markers",
    "retrieved_count", "user_text_size",
];
export const N_FEATURES = FEATURE_NAMES.length;
const WORKLOAD_ONEHOT = [
    WorkloadClass.Agent, WorkloadClass.Chat, WorkloadClass.Rag,
    WorkloadClass.Batch, WorkloadClass.Code, WorkloadClass.Vision,
];
function logNorm(chars, divisor) {
    return Math.min(1.0, Math.log10(1.0 + chars) / divisor);
}
export function extractFeatures(req, workload) {
    const onehot = WORKLOAD_ONEHOT.map((w) => (w === workload ? 1.0 : 0.0));
    let toolChars = 0;
    for (const t of req.tools)
        toolChars += JSON.stringify(t).length;
    let prefixChars = 0;
    for (const m of req.messages) {
        if (m.role === "system")
            prefixChars += m.text.length;
        else if (m.role !== "user" || !req.isFirstTurn)
            prefixChars += m.text.length;
    }
    for (const r of req.retrieved)
        prefixChars += r.length;
    const userText = req.firstUserText();
    const codeHits = CODE_MARKERS.filter((mk) => userText.includes(mk)).length;
    return [
        ...onehot,
        req.tools.length ? 1.0 : 0.0,
        req.hasToolResults() ? 1.0 : 0.0,
        req.hasImages() ? 1.0 : 0.0,
        Math.min(1.0, req.tools.length / 8.0),
        logNorm(toolChars, 5.0),
        logNorm(prefixChars + toolChars, 5.0),
        Math.min(1.0, req.turnIndex / 25.0),
        req.isFirstTurn ? 1.0 : 0.0,
        Math.min(1.0, codeHits / 6.0),
        Math.min(1.0, req.retrieved.length / 4.0),
        logNorm(userText.length, 4.0),
    ];
}
export class QualityModel {
    heads;
    shrinkageK;
    constructor(heads, shrinkageK = 20.0) {
        this.heads = heads;
        this.shrinkageK = shrinkageK;
    }
    /** b + w·x, accumulated in feature order (same float64 ops as Python). */
    static raw(head, x) {
        let acc = head.bias;
        for (let i = 0; i < x.length; i++)
            acc += head.weights[i] * x[i];
        return acc;
    }
    /** Blended quality: prior + shrunk learned delta. Falls back to the prior
     *  when the model has no head or the prior says "unsupported" (q < 0). */
    quality(card, workload, x) {
        const base = card.quality(workload);
        const head = this.heads.get(card.modelId);
        if (head === undefined || base < 0)
            return base;
        const lam = head.samples / (head.samples + this.shrinkageK);
        return Math.min(1.0, Math.max(0.0, base + lam * QualityModel.raw(head, x)));
    }
    coverage(modelIds) {
        let n = 0;
        for (const m of modelIds)
            if (this.heads.has(m))
                n += 1;
        return [n, modelIds.length];
    }
    static fromJson(obj) {
        if (obj.format !== CHECKPOINT_FORMAT) {
            throw new Error(`unsupported checkpoint format ${String(obj.format)}, ` +
                `expected ${CHECKPOINT_FORMAT}`);
        }
        const names = obj.feature_names;
        if (!names || names.length !== FEATURE_NAMES.length ||
            names.some((n, i) => n !== FEATURE_NAMES[i])) {
            throw new Error("checkpoint feature set does not match this engine version — " +
                "retrain it with `llmrouter train`");
        }
        const heads = new Map();
        for (const [mid, h] of Object.entries((obj.heads ?? {}))) {
            if (h.weights.length !== N_FEATURES) {
                throw new Error(`head ${mid} has ${h.weights.length} weights, expected ${N_FEATURES}`);
            }
            heads.set(mid, {
                bias: Number(h.bias),
                weights: h.weights.map(Number),
                samples: Number(h.samples),
            });
        }
        return new QualityModel(heads, Number(obj.shrinkage_k ?? 20.0));
    }
}
