/**
 * `llmrouter serve` — the decision engine as an HTTP decide-sidecar.
 *
 *   POST /decide   {RoutingRequest payload}  -> {Decision payload}
 *   GET  /stats    session/backend telemetry
 *   GET  /healthz  liveness
 *
 * Wire-compatible with the Python `llmrouter serve` (same snake_case payloads)
 * and with `llmrouter-state-v1` snapshots via store.ts, so a LiteLLM/Bifrost/
 * custom gateway can point at either runtime.
 *
 * Feedback loop across the HTTP boundary: include
 *   "observed_usage": {cache_read_tokens, cache_write_tokens, ...}
 * with the NEXT request — the counters the gateway saw from the provider —
 * and the learned hit rate for this session is updated before deciding.
 */
import { createServer, type IncomingMessage, type Server, type ServerResponse } from "node:http";
import { readFileSync, renameSync, writeFileSync } from "node:fs";
import { PolicyEngine, DEFAULT_CONFIG, type RouterConfig } from "./policy.js";
import { BackendRegistry, BackendSpec } from "./backends.js";
import { QualityModel } from "./learning.js";
import { PriceRegistry } from "./pricing.js";
import { WorkloadResolver } from "./signals.js";
import { restoreSnapshot, takeSnapshot, type Snapshot } from "./store.js";
import { DeploymentClass, RoutingRequest } from "./types.js";

const MAX_BODY_BYTES = 10 * 1024 * 1024;

export interface Sidecar {
  server: Server;
  engine: PolicyEngine;
  statePath: string | null;
}

export function buildEngine(
  qualityModel: QualityModel | null = null, config?: RouterConfig,
): PolicyEngine {
  const prices = new PriceRegistry();
  const ids = prices.ids();
  const backends = new BackendRegistry([
    new BackendSpec("anthropic", {
      deployment: DeploymentClass.Api,
      models: ids.filter((m) => m.startsWith("claude")),
      priority: 0,
    }),
    new BackendSpec("openai", {
      deployment: DeploymentClass.Api,
      models: ids.filter((m) => m.startsWith("gpt")),
      priority: 1,
    }),
  ]);
  return new PolicyEngine(
    prices, backends, new WorkloadResolver(), config ?? DEFAULT_CONFIG,
    null, qualityModel,
  );
}

function readBody(req: IncomingMessage): Promise<string> {
  return new Promise((resolve, reject) => {
    const chunks: Buffer[] = [];
    let size = 0;
    req.on("data", (c: Buffer) => {
      size += c.length;
      if (size > MAX_BODY_BYTES) {
        reject(new Error("body too large"));
        req.destroy();
        return;
      }
      chunks.push(c);
    });
    req.on("end", () => resolve(Buffer.concat(chunks).toString("utf8")));
    req.on("error", reject);
  });
}

function send(res: ServerResponse, status: number, obj: unknown): void {
  const body = JSON.stringify(obj);
  res.writeHead(status, { "content-type": "application/json" });
  res.end(body);
}

export function createDecideServer(options: {
  host?: string;
  port?: number;
  token?: string | null;
  qualityModelPath?: string | null;
  statePath?: string | null;
  engine?: PolicyEngine;
} = {}): Sidecar {
  const qualityModel = options.qualityModelPath
    ? QualityModel.fromJson(JSON.parse(readFileSync(options.qualityModelPath, "utf8")))
    : null;
  const engine = options.engine ?? buildEngine(qualityModel);

  let sidecar: Sidecar; // declared before the handler closes over it
  const handler = async (req: IncomingMessage, res: ServerResponse) => {
    if (options.token) {
      const got = String(req.headers.authorization ?? "");
      if (got !== `Bearer ${options.token}`) {
        return send(res, 401, { error: "unauthorized" });
      }
    }
    const path = (req.url ?? "").split("?")[0];
    if (req.method === "GET" && path === "/healthz") {
      return send(res, 200, { ok: true, engine: "llmrouter" });
    }
    if (req.method === "GET" && path === "/stats") {
      const backends: Record<string, unknown> = {};
      for (const h of engine.backends.allHealth()) {
        backends[h.backendId] = {
          in_flight: h.inFlight, error_rate: h.errorRate,
          ewma_latency_ms: h.ewmaLatencyMs,
          consecutive_failures: h.consecutiveFailures,
          circuit_open: h.openUntil ? 1 : 0,
        };
      }
      return send(res, 200, {
        sessions: engine.sessions.stats(), backends,
        fingerprints: engine.resolver.fingerprints.cardinalityByClient(),
      });
    }
    if (req.method === "POST" && path === "/decide") {
      let payload: Record<string, any>;
      try {
        payload = JSON.parse(await readBody(req));
        if (typeof payload !== "object" || payload === null || Array.isArray(payload)) {
          throw new Error("body must be a JSON object");
        }
      } catch (err) {
        return send(res, 400, {
          error: `invalid JSON: ${err instanceof Error ? err.message : err}`,
        });
      }
      let decided: RoutingRequest;
      try {
        decided = RoutingRequest.fromPayload(payload);
      } catch (err) {
        return send(res, 400, { error: err instanceof Error ? err.message : err });
      }
      // Fold the previous turn's provider counters in FIRST, so this decision
      // already sees the learned hit rate (same order as the Python sidecar).
      const usage = payload.observed_usage;
      if (usage) {
        const sess = engine.sessions.peek(decided.stickyKey());
        if (sess) {
          sess.recordUsage(
            Number(usage.cache_read_tokens ?? 0),
            Number(usage.cache_write_tokens ?? 0),
          );
          engine.sessions.put(sess);
        }
      }
      try {
        const decision = engine.decide(decided);
        engine.recordDecision(decided, decision, usage ? {
          cacheReadTokens: Number(usage.cache_read_tokens ?? 0),
          cacheWriteTokens: Number(usage.cache_write_tokens ?? 0),
        } : undefined);
        if (sidecar?.statePath) {
          saveSnapshotFile(sidecar.statePath, sidecar.engine);
        }
        return send(res, 200, decision.toPayload());
      } catch (err) {
        return send(res, 422, {
          error: err instanceof Error ? err.message : String(err),
        });
      }
    }
    return send(res, 404, { error: "not found" });
  };

  const server = createServer(handler);
  sidecar = {
    server, engine,
    statePath: options.statePath ?? null,
  };
  if (options.statePath) {
    loadSnapshotFile(options.statePath, engine);
  }
  return sidecar;
}

// --------------------------------------------------------------------------- //
// Durable state (atomic: tmp + rename)
// --------------------------------------------------------------------------- //
export function saveSnapshotFile(path: string, engine: PolicyEngine): void {
  const tmp = `${path}.tmp`;
  writeFileSync(tmp, JSON.stringify(takeSnapshot(engine), null, 1));
  renameSync(tmp, path);
}

export function loadSnapshotFile(path: string, engine: PolicyEngine): boolean {
  try {
    const snap = JSON.parse(readFileSync(path, "utf8")) as Snapshot;
    restoreSnapshot(engine, snap);
    return true;
  } catch (err) {
    if ((err as NodeJS.ErrnoException).code === "ENOENT") return false; // first boot
    throw err;
  }
}

// --------------------------------------------------------------------------- //
export function serveMain(argv: string[]): number {
  const get = (flag: string): string | undefined => {
    const i = argv.indexOf(`--${flag}`);
    return i >= 0 && i + 1 < argv.length ? argv[i + 1] : undefined;
  };
  const host = get("host") ?? "127.0.0.1";
  const port = Number(get("port") ?? 8787);
  const sc = createDecideServer({
    token: get("token") ?? null,
    qualityModelPath: get("quality-model") ?? null,
    statePath: get("state") ?? null,
  });
  sc.server.listen(port, host, () => {
    console.log(
      `llmrouter decide-sidecar on http://${host}:${port}  ` +
        `(POST /decide, GET /stats, GET /healthz)` +
        (get("token") ? "  [auth: bearer token required]" : ""),
    );
  });
  const stop = () => {
    if (sc.statePath) saveSnapshotFile(sc.statePath, sc.engine);
    sc.server.close();
    process.exit(0);
  };
  process.on("SIGINT", stop);
  process.on("SIGTERM", stop);
  return 0;
}

// Direct execution: node dist/serve.js --port 8787
if (process.argv[1] && process.argv[1].endsWith("serve.js")) {
  serveMain(process.argv.slice(2));
}
