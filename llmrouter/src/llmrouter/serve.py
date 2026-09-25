"""`llmrouter serve` — the decision engine as an HTTP sidecar.

Zero dependencies (stdlib `http.server`), single-purpose: a gateway that already
owns its transport (LiteLLM custom router, Bifrost routing rule, Envoy ext_authz,
your own proxy) asks *us* which (model, backend) to use and executes the call
itself. This is the integration surface that lets the engine ride alongside
production gateways instead of replacing them.

    POST /decide   {RoutingRequest payload}          -> {Decision payload}
    GET  /stats    routing/session/backend telemetry -> {...}
    GET  /healthz  liveness for load balancers       -> {"ok": true}

Feedback closes the loop here too: when the caller's *next* request includes
`"observed_usage": {"cache_read_tokens": N, "cache_write_tokens": M}` (the
provider counters from the previous turn), the sidecar folds them into session
affinity — the learned-hit-rate machinery works across the HTTP boundary.

Security posture: binds 127.0.0.1 by default; pass --token to require
`Authorization: Bearer <token>`. This is a sidecar, not a public endpoint.

The Decision/Request JSON mirrors the TypeScript `serve.ts` byte-for-byte, so
one gateway client works against either runtime.
"""
from __future__ import annotations

import argparse
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Mapping

from .learning import QualityModel
from .router import Router
from .types import RoutingRequest

MAX_BODY_BYTES = 10 * 1024 * 1024


class _Handler(BaseHTTPRequestHandler):
    server: "_Sidecar"

    # ---- plumbing --------------------------------------------------------- #
    def log_message(self, fmt: str, *args: Any) -> None:  # quiet by default
        if self.server.verbose:
            super().log_message(fmt, *args)

    def _send(self, status: int, obj: Mapping[str, Any]) -> None:
        body = json.dumps(obj).encode("utf-8")
        self.send_response(status)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _authorized(self) -> bool:
        token = self.server.token
        if token is None:
            return True
        got = self.headers.get("authorization", "")
        return got == f"Bearer {token}"

    # ---- routes ----------------------------------------------------------- #
    def do_GET(self) -> None:  # noqa: N802 — BaseHTTPRequestHandler API
        if not self._authorized():
            return self._send(401, {"error": "unauthorized"})
        if self.path == "/healthz":
            return self._send(200, {"ok": True, "engine": "llmrouter"})
        if self.path == "/stats":
            return self._send(200, self.server.router.stats())
        return self._send(404, {"error": "not found"})

    def do_POST(self) -> None:  # noqa: N802
        if not self._authorized():
            return self._send(401, {"error": "unauthorized"})
        if self.path != "/decide":
            return self._send(404, {"error": "not found"})
        try:
            length = int(self.headers.get("content-length") or 0)
            if length <= 0 or length > MAX_BODY_BYTES:
                return self._send(400, {"error": "body must be 10MB or less"})
            payload = json.loads(self.rfile.read(length))
            if not isinstance(payload, dict):
                return self._send(400, {"error": "body must be a JSON object"})
        except (ValueError, json.JSONDecodeError) as exc:
            return self._send(400, {"error": f"invalid JSON: {exc}"})

        try:
            req = RoutingRequest.from_payload(payload)
        except (ValueError, KeyError) as exc:
            return self._send(400, {"error": str(exc)})

        try:
            with self.server.lock:
                # Feed the previous turn's provider counters in first, so THIS
                # decision already sees the learned hit rate.
                usage = payload.get("observed_usage")
                if usage:
                    from .adapters import Usage
                    self.server.router.note_usage(
                        req,
                        Usage(
                            input_tokens=int(usage.get("input_tokens", 0) or 0),
                            output_tokens=int(usage.get("output_tokens", 0) or 0),
                            cache_read_tokens=int(usage.get("cache_read_tokens", 0) or 0),
                            cache_write_tokens=int(usage.get("cache_write_tokens", 0) or 0),
                        ),
                    )
                decision = self.server.router.decide(req)
                # Pin the session for THIS turn so the next one is sticky —
                # the caller executes; we keep the affinity memory.
                self.server.router.note_decision(req, decision)
        except RuntimeError as exc:  # every candidate filtered out
            return self._send(422, {"error": str(exc)})
        return self._send(200, decision.to_payload())


class _Sidecar(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, router: Router, host: str, port: int,
                 token: str | None, verbose: bool) -> None:
        super().__init__((host, port), _Handler)
        self.router = router
        self.token = token
        self.verbose = verbose
        # SessionStore and BackendHealth are plain dicts — one lock keeps the
        # decide+learn critical section honest under ThreadingHTTPServer.
        self.lock = threading.Lock()


def build_router(quality_model_path: str | None = None,
                 state_path: str | None = None) -> Router:
    """Decision-capable router over the bundled defaults (no keys, no network)."""
    quality_model = (QualityModel.load(quality_model_path)
                     if quality_model_path else None)
    state_store = None
    if state_path:
        from .store import FileStateStore
        # autosave every decision: sidecar traffic is low-rate and durable
        # affinity is the entire point of running one.
        state_store = FileStateStore(state_path, autosave_every=1)
    return Router.pure(quality_model=quality_model, state_store=state_store)


def create_server(router: Router, host: str = "127.0.0.1", port: int = 8787,
                  token: str | None = None, verbose: bool = False) -> _Sidecar:
    return _Sidecar(router, host, port, token, verbose)


def cmd_serve(args: argparse.Namespace) -> int:
    router = build_router(args.quality_model, args.state)
    httpd = create_server(router, args.host, args.port, args.token, args.verbose)
    where = f"http://{args.host}:{args.port}"
    print(f"llmrouter decide-sidecar on {where}  "
          f"(POST {where}/decide, GET {where}/stats, GET {where}/healthz)"
          + ("  [auth: bearer token required]" if args.token else ""))
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nshutting down")
    finally:
        if args.state:
            router.save_state()
        httpd.server_close()
    return 0


def main(argv: list[str] | None = None) -> int:
    """Standalone entry (`python -m llmrouter.serve`); normally reached via
    `python -m llmrouter serve` in cli.main."""
    parser = argparse.ArgumentParser(prog="llmrouter serve")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8787)
    parser.add_argument("--token", default=None)
    parser.add_argument("--quality-model", default=None)
    parser.add_argument("--state", default=None)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)
    return cmd_serve(args)


__all__ = ["build_router", "create_server", "cmd_serve", "MAX_BODY_BYTES"]
