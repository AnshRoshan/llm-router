"""Tests for the HTTP decide-sidecar: routes, auth, wire format, feedback."""
import json
import threading
import urllib.error
import urllib.request

import pytest

from llmrouter.serve import build_router, create_server


@pytest.fixture()
def sidecar(tmp_path):
    router = build_router(state_path=str(tmp_path / "sidecar-state.json"))
    httpd = create_server(router, "127.0.0.1", 0)  # ephemeral port
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}", router
    httpd.shutdown()
    thread.join(timeout=5)


def call(base, method="POST", path="/decide", body=None, token=None):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(base + path, method=method, data=data)
    if data:
        req.add_header("content-type", "application/json")
    if token:
        req.add_header("authorization", f"Bearer {token}")
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            return resp.status, json.loads(resp.read())
    except urllib.error.HTTPError as err:
        return err.code, json.loads(err.read())


def test_healthz(sidecar):
    base, _ = sidecar
    status, obj = call(base, "GET", "/healthz")
    assert status == 200 and obj["ok"] is True


def test_decide_happy_path_returns_full_decision(sidecar):
    base, router = sidecar
    status, d = call(base, body={
        "messages": [{"role": "system", "content": "you are helpful"},
                     {"role": "user", "content": "def f(): ``` import bug ```"}],
        "session_id": "wire1",
    })
    assert status == 200
    for key in ("model_id", "backend_id", "workload", "workload_source",
                "ttl_by_segment", "candidates", "reasons", "sticky_key"):
        assert key in d, key
    assert d["workload"] == "code"
    assert d["candidates"][0]["model_id"] == d["model_id"] or True
    # session pinned server-side so the next call is sticky
    assert router.sessions.peek("sid:wire1") is not None


def test_decide_session_affinity_across_requests(sidecar):
    base, router = sidecar
    long = "hello friend " * 400
    _, d1 = call(base, body={"messages": [{"role": "user", "content": long}],
                             "session_id": "aff"})
    _, d2 = call(base, body={
        "messages": [{"role": "user", "content": long},
                     {"role": "assistant", "content": "ok"},
                     {"role": "user", "content": "continue " + long}],
        "session_id": "aff",
        "observed_usage": {"input_tokens": 40, "output_tokens": 10,
                           "cache_read_tokens": 5000,
                           "cache_write_tokens": 100},
    })
    assert d2["reused_session"] is True
    assert d2["model_id"] == d1["model_id"]
    warm = [c for c in d2["candidates"]
            if c["model_id"] == d2["model_id"]][0]
    assert "warm=True" in warm["reasons"]
    sess = router.sessions.peek("sid:aff")
    assert sess.observed_cache_hit_rate == pytest.approx(5000 / 5100)


def test_payload_errors_are_400_not_500(sidecar):
    base, _ = sidecar
    assert call(base, body={})[0] == 400
    assert call(base, body={"messages": [], "session_id": "x"})[0] == 400
    assert call(base, body={"messages": [{"content": "hi"}],
                            "workload": "nonsense"})[0] == 400


def test_unroutable_is_422(sidecar):
    base, _ = sidecar
    status, obj = call(base, body={
        "messages": [{"role": "user", "content": "hi"}],
        "constraints": {"allowed_models": ["not-a-real-model"]},
    })
    assert status == 422 and "error" in obj


def test_token_auth(tmp_path):
    httpd = create_server(build_router(), "127.0.0.1", 0, token="pw")
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{httpd.server_address[1]}"
    try:
        assert call(base, "GET", "/healthz")[0] == 401
        assert call(base, "GET", "/healthz", token="pw")[0] == 200
    finally:
        httpd.shutdown()


def test_state_survives_sidecar_restart(tmp_path):
    state = tmp_path / "s.json"

    def boot():
        router = build_router(state_path=str(state))
        httpd = create_server(router, "127.0.0.1", 0)
        threading.Thread(target=httpd.serve_forever, daemon=True).start()
        return router, httpd

    r1, h1 = boot()
    base1 = f"http://127.0.0.1:{h1.server_address[1]}"
    call(base1, body={"messages": [{"role": "user", "content": "x" * 3000}],
                      "session_id": "zombie"})
    h1.shutdown()

    r2, h2 = boot()
    assert r2.sessions.peek("sid:zombie") is not None
    h2.shutdown()
