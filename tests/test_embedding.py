"""The opt-in backend switches the VS Code extension sets (see
vscode-extension-plan.md §3): the auth token, the CORS origin regex,
/api/version, and the copilot provider over a fake relay. With none of
them set, the rest of the suite covers that nothing changed."""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.testclient import TestClient

import modelmaker.api as api_module
from modelmaker.agent.loop import CopilotLoop, make_loop
from modelmaker.llm import get_provider
from modelmaker.llm.base import DraftContext


@pytest.fixture
def client():
    return TestClient(api_module.app)


# ---- auth token ------------------------------------------------------------


def test_no_token_configured_means_no_auth(client, monkeypatch):
    monkeypatch.delenv("MODELMAKER_AUTH_TOKEN", raising=False)
    assert client.get("/api/registry").status_code == 200


def test_token_required_when_configured(client, monkeypatch):
    monkeypatch.setenv("MODELMAKER_AUTH_TOKEN", "s3cret")
    assert client.get("/api/registry").status_code == 401
    assert client.get("/api/registry", headers={"Authorization": "Bearer wrong"}).status_code == 401
    assert client.get("/api/registry", headers={"Authorization": "Bearer s3cret"}).status_code == 200


def test_query_token_only_for_reads(client, monkeypatch):
    monkeypatch.setenv("MODELMAKER_AUTH_TOKEN", "s3cret")
    assert client.get("/api/registry?token=s3cret").status_code == 200
    # A mutating request can't authenticate by query string.
    assert client.post("/api/project/new?token=s3cret").status_code == 401


def test_health_and_mcp_bridge_stay_open(client, monkeypatch):
    monkeypatch.setenv("MODELMAKER_AUTH_TOKEN", "s3cret")
    assert client.get("/api/health").status_code == 200
    # Reaches the route, which checks its own X-Agent-Token instead.
    assert client.get("/api/agent/mcp/tools").status_code != 401


def test_non_ascii_token_header_is_rejected_not_crashed(client, monkeypatch):
    monkeypatch.setenv("MODELMAKER_AUTH_TOKEN", "s3cret")
    resp = client.get("/api/registry", headers={"Authorization": "Bearer café".encode("latin-1")})
    assert resp.status_code == 401


# ---- CORS ------------------------------------------------------------------


def _cors_app(monkeypatch, regex):
    if regex is None:
        monkeypatch.delenv("MODELMAKER_CORS_ORIGIN_REGEX", raising=False)
    else:
        monkeypatch.setenv("MODELMAKER_CORS_ORIGIN_REGEX", regex)
    app = FastAPI()
    app.add_middleware(CORSMiddleware, **api_module._cors_kwargs())

    @app.get("/x")
    def x():
        return {}

    return TestClient(app)


def test_cors_regex_unset_keeps_dev_origins_only(monkeypatch):
    c = _cors_app(monkeypatch, None)
    assert "allow_origin_regex" not in api_module._cors_kwargs()
    assert c.get("/x", headers={"Origin": "http://localhost:5173"}).headers["access-control-allow-origin"] == "http://localhost:5173"
    assert "access-control-allow-origin" not in c.get("/x", headers={"Origin": "vscode-webview://abc123"}).headers


def test_cors_regex_admits_matching_origin(monkeypatch):
    c = _cors_app(monkeypatch, r"^vscode-webview://[a-z0-9-]+$")
    resp = c.get("/x", headers={"Origin": "vscode-webview://abc123"})
    assert resp.headers["access-control-allow-origin"] == "vscode-webview://abc123"
    assert "access-control-allow-origin" not in c.get("/x", headers={"Origin": "https://evil.example"}).headers


def test_auth_failure_carries_cors_headers(client, monkeypatch):
    # The app's own CORS sits outside the auth check, so a cross-origin
    # client can read the 401. A dev origin is always allowed.
    monkeypatch.setenv("MODELMAKER_AUTH_TOKEN", "s3cret")
    resp = client.get("/api/registry", headers={"Origin": "http://localhost:5173"})
    assert resp.status_code == 401
    assert resp.headers["access-control-allow-origin"] == "http://localhost:5173"


def test_preflight_passes_without_token(client, monkeypatch):
    monkeypatch.setenv("MODELMAKER_AUTH_TOKEN", "s3cret")
    resp = client.options(
        "/api/project/save",
        headers={"Origin": "http://localhost:5173", "Access-Control-Request-Method": "POST"},
    )
    assert resp.status_code == 200


# ---- version ---------------------------------------------------------------


def test_version_reports_package_and_compat_level(client):
    from importlib import metadata

    body = client.get("/api/version").json()
    assert body["api"] == api_module.API_COMPAT_LEVEL
    assert body["version"] == metadata.version("quantology-modelmaker")


# ---- copilot ---------------------------------------------------------------


class FakeRelay:
    """An OpenAI-compatible endpoint like the extension's Copilot relay:
    /models in preference order, scripted /chat/completions replies, and a
    bearer token it checks."""

    def __init__(self, replies, token="relay-token"):
        self.replies = list(replies)
        self.requests: list[dict] = []
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def _authorized(self):
                if self.headers.get("Authorization") != f"Bearer {token}":
                    self.send_response(401)
                    self.end_headers()
                    return False
                return True

            def _json(self, body):
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps(body).encode())

            def do_GET(self):  # noqa: N802
                if self._authorized():
                    self._json({"data": [
                        {"id": "gpt-4.1", "name": "GPT-4.1", "max_input_tokens": 1000},
                        {"id": "claude-sonnet", "name": "Claude Sonnet", "max_input_tokens": 64000},
                    ]})

            def do_POST(self):  # noqa: N802
                if self._authorized():
                    outer.requests.append(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
                    self._json(outer.replies.pop(0))

            def log_message(self, *args):
                pass

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}/v1"

    def close(self):
        self.server.shutdown()


@pytest.fixture
def relay(monkeypatch):
    fake = FakeRelay([])
    monkeypatch.setenv("MODELMAKER_COPILOT_BASE_URL", fake.url)
    monkeypatch.setenv("MODELMAKER_COPILOT_TOKEN", "relay-token")
    monkeypatch.delenv("MODELMAKER_COPILOT_MODEL", raising=False)
    monkeypatch.delenv("MODELMAKER_LLM_MODEL", raising=False)
    yield fake
    fake.close()


def test_copilot_unavailable_outside_vscode(client, monkeypatch):
    monkeypatch.delenv("MODELMAKER_COPILOT_BASE_URL", raising=False)
    monkeypatch.delenv("MODELMAKER_COPILOT_TOKEN", raising=False)
    with pytest.raises(RuntimeError, match="only available .* inside VS Code"):
        get_provider("copilot")
    with pytest.raises(RuntimeError, match="inside VS Code"):
        make_loop("copilot", None, api_base_url="", token="")
    body = client.get("/api/llm/settings").json()
    assert "copilot" in body["providers"]
    assert body["settings"]["copilot"]["available"] is False
    assert "copilot" in body["agent"]["capable_providers"]
    assert client.get("/api/llm/models", params={"provider": "copilot"}).status_code == 400


def test_copilot_settings_and_models_through_relay(client, relay):
    assert client.get("/api/llm/settings").json()["settings"]["copilot"]["available"] is True
    # The relay's order, not sorted: the first is the default.
    assert client.get("/api/llm/models", params={"provider": "copilot"}).json() == {"models": ["gpt-4.1", "claude-sonnet"]}


def test_copilot_provider_drafts_through_relay(relay):
    relay.replies.append({"choices": [{"message": {"content": json.dumps({"code": "def f(df):\n    return df"})}, "finish_reason": "stop"}]})
    provider = get_provider("copilot")
    assert provider.model == "gpt-4.1"
    result = provider.draft(DraftContext(instruction="identity", function_name="f", input_ports={}, param_names=[]))
    assert "return df" in result.code
    assert relay.requests[0]["model"] == "gpt-4.1"


def test_copilot_loop_takes_model_and_context_from_relay(relay):
    loop = make_loop("copilot", "claude-sonnet", api_base_url="", token="")
    assert isinstance(loop, CopilotLoop)
    assert loop.model == "claude-sonnet" and loop.context_tokens == 64000
    assert make_loop("copilot", None, api_base_url="", token="").model == "gpt-4.1"
