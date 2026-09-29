"""The OpenAI and Gemini agent loops (see agent/loop.py), driven against a
local fake endpoint that replays scripted vendor responses and records
every request -- so what's checked is our side of the wire format: tool
declarations, echoing the model's turn, tool results matched to their
calls, resume, usage, and error handling. Not checked (no keys here): that
the real services accept exactly these payloads."""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from modelmaker.agent.build import AWAITING_APPROVAL, PLANNING, AgentBuild
from modelmaker.agent.loop import GeminiLoop, OpenAILoop, make_loop
from modelmaker.agent.tools import tools_for_phase
from modelmaker.runslot import RunSlot
from modelmaker.session import ProjectSession

DATA = Path(__file__).resolve().parents[1] / "sample_data" / "pd_model_data.csv"


class FakeEndpoint:
    """Serves `responses` in order (each a dict, or a callable of the request
    body returning one) and records (path, headers, body) per request."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.requests: list[tuple[str, dict, dict]] = []
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):  # noqa: N802
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                outer.requests.append((self.path, dict(self.headers), body))
                if not outer.responses:
                    self.send_response(500)
                    self.end_headers()
                    self.wfile.write(b'{"error": "script exhausted"}')
                    return
                r = outer.responses.pop(0)
                r = r(body) if callable(r) else r
                status = r.pop("_status", 200)
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps(r).encode())

            def log_message(self, *args):
                pass

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}"

    def close(self):
        self.server.shutdown()


@pytest.fixture
def planning_build(tmp_path):
    session = ProjectSession(recovery_path=tmp_path / "r.json")
    with session.edit():
        load = session.add_block("read_csv", name="applications", params={"path": str(DATA)})
    b = AgentBuild(session, RunSlot(), "split it", [load.id])
    b.phase = PLANNING
    return b, load.id


def _plan(anchor):
    return {
        "summary": "split",
        "lanes": [{"key": "est", "name": "Estimation"}],
        "steps": [
            {"ref": "s1", "category": "train_test_split", "lane": "est", "name": "split",
             "inputs": [{"port": "df", "from": anchor, "from_port": "out"}], "why": "holdout"}
        ],
    }


# ---- OpenAI --------------------------------------------------------------------


def _oa_message(content=None, calls=(), usage=(10, 5)):
    msg = {"role": "assistant", "content": content}
    if calls:
        msg["tool_calls"] = [
            {"id": cid, "type": "function", "function": {"name": name, "arguments": args}} for cid, name, args in calls
        ]
    return {"choices": [{"message": msg, "finish_reason": "tool_calls" if calls else "stop"}],
            "usage": {"prompt_tokens": usage[0], "completion_tokens": usage[1]}}


def test_openai_loop_plans_through_the_tools(planning_build):
    b, anchor = planning_build
    fake = FakeEndpoint(
        [
            _oa_message("Looking around.", [("c1", "get_graph", "{}"), ("c2", "describe_block_type", '{"category": "train_test_split"}')]),
            _oa_message(None, [("c3", "submit_plan", "{not json")]),
            _oa_message(None, [("c4", "submit_plan", json.dumps({"plan": _plan(anchor)}))]),
            _oa_message("Plan submitted."),
        ]
    )
    try:
        loop = OpenAILoop("gpt-test", api_key="sk-test", base_url=fake.url + "/v1")
        outcome = loop.start(b, "SYSTEM", "PROMPT", tools_for_phase(PLANNING))
    finally:
        fake.close()

    assert outcome.final_text == "Plan submitted."
    assert b.phase == AWAITING_APPROVAL and b.plan["steps"][0]["ref"] == "s1"
    assert b.usage["input_tokens"] == 40 and b.usage["output_tokens"] == 20

    path, headers, first = fake.requests[0]
    assert path == "/v1/chat/completions" and headers["Authorization"] == "Bearer sk-test"
    assert first["model"] == "gpt-test"
    assert "temperature" not in first and "max_tokens" not in first
    assert first["messages"][:2] == [{"role": "system", "content": "SYSTEM"}, {"role": "user", "content": "PROMPT"}]
    names = {t["function"]["name"] for t in first["tools"]}
    assert "submit_plan" in names and "add_block" not in names
    assert all(t["type"] == "function" and "parameters" in t["function"] for t in first["tools"])

    # Second request: the assistant turn echoed with its tool_calls, then one
    # tool message per call, in order, matched by id.
    second = fake.requests[1][2]["messages"]
    assert second[2]["role"] == "assistant" and [c["id"] for c in second[2]["tool_calls"]] == ["c1", "c2"]
    assert [(m["role"], m["tool_call_id"]) for m in second[3:5]] == [("tool", "c1"), ("tool", "c2")]
    assert anchor in second[3]["content"]
    # Malformed arguments go back as a tool error, not a crash.
    third = fake.requests[2][2]["messages"]
    assert "valid JSON" in json.loads(third[-1]["content"])["error"]


def test_openai_loop_resumes_with_history(planning_build):
    b, _ = planning_build
    fake = FakeEndpoint([_oa_message("first"), _oa_message("second")])
    try:
        loop = OpenAILoop("gpt-test", api_key="sk-test", base_url=fake.url)
        loop.start(b, "S", "P", tools_for_phase(PLANNING))
        loop.resume(b, "FEEDBACK", tools_for_phase(PLANNING))
    finally:
        fake.close()
    msgs = fake.requests[1][2]["messages"]
    assert [m["role"] for m in msgs] == ["system", "user", "assistant", "user"]
    assert msgs[-1]["content"] == "FEEDBACK"


def test_openai_http_errors_are_readable(planning_build):
    b, _ = planning_build
    fake = FakeEndpoint([{"_status": 401, "error": {"message": "bad key"}}])
    try:
        loop = OpenAILoop("gpt-test", api_key="sk-test", base_url=fake.url)
        with pytest.raises(RuntimeError, match="HTTP 401.*bad key"):
            loop.start(b, "S", "P", tools_for_phase(PLANNING))
    finally:
        fake.close()


# ---- Gemini --------------------------------------------------------------------


def _gm(parts, usage=(7, 3)):
    return {"candidates": [{"content": {"role": "model", "parts": parts}, "finishReason": "STOP"}],
            "usageMetadata": {"promptTokenCount": usage[0], "candidatesTokenCount": usage[1]}}


def test_gemini_loop_plans_through_the_tools(planning_build):
    b, anchor = planning_build
    fake = FakeEndpoint(
        [
            _gm(
                [
                    {"text": "thinking...", "thought": True},
                    {"functionCall": {"name": "get_graph", "args": {}, "id": "g1"}, "thoughtSignature": "SIG"},
                    {"functionCall": {"name": "no_such_tool", "args": {}}},
                ]
            ),
            _gm([{"functionCall": {"name": "submit_plan", "args": {"plan": _plan(anchor)}}}]),
            _gm([{"text": "Plan submitted."}]),
        ]
    )
    try:
        loop = GeminiLoop("gemini-test", api_key="g-key", base_url=fake.url)
        outcome = loop.start(b, "SYSTEM", "PROMPT", tools_for_phase(PLANNING))
    finally:
        fake.close()

    assert outcome.final_text == "Plan submitted."
    assert b.phase == AWAITING_APPROVAL
    assert b.usage["input_tokens"] == 21 and b.usage["output_tokens"] == 9
    assert all(e.get("text") != "thinking..." for e in b.events_since(0) if e["kind"] == "assistant")

    path, headers, first = fake.requests[0]
    assert path == "/models/gemini-test:generateContent"
    lower = {k.lower(): v for k, v in headers.items()}  # header names are case-insensitive
    assert lower["x-goog-api-key"] == "g-key" and "key=" not in path
    assert first["system_instruction"] == {"parts": [{"text": "SYSTEM"}]}
    decls = first["tools"][0]["functionDeclarations"]
    assert all("parametersJsonSchema" in d and "parameters" not in d for d in decls)

    second = fake.requests[1][2]["contents"]
    # The model turn goes back unchanged -- thought signature included.
    assert second[1]["role"] == "model" and second[1]["parts"][1]["thoughtSignature"] == "SIG"
    responses = second[2]["parts"]
    assert second[2]["role"] == "user" and len(responses) == 2
    assert responses[0]["functionResponse"]["name"] == "get_graph" and responses[0]["functionResponse"]["id"] == "g1"
    assert anchor in json.dumps(responses[0]["functionResponse"]["response"])
    assert "unknown tool" in responses[1]["functionResponse"]["response"]["error"]


def test_gemini_loop_resumes_with_history(planning_build):
    b, _ = planning_build
    fake = FakeEndpoint([_gm([{"text": "first"}]), _gm([{"text": "second"}])])
    try:
        loop = GeminiLoop("gemini-test", api_key="g-key", base_url=fake.url)
        loop.start(b, "S", "P", [])
        loop.resume(b, "FEEDBACK", [])
    finally:
        fake.close()
    contents = fake.requests[1][2]["contents"]
    assert [c["role"] for c in contents] == ["user", "model", "user"]
    assert contents[-1]["parts"] == [{"text": "FEEDBACK"}]
    assert "tools" not in fake.requests[1][2]


def test_stopped_build_makes_no_requests(planning_build):
    b, _ = planning_build
    b.stop_requested = True
    fake = FakeEndpoint([])
    try:
        OpenAILoop("m", api_key="k", base_url=fake.url).start(b, "S", "P", [])
        GeminiLoop("m", api_key="k", base_url=fake.url).start(b, "S", "P", [])
    finally:
        fake.close()
    assert fake.requests == []


# ---- configuration -----------------------------------------------------------------


def test_missing_key_or_model_is_a_clear_error(monkeypatch):
    for var in ("OPENAI_API_KEY", "MODELMAKER_OPENAI_API_KEY", "GEMINI_API_KEY", "GOOGLE_API_KEY",
                "MODELMAKER_GEMINI_API_KEY", "MODELMAKER_LLM_MODEL"):
        monkeypatch.delenv(var, raising=False)
    with pytest.raises(RuntimeError, match="no OpenAI API key"):
        make_loop("openai", "gpt", api_base_url="", token="")
    with pytest.raises(RuntimeError, match="no model configured for gemini"):
        make_loop("gemini", None, api_base_url="", token="", api_key="k")


def test_start_endpoint_rejects_an_unconfigured_provider_up_front(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    from modelmaker import api
    from modelmaker.agent.controller import BuildController
    from modelmaker.llm.settings import LLMSettingsStore

    for var in ("OPENAI_API_KEY", "MODELMAKER_OPENAI_API_KEY", "MODELMAKER_LLM_MODEL"):
        monkeypatch.delenv(var, raising=False)
    slot = RunSlot()
    monkeypatch.setattr(api, "SESSION", ProjectSession(recovery_path=tmp_path / "r.json"))
    monkeypatch.setattr(api, "RUN_SLOT", slot)
    monkeypatch.setattr(api, "AGENT", BuildController(lambda: api.SESSION, slot, api._agent_loop_factory))
    monkeypatch.setattr(api, "LLM_SETTINGS", LLMSettingsStore(active_provider="openai"))
    c = TestClient(api.app)
    block = c.post("/api/blocks", json={"category": "read_csv", "params": {"path": str(DATA)}}).json()
    r = c.post("/api/agent/builds", json={"goal": "x", "anchors": [block["id"]]})
    assert r.status_code == 400 and "plan LLM (openai)" in r.json()["detail"] and "API key" in r.json()["detail"]
    assert api.AGENT.build is None
    # Settings key + model make it acceptable (still no network call at start).
    api.LLM_SETTINGS.update(None, None, {"openai": {"api_key": "sk-x", "model": "gpt-x"}})
    assert c.get("/api/llm/settings").json()["agent"]["capable_providers"] == ["claude_cli", "anthropic", "openai", "gemini"]
