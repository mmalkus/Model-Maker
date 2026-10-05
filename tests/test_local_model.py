"""The built-in local model: the one-click download (resumed, checksummed)
against a local HTTP server, and LocalLoop's think-then-answer against a
stand-in for llama.cpp."""

from __future__ import annotations

import hashlib
import http.server
import json
import threading
from dataclasses import replace

import pytest

from modelmaker.agent.guided import BUILD_TOOLS
from modelmaker.agent.loop import ANSWER_TOKENS, LocalLoop, budgeted_answer
from modelmaker.llm import local_model

PAYLOAD = bytes(range(256)) * 4096  # 1 MiB


class _Files(http.server.BaseHTTPRequestHandler):
    ranges: list[str | None] = []

    def do_GET(self):  # noqa: N802 -- serves PAYLOAD, honouring Range like Hugging Face
        rng = self.headers.get("Range")
        type(self).ranges.append(rng)
        body = PAYLOAD[int(rng.split("=")[1].rstrip("-")) if rng else 0:]
        self.send_response(206 if rng else 200)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass


@pytest.fixture
def spec(tmp_path, monkeypatch):
    _Files.ranges = []
    httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _Files)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    monkeypatch.setenv("MODELMAKER_MODELS_DIR", str(tmp_path))
    monkeypatch.delenv("MODELMAKER_LOCAL_MODEL_PATH", raising=False)
    s = replace(local_model.DEFAULT_MODEL, url=f"http://127.0.0.1:{httpd.server_address[1]}/m.gguf",
                size=len(PAYLOAD), sha256=hashlib.sha256(PAYLOAD).hexdigest())
    monkeypatch.setattr(local_model, "DEFAULT_MODEL", s)
    monkeypatch.setattr(local_model, "_download", None)
    yield s
    httpd.shutdown()


def _download(spec):
    d = local_model.Download(spec, local_model.model_path())
    d.thread.join(30)
    return d


def test_the_download_resumes_checks_out_and_counts_as_downloaded(spec):
    local_model.model_path().with_name(spec.filename + ".part").write_bytes(PAYLOAD[:300_000])
    assert not local_model.is_downloaded()
    assert _download(spec).state == "done" and _Files.ranges == ["bytes=300000-"]
    assert local_model.model_path().read_bytes() == PAYLOAD and local_model.is_downloaded()


def test_a_download_that_doesnt_match_its_checksum_is_thrown_away(spec):
    d = _download(replace(spec, sha256="0" * 64))
    assert d.state == "failed" and "checksum" in d.error and not local_model.model_path().exists()


def test_the_settings_panel_starts_the_download_and_sees_it_finish(spec):
    from fastapi.testclient import TestClient

    import modelmaker.api as api

    client = TestClient(api.app)
    assert client.get("/api/local-model/").json()["downloaded"] is False
    client.post("/api/local-model/download")
    local_model._download.thread.join(30)
    assert client.get("/api/local-model/").json()["downloaded"] is True


def test_the_ai_builder_defaults_to_the_local_model_once_it_is_downloaded(monkeypatch):
    import modelmaker.api as api

    monkeypatch.delenv("MODELMAKER_LLM_PROVIDER", raising=False)
    monkeypatch.setattr(api.LLM_SETTINGS, "active_provider", None)
    monkeypatch.setattr(api._local_model, "runtime_available", lambda: True)
    monkeypatch.setattr(api._local_model, "is_downloaded", lambda: False)
    assert api.agent_llm_choice({}, None).provider == "claude_cli"
    monkeypatch.setattr(api._local_model, "is_downloaded", lambda: True)
    assert api.agent_llm_choice({}, None).provider == "local"
    assert api.agent_llm_choice({"provider": "anthropic"}, None).provider == "anthropic"  # a choice wins


class _FakeLlama:
    """Records LocalLoop's two completions: the think, then the answer."""

    metadata = {"tokenizer.chat_template": (
        "{% for m in messages %}<|im_start|>{{ m.role }}\n{{ m.content }}<|im_end|>\n{% endfor %}"
        "{% if add_generation_prompt %}<|im_start|>assistant\n{% if enable_thinking %}<think>\n{% endif %}{% endif %}"
    )}

    def __init__(self, answer: str):
        self.answer, self.calls = answer, []

    def create_completion(self, prompt, **kw):
        self.calls.append({"prompt": prompt, **kw})
        text = "Development should be the earlier 75%." if len(self.calls) == 1 else self.answer
        return {"choices": [{"text": text}], "usage": {"prompt_tokens": 10, "completion_tokens": 5}}


class _Build:
    stop_requested = False

    def __init__(self):
        self.usage = {"input_tokens": 0, "output_tokens": 0}

    def check_stop(self):
        pass


def test_the_local_loop_thinks_within_its_budget_then_answers_in_the_schema(monkeypatch):
    pytest.importorskip("llama_cpp")
    args = {"block": "time_split", "inputs": ["D1"], "settings": {"cutoff": "2024-10-08"}}
    fake = _FakeLlama(json.dumps({"tool": "place_block", "args": args}))
    monkeypatch.setattr(local_model, "load", lambda: fake)
    loop = LocalLoop()
    loop.think_tokens = 300
    build = _Build()
    got = loop.ask(build, "SYSTEM", "Which block next?", BUILD_TOOLS)
    assert (got.tool, got.args) == ("place_block", args)
    think, final = fake.calls
    assert think["prompt"].endswith("<|im_start|>assistant\n<think>\n") and "Which block next?" in think["prompt"]
    assert think["max_tokens"] == 300 and think["stop"] == ["</think>"]
    assert final["prompt"].endswith("earlier 75%.\n</think>\n\n") and final["grammar"] is not None
    assert build.usage == {"input_tokens": 20, "output_tokens": 10}
    # Cut off mid-answer: an error for the next question, not a crash.
    fake.answer, fake.calls = '{"tool": "place_block", "args": {"blo', []
    assert "complete JSON" in loop.ask(build, "S", "P", BUILD_TOOLS).error
    with pytest.raises(RuntimeError, match="guided builds only"):
        loop.start()


def test_a_zero_budget_answers_without_thinking_and_a_budget_thinks_first():
    calls = []

    def complete(text, max_tokens, stop, schema):
        calls.append((text, max_tokens, stop, schema is not None))
        return "the cutoff is 2024-10-08" if schema is None else '{"tool": "stage_done", "args": {"summary": "s"}}'

    assert budgeted_answer(_Build(), complete, "PROMPT", 0, BUILD_TOOLS).tool == "stage_done"
    assert calls == [("PROMPT<think>\n\n</think>\n\n", ANSWER_TOKENS, [], True)]  # one call: thinking closed at once
    calls.clear()
    budgeted_answer(_Build(), complete, "PROMPT", 200, BUILD_TOOLS)
    assert calls[0] == ("PROMPT<think>\n", 200, ["</think>"], False)
    assert calls[1][0] == "PROMPT<think>\nthe cutoff is 2024-10-08\n</think>\n\n" and calls[1][3]
