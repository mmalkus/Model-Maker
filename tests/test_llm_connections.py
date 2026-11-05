"""Named LLM connections and model lists (a hosted server's AI settings).

A settings entry with a "kind" is a named connection: e.g. an
OpenAI-compatible service under its own name. "models" lists the models its
users may pick. The provider, the AI builder's loop, the Settings view and
/api/llm/models all follow them.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from modelmaker import api
from modelmaker.agent.build import LLMChoice
from modelmaker.agent.loop import OpenAILoop
from modelmaker.llm.openai_provider import OpenAIProvider
from modelmaker.workspace import Workspace


@pytest.fixture
def hosted(tmp_path, monkeypatch):
    ws = Workspace("w", cache_dir=tmp_path / "c", projects_dir=tmp_path / "p", root=tmp_path, features=["ai_builder", "ai_drafting"])
    ws.llm_settings.update(
        active_provider="openrouter",
        include_reference=None,
        settings={
            "openrouter": {"kind": "openai", "base_url": "https://router.example/v1", "api_key": "k-router",
                           "model": "m-small", "models": ["m-small", "m-large"]},
            "anthropic": {"api_key": "k-anthropic", "models": ["claude-sonnet-5-5"]},
        },
    )
    monkeypatch.setattr(api, "DEFAULT_WORKSPACE", ws)
    return ws


def test_a_named_connection_builds_its_kinds_provider(hosted):
    p = hosted.provider()
    assert isinstance(p, OpenAIProvider)
    assert p.base_url == "https://router.example/v1" and p.api_key == "k-router"
    assert hosted.kind_of("openrouter") == "openai" and hosted.kind_of("anthropic") == "anthropic"
    assert hosted.resolve_api_key("openrouter") == "k-router"


def test_the_ai_builder_runs_a_named_connection_as_its_kind(hosted):
    loop = hosted.agent_loop(LLMChoice(provider="openrouter", model="m-large"), "plan")
    assert isinstance(loop, OpenAILoop)


def test_settings_show_connections_and_model_lists(hosted):
    s = TestClient(api.app).get("/api/llm/settings").json()
    assert s["settings"]["openrouter"]["kind"] == "openai"
    assert s["settings"]["openrouter"]["models"] == ["m-small", "m-large"]
    assert s["settings"]["anthropic"]["models"] == ["claude-sonnet-5-5"]
    assert "api_key" not in s["settings"]["openrouter"] and s["settings"]["openrouter"]["api_key_set"]
    # Users who can't set providers up are offered only the configured ones.
    assert s["agent"]["capable_providers"] == ["anthropic", "openrouter"]


def test_models_endpoint_answers_with_the_allowed_list(hosted, tmp_path, monkeypatch):
    # Users without LLM settings get the list from /api/llm/settings; the
    # endpoint itself stays theirs to refuse.
    c = TestClient(api.app)
    assert c.get("/api/llm/models", params={"provider": "openrouter"}).status_code == 403
    ws = Workspace("local", cache_dir=tmp_path / "c2", projects_dir=tmp_path / "p2")
    ws.llm_settings.update(None, None, {"openrouter": {"kind": "openai", "base_url": "https://r/v1", "models": ["m-a", "m-b"]}})
    monkeypatch.setattr(api, "DEFAULT_WORKSPACE", ws)
    assert c.get("/api/llm/models", params={"provider": "openrouter"}).json() == {"models": ["m-a", "m-b"]}


def test_without_a_list_and_kind_nothing_changes(tmp_path, monkeypatch):
    ws = Workspace("local", cache_dir=tmp_path / "c", projects_dir=tmp_path / "p")
    monkeypatch.setattr(api, "DEFAULT_WORKSPACE", ws)
    assert ws.kind_of("openai") == "openai" and ws.connection_models("openai") == []
    s = TestClient(api.app).get("/api/llm/settings").json()
    assert "models" not in s["settings"]["openai"]
    assert "claude_cli" in s["agent"]["capable_providers"]
