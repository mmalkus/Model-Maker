"""Restricted mode (features.py): a workspace can switch off custom block
code, the AI builder, AI drafting, LLM settings, env vars, git and git
remotes. Switched-off features are refused by the API itself, whatever the
UI shows; the default workspace (the local app) keeps all of them."""

from __future__ import annotations

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

import modelmaker.api as api
from modelmaker import features
from modelmaker.project import save_project
from modelmaker.workspace import Workspace

from .helpers import write_library_file

HEADER = "x-test-workspace"


@pytest.fixture
def pair(tmp_path):
    full = Workspace("full", cache_dir=tmp_path / "c-full", projects_dir=tmp_path / "p-full", root=tmp_path)
    locked = Workspace(
        "locked",
        cache_dir=tmp_path / "c-locked",
        projects_dir=tmp_path / "p-locked",
        root=tmp_path,
        features=set(),
    )
    by_id = {"full": full, "locked": locked}

    def resolve(request):
        ws = by_id.get(request.headers.get(HEADER, ""))
        if ws is None:
            raise HTTPException(404, "no such workspace")
        return ws

    api.set_workspace_resolver(resolve)
    try:
        yield full, locked
    finally:
        api.set_workspace_resolver(None)


def _client(ws: Workspace) -> TestClient:
    return TestClient(api.app, headers={HEADER: ws.id})


def _lib_block() -> None:
    write_library_file("mine.py", {"double": {"inputs": [{"name": "df"}]}}, "def double(df):\n    return df\n")


# (method, path, body, feature) -- one or more per switched-off feature.
REFUSED = [
    ("GET", "/api/library", None, features.CUSTOM_BLOCKS),
    ("GET", "/api/library/template", None, features.CUSTOM_BLOCKS),
    ("PUT", "/api/library/files/x.py", {"source": "x = 1\n", "create": True}, features.CUSTOM_BLOCKS),
    ("POST", "/api/library/check", {"source": "x = 1\n", "name": "x.py"}, features.CUSTOM_BLOCKS),
    ("POST", "/api/library/import", {"filename": "x.py", "content_base64": "", "actions": {}}, features.CUSTOM_BLOCKS),
    ("POST", "/api/library/duplicate", {"id": "filter", "new_id": "f2", "to": "x.py"}, features.CUSTOM_BLOCKS),
    ("POST", "/api/library/ai/draft", {"instruction": "x"}, features.CUSTOM_BLOCKS),
    ("POST", "/api/blocks/b1/update", None, features.CUSTOM_BLOCKS),
    ("GET", "/api/blocks/b1/update_preview", None, features.CUSTOM_BLOCKS),
    ("POST", "/api/agent/builds", {"goal": "x", "anchors": []}, features.AI_BUILDER),
    ("GET", "/api/agent/builds/current", None, features.AI_BUILDER),
    ("POST", "/api/blocks/b1/draft", {"instruction": "x"}, features.AI_DRAFTING),
    ("POST", "/api/blocks/b1/suggest_fix", {}, features.AI_DRAFTING),
    ("POST", "/api/blocks/b1/analyze_data", {}, features.AI_DRAFTING),
    ("POST", "/api/blocks/b1/suggest_names", {}, features.AI_DRAFTING),
    ("PUT", "/api/llm/settings", {"active_provider": "openai"}, features.LLM_SETTINGS),
    ("GET", "/api/llm/models?provider=openai", None, features.LLM_SETTINGS),
    ("GET", "/api/env_vars/FOO", None, features.ENV_VARS),
    ("PUT", "/api/env_vars/FOO", {"value": "1"}, features.ENV_VARS),
    ("DELETE", "/api/env_vars/FOO", None, features.ENV_VARS),
    ("GET", "/api/project/git/status", None, features.GIT),
    ("POST", "/api/project/git/init", None, features.GIT),
    ("POST", "/api/project/git/remote", {"url": "https://example.com/x.git"}, features.GIT),
    ("POST", "/api/project/git/push", None, features.GIT),
]


@pytest.mark.parametrize(("method", "path", "body", "feature"), REFUSED)
def test_switched_off_features_are_refused(pair, method, path, body, feature):
    _, locked = pair
    resp = _client(locked).request(method, path, json=body)
    assert resp.status_code == 403, (path, resp.text)
    assert resp.json()["detail"] == features.refusal(feature)


@pytest.mark.parametrize(("method", "path", "body", "feature"), REFUSED)
def test_the_same_requests_reach_their_routes_with_every_feature(pair, method, path, body, feature):
    full, _ = pair
    resp = _client(full).request(method, path, json=body)
    assert resp.status_code != 403 or resp.json()["detail"] != features.refusal(feature)


def test_each_feature_is_independent(tmp_path, pair):
    full, _ = pair
    ws = Workspace("git-only", cache_dir=tmp_path / "c3", projects_dir=tmp_path / "p3", root=tmp_path,
                   features=features.ALL - {features.GIT_REMOTE})
    api.set_workspace_resolver(lambda request: ws)
    client = TestClient(api.app)
    # Local git still works; remotes don't.
    assert client.get("/api/project/git/status").status_code != 403
    for path in ("/api/project/git/remote", "/api/project/git/push", "/api/project/git/pull"):
        resp = client.post(path, json={"url": "https://example.com/x.git"})
        assert resp.status_code == 403
        assert resp.json()["detail"] == features.refusal(features.GIT_REMOTE)
    assert client.get("/api/library").status_code == 200


def test_reading_llm_settings_stays_allowed(pair):
    _, locked = pair
    assert _client(locked).get("/api/llm/settings").status_code == 200


def test_capabilities_endpoint_lists_features(pair):
    full, locked = pair
    assert _client(full).get("/api/capabilities").json()["features"] == sorted(features.ALL)
    assert _client(locked).get("/api/capabilities").json()["features"] == []


def test_default_workspace_keeps_everything():
    assert api.DEFAULT_WORKSPACE.features == features.LOCAL == features.ALL
    client = TestClient(api.app)
    body = client.get("/api/capabilities").json()
    assert body["features"] == sorted(features.ALL)
    assert client.get("/api/library").status_code == 200
    assert client.get("/api/env_vars/FOO").status_code == 200


def test_unknown_feature_names_fail_loudly(tmp_path):
    with pytest.raises(ValueError, match="unknown features"):
        Workspace("x", cache_dir=tmp_path, features={"custom_block"})


def test_placing_a_library_block_is_refused_without_custom_blocks(pair):
    full, locked = pair
    _lib_block()
    body = {"library": {"file": "mine.py", "id": "double"}}
    assert _client(full).post("/api/blocks", json=body).status_code == 200
    resp = _client(locked).post("/api/blocks", json=body)
    assert resp.status_code == 403
    assert locked.session.graph.blocks == {}
    # Built-ins still go.
    assert _client(locked).post("/api/blocks", json={"category": "filter"}).status_code == 200


def test_a_project_with_embedded_code_is_refused_without_custom_blocks(pair, tmp_path):
    full, locked = pair
    _lib_block()
    assert _client(full).post("/api/blocks", json={"library": {"file": "mine.py", "id": "double"}}).status_code == 200
    target = tmp_path / "proj"
    target.mkdir()
    save_project(full.session.graph, target / "project.json")
    resp = _client(locked).post("/api/project/load", json={"path": str(target)})
    assert resp.status_code == 400
    assert "carries its own code" in resp.json()["detail"]
    assert locked.session.graph.blocks == {}
    assert _client(full).post("/api/project/load", json={"path": str(target)}).status_code == 200


def test_session_layer_refuses_embedded_code_too(pair):
    _, locked = pair
    _lib_block()
    from modelmaker import library_store
    from modelmaker.capabilities import CapabilityError

    with api.workspace_context(locked):
        definition, ref = library_store.placeable("mine.py", "double")
    with pytest.raises(CapabilityError):
        locked.session.add_block(category="double", definition=definition, library_ref=ref)


def test_needed_matches_routes():
    assert features.needed("GET", "/api/graph") == []
    assert features.needed("GET", "/api/llm/settings") == []
    assert features.needed("POST", "/api/library/ai/draft") == [features.CUSTOM_BLOCKS, features.AI_DRAFTING]
    assert features.needed("POST", "/api/project/git/push") == [features.GIT, features.GIT_REMOTE]
    assert features.needed("POST", "/api/blocks/abc/run") == []
