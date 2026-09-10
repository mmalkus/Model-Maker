import pytest
from fastapi.testclient import TestClient

import modelmaker.api as api_module
from modelmaker.llm import ColumnInfo, DraftContext, get_provider
from modelmaker.llm.settings import LLMSettingsStore
from modelmaker.session import ProjectSession


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setenv("MODELMAKER_LLM_PROVIDER", "stub")
    api_module.SESSION = ProjectSession()
    api_module.LLM_SETTINGS = LLMSettingsStore()
    return TestClient(api_module.app)


def test_stub_provider_registered():
    provider = get_provider("stub")
    ctx = DraftContext(
        instruction="add a high_b flag",
        function_name="bucket_income",
        input_ports={"df": [ColumnInfo(name="b", dtype="Int64", role="unassigned")]},
    )
    result = provider.draft(ctx)
    assert "def bucket_income(df):" in result.code
    assert result.metadata_transform == {"kind": "passthrough"}


def test_draft_endpoint_returns_proposal_without_mutating_block(client, tmp_path):
    csv_path = tmp_path / "data.csv"
    csv_path.write_text("a,b\n1,10\n")
    custom = client.post(
        "/api/blocks",
        json={
            "category": "bucket_income",
            "block_type": "standard",
            "inputs": [{"name": "df", "type": "dataframe"}],
            "outputs": [{"name": "out", "type": "dataframe"}],
            "code": "def bucket_income(df):\n    return df\n",
        },
    ).json()

    resp = client.post(f"/api/blocks/{custom['id']}/draft", json={"instruction": "flag rows where b > 15"})
    assert resp.status_code == 200
    body = resp.json()
    assert "def bucket_income" in body["code"]
    assert body["metadata_transform"] == {"kind": "passthrough"}
    assert "explanation" in body

    # never auto-applied: the stored block is untouched
    graph = client.get("/api/graph").json()
    assert graph["blocks"][custom["id"]]["code"] == "def bucket_income(df):\n    return df\n"
    assert graph["blocks"][custom["id"]]["code_version"] == 1


def test_draft_on_a_standard_block_is_params_only(client, tmp_path):
    csv_path = tmp_path / "data.csv"
    csv_path.write_text("a\n1\n")
    read = client.post("/api/blocks", json={"category": "read_csv", "params": {"path": str(csv_path)}}).json()
    resp = client.post(f"/api/blocks/{read['id']}/draft", json={"instruction": "point it at some other file"})
    assert resp.status_code == 200
    body = resp.json()
    # stub provider never calls an LLM -- it just proves the params_only path
    # runs end to end without touching the block's fixed code.
    assert body["code"] == ""
    assert body["params"] == {}
    assert "explanation" in body

    graph = client.get("/api/graph").json()
    assert graph["blocks"][read["id"]]["params"] == {"path": str(csv_path)}


def test_draft_requires_instruction_for_standard_blocks_too(client, tmp_path):
    csv_path = tmp_path / "data.csv"
    csv_path.write_text("a\n1\n")
    read = client.post("/api/blocks", json={"category": "read_csv", "params": {"path": str(csv_path)}}).json()
    resp = client.post(f"/api/blocks/{read['id']}/draft", json={"instruction": "  "})
    assert resp.status_code == 400


def test_draft_requires_nonempty_instruction(client):
    custom = client.post(
        "/api/blocks",
        json={
            "category": "custom_block",
            "block_type": "standard",
            "inputs": [{"name": "df", "type": "dataframe"}],
            "outputs": [{"name": "out", "type": "dataframe"}],
            "code": "def custom_block(df):\n    return df\n",
        },
    ).json()
    resp = client.post(f"/api/blocks/{custom['id']}/draft", json={"instruction": "   "})
    assert resp.status_code == 400


def test_suggest_fix_requires_recorded_error(client):
    custom = client.post(
        "/api/blocks",
        json={
            "category": "custom_block",
            "block_type": "standard",
            "inputs": [{"name": "df", "type": "dataframe"}],
            "outputs": [{"name": "out", "type": "dataframe"}],
            "code": "def custom_block(df):\n    return df\n",
        },
    ).json()
    resp = client.post(f"/api/blocks/{custom['id']}/suggest_fix", json={})
    assert resp.status_code == 400


def test_suggest_fix_after_failed_run(client, tmp_path):
    csv_path = tmp_path / "data.csv"
    csv_path.write_text("a,b\n1,10\n")
    read = client.post("/api/blocks", json={"category": "read_csv", "params": {"path": str(csv_path)}}).json()
    custom = client.post(
        "/api/blocks",
        json={
            "category": "bucket_income",
            "block_type": "standard",
            "inputs": [{"name": "df", "type": "dataframe"}],
            "outputs": [{"name": "out", "type": "dataframe"}],
            "code": "def bucket_income(df):\n    return df.this_method_does_not_exist()\n",
        },
    ).json()
    client.post(
        "/api/wires",
        json={"from_block": read["id"], "from_port": "out", "to_block": custom["id"], "to_port": "df"},
    )
    client.post(f"/api/blocks/{read['id']}/refresh")
    run_resp = client.post(f"/api/blocks/{custom['id']}/run")
    assert run_resp.json()["status"] == "red"

    resp = client.post(f"/api/blocks/{custom['id']}/suggest_fix", json={})
    assert resp.status_code == 200
    assert "def bucket_income" in resp.json()["code"]


def test_suggest_fix_on_a_standard_block_is_params_only(client, tmp_path):
    csv_path = tmp_path / "data.csv"
    csv_path.write_text("a,b\n1,10\n")
    read = client.post("/api/blocks", json={"category": "read_csv", "params": {"path": str(csv_path)}}).json()
    filt = client.post("/api/blocks", json={"category": "filter", "params": {"expr": "not_a_real_column > 1"}}).json()
    client.post(
        "/api/wires",
        json={"from_block": read["id"], "from_port": "out", "to_block": filt["id"], "to_port": "df"},
    )
    client.post(f"/api/blocks/{read['id']}/refresh")
    run_resp = client.post(f"/api/blocks/{filt['id']}/run")
    assert run_resp.json()["status"] == "red"

    resp = client.post(f"/api/blocks/{filt['id']}/suggest_fix", json={})
    assert resp.status_code == 200
    body = resp.json()
    assert body["code"] == ""

    # the block's fixed code and category are untouched -- only params could
    # ever be proposed for a standard block
    graph = client.get("/api/graph").json()
    assert graph["blocks"][filt["id"]]["category"] == "filter"


def test_llm_settings_endpoint_lists_registered_providers(client):
    resp = client.get("/api/llm/settings")
    assert resp.status_code == 200
    body = resp.json()
    assert {"stub", "claude_cli"} <= set(body["providers"])
    assert body["active_provider"] == "stub"


def test_llm_settings_can_be_updated(client):
    resp = client.put(
        "/api/llm/settings",
        json={"active_provider": "lmstudio", "settings": {"lmstudio": {"base_url": "http://example:9999/v1"}}},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["active_provider"] == "lmstudio"
    assert body["settings"]["lmstudio"]["base_url"] == "http://example:9999/v1"


def test_llm_settings_rejects_unknown_provider(client):
    resp = client.put("/api/llm/settings", json={"active_provider": "not_a_real_provider"})
    assert resp.status_code == 400


def test_llm_settings_lists_openai_and_gemini_providers(client):
    resp = client.get("/api/llm/settings")
    body = resp.json()
    assert {"openai", "gemini"} <= set(body["providers"])
    assert "openai" in body["settings"]
    assert "gemini" in body["settings"]


def test_api_key_is_never_echoed_back_but_presence_is_reported(client):
    resp = client.put(
        "/api/llm/settings",
        json={"settings": {"openai": {"api_key": "sk-super-secret-value", "model": "gpt-test"}}},
    )
    assert resp.status_code == 200
    body = resp.json()
    openai_settings = body["settings"]["openai"]
    assert "api_key" not in openai_settings
    assert "sk-super-secret-value" not in resp.text
    assert openai_settings["api_key_set"] is True
    assert openai_settings["api_key_source"] == "override"
    assert openai_settings["model"] == "gpt-test"

    # a second GET reflects the same redacted state
    resp2 = client.get("/api/llm/settings")
    assert resp2.json()["settings"]["openai"]["api_key_set"] is True
    assert "sk-super-secret-value" not in resp2.text


def test_api_key_can_be_cleared_with_explicit_null(client):
    client.put("/api/llm/settings", json={"settings": {"openai": {"api_key": "sk-abc"}}})
    resp = client.put("/api/llm/settings", json={"settings": {"openai": {"api_key": None}}})
    assert resp.json()["settings"]["openai"]["api_key_set"] is False
    assert resp.json()["settings"]["openai"]["api_key_source"] is None


def test_api_key_reported_as_env_sourced_when_not_overridden(client, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-from-env")
    resp = client.get("/api/llm/settings")
    openai_settings = resp.json()["settings"]["openai"]
    assert openai_settings["api_key_set"] is True
    assert openai_settings["api_key_source"] == "env"
    assert "sk-from-env" not in resp.text


def test_openai_provider_requires_api_key_and_model(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("MODELMAKER_OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("MODELMAKER_LLM_MODEL", raising=False)
    with pytest.raises(RuntimeError, match="API key"):
        get_provider("openai")

    with pytest.raises(RuntimeError, match="model"):
        get_provider("openai", api_key="sk-abc")


def test_openai_provider_custom_base_url_is_configurable(monkeypatch):
    provider = get_provider("openai", api_key="sk-abc", model="gpt-test", base_url="http://localhost:9999/v1")
    assert provider.base_url == "http://localhost:9999/v1"


def test_gemini_provider_requires_api_key_and_model(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.delenv("MODELMAKER_GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("MODELMAKER_LLM_MODEL", raising=False)
    with pytest.raises(RuntimeError, match="API key"):
        get_provider("gemini")

    with pytest.raises(RuntimeError, match="model"):
        get_provider("gemini", api_key="key-abc")
