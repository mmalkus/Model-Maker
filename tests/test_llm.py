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


def test_stub_provider_analyze_data_mode_proposes_tags_and_a_document():
    ctx = DraftContext(
        instruction="analyze",
        function_name="n/a",
        input_ports={"columns": [ColumnInfo(name="age", dtype="Int64", role="feature", count=100, null_count=2)]},
        mode="analyze_data",
    )
    result = get_provider("stub").draft(ctx)
    assert result.params == {"age": "stub-tag"}
    assert "Stub provider" in result.explanation


def test_analyze_data_endpoint_applies_tags_and_returns_document(client, tmp_path):
    csv_path = tmp_path / "data.csv"
    csv_path.write_text("age,income\n25,50000\n40,80000\n")
    read = client.post("/api/blocks", json={"category": "read_csv", "params": {"path": str(csv_path)}}).json()
    client.post(f"/api/blocks/{read['id']}/refresh")

    resp = client.post(f"/api/blocks/{read['id']}/analyze_data")
    assert resp.status_code == 200
    body = resp.json()
    assert body["tags"] == {"age": ["stub-tag"], "income": ["stub-tag"]}
    assert "Stub provider" in body["document"]
    assert body["document_path"] is None  # no project has been saved yet

    preview = client.get(f"/api/blocks/{read['id']}/preview").json()
    tags_by_column = {c["name"]: c["tags"] for c in preview["columns"]}
    assert tags_by_column == {"age": ["stub-tag"], "income": ["stub-tag"]}


def test_analyze_data_writes_the_document_into_the_saved_projects_files_dir(client, tmp_path):
    csv_path = tmp_path / "data.csv"
    csv_path.write_text("age\n25\n")
    read = client.post("/api/blocks", json={"category": "read_csv", "params": {"path": str(csv_path)}}).json()
    client.post(f"/api/blocks/{read['id']}/refresh")

    project_dir = tmp_path / "proj"
    client.post("/api/project/save", json={"path": str(project_dir)})

    body = client.post(f"/api/blocks/{read['id']}/analyze_data").json()
    assert body["document_path"] is not None
    from pathlib import Path

    assert Path(body["document_path"]).read_text(encoding="utf-8") == body["document"]
    assert Path(body["document_path"]).parent == project_dir / "files"


def test_analyze_data_requires_a_dataframe_output(client):
    display = client.post("/api/blocks", json={"category": "display_value"}).json()
    resp = client.post(f"/api/blocks/{display['id']}/analyze_data")
    assert resp.status_code in (400, 409)  # no output at all yet, or wrong type once run


def test_stub_provider_rename_mode_proposes_block_and_port_names():
    ctx = DraftContext(
        instruction="rename",
        function_name="n/a",
        input_ports={"out": [ColumnInfo(name="a", dtype="Int64", role="feature")]},
        mode="rename",
    )
    result = get_provider("stub").draft(ctx)
    assert result.params == {"name": "stub_renamed_block", "out": "stub_out"}
    assert "Stub provider" in result.explanation


def test_suggest_names_endpoint_applies_block_name_and_port_names(client, tmp_path):
    csv_path = tmp_path / "data.csv"
    csv_path.write_text("a,b\n1,10\n")
    read = client.post("/api/blocks", json={"category": "read_csv", "params": {"path": str(csv_path)}}).json()

    resp = client.post(f"/api/blocks/{read['id']}/suggest_names")
    assert resp.status_code == 200
    body = resp.json()
    assert body["name"] == "stub_renamed_block"
    assert body["port_names"] == {"out": "stub_out"}
    assert "explanation" in body

    graph = client.get("/api/graph").json()
    assert graph["blocks"][read["id"]]["name"] == "stub_renamed_block"
    assert graph["blocks"][read["id"]]["port_names"] == {"out": "stub_out"}


def test_suggest_names_resolves_a_collision_with_a_numeric_suffix(client, tmp_path):
    csv_path = tmp_path / "data.csv"
    csv_path.write_text("a,b\n1,10\n")
    first = client.post("/api/blocks", json={"category": "read_csv", "params": {"path": str(csv_path)}}).json()
    second = client.post(
        "/api/blocks", json={"category": "read_csv", "params": {"path": str(csv_path)}, "x": 200}
    ).json()

    # The stub provider always proposes the same "stub_out" name -- taking
    # it on the first block leaves nothing but a collision for the second.
    client.post(f"/api/blocks/{first['id']}/suggest_names")
    resp = client.post(f"/api/blocks/{second['id']}/suggest_names")
    assert resp.status_code == 200
    assert resp.json()["port_names"] == {"out": "stub_out_2"}

    # both stick -- no collision ever reached the graph
    graph = client.get("/api/graph").json()
    assert graph["blocks"][first["id"]]["port_names"] == {"out": "stub_out"}
    assert graph["blocks"][second["id"]]["port_names"] == {"out": "stub_out_2"}
