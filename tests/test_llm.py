import pytest
from fastapi.testclient import TestClient

import modelmaker.api as api_module
from modelmaker.llm import ColumnInfo, DraftContext, get_provider
from modelmaker.session import ProjectSession


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setenv("MODELMAKER_LLM_PROVIDER", "stub")
    api_module.SESSION = ProjectSession()
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
            "block_type": "llm_authored",
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


def test_draft_rejects_non_custom_block(client, tmp_path):
    csv_path = tmp_path / "data.csv"
    csv_path.write_text("a\n1\n")
    read = client.post("/api/blocks", json={"category": "read_csv", "params": {"path": str(csv_path)}}).json()
    resp = client.post(f"/api/blocks/{read['id']}/draft", json={"instruction": "do something"})
    assert resp.status_code == 400


def test_draft_requires_nonempty_instruction(client):
    custom = client.post(
        "/api/blocks",
        json={
            "category": "custom_block",
            "block_type": "llm_authored",
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
            "block_type": "llm_authored",
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
            "block_type": "llm_authored",
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


def test_providers_endpoint_lists_registered_providers(client):
    resp = client.get("/api/llm/providers")
    assert resp.status_code == 200
    body = resp.json()
    assert {"stub", "claude_cli"} <= set(body["providers"])
    assert body["active"] == "stub"
