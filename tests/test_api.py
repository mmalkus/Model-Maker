import pytest
from fastapi.testclient import TestClient

import modelmaker.api as api_module
from modelmaker.session import ProjectSession


@pytest.fixture
def client():
    api_module.SESSION = ProjectSession()
    return TestClient(api_module.app)


def test_registry_lists_standard_blocks(client):
    resp = client.get("/api/registry")
    assert resp.status_code == 200
    categories = {b["category"] for b in resp.json()}
    assert {"read_csv", "filter", "select", "join", "train_test_split", "write_csv"} <= categories


def test_create_wire_run_and_preview(client, tmp_path):
    csv_path = tmp_path / "data.csv"
    csv_path.write_text("a,b\n1,10\n2,20\n3,30\n")

    read = client.post("/api/blocks", json={"category": "read_csv", "params": {"path": str(csv_path)}}).json()
    filt = client.post(
        "/api/blocks", json={"category": "filter", "params": {"expr": "a > 1"}, "x": 200}
    ).json()

    assert read["status"] == "grey"

    wire = client.post(
        "/api/wires",
        json={"from_block": read["id"], "from_port": "out", "to_block": filt["id"], "to_port": "df"},
    ).json()
    assert wire["valid"] is True

    client.post(f"/api/blocks/{read['id']}/refresh")
    graph = client.get("/api/graph").json()
    assert graph["blocks"][read["id"]]["status"] == "green"

    run_resp = client.post(f"/api/blocks/{filt['id']}/run")
    assert run_resp.json()["status"] == "green"

    preview = client.get(f"/api/blocks/{filt['id']}/preview").json()
    assert preview["row_count"] == 2
    assert {r["a"] for r in preview["rows"]} == {2, 3}


def test_invalid_wire_type_mismatch_is_flagged_not_rejected(client, tmp_path):
    csv_path = tmp_path / "data.csv"
    csv_path.write_text("a,b\n1,10\n")
    read = client.post("/api/blocks", json={"category": "read_csv", "params": {"path": str(csv_path)}}).json()
    split = client.post("/api/blocks", json={"category": "train_test_split", "x": 200}).json()

    wire = client.post(
        "/api/wires",
        json={"from_block": read["id"], "from_port": "out", "to_block": split["id"], "to_port": "nonexistent"},
    )
    assert wire.status_code == 200
    assert wire.json()["valid"] is False


def test_run_all_and_compile(client, tmp_path):
    csv_path = tmp_path / "data.csv"
    csv_path.write_text("a,b\n1,10\n2,20\n3,30\n")
    read = client.post("/api/blocks", json={"category": "read_csv", "params": {"path": str(csv_path)}}).json()
    filt = client.post("/api/blocks", json={"category": "filter", "params": {"expr": "a > 1"}, "x": 200}).json()
    client.post(
        "/api/wires",
        json={"from_block": read["id"], "from_port": "out", "to_block": filt["id"], "to_port": "df"},
    )

    blocked = client.post("/api/run_all").json()
    assert blocked[filt["id"]].startswith("blocked")

    client.post(f"/api/blocks/{read['id']}/refresh")
    report = client.post("/api/run_all").json()
    assert report[filt["id"]] == "green"

    compiled = client.post("/api/compile", json={}).json()
    assert "def filter_" in compiled["source"]


def test_compile_before_run_returns_409(client, tmp_path):
    csv_path = tmp_path / "data.csv"
    csv_path.write_text("a,b\n1,10\n")
    client.post("/api/blocks", json={"category": "read_csv", "params": {"path": str(csv_path)}})
    resp = client.post("/api/compile", json={})
    assert resp.status_code == 409


def test_save_and_load_round_trip(client, tmp_path):
    csv_path = tmp_path / "data.csv"
    csv_path.write_text("a,b\n1,10\n")
    client.post("/api/blocks", json={"category": "read_csv", "name": "Load data", "params": {"path": str(csv_path)}})

    project_path = tmp_path / "project.json"
    save_resp = client.post("/api/project/save", json={"path": str(project_path)})
    assert save_resp.status_code == 200
    assert project_path.exists()

    api_module.SESSION = ProjectSession()
    client2 = TestClient(api_module.app)
    load_resp = client2.post("/api/project/load", json={"path": str(project_path)})
    assert load_resp.status_code == 200
    blocks = load_resp.json()["blocks"]
    assert len(blocks) == 1
    assert next(iter(blocks.values()))["name"] == "Load data"


def test_delete_block_removes_dependent_wires(client, tmp_path):
    csv_path = tmp_path / "data.csv"
    csv_path.write_text("a,b\n1,10\n")
    read = client.post("/api/blocks", json={"category": "read_csv", "params": {"path": str(csv_path)}}).json()
    filt = client.post("/api/blocks", json={"category": "filter", "params": {"expr": "a > 0"}, "x": 200}).json()
    client.post(
        "/api/wires",
        json={"from_block": read["id"], "from_port": "out", "to_block": filt["id"], "to_port": "df"},
    )

    client.delete(f"/api/blocks/{read['id']}")
    graph = client.get("/api/graph").json()
    assert graph["wires"] == {}
