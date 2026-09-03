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


def test_registry_groups_modelling_and_tests_blocks(client):
    resp = client.get("/api/registry")
    by_category = {b["category"]: b for b in resp.json()}
    for cat in ("glm_fit", "logistic_regression", "woe_transform"):
        assert by_category[cat]["group"] == "modelling"
    for cat in ("ks_test", "auc_gini", "psi_test"):
        assert by_category[cat]["group"] == "tests"
    # group is cosmetic only -- block_type still governs runtime role
    assert by_category["logistic_regression"]["block_type"] == "standard"
    assert by_category["ks_test"]["block_type"] == "output"
    # an unrelated block with no explicit group falls back to its block_type
    assert by_category["filter"]["group"] == "standard"


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


def _wired_read_csv(client, tmp_path, target_id, target_port="df"):
    csv_path = tmp_path / "data.csv"
    csv_path.write_text("a,b\n1,10\n2,20\n3,30\n")
    read = client.post("/api/blocks", json={"category": "read_csv", "params": {"path": str(csv_path)}}).json()
    client.post(
        "/api/wires",
        json={"from_block": read["id"], "from_port": "out", "to_block": target_id, "to_port": target_port},
    )
    client.post(f"/api/blocks/{read['id']}/refresh")
    return read


def test_display_table_passes_data_through_for_preview(client, tmp_path):
    display = client.post("/api/blocks", json={"category": "display_table", "x": 200}).json()
    _wired_read_csv(client, tmp_path, display["id"])

    run_resp = client.post(f"/api/blocks/{display['id']}/run")
    assert run_resp.json()["status"] == "green"

    preview = client.get(f"/api/blocks/{display['id']}/preview").json()
    assert preview["row_count"] == 3
    assert {r["a"] for r in preview["rows"]} == {1, 2, 3}


def test_generate_image_produces_a_png(client, tmp_path):
    img_block = client.post(
        "/api/blocks", json={"category": "generate_image", "params": {"kind": "hist", "x": "a"}, "x": 200}
    ).json()
    _wired_read_csv(client, tmp_path, img_block["id"])

    run_resp = client.post(f"/api/blocks/{img_block['id']}/run")
    assert run_resp.json()["status"] == "green"

    resp = client.get(f"/api/blocks/{img_block['id']}/image")
    assert resp.status_code == 200
    assert resp.headers["content-type"] == "image/png"
    assert resp.content[:8] == b"\x89PNG\r\n\x1a\n"


def test_image_endpoint_rejects_a_dataframe_block(client, tmp_path):
    read = _wired_read_csv(client, tmp_path, client.post("/api/blocks", json={"category": "filter", "params": {"expr": "a > 0"}}).json()["id"])
    resp = client.get(f"/api/blocks/{read['id']}/image")
    assert resp.status_code == 400


def test_custom_block_can_be_an_input_block(client):
    # An AI-authored block isn't forced into "standard" (mid-pipeline
    # transform) role -- block_type is orthogonal to is_custom, so a custom
    # block can be a real input block: no upstream required, and eligible
    # for refresh/check_for_changes like read_csv.
    block = client.post(
        "/api/blocks",
        json={
            "category": "synthetic_source",
            "block_type": "input",
            "inputs": [],
            "outputs": [{"name": "out", "type": "dataframe"}],
            "code": "def synthetic_source():\n    import polars as pl\n    return pl.DataFrame({'x': [1, 2, 3]})\n",
            "metadata_transform": {"kind": "infer_dtypes"},
        },
    ).json()
    assert block["block_type"] == "input"
    assert block["is_custom"] is True

    resp = client.post(f"/api/blocks/{block['id']}/refresh")
    assert resp.status_code == 200
    assert resp.json()["status"] == "green"


def test_custom_block_can_be_an_output_block(client, tmp_path):
    read = client.post("/api/blocks", json={"category": "read_csv", "params": {"path": str(tmp_path / "x.csv")}}).json()
    (tmp_path / "x.csv").write_text("a\n1\n2\n")
    custom = client.post(
        "/api/blocks",
        json={
            "category": "row_count_report",
            "block_type": "output",
            "inputs": [{"name": "df", "type": "dataframe"}],
            "outputs": [],
            "code": "def row_count_report(df, output_dir='.'):\n    pass\n",
            "metadata_transform": {"kind": "passthrough"},
        },
    ).json()
    assert custom["block_type"] == "output"
    assert custom["is_custom"] is True
    client.post(
        "/api/wires",
        json={"from_block": read["id"], "from_port": "out", "to_block": custom["id"], "to_port": "df"},
    )
    client.post(f"/api/blocks/{read['id']}/refresh")
    resp = client.post(f"/api/blocks/{custom['id']}/run")
    assert resp.status_code == 200
    assert resp.json()["status"] == "green"


def _wired_classification_csv(client, tmp_path, target_id, target_port="df"):
    # Mostly-separable by x (low x -> 0, high x -> 1) but with a couple of
    # flips near the boundary (x=8,10 -> 1 early; x=11 -> 0 late), so
    # LogisticRegression fits cleanly without a perfect-separation warning.
    y = [0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    csv_path = tmp_path / "clf.csv"
    rows = ["x,y"] + [f"{x},{yi}" for x, yi in zip(range(1, 21), y)]
    csv_path.write_text("\n".join(rows) + "\n")
    read = client.post("/api/blocks", json={"category": "read_csv", "params": {"path": str(csv_path)}}).json()
    client.post(
        "/api/wires",
        json={"from_block": read["id"], "from_port": "out", "to_block": target_id, "to_port": target_port},
    )
    client.post(f"/api/blocks/{read['id']}/refresh")
    return read


def test_logistic_regression_fits_and_scores(client, tmp_path):
    logreg = client.post(
        "/api/blocks",
        json={"category": "logistic_regression", "params": {"target": "y", "features": ["x"]}},
    ).json()
    _wired_classification_csv(client, tmp_path, logreg["id"])

    run_resp = client.post(f"/api/blocks/{logreg['id']}/run")
    assert run_resp.json()["status"] == "green", run_resp.json()

    preview = client.get(f"/api/blocks/{logreg['id']}/preview", params={"port": "predictions"}).json()
    cols = {c["name"] for c in preview["columns"]}
    assert {"x", "y", "predicted_proba", "predicted_class"} <= cols
    assert preview["row_count"] == 20

    st = api_module.SESSION.runner.state[logreg["id"]]
    model = api_module.SESSION.runner.cache.get(st.last_successful_key).outputs["model"]
    assert model["kind"] == "logistic_regression"
    assert set(model["coefficients"]) == {"x"}


def test_glm_fit_smoke(client, tmp_path):
    glm = client.post(
        "/api/blocks",
        json={"category": "glm_fit", "params": {"target": "y", "features": ["x"], "family": "gaussian"}},
    ).json()
    _wired_classification_csv(client, tmp_path, glm["id"])
    resp = client.post(f"/api/blocks/{glm['id']}/run")
    assert resp.json()["status"] == "green", resp.json()


def test_woe_transform_adds_expected_column(client, tmp_path):
    woe = client.post("/api/blocks", json={"category": "woe_transform", "params": {"col": "x", "target": "y", "bins": 4}}).json()
    _wired_classification_csv(client, tmp_path, woe["id"])
    resp = client.post(f"/api/blocks/{woe['id']}/run")
    assert resp.json()["status"] == "green", resp.json()

    preview = client.get(f"/api/blocks/{woe['id']}/preview").json()
    assert "x_woe" in {c["name"] for c in preview["columns"]}
    assert all(r["x_woe"] is not None for r in preview["rows"])


def test_ks_and_auc_gini_on_model_predictions(client, tmp_path):
    logreg = client.post(
        "/api/blocks",
        json={"category": "logistic_regression", "params": {"target": "y", "features": ["x"]}},
    ).json()
    _wired_classification_csv(client, tmp_path, logreg["id"])
    client.post(f"/api/blocks/{logreg['id']}/run")

    ks = client.post(
        "/api/blocks", json={"category": "ks_test", "params": {"score_col": "predicted_proba", "target_col": "y"}}
    ).json()
    auc = client.post(
        "/api/blocks", json={"category": "auc_gini", "params": {"score_col": "predicted_proba", "target_col": "y"}}
    ).json()
    for test_block in (ks, auc):
        client.post(
            "/api/wires",
            json={"from_block": logreg["id"], "from_port": "predictions", "to_block": test_block["id"], "to_port": "df"},
        )

    ks_resp = client.post(f"/api/blocks/{ks['id']}/run")
    auc_resp = client.post(f"/api/blocks/{auc['id']}/run")
    assert ks_resp.json()["status"] == "green", ks_resp.json()
    assert auc_resp.json()["status"] == "green", auc_resp.json()

    ks_metric = api_module.SESSION.runner.cache.get(
        api_module.SESSION.runner.state[ks["id"]].last_successful_key
    ).outputs["metric"]
    auc_metric = api_module.SESSION.runner.cache.get(
        api_module.SESSION.runner.state[auc["id"]].last_successful_key
    ).outputs["metric"]
    assert 0 < ks_metric["ks_statistic"] <= 1
    assert 0.5 < auc_metric["auc"] <= 1  # clearly-separable synthetic data
    assert auc_metric["gini"] == pytest.approx(2 * auc_metric["auc"] - 1)


def test_psi_test_compares_two_inputs(client, tmp_path):
    psi = client.post("/api/blocks", json={"category": "psi_test", "params": {"col": "x", "bins": 4}}).json()
    expected = _wired_classification_csv(client, tmp_path, psi["id"], target_port="expected")
    # reuse the same data as "actual" too -- comparing a distribution to
    # itself should yield ~0 drift
    client.post(
        "/api/wires",
        json={"from_block": expected["id"], "from_port": "out", "to_block": psi["id"], "to_port": "actual"},
    )
    resp = client.post(f"/api/blocks/{psi['id']}/run")
    assert resp.json()["status"] == "green", resp.json()

    metric = api_module.SESSION.runner.cache.get(
        api_module.SESSION.runner.state[psi["id"]].last_successful_key
    ).outputs["metric"]
    assert metric["psi"] == pytest.approx(0.0, abs=1e-9)


def test_display_value_wires_up_to_a_scalar_metric_port(client, tmp_path):
    # display_value's ports are typed "any" -- wiring a scalar_metric port
    # (auc_gini) into it must be accepted despite the type mismatch.
    logreg = client.post(
        "/api/blocks",
        json={"category": "logistic_regression", "params": {"target": "y", "features": ["x"]}},
    ).json()
    _wired_classification_csv(client, tmp_path, logreg["id"])
    client.post(f"/api/blocks/{logreg['id']}/run")

    auc = client.post(
        "/api/blocks", json={"category": "auc_gini", "params": {"score_col": "predicted_proba", "target_col": "y"}}
    ).json()
    client.post(
        "/api/wires",
        json={"from_block": logreg["id"], "from_port": "predictions", "to_block": auc["id"], "to_port": "df"},
    )
    client.post(f"/api/blocks/{auc['id']}/run")

    viewer = client.post("/api/blocks", json={"category": "display_value"}).json()
    wire = client.post(
        "/api/wires",
        json={"from_block": auc["id"], "from_port": "metric", "to_block": viewer["id"], "to_port": "value"},
    ).json()
    assert wire["valid"] is True

    run_resp = client.post(f"/api/blocks/{viewer['id']}/run")
    assert run_resp.json()["status"] == "green", run_resp.json()

    value_resp = client.get(f"/api/blocks/{viewer['id']}/value")
    assert value_resp.status_code == 200
    body = value_resp.json()
    assert body["kind"] == "auc_gini"
    assert 0.5 < body["auc"] <= 1


def test_display_value_also_works_for_a_dataframe(client, tmp_path):
    viewer = client.post("/api/blocks", json={"category": "display_value"}).json()
    _wired_classification_csv(client, tmp_path, viewer["id"], target_port="value")
    run_resp = client.post(f"/api/blocks/{viewer['id']}/run")
    assert run_resp.json()["status"] == "green", run_resp.json()

    preview = client.get(f"/api/blocks/{viewer['id']}/preview").json()
    assert preview["row_count"] == 20
    assert {"x", "y"} <= {c["name"] for c in preview["columns"]}

    # /value correctly refuses since this port turned out to hold a dataframe
    assert client.get(f"/api/blocks/{viewer['id']}/value").status_code == 400


def test_value_endpoint_rejects_an_image_port(client, tmp_path):
    img_block = client.post("/api/blocks", json={"category": "generate_image", "params": {"kind": "hist", "x": "a"}}).json()
    _wired_read_csv(client, tmp_path, img_block["id"])
    client.post(f"/api/blocks/{img_block['id']}/run")
    resp = client.get(f"/api/blocks/{img_block['id']}/value")
    assert resp.status_code == 400
