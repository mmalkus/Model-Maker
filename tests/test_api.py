import pytest
from fastapi.testclient import TestClient

import modelmaker.api as api_module
from modelmaker.session import ProjectSession


@pytest.fixture
def client(tmp_path):
    # Recovery snapshots go to a temp path so a test run never writes one
    # into the working directory (see ProjectSession._write_recovery).
    api_module.SESSION = ProjectSession(recovery_path=tmp_path / "recovery.json")
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


def test_rename_port_sets_and_clears_a_data_name(client, tmp_path):
    csv_path = tmp_path / "data.csv"
    csv_path.write_text("a,b\n1,10\n2,20\n3,30\n")
    read = client.post("/api/blocks", json={"category": "read_csv", "params": {"path": str(csv_path)}}).json()

    resp = client.patch(f"/api/blocks/{read['id']}/port_name", json={"port": "out", "name": "raw applications"})
    assert resp.status_code == 200
    assert resp.json()["port_names"] == {"out": "raw applications"}

    graph = client.get("/api/graph").json()
    assert graph["blocks"][read["id"]]["port_names"] == {"out": "raw applications"}

    cleared = client.patch(f"/api/blocks/{read['id']}/port_name", json={"port": "out", "name": None})
    assert cleared.json()["port_names"] == {}

    bad_port = client.patch(f"/api/blocks/{read['id']}/port_name", json={"port": "nonexistent", "name": "x"})
    assert bad_port.status_code == 400


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


def test_wire_that_would_create_a_cycle_is_rejected_not_a_crash(client):
    # Regression: connecting two blocks into a loop used to crash the whole
    # app (RecursionError out of runner.compute_key on the very next graph
    # read) instead of a clean 4xx -- see graph.creates_cycle /
    # session.add_wire.
    a = client.post("/api/blocks", json={"category": "filter", "params": {"expr": "1=1"}}).json()
    b = client.post("/api/blocks", json={"category": "filter", "params": {"expr": "1=1"}, "x": 200}).json()

    first = client.post(
        "/api/wires", json={"from_block": a["id"], "from_port": "out", "to_block": b["id"], "to_port": "df"}
    )
    assert first.status_code == 200

    looped = client.post(
        "/api/wires", json={"from_block": b["id"], "from_port": "out", "to_block": a["id"], "to_port": "df"}
    )
    assert looped.status_code == 400

    # the graph is still perfectly readable afterwards
    graph = client.get("/api/graph")
    assert graph.status_code == 200
    assert len(graph.json()["wires"]) == 1


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
    assert "def filter(" in compiled["source"]


def test_run_all_streaming_endpoint(client, tmp_path):
    csv_path = tmp_path / "data.csv"
    csv_path.write_text("a,b\n" + "".join(f"{i},{i * 10}\n" for i in range(1, 11)))
    read = client.post("/api/blocks", json={"category": "read_csv", "params": {"path": str(csv_path)}}).json()
    filt = client.post("/api/blocks", json={"category": "filter", "params": {"expr": "a > 5"}, "x": 200}).json()
    client.post(
        "/api/wires",
        json={"from_block": read["id"], "from_port": "out", "to_block": filt["id"], "to_port": "df"},
    )

    # No refresh() needed first -- a fusable input block (read_csv) is
    # scanned live as part of the streaming run's fused chain.
    report = client.post("/api/run_all_streaming").json()
    assert report[filt["id"]] == "green"
    assert report[read["id"]] == "fused"

    preview = client.get(f"/api/blocks/{filt['id']}/preview").json()
    assert sorted(r["a"] for r in preview["rows"]) == [6, 7, 8, 9, 10]


def test_run_all_streaming_refuses_in_sample_mode(client, tmp_path):
    csv_path = tmp_path / "data.csv"
    csv_path.write_text("a\n" + "".join(f"{i}\n" for i in range(1, 6)))
    client.post("/api/blocks", json={"category": "read_csv", "params": {"path": str(csv_path)}})
    client.put("/api/sample_mode", json={"rows": 5})

    resp = client.post("/api/run_all_streaming")
    assert resp.status_code == 409
    assert "sample mode" in resp.json()["detail"]


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


def test_input_schema_lists_all_declared_ports_even_when_unwired(client):
    # Regression: a block whose input ports aren't named "df" (e.g.
    # psi_test's expected/actual) used to disappear entirely from the input
    # schema when unwired, which fed a misleading "df" fallback into the
    # AI-draft prompt.
    psi = client.post("/api/blocks", json={"category": "psi_test", "params": {"col": "x", "bins": 4}}).json()
    schema = client.get(f"/api/blocks/{psi['id']}/input_schema").json()
    assert schema == {"expected": [], "actual": []}


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


def test_column_role_tag_flows_downstream_and_auto_fills_target(client, tmp_path):
    read = _wired_read_csv(client, tmp_path, client.post("/api/blocks", json={"category": "display_table"}).json()["id"])

    resp = client.post(f"/api/blocks/{read['id']}/column_role", json={"column": "a", "role": "target"})
    assert resp.status_code == 200
    assert resp.json()["status"] == "orange"  # retagging invalidated the cached run, same as any param edit

    client.post(f"/api/blocks/{read['id']}/run")
    preview = client.get(f"/api/blocks/{read['id']}/preview").json()
    roles = {c["name"]: c["role"] for c in preview["columns"]}
    assert roles == {"a": "target", "b": "unassigned"}

    # clearing it back to unassigned
    client.post(f"/api/blocks/{read['id']}/run")
    resp = client.post(f"/api/blocks/{read['id']}/column_role", json={"column": "a", "role": "unassigned"})
    assert resp.status_code == 200
    client.post(f"/api/blocks/{read['id']}/run")
    preview2 = client.get(f"/api/blocks/{read['id']}/preview").json()
    assert {c["name"]: c["role"] for c in preview2["columns"]} == {"a": "unassigned", "b": "unassigned"}


def test_column_role_rejects_a_second_column_with_the_same_unique_role(client, tmp_path):
    read = _wired_read_csv(client, tmp_path, client.post("/api/blocks", json={"category": "display_table"}).json()["id"])
    client.post(f"/api/blocks/{read['id']}/column_role", json={"column": "a", "role": "target"})
    client.post(f"/api/blocks/{read['id']}/run")

    resp = client.post(f"/api/blocks/{read['id']}/column_role", json={"column": "b", "role": "target"})
    assert resp.status_code == 400
    assert "already set on 'a'" in resp.json()["detail"]


def test_column_role_rejects_unknown_role_and_manual_predicted(client, tmp_path):
    read = _wired_read_csv(client, tmp_path, client.post("/api/blocks", json={"category": "display_table"}).json()["id"])

    resp = client.post(f"/api/blocks/{read['id']}/column_role", json={"column": "a", "role": "nonsense"})
    assert resp.status_code == 400

    resp2 = client.post(f"/api/blocks/{read['id']}/column_role", json={"column": "a", "role": "predicted"})
    assert resp2.status_code == 400
    assert "predicted" in resp2.json()["detail"].lower()


def test_logistic_regression_predictions_are_tagged_predicted_role(client, tmp_path):
    logreg = client.post("/api/blocks", json={"category": "logistic_regression", "params": {"features": ["x"]}}).json()
    read = _wired_classification_csv(client, tmp_path, logreg["id"])
    client.post(f"/api/blocks/{read['id']}/column_role", json={"column": "y", "role": "target"})
    client.post(f"/api/blocks/{read['id']}/run")

    resp = client.post(f"/api/blocks/{logreg['id']}/run")
    assert resp.json()["status"] == "green", resp.json()

    preview = client.get(f"/api/blocks/{logreg['id']}/preview", params={"port": "predictions"}).json()
    roles = {c["name"]: c["role"] for c in preview["columns"]}
    assert roles["predicted_proba"] == "predicted"
    assert roles["predicted_class"] == "feature"
    assert roles["y"] == "target"


def test_undo_and_redo_endpoints_round_trip_a_block(client):
    block = client.post("/api/blocks", json={"category": "filter", "params": {"expr": "a > 1"}}).json()
    graph = client.get("/api/graph").json()
    assert graph["can_undo"] is True and graph["can_redo"] is False
    assert graph["dirty"] is True

    graph = client.post("/api/undo").json()
    assert block["id"] not in graph["blocks"]
    assert graph["can_redo"] is True

    graph = client.post("/api/redo").json()
    assert block["id"] in graph["blocks"]


def test_undo_with_nothing_to_undo_is_a_conflict_not_a_crash(client):
    assert client.post("/api/undo").status_code == 409
    assert client.post("/api/redo").status_code == 409


def test_dirty_clears_once_the_project_is_saved(client, tmp_path):
    client.post("/api/blocks", json={"category": "filter", "params": {"expr": "a > 1"}})
    assert client.get("/api/graph").json()["dirty"] is True

    client.post("/api/project/save", json={"path": str(tmp_path / "proj.json")})
    assert client.get("/api/graph").json()["dirty"] is False


def test_sample_mode_turns_a_green_block_orange_with_a_reason(client, tmp_path):
    csv_path = tmp_path / "data.csv"
    csv_path.write_text("a\n" + "".join(f"{i}\n" for i in range(1, 21)))
    read = client.post("/api/blocks", json={"category": "read_csv", "params": {"path": str(csv_path)}}).json()
    select = client.post("/api/blocks", json={"category": "select", "params": {"cols": ["a"]}, "x": 200}).json()
    client.post(
        "/api/wires",
        json={"from_block": read["id"], "from_port": "out", "to_block": select["id"], "to_port": "df"},
    )
    client.post(f"/api/blocks/{read['id']}/refresh")
    client.post(f"/api/blocks/{select['id']}/run")
    assert client.get("/api/graph").json()["blocks"][select["id"]]["status"] == "green"

    graph = client.put("/api/sample_mode", json={"rows": 5}).json()
    assert graph["sample_rows"] == 5
    stale = graph["blocks"][select["id"]]
    assert stale["status"] == "orange"
    assert "sample mode" in stale["stale_reason"]

    client.post(f"/api/blocks/{read['id']}/refresh")
    client.post(f"/api/blocks/{select['id']}/run")
    assert client.get(f"/api/blocks/{select['id']}/preview").json()["row_count"] == 5

    # Turning it off goes stale again -- the source has been re-read since
    # the full-data run, so those cached results are genuinely not current --
    # and re-running gives the whole dataset back.
    graph = client.put("/api/sample_mode", json={"rows": None}).json()
    assert graph["blocks"][select["id"]]["status"] == "orange"
    client.post(f"/api/blocks/{read['id']}/refresh")
    client.post(f"/api/blocks/{select['id']}/run")
    assert client.get(f"/api/blocks/{select['id']}/preview").json()["row_count"] == 20


def test_sample_mode_rejects_a_nonsense_row_count(client):
    assert client.put("/api/sample_mode", json={"rows": 0}).status_code == 400


def test_stale_reason_is_absent_for_green_and_grey_blocks(client, tmp_path):
    csv_path = tmp_path / "data.csv"
    csv_path.write_text("a\n1\n2\n")
    read = client.post("/api/blocks", json={"category": "read_csv", "params": {"path": str(csv_path)}}).json()
    assert read["stale_reason"] is None  # grey

    client.post(f"/api/blocks/{read['id']}/refresh")
    assert client.get("/api/graph").json()["blocks"][read["id"]]["stale_reason"] is None  # green


def test_recovery_is_offered_after_edits_and_rebuilds_the_graph(client, tmp_path):
    client.post("/api/blocks", json={"category": "filter", "params": {"expr": "a > 1"}})
    info = client.get("/api/project/recovery").json()["recovery"]
    assert info["block_count"] == 1

    # A fresh session (as after a restart) can pick that work back up.
    api_module.SESSION = ProjectSession(recovery_path=tmp_path / "recovery.json")
    assert client.get("/api/graph").json()["blocks"] == {}
    graph = client.post("/api/project/recover").json()
    assert len(graph["blocks"]) == 1
