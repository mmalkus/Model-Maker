from modelmaker.cache import CacheStore
from modelmaker.graph import Graph, Wire
from modelmaker.runner import Runner

from .helpers import make_block


def _classification_graph(tmp_path):
    csv_path = tmp_path / "data.csv"
    csv_path.write_text("x,y\n1,0\n2,1\n3,0\n4,1\n5,0\n6,1\n")

    read = make_block("b_read", "read_csv", params={"path": str(csv_path)})
    logreg = make_block("b_logreg", "logistic_regression", params={"features": ["x"]}, x=1)
    graph = Graph(
        blocks={"b_read": read, "b_logreg": logreg},
        wires={"w1": Wire("w1", "b_read", "out", "b_logreg", "df")},
    )
    return graph


def _csv_graph(tmp_path):
    csv_path = tmp_path / "data.csv"
    csv_path.write_text("a,b\n1,10\n2,20\n3,30\n")

    read = make_block("b_read", "read_csv", params={"path": str(csv_path)})
    filt = make_block("b_filter", "filter", params={"expr": "a > 1"}, x=1)
    graph = Graph(
        lanes={},
        blocks={"b_read": read, "b_filter": filt},
        wires={"w1": Wire(id="w1", from_block="b_read", from_port="out", to_block="b_filter", to_port="df")},
    )
    return graph, csv_path


def test_status_reports_red_not_recursionerror_for_a_cyclic_graph():
    # A cycle can't be created through the API/session anymore (see
    # graph.creates_cycle), but a hand-built or manually-edited project file
    # could still load one -- status() must fail cleanly, not blow the stack.
    a = make_block("a", "filter", params={"expr": "1=1"})
    b = make_block("b", "filter", params={"expr": "1=1"})
    graph = Graph(
        blocks={"a": a, "b": b},
        wires={
            "w1": Wire("w1", "a", "out", "b", "df"),
            "w2": Wire("w2", "b", "out", "a", "df"),
        },
    )
    runner = Runner(graph, CacheStore())
    assert runner.status("a") == "red"
    assert "cycle" in runner.state["a"].last_error.lower()


def test_status_lifecycle_grey_to_green(tmp_path):
    graph, _ = _csv_graph(tmp_path)
    runner = Runner(graph, CacheStore())

    assert runner.status("b_read") == "grey"
    assert runner.status("b_filter") == "grey"

    runner.refresh("b_read")
    assert runner.status("b_read") == "green"
    assert runner.status("b_filter") == "grey"

    runner.run_block("b_filter")
    assert runner.status("b_filter") == "green"


def test_editing_params_cascades_orange_not_upstream(tmp_path):
    graph, _ = _csv_graph(tmp_path)
    runner = Runner(graph, CacheStore())
    runner.refresh("b_read")
    runner.run_block("b_filter")
    assert runner.status("b_filter") == "green"

    graph.blocks["b_filter"].params["expr"] = "a > 2"
    assert runner.status("b_filter") == "orange"
    assert runner.status("b_read") == "green"


def test_run_all_blocks_on_ungread_input_then_succeeds(tmp_path):
    graph, _ = _csv_graph(tmp_path)
    runner = Runner(graph, CacheStore())

    report = runner.run_all()
    assert report["b_filter"].startswith("blocked")
    assert runner.status("b_filter") == "grey"

    runner.refresh("b_read")
    report = runner.run_all()
    assert report["b_filter"] == "green"

    report2 = runner.run_all()
    assert report2["b_filter"] == "green"


def test_red_status_keeps_stale_green_output_and_recovers(tmp_path):
    graph, _ = _csv_graph(tmp_path)
    runner = Runner(graph, CacheStore())
    runner.refresh("b_read")
    runner.run_block("b_filter")
    assert runner.status("b_filter") == "green"
    good_key = runner.state["b_filter"].last_successful_key

    graph.blocks["b_filter"].params["expr"] = "not a valid expr((("
    runner.run_block("b_filter")
    assert runner.status("b_filter") == "red"
    assert runner.state["b_filter"].last_successful_key == good_key
    assert runner.cache.get(good_key) is not None

    graph.blocks["b_filter"].params["expr"] = "a > 1"
    assert runner.status("b_filter") == "green"


def test_force_run_all_recomputes_even_when_green(tmp_path):
    graph, _ = _csv_graph(tmp_path)
    runner = Runner(graph, CacheStore())
    runner.refresh("b_read")
    runner.run_all()
    key_before = runner.state["b_filter"].last_successful_key

    report = runner.force_run_all()
    assert report["b_filter"] == "green"
    assert runner.state["b_filter"].last_successful_key == key_before


def test_refresh_keeps_last_known_good_on_failed_reread(tmp_path):
    graph, csv_path = _csv_graph(tmp_path)
    runner = Runner(graph, CacheStore())
    runner.refresh("b_read")
    assert runner.status("b_read") == "green"
    good_key = runner.state["b_read"].last_successful_key
    read_at = runner.state["b_read"].last_successful_read_at

    graph.blocks["b_read"].params["path"] = str(csv_path.parent / "missing.csv")
    runner.refresh("b_read")
    assert runner.status("b_read") == "red"
    assert runner.state["b_read"].last_successful_key == good_key
    assert runner.state["b_read"].last_successful_read_at == read_at


def test_check_for_changes_flags_without_touching_cache(tmp_path):
    graph, csv_path = _csv_graph(tmp_path)
    runner = Runner(graph, CacheStore())
    runner.refresh("b_read")
    key_before = runner.state["b_read"].last_successful_key

    assert runner.check_for_changes("b_read") is False  # first probe just baselines

    csv_path.write_text("a,b\n1,10\n2,20\n3,30\n4,40\n")
    assert runner.check_for_changes("b_read") is True
    # a probe never mutates the cached packet or its key
    assert runner.state["b_read"].last_successful_key == key_before
    assert runner.status("b_read") == "green"


def test_custom_block_code_has_polars_available_without_local_import(tmp_path):
    # Regression: the compiled script gets `import polars as pl` for free at
    # module level, so LLM-drafted / hand-written custom block code is
    # written assuming `pl` is in scope. The live engine must match that,
    # not just the compile-to-script path.
    graph, _ = _csv_graph(tmp_path)
    graph.blocks["b_custom"] = make_block(
        "b_custom",
        "bucket_income",
        block_type="standard",
        code="def bucket_income(df: pl.DataFrame) -> pl.DataFrame:\n    return df.with_columns((pl.col('a') > 1).alias('high_a'))\n",
        inputs=[graph.blocks["b_filter"].inputs[0]],
        outputs=[graph.blocks["b_filter"].outputs[0]],
        metadata_transform={"kind": "declared", "base": "df", "adds": [{"name": "high_a", "dtype": "Boolean"}]},
        x=2,
    )
    graph.wires["w2"] = Wire("w2", "b_read", "out", "b_custom", "df")

    runner = Runner(graph, CacheStore())
    runner.refresh("b_read")
    status = runner.run_block("b_custom")
    assert status == "green", runner.state["b_custom"].last_error


def test_run_to_here_cascades_upstream(tmp_path):
    graph, _ = _csv_graph(tmp_path)
    runner = Runner(graph, CacheStore())
    runner.refresh("b_read")

    status = runner.run_to_here("b_filter")
    assert status == "green"
    assert runner.status("b_filter") == "green"


def test_two_generate_image_blocks_write_distinct_files(tmp_path):
    # Regression: generate_image used to always write to a hardcoded
    # "generate_image.png", so a second image block in the same pipeline
    # silently overwrote the first block's saved output file.
    graph, _ = _csv_graph(tmp_path)
    graph.blocks["b_img1"] = make_block(
        "b_img1", "generate_image", params={"kind": "hist", "x": "a"}, x=2
    )
    graph.blocks["b_img2"] = make_block(
        "b_img2", "generate_image", params={"kind": "hist", "x": "b"}, x=2, y=1
    )
    graph.wires["w2"] = Wire("w2", "b_read", "out", "b_img1", "df")
    graph.wires["w3"] = Wire("w3", "b_read", "out", "b_img2", "df")

    output_dir = tmp_path / "out"
    runner = Runner(graph, CacheStore(), output_dir=str(output_dir))
    runner.refresh("b_read")
    assert runner.run_block("b_img1") == "green"
    assert runner.run_block("b_img2") == "green"

    pngs = sorted(p.name for p in output_dir.glob("*.png"))
    assert len(pngs) == 2


def test_target_param_left_unset_fails_clearly_with_no_role_tagged(tmp_path):
    graph = _classification_graph(tmp_path)
    runner = Runner(graph, CacheStore())
    runner.refresh("b_read")

    assert runner.run_block("b_logreg") == "red"
    assert "target" in runner.state["b_logreg"].last_error.lower()


def test_target_param_auto_fills_from_role_tagged_upstream_column(tmp_path):
    graph = _classification_graph(tmp_path)
    graph.blocks["b_read"].column_role_overrides = {"y": "target"}
    runner = Runner(graph, CacheStore())
    runner.refresh("b_read")

    assert runner.run_block("b_logreg") == "green"
    predictions = runner.cache.get(runner.state["b_logreg"].last_successful_key).outputs["predictions"]
    assert predictions.schema_meta["predicted_proba"].role.value == "predicted"
    assert predictions.schema_meta["predicted_class"].role.value == "feature"
    assert predictions.schema_meta["y"].role.value == "target"


def test_explicit_target_param_overrides_the_role_tagged_default(tmp_path):
    # y is tagged target, but the block explicitly names x instead -- the
    # explicit choice must win, not the upstream tag (sklearn will happily
    # "fit" on a constant-ish column, this only checks which name was used).
    csv_path = tmp_path / "data.csv"
    csv_path.write_text("x,y,z\n1,0,0\n2,1,1\n3,0,0\n4,1,1\n5,0,0\n6,1,1\n")
    read = make_block("b_read", "read_csv", params={"path": str(csv_path)})
    logreg = make_block("b_logreg", "logistic_regression", params={"features": ["z"], "target": "x"}, x=1)
    graph = Graph(
        blocks={"b_read": read, "b_logreg": logreg},
        wires={"w1": Wire("w1", "b_read", "out", "b_logreg", "df")},
    )
    graph.blocks["b_read"].column_role_overrides = {"y": "target"}
    runner = Runner(graph, CacheStore())
    runner.refresh("b_read")

    assert runner.run_block("b_logreg") == "green"
    artifact = runner.cache.get(runner.state["b_logreg"].last_successful_key).outputs["model"]
    assert artifact["target"] == "x"


def test_retagging_the_target_column_invalidates_downstream_and_repropagates(tmp_path):
    graph = _classification_graph(tmp_path)
    graph.blocks["b_read"].column_role_overrides = {"y": "target"}
    runner = Runner(graph, CacheStore())
    runner.refresh("b_read")
    assert runner.run_block("b_logreg") == "green"

    # Retag: y is no longer target, x is (a nonsense model, but exercises
    # the propagation, not the statistics).
    graph.blocks["b_read"].column_role_overrides = {"x": "target"}
    assert runner.status("b_read") == "orange"
    runner.run_block("b_read")
    assert runner.status("b_logreg") == "orange"  # cache key changed via upstream, not yet re-run

    assert runner.run_block("b_logreg") == "green"
    artifact = runner.cache.get(runner.state["b_logreg"].last_successful_key).outputs["model"]
    assert artifact["target"] == "x"


def test_two_upstream_branches_tagging_the_same_role_collide_only_once_joined(tmp_path):
    csv1 = tmp_path / "a.csv"
    csv2 = tmp_path / "b.csv"
    csv1.write_text("id,y1\n1,0\n2,1\n")
    csv2.write_text("id,y2\n1,1\n2,0\n")

    r1 = make_block("b_r1", "read_csv", params={"path": str(csv1)})
    r2 = make_block("b_r2", "read_csv", params={"path": str(csv2)}, x=0, y=1)
    r1.column_role_overrides = {"y1": "target"}
    r2.column_role_overrides = {"y2": "target"}
    j = make_block("b_join", "join", params={"on": ["id"], "how": "inner"}, x=1)
    graph = Graph(
        blocks={"b_r1": r1, "b_r2": r2, "b_join": j},
        wires={
            "w1": Wire("w1", "b_r1", "out", "b_join", "left"),
            "w2": Wire("w2", "b_r2", "out", "b_join", "right"),
        },
    )
    runner = Runner(graph, CacheStore())
    runner.refresh("b_r1")
    runner.refresh("b_r2")

    assert runner.run_block("b_join") == "red"
    assert "target" in runner.state["b_join"].last_error.lower()
