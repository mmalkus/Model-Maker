from modelmaker.cache import CacheStore
from modelmaker.graph import Graph, Wire
from modelmaker.runner import Runner

from .helpers import make_block


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
        block_type="llm_authored",
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
