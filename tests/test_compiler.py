import pytest

from modelmaker.cache import CacheStore
from modelmaker.compiler import CompileError, compile_graph
from modelmaker.graph import Graph, Wire
from modelmaker.runner import Runner

from .helpers import make_block


def _csv_graph(tmp_path):
    csv_path = tmp_path / "data.csv"
    csv_path.write_text("a,b\n1,10\n2,20\n3,30\n")

    read = make_block("b_read", "read_csv", params={"path": str(csv_path)})
    filt = make_block("b_filter", "filter", params={"expr": "a > 1"}, x=1)
    select = make_block("b_select", "select", params={"cols": ["a"]}, x=2)
    graph = Graph(
        blocks={"b_read": read, "b_filter": filt, "b_select": select},
        wires={
            "w1": Wire("w1", "b_read", "out", "b_filter", "df"),
            "w2": Wire("w2", "b_filter", "out", "b_select", "df"),
        },
    )
    return graph, csv_path


def test_compile_refuses_when_a_block_is_grey(tmp_path):
    graph, _ = _csv_graph(tmp_path)
    runner = Runner(graph, CacheStore())
    with pytest.raises(CompileError):
        compile_graph(graph, runner=runner)


def test_compiled_script_runs_and_matches_engine_output(tmp_path):
    graph, _ = _csv_graph(tmp_path)
    runner = Runner(graph, CacheStore())
    runner.refresh("b_read")
    runner.run_all()

    source = compile_graph(graph, runner=runner)
    assert "# === Function: read_csv | type=input | category=read_csv ===" in source
    assert '# --- Call: b_select | name="b_select" ---' in source
    assert "OUTPUT_DIR" in source

    ns = {}
    exec(compile(source, "<compiled>", "exec"), ns)

    engine_out = runner.cache.get(runner.state["b_select"].last_successful_key).outputs["out"]
    compiled_out = ns["b_select_b_select"]
    assert compiled_out.to_dicts() == engine_out.data.to_dicts()


def test_compile_scopes_to_requested_output_only(tmp_path):
    graph, _ = _csv_graph(tmp_path)
    runner = Runner(graph, CacheStore())
    runner.refresh("b_read")
    runner.run_all()

    source = compile_graph(graph, runner=runner, output_blocks=["b_filter"])
    assert "b_select" not in source
    assert "b_filter" in source


def test_output_blocks_only_get_output_dir_kwarg_when_their_signature_wants_it(tmp_path, monkeypatch):
    # compiled write_csv writes a real file under OUTPUT_DIR when exec'd below
    monkeypatch.setenv("OUTPUT_DIR", str(tmp_path / "compiled_output"))
    # write_csv takes output_dir; display_table doesn't. Mixing them exercises
    # the compiler's per-block check rather than a blanket "block_type ==
    # output" assumption, which would crash display_table with an
    # unexpected-keyword TypeError at runtime.
    graph, _ = _csv_graph(tmp_path)
    graph.blocks["b_display"] = make_block("b_display", "display_table", x=3)
    graph.blocks["b_write"] = make_block("b_write", "write_csv", params={"filename": "out.csv"}, x=4)
    graph.wires["w3"] = Wire("w3", "b_select", "out", "b_display", "df")
    graph.wires["w4"] = Wire("w4", "b_select", "out", "b_write", "df")

    runner = Runner(graph, CacheStore())
    runner.refresh("b_read")
    report = runner.run_all()
    assert report["b_display"] == "green"
    assert report["b_write"] == "green"

    source = compile_graph(graph, runner=runner)
    assert "display_table(df=" in source
    assert "output_dir=OUTPUT_DIR" in source
    # the display_table call site must not have picked up output_dir too
    display_call_line = next(line for line in source.splitlines() if "display_table(" in line and "=" in line)
    assert "output_dir" not in display_call_line

    ns = {}
    exec(compile(source, "<compiled>", "exec"), ns)  # would raise TypeError if the bug regressed


def test_compiled_generate_image_calls_get_a_distinct_block_id(tmp_path):
    # Regression: generate_image's compiled call site must carry block_id
    # (like output_dir) so two image blocks in one script don't collide on
    # the same output filename.
    graph, _ = _csv_graph(tmp_path)
    graph.blocks["b_img"] = make_block("b_img", "generate_image", params={"kind": "hist", "x": "a"}, x=3)
    graph.wires["w3"] = Wire("w3", "b_select", "out", "b_img", "df")

    runner = Runner(graph, CacheStore())
    runner.refresh("b_read")
    assert runner.run_all()["b_img"] == "green"

    source = compile_graph(graph, runner=runner)
    call_line = next(line for line in source.splitlines() if "generate_image(" in line and "=" in line)
    assert "block_id='b_img'" in call_line


def test_two_blocks_of_the_same_type_compile_to_one_shared_function(tmp_path):
    # Two "filter" blocks (same registry category => identical source) must
    # produce exactly one function definition, called twice with each
    # instance's own params -- not two near-duplicate function bodies.
    graph, _ = _csv_graph(tmp_path)
    graph.blocks["b_filter2"] = make_block("b_filter2", "filter", params={"expr": "a > 2"}, x=3)
    graph.wires["w3"] = Wire("w3", "b_read", "out", "b_filter2", "df")

    runner = Runner(graph, CacheStore())
    runner.refresh("b_read")
    runner.run_all()

    source = compile_graph(graph, runner=runner)
    assert source.count("# === Function: filter |") == 1
    assert source.count("def filter(") == 1
    assert '# --- Call: b_filter | name="b_filter" ---' in source
    assert '# --- Call: b_filter2 | name="b_filter2" ---' in source

    shared_fn = next(line for line in source.splitlines() if line.startswith("def filter(")).split("(")[0][len("def ") :]
    assert f"{shared_fn}(df=" in source
    assert source.count(f"{shared_fn}(df=") == 2

    ns = {}
    exec(compile(source, "<compiled>", "exec"), ns)
    assert ns["b_filter_b_filter"].to_dicts() != ns["b_filter2_b_filter2"].to_dicts()


def test_compiled_calls_are_sectioned_by_lane(tmp_path):
    from modelmaker.graph import Lane

    graph, _ = _csv_graph(tmp_path)
    graph.lanes["prep"] = Lane("Data Prep", 0)
    graph.lanes["report"] = Lane("Reporting", 1)
    graph.blocks["b_read"].lane = "prep"
    graph.blocks["b_filter"].lane = "prep"
    graph.blocks["b_select"].lane = "report"

    runner = Runner(graph, CacheStore())
    runner.refresh("b_read")
    runner.run_all()

    source = compile_graph(graph, runner=runner)
    assert "# ===== Lane: Data Prep =====" in source
    assert "# ===== Lane: Reporting =====" in source
    assert source.index("# ===== Lane: Data Prep =====") < source.index("# ===== Lane: Reporting =====")


def test_compile_bakes_in_the_role_tagged_target_when_param_left_unset(tmp_path):
    # target is never in logreg's own params -- it's resolved from the
    # upstream role=target tag, same as the live runner does (see
    # test_runner.test_target_param_auto_fills_from_role_tagged_upstream_column),
    # and baked into the generated call as a literal, same as output_dir/block_id.
    csv_path = tmp_path / "clf.csv"
    csv_path.write_text("x,y\n1,0\n2,1\n3,0\n4,1\n5,0\n6,1\n")
    read = make_block("b_read", "read_csv", params={"path": str(csv_path)})
    read.column_role_overrides = {"y": "target"}
    logreg = make_block("b_logreg", "logistic_regression", params={"features": ["x"]}, x=1)
    graph = Graph(
        blocks={"b_read": read, "b_logreg": logreg},
        wires={"w1": Wire("w1", "b_read", "out", "b_logreg", "df")},
    )

    runner = Runner(graph, CacheStore())
    runner.refresh("b_read")
    assert runner.run_block("b_logreg") == "green"

    source = compile_graph(graph, runner=runner)
    call_line = next(line for line in source.splitlines() if "logistic_regression(" in line and "=" in line)
    assert "target='y'" in call_line


def test_compile_bakes_in_the_role_tagged_predicted_column_when_param_left_unset(tmp_path):
    # score_col is never in auc_gini's own params -- it's resolved from the
    # upstream role=predicted tag left by logistic_regression, same as the
    # live runner (see test_runner.test_predicted_param_auto_fills_from_role_tagged_upstream_column).
    csv_path = tmp_path / "clf.csv"
    csv_path.write_text("x,y\n1,0\n2,1\n3,0\n4,1\n5,0\n6,1\n")
    read = make_block("b_read", "read_csv", params={"path": str(csv_path)})
    read.column_role_overrides = {"y": "target"}
    logreg = make_block("b_logreg", "logistic_regression", params={"features": ["x"]}, x=1)
    gini = make_block("b_gini", "auc_gini", x=2)
    graph = Graph(
        blocks={"b_read": read, "b_logreg": logreg, "b_gini": gini},
        wires={
            "w1": Wire("w1", "b_read", "out", "b_logreg", "df"),
            "w2": Wire("w2", "b_logreg", "predictions", "b_gini", "df"),
        },
    )

    runner = Runner(graph, CacheStore())
    runner.refresh("b_read")
    assert runner.run_block("b_logreg") == "green"
    assert runner.run_block("b_gini") == "green"

    source = compile_graph(graph, runner=runner)
    call_line = next(line for line in source.splitlines() if "auc_gini(" in line and "=" in line)
    assert "target_col='y'" in call_line
    assert "score_col='predicted_proba'" in call_line


def test_custom_block_compiles_to_a_function_named_after_the_block_not_its_category(tmp_path):
    graph, _ = _csv_graph(tmp_path)
    graph.blocks["b_custom"] = make_block(
        "b_custom",
        "ai_block_lz3k9f",
        code="def ai_block_lz3k9f(df):\n    return df\n",
        x=3,
    )
    graph.blocks["b_custom"].name = "Score bucketer"
    graph.wires["w3"] = Wire("w3", "b_select", "out", "b_custom", "df")

    runner = Runner(graph, CacheStore())
    runner.refresh("b_read")
    assert runner.run_all()["b_custom"] == "green"

    source = compile_graph(graph, runner=runner)
    assert "def Score_bucketer(" in source
    assert "def ai_block_lz3k9f(" not in source


def test_compiled_grouped_block_matches_engine_output(tmp_path):
    # A group_by block compiles to a partition/dispatch/recombine sequence
    # (see compiler._grouped_call_lines) instead of a single call -- the
    # exec'd script's result must match what the live engine already
    # computed for the same grouped run (see
    # test_runner.test_group_by_produces_one_metric_row_per_group).
    csv_path = tmp_path / "scores.csv"
    csv_path.write_text(
        "region,score,target\n"
        "north,0.9,1\nnorth,0.1,0\nnorth,0.8,1\nnorth,0.2,0\n"
        "south,0.7,1\nsouth,0.3,0\nsouth,0.6,1\nsouth,0.4,0\n"
    )
    read = make_block("b_read", "read_csv", params={"path": str(csv_path)})
    gini = make_block("b_gini", "auc_gini", params={"score_col": "score", "target_col": "target"}, x=1)
    gini.group_by = "region"
    graph = Graph(
        blocks={"b_read": read, "b_gini": gini},
        wires={"w1": Wire("w1", "b_read", "out", "b_gini", "df")},
    )
    runner = Runner(graph, CacheStore())
    runner.refresh("b_read")
    assert runner.run_block("b_gini") == "green"

    source = compile_graph(graph, runner=runner)
    assert "ThreadPoolExecutor" in source
    assert "_combine_group_results" in source

    ns = {}
    exec(compile(source, "<compiled>", "exec"), ns)

    engine_out = runner.cache.get(runner.state["b_gini"].last_successful_key).outputs["metric"]
    compiled_out = ns["b_gini_b_gini"]
    assert compiled_out.sort("region").to_dicts() == engine_out.data.sort("region").to_dicts()


def test_naming_a_port_uses_that_name_as_the_compiled_variable(tmp_path):
    graph, _ = _csv_graph(tmp_path)
    graph.blocks["b_select"].port_names = {"out": "clean_rows"}

    runner = Runner(graph, CacheStore())
    runner.refresh("b_read")
    runner.run_all()

    source = compile_graph(graph, runner=runner)
    assert "clean_rows = " in source
    assert "b_select_b_select" not in source


def test_stream_false_by_default_leaves_compiled_output_unchanged(tmp_path):
    graph, _ = _csv_graph(tmp_path)
    runner = Runner(graph, CacheStore())
    runner.refresh("b_read")
    runner.run_all()

    source = compile_graph(graph, runner=runner)
    assert "Fused (streaming)" not in source
    assert "collect_all" not in source


def test_stream_true_fuses_a_chain_into_one_collect_all_and_matches_engine_output(tmp_path):
    csv_path = tmp_path / "data.csv"
    csv_path.write_text("a,b\n" + "".join(f"{i},{'x' if i % 2 == 0 else 'y'}\n" for i in range(1, 21)))
    read = make_block("b_read", "read_csv", params={"path": str(csv_path)})
    filt = make_block("b_filter", "filter", params={"expr": "a > 5"}, x=1)
    select = make_block("b_select", "select", params={"cols": ["a", "b"]}, x=2)
    agg = make_block("b_agg", "groupby_agg", params={"by": ["b"], "aggs": {"a": "sum"}}, x=3)
    graph = Graph(
        blocks={"b_read": read, "b_filter": filt, "b_select": select, "b_agg": agg},
        wires={
            "w1": Wire("w1", "b_read", "out", "b_filter", "df"),
            "w2": Wire("w2", "b_filter", "out", "b_select", "df"),
            "w3": Wire("w3", "b_select", "out", "b_agg", "df"),
        },
    )

    runner = Runner(graph, CacheStore())
    runner.refresh("b_read")
    runner.run_all()

    source = compile_graph(graph, runner=runner, stream=True)
    assert "# --- Fused (streaming): b_read, b_filter, b_select, b_agg ---" in source
    assert source.count("pl.collect_all(") == 1
    # filter/select/groupby_agg's lazy_fn is literally the same function
    # object as their eager fn (see BlockSpec.lazy_fn) -- a single fused
    # chain should still get exactly one def each, not a duplicate.
    assert source.count("def filter(") == 1
    assert source.count("def select(") == 1
    assert source.count("def groupby_agg(") == 1

    ns = {}
    exec(compile(source, "<compiled>", "exec"), ns)

    engine_out = runner.cache.get(runner.state["b_agg"].last_successful_key).outputs["out"]
    compiled_out = ns["b_agg_b_agg"]
    assert compiled_out.sort("b").to_dicts() == engine_out.data.sort("b").to_dicts()


def test_stream_true_fan_out_uses_a_single_collect_all_for_both_exits(tmp_path):
    csv_path = tmp_path / "data.csv"
    csv_path.write_text("a,b\n" + "".join(f"{i},{'x' if i % 2 == 0 else 'y'}\n" for i in range(1, 11)))
    read = make_block("b_read", "read_csv", params={"path": str(csv_path)})
    filt = make_block("b_filter", "filter", params={"expr": "a > 3"}, x=1)
    sel1 = make_block("b_sel1", "select", params={"cols": ["a"]}, x=2, y=0)
    sel2 = make_block("b_sel2", "select", params={"cols": ["b"]}, x=2, y=1)
    graph = Graph(
        blocks={"b_read": read, "b_filter": filt, "b_sel1": sel1, "b_sel2": sel2},
        wires={
            "w1": Wire("w1", "b_read", "out", "b_filter", "df"),
            "w2": Wire("w2", "b_filter", "out", "b_sel1", "df"),
            "w3": Wire("w3", "b_filter", "out", "b_sel2", "df"),
        },
    )
    runner = Runner(graph, CacheStore())
    runner.refresh("b_read")
    runner.run_all()

    source = compile_graph(graph, runner=runner, stream=True)
    # One fused group covering all four blocks, one collect_all producing
    # both exits -- b_filter is never independently collected.
    assert source.count("pl.collect_all(") == 1
    assert "b_sel1_b_sel1, b_sel2_b_sel2 = pl.collect_all(" in source or (
        "b_sel2_b_sel2, b_sel1_b_sel1 = pl.collect_all(" in source
    )

    ns = {}
    exec(compile(source, "<compiled>", "exec"), ns)
    assert sorted(ns["b_sel1_b_sel1"]["a"].to_list()) == [4, 5, 6, 7, 8, 9, 10]
    assert sorted(ns["b_sel2_b_sel2"]["b"].to_list()) == sorted(
        ["x" if i % 2 == 0 else "y" for i in range(4, 11)]
    )


def test_stream_true_output_blocks_forces_an_interior_member_to_be_named(tmp_path):
    graph, _ = _csv_graph(tmp_path)
    runner = Runner(graph, CacheStore())
    runner.refresh("b_read")
    runner.run_all()

    # b_filter's only consumer (b_select) is in the same fusable stretch,
    # so it wouldn't normally need its own exit -- requesting it as an
    # explicit output_blocks target forces one anyway.
    source = compile_graph(graph, runner=runner, stream=True, output_blocks=["b_filter"])
    assert "b_filter_b_filter" in source
    assert "pl.collect_all(" in source

    ns = {}
    exec(compile(source, "<compiled>", "exec"), ns)
    assert ns["b_filter_b_filter"].to_dicts() == [{"a": 2, "b": 20}, {"a": 3, "b": 30}]


def test_stream_true_splits_around_a_non_fusable_custom_block(tmp_path):
    from modelmaker.blocks.base import PortSpec

    graph, _ = _csv_graph(tmp_path)
    custom_code = "def double_a(df: pl.DataFrame) -> pl.DataFrame:\n    return df.with_columns((pl.col('a') * 2).alias('a'))\n"
    graph.blocks["b_custom"] = make_block(
        "b_custom",
        "ai_double",
        params={},
        code=custom_code,
        block_type="standard",
        x=3,
        inputs=[PortSpec("df")],
        outputs=[PortSpec("out")],
    )
    graph.wires["w3"] = Wire("w3", "b_select", "out", "b_custom", "df")

    runner = Runner(graph, CacheStore())
    runner.refresh("b_read")
    runner.run_all()

    source = compile_graph(graph, runner=runner, stream=True)
    # b_custom is never fusable -- it stays a plain eager call, wired to
    # the fused stretch's own exit variable exactly as an ordinary,
    # non-streaming compile would. Custom blocks are compiled under their
    # own block name (see compile_graph's fn-naming rule), not the
    # function's own def name in the source.
    assert "def b_custom(" in source
    assert "pl.collect_all(" in source

    ns = {}
    exec(compile(source, "<compiled>", "exec"), ns)
    engine_out = runner.cache.get(runner.state["b_custom"].last_successful_key).outputs["out"]
    assert ns["b_custom_b_custom"].to_dicts() == engine_out.data.to_dicts()
