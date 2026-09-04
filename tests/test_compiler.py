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


def test_naming_a_port_uses_that_name_as_the_compiled_variable(tmp_path):
    graph, _ = _csv_graph(tmp_path)
    graph.blocks["b_select"].port_names = {"out": "clean_rows"}

    runner = Runner(graph, CacheStore())
    runner.refresh("b_read")
    runner.run_all()

    source = compile_graph(graph, runner=runner)
    assert "clean_rows = " in source
    assert "b_select_b_select" not in source
