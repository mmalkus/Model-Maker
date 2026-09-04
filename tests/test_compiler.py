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
    assert "# === Block: b_read" in source
    assert "# --- Call: b_select ---" in source
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
    assert "display_table_b_display(df=" in source
    assert "output_dir=OUTPUT_DIR" in source
    # the display_table call site must not have picked up output_dir too
    display_call_line = next(line for line in source.splitlines() if "display_table_b_display(" in line and "=" in line)
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
    call_line = next(line for line in source.splitlines() if "generate_image_b_img(" in line and "=" in line)
    assert "block_id='b_img'" in call_line
