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
