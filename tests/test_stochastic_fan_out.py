"""Tests for the graph fan-out engine primitive (stochastic-engine-proposal.md
S4): the 'iterate'/'collect' block pair and Runner._run_region_iterations/
_dispatch_iterations."""

from __future__ import annotations

import pytest

from modelmaker.blocks.base import PortSpec
from modelmaker.cache import CacheStore
from modelmaker.compiler import CompileError, compile_graph
from modelmaker.graph import Graph, Wire
from modelmaker.runner import Runner

from .helpers import make_block

_MEAN_CODE = "def compute_mean(df):\n    import polars as pl\n    return df.select(pl.col('x').mean().alias('mean'))\n"


def _bootstrap_ci_graph(tmp_path, n_iterations: int = 50):
    csv_path = tmp_path / "data.csv"
    csv_path.write_text("x\n" + "\n".join(str(v) for v in range(1, 101)) + "\n")
    read = make_block("b_read", "read_csv", params={"path": str(csv_path)})
    iterate_block = make_block(
        "b_iter", "iterate", params={"n_iterations": n_iterations, "iterator": "bootstrap_resample", "seed": 7}
    )
    mean_block = make_block("b_mean", "compute_mean", code=_MEAN_CODE, inputs=[PortSpec("df")], outputs=[PortSpec("out")])
    collect_block = make_block(
        "b_collect",
        "collect",
        params={"iterate_block": "b_iter", "reducer": "risk_measures", "value_col": "mean", "alpha_levels": [0.05, 0.5, 0.95], "seed": 3},
    )
    return Graph(
        blocks={"b_read": read, "b_iter": iterate_block, "b_mean": mean_block, "b_collect": collect_block},
        wires={
            "w1": Wire("w1", "b_read", "out", "b_iter", "df"),
            "w2": Wire("w2", "b_iter", "out", "b_mean", "df"),
            "w3": Wire("w3", "b_mean", "out", "b_collect", "value"),
        },
    )


def _scenario_row_graph(tmp_path, n_iterations: int = 3):
    csv_path = tmp_path / "scenarios.csv"
    csv_path.write_text("factor\n10\n20\n30\n")
    read = make_block("b_read", "read_csv", params={"path": str(csv_path)})
    iterate_block = make_block("b_iter", "iterate", params={"n_iterations": n_iterations, "iterator": "scenario_row", "seed": 1})
    select_block = make_block("b_select", "select", params={"cols": ["factor"]})
    collect_block = make_block("b_collect", "collect", params={"iterate_block": "b_iter", "reducer": "concat"})
    return Graph(
        blocks={"b_read": read, "b_iter": iterate_block, "b_select": select_block, "b_collect": collect_block},
        wires={
            "w1": Wire("w1", "b_read", "out", "b_iter", "df"),
            "w2": Wire("w2", "b_iter", "out", "b_select", "df"),
            "w3": Wire("w3", "b_select", "out", "b_collect", "value"),
        },
    )


def test_bootstrap_ci_fan_out_end_to_end(tmp_path):
    graph = _bootstrap_ci_graph(tmp_path)
    runner = Runner(graph, CacheStore())
    runner.run_block("b_read")
    assert runner.run_to_here("b_collect") == "green"

    outputs = runner.cache.get(runner.compute_key("b_collect")).outputs
    table = outputs["table"].data
    assert table.height == 50
    assert set(table.columns) >= {"mean", "iteration"}
    assert sorted(table["iteration"].to_list()) == list(range(50))

    result = outputs["simulation_result"]
    assert result["kind"] == "simulation_result"
    assert result["n_paths"] == 50
    # true population mean of 1..100 is 50.5 -- the bootstrap distribution
    # of the mean should be centered close to it
    assert result["mean"] == pytest.approx(50.5, abs=3.0)
    assert set(result["quantiles"]) == {0.05, 0.5, 0.95}
    assert result["quantiles"][0.05] < result["quantiles"][0.5] < result["quantiles"][0.95]


def test_bootstrap_fan_out_is_deterministic_across_runner_instances(tmp_path):
    graph = _bootstrap_ci_graph(tmp_path, n_iterations=20)

    runner_a = Runner(graph, CacheStore())
    runner_a.run_block("b_read")
    runner_a.run_to_here("b_collect")
    table_a = runner_a.cache.get(runner_a.compute_key("b_collect")).outputs["table"].data

    runner_b = Runner(graph, CacheStore())
    runner_b.run_block("b_read")
    runner_b.run_to_here("b_collect")
    table_b = runner_b.cache.get(runner_b.compute_key("b_collect")).outputs["table"].data

    assert table_a.sort("iteration").to_dicts() == table_b.sort("iteration").to_dicts()


def test_region_re_executes_per_iteration_not_just_once(tmp_path):
    """A weak version of this test would pass even if the fan-out silently
    ran the region only once and replicated it -- guard against that by
    checking the per-iteration means actually vary (bootstrap resampling
    really did draw a different sample each time)."""
    graph = _bootstrap_ci_graph(tmp_path, n_iterations=20)
    runner = Runner(graph, CacheStore())
    runner.run_block("b_read")
    runner.run_to_here("b_collect")
    table = runner.cache.get(runner.compute_key("b_collect")).outputs["table"].data
    assert table["mean"].n_unique() > 1


def test_scenario_row_fan_out_produces_one_row_per_iteration(tmp_path):
    graph = _scenario_row_graph(tmp_path)
    runner = Runner(graph, CacheStore())
    runner.run_block("b_read")
    assert runner.run_to_here("b_collect") == "green"

    table = runner.cache.get(runner.compute_key("b_collect")).outputs["table"].data
    assert table.sort("iteration")["factor"].to_list() == [10, 20, 30]
    assert table["iteration"].to_list() == sorted(table["iteration"].to_list())


def test_scenario_row_out_of_range_iteration_is_a_clean_error(tmp_path):
    graph = _scenario_row_graph(tmp_path, n_iterations=5)  # only 3 scenario rows exist
    runner = Runner(graph, CacheStore())
    runner.run_block("b_read")
    assert runner.run_to_here("b_collect") == "red"
    assert "out of range" in runner.state["b_collect"].last_error


def test_collect_requires_iterate_block_param(tmp_path):
    graph = _scenario_row_graph(tmp_path)
    graph.blocks["b_collect"].params.pop("iterate_block")
    runner = Runner(graph, CacheStore())
    runner.run_block("b_read")
    assert runner.run_to_here("b_collect") == "red"
    assert "iterate_block" in runner.state["b_collect"].last_error


def test_collect_iterate_block_must_be_an_iterate_category(tmp_path):
    graph = _scenario_row_graph(tmp_path)
    graph.blocks["b_collect"].params["iterate_block"] = "b_select"  # exists, but isn't an 'iterate' block
    runner = Runner(graph, CacheStore())
    runner.run_block("b_read")
    assert runner.run_to_here("b_collect") == "red"
    assert "not an 'iterate' block" in runner.state["b_collect"].last_error


def test_iterate_block_must_actually_reach_collect(tmp_path):
    graph = _scenario_row_graph(tmp_path)
    # rewire collect's 'value' straight from b_read -- b_iter no longer
    # reaches b_collect at all (collect has only one input wire, so this is
    # the only way anything reaches it now)
    graph.wires["w3"] = Wire("w3", "b_read", "out", "b_collect", "value")
    runner = Runner(graph, CacheStore())
    runner.run_block("b_read")
    assert runner.run_to_here("b_collect") == "red"
    assert "does not reach" in runner.state["b_collect"].last_error


def test_region_may_not_contain_an_input_block(tmp_path):
    graph = _scenario_row_graph(tmp_path)
    graph.blocks["b_select"].block_type = "input"  # malformed on purpose, like the cyclic-graph test elsewhere
    runner = Runner(graph, CacheStore())
    runner.run_block("b_read")
    assert runner.run_to_here("b_collect") == "red"
    assert "input block" in runner.state["b_collect"].last_error


def test_region_may_not_contain_a_grouped_block(tmp_path):
    graph = _scenario_row_graph(tmp_path)
    graph.blocks["b_select"].group_by = "factor"
    runner = Runner(graph, CacheStore())
    runner.run_block("b_read")
    assert runner.run_to_here("b_collect") == "red"
    assert "group_by" in runner.state["b_collect"].last_error


def test_n_iterations_must_be_positive(tmp_path):
    graph = _scenario_row_graph(tmp_path, n_iterations=0)
    runner = Runner(graph, CacheStore())
    runner.run_block("b_read")
    assert runner.run_to_here("b_collect") == "red"
    assert "n_iterations" in runner.state["b_collect"].last_error


def test_n_iterations_over_the_safety_limit_fails_fast(tmp_path):
    graph = _scenario_row_graph(tmp_path, n_iterations=Runner.MAX_ITERATIONS + 1)
    runner = Runner(graph, CacheStore())
    runner.run_block("b_read")
    assert runner.run_to_here("b_collect") == "red"
    assert "safety limit" in runner.state["b_collect"].last_error


def test_collect_concat_reducer_passes_table_through_unreduced(tmp_path):
    graph = _scenario_row_graph(tmp_path)
    runner = Runner(graph, CacheStore())
    runner.run_block("b_read")
    runner.run_to_here("b_collect")
    outputs = runner.cache.get(runner.compute_key("b_collect")).outputs
    assert outputs["simulation_result"] is None


def test_compile_graph_rejects_a_fan_out_region(tmp_path):
    graph = _scenario_row_graph(tmp_path)
    runner = Runner(graph, CacheStore())
    runner.run_block("b_read")
    runner.run_to_here("b_collect")
    with pytest.raises(CompileError, match="fan-out"):
        compile_graph(graph, runner=runner)
