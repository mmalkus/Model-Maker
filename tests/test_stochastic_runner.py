"""End-to-end Runner tests for the stochastic blocks -- specifically that
`block_id` injection (see runner.run_block's widened `accepts_param(fn,
"block_id")` check) actually reaches a "standard"-type block through the
real subprocess dispatch path, not just via a direct function call."""

from __future__ import annotations

import polars as pl

from modelmaker.cache import CacheStore
from modelmaker.compiler import compile_graph
from modelmaker.graph import Graph, Wire
from modelmaker.runner import Runner

from .helpers import make_block


def _dependency_graph(tmp_path):
    csv_path = tmp_path / "factors.csv"
    csv_path.write_text("a,b\n1.0,2.1\n1.2,1.9\n0.9,2.4\n1.1,2.0\n1.3,2.2\n0.8,1.8\n")
    read = make_block("b_read", "read_csv", params={"path": str(csv_path)})
    dep = make_block("b_dep", "build_dependency", params={"columns": ["a", "b"], "copula_type": "gaussian"})
    graph = Graph(blocks={"b_read": read, "b_dep": dep}, wires={"w1": Wire("w1", "b_read", "out", "b_dep", "df")})
    return graph


def test_build_dependency_runs_green_through_the_runner(tmp_path):
    graph = _dependency_graph(tmp_path)
    runner = Runner(graph, CacheStore())
    runner.run_block("b_read")
    assert runner.status("b_dep") == "grey"
    runner.run_to_here("b_dep")
    assert runner.status("b_dep") == "green"
    key = runner.compute_key("b_dep")
    dependency = runner.cache.get(key).outputs["dependency"]
    assert dependency["kind"] == "dependency"
    assert dependency["labels"] == ["a", "b"]


def test_sample_distribution_block_id_injection_is_deterministic_across_runs(tmp_path):
    """sample_distribution has no dataframe input at all -- it's driven
    purely by a literal `distribution` param -- so this exercises block_id
    injection for a block with zero wired inputs, through the full
    subprocess dispatch (run_worker.run_worker_entry), not a direct call."""
    dist_param = {"kind": "distribution", "family": "norm", "params": {"loc": 0.0, "scale": 1.0}}
    block = make_block("b_sample", "sample_distribution", params={"distribution": dist_param, "n": 500, "seed": 3})
    graph = Graph(blocks={"b_sample": block}, wires={})

    runner_a = Runner(graph, CacheStore())
    runner_a.run_block("b_sample")
    key_a = runner_a.compute_key("b_sample")
    samples_a = runner_a.cache.get(key_a).outputs["samples"].data

    runner_b = Runner(graph, CacheStore())
    runner_b.run_block("b_sample")
    key_b = runner_b.compute_key("b_sample")
    samples_b = runner_b.cache.get(key_b).outputs["samples"].data

    assert isinstance(samples_a, pl.DataFrame)
    assert samples_a["sample"].to_list() == samples_b["sample"].to_list()


def test_sample_distribution_differs_when_block_id_differs(tmp_path):
    dist_param = {"kind": "distribution", "family": "norm", "params": {"loc": 0.0, "scale": 1.0}}
    block_x = make_block("b_x", "sample_distribution", params={"distribution": dist_param, "n": 200, "seed": 9})
    block_y = make_block("b_y", "sample_distribution", params={"distribution": dist_param, "n": 200, "seed": 9})

    runner = Runner(Graph(blocks={"b_x": block_x, "b_y": block_y}, wires={}), CacheStore())
    runner.run_block("b_x")
    runner.run_block("b_y")
    samples_x = runner.cache.get(runner.compute_key("b_x")).outputs["samples"].data
    samples_y = runner.cache.get(runner.compute_key("b_y")).outputs["samples"].data
    assert samples_x["sample"].to_list() != samples_y["sample"].to_list()


def test_compiled_script_passes_block_id_for_a_standard_stochastic_block(tmp_path):
    """compiler.compile_graph has its own, separate block_id-injection
    check (wants_block_id in compiler.py) that has to agree with the
    runner's -- otherwise a compiled script would call sample_distribution
    without block_id and silently draw a different (still-deterministic,
    but non-matching) stream than the live graph did."""
    dist_param = {"kind": "distribution", "family": "norm", "params": {"loc": 0.0, "scale": 1.0}}
    block = make_block("b_sample", "sample_distribution", params={"distribution": dist_param, "n": 50, "seed": 1})
    graph = Graph(blocks={"b_sample": block}, wires={})
    runner = Runner(graph, CacheStore())
    runner.run_all()

    script = compile_graph(graph, runner=runner)
    assert "block_id='b_sample'" in script

    namespace: dict = {}
    exec(compile(script, "<compiled>", "exec"), namespace)  # noqa: S102 -- test-only, compiling our own output
    compiled_samples = namespace["b_sample_b_sample"]  # {block.name}_{block_id}, see compiler._alloc_output_vars
    live_samples = runner.cache.get(runner.compute_key("b_sample")).outputs["samples"].data
    assert compiled_samples["sample"].to_list() == live_samples["sample"].to_list()
