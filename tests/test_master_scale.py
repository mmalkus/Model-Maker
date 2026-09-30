import random

import pytest

from modelmaker.cache import CacheStore
from modelmaker.compiler import compile_graph
from modelmaker.graph import Graph, Wire
from modelmaker.runner import Runner

from .helpers import make_block


def _scored_graph(tmp_path, n=400, seed=0):
    """A score column whose default rate genuinely rises with the score, so
    every algorithm should come out monotonic on it."""
    rng = random.Random(seed)
    csv_path = tmp_path / "scores.csv"
    lines = ["score,y"]
    for _ in range(n):
        score = rng.random()
        y = 1 if rng.random() < 0.02 + 0.9 * score else 0
        lines.append(f"{score},{y}")
    csv_path.write_text("\n".join(lines))

    read = make_block("b_read", "read_csv", params={"path": str(csv_path)})
    fit = make_block("b_fit", "fit_master_scale", params={"score_col": "score", "target_col": "y", "n_grades": 8})
    assign = make_block("b_assign", "assign_rating_grade", params={})
    summary = make_block("b_summary", "rating_summary", params={"grade_col": "grade", "target_col": "y"})
    graph = Graph(
        blocks={"b_read": read, "b_fit": fit, "b_assign": assign, "b_summary": summary},
        wires={
            "w1": Wire("w1", "b_read", "out", "b_fit", "df"),
            "w2": Wire("w2", "b_read", "out", "b_assign", "df"),
            "w3": Wire("w3", "b_fit", "master_scale", "b_assign", "master_scale"),
            "w4": Wire("w4", "b_assign", "out", "b_summary", "df"),
        },
    )
    return graph


@pytest.mark.parametrize("algorithm", ["quantile", "equal_width", "monotonic_default_rate"])
def test_fit_assign_summarize_round_trip(tmp_path, algorithm):
    graph = _scored_graph(tmp_path)
    graph.blocks["b_fit"].params["algorithm"] = algorithm
    runner = Runner(graph, CacheStore())
    runner.refresh("b_read")

    assert runner.run_block("b_fit") == "green"
    assert runner.run_block("b_assign") == "green"
    assert runner.run_block("b_summary") == "green"

    scale = runner.cache.get(runner.state["b_fit"].last_successful_key).outputs["master_scale"]
    assert scale["algorithm"] == algorithm
    assert 1 <= len(scale["grades"]) <= 8
    assert scale["grades"][0]["lower"] == float("-inf")
    assert scale["grades"][-1]["upper"] == float("inf")

    assigned = runner.cache.get(runner.state["b_assign"].last_successful_key).outputs["out"]
    assert assigned.schema_meta["grade"].role.value == "segment"
    assert set(assigned.data["grade"].unique().to_list()) <= {str(g["grade"]) for g in scale["grades"]}

    summary = runner.cache.get(runner.state["b_summary"].last_successful_key).outputs["metric"]
    assert sum(row["n"] for row in summary["grades"]) == assigned.data.height
    if algorithm == "monotonic_default_rate":
        # The one algorithm that actually guarantees this -- quantile and
        # equal_width bin on the score, not the default rate, so a noisy
        # sample can easily produce a local dip even though risk rises
        # with score on average (that's the whole reason to offer this
        # algorithm as an alternative).
        assert summary["monotonic"] is True


def test_assign_rating_grade_uses_the_scales_own_score_col_by_default(tmp_path):
    # b_assign's params are left empty in _scored_graph -- score_col should
    # fall back to whatever fit_master_scale was fit on, not error out.
    graph = _scored_graph(tmp_path)
    runner = Runner(graph, CacheStore())
    runner.refresh("b_read")
    runner.run_block("b_fit")
    assert runner.run_block("b_assign") == "green"


def test_fit_master_scale_rejects_a_constant_score_column(tmp_path):
    csv_path = tmp_path / "data.csv"
    csv_path.write_text("score,y\n0.5,0\n0.5,1\n0.5,0\n0.5,1\n")
    read = make_block("b_read", "read_csv", params={"path": str(csv_path)})
    fit = make_block("b_fit", "fit_master_scale", params={"score_col": "score", "target_col": "y"})
    graph = Graph(blocks={"b_read": read, "b_fit": fit}, wires={"w1": Wire("w1", "b_read", "out", "b_fit", "df")})
    runner = Runner(graph, CacheStore())
    runner.refresh("b_read")

    assert runner.run_block("b_fit") == "red"
    assert "distinct values" in runner.state["b_fit"].last_error


def test_fit_master_scale_rejects_an_unknown_algorithm(tmp_path):
    graph = _scored_graph(tmp_path)
    graph.blocks["b_fit"].params["algorithm"] = "made_up"
    runner = Runner(graph, CacheStore())
    runner.refresh("b_read")

    assert runner.run_block("b_fit") == "red"
    assert "unknown algorithm" in runner.state["b_fit"].last_error


def test_monotonic_default_rate_can_return_fewer_grades_than_requested(tmp_path):
    # Every row shares the same default rate (~0.5) regardless of score --
    # pool-adjacent-violators should collapse everything into far fewer
    # than the 8 requested grades rather than force a monotonic split that
    # isn't really there.
    rng = random.Random(1)
    csv_path = tmp_path / "flat.csv"
    lines = ["score,y"]
    for _ in range(300):
        lines.append(f"{rng.random()},{1 if rng.random() < 0.5 else 0}")
    csv_path.write_text("\n".join(lines))

    read = make_block("b_read", "read_csv", params={"path": str(csv_path)})
    fit = make_block(
        "b_fit",
        "fit_master_scale",
        params={"score_col": "score", "target_col": "y", "n_grades": 8, "algorithm": "monotonic_default_rate"},
    )
    graph = Graph(blocks={"b_read": read, "b_fit": fit}, wires={"w1": Wire("w1", "b_read", "out", "b_fit", "df")})
    runner = Runner(graph, CacheStore())
    runner.refresh("b_read")

    assert runner.run_block("b_fit") == "green"
    scale = runner.cache.get(runner.state["b_fit"].last_successful_key).outputs["master_scale"]
    assert len(scale["grades"]) < 8


def test_compiled_master_scale_pipeline_matches_engine_output(tmp_path):
    # fit_master_scale's PAVA logic lives in a nested function -- the
    # compiler only inlines a block's own source (see compiler.py's
    # inspect.getsource(fn) call), so this is the regression test that the
    # nested def actually survives being pasted verbatim into a script.
    graph = _scored_graph(tmp_path)
    graph.blocks["b_fit"].params["algorithm"] = "monotonic_default_rate"
    runner = Runner(graph, CacheStore())
    runner.refresh("b_read")
    runner.run_all()

    source = compile_graph(graph, runner=runner)
    ns = {}
    exec(compile(source, "<compiled>", "exec"), ns)

    engine_out = runner.cache.get(runner.state["b_assign"].last_successful_key).outputs["out"]
    compiled_out = ns["b_assign_b_assign"]
    assert compiled_out.to_dicts() == engine_out.data.to_dicts()


def test_monotonic_scale_spreads_a_zero_default_tail_over_several_grades():
    """Leftover pools are merged by closest mean *predicted* score, not
    observed rate -- every zero-default pool has observed rate 0, so
    merging on that folded the whole low-risk tail into one grade."""
    import numpy as np
    import polars as pl

    from modelmaker.blocks.modelling import fit_master_scale

    rng = np.random.default_rng(0)
    pd_ = np.exp(rng.uniform(np.log(0.0005), np.log(0.5), size=600))
    y = (rng.random(600) < pd_).astype(int)
    scale = fit_master_scale(pl.DataFrame({"pd": pd_, "y": y}), "pd", "y", n_grades=7, algorithm="monotonic_default_rate")
    grades = scale["grades"]
    zero_default = [g for g in grades if g["observed_rate"] == 0.0]
    assert len(zero_default) >= 2
    assert max(g["n"] for g in grades) / 600 < 0.5
    rates = [g["observed_rate"] for g in grades]
    assert all(a <= b for a, b in zip(rates, rates[1:]))


def test_monotonic_scale_stays_monotonic_with_tied_scores_on_the_edges():
    """A binned/WoE model has few distinct scores, so quantile edges land on
    clumps of ties; pooling and grade assignment must use the same
    (lo, hi] convention or the clumps land in a different grade than the
    one monotonicity was checked for. This is the PD sample data case that
    broke it."""
    from pathlib import Path

    import polars as pl

    from modelmaker.blocks.binning import apply_binning, fit_binning
    from modelmaker.blocks.library import time_split, train_test_split
    from modelmaker.blocks.modelling import assign_rating_grade, calibrate_model, fit_master_scale, logistic_regression, predict

    df = pl.read_csv(Path(__file__).resolve().parents[1] / "sample_data" / "credit_risk_data.csv")
    dev, _ = time_split(df, "reference_date", "2025-01-01")
    train, _ = train_test_split(dev, 0.3, 1, stratify_col="default_flag")
    features = ["credit_score", "purpose", "employment_years", "home_ownership"]
    binning, _, _ = fit_binning(train, "default_flag", features)
    woe = apply_binning(train, binning)
    _, model = logistic_regression(woe, "default_flag", [f"{f}_woe" for f in features])
    scored = predict(woe, calibrate_model(woe, model, 0.08))

    for share in (0.0, 0.05):
        scale = fit_master_scale(scored, "predicted_proba", "default_flag", 7, "monotonic_default_rate", share)
        graded = assign_rating_grade(scored, scale)
        observed = graded.group_by("grade").agg(pl.col("default_flag").mean(), pl.len()).sort(pl.col("grade").cast(pl.Int64))
        # The artifact's own per-grade stats match what assignment produces...
        assert observed["len"].to_list() == [g["n"] for g in scale["grades"]]
        # ...and observed default rate rises monotonically across grades.
        rates = observed["default_flag"].to_list()
        assert all(a <= b for a, b in zip(rates, rates[1:])), rates
