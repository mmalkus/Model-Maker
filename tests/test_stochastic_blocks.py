"""Block-level (pl.DataFrame in/out) tests for modelmaker/blocks/stochastic.py."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest
from scipy import stats

from modelmaker.blocks import stochastic as blocks


def test_fit_distribution_block_and_fit_table():
    rng = np.random.default_rng(0)
    df = pl.DataFrame({"x": rng.normal(loc=100.0, scale=5.0, size=3000)})
    dist, table = blocks.fit_distribution(df, "x", families=["norm", "expon"])
    assert dist["family"] == "norm"
    assert set(table["family"].to_list()) == {"norm", "expon"}


def test_fit_distribution_block_spliced_tail():
    rng = np.random.default_rng(1)
    df = pl.DataFrame({"x": rng.lognormal(mean=1.0, sigma=0.5, size=3000)})
    dist, table = blocks.fit_distribution(df, "x", spliced_tail=True, threshold_quantile=0.9)
    assert dist["family"] == "spliced_gpd"
    assert table["family"][0] == "spliced_gpd"


def test_sample_distribution_block_matches_fitted_moments():
    rng = np.random.default_rng(2)
    df = pl.DataFrame({"x": rng.normal(loc=10.0, scale=2.0, size=3000)})
    dist, _ = blocks.fit_distribution(df, "x", families=["norm"])
    samples = blocks.sample_distribution(dist, n=50_000, seed=0, block_id="b_sample")
    assert samples["sample"].mean() == pytest.approx(10.0, abs=0.1)
    assert samples["sample"].std() == pytest.approx(2.0, abs=0.1)


def test_sample_distribution_block_is_deterministic_given_seed_and_block_id():
    dist = {"kind": "distribution", "family": "norm", "params": {"loc": 0.0, "scale": 1.0}}
    a = blocks.sample_distribution(dist, n=100, seed=7, block_id="b_x")
    b = blocks.sample_distribution(dist, n=100, seed=7, block_id="b_x")
    c = blocks.sample_distribution(dist, n=100, seed=7, block_id="b_y")
    assert a["sample"].to_list() == b["sample"].to_list()
    assert a["sample"].to_list() != c["sample"].to_list()


def test_build_dependency_block_recovers_target_correlation():
    rng = np.random.default_rng(3)
    n = 20_000
    z1 = rng.normal(size=n)
    z2 = 0.7 * z1 + np.sqrt(1 - 0.7**2) * rng.normal(size=n)
    df = pl.DataFrame({"a": z1, "b": z2})
    dep = blocks.build_dependency(df, ["a", "b"], copula_type="gaussian")
    assert dep["labels"] == ["a", "b"]
    assert dep["corr"][0][1] == pytest.approx(0.7, abs=0.03)
    assert dep["stds"]["a"] == pytest.approx(1.0, abs=0.05)


def test_simulate_op_risk_lda_chunking_does_not_bias_the_estimate():
    # Different chunk sizes call the RNG in a different sequence of draw
    # sizes, so this can't expect bit-identical output (that would need the
    # same seed *and* the same chunking) -- it checks that chunking is a
    # pure memory-management detail: both chunkings are unbiased Monte
    # Carlo estimates of the same underlying model, so their summary
    # statistics should agree within ordinary MC noise.
    severity = {"kind": "distribution", "family": "lognorm", "params": {"s": 0.5, "loc": 0.0, "scale": 1.0}}
    kwargs = dict(severity_dist=severity, n_paths=200_000, frequency_mean=5.0, seed=1)
    result_a, table_a, paths_a = blocks.simulate_op_risk_lda(chunk_size=200_000, block_id="b_lda_a", **kwargs)
    result_b, _table_b, _paths_b = blocks.simulate_op_risk_lda(chunk_size=7_000, block_id="b_lda_b", **kwargs)
    assert result_a["mean"] == pytest.approx(result_b["mean"], rel=0.03)
    assert result_a["quantiles"][0.999] == pytest.approx(result_b["quantiles"][0.999], rel=0.05)
    assert paths_a.height == 0  # keep_paths defaults to False
    assert table_a.height == 3


def test_simulate_op_risk_lda_keep_paths_and_quantile_table_shape():
    severity = {"kind": "distribution", "family": "norm", "params": {"loc": 100.0, "scale": 10.0}}
    result, table, paths = blocks.simulate_op_risk_lda(
        severity_dist=severity,
        n_paths=2_000,
        chunk_size=500,
        frequency_family="negbinom",
        frequency_mean=3.0,
        frequency_dispersion=2.0,
        alpha_levels=[0.9, 0.99],
        keep_paths=True,
        seed=2,
        block_id="b_lda2",
    )
    assert paths.height == 2_000
    assert table.height == 2
    assert result["n_paths"] == 2_000
    assert "convergence" in result and result["convergence"][-1]["n"] == 2_000


def test_simulate_op_risk_lda_rejects_unknown_frequency_family():
    severity = {"kind": "distribution", "family": "norm", "params": {"loc": 0.0, "scale": 1.0}}
    with pytest.raises(ValueError):
        blocks.simulate_op_risk_lda(severity_dist=severity, n_paths=10, frequency_family="bogus")


def test_risk_measures_block():
    df = pl.DataFrame({"loss": np.arange(1, 1001, dtype=float)})
    result, table = blocks.risk_measures(df, "loss", alpha_levels=[0.95])
    assert result["quantiles"][0.95] == pytest.approx(950.0)
    assert table.height == 1


def test_aggregate_simulation_dependency_changes_the_aggregate_tail():
    rng = np.random.default_rng(4)
    n = 30_000
    components = pl.DataFrame({"a": rng.exponential(size=n), "b": rng.exponential(size=n)})

    independent = {"kind": "dependency", "type": "gaussian", "labels": ["a", "b"], "corr": [[1.0, 0.0], [0.0, 1.0]]}
    correlated = {"kind": "dependency", "type": "gaussian", "labels": ["a", "b"], "corr": [[1.0, 0.95], [0.95, 1.0]]}

    result_indep, _, contrib_indep = blocks.aggregate_simulation(components, independent, alpha_levels=[0.99], seed=0)
    result_corr, _, _ = blocks.aggregate_simulation(components, correlated, alpha_levels=[0.99], seed=0)

    # near-comonotonic aggregation should have a materially heavier tail
    # than independent aggregation of the same two marginals
    assert result_corr["quantiles"][0.99] > result_indep["quantiles"][0.99]
    assert set(contrib_indep["component"].to_list()) == {"a", "b"}


def test_aggregate_simulation_rejects_missing_dependency_labels():
    components = pl.DataFrame({"a": [1.0, 2.0]})
    dep = {"type": "gaussian", "labels": ["a", "b"], "corr": [[1.0, 0.0], [0.0, 1.0]]}
    with pytest.raises(ValueError, match="not found"):
        blocks.aggregate_simulation(components, dep)


def test_fit_evaluate_validate_proxy_round_trip():
    rng = np.random.default_rng(5)
    x = rng.uniform(-2, 2, size=100)
    y = rng.uniform(-2, 2, size=100)
    value = x**2 + 2 * x * y + 3
    fitting = pl.DataFrame({"x": x, "y": y, "value": value})

    proxy = blocks.fit_proxy(fitting, ["x", "y"], value_col="value", method="polynomial", degree=2, regressor="ols")
    assert proxy["method"] == "polynomial"

    scenarios = pl.DataFrame({"x": [1.0, -1.0], "y": [0.5, 0.5]})
    valued = blocks.evaluate_proxy(proxy, scenarios)
    assert "value" in valued.columns
    assert valued.height == 2

    validation_x = rng.uniform(-2, 2, size=30)
    validation_y = rng.uniform(-2, 2, size=30)
    validation = pl.DataFrame(
        {"x": validation_x, "y": validation_y, "truth": validation_x**2 + 2 * validation_x * validation_y + 3}
    )
    diagnostics, error_table = blocks.validate_proxy(proxy, validation, "truth")
    assert diagnostics["kind"] == "scalar_metric"
    assert diagnostics["out_of_sample_r2"] == pytest.approx(1.0, abs=1e-6)
    assert "predicted" in error_table.columns and "abs_error" in error_table.columns


def test_fit_proxy_requires_expr_for_closed_form():
    fitting = pl.DataFrame({"x": [1.0, 2.0]})
    with pytest.raises(ValueError, match="expr"):
        blocks.fit_proxy(fitting, ["x"], method="closed_form")


def test_var_covar_aggregate_block_end_to_end():
    fitting = pl.DataFrame({"x": [-1.0, 0.0, 1.0], "y": [-1.0, 0.0, 1.0], "value": [-2.0, 0.0, 2.0]})
    proxy = blocks.fit_proxy(fitting, ["x", "y"], value_col="value", method="polynomial", degree=1, regressor="ols")

    base = pl.DataFrame({"x": [0.0], "y": [0.0]})
    dependency = {
        "kind": "dependency",
        "type": "gaussian",
        "labels": ["x", "y"],
        "corr": [[1.0, 0.0], [0.0, 1.0]],
        "stds": {"x": 1.0, "y": 1.0},
    }
    result, contributions = blocks.var_covar_aggregate(proxy, base, dependency, alpha_levels=[0.975], method="normal")
    assert result["quantiles"][0.975] == pytest.approx(result["portfolio_std"] * stats.norm.ppf(0.975), abs=1e-3)
    assert set(contributions["factor"].to_list()) == {"x", "y"}
    # symmetric setup -> roughly equal contributions
    contrib_vals = dict(zip(contributions["factor"].to_list(), contributions["euler_contribution"].to_list()))
    assert contrib_vals["x"] == pytest.approx(contrib_vals["y"], rel=0.05)


def test_var_covar_aggregate_block_rejects_multi_row_base():
    fitting = pl.DataFrame({"x": [-1.0, 0.0, 1.0], "value": [-2.0, 0.0, 2.0]})
    proxy = blocks.fit_proxy(fitting, ["x"], value_col="value", method="polynomial", degree=1, regressor="ols")
    base = pl.DataFrame({"x": [0.0, 1.0]})
    dependency = {"labels": ["x"], "corr": [[1.0]], "stds": {"x": 1.0}}
    with pytest.raises(ValueError, match="one row"):
        blocks.var_covar_aggregate(proxy, base, dependency)
