"""Unit tests for the pure-numpy stochastic core (modelmaker/stochastic/*).
Block-level (pl.DataFrame in/out) behavior is covered separately in
tests/test_stochastic_blocks.py."""

from __future__ import annotations

import numpy as np
import pytest
from scipy import stats

from modelmaker.stochastic import accumulate, dependency, distributions, proxy, seed, var_covar


# ---------------------------------------------------------------------------
# seed
# ---------------------------------------------------------------------------


def test_spawn_rng_is_deterministic_for_the_same_seed_and_path():
    a = seed.spawn_rng(42, ["b_1"]).standard_normal(10)
    b = seed.spawn_rng(42, ["b_1"]).standard_normal(10)
    assert a.tolist() == b.tolist()


def test_spawn_rng_differs_across_path_or_seed():
    base = seed.spawn_rng(42, ["b_1"]).standard_normal(10)
    other_path = seed.spawn_rng(42, ["b_2"]).standard_normal(10)
    other_seed = seed.spawn_rng(43, ["b_1"]).standard_normal(10)
    assert base.tolist() != other_path.tolist()
    assert base.tolist() != other_seed.tolist()


# ---------------------------------------------------------------------------
# distributions
# ---------------------------------------------------------------------------


def test_fit_distribution_recovers_normal_params_and_prefers_norm():
    rng = np.random.default_rng(0)
    x = rng.normal(loc=100.0, scale=5.0, size=5000)
    fitted = distributions.fit_distribution(x, ["norm", "expon"])
    assert fitted["family"] == "norm"
    assert fitted["params"]["loc"] == pytest.approx(100.0, abs=0.5)
    assert fitted["params"]["scale"] == pytest.approx(5.0, abs=0.5)
    candidates = {c["family"]: c for c in fitted["candidates"]}
    assert set(candidates) == {"norm", "expon"}
    assert "aic" in candidates["norm"]


def test_fit_distribution_raises_when_no_family_fits():
    with pytest.raises(ValueError):
        distributions.fit_distribution(np.array([1.0, 2.0, 3.0]), ["not_a_family"])


def test_sample_and_ppf_are_consistent_for_a_fitted_normal():
    rng = np.random.default_rng(1)
    x = rng.normal(loc=0.0, scale=1.0, size=2000)
    dist = distributions.fit_distribution(x, ["norm"])
    q = np.array([0.1, 0.5, 0.9])
    values = distributions.ppf(dist, q)
    back = distributions.cdf(dist, values)
    assert back == pytest.approx(q, abs=1e-6)

    samples = distributions.sample(dist, 20000, np.random.default_rng(2))
    assert samples.mean() == pytest.approx(dist["params"]["loc"], abs=0.1)
    assert samples.std() == pytest.approx(dist["params"]["scale"], abs=0.1)


def test_fit_spliced_gpd_threshold_and_sampling():
    rng = np.random.default_rng(3)
    x = rng.lognormal(mean=1.0, sigma=0.5, size=5000)
    dist = distributions.fit_spliced_gpd(x, threshold_quantile=0.9)
    assert dist["family"] == "spliced_gpd"
    assert dist["params"]["threshold"] == pytest.approx(np.quantile(x, 0.9))

    samples = distributions.sample(dist, 50000, np.random.default_rng(4))
    # roughly p_body of samples should fall below the threshold
    frac_below = (samples <= dist["params"]["threshold"]).mean()
    assert frac_below == pytest.approx(0.9, abs=0.02)

    cdf_at_threshold = distributions.cdf(dist, np.array([dist["params"]["threshold"]]))[0]
    assert cdf_at_threshold == pytest.approx(0.9, abs=1e-6)


def test_fit_spliced_gpd_rejects_too_few_exceedances():
    x = np.arange(20, dtype=float)
    with pytest.raises(ValueError, match="exceedances"):
        distributions.fit_spliced_gpd(x, threshold_quantile=0.95)


# ---------------------------------------------------------------------------
# dependency / copulas
# ---------------------------------------------------------------------------


def test_nearest_psd_correlation_repairs_an_indefinite_matrix():
    bad = np.array([[1.0, 0.9, -0.9], [0.9, 1.0, 0.9], [-0.9, 0.9, 1.0]])
    assert np.min(np.linalg.eigvalsh(bad)) < 0
    fixed = dependency.nearest_psd_correlation(bad)
    assert np.min(np.linalg.eigvalsh(fixed)) >= -1e-8
    assert np.allclose(np.diag(fixed), 1.0)


def test_gaussian_copula_reproduces_target_correlation():
    corr = np.array([[1.0, 0.6], [0.6, 1.0]])
    rng = np.random.default_rng(5)
    u = dependency.sample_gaussian_copula(corr, 200_000, rng)
    z = stats.norm.ppf(u)
    empirical = np.corrcoef(z, rowvar=False)[0, 1]
    assert empirical == pytest.approx(0.6, abs=0.02)


def test_t_copula_shape_and_range():
    corr = np.array([[1.0, 0.3], [0.3, 1.0]])
    u = dependency.sample_t_copula(corr, dof=5, n=1000, rng=np.random.default_rng(6))
    assert u.shape == (1000, 2)
    assert u.min() >= 0 and u.max() <= 1


def test_clayton_copula_has_expected_lower_tail_dependence():
    theta = 3.0
    u = dependency.sample_clayton_copula(theta, k=2, n=300_000, rng=np.random.default_rng(7))
    q = 0.02
    lower_joint = np.mean((u[:, 0] < q) & (u[:, 1] < q))
    empirical_lambda_l = lower_joint / q
    theoretical = 2 ** (-1 / theta)
    assert empirical_lambda_l == pytest.approx(theoretical, abs=0.1)


def test_gumbel_copula_has_expected_upper_tail_dependence():
    theta = 3.0
    u = dependency.sample_gumbel_copula(theta, k=2, n=300_000, rng=np.random.default_rng(8))
    q = 0.98
    upper_joint = np.mean((u[:, 0] > q) & (u[:, 1] > q))
    empirical_lambda_u = upper_joint / (1 - q)
    theoretical = 2 - 2 ** (1 / theta)
    assert empirical_lambda_u == pytest.approx(theoretical, abs=0.1)


def test_sample_uniforms_dispatches_on_dependency_type():
    dep = {"type": "gaussian", "corr": [[1.0, 0.0], [0.0, 1.0]]}
    u = dependency.sample_uniforms(dep, 100, np.random.default_rng(9))
    assert u.shape == (100, 2)
    with pytest.raises(ValueError):
        dependency.sample_uniforms({"type": "bogus", "corr": [[1.0]]}, 10, np.random.default_rng(0))


# ---------------------------------------------------------------------------
# accumulate
# ---------------------------------------------------------------------------


def test_risk_measures_on_a_known_array():
    losses = np.arange(1, 101, dtype=float)  # 1..100
    rm = accumulate.risk_measures(losses, [0.95])
    assert rm["quantiles"][0.95] == pytest.approx(95.0)
    assert rm["tvar"][0.95] == pytest.approx(97.5)
    assert rm["mean"] == pytest.approx(50.5)
    assert rm["unexpected_loss"][0.95] == pytest.approx(95.0 - 50.5)


def test_running_estimate_converges_toward_final_value():
    rng = np.random.default_rng(10)
    losses = rng.normal(size=200_000)
    series = accumulate.running_estimate(losses, 0.99, n_checkpoints=6)
    assert series[-1]["n"] == len(losses)
    final = series[-1]["var_estimate"]
    assert final == pytest.approx(stats.norm.ppf(0.99), abs=0.05)


def test_bootstrap_quantile_ci_brackets_a_reasonable_range():
    rng = np.random.default_rng(11)
    losses = rng.normal(size=20_000)
    lo, hi = accumulate.bootstrap_quantile_ci(losses, 0.95, np.random.default_rng(12), n_boot=100)
    true_q = stats.norm.ppf(0.95)
    assert lo <= true_q <= hi
    assert hi - lo < 0.2


def test_euler_contributions_sum_to_total_tvar():
    rng = np.random.default_rng(13)
    n = 50_000
    components = {"a": rng.exponential(size=n), "b": rng.exponential(size=n)}
    alpha = 0.95
    contributions = accumulate.euler_contributions(components, alpha)
    total = components["a"] + components["b"]
    total_tvar = accumulate.risk_measures(total, [alpha])["tvar"][alpha]
    assert sum(contributions.values()) == pytest.approx(total_tvar, rel=1e-9)
    assert contributions["a"] == pytest.approx(contributions["b"], rel=0.1)  # symmetric by construction


def test_shapley_contributions_are_efficient_and_symmetric():
    """The two defining properties of a Shapley value, checked directly
    rather than against a hand-derived reference number: efficiency (shares
    sum exactly to the grand coalition's own value) and symmetry (component
    with an identical marginal effect on every coalition get equal shares --
    guaranteed here by construction, since 'a' and 'b' are iid)."""
    rng = np.random.default_rng(20)
    n = 5_000
    components = {"a": rng.exponential(size=n), "b": rng.exponential(size=n), "c": rng.exponential(size=n)}
    alpha = 0.9
    shapley = accumulate.shapley_contributions(components, alpha, measure="var")
    total = components["a"] + components["b"] + components["c"]
    grand_value = accumulate.risk_measures(total, [alpha])["quantiles"][alpha]
    assert sum(shapley.values()) == pytest.approx(grand_value, rel=1e-9)
    assert shapley["a"] == pytest.approx(shapley["b"], rel=0.15)
    assert shapley["b"] == pytest.approx(shapley["c"], rel=0.15)


def test_shapley_contributions_tvar_measure_is_also_efficient():
    rng = np.random.default_rng(21)
    n = 5_000
    components = {"a": rng.exponential(size=n), "b": 2 * rng.exponential(size=n)}
    alpha = 0.9
    shapley = accumulate.shapley_contributions(components, alpha, measure="tvar")
    total = components["a"] + components["b"]
    grand_value = accumulate.risk_measures(total, [alpha])["tvar"][alpha]
    assert sum(shapley.values()) == pytest.approx(grand_value, rel=1e-9)


def test_shapley_contributions_empty_input():
    assert accumulate.shapley_contributions({}, 0.9) == {}


def test_shapley_contributions_rejects_too_many_components():
    components = {f"c{i}": np.array([1.0, 2.0, 3.0]) for i in range(accumulate.MAX_SHAPLEY_COMPONENTS + 1)}
    with pytest.raises(ValueError, match="at most"):
        accumulate.shapley_contributions(components, 0.9)


# ---------------------------------------------------------------------------
# proxy
# ---------------------------------------------------------------------------


def test_fit_closed_form_and_evaluate():
    import polars as pl

    p = proxy.fit_closed_form("x + y", ["x", "y"])
    scenarios = pl.DataFrame({"x": [1.0, 2.0], "y": [10.0, 20.0]})
    values = proxy.evaluate_proxy(p, scenarios)
    assert values.tolist() == pytest.approx([11.0, 22.0])


def test_fit_polynomial_recovers_an_exact_quadratic():
    import polars as pl

    rng = np.random.default_rng(14)
    x = rng.uniform(-2, 2, size=200)
    y_factor = rng.uniform(-2, 2, size=200)
    value = 2 * x**2 + 3 * x * y_factor + y_factor + 5
    fitting = pl.DataFrame({"x": x, "y": y_factor, "value": value})

    p = proxy.fit_polynomial(fitting, ["x", "y"], "value", degree=2, regressor="ols")
    assert p["fit_diagnostics"]["in_sample_r2"] == pytest.approx(1.0, abs=1e-6)

    scenarios = pl.DataFrame({"x": [1.0, -1.0], "y": [1.0, 2.0]})
    predicted = proxy.evaluate_proxy(p, scenarios)
    expected = 2 * scenarios["x"].to_numpy() ** 2 + 3 * scenarios["x"].to_numpy() * scenarios["y"].to_numpy() + scenarios["y"].to_numpy() + 5
    assert predicted == pytest.approx(expected, abs=1e-6)


def test_fit_polynomial_rejects_too_few_scenarios():
    import polars as pl

    fitting = pl.DataFrame({"x": [1.0], "value": [1.0]})
    with pytest.raises(ValueError):
        proxy.fit_polynomial(fitting, ["x"], "value")


def test_validate_proxy_reports_perfect_fit_for_exact_predictions():
    actual = np.array([1.0, 2.0, 3.0])
    diagnostics = proxy.validate_proxy({"method": "closed_form"}, actual, actual.copy())
    assert diagnostics["out_of_sample_r2"] == pytest.approx(1.0)
    assert diagnostics["mae"] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# var_covar
# ---------------------------------------------------------------------------


def test_bump_and_reval_recovers_derivatives_of_a_quadratic():
    p = proxy.fit_closed_form("x * x", ["x"])
    sens = var_covar.bump_and_reval(p, {"x": 3.0}, ["x"], bump_sizes=0.01)
    assert sens["base_value"] == pytest.approx(9.0)
    assert sens["delta"]["x"] == pytest.approx(6.0, abs=1e-3)
    assert sens["gamma"]["x"] == pytest.approx(2.0, abs=1e-2)


def test_bump_and_reval_cross_gamma_for_a_bilinear_proxy():
    p = proxy.fit_closed_form("x * y", ["x", "y"])
    sens = var_covar.bump_and_reval(p, {"x": 2.0, "y": 3.0}, ["x", "y"], bump_sizes=0.01)
    assert sens["cross_gamma"]["x|y"] == pytest.approx(1.0, abs=1e-2)


def test_var_covar_normal_aggregation_matches_closed_form():
    p = proxy.fit_closed_form("x", ["x"])
    sens = var_covar.bump_and_reval(p, {"x": 0.0}, ["x"], bump_sizes=0.01)
    corr = np.array([[1.0]])
    result = var_covar.aggregate(sens, {"x": 2.0}, corr, ["x"], [0.975], method="normal")
    assert result["portfolio_std"] == pytest.approx(2.0, abs=1e-6)
    assert result["quantiles"][0.975] == pytest.approx(2.0 * stats.norm.ppf(0.975), abs=1e-6)
    assert result["euler_contributions"]["x"] == pytest.approx(2.0, abs=1e-6)


def test_var_covar_non_normal_methods_reduce_to_normal_when_gamma_is_zero():
    p = proxy.fit_closed_form("x + y", ["x", "y"])
    sens = var_covar.bump_and_reval(p, {"x": 0.0, "y": 0.0}, ["x", "y"], bump_sizes=0.01)
    corr = np.array([[1.0, 0.0], [0.0, 1.0]])
    stds = {"x": 1.0, "y": 1.0}
    normal = var_covar.aggregate(sens, stds, corr, ["x", "y"], [0.99], method="normal")
    for method in ("cornish_fisher", "moment_matching", "delta_gamma_copula"):
        result = var_covar.aggregate(sens, stds, corr, ["x", "y"], [0.99], method=method, n_mc=200_000, seed=0)
        assert result["quantiles"][0.99] == pytest.approx(normal["quantiles"][0.99], rel=0.05)


def test_var_covar_delta_gamma_copula_uses_dependency_object_directly():
    p = proxy.fit_closed_form("x + y", ["x", "y"])
    sens = var_covar.bump_and_reval(p, {"x": 0.0, "y": 0.0}, ["x", "y"], bump_sizes=0.01)
    corr = np.array([[1.0, 0.8], [0.8, 1.0]])
    dep = {"type": "gaussian", "corr": corr.tolist()}
    result = var_covar.aggregate(
        sens, {"x": 1.0, "y": 1.0}, corr, ["x", "y"], [0.99], method="delta_gamma_copula", n_mc=100_000, seed=0, copula=dep
    )
    # x+y with corr=0.8, unit std each -> var = 1+1+2*0.8 = 3.6
    assert result["quantiles"][0.99] == pytest.approx(np.sqrt(3.6) * stats.norm.ppf(0.99), rel=0.05)


def test_var_covar_aggregate_raises_for_unknown_method():
    p = proxy.fit_closed_form("x", ["x"])
    sens = var_covar.bump_and_reval(p, {"x": 0.0}, ["x"], bump_sizes=0.01)
    with pytest.raises(ValueError):
        var_covar.aggregate(sens, {"x": 1.0}, np.array([[1.0]]), ["x"], [0.99], method="bogus")
