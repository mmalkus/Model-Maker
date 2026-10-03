"""Decision hints (BuildOptions.decision_hints, see modelmaker/agent/hints.py):
the rules on their own, and their place in run_to, plan_stage and the
prompts -- only when the option is on."""

from __future__ import annotations

import polars as pl

from modelmaker.agent import prompts
from modelmaker.agent.hints import decision_hints
from modelmaker.packet import ColumnMeta, ColumnRole, DataFramePacket

from .test_agent_build import ANCHOR, FEATURES, _binned, est_steps, prepared, staged  # noqa: F401 -- fixture


def _table(rows: list[dict]) -> DataFramePacket:
    return DataFramePacket(pl.DataFrame(rows), {})


def _sample(target: list, column: str = "default_flag") -> DataFramePacket:
    df = pl.DataFrame({column: target})
    return DataFramePacket(df, {column: ColumnMeta(dtype=str(df.schema[column]), role=ColumnRole.TARGET)})


def _hint(category: str, **outputs) -> str:
    """Every hint for the outputs, as one string to search."""
    return " | ".join(decision_hints(category, outputs))


# ---- the rules -------------------------------------------------------------------


def test_binning_sorts_features_by_iv_band():
    summary = _table(
        [
            {"feature": "leaky", "iv": 0.9, "monotonic": True},
            {"feature": "good", "iv": 0.25, "monotonic": True},
            {"feature": "bumpy", "iv": 0.12, "monotonic": False},
            {"feature": "noise", "iv": 0.005, "monotonic": True},
        ]
    )
    hints = decision_hints("fit_binning", {"summary": summary})
    assert any("Candidate" in h and "good" in h and "bumpy" in h and "leaky" not in h for h in hints)
    assert any("Leave out" in h and "noise" in h for h in hints)
    assert any("leaky" in h and "ask_user" in h for h in hints)
    assert any("Not monotonic" in h and "bumpy" in h and "good" not in h for h in hints)


def test_binning_with_a_continuous_target_uses_r2():
    summary = _table([{"feature": "a", "r2_binned": 0.08}, {"feature": "b", "r2_binned": 0.001}])
    hints = decision_hints("fit_binning", {"summary": summary})
    assert any("Candidate" in h and "a" in h for h in hints)
    assert any("Leave out" in h and "b" in h for h in hints)


def test_coefficients_flag_wrong_signs_and_insignificance():
    model = {
        "coefficients": {"x_woe": 0.4, "y_woe": -0.8, "z_woe": -0.1},
        "statistics": {
            "intercept": {"p_value": 0.9},
            "x_woe": {"p_value": 0.001},
            "y_woe": {"p_value": 0.001},
            "z_woe": {"p_value": 0.4},
        },
    }
    hints = decision_hints("logistic_regression", {"model": model})
    assert any("wrong sign" in h and "x_woe" in h and "y_woe" not in h for h in hints)
    assert any("Not significant" in h and "z_woe" in h and "intercept" not in h for h in hints)

    sound = {"coefficients": {"y_woe": -0.8}, "statistics": {"y_woe": {"p_value": 0.001}}}
    assert decision_hints("logistic_regression", {"model": sound}) == [
        "Coefficients look sound: signs as expected and all significant."
    ]


def test_gini_bands():
    def hint(g):
        return decision_hints("auc_gini", {"metric": {"auc": (g + 1) / 2, "gini": g}})[0]

    assert "wrong way" in hint(-0.1)
    assert "weak" in hint(0.1)
    assert "usual range" in hint(0.5)
    assert "leakage" in hint(0.95)


def test_compare_samples_flags_overfitting_and_calibration():
    table = _table(
        [
            {"sample": "train", "gini": 0.62, "mean_actual": 0.05, "mean_predicted": 0.05},
            {"sample": "test", "gini": 0.41, "mean_actual": 0.05, "mean_predicted": 0.08},
        ]
    )
    hints = decision_hints("compare_samples", {"table": table})
    assert any("overfitting" in h and "train" in h and "test" in h for h in hints)
    assert any("calibrate" in h and "test" in h and "above" in h for h in hints)

    steady = _table(
        [
            {"sample": "train", "gini": 0.6, "mean_actual": 0.05, "mean_predicted": 0.05},
            {"sample": "test", "gini": 0.58, "mean_actual": 0.05, "mean_predicted": 0.051},
        ]
    )
    assert "holds" in decision_hints("compare_samples", {"table": steady})[0]


def test_psi_and_rating_summary():
    assert "stable" in decision_hints("psi_test", {"metric": {"psi": 0.05}})[0]
    assert "moderate" in decision_hints("psi_test", {"metric": {"psi": 0.15}})[0]
    assert "ask_user" in decision_hints("psi_test", {"metric": {"psi": 0.4}})[0]
    assert "refit" in decision_hints("rating_summary", {"metric": {"monotonic": False}})[0]


def test_blocks_without_rules_and_odd_outputs_get_no_hints():
    assert decision_hints("train_test_split", {"train": _table([{"a": 1}])}) == []
    assert decision_hints("auc_gini", {"metric": "not a dict"}) == []
    assert decision_hints("fit_binning", {}) == []


# ---- data ----------------------------------------------------------------------


def test_data_profile_flags_empty_constant_and_id_columns():
    cols = [
        {"column": "sparse", "dtype": "Float64", "fill_rate": 0.2, "dominant_value_share": 0.1, "distinct_ratio": 0.1},
        {"column": "flat", "dtype": "Int64", "fill_rate": 1.0, "dominant_value_share": 0.99, "distinct_ratio": 0.0},
        {"column": "app_id", "dtype": "String", "fill_rate": 1.0, "dominant_value_share": 0.0, "distinct_ratio": 1.0},
        {"column": "income", "dtype": "Float64", "fill_rate": 1.0, "dominant_value_share": 0.01, "distinct_ratio": 0.99},
    ]
    text = _hint("data_profile", metric={"columns": cols})
    assert "Mostly empty" in text and "sparse" in text
    assert "Near-constant" in text and "flat" in text
    assert "Id-like" in text and "app_id" in text
    assert "income" not in text  # a continuous float is distinct by nature
    assert "No empty" in _hint("data_profile", metric={"columns": cols[3:]})


def test_exclusions_flag_big_and_empty_rules():
    summary = {
        "starting_population": 1000,
        "final_population": 600,
        "steps": [
            {"step": "starting population", "rule": None, "dropped": 0},
            {"step": "adults", "rule": "age >= 18", "dropped": 400},
            {"step": "has income", "rule": "income > 0", "dropped": 0},
        ],
    }
    text = _hint("apply_exclusions", summary=summary)
    assert "'adults' dropped 40%" in text and "'has income' dropped nothing" in text
    summary["steps"] = summary["steps"][:1] + [{"step": "adults", "rule": "age >= 18", "dropped": 50}]
    assert "kept 60%" in _hint("apply_exclusions", summary=summary)


def test_quality_rules_split_errors_from_warnings():
    results = [
        {"name": "ids", "severity": "error", "passed": False},
        {"name": "range", "severity": "warning", "passed": False},
        {"name": "nulls", "severity": "error", "passed": True},
    ]
    text = _hint("data_quality_rules", metric={"results": results})
    assert "Failed: ids" in text and "Warnings: range" in text and "nulls" not in text
    assert "passed" in _hint("data_quality_rules", metric={"results": results[2:]})


def test_splits_flag_few_events_rate_gaps_and_empty_samples():
    train = _sample([1] * 100 + [0] * 900)
    test = _sample([1] * 10 + [0] * 290)
    text = _hint("train_test_split", train=train, test=test)
    assert "Few events" in text and "test has 10" in text and "train has" not in text
    assert "stratify_col" in text
    assert _hint("train_test_split", train=train, test=_sample([1] * 60 + [0] * 540)) == ""
    empty = _sample([], "default_flag")
    assert "out_of_time is empty" in _hint("time_split", development=train, out_of_time=empty)


def test_computed_lgd_and_ccf_flag_nulls():
    out = DataFramePacket(pl.DataFrame({"lgd": [0.1, None, 0.5, 0.9]}), {})
    assert "25% of rows have a null lgd" in _hint("compute_lgd", out=out)
    assert _hint("compute_ccf", out=DataFramePacket(pl.DataFrame({"ccf": [0.1, 0.2]}), {})) == ""


# ---- feature screens ---------------------------------------------------------------


def test_iv_table_uses_the_binning_bands():
    rows = [{"feature": "a", "iv": 0.9}, {"feature": "b", "iv": 0.1}, {"feature": "c", "iv": 0.001}]
    text = _hint("iv_table", metric={"rows": rows})
    assert "Candidate features" in text and "b" in text
    assert "Leave out" in text and "c" in text and "leakage: a" in text


def test_correlation_flags_pairs_and_vifs():
    metric = {
        "features": ["a", "b", "c"],
        "correlation": [[1.0, 0.92, 0.1], [0.92, 1.0, 0.0], [0.1, 0.0, 1.0]],
        "vif": [{"feature": "a", "vif": 12.0}, {"feature": "b", "vif": 11.0}, {"feature": "c", "vif": 1.1}],
    }
    text = _hint("correlation_matrix", metric=metric)
    assert "a/b (+0.92)" in text and "VIF >= 10: a, b" in text and "c" not in text.split("VIF")[1]
    calm = {"features": ["a", "c"], "correlation": [[1.0, 0.1], [0.1, 1.0]], "vif": [{"feature": "a", "vif": None}]}
    assert "No redundant" in _hint("correlation_matrix", metric=calm)


def test_characteristic_stability_bands():
    table = _table([{"feature": "a", "psi": 0.4}, {"feature": "b", "psi": 0.15}, {"feature": "c", "psi": 0.01}])
    text = _hint("characteristic_stability", table=table)
    assert "Unstable" in text and "a" in text and "Shifting" in text and "b" in text
    assert "stable" in _hint("characteristic_stability", table=_table([{"feature": "c", "psi": 0.01}]))


def test_target_trend_flags_an_immature_last_period_and_thin_ones():
    rows = [{"period": f"2020-Q{q}", "n": 200, "target_mean": 0.05} for q in range(1, 5)]
    text = _hint("target_trend", table=_table(rows + [{"period": "2021-Q1", "n": 20, "target_mean": 0.01}]))
    assert "2021-Q1" in text and "immature" in text and "< 30 rows: 2021-Q1" in text
    assert _hint("target_trend", table=_table(rows)) == ""


# ---- fits -------------------------------------------------------------------------


def test_lgd_regression_flags_non_convergence_but_not_woe_signs():
    model = {
        "converged": False,
        "coefficients": {"x_woe": 0.4},
        "statistics": {"x_woe": {"p_value": 0.001}},
    }
    text = _hint("lgd_regression", model=model)
    assert "didn't converge" in text and "wrong sign" not in text


def test_stepwise_names_the_selection_and_wrong_signs():
    metric = {
        "target_type": "binary",
        "selected": ["a_woe", "b_woe"],
        "dropped": ["c_woe"],
        "coefficients": [{"feature": "a_woe", "sign": "-"}, {"feature": "b_woe", "sign": "+"}],
    }
    text = _hint("stepwise_selection", metric=metric)
    assert "Use these as the model's features: a_woe, b_woe" in text
    assert "Dropped along the way: c_woe" in text and "Wrong (positive) sign on b_woe" in text
    assert "Nothing passed" in _hint("stepwise_selection", metric={"selected": []})


def test_calibration_shift_size():
    def cal(shift):
        return {"calibration": {"intercept_shift": shift, "mean_prediction_before": 0.1, "mean_prediction_after": 0.05}}

    assert "Calibrated" in _hint("calibrate_model", model=cal(-0.3))
    assert "large shift" in _hint("calibrate_model", model=cal(-1.5))


def test_master_scale_monotonicity_and_thin_grades():
    grades = [
        {"grade": 1, "n": 500, "observed_rate": 0.01},
        {"grade": 2, "n": 480, "observed_rate": 0.03},
        {"grade": 3, "n": 20, "observed_rate": 0.02},
    ]
    text = _hint("fit_master_scale", master_scale={"grades": grades})
    assert "monotonic_default_rate" in text and "Thin grades (3" in text
    grades[2] = {"grade": 3, "n": 400, "observed_rate": 0.2}
    assert "monotonic and none too thin" in _hint("fit_master_scale", master_scale={"grades": grades})


def test_long_run_average_history_and_weighting():
    text = _hint("long_run_average", metric={"n_periods": 3, "default_weighted": 0.4, "time_weighted": 0.5})
    assert "Only 3 period" in text and "differ by more than 10%" in text
    assert "agree" in _hint("long_run_average", metric={"n_periods": 8, "default_weighted": 0.4, "time_weighted": 0.41})


# ---- discrimination, calibration ----------------------------------------------------------


def test_ks_and_roc_bands():
    assert "weak" in _hint("ks_test", metric={"ks_statistic": 0.1})
    assert "usual range" in _hint("ks_test", metric={"ks_statistic": 0.4})
    assert "leakage" in _hint("ks_test", metric={"ks_statistic": 0.8})
    assert "Gini 0.50" in _hint("roc_curve", metric={"auc": 0.75})


def test_continuous_accuracy_flags_ranking_fit_and_bias():
    bad = {"r2": -0.1, "spearman": 0.05, "mean_actual": 0.4, "bias": 0.12}
    text = _hint("continuous_accuracy", metric=bad)
    assert "R^2 is negative" in text and "Spearman 0.05 is weak" in text and "30% above" in text
    assert "reasonable" in _hint("continuous_accuracy", metric={"r2": 0.3, "spearman": 0.5, "mean_actual": 0.4, "bias": 0.0})


def test_compare_samples_with_a_continuous_target():
    table = _table(
        [
            {"sample": "train", "spearman": 0.6, "mean_actual": 0.4, "mean_predicted": 0.4},
            {"sample": "test", "spearman": 0.45, "mean_actual": 0.4, "mean_predicted": 0.41},
        ]
    )
    assert "Spearman falls 0.15" in _hint("compare_samples", table=table)


def test_hosmer_lemeshow_and_bucketed_calibration():
    assert "don't match" in _hint("calibration_test", metric={"p_value": 0.001})
    assert "acceptable" in _hint("calibration_test", metric={"p_value": 0.4})
    buckets = [
        {"bucket": "low", "observed_mean": 0.2, "predicted_mean": 0.1},
        {"bucket": "high", "observed_mean": 0.8, "predicted_mean": 0.79},
    ]
    text = _hint("bucketed_calibration", metric={"buckets": buckets})
    assert "1 of 2 buckets" in text and "low" in text and "high" not in text
    assert "close" in _hint("bucketed_calibration", metric={"buckets": buckets[1:]})


def test_grade_backtest():
    grades = [{"grade": str(g), "traffic_light": "red" if g == 2 else "green"} for g in range(1, 6)]
    metric = {"grades": grades, "portfolio": {"traffic_light": "red"}, "monotonic": False, "herfindahl": 0.9}
    text = _hint("grade_backtest", metric=metric)
    assert "Red grades (2)" in text and "portfolio-level" in text and "monotonic" in text and "concentrated" in text
    calm = {"grades": grades[:1] * 5, "portfolio": {"traffic_light": "green"}, "monotonic": True, "herfindahl": 0.25}
    assert "passes" in _hint("grade_backtest", metric=calm)


# ---- stochastic ------------------------------------------------------------------


def test_fit_distribution_and_proxy():
    good = {"family": "lognorm", "fit_stats": {"ks_p": 0.6}}
    assert "fits well" in _hint("fit_distribution", distribution=good)
    assert "fits poorly" in _hint("fit_distribution", distribution={"family": "norm", "fit_stats": {"ks_p": 0.001}})
    assert "accurate enough to use" in _hint("validate_proxy", diagnostics={"out_of_sample_r2": 0.995})
    assert "isn't accurate" in _hint("validate_proxy", diagnostics={"out_of_sample_r2": 0.9})


# ---- in the build ------------------------------------------------------------------


def test_run_to_carries_hints_only_when_on(prepared):  # noqa: F811
    session, anchor = prepared
    b = staged(session, anchor)
    _, ran = _binned(session, anchor, b)
    assert "next" not in ran

    b.options.decision_hints = True
    fit = b.call_tool("add_block", {"category": "fit_binning", "lane": "est", "params": {"features": FEATURES}})["block"]
    b.call_tool("connect", {"from_block": anchor, "from_port": "out", "to_block": fit, "to_port": "df"})
    ran = b.call_tool("run_to", {"block": fit})
    assert ran["status"] == "green", ran
    # credit_score is banded 'suspicious' in the sample data (see
    # test_a_suspicious_feature_in_a_model_is_flagged_by_the_app).
    assert any("credit_score" in h and "ask_user" in h for h in ran["next"])


def test_plan_stage_steps_and_the_next_stage_prompt_carry_hints(prepared):  # noqa: F811
    session, anchor = prepared
    ANCHOR[:] = [anchor]
    b = staged(session, anchor)
    b.options.decision_hints = True
    built = b.call_tool("plan_stage", {"steps": est_steps()})
    assert built.get("ok"), built
    split, fit = built["steps"]
    assert "next" not in split  # no rule for a split
    assert fit["next"] and fit["next"] == ["All coefficients are significant."]
    assert "  next: " in prompts.built_so_far(b)
    assert "## Hints" in prompts.build_system(b)

    b.options.decision_hints = False
    assert "  next: " not in prompts.built_so_far(b)
    assert "## Hints" not in prompts.build_system(b)
