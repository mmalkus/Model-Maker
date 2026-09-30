"""LGD/CCF/EAD and calibration additions: discounted recoveries, EAD from
CCF, long-run averages, margin of conservatism, calibration to a central
tendency, stepwise selection, coefficient statistics, master-scale grade
statistics, and the realised LGD/CCF being tagged as the target -- plus an
end-to-end LGD and CCF pipeline run through the engine and the compiler."""

from __future__ import annotations

import math
from datetime import date
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from modelmaker.blocks.modelling import (
    _compute_lgd_meta,
    _moc_meta,
    assign_rating_grade,
    calibrate_model,
    compute_ead,
    discount_recoveries,
    fit_master_scale,
    lgd_regression,
    logistic_regression,
    long_run_average,
    margin_of_conservatism,
    predict,
    stepwise_selection,
)
from modelmaker.cache import CacheStore
from modelmaker.compiler import compile_graph
from modelmaker.graph import Graph, Wire
from modelmaker.packet import ColumnMeta, ColumnRole
from modelmaker.runner import Runner

from .helpers import make_block

SAMPLE = Path(__file__).resolve().parents[1] / "sample_data"


def test_discount_recoveries_discounts_and_totals_per_facility():
    facilities = pl.DataFrame(
        {"id": ["A", "B", "C"], "default_date": ["2020-01-01", "2020-01-01", "2020-01-01"], "rate": [0.0, 0.1, 0.1]}
    )
    cashflows = pl.DataFrame(
        {
            "id": ["A", "A", "B"],
            "cf_date": ["2021-01-01", "2022-01-01", "2021-01-01"],
            "rec": [100.0, 100.0, 110.0],
            "cost": [10.0, None, 0.0],
        }
    )
    out = discount_recoveries(facilities, cashflows, "id", "default_date", "cf_date", "rec", "cost", annual_rate=0.1).sort("id")
    years_a1 = 366 / 365.25  # 2020 is a leap year
    years_a2 = 731 / 365.25
    assert out["pv_recovery"][0] == pytest.approx(100 * 1.1**-years_a1 + 100 * 1.1**-years_a2)
    assert out["pv_cost"][0] == pytest.approx(10 * 1.1**-years_a1)
    assert out["n_cashflows"].to_list() == [2, 1, 0]
    assert out.row(2, named=True)["pv_recovery"] == 0.0  # no cash flows
    assert out["workout_years"][0] == pytest.approx(years_a2)
    # A per-facility rate column overrides the flat rate.
    by_rate = discount_recoveries(facilities, cashflows, "id", "default_date", "cf_date", "rec", rate_col="rate").sort("id")
    assert by_rate["pv_recovery"][0] == pytest.approx(200.0)


def test_compute_ead_and_lgd_target_role():
    df = pl.DataFrame({"bal": [60.0, 120.0], "lim": [100.0, 100.0], "predicted": [0.5, 0.5]})
    out = compute_ead(df, "bal", "lim", "predicted")
    assert out["ead_predicted"].to_list() == [80.0, 120.0]

    lgd_df = pl.DataFrame({"ead": [1.0], "lgd": [0.3]})
    meta = _compute_lgd_meta({"df": {"ead": ColumnMeta("Float64")}}, {"out": lgd_df}, {})["out"]
    assert meta["lgd"].role == ColumnRole.TARGET
    taken = {"ead": ColumnMeta("Float64"), "flag": ColumnMeta("Int64", role=ColumnRole.TARGET)}
    meta = _compute_lgd_meta({"df": taken}, {"out": lgd_df.with_columns(flag=pl.lit(1))}, {})["out"]
    assert meta["lgd"].role == ColumnRole.FEATURE


def test_compute_ccf_no_headroom_can_be_null():
    from modelmaker.blocks.modelling import compute_ccf

    df = pl.DataFrame({"lim": [100.0, 100.0], "ref": [50.0, 105.0], "dft": [75.0, 110.0]})
    assert compute_ccf(df, "lim", "ref", "dft")["ccf"].to_list() == [0.5, 0.0]
    assert compute_ccf(df, "lim", "ref", "dft", no_headroom="null")["ccf"].to_list() == [0.5, None]


def test_long_run_average_weightings():
    df = pl.DataFrame(
        {
            "d": [date(2020, 3, 1), date(2020, 6, 1), date(2021, 1, 1)],
            "lgd": [0.2, 0.4, 0.9],
            "ead": [100.0, 300.0, 100.0],
        }
    )
    table, metric = long_run_average(df, "lgd", "d", weight_col="ead")
    assert table["period"].to_list() == ["2020", "2021"]
    assert metric["default_weighted"] == pytest.approx(0.5)
    assert metric["time_weighted"] == pytest.approx((0.3 + 0.9) / 2)
    assert metric["exposure_weighted"] == pytest.approx((20 + 120 + 90) / 500)


def test_margin_of_conservatism_moves_the_predicted_role():
    df = pl.DataFrame({"predicted": [0.1, 0.95]})
    out = margin_of_conservatism(df, "predicted", add_on=0.1, multiplier=1.0, cap=1.0)
    assert out["predicted_moc"].to_list() == pytest.approx([0.2, 1.0])
    meta = _moc_meta({"df": {"predicted": ColumnMeta("Float64", role=ColumnRole.PREDICTED)}}, {"out": out}, {})["out"]
    assert meta["predicted_moc"].role == ColumnRole.PREDICTED
    assert meta["predicted"].role == ColumnRole.FEATURE


@pytest.fixture(scope="module")
def pd_data():
    rng = np.random.default_rng(4)
    n = 2000
    x1, x2, noise = rng.normal(size=(3, n))
    y = (rng.random(n) < 1 / (1 + np.exp(-(-2.5 + 1.2 * x1 - 0.8 * x2)))).astype(int)
    return pl.DataFrame({"x1": x1, "x2": x2, "noise": noise, "y": y})


def test_logistic_regression_reports_coefficient_statistics(pd_data):
    _, model = logistic_regression(pd_data, "y", ["x1", "x2", "noise"], C=1e6)
    stats = model["statistics"]
    assert set(stats) == {"intercept", "x1", "x2", "noise"}
    assert stats["x1"]["p_value"] < 1e-6 and stats["x2"]["p_value"] < 1e-6
    assert stats["noise"]["p_value"] > 0.01
    assert stats["x1"]["z"] == pytest.approx(stats["x1"]["estimate"] / stats["x1"]["std_error"])


def test_stepwise_selection_keeps_real_drivers(pd_data):
    for direction in ("forward", "backward", "both"):
        result = stepwise_selection(pd_data, "y", ["noise", "x1", "x2"], direction=direction)
        assert sorted(result["selected"]) == ["x1", "x2"], direction
    capped = stepwise_selection(pd_data, "y", ["noise", "x1", "x2"], direction="forward", max_features=1)
    assert capped["selected"] == ["x1"]
    signs = {c["feature"]: c["sign"] for c in capped["coefficients"]}
    assert signs["x1"] == "+"


def test_stepwise_selection_on_a_fractional_target():
    rng = np.random.default_rng(2)
    x, noise = rng.normal(size=(2, 800))
    lgd = np.clip(1 / (1 + np.exp(-(0.2 + 0.9 * x))) + rng.normal(scale=0.1, size=800), 0, 1)
    result = stepwise_selection(pl.DataFrame({"x": x, "noise": noise, "lgd": lgd}), "lgd", ["x", "noise"])
    assert result["target_type"] == "continuous"
    assert result["selected"] == ["x"]


def test_lgd_regression_reports_robust_statistics():
    rng = np.random.default_rng(5)
    x = rng.normal(size=500)
    lgd = np.clip(1 / (1 + np.exp(-(0.3 + 0.8 * x))) + rng.normal(scale=0.15, size=500), 0, 1)
    _, model = lgd_regression(pl.DataFrame({"x": x, "lgd": lgd}), "lgd", ["x"])
    assert model["statistics"]["x"]["p_value"] < 1e-6
    assert model["statistics"]["x"]["std_error"] > 0


def test_calibrate_model_hits_the_central_tendency_and_keeps_ranking(pd_data):
    _, model = logistic_regression(pd_data, "y", ["x1", "x2"])
    calibrated = calibrate_model(pd_data, model, central_tendency=0.2)
    before = predict(pd_data, model)["predicted_proba"]
    after = predict(pd_data, calibrated)["predicted_proba"]
    assert after.mean() == pytest.approx(0.2, abs=1e-6)
    assert calibrated["calibration"]["mean_prediction_after"] == pytest.approx(0.2, abs=1e-6)
    assert calibrated["coefficients"] == model["coefficients"]
    assert (before.rank() == after.rank()).all()
    assert model["intercept"] != calibrated["intercept"]  # input artifact not mutated in place
    with pytest.raises(ValueError):
        calibrate_model(pd_data, model, central_tendency=1.5)


def test_master_scale_min_grade_share_and_grade_pd():
    rng = np.random.default_rng(0)
    score = rng.beta(0.5, 5, size=2000)
    y = (rng.random(2000) < score).astype(int)
    df = pl.DataFrame({"pd": score, "y": y})
    scale = fit_master_scale(df, "pd", "y", n_grades=12, algorithm="monotonic_default_rate", min_grade_share=0.05)
    counts = [g["n"] for g in scale["grades"]]
    assert sum(counts) == 2000
    assert min(counts) >= 0.05 * 2000
    assert all(g["pd"] is not None and g["observed_rate"] is not None for g in scale["grades"])
    graded = assign_rating_grade(df, scale)
    first = scale["grades"][0]
    assert graded.filter(pl.col("grade") == "1")["grade_pd"].unique().to_list() == [pytest.approx(first["pd"])]
    # An old artifact without per-grade PDs still assigns grades, just no grade_pd.
    legacy = {**scale, "grades": [{k: g[k] for k in ("grade", "lower", "upper")} for g in scale["grades"]]}
    assert "grade_pd" not in assign_rating_grade(df, legacy).columns


def _run(blocks, wires):
    graph = Graph(blocks=blocks, wires={f"w{i}": Wire(f"w{i}", *w) for i, w in enumerate(wires)})
    runner = Runner(graph, CacheStore())
    for bid, blk in blocks.items():
        if blk.block_type == "input":
            runner.refresh(bid)
    runner.run_all()
    for bid in blocks:
        assert runner.status(bid) == "green", (bid, runner.state[bid].last_error)
    return graph, runner


def _output(runner, bid, port):
    value = runner.cache.get(runner.state[bid].last_successful_key).outputs[port]
    return getattr(value, "data", value)


def test_lgd_pipeline_end_to_end_and_compiled():
    """The demo dataset's defaults to a validated, downturn-adjusted LGD
    model using registry blocks only."""
    B = make_block
    blocks = {
        "raw": B("raw", "read_csv", params={"path": str(SAMPLE / "credit_risk_data.csv")}),
        "excl": B("excl", "apply_exclusions", params={"rules": [{"name": "defaulted", "expr": "default_flag = 1"}, {"name": "ead", "expr": "balance_at_default > 0"}]}),
        "lgd": B("lgd", "compute_lgd", params={"ead_col": "balance_at_default", "recovered_col": "recovery_amount", "cost_col": "workout_cost"}),
        "fill": B("fill", "missing_value_treatment", params={"strategies": {"collateral_value": {"method": "constant", "value": 0.0}, "collateral_type": {"method": "constant", "value": "none"}}}),
        "feat": B("feat", "derive_columns", params={"expressions": {"cover": "collateral_value / balance_at_default"}}),
        "lra": B("lra", "long_run_average", params={"date_col": "default_date", "weight_col": "balance_at_default"}),
        "uni": B("uni", "fit_binning", params={"features": ["cover", "is_secured", "product_type", "collateral_type", "credit_score", "region"]}),
        "ohe": B("ohe", "one_hot_encode", params={"columns": ["product_type"]}),
        "split": B("split", "train_test_split", params={"test_size": 0.3, "seed": 1}),
        "step": B("step", "stepwise_selection", params={"features": ["cover", "is_secured", "product_type_term_loan", "credit_score"]}),
        "reg": B("reg", "lgd_regression", params={"features": ["cover", "is_secured", "product_type_term_loan"]}),
        "pred": B("pred", "predict", params={}),
        "moc": B("moc", "margin_of_conservatism", params={"add_on": 0.05}),
        "acc": B("acc", "continuous_accuracy", params={}),
        "cmp": B("cmp", "compare_samples", params={}),
    }
    wires = [
        ("raw", "out", "excl", "df"),
        ("excl", "out", "lgd", "df"),
        ("lgd", "out", "fill", "df"),
        ("fill", "out", "feat", "df"),
        ("feat", "out", "lra", "df"),
        ("feat", "out", "uni", "df"),
        ("feat", "out", "ohe", "df"),
        ("ohe", "out", "split", "df"),
        ("split", "train", "step", "df"),
        ("split", "train", "reg", "df"),
        ("split", "test", "pred", "df"),
        ("reg", "model", "pred", "model"),
        ("pred", "predictions", "moc", "df"),
        ("pred", "predictions", "acc", "df"),
        ("reg", "predictions", "cmp", "sample_1"),
        ("pred", "predictions", "cmp", "sample_2"),
    ]
    graph, runner = _run(blocks, wires)

    lra = _output(runner, "lra", "metric")
    assert 0.1 < lra["default_weighted"] < 0.9
    summary = _output(runner, "uni", "summary")
    assert set(summary["feature"]) == {"cover", "is_secured", "product_type", "collateral_type", "credit_score", "region"}
    assert summary["r2_binned"][0] > 0.1  # secured vs unsecured drives LGD in the demo data
    acc = _output(runner, "acc", "metric")
    assert acc["r2"] > 0.1 and acc["spearman"] > 0.3
    cmp = _output(runner, "cmp", "table")
    assert cmp["sample"].to_list() == ["train", "test"] and "rmse" in cmp.columns
    moc = _output(runner, "moc", "out")
    assert (moc["predicted_moc"] >= moc["predicted"]).all()

    ns: dict = {}
    exec(compile(compile_graph(graph, runner=runner), "<compiled>", "exec"), ns)
    compiled_cmp = next(v for k, v in ns.items() if k.startswith("cmp") and isinstance(v, pl.DataFrame))
    assert compiled_cmp.to_dicts() == cmp.to_dicts()


def test_ccf_pipeline_end_to_end():
    B = make_block
    blocks = {
        "raw": B("raw", "read_csv", params={"path": str(SAMPLE / "credit_risk_data.csv")}),
        "cards": B("cards", "apply_exclusions", params={"rules": [{"name": "defaulted cards", "expr": "product_type = 'credit_card' AND default_flag = 1"}]}),
        "ccf": B("ccf", "compute_ccf", params={"limit_col": "credit_limit", "balance_ref_col": "balance_at_reference", "balance_default_col": "balance_at_default", "no_headroom": "null"}),
        "excl": B("excl", "apply_exclusions", params={"rules": [{"name": "headroom", "expr": "ccf IS NOT NULL"}]}),
        "feat": B("feat", "derive_columns", params={"expressions": {"utilisation_at_ref": "balance_at_reference / credit_limit"}}),
        "tsplit": B("tsplit", "time_split", params={"date_col": "reference_date", "cutoff": "2024-07-01"}),
        "reg": B("reg", "lgd_regression", params={"features": ["utilisation_at_ref", "credit_score"]}),
        "pred": B("pred", "predict", params={}),
        "ead": B("ead", "compute_ead", params={"balance_col": "balance_at_reference", "limit_col": "credit_limit"}),
        "ead_acc": B("ead_acc", "continuous_accuracy", params={"actual_col": "balance_at_default", "predicted_col": "ead_predicted"}),
        "stab": B("stab", "characteristic_stability", params={"features": ["utilisation_at_ref", "credit_score", "region"]}),
    }
    wires = [
        ("raw", "out", "cards", "df"),
        ("cards", "out", "ccf", "df"),
        ("ccf", "out", "excl", "df"),
        ("excl", "out", "feat", "df"),
        ("feat", "out", "tsplit", "df"),
        ("tsplit", "development", "reg", "df"),
        ("tsplit", "out_of_time", "pred", "df"),
        ("reg", "model", "pred", "model"),
        ("pred", "predictions", "ead", "df"),
        ("ead", "out", "ead_acc", "df"),
        ("tsplit", "development", "stab", "expected"),
        ("tsplit", "out_of_time", "stab", "actual"),
    ]
    graph, runner = _run(blocks, wires)
    ccf_meta = runner.cache.get(runner.state["ccf"].last_successful_key).outputs["out"].schema_meta
    assert ccf_meta["ccf"].role == ColumnRole.TARGET
    assert _output(runner, "reg", "model")["statistics"]["utilisation_at_ref"]["std_error"] > 0
    assert _output(runner, "ead_acc", "metric")["r2"] > 0.8
    assert set(_output(runner, "stab", "table")["feature"]) == {"utilisation_at_ref", "credit_score", "region"}
    assert not math.isnan(_output(runner, "ead_acc", "metric")["mae"])
