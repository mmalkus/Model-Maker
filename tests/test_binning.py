"""fit_binning / apply_binning / scorecard_table: the fit-once, apply-
everywhere WoE pair and the univariate analysis it doubles as."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
import pytest

from modelmaker.blocks.binning import apply_binning, fit_binning, scorecard_table
from modelmaker.blocks.feature_analysis import bin_chart
from modelmaker.blocks.library import train_test_split
from modelmaker.blocks.modelling import logistic_regression, predict, woe_transform
from modelmaker.blocks.stat_tests import auc_gini
from modelmaker.cache import CacheStore
from modelmaker.compiler import compile_graph
from modelmaker.graph import Graph, Wire
from modelmaker.runner import Runner

from .helpers import make_block

PD_DATA = Path(__file__).resolve().parents[1] / "sample_data" / "credit_risk_data.csv"
FEATURES = ["credit_score", "dti", "revolving_utilization", "num_late_payments_2yr", "loan_amount", "purpose"]


@pytest.fixture(scope="module")
def pd_split():
    return train_test_split(pl.read_csv(PD_DATA), 0.3, 1)


def test_fit_binning_outputs_artifact_bin_table_and_summary(pd_split):
    train, _ = pd_split
    artifact, bins, summary = fit_binning(train, "default_flag", FEATURES, max_bins=6)

    assert artifact["kind"] == "binning" and artifact["target_type"] == "binary"
    assert set(artifact["features"]) == set(FEATURES)
    assert set(summary["feature"]) == set(FEATURES)
    assert {"iv", "iv_band", "gini", "ks", "monotonic", "fill_rate"} <= set(summary.columns)
    # Sorted strongest first.
    ivs = summary["iv"].to_list()
    assert ivs == sorted(ivs, reverse=True)

    cs = bins.filter(pl.col("feature") == "credit_score").sort("bin")
    # Every feature gets a missing bin, even with no nulls at fit time.
    assert cs["label"][-1] == "__missing__"
    value_bins = cs.filter(pl.col("count") > 0)
    assert value_bins["count"].sum() == train.height
    assert 2 <= value_bins.height <= 6
    # Monotonic: credit_score's event rate falls as the score rises.
    rates = value_bins["target_mean"].to_list()
    assert all(a >= b for a, b in zip(rates, rates[1:]))
    # Direction-adjusted: a higher-is-safer feature still gets a positive Gini.
    row = summary.filter(pl.col("feature") == "credit_score").row(0, named=True)
    assert row["gini"] > 0.5
    assert row["raw_direction"] == "higher = safer"
    assert row["monotonic"] is True


def test_min_bin_share_and_categorical_other_bin():
    rng = np.random.default_rng(0)
    n = 1000
    cat = rng.choice(["a", "b", "c", "rare1", "rare2"], size=n, p=[0.4, 0.3, 0.26, 0.02, 0.02])
    y = (rng.random(n) < np.where(cat == "a", 0.05, 0.2)).astype(int)
    df = pl.DataFrame({"cat": cat, "y": y, "x": rng.normal(size=n)})
    artifact, bins, _ = fit_binning(df, "y", ["cat", "x"], min_bin_share=0.05)

    cat_bins = bins.filter(pl.col("feature") == "cat")
    assert set(cat_bins["label"]) == {"a", "b", "c", "__other__", "__missing__"}
    assert artifact["features"]["cat"]["mapping"]["rare1"] == artifact["features"]["cat"]["other_bin"]
    x_bins = bins.filter((pl.col("feature") == "x") & (pl.col("count") > 0))
    assert (x_bins["share"] >= 0.05 - 1e-9).all()


def test_apply_binning_uses_fit_sample_woe_and_handles_nulls_and_unseen(pd_split):
    train, test = pd_split
    artifact, _, _ = fit_binning(train, "default_flag", FEATURES)
    applied_train = apply_binning(train, artifact)
    # On the fit sample, apply reproduces the fit's own WoE per bin.
    woe_by_bin = {b["label"]: b["woe"] for b in artifact["features"]["purpose"]["bins"]}
    got = applied_train.select("purpose", "purpose_woe").unique()
    for purpose, woe in got.iter_rows():
        assert woe in woe_by_bin.values()

    odd = test.head(3).with_columns(
        pl.Series("credit_score", [None, 700, 500], dtype=pl.Int64),
        pl.Series("purpose", ["never_seen", "car", None]),
    )
    out = apply_binning(odd, artifact, output="both")
    cs_bins = artifact["features"]["credit_score"]["bins"]
    assert out["credit_score_bin"][0] == "__missing__"
    assert out["credit_score_woe"][0] == cs_bins[artifact["features"]["credit_score"]["missing_bin"]]["woe"]
    assert out["purpose_bin"][0] == "__other__"
    assert out["purpose_bin"][2] == "__missing__"
    assert out.columns[: len(odd.columns)] == odd.columns


def test_fit_apply_avoids_the_leakage_of_refitting_woe_on_test(pd_split):
    """The regression the pair exists for: woe_transform run on the test
    sample re-derives WoE from the test target and inflates test Gini."""
    train, test = pd_split
    woe_cols = [f"{f}_woe" for f in FEATURES]

    leaky_train, leaky_test = train, test
    for f in FEATURES:
        leaky_train = woe_transform(leaky_train, f, "default_flag", 5)
        leaky_test = woe_transform(leaky_test, f, "default_flag", 5)
    _, leaky_model = logistic_regression(leaky_train, "default_flag", woe_cols)
    leaky_gini = auc_gini(predict(leaky_test, leaky_model), "predicted_proba", "default_flag")["gini"]

    artifact, _, _ = fit_binning(train, "default_flag", FEATURES)
    _, model = logistic_regression(apply_binning(train, artifact), "default_flag", woe_cols)
    honest_gini = auc_gini(predict(apply_binning(test, artifact), model), "predicted_proba", "default_flag")["gini"]

    assert honest_gini < leaky_gini


def test_continuous_target_binning_and_target_mean_encoding():
    rng = np.random.default_rng(1)
    x = rng.normal(size=600)
    lgd = np.clip(0.4 + 0.2 * x + rng.normal(scale=0.1, size=600), 0, 1)
    df = pl.DataFrame({"x": x, "product": rng.choice(["m", "u"], size=600), "lgd": lgd})
    artifact, bins, summary = fit_binning(df, "lgd", ["x", "product"])

    assert artifact["target_type"] == "continuous"
    assert "woe" not in bins.columns
    row = summary.filter(pl.col("feature") == "x").row(0, named=True)
    assert row["r2_binned"] > 0.5 and row["spearman_raw"] > 0.7
    out = apply_binning(df, artifact, output="target_mean")
    assert "x_tm" in out.columns
    with pytest.raises(ValueError, match="binary target"):
        apply_binning(df, artifact, output="woe")


def test_constant_feature_has_no_spurious_correlation():
    df = pl.DataFrame({"c": ["a"] * 50, "y": [0.1, 0.9] * 25})
    _, _, summary = fit_binning(df, "y", ["c"])
    assert summary["spearman_binned"][0] is None
    assert summary["r2_binned"][0] == pytest.approx(0.0)


def test_scorecard_table_points_add_up_to_the_model_score(pd_split):
    train, _ = pd_split
    artifact, _, _ = fit_binning(train, "default_flag", FEATURES)
    applied = apply_binning(train, artifact, output="both")
    _, model = logistic_regression(applied, "default_flag", [f"{f}_woe" for f in FEATURES])
    table = scorecard_table(model, artifact, base_score=600, base_odds=50, pdo=20)

    assert set(table["feature"]) == set(FEATURES)
    # Total points for one applicant == the PDO scaling of their log-odds.
    row = applied.row(0, named=True)
    total = sum(
        table.filter((pl.col("feature") == f) & (pl.col("label") == row[f"{f}_bin"]))["points"][0] for f in FEATURES
    )
    log_odds_bad = model["intercept"] + sum(model["coefficients"][f"{f}_woe"] * row[f"{f}_woe"] for f in FEATURES)
    factor = 20 / np.log(2)
    expected = 600 - factor * np.log(50) - factor * log_odds_bad
    assert total == pytest.approx(expected, abs=0.02 * len(FEATURES))


def test_bin_chart_renders_png(pd_split, tmp_path):
    _, bins, _ = fit_binning(pd_split[0], "default_flag", ["credit_score"])
    png = bin_chart(bins, "credit_score", output_dir=str(tmp_path), block_id="b1")
    assert png[:4] == b"\x89PNG"
    assert (tmp_path / "bin_chart_b1.png").exists()


def test_binning_pipeline_runs_in_engine_and_compiles_to_the_same_result():
    blocks = {
        "b_read": make_block("b_read", "read_csv", params={"path": str(PD_DATA)}),
        "b_split": make_block("b_split", "train_test_split", params={"test_size": 0.3, "seed": 1, "stratify_col": "default_flag"}),
        "b_fit": make_block("b_fit", "fit_binning", params={"target": "default_flag", "features": FEATURES}),
        "b_apply_tr": make_block("b_apply_tr", "apply_binning", params={}),
        "b_apply_te": make_block("b_apply_te", "apply_binning", params={}),
        "b_lr": make_block("b_lr", "logistic_regression", params={"target": "default_flag", "features": [f"{f}_woe" for f in FEATURES]}),
        "b_card": make_block("b_card", "scorecard_table", params={}),
    }
    wires = [
        ("b_read", "out", "b_split", "df"),
        ("b_split", "train", "b_fit", "df"),
        ("b_split", "train", "b_apply_tr", "df"),
        ("b_fit", "binning", "b_apply_tr", "binning"),
        ("b_split", "test", "b_apply_te", "df"),
        ("b_fit", "binning", "b_apply_te", "binning"),
        ("b_apply_tr", "out", "b_lr", "df"),
        ("b_lr", "model", "b_card", "model"),
        ("b_fit", "binning", "b_card", "binning"),
    ]
    graph = Graph(blocks=blocks, wires={f"w{i}": Wire(f"w{i}", *w) for i, w in enumerate(wires)})
    runner = Runner(graph, CacheStore())
    runner.refresh("b_read")
    runner.run_all()
    for bid in blocks:
        assert runner.status(bid) == "green", (bid, runner.state[bid].last_error)

    applied = runner.cache.get(runner.state["b_apply_te"].last_successful_key).outputs["out"]
    assert applied.schema_meta["credit_score_woe"].role.value == "feature"

    ns: dict = {}
    exec(compile(compile_graph(graph, runner=runner), "<compiled>", "exec"), ns)
    engine_card = runner.cache.get(runner.state["b_card"].last_successful_key).outputs["table"].data
    compiled_card = next(v for k, v in ns.items() if k.startswith("b_card") and isinstance(v, pl.DataFrame))
    assert compiled_card.to_dicts() == engine_card.to_dicts()
