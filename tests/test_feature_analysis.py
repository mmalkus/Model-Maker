import polars as pl
import pytest

from modelmaker.blocks import feature_analysis


def test_iv_table_raises_when_target_is_not_binary():
    df = pl.DataFrame({"x": [1, 2, 3], "y": [1, 1, 1]})
    with pytest.raises(ValueError, match="must contain both 0 and 1"):
        feature_analysis.iv_table(df, target="y")


def test_iv_table_bands_a_useless_feature_as_useless():
    # x carries no information about y at all (constant per group is
    # irrelevant here) -- a single bin with identical good/bad shares should
    # land IV at (or extremely close to) zero, banded "useless".
    df = pl.DataFrame({"x": [1, 1, 1, 1], "y": [0, 1, 0, 1]})
    result = feature_analysis.iv_table(df, target="y", bins=1)
    row = result["rows"][0]
    assert row["feature"] == "x"
    assert row["iv"] == pytest.approx(0.0, abs=1e-9)
    assert row["iv_band"] == "useless"


def test_iv_table_defaults_to_every_non_target_column():
    df = pl.DataFrame({"x": [1, 2, 3, 4], "z": [4, 3, 2, 1], "y": [0, 0, 1, 1]})
    result = feature_analysis.iv_table(df, target="y")
    assert {r["feature"] for r in result["rows"]} == {"x", "z"}


def test_correlation_matrix_requires_at_least_two_features():
    df = pl.DataFrame({"x": [1, 2, 3]})
    with pytest.raises(ValueError, match="at least 2"):
        feature_analysis.correlation_matrix(df, features=["x"])


def test_correlation_matrix_nulls_out_vif_for_a_constant_column():
    df = pl.DataFrame({"a": [1, 1, 1, 1], "b": [1, 2, 3, 4]})
    result = feature_analysis.correlation_matrix(df)
    vif_by_feature = {row["feature"]: row["vif"] for row in result["vif"]}
    assert vif_by_feature["a"] is None
