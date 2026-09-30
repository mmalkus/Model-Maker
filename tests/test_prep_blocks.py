"""Data-prep additions: out-of-time split, stratified train/test split,
derived columns, one-hot encoding, multi-aggregation group-by, and the
feature-level analysis tables (characteristic stability, target trend)."""

from __future__ import annotations

from datetime import date

import polars as pl
import pytest

from modelmaker.blocks.feature_analysis import characteristic_stability, target_trend
from modelmaker.blocks.library import (
    _derive_columns_meta,
    _one_hot_meta,
    derive_columns,
    groupby_agg,
    one_hot_encode,
    time_split,
    train_test_split,
)
from modelmaker.packet import ColumnMeta, ColumnRole


def test_time_split_on_iso_strings_and_dates():
    df = pl.DataFrame({"d": ["2025-06-30", "2025-12-31", "2026-01-01", "2026-05-01", None], "v": [1, 2, 3, 4, 5]})
    dev, oot = time_split(df, "d", "2026-01-01")
    assert dev["v"].to_list() == [1, 2]
    assert oot["v"].to_list() == [3, 4]

    dev, oot = time_split(df.with_columns(pl.col("d").str.to_date()), "d", "2026-01-01", oot_end="2026-03-01")
    assert oot["v"].to_list() == [3]


def test_stratified_split_keeps_the_event_rate():
    df = pl.DataFrame({"y": [1] * 50 + [0] * 950, "i": range(1000)})
    train, test = train_test_split(df, 0.3, seed=3, stratify_col="y")
    assert train.height + test.height == 1000
    assert set(train["i"]).isdisjoint(set(test["i"]))
    assert test["y"].sum() == 15
    assert train["y"].mean() == pytest.approx(test["y"].mean(), abs=0.002)


def test_unstratified_split_is_unchanged():
    df = pl.DataFrame({"i": range(100)})
    train, test = train_test_split(df, 0.2, seed=0)
    shuffled = df.sample(fraction=1.0, shuffle=True, seed=0)
    assert test["i"].to_list() == shuffled.head(20)["i"].to_list()
    assert train.height == 80


def test_derive_columns_adds_features_and_keeps_replaced_roles():
    df = pl.DataFrame({"bal": [50.0, 0.0], "lim": [100.0, 0.0], "age": [25, 40]})
    out = derive_columns(
        df,
        {
            "util": "bal / NULLIF(lim, 0)",
            "young": "CASE WHEN age < 30 THEN 'y' ELSE 'n' END",
            "age": "age + 1",
        },
    )
    assert out["util"].to_list() == [0.5, None]
    assert out["young"].to_list() == ["y", "n"]
    assert out["age"].to_list() == [26, 41]

    in_meta = {c: ColumnMeta(dtype=str(df.schema[c])) for c in df.columns}
    in_meta["age"] = ColumnMeta(dtype="Int64", role=ColumnRole.SEGMENT)
    meta = _derive_columns_meta({"df": in_meta}, {"out": out}, {})["out"]
    assert meta["util"].role == ColumnRole.FEATURE
    assert meta["age"].role == ColumnRole.SEGMENT


def test_one_hot_encode_drops_reference_level_and_guards_ids():
    df = pl.DataFrame({"product": ["a", "b", "c", None], "x": [1, 2, 3, 4]})
    out = one_hot_encode(df, ["product"])
    dummies = [c for c in out.columns if c.startswith("product_")]
    assert "product" not in out.columns
    assert len(dummies) == 3  # 4 levels incl. null, first dropped
    assert all(out[c].dtype == pl.Int8 for c in dummies)
    # "a" (alphabetically first) is the reference level: its row is all zeros.
    assert dummies == ["product_b", "product_c", "product_null"]
    assert out.select(dummies).sum_horizontal().to_list() == [0, 1, 1, 1]
    # ...whatever order the rows come in.
    reordered = one_hot_encode(df.reverse(), ["product"])
    assert [c for c in reordered.columns if c.startswith("product_")] == dummies

    meta = _one_hot_meta({"df": {"product": ColumnMeta("String"), "x": ColumnMeta("Int64")}}, {"out": out}, {})["out"]
    assert all(meta[c].role == ColumnRole.FEATURE for c in dummies)

    with pytest.raises(ValueError, match="max_categories"):
        one_hot_encode(pl.DataFrame({"id": [str(i) for i in range(40)]}), ["id"])


def test_groupby_agg_accepts_a_list_of_aggregations():
    df = pl.DataFrame({"g": ["a", "a", "b"], "v": [1.0, 3.0, 5.0], "w": [1, 1, 1]})
    out = groupby_agg(df, ["g"], {"v": ["mean", "count"], "w": "sum"}).sort("g")
    assert out.columns == ["g", "v_mean", "v_count", "w"]
    assert out["v_mean"].to_list() == [2.0, 5.0]


def test_characteristic_stability_handles_numeric_categorical_and_nulls():
    exp = pl.DataFrame({"x": list(range(100)), "c": ["a"] * 50 + ["b"] * 50, "n": [1.0] * 100})
    act = pl.DataFrame({"x": list(range(50, 150)), "c": ["a"] * 50 + ["z"] * 50, "n": [None] * 50 + [1.0] * 50})
    table = characteristic_stability(exp, act, bins=5)
    rows = {r["feature"]: r for r in table.to_dicts()}
    assert rows["c"]["type"] == "categorical" and rows["c"]["band"] == "unstable"
    assert rows["x"]["psi"] > 0.25
    assert rows["n"]["actual_null_share"] == 0.5 and rows["n"]["psi"] > 0.1
    assert table["psi"].to_list() == sorted(table["psi"].to_list(), reverse=True)
    same = characteristic_stability(exp, exp)
    assert same["psi"].max() == pytest.approx(0.0, abs=1e-9)


def test_target_trend_by_quarter():
    df = pl.DataFrame(
        {
            "d": [date(2025, 1, 5), date(2025, 2, 1), date(2025, 4, 1), date(2026, 1, 1)],
            "y": [1, 0, 0, 1],
            "x": [1.0, 3.0, 5.0, None],
            "s": ["a", None, "b", "c"],
        }
    )
    out = target_trend(df, "d", "y", period="quarter", features=["x", "s"])
    assert out["period"].to_list() == ["2025-Q1", "2025-Q2", "2026-Q1"]
    assert out["target_mean"].to_list() == [0.5, 0.0, 1.0]
    assert out["x_mean"].to_list() == [2.0, 5.0, None]
    assert out["s_null_share"].to_list() == [0.5, 0.0, 0.0]
