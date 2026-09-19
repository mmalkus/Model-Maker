import polars as pl
import pytest

from modelmaker.blocks import data_quality
from modelmaker.packet import ColumnMeta, ColumnRole


def test_data_profile_reports_fill_rate_and_dominant_value():
    df = pl.DataFrame({"a": [1, 1, 2, None], "b": ["x", "x", "x", "y"]})
    result = data_quality.data_profile(df)
    by_col = {r["column"]: r for r in result["columns"]}
    assert by_col["a"]["fill_rate"] == 0.75
    assert by_col["a"]["null_count"] == 1
    assert by_col["b"]["dominant_value"] == "x"
    assert by_col["b"]["dominant_value_share"] == 0.75


def test_data_profile_computes_percentiles_for_numeric_columns_only():
    df = pl.DataFrame({"a": [1.0, 2.0, 3.0, 4.0], "b": ["w", "x", "y", "z"]})
    result = data_quality.data_profile(df)
    by_col = {r["column"]: r for r in result["columns"]}
    assert by_col["a"]["percentiles"]["p50"] == pytest.approx(df["a"].quantile(0.5))
    assert by_col["b"]["percentiles"] is None


def test_data_profile_handles_an_entirely_null_column():
    df = pl.DataFrame({"a": [None, None, None]}, schema={"a": pl.Int64})
    result = data_quality.data_profile(df)
    row = result["columns"][0]
    assert row["fill_rate"] == 0.0
    assert row["dominant_value"] is None
    assert row["min"] is None


def test_data_profile_respects_features_filter():
    df = pl.DataFrame({"a": [1], "b": [2], "c": [3]})
    result = data_quality.data_profile(df, features=["a", "c"])
    assert {r["column"] for r in result["columns"]} == {"a", "c"}


def test_apply_exclusions_tracks_the_waterfall_in_order():
    df = pl.DataFrame({"amount": [10, -5, 20, 0, 30]})
    rules = [
        {"name": "drop negative", "expr": "amount >= 0"},
        {"name": "drop zero", "expr": "amount > 0"},
    ]
    kept, summary = data_quality.apply_exclusions(df, rules)
    assert kept["amount"].to_list() == [10, 20, 30]
    assert summary["starting_population"] == 5
    assert summary["final_population"] == 3
    assert summary["total_dropped"] == 2
    assert [s["step"] for s in summary["steps"]] == ["starting population", "drop negative", "drop zero"]
    assert summary["steps"][1]["dropped"] == 1  # the one negative row
    assert summary["steps"][2]["dropped"] == 1  # the one zero row


def test_apply_exclusions_metadata_passes_through_only_the_dataframe_port():
    df = pl.DataFrame({"a": [1, 2]})
    in_meta = {"a": ColumnMeta(dtype="Int64", role=ColumnRole.FEATURE)}
    kept, summary = data_quality.apply_exclusions(df, [])
    result = data_quality._apply_exclusions_meta({"df": in_meta}, {"out": kept}, {})
    assert result == {"out": {"a": in_meta["a"]}}


def test_data_quality_rules_not_null_flags_nulls():
    df = pl.DataFrame({"a": [1, None, 3]})
    result = data_quality.data_quality_rules(df, [{"name": "a required", "rule": "not_null", "column": "a"}])
    row = result["results"][0]
    assert row["violations"] == 1
    assert row["passed"] is False
    assert result["passed"] is False


def test_data_quality_rules_unique_on_a_composite_key():
    df = pl.DataFrame({"acct": [1, 1, 2], "month": ["jan", "jan", "jan"]})
    result = data_quality.data_quality_rules(
        df, [{"name": "no dup account-months", "rule": "unique", "columns": ["acct", "month"]}]
    )
    assert result["results"][0]["violations"] == 1


def test_data_quality_rules_in_set_and_between():
    df = pl.DataFrame({"region": ["east", "west", "nowhere"], "lgd": [0.1, 1.5, -0.2]})
    result = data_quality.data_quality_rules(
        df,
        [
            {"name": "known region", "rule": "in_set", "column": "region", "values": ["east", "west"]},
            {"name": "lgd bounded", "rule": "between", "column": "lgd", "min": 0.0, "max": 1.0},
        ],
    )
    by_name = {r["name"]: r for r in result["results"]}
    assert by_name["known region"]["violations"] == 1
    assert by_name["lgd bounded"]["violations"] == 2


def test_data_quality_rules_row_count_between():
    df = pl.DataFrame({"a": [1, 2, 3]})
    result = data_quality.data_quality_rules(df, [{"name": "enough rows", "rule": "row_count_between", "min": 10}])
    assert result["results"][0]["violations"] == 1


def test_data_quality_rules_passed_ignores_warning_severity():
    df = pl.DataFrame({"a": [1, None]})
    result = data_quality.data_quality_rules(
        df, [{"name": "soft check", "rule": "not_null", "column": "a", "severity": "warning"}]
    )
    assert result["results"][0]["passed"] is False
    assert result["passed"] is True  # only error-severity rules gate the top-level flag


def test_missing_value_treatment_fills_constant_mean_and_mode():
    df = pl.DataFrame({"a": [1.0, None, 3.0], "b": [None, "x", "x"], "c": [None, 2, 4]})
    out = data_quality.missing_value_treatment(
        df,
        {
            "a": {"method": "mean"},
            "b": {"method": "mode"},
            "c": {"method": "constant", "value": 0},
        },
    )
    assert out["a"].to_list() == [1.0, 2.0, 3.0]
    assert out["b"].to_list() == ["x", "x", "x"]
    assert out["c"].to_list() == [0, 2, 4]


def test_missing_value_treatment_adds_indicator_before_filling():
    df = pl.DataFrame({"a": [1.0, None, 3.0]})
    out = data_quality.missing_value_treatment(df, {"a": {"method": "mean", "add_indicator": True}})
    assert out["a_was_missing"].to_list() == [0, 1, 0]
    assert out["a"].to_list() == [1.0, 2.0, 3.0]


def test_missing_value_treatment_drop_rows_does_not_touch_other_columns_fills():
    df = pl.DataFrame({"a": [1.0, None, 3.0], "b": [None, 5.0, 6.0]})
    out = data_quality.missing_value_treatment(df, {"a": {"method": "drop_rows"}, "b": {"method": "mean"}})
    assert out.height == 2
    assert out["a"].to_list() == [1.0, 3.0]
    assert out["b"].null_count() == 0


def test_missing_value_treatment_flag_leaves_values_untouched():
    df = pl.DataFrame({"a": [1.0, None, 3.0]})
    out = data_quality.missing_value_treatment(df, {"a": {"method": "flag"}})
    assert out["a"].to_list() == [1.0, None, 3.0]


def test_missing_value_treatment_meta_tags_the_strategy_and_the_indicator_column():
    df = pl.DataFrame({"a": [1.0, None, 3.0]})
    in_meta = {"a": ColumnMeta(dtype="Float64", role=ColumnRole.FEATURE)}
    out = data_quality.missing_value_treatment(df, {"a": {"method": "mean", "add_indicator": True}})
    result = data_quality._missing_value_treatment_meta(
        {"df": in_meta}, {"out": out}, {"strategies": {"a": {"method": "mean", "add_indicator": True}}}
    )
    assert "missing_treatment:mean" in result["out"]["a"].tags
    assert result["out"]["a_was_missing"].role == ColumnRole.FEATURE
