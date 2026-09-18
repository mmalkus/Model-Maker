import polars as pl
import pytest

from modelmaker.blocks import stat_tests


def test_continuous_accuracy_computes_mae_mse_rmse_r2():
    df = pl.DataFrame({"actual": [0.0, 0.5, 1.0], "predicted": [0.1, 0.4, 0.9]})

    result = stat_tests.continuous_accuracy(df, actual_col="actual", predicted_col="predicted")

    assert result["kind"] == "continuous_accuracy"
    assert result["mae"] == pytest.approx(0.1)
    assert result["mse"] == pytest.approx(0.01)
    assert result["rmse"] == pytest.approx(0.1)
    assert result["r2"] is not None


def test_continuous_accuracy_returns_none_r2_when_actual_has_no_variance():
    df = pl.DataFrame({"actual": [0.5, 0.5, 0.5], "predicted": [0.4, 0.5, 0.6]})

    result = stat_tests.continuous_accuracy(df, actual_col="actual", predicted_col="predicted")

    assert result["r2"] is None


def test_bucketed_calibration_reports_observed_and_predicted_means_per_bucket():
    df = pl.DataFrame(
        {
            "actual": [0.0, 0.1, 0.4, 0.5, 0.9, 1.0],
            "predicted": [0.05, 0.15, 0.35, 0.45, 0.85, 0.95],
        }
    )

    result = stat_tests.bucketed_calibration(df, actual_col="actual", predicted_col="predicted", bins=3)

    assert result["kind"] == "bucketed_calibration"
    assert len(result["buckets"]) == 3
    total_n = sum(b["n"] for b in result["buckets"])
    assert total_n == 6
    # buckets are sorted by increasing predicted mean
    predicted_means = [b["predicted_mean"] for b in result["buckets"]]
    assert predicted_means == sorted(predicted_means)


def test_bucketed_calibration_flags_a_miscalibrated_bucket():
    df = pl.DataFrame({"actual": [0.0, 0.0, 1.0, 1.0], "predicted": [0.1, 0.1, 0.1, 0.1]})

    result = stat_tests.bucketed_calibration(df, actual_col="actual", predicted_col="predicted", bins=1)

    bucket = result["buckets"][0]
    assert bucket["observed_mean"] == pytest.approx(0.5)
    assert bucket["predicted_mean"] == pytest.approx(0.1)
