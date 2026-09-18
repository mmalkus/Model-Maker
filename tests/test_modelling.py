import polars as pl
import pytest

from modelmaker.blocks import modelling


def test_predict_applies_glm_gaussian_coefficients_directly():
    df = pl.DataFrame({"x": [1.0, 2.0, 3.0]})
    model = {"kind": "glm", "family": "gaussian", "features": ["x"], "coefficients": {"x": 2.0}, "intercept": 1.0}

    out = modelling.predict(df, model)

    assert out["predicted"].to_list() == pytest.approx([3.0, 5.0, 7.0])


def test_predict_applies_glm_log_link_for_non_gaussian_families():
    import math

    df = pl.DataFrame({"x": [0.0, 1.0]})
    model = {"kind": "glm", "family": "poisson", "features": ["x"], "coefficients": {"x": 1.0}, "intercept": 0.0}

    out = modelling.predict(df, model)

    assert out["predicted"].to_list() == pytest.approx([math.exp(0.0), math.exp(1.0)])


def test_predict_applies_logistic_regression_sigmoid():
    df = pl.DataFrame({"x": [-100.0, 100.0]})
    model = {"kind": "logistic_regression", "features": ["x"], "coefficients": {"x": 1.0}, "intercept": 0.0}

    out = modelling.predict(df, model)

    assert out["predicted_proba"].to_list() == pytest.approx([0.0, 1.0], abs=1e-6)
    assert out["predicted_class"].to_list() == [0, 1]


def test_predict_fits_and_applies_a_model_end_to_end():
    train = pl.DataFrame({"x": [1.0, 2.0, 3.0, 4.0], "y": [0.0, 0.0, 1.0, 1.0]})
    _, model = modelling.logistic_regression(train, target="y", features=["x"])

    test = pl.DataFrame({"x": [1.5, 3.5]})
    out = modelling.predict(test, model)

    assert set(out.columns) >= {"x", "predicted_proba", "predicted_class"}
    # a model that separates the training data cleanly should also rank the
    # held-out points the same way: the higher x should score higher risk
    proba = out["predicted_proba"].to_list()
    assert proba[1] > proba[0]


def test_predict_raises_on_missing_feature_columns():
    df = pl.DataFrame({"a": [1.0]})
    model = {"kind": "glm", "family": "gaussian", "features": ["x"], "coefficients": {"x": 1.0}, "intercept": 0.0}

    with pytest.raises(ValueError, match="not present"):
        modelling.predict(df, model)


def test_predict_raises_on_unsupported_model_kind():
    df = pl.DataFrame({"x": [1.0]})
    model = {"kind": "some_other_model", "features": ["x"], "coefficients": {"x": 1.0}, "intercept": 0.0}

    with pytest.raises(ValueError, match="unsupported model kind"):
        modelling.predict(df, model)


def test_scorecard_scale_matches_the_pdo_formula_directly():
    import math

    model = {"kind": "logistic_regression", "features": ["x"], "coefficients": {"x": -0.5}, "intercept": 0.3}
    result = modelling.scorecard_scale(model, base_score=600.0, base_odds=50.0, pdo=20.0)

    factor = 20.0 / math.log(2)
    offset = 600.0 - factor * math.log(50.0)
    assert result["factor"] == pytest.approx(factor)
    assert result["offset"] == pytest.approx(offset)
    assert result["intercept_points"] == pytest.approx(offset - factor * 0.3)
    assert result["rows"] == [{"feature": "x", "coefficient": -0.5, "points_per_unit": pytest.approx(0.5 * factor)}]


def test_scorecard_scale_raises_on_a_non_logistic_model():
    model = {"kind": "glm", "family": "gaussian", "features": ["x"], "coefficients": {"x": 1.0}, "intercept": 0.0}

    with pytest.raises(ValueError, match="logistic_regression"):
        modelling.scorecard_scale(model)


def test_lgd_regression_fits_a_bounded_target_and_predictions_stay_in_0_1():
    df = pl.DataFrame({"x": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0], "lgd": [0.0, 0.05, 0.3, 0.6, 0.9, 1.0]})

    predictions, model = modelling.lgd_regression(df, target="lgd", features=["x"])

    assert model["kind"] == "lgd_regression"
    assert model["converged"] is True
    assert set(predictions.columns) >= {"x", "lgd", "predicted"}
    preds = predictions["predicted"].to_list()
    assert all(0.0 <= p <= 1.0 for p in preds)
    # higher x -> higher loss, matching the fitted data's trend
    assert preds[-1] > preds[0]


def test_lgd_regression_raises_when_target_is_outside_0_1():
    df = pl.DataFrame({"x": [1.0, 2.0], "lgd": [0.5, 1.5]})

    with pytest.raises(ValueError, match=r"within \[0, 1\]"):
        modelling.lgd_regression(df, target="lgd", features=["x"])


def test_predict_applies_lgd_regression_sigmoid():
    df = pl.DataFrame({"x": [-100.0, 100.0]})
    model = {"kind": "lgd_regression", "features": ["x"], "coefficients": {"x": 1.0}, "intercept": 0.0}

    out = modelling.predict(df, model)

    assert out["predicted"].to_list() == pytest.approx([0.0, 1.0], abs=1e-6)


def test_compute_lgd_matches_the_1_minus_recovery_rate_formula():
    df = pl.DataFrame({"ead": [100.0, 200.0], "recovered": [80.0, 50.0], "cost": [5.0, 10.0]})

    out = modelling.compute_lgd(df, ead_col="ead", recovered_col="recovered", cost_col="cost")

    assert out["lgd"].to_list() == pytest.approx([0.25, 0.8])


def test_compute_lgd_defaults_cost_to_zero_when_omitted():
    df = pl.DataFrame({"ead": [100.0], "recovered": [40.0]})

    out = modelling.compute_lgd(df, ead_col="ead", recovered_col="recovered")

    assert out["lgd"].to_list() == pytest.approx([0.6])


def test_compute_lgd_nulls_out_non_positive_ead_instead_of_dividing_by_zero():
    df = pl.DataFrame({"ead": [0.0, -5.0], "recovered": [0.0, 0.0]})

    out = modelling.compute_lgd(df, ead_col="ead", recovered_col="recovered")

    assert out["lgd"].to_list() == [None, None]


def test_compute_lgd_truncates_to_floor_and_cap():
    df = pl.DataFrame({"ead": [100.0, 100.0], "recovered": [150.0, -50.0]})

    out = modelling.compute_lgd(df, ead_col="ead", recovered_col="recovered", floor=0.0, cap=1.0)

    assert out["lgd"].to_list() == pytest.approx([0.0, 1.0])


def test_compute_ccf_matches_the_undrawn_drawdown_formula():
    df = pl.DataFrame({"limit": [1000.0], "balance_ref": [400.0], "balance_default": [700.0]})

    out = modelling.compute_ccf(df, limit_col="limit", balance_ref_col="balance_ref", balance_default_col="balance_default")

    # undrawn = 1000 - 400 = 600; ccf = (700 - 400) / 600 = 0.5
    assert out["undrawn_at_reference"].to_list() == pytest.approx([600.0])
    assert out["ccf"].to_list() == pytest.approx([0.5])


def test_compute_ccf_zeroes_out_when_already_at_or_over_limit():
    df = pl.DataFrame({"limit": [500.0], "balance_ref": [500.0], "balance_default": [500.0]})

    out = modelling.compute_ccf(df, limit_col="limit", balance_ref_col="balance_ref", balance_default_col="balance_default")

    assert out["undrawn_at_reference"].to_list() == pytest.approx([0.0])
    assert out["ccf"].to_list() == pytest.approx([0.0])


def test_compute_ccf_truncates_to_floor_and_cap():
    df = pl.DataFrame({"limit": [1000.0, 1000.0], "balance_ref": [400.0, 900.0], "balance_default": [100.0, 2000.0]})

    out = modelling.compute_ccf(df, limit_col="limit", balance_ref_col="balance_ref", balance_default_col="balance_default", floor=0.0, cap=1.0)

    assert out["ccf"].to_list() == pytest.approx([0.0, 1.0])


def test_compute_ccf_skips_truncation_when_floor_and_cap_are_none():
    df = pl.DataFrame({"limit": [1000.0], "balance_ref": [400.0], "balance_default": [100.0]})

    out = modelling.compute_ccf(df, limit_col="limit", balance_ref_col="balance_ref", balance_default_col="balance_default", floor=None, cap=None)

    # (100 - 400) / 600 = -0.5, left untouched
    assert out["ccf"].to_list() == pytest.approx([-0.5])
