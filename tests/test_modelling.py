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
