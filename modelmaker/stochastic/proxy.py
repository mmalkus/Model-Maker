"""Proxy functions -- the scenario-valuation escape from nested simulation
(see stochastic-engine-proposal.md S8): fit a cheap valuation surrogate at a
small number of fitting scenarios, then evaluate it many times. Every
method shares one contract:

    fit_*(fitting_data, ...) -> proxy      # a plain dict, {"kind": "proxy_function", ...}
    evaluate_proxy(proxy, scenarios) -> np.ndarray

so a block can swap `method="closed_form"` for `method="polynomial"`
without anything downstream (evaluate_proxy, validate_proxy, var_covar's
bump-and-reval) needing to know which one fitted it -- the point of the
abstraction per review S3.6.

v1 ships closed_form and polynomial (the proposal's recommended entry
point); LSMC and replicating portfolios are increments on the same
`payload`/`method` shape, not a different interface.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import polars as pl


def fit_closed_form(expr: str, risk_factors: list[str]) -> dict[str, Any]:
    """No fitting: `expr` is evaluated directly per scenario (Polars SQL
    expression syntax, same convention as blocks/library.py's filter block).
    The right default wherever an analytic valuation function exists --
    review S3.6 case 1, which is all IFRS 9 scenario valuation needs."""
    return {
        "kind": "proxy_function",
        "method": "closed_form",
        "risk_factors": list(risk_factors),
        "payload": {"expr": expr},
        "fit_diagnostics": {},
    }


def fit_polynomial(
    fitting: pl.DataFrame,
    risk_factors: list[str],
    value_col: str,
    degree: int = 2,
    regressor: str = "ridge",
    alpha: float = 1.0,
) -> dict[str, Any]:
    """Curve fit at a small, deliberately-chosen set of fitting scenarios
    (review S3.6 case 2): polynomial basis expansion + Ridge/Lasso/plain
    least squares over sklearn (already a dependency) -- no new numerical
    dependency for the entry-level version."""
    from sklearn.linear_model import Lasso, LinearRegression, Ridge
    from sklearn.preprocessing import PolynomialFeatures

    if fitting.height <= len(risk_factors):
        raise ValueError(
            f"only {fitting.height} fitting scenario(s) for {len(risk_factors)} risk factor(s) -- "
            "need more fitting scenarios than risk factors to fit a stable curve"
        )
    x = fitting.select(risk_factors).to_numpy()
    y = fitting[value_col].to_numpy()
    poly = PolynomialFeatures(degree=degree, include_bias=False)
    x_poly = poly.fit_transform(x)

    model: Any
    if regressor == "ridge":
        model = Ridge(alpha=alpha)
    elif regressor == "lasso":
        model = Lasso(alpha=alpha, max_iter=5000)
    elif regressor == "ols":
        model = LinearRegression()
    else:
        raise ValueError(f"unknown regressor: {regressor!r} (use ridge, lasso, or ols)")
    model.fit(x_poly, y)

    fitted = model.predict(x_poly)
    ss_res = float(np.sum((y - fitted) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0

    return {
        "kind": "proxy_function",
        "method": "polynomial",
        "risk_factors": list(risk_factors),
        "payload": {
            "degree": degree,
            "regressor": regressor,
            "alpha": alpha,
            "powers": poly.powers_.tolist(),
            "coefficients": [float(c) for c in model.coef_],
            "intercept": float(model.intercept_),
        },
        "fit_diagnostics": {"in_sample_r2": r2, "n_fitting_scenarios": fitting.height},
    }


def evaluate_proxy(proxy: dict[str, Any], scenarios: pl.DataFrame) -> np.ndarray:
    method = proxy["method"]
    if method == "closed_form":
        expr = proxy["payload"]["expr"]
        return scenarios.select(pl.sql_expr(expr).alias("__value__"))["__value__"].to_numpy().astype(float)
    if method == "polynomial":
        return _evaluate_polynomial(proxy, scenarios)
    raise ValueError(f"unknown proxy method: {method!r} (have: closed_form, polynomial)")


def _evaluate_polynomial(proxy: dict[str, Any], scenarios: pl.DataFrame) -> np.ndarray:
    payload = proxy["payload"]
    x = scenarios.select(proxy["risk_factors"]).to_numpy()
    powers = np.array(payload["powers"])
    if powers.shape[0] == 0:
        x_poly = np.zeros((x.shape[0], 0))
    else:
        x_poly = np.column_stack([np.prod(x ** powers[j], axis=1) for j in range(powers.shape[0])])
    coefficients = np.array(payload["coefficients"])
    return x_poly @ coefficients + payload["intercept"]


def validate_proxy(proxy: dict[str, Any], actual: np.ndarray, predicted: np.ndarray, n_worst: int = 10) -> dict[str, Any]:
    """Proxy-vs-truth diagnostics (review S3.6's "mandatory, and a real
    product opportunity" validation pack): overall error, and the worst
    individual scenarios by absolute error -- the region that matters, not
    just an average that a validator will ask to see broken down."""
    error = predicted - actual
    ss_res = float(np.sum(error**2))
    ss_tot = float(np.sum((actual - actual.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0
    abs_error = np.abs(error)
    worst = np.argsort(-abs_error)[: min(n_worst, len(abs_error))]
    return {
        "out_of_sample_r2": r2,
        "mae": float(abs_error.mean()),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "max_abs_error": float(abs_error.max()) if len(abs_error) else 0.0,
        "worst_indices": [int(i) for i in worst],
    }
