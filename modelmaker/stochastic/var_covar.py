"""Var-covar sensitivity/aggregation -- review S3.6 method 5, and the
concrete link between "curve fitting" and "var-covar" the proposal draws in
S7.1: var-covar *is* a curve fit, just one truncated to a quadratic Taylor
expansion (delta/gamma) instead of a general basis. Sensitivities are
computed by bumping-and-revaluing a `ProxyFunctionPacket` (proxy.py) --
whatever fitted it (closed-form or a polynomial curve fit) -- so this module
never needs its own valuation logic.

Four aggregation methods, sharing one entry point (`aggregate`):
  - normal              closed-form: portfolio std from delta'*Cov*delta,
                        quantile = std * z_alpha. Cheap, transparent,
                        the right default when it's defensible.
  - cornish_fisher       skew + kurtosis Taylor correction to the normal
                        quantile, estimated from a cheap Monte Carlo pass
                        over the *own* quadratic (delta-gamma) expansion --
                        deliberately not derived analytically from cumulant
                        algebra (fragile to get right for a full Hessian
                        with cross-gamma terms); simulating the expansion's
                        own distribution is simple, correct, and standard
                        practice ("delta-gamma Monte Carlo").
  - moment_matching      the same skew estimate, but only the skew term of
                        the Cornish-Fisher series (no kurtosis term) -- a
                        lighter-weight correction, distinct from the full
                        cornish_fisher method above.
  - delta_gamma_copula   full empirical quantile of the MC sample, with the
                        correlated normal draws replaced by copula draws
                        (dependency.sample_uniforms) -- captures tail
                        dependence beyond Gaussian.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import polars as pl
from scipy import stats

from . import dependency as dependency_mod
from . import proxy as proxy_mod


def bump_and_reval(
    proxy: dict[str, Any],
    base: dict[str, float],
    risk_factors: list[str],
    bump_sizes: dict[str, float] | float = 0.01,
) -> dict[str, Any]:
    """Delta, gamma, and cross-gamma of `proxy` around `base` by central
    finite differences -- cheap because evaluating a fitted proxy is cheap
    (that's the point of fitting one), so a handful of extra evaluations per
    risk factor (and per pair, for cross-gamma) costs nothing next to
    whatever building the proxy itself cost."""
    if isinstance(bump_sizes, (int, float)):
        bump_sizes = {f: float(bump_sizes) for f in risk_factors}

    def value_at(overrides: dict[str, float]) -> float:
        row = dict(base)
        row.update(overrides)
        return float(proxy_mod.evaluate_proxy(proxy, pl.DataFrame([row]))[0])

    base_value = value_at({})
    delta: dict[str, float] = {}
    gamma: dict[str, float] = {}
    for f in risk_factors:
        h = bump_sizes[f]
        v_up = value_at({f: base[f] + h})
        v_down = value_at({f: base[f] - h})
        delta[f] = (v_up - v_down) / (2 * h)
        gamma[f] = (v_up - 2 * base_value + v_down) / (h**2)

    cross_gamma: dict[str, float] = {}
    for i, fi in enumerate(risk_factors):
        for fj in risk_factors[i + 1 :]:
            hi, hj = bump_sizes[fi], bump_sizes[fj]
            v_pp = value_at({fi: base[fi] + hi, fj: base[fj] + hj})
            v_pm = value_at({fi: base[fi] + hi, fj: base[fj] - hj})
            v_mp = value_at({fi: base[fi] - hi, fj: base[fj] + hj})
            v_mm = value_at({fi: base[fi] - hi, fj: base[fj] - hj})
            cross_gamma[f"{fi}|{fj}"] = (v_pp - v_pm - v_mp + v_mm) / (4 * hi * hj)

    return {"base_value": base_value, "delta": delta, "gamma": gamma, "cross_gamma": cross_gamma}


def _hessian_matrix(sensitivities: dict[str, Any], factor_order: list[str]) -> np.ndarray:
    k = len(factor_order)
    index = {f: i for i, f in enumerate(factor_order)}
    h = np.zeros((k, k))
    for f, g in sensitivities["gamma"].items():
        h[index[f], index[f]] = g
    for pair, g in sensitivities.get("cross_gamma", {}).items():
        fi, fj = pair.split("|")
        h[index[fi], index[fj]] = g
        h[index[fj], index[fi]] = g
    return h


def _cornish_fisher_quantile(z: float, skew: float, excess_kurtosis: float) -> float:
    return z + (z**2 - 1) * skew / 6 + (z**3 - 3 * z) * excess_kurtosis / 24 - (2 * z**3 - 5 * z) * skew**2 / 36


def _skew_only_quantile(z: float, skew: float) -> float:
    return z + (z**2 - 1) * skew / 6


def _euler_contributions_linear(delta_vec: np.ndarray, cov: np.ndarray, factor_order: list[str], portfolio_std: float) -> dict[str, float]:
    if portfolio_std <= 0:
        return {f: 0.0 for f in factor_order}
    marginal = cov @ delta_vec
    return {f: float(marginal[i] * delta_vec[i] / portfolio_std) for i, f in enumerate(factor_order)}


def aggregate(
    sensitivities: dict[str, Any],
    stds: dict[str, float],
    corr: np.ndarray,
    factor_order: list[str],
    alpha_levels: list[float],
    method: str = "normal",
    n_mc: int = 50_000,
    seed: int = 0,
    copula: dict[str, Any] | None = None,
) -> dict[str, Any]:
    missing_delta = [f for f in factor_order if f not in sensitivities["delta"]]
    if missing_delta:
        raise ValueError(f"sensitivities have no delta for factor(s) {missing_delta} -- check risk_factors match dependency labels")
    delta_vec = np.array([sensitivities["delta"][f] for f in factor_order])
    std_vec = np.array([stds[f] for f in factor_order])
    cov = corr * np.outer(std_vec, std_vec)
    portfolio_std = float(np.sqrt(max(delta_vec @ cov @ delta_vec, 0.0)))
    base_value = sensitivities["base_value"]

    if method == "normal":
        quantiles = {a: portfolio_std * float(stats.norm.ppf(a)) for a in alpha_levels}
        return {
            "method": method,
            "base_value": base_value,
            "portfolio_std": portfolio_std,
            "quantiles": quantiles,
            "euler_contributions": _euler_contributions_linear(delta_vec, cov, factor_order, portfolio_std),
        }

    if method not in ("cornish_fisher", "moment_matching", "delta_gamma_copula"):
        raise ValueError(f"unknown var-covar aggregation method: {method!r}")

    rng = np.random.default_rng(seed)
    hessian = _hessian_matrix(sensitivities, factor_order)
    if method == "delta_gamma_copula":
        dep = copula or {"type": "gaussian", "corr": corr.tolist()}
        u = dependency_mod.sample_uniforms(dep, n_mc, rng)
        x = stats.norm.ppf(np.clip(u, 1e-12, 1 - 1e-12)) * std_vec
    else:
        chol = dependency_mod.cholesky_factor(corr)
        x = (rng.standard_normal((n_mc, len(factor_order))) @ chol.T) * std_vec

    y = x @ delta_vec + 0.5 * np.einsum("ij,jk,ik->i", x, hessian, x)
    skew = float(stats.skew(y))
    excess_kurtosis = float(stats.kurtosis(y))

    quantiles = {}
    if method == "delta_gamma_copula":
        y_sorted = np.sort(y)
        n = len(y_sorted)
        for a in alpha_levels:
            idx = max(0, min(n - 1, int(np.ceil(a * n)) - 1))
            quantiles[a] = float(y_sorted[idx])
    else:
        for a in alpha_levels:
            z = float(stats.norm.ppf(a))
            adjusted_z = _cornish_fisher_quantile(z, skew, excess_kurtosis) if method == "cornish_fisher" else _skew_only_quantile(z, skew)
            quantiles[a] = portfolio_std * adjusted_z

    # Euler contributions: conditional mean of each factor's own quadratic
    # contribution (delta_i*x_i + 0.5*gamma_ii*x_i^2) given the aggregate
    # exceeds VaR(max(alpha_levels)) -- cross-gamma terms are split evenly
    # between the two factors involved, a common allocation convention.
    contributions = _mc_euler_contributions(x, hessian, delta_vec, y, factor_order, max(alpha_levels))

    return {
        "method": method,
        "base_value": base_value,
        "portfolio_std": portfolio_std,
        "skewness": skew,
        "excess_kurtosis": excess_kurtosis,
        "quantiles": quantiles,
        "euler_contributions": contributions,
    }


def _mc_euler_contributions(
    x: np.ndarray, hessian: np.ndarray, delta_vec: np.ndarray, y: np.ndarray, factor_order: list[str], alpha: float
) -> dict[str, float]:
    n = len(y)
    idx = max(0, min(n - 1, int(np.ceil(alpha * n)) - 1))
    threshold = np.partition(y, idx)[idx]
    mask = y >= threshold
    if not mask.any():
        mask = y >= y.max()
    k = len(factor_order)
    per_factor = x[mask] * delta_vec + 0.5 * (x[mask] ** 2) * np.diag(hessian)
    for i in range(k):
        for j in range(i + 1, k):
            cross_term = x[mask, i] * x[mask, j] * hessian[i, j]
            per_factor[:, i] += cross_term / 2
            per_factor[:, j] += cross_term / 2
    return {f: float(per_factor[:, i].mean()) for i, f in enumerate(factor_order)}
