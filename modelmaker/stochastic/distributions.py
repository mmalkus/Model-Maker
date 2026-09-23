"""Distribution fitting, sampling, and a goodness-of-fit battery (see
stochastic-engine-proposal.md S2/S8). A fitted distribution is a plain
JSON-shaped dict, never a pickled scipy object -- same "model" port
convention as blocks/modelling.py -- so it's inspectable over the API's
generic /value endpoint and travels through the normal pickle cache without
any special handling.

Two shapes exist:
  {"kind": "distribution", "family": "lognorm", "params": {...}, ...}
  {"kind": "distribution", "family": "spliced_gpd", "params": {...}, ...}

`sample`/`ppf`/`cdf` dispatch on `family` and work identically for either.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy import stats

CONTINUOUS_FAMILIES: dict[str, Any] = {
    "norm": stats.norm,
    "lognorm": stats.lognorm,
    "gamma": stats.gamma,
    "expon": stats.expon,
    "genpareto": stats.genpareto,
    "t": stats.t,
}


def _param_names(dist: Any) -> list[str]:
    shapes = [s.strip() for s in dist.shapes.split(",")] if dist.shapes else []
    return [*shapes, "loc", "scale"]


def _params_to_args(dist: Any, params: dict[str, float]) -> tuple[float, ...]:
    return tuple(params[name] for name in _param_names(dist))


def _anderson_darling_statistic(x: np.ndarray, cdf: Any) -> float:
    """Generic (family-agnostic) Anderson-Darling statistic -- tail-weighted,
    unlike KS, which is what you actually want for a severity distribution
    (review S3.1(4)). Reported as a raw statistic for candidate comparison
    (lower is better), not converted to a p-value: asymptotic AD critical
    values depend on the family and on how many parameters were estimated
    from the same data, so a single generic table doesn't apply here."""
    n = len(x)
    xs = np.sort(x)
    f = np.clip(cdf(xs), 1e-12, 1 - 1e-12)
    i = np.arange(1, n + 1)
    s = np.sum((2 * i - 1) * (np.log(f) + np.log(1 - f[::-1])))
    return float(-n - s / n)


def fit_distribution(x: np.ndarray, families: list[str] | None = None) -> dict[str, Any]:
    """MLE-fit each candidate family, rank by AIC, and report a full
    candidate-comparison table (AIC/BIC/KS/AD) -- review S3.1(4)'s
    goodness-of-fit battery. Raises if every requested family fails to fit
    (e.g. a family whose support doesn't match the data's sign)."""
    families = list(families or ["norm", "lognorm", "gamma", "expon"])
    x = np.asarray(x, dtype=float)
    n = len(x)
    candidates: list[dict[str, Any]] = []
    for name in families:
        dist = CONTINUOUS_FAMILIES.get(name)
        if dist is None:
            raise ValueError(f"unknown distribution family: {name!r} (have: {sorted(CONTINUOUS_FAMILIES)})")
        try:
            params = dist.fit(x)
            loglik = float(np.sum(dist.logpdf(x, *params)))
            k = len(params)
            ks_stat, ks_p = stats.kstest(x, name, args=params)
            ad_stat = _anderson_darling_statistic(x, lambda v, d=dist, p=params: d.cdf(v, *p))
            candidates.append(
                {
                    "family": name,
                    "params": dict(zip(_param_names(dist), (float(p) for p in params))),
                    "loglik": loglik,
                    "aic": 2 * k - 2 * loglik,
                    "bic": k * np.log(n) - 2 * loglik,
                    "ks_stat": float(ks_stat),
                    "ks_p": float(ks_p),
                    "ad_stat": ad_stat,
                }
            )
        except Exception as e:  # noqa: BLE001 -- a family that can't fit this data is reported, not fatal
            candidates.append({"family": name, "error": f"{type(e).__name__}: {e}"})

    fitted = [c for c in candidates if "error" not in c]
    if not fitted:
        raise ValueError(f"no candidate family could be fit to this data: {candidates}")
    best = min(fitted, key=lambda c: c["aic"])
    return {
        "kind": "distribution",
        "family": best["family"],
        "params": best["params"],
        "fit_method": "mle",
        "fit_stats": {k: best[k] for k in ("loglik", "aic", "bic", "ks_stat", "ks_p", "ad_stat")},
        "candidates": candidates,
    }


def fit_spliced_gpd(x: np.ndarray, threshold_quantile: float = 0.9, body_family: str = "lognorm") -> dict[str, Any]:
    """Spliced body-tail distribution: `body_family` below the threshold,
    a GPD tail above it (review S3.1(3) -- EVT/GPD tail fitting and spliced
    body-tail distributions). `threshold_quantile` is the split point as a
    quantile of the data, e.g. 0.9 fits the GPD to the top decile only."""
    x = np.asarray(x, dtype=float)
    threshold = float(np.quantile(x, threshold_quantile))
    body = x[x <= threshold]
    exceedances = x[x > threshold] - threshold
    if len(exceedances) < 10:
        raise ValueError(
            f"only {len(exceedances)} exceedances above the {threshold_quantile:.0%} threshold -- need >= 10 to "
            "fit a GPD tail; lower threshold_quantile or provide more data"
        )
    body_dist = CONTINUOUS_FAMILIES.get(body_family)
    if body_dist is None:
        raise ValueError(f"unknown body family: {body_family!r}")
    body_params = body_dist.fit(body)
    tail_c, tail_loc, tail_scale = stats.genpareto.fit(exceedances, floc=0.0)
    return {
        "kind": "distribution",
        "family": "spliced_gpd",
        "fit_method": "mle",
        "fit_stats": {"n_exceedances": len(exceedances), "threshold": threshold},
        "params": {
            "threshold": threshold,
            "p_body": threshold_quantile,
            "body_family": body_family,
            "body_params": dict(zip(_param_names(body_dist), (float(p) for p in body_params))),
            "tail_params": {"c": float(tail_c), "loc": float(tail_loc), "scale": float(tail_scale)},
        },
        "candidates": [],
    }


def sample(dist: dict[str, Any], n: int, rng: np.random.Generator) -> np.ndarray:
    if n == 0:
        return np.empty(0)
    if dist["family"] == "spliced_gpd":
        return _sample_spliced(dist, n, rng)
    scipy_dist = CONTINUOUS_FAMILIES[dist["family"]]
    args = _params_to_args(scipy_dist, dist["params"])
    return scipy_dist.ppf(rng.uniform(size=n), *args)


def _sample_spliced(dist: dict[str, Any], n: int, rng: np.random.Generator) -> np.ndarray:
    p = dist["params"]
    body_dist = CONTINUOUS_FAMILIES[p["body_family"]]
    body_args = _params_to_args(body_dist, p["body_params"])
    threshold, p_body = p["threshold"], p["p_body"]
    body_cdf_at_threshold = body_dist.cdf(threshold, *body_args)

    u = rng.uniform(size=n)
    is_body = u < p_body
    out = np.empty(n)
    u_body = (u[is_body] / p_body) * body_cdf_at_threshold
    out[is_body] = body_dist.ppf(u_body, *body_args)
    tail = p["tail_params"]
    u_tail = (u[~is_body] - p_body) / (1 - p_body)
    out[~is_body] = threshold + stats.genpareto.ppf(u_tail, tail["c"], loc=tail["loc"], scale=tail["scale"])
    return out


def ppf(dist: dict[str, Any], q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=float)
    if dist["family"] == "spliced_gpd":
        p = dist["params"]
        body_dist = CONTINUOUS_FAMILIES[p["body_family"]]
        body_args = _params_to_args(body_dist, p["body_params"])
        threshold, p_body = p["threshold"], p["p_body"]
        body_cdf_at_threshold = body_dist.cdf(threshold, *body_args)
        out = np.empty_like(q)
        in_body = q < p_body
        out[in_body] = body_dist.ppf((q[in_body] / p_body) * body_cdf_at_threshold, *body_args)
        tail = p["tail_params"]
        out[~in_body] = threshold + stats.genpareto.ppf((q[~in_body] - p_body) / (1 - p_body), tail["c"], loc=tail["loc"], scale=tail["scale"])
        return out
    scipy_dist = CONTINUOUS_FAMILIES[dist["family"]]
    return scipy_dist.ppf(q, *_params_to_args(scipy_dist, dist["params"]))


def cdf(dist: dict[str, Any], x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if dist["family"] == "spliced_gpd":
        p = dist["params"]
        body_dist = CONTINUOUS_FAMILIES[p["body_family"]]
        body_args = _params_to_args(body_dist, p["body_params"])
        threshold, p_body = p["threshold"], p["p_body"]
        body_cdf_at_threshold = body_dist.cdf(threshold, *body_args)
        out = np.empty_like(x)
        in_body = x <= threshold
        out[in_body] = (body_dist.cdf(x[in_body], *body_args) / body_cdf_at_threshold) * p_body
        tail = p["tail_params"]
        out[~in_body] = p_body + (1 - p_body) * stats.genpareto.cdf(x[~in_body] - threshold, tail["c"], loc=tail["loc"], scale=tail["scale"])
        return out
    scipy_dist = CONTINUOUS_FAMILIES[dist["family"]]
    return scipy_dist.cdf(x, *_params_to_args(scipy_dist, dist["params"]))
