"""Dependency layer: correlation/PSD repair, Cholesky, and hand-rolled
Gaussian/t/Clayton/Gumbel copulas (see stochastic-engine-proposal.md S7.2,
open decision 2 -- hand-rolled over numpy/scipy rather than a third-party
copula package). Shared by the simulation-based aggregation path
(blocks.stochastic.aggregate_simulation) and the var-covar path's
delta_gamma_copula method (stochastic.var_covar).

A `dependency` object is a plain dict:
    {"kind": "dependency", "type": "gaussian" | "t" | "clayton" | "gumbel",
     "corr": [[...]], "stds": {...}, "labels": [...], "dof": ..., "theta": ...}
"""

from __future__ import annotations

import numpy as np
from scipy import stats


def nearest_psd_correlation(corr: np.ndarray, min_eig: float = 1e-8) -> np.ndarray:
    """Real historical correlation matrices built from mismatched windows
    routinely aren't PSD. Simple eigenvalue-clipping projection (not the
    full iterative Higham algorithm): symmetrize, clip negative/near-zero
    eigenvalues, rebuild, and rescale back to a correlation matrix (unit
    diagonal). Good enough to make Cholesky/copula sampling well-defined
    without silently producing garbage; a tighter nearest-correlation
    projection is a follow-up if this proves too coarse in practice."""
    sym = (corr + corr.T) / 2
    eigvals, eigvecs = np.linalg.eigh(sym)
    clipped = np.clip(eigvals, min_eig, None)
    repaired = eigvecs @ np.diag(clipped) @ eigvecs.T
    d = np.sqrt(np.diag(repaired))
    repaired = repaired / np.outer(d, d)
    np.fill_diagonal(repaired, 1.0)
    return (repaired + repaired.T) / 2


def cholesky_factor(corr: np.ndarray) -> np.ndarray:
    """Cholesky of `corr`, repairing to the nearest PSD correlation matrix
    first if it isn't already (see nearest_psd_correlation)."""
    try:
        return np.linalg.cholesky(corr)
    except np.linalg.LinAlgError:
        return np.linalg.cholesky(nearest_psd_correlation(corr))


def sample_correlated_normal(corr: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    """Jointly normal (not uniform) correlated draws -- the Cholesky
    ingredient shared by sample_gaussian_copula below and var_covar's
    delta-gamma Monte Carlo, exposed directly for a Merton-style
    multi-factor credit model (stochastic.credit), where the systematic
    factors themselves are assumed jointly normal, not merely
    rank-correlated via a copula."""
    return rng.standard_normal((n, corr.shape[0])) @ cholesky_factor(corr).T


def sample_gaussian_copula(corr: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    return stats.norm.cdf(sample_correlated_normal(corr, n, rng))


def sample_t_copula(corr: np.ndarray, dof: float, n: int, rng: np.random.Generator) -> np.ndarray:
    k = corr.shape[0]
    z = rng.standard_normal((n, k)) @ cholesky_factor(corr).T
    chi2 = rng.chisquare(dof, size=n)
    t = z / np.sqrt(chi2 / dof)[:, None]
    return stats.t.cdf(t, dof)


def _sample_positive_stable(alpha: float, n: int, rng: np.random.Generator) -> np.ndarray:
    """Chambers-Mallows-Stuck sampler for a totally-skewed positive
    alpha-stable frailty (beta=1) -- the Marshall-Olkin ingredient shared by
    Clayton and Gumbel below (Hofert 2011, "Sampling Archimedean copulas")."""
    u = rng.uniform(-np.pi / 2, np.pi / 2, size=n)
    w = rng.exponential(size=n)
    a = np.sin(alpha * (u + np.pi / 2)) / (np.cos(u) ** (1 / alpha))
    b = (np.cos(u - alpha * (u + np.pi / 2)) / w) ** ((1 - alpha) / alpha)
    return a * b


def sample_clayton_copula(theta: float, k: int, n: int, rng: np.random.Generator) -> np.ndarray:
    """Exchangeable Clayton copula via the Marshall-Olkin gamma-frailty
    construction: V ~ Gamma(1/theta, 1), U_i = (1 + E_i/V)^(-1/theta) with
    iid E_i ~ Exp(1). theta > 0 (higher = stronger lower-tail dependence)."""
    if theta <= 0:
        raise ValueError("Clayton copula requires theta > 0")
    v = rng.gamma(shape=1.0 / theta, scale=1.0, size=n)
    e = rng.exponential(size=(n, k))
    return (1 + e / v[:, None]) ** (-1.0 / theta)


def sample_gumbel_copula(theta: float, k: int, n: int, rng: np.random.Generator) -> np.ndarray:
    """Exchangeable Gumbel copula via Marshall-Olkin with a positive
    alpha-stable frailty (alpha = 1/theta). theta >= 1 (higher = stronger
    upper-tail dependence; theta=1 is independence)."""
    if theta < 1:
        raise ValueError("Gumbel copula requires theta >= 1")
    alpha = 1.0 / theta
    v = _sample_positive_stable(alpha, n, rng)
    e = rng.exponential(size=(n, k))
    return np.exp(-((e / v[:, None]) ** alpha))


def sample_uniforms(dependency: dict, n: int, rng: np.random.Generator) -> np.ndarray:
    """Dispatch on dependency["type"] -> (n, k) array of Uniform(0,1) draws
    with the declared dependence structure. `corr` drives Gaussian/t;
    `theta` drives Clayton/Gumbel (dimension taken from `corr`'s shape so
    the same dependency object always carries k, even for the Archimedean
    families that don't otherwise use `corr` beyond its size)."""
    corr = np.array(dependency["corr"], dtype=float)
    kind = dependency["type"]
    if kind == "gaussian":
        return sample_gaussian_copula(corr, n, rng)
    if kind == "t":
        return sample_t_copula(corr, dependency.get("dof", 5.0), n, rng)
    if kind == "clayton":
        return sample_clayton_copula(dependency.get("theta", 2.0), corr.shape[0], n, rng)
    if kind == "gumbel":
        return sample_gumbel_copula(dependency.get("theta", 2.0), corr.shape[0], n, rng)
    raise ValueError(f"unknown dependency type: {kind!r} (have: gaussian, t, clayton, gumbel)")
