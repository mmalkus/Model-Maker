"""Closed-form single-factor ASRF credit portfolio capital (Vasicek/Basel
II IRB formula) -- stochastic-engine-proposal.md S3.2. Deterministic, no
simulation: the point of it, per the proposal, is to be a *benchmark* a
future multi-factor obligor-level Monte Carlo simulation can be validated
against (not built here -- see the proposal's Implementation status), and
a real, standalone economic-capital number in its own right for a
single-factor (conditional-independence) portfolio.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import norm


def basel_corporate_correlation(pd: np.ndarray) -> np.ndarray:
    """Basel II IRB regulatory asset correlation for corporate/sovereign/
    bank exposures: R(PD) interpolates from 0.24 (PD near 0) to 0.12 (PD
    near 1) via an exponential weight -- the regulatory assumption that
    lower-PD obligors are more correlated with the systematic factor
    (idiosyncratic risk dominates less for safer borrowers)."""
    pd = np.asarray(pd, dtype=float)
    w = (1 - np.exp(-50 * pd)) / (1 - np.exp(-50))
    return 0.12 * w + 0.24 * (1 - w)


def asrf_capital_rate(pd: np.ndarray, correlation: np.ndarray, confidence: float) -> np.ndarray:
    """Unexpected-loss capital rate per unit EAD, *before* multiplying by
    LGD (the caller does that, since LGD can vary independently of PD/R):
    the conditional PD at the target confidence level under the
    single-factor Vasicek model, minus the unconditional PD. Always >= 0,
    and exactly 0 when correlation is 0 (conditional PD collapses to the
    unconditional PD when there's no systematic factor to condition on)."""
    pd = np.asarray(pd, dtype=float)
    correlation = np.asarray(correlation, dtype=float)
    conditional_pd = norm.cdf((norm.ppf(pd) + np.sqrt(correlation) * norm.ppf(confidence)) / np.sqrt(1 - correlation))
    return conditional_pd - pd


def asrf_economic_capital(
    pd: np.ndarray, lgd: np.ndarray, ead: np.ndarray, correlation: np.ndarray, confidence: float
) -> tuple[dict, np.ndarray, np.ndarray]:
    """Portfolio-level EC/EL under conditional independence: sum obligor-
    level unexpected/expected loss directly (no simulation -- this is
    exactly what "closed form" buys you here). Returns the portfolio
    summary dict plus the two per-obligor arrays, so the caller (see
    blocks/stochastic.py) can attach them to the output table."""
    pd = np.asarray(pd, dtype=float)
    lgd = np.asarray(lgd, dtype=float)
    ead = np.asarray(ead, dtype=float)
    capital_rate = asrf_capital_rate(pd, correlation, confidence)
    ec_per_obligor = capital_rate * lgd * ead
    el_per_obligor = pd * lgd * ead
    total_ead = float(ead.sum())
    summary = {
        "kind": "simulation_result",
        "confidence": confidence,
        "n_obligors": int(len(pd)),
        "total_ead": total_ead,
        "expected_loss": float(el_per_obligor.sum()),
        "economic_capital": float(ec_per_obligor.sum()),
        "capital_requirement": float(ec_per_obligor.sum() + el_per_obligor.sum()),
        "weighted_avg_pd": float(np.average(pd, weights=ead)) if total_ead > 0 else float(pd.mean()),
        "weighted_avg_correlation": float(np.average(correlation, weights=ead)) if total_ead > 0 else float(np.mean(correlation)),
    }
    return summary, ec_per_obligor, el_per_obligor
