"""Risk measures, convergence diagnostics, and Euler/ES allocation over a
1-D array of per-path losses (see stochastic-engine-proposal.md S6/S7.3).

Deliberately simple: a length-N array of scalar losses is cheap to hold in
memory even at N ~ 1e7 (~80MB of float64) -- the actual memory risk the
proposal calls out is materialising *obligor x path* detail during
generation, not the final per-path total. That part is the simulation
block's job (see blocks/stochastic.py's chunked draw loop); once it hands
this module a plain 1-D array, everything here can just use numpy directly.
"""

from __future__ import annotations

import numpy as np


def risk_measures(losses: np.ndarray, alpha_levels: list[float]) -> dict:
    """VaR(alpha) = the alpha-quantile; TVaR/ES(alpha) = mean of the tail at
    or beyond it. EL/UL are the mean and VaR-minus-mean, the two headline
    numbers behind "EC = VaR(alpha) - EL"."""
    losses = np.sort(np.asarray(losses, dtype=float))
    n = len(losses)
    if n == 0:
        raise ValueError("no losses to compute risk measures from")
    quantiles: dict[float, float] = {}
    tvars: dict[float, float] = {}
    for alpha in alpha_levels:
        idx = max(0, min(n - 1, int(np.ceil(alpha * n)) - 1))
        var = float(losses[idx])
        quantiles[alpha] = var
        tail = losses[idx:]
        tvars[alpha] = float(tail.mean())
    mean = float(losses.mean())
    return {
        "mean": mean,
        "std": float(losses.std(ddof=1)) if n > 1 else 0.0,
        "el": mean,
        "quantiles": quantiles,
        "tvar": tvars,
        "unexpected_loss": {a: quantiles[a] - mean for a in alpha_levels},
    }


def running_estimate(losses_in_arrival_order: np.ndarray, alpha: float, n_checkpoints: int = 8) -> list[dict]:
    """VaR(alpha) recomputed on geometrically-growing prefixes (N/2^k, ...,
    N) of the loss array *in the order paths were generated* -- the
    running-estimate series that answers "how many paths is enough"
    (proposal S6), without a second pass over the full data: each
    checkpoint uses np.partition (O(prefix length)), not a sort."""
    n = len(losses_in_arrival_order)
    sizes = sorted({max(1, n // (2**k)) for k in range(n_checkpoints)} | {n})
    out = []
    for size in sizes:
        prefix = losses_in_arrival_order[:size]
        idx = max(0, min(size - 1, int(np.ceil(alpha * size)) - 1))
        out.append({"n": size, "var_estimate": float(np.partition(prefix, idx)[idx])})
    return out


def quantile_standard_error(losses_sorted: np.ndarray, alpha: float) -> float:
    """Order-statistic-based SE of the quantile estimator: se ~ sqrt(alpha *
    (1-alpha) / n) / f(quantile), with a local density estimate from a small
    window of neighbouring order statistics (a lightweight Maritz-Jarrett-
    style approximation, not the full weighted-order-statistic version)."""
    n = len(losses_sorted)
    if n < 3:
        return 0.0
    idx = max(1, min(n - 2, int(round(alpha * n))))
    window = max(1, int(0.005 * n))
    lo, hi = max(0, idx - window), min(n - 1, idx + window)
    spread = losses_sorted[hi] - losses_sorted[lo]
    if hi == lo or spread <= 0:
        return 0.0
    density = (hi - lo) / n / spread
    return float(np.sqrt(alpha * (1 - alpha) / n) / density)


def bootstrap_quantile_ci(
    losses: np.ndarray, alpha: float, rng: np.random.Generator, n_boot: int = 200, ci: float = 0.9
) -> tuple[float, float]:
    """Bootstrap CI on VaR(alpha) by resampling the loss vector with
    replacement. Uses np.partition (O(n)) rather than a full sort per
    replicate, since only the one order statistic is needed each time --
    matters once n_boot x n gets into the billions of comparisons at
    n ~ 1e6. A local numpy resample for now (see stochastic-engine-proposal
    S6): the "reuse graph fan-out for this" version is a phase-4 follow-up,
    once sub-graph iteration exists."""
    n = len(losses)
    idx = max(0, min(n - 1, int(np.ceil(alpha * n)) - 1))
    estimates = np.empty(n_boot)
    for b in range(n_boot):
        resample = losses[rng.integers(0, n, size=n)]
        estimates[b] = np.partition(resample, idx)[idx]
    lo = float(np.quantile(estimates, (1 - ci) / 2))
    hi = float(np.quantile(estimates, 1 - (1 - ci) / 2))
    return lo, hi


def euler_contributions(component_losses: dict[str, np.ndarray], alpha: float) -> dict[str, float]:
    """ES-based (Euler) risk contribution per component: each component's
    mean loss conditional on the *total* exceeding VaR(alpha) -- far more
    stable than VaR contributions at realistic path counts (proposal
    S3.2/S7.3), and a plain conditional expectation over paths already held
    in memory, no extra simulation needed."""
    names = list(component_losses)
    stacked = np.column_stack([component_losses[name] for name in names])
    total = stacked.sum(axis=1)
    n = len(total)
    idx = max(0, min(n - 1, int(np.ceil(alpha * n)) - 1))
    threshold = np.partition(total, idx)[idx]
    mask = total >= threshold
    if not mask.any():
        mask = total >= total.max()
    return {name: float(stacked[mask, i].mean()) for i, name in enumerate(names)}
