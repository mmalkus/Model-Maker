"""Model-validation test blocks: KS statistic, AUC/Gini, and Population
Stability Index (PSI) -- the standard discrimination and stability checks
for a PD-style model. Each produces a scalar_metric artifact (a plain
dict), not a dataframe, so it terminates a branch of the pipeline the way
generate_image/display_table do. Extra libraries (scipy, sklearn, numpy)
are imported locally inside each fn -- see generate_image in library.py for
why: the function body is inlined verbatim into the compiled script, which
only guarantees `polars as pl` at module level.
"""

from __future__ import annotations

import polars as pl

from .base import BlockSpec, PortSpec, register_block


def ks_test(df: pl.DataFrame, score_col: str, target_col: str) -> dict:
    from scipy.stats import ks_2samp

    scores = df[score_col].to_numpy()
    target = df[target_col].to_numpy()
    good = scores[target == 0]
    bad = scores[target == 1]
    if len(good) == 0 or len(bad) == 0:
        raise ValueError(f"'{target_col}' must contain both 0 and 1 values to run a KS test")
    result = ks_2samp(good, bad)
    return {"kind": "ks_test", "ks_statistic": float(result.statistic), "p_value": float(result.pvalue)}


register_block(
    BlockSpec(
        category="ks_test",
        block_type="output",
        group="tests",
        display_name="KS test",
        inputs=[PortSpec("df")],
        outputs=[PortSpec("metric", type="scalar_metric")],
        fn=ks_test,
        metadata_transform=lambda *_a, **_k: {},
    )
)


def auc_gini(df: pl.DataFrame, score_col: str, target_col: str) -> dict:
    from sklearn.metrics import roc_auc_score

    auc = float(roc_auc_score(df[target_col].to_numpy(), df[score_col].to_numpy()))
    return {"kind": "auc_gini", "auc": auc, "gini": 2 * auc - 1}


register_block(
    BlockSpec(
        category="auc_gini",
        block_type="output",
        group="tests",
        display_name="AUC / Gini",
        inputs=[PortSpec("df")],
        outputs=[PortSpec("metric", type="scalar_metric")],
        fn=auc_gini,
        metadata_transform=lambda *_a, **_k: {},
    )
)


def psi_test(expected: pl.DataFrame, actual: pl.DataFrame, col: str, bins: int = 10) -> dict:
    """Population Stability Index: how much `col`'s distribution in `actual`
    has drifted from `expected` (e.g. current score distribution vs. the
    distribution the model was built/validated on)."""
    import numpy as np

    exp_vals = expected[col].to_numpy()
    act_vals = actual[col].to_numpy()
    edges = np.quantile(exp_vals, np.linspace(0, 1, bins + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    edges = np.unique(edges)

    exp_counts, _ = np.histogram(exp_vals, bins=edges)
    act_counts, _ = np.histogram(act_vals, bins=edges)
    exp_pct = exp_counts / max(len(exp_vals), 1)
    act_pct = act_counts / max(len(act_vals), 1)
    # Laplace-style smoothing so an empty bucket never produces ln(0).
    eps = 1e-6
    contributions = (act_pct - exp_pct) * np.log((act_pct + eps) / (exp_pct + eps))
    return {
        "kind": "psi_test",
        "psi": float(contributions.sum()),
        "buckets": [
            {"expected_pct": float(e), "actual_pct": float(a), "contribution": float(c)}
            for e, a, c in zip(exp_pct, act_pct, contributions)
        ],
    }


register_block(
    BlockSpec(
        category="psi_test",
        block_type="output",
        group="tests",
        display_name="PSI test",
        inputs=[PortSpec("expected"), PortSpec("actual")],
        outputs=[PortSpec("metric", type="scalar_metric")],
        fn=psi_test,
        metadata_transform=lambda *_a, **_k: {},
    )
)
