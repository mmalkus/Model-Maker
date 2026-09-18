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


def rating_summary(df: pl.DataFrame, grade_col: str, target_col: str) -> dict:
    """Per-grade population and observed default rate -- the standard check
    that a rating scale (see modelling.fit_master_scale/assign_rating_grade)
    is monotonic: default rate should rise, never fall, from the
    lowest-risk grade to the highest."""
    stats = df.group_by(grade_col).agg(n=pl.len(), default_rate=pl.col(target_col).mean())
    stats = stats.sort(pl.col(grade_col).cast(pl.Int64, strict=False), nulls_last=True)
    rows = stats.to_dicts()
    rates = [r["default_rate"] for r in rows]
    monotonic = all(a <= b for a, b in zip(rates, rates[1:]))
    return {"kind": "rating_summary", "grades": rows, "monotonic": monotonic}


register_block(
    BlockSpec(
        category="rating_summary",
        block_type="output",
        group="tests",
        display_name="Rating scale summary",
        inputs=[PortSpec("df")],
        outputs=[PortSpec("metric", type="scalar_metric")],
        fn=rating_summary,
        metadata_transform=lambda *_a, **_k: {},
    )
)


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


def roc_curve(df: pl.DataFrame, score_col: str, target_col: str) -> dict:
    """The full ROC curve -- false/true positive rate at each distinct
    score threshold -- not just the scalar AUC that auc_gini reports, for
    plotting or a closer look at where a model's discrimination actually
    comes from."""
    from sklearn.metrics import roc_auc_score, roc_curve as _roc_curve

    y = df[target_col].to_numpy()
    scores = df[score_col].to_numpy()
    fpr, tpr, thresholds = _roc_curve(y, scores)
    auc = float(roc_auc_score(y, scores))
    return {
        "kind": "roc_curve",
        "auc": auc,
        # sklearn's first threshold is a synthetic max(score)+1 (not a real
        # cutoff, just "reject everything") and can come back as +inf --
        # nulled out rather than left as a non-JSON float.
        "points": [
            {"fpr": float(f), "tpr": float(t), "threshold": float(th) if th == th and th != float("inf") else None}
            for f, t, th in zip(fpr, tpr, thresholds)
        ],
    }


register_block(
    BlockSpec(
        category="roc_curve",
        block_type="output",
        group="tests",
        display_name="ROC curve",
        inputs=[PortSpec("df")],
        outputs=[PortSpec("metric", type="scalar_metric")],
        fn=roc_curve,
        metadata_transform=lambda *_a, **_k: {},
    )
)


def calibration_test(df: pl.DataFrame, score_col: str, target_col: str, bins: int = 10) -> dict:
    """Hosmer-Lemeshow goodness-of-fit test: buckets rows into `bins`
    quantile groups by predicted probability (`score_col`), then compares
    each bucket's observed event count to its expected count (the sum of
    predicted probabilities in it). A large HL statistic (low p-value)
    means the model's predicted probabilities don't match reality well,
    even when discrimination (AUC/KS) looks fine -- calibration and
    discrimination are different things, and this is what checks the
    former."""
    from scipy.stats import chi2

    bucketed = df.with_columns(pl.col(score_col).qcut(bins, allow_duplicates=True).alias("_bucket"))
    stats = bucketed.group_by("_bucket").agg(
        n=pl.len(),
        observed=pl.col(target_col).sum(),
        expected=pl.col(score_col).sum(),
    ).sort("expected")
    rows = stats.to_dicts()

    hl_statistic = 0.0
    for r in rows:
        n, observed, expected = r["n"], float(r["observed"]), float(r["expected"])
        # A bucket with nothing expected/expected==n contributes an
        # undefined (0/0) term -- excluded rather than let it poison the
        # sum, the same way psi_test/iv_table smooth an empty bucket away
        # instead of blowing up on it.
        if expected <= 0 or expected >= n:
            continue
        hl_statistic += (observed - expected) ** 2 / (expected * (1 - expected / n))
    degrees_of_freedom = max(len(rows) - 2, 1)

    return {
        "kind": "calibration_test",
        "hl_statistic": float(hl_statistic),
        "p_value": float(chi2.sf(hl_statistic, degrees_of_freedom)),
        "degrees_of_freedom": degrees_of_freedom,
        "buckets": [
            {
                "bucket": str(r["_bucket"]),
                "n": r["n"],
                "observed_count": float(r["observed"]),
                "expected_count": float(r["expected"]),
            }
            for r in rows
        ],
    }


register_block(
    BlockSpec(
        category="calibration_test",
        block_type="output",
        group="tests",
        display_name="Calibration (Hosmer-Lemeshow)",
        inputs=[PortSpec("df")],
        outputs=[PortSpec("metric", type="scalar_metric")],
        fn=calibration_test,
        metadata_transform=lambda *_a, **_k: {},
    )
)
