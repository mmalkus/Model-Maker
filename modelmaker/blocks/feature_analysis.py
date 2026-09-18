"""Univariate/multivariate feature-screening blocks: Information Value and
a correlation/VIF matrix -- the standard pre-modelling variable-selection
step for a PD-style workflow. Same contract as stat_tests.py: each produces
a scalar_metric artifact (a plain dict), not a dataframe, so it terminates
a branch of the pipeline. Extra libraries, and any private helper a block
needs, are defined *inside* the block's own fn -- the compiled single-file
script inlines only inspect.getsource(fn) itself (see compiler.py), not
sibling module-level helpers, so a block's fn must be fully self-contained.
"""

from __future__ import annotations

import polars as pl

from .base import BlockSpec, PortSpec, register_block


def iv_table(df: pl.DataFrame, target: str, features: list[str] | None = None, bins: int = 10) -> dict:
    """Information Value per candidate feature against a binary `target`
    (1 = event/bad, 0 = non-event/good) -- the standard univariate screen
    run before variable selection, across every candidate feature at once
    instead of one woe_transform at a time. Bins the same way
    modelling.woe_transform does (quantile bins for numeric columns, raw
    categories otherwise), then sums each bin's (%good - %bad) * WoE."""

    def _iv_band(iv: float) -> str:
        # Standard Information Value interpretation bands.
        if iv < 0.02:
            return "useless"
        if iv < 0.1:
            return "weak"
        if iv < 0.3:
            return "medium"
        if iv < 0.5:
            return "strong"
        return "suspicious"  # almost always a leak, e.g. a near-duplicate of the target

    candidate_features = features or [c for c in df.columns if c != target]
    total_goods = (df[target] == 0).sum()
    total_bads = (df[target] == 1).sum()
    if total_goods == 0 or total_bads == 0:
        raise ValueError(f"'{target}' must contain both 0 and 1 values to compute IV")

    rows = []
    for feature in candidate_features:
        is_numeric = df.schema[feature].is_numeric()
        bucket_expr = (
            pl.col(feature).qcut(bins, allow_duplicates=True).alias("_bucket")
            if is_numeric
            else pl.col(feature).cast(pl.Utf8).alias("_bucket")
        )
        tagged = df.select(bucket_expr, pl.col(target).alias("_target"))
        stats = tagged.group_by("_bucket").agg(
            goods=(pl.col("_target") == 0).sum(),
            bads=(pl.col("_target") == 1).sum(),
        )
        # Laplace-style smoothing (+0.5), same as woe_transform, so an
        # all-good or all-bad bucket never produces ln(0).
        stats = stats.with_columns(
            (((pl.col("goods") + 0.5) / (total_goods + 0.5))).alias("_good_pct"),
            (((pl.col("bads") + 0.5) / (total_bads + 0.5))).alias("_bad_pct"),
        )
        stats = stats.with_columns((pl.col("_good_pct") / pl.col("_bad_pct")).log().alias("_woe"))
        stats = stats.with_columns(
            ((pl.col("_good_pct") - pl.col("_bad_pct")) * pl.col("_woe")).alias("_contribution")
        )
        iv = float(stats["_contribution"].sum())
        rows.append({"feature": feature, "iv": iv, "iv_band": _iv_band(iv), "n_bins": stats.height})

    rows.sort(key=lambda r: r["iv"], reverse=True)
    return {"kind": "iv_table", "target": target, "rows": rows}


register_block(
    BlockSpec(
        category="iv_table",
        block_type="output",
        group="tests",
        display_name="IV / univariate table",
        inputs=[PortSpec("df")],
        outputs=[PortSpec("metric", type="scalar_metric")],
        fn=iv_table,
        metadata_transform=lambda *_a, **_k: {},
    )
)


def correlation_matrix(df: pl.DataFrame, features: list[str] | None = None) -> dict:
    """Pairwise Pearson correlation and variance-inflation factor (VIF) for
    numeric columns -- the standard multicollinearity screen before
    variable selection. VIF_i = 1 / (1 - R^2) from regressing feature i on
    every other candidate feature; a VIF above ~5-10 flags a feature that's
    largely redundant given the others. Undefined values (a constant
    column, a perfect fit) come back as null rather than inf/NaN, which
    aren't valid JSON."""
    import numpy as np
    from sklearn.linear_model import LinearRegression

    def _finite_or_none(x: float) -> float | None:
        return x if np.isfinite(x) else None

    candidate_features = features or [c for c, dt in zip(df.columns, df.dtypes) if dt.is_numeric()]
    if len(candidate_features) < 2:
        raise ValueError("correlation_matrix needs at least 2 numeric features")

    x = df.select(candidate_features).to_numpy().astype(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        corr = np.corrcoef(x, rowvar=False)

    vifs: list[float | None] = []
    for i in range(len(candidate_features)):
        others = [j for j in range(len(candidate_features)) if j != i]
        y = x[:, i]
        if np.std(y) == 0:
            vifs.append(None)
            continue
        reg = LinearRegression().fit(x[:, others], y)
        r2 = reg.score(x[:, others], y)
        vifs.append(_finite_or_none(1.0 / (1.0 - r2)) if r2 < 1.0 else None)

    return {
        "kind": "correlation_matrix",
        "features": candidate_features,
        "correlation": [[_finite_or_none(float(v)) for v in row] for row in corr],
        "vif": [{"feature": f, "vif": v} for f, v in zip(candidate_features, vifs)],
    }


register_block(
    BlockSpec(
        category="correlation_matrix",
        block_type="output",
        group="tests",
        display_name="Correlation / VIF",
        inputs=[PortSpec("df")],
        outputs=[PortSpec("metric", type="scalar_metric")],
        fn=correlation_matrix,
        metadata_transform=lambda *_a, **_k: {},
    )
)
