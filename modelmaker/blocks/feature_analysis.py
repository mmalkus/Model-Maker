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

from ..metadata_transforms import infer_dtypes
from .base import BlockSpec, PortSpec, register_block


def iv_table(df: pl.DataFrame, target: str, features: list[str] | None = None, bins: int = 10) -> dict:
    """Information Value per candidate feature against a binary `target`
    (1 = event/bad, 0 = non-event/good) -- the standard univariate screen
    run before variable selection, across every candidate feature at once
    instead of one woe_transform at a time. Bins the same way
    modelling.woe_transform does (quantile bins for numeric columns, raw
    categories otherwise), then sums each bin's (%good - %bad) * WoE.
    A quick screen only: for the per-bin table, monotonicity, univariate
    Gini/KS, a missing bin, and bins you can reuse on test data, use
    fit_binning ("Univariate analysis & binning")."""

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


def characteristic_stability(
    expected: pl.DataFrame, actual: pl.DataFrame, features: list[str] | None = None, bins: int = 10
) -> pl.DataFrame:
    """Characteristic stability: the Population Stability Index of every
    feature between two samples at once (e.g. development vs. out-of-time,
    or train vs. a recent application month) -- one row per feature, with
    the band ("stable" < 0.1, "monitor" < 0.25, "unstable" otherwise).
    Numeric features are bucketed on `expected`'s quantiles; text/boolean
    features compare category shares (categories only seen in one sample
    count too); nulls are their own bucket either way. `features` defaults
    to every column the two samples share. `psi_test` is the single-column,
    bucket-level version of the same measure."""
    import numpy as np

    def _band(psi: float) -> str:
        return "stable" if psi < 0.1 else ("monitor" if psi < 0.25 else "unstable")

    eps = 1e-6
    cols = features or [c for c in expected.columns if c in actual.columns]
    rows = []
    for c in cols:
        e, a = expected[c], actual[c]
        is_numeric = e.dtype.is_numeric() and e.dtype != pl.Boolean
        if is_numeric:
            ev = e.drop_nulls().cast(pl.Float64).to_numpy()
            inner = np.unique(np.quantile(ev, np.linspace(0, 1, bins + 1))[1:-1]) if len(ev) else np.array([])
            def _shares(s):
                vals = s.cast(pl.Float64).to_numpy()
                nulls = np.isnan(vals)
                idx = np.searchsorted(inner, vals[~nulls], side="left")
                counts = np.bincount(idx, minlength=len(inner) + 1).astype(float)
                return np.append(counts, nulls.sum()) / max(len(vals), 1)
            ep, ap = _shares(e), _shares(a)
        else:
            ek = e.cast(pl.Utf8).fill_null("__missing__")
            ak = a.cast(pl.Utf8).fill_null("__missing__")
            cats = sorted(set(ek.unique().to_list()) | set(ak.unique().to_list()))
            ec, ac = ek.value_counts(), ak.value_counts()
            emap = dict(zip(ec[ek.name].to_list(), ec["count"].to_list()))
            amap = dict(zip(ac[ak.name].to_list(), ac["count"].to_list()))
            ep = np.array([emap.get(k, 0) for k in cats], dtype=float) / max(len(ek), 1)
            ap = np.array([amap.get(k, 0) for k in cats], dtype=float) / max(len(ak), 1)
        psi = float(((ap - ep) * np.log((ap + eps) / (ep + eps))).sum())
        rows.append(
            {
                "feature": c,
                "type": "numeric" if is_numeric else "categorical",
                "psi": psi,
                "band": _band(psi),
                "n_buckets": int(((ep > 0) | (ap > 0)).sum()),
                "expected_null_share": float(e.null_count() / max(len(e), 1)),
                "actual_null_share": float(a.null_count() / max(len(a), 1)),
            }
        )
    rows.sort(key=lambda r: -r["psi"])
    return pl.DataFrame(rows)


register_block(
    BlockSpec(
        category="characteristic_stability",
        block_type="output",
        group="tests",
        display_name="Characteristic stability (PSI per feature)",
        inputs=[PortSpec("expected"), PortSpec("actual")],
        outputs=[PortSpec("table")],
        fn=characteristic_stability,
        metadata_transform=infer_dtypes,
    )
)


def target_trend(
    df: pl.DataFrame, date_col: str, target_col: str, period: str = "quarter", features: list[str] | None = None
) -> pl.DataFrame:
    """The target over time: per `period` ("month", "quarter" or "year")
    of `date_col`, the row count, the mean of `target_col` (the observed
    default rate for a 0/1 target, the average LGD/CCF for a continuous
    one) and, for each of `features`, its mean (numeric) or null share
    (text) -- the time-series view of the univariate analysis, used to spot
    drift, seasonality, a policy change, or an immature last period.
    `date_col` may be a Date/Datetime column or ISO date strings;
    `target_col` auto-fills from the role=target column."""
    if period not in ("month", "quarter", "year"):
        raise ValueError("period must be 'month', 'quarter' or 'year'")
    dtype = df.schema[date_col]
    if dtype == pl.Utf8:
        d = pl.col(date_col).str.slice(0, 10).str.to_date("%Y-%m-%d", strict=False)
    elif isinstance(dtype, pl.Datetime):
        d = pl.col(date_col).dt.date()
    else:
        d = pl.col(date_col).cast(pl.Date)
    if period == "month":
        label = d.dt.strftime("%Y-%m")
    elif period == "quarter":
        label = pl.format("{}-Q{}", d.dt.year(), d.dt.quarter())
    else:
        label = d.dt.year().cast(pl.Utf8)
    aggs = [pl.len().alias("n"), pl.col(target_col).cast(pl.Float64).mean().alias("target_mean")]
    for f in features or []:
        if df.schema[f].is_numeric():
            aggs.append(pl.col(f).cast(pl.Float64).mean().alias(f"{f}_mean"))
        else:
            aggs.append(pl.col(f).is_null().mean().alias(f"{f}_null_share"))
    return df.with_columns(label.alias("period")).filter(pl.col("period").is_not_null()).group_by("period").agg(aggs).sort("period")


register_block(
    BlockSpec(
        category="target_trend",
        block_type="output",
        group="tests",
        display_name="Target trend over time",
        inputs=[PortSpec("df")],
        outputs=[PortSpec("table")],
        fn=target_trend,
        metadata_transform=infer_dtypes,
    )
)


def bin_chart(bins: pl.DataFrame, feature: str, title: str = "", output_dir: str = ".", block_id: str = "") -> bytes:
    """The standard univariate chart for one feature, from fit_binning's
    `bins` table: bars for each bin's share of the population, and a line
    for its event rate (binary target) or mean target (continuous) on a
    second axis -- showing at a glance whether the relationship is
    monotonic and where the population sits. Saved as
    bin_chart_<block_id>.png in `output_dir` too; `output_dir` and
    `block_id` are engine-injected."""
    import io
    import os

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = bins.filter((pl.col("feature") == feature) & (pl.col("count") > 0)).sort("bin")
    if rows.height == 0:
        raise ValueError(f"no bins for feature '{feature}' in this table")
    labels = rows["label"].to_list()
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(range(len(labels)), rows["share"].to_list(), color="#93c5fd")
    ax.set_ylabel("share of population")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
    ax2 = ax.twinx()
    ax2.plot(range(len(labels)), rows["target_mean"].to_list(), color="#b91c1c", marker="o")
    ax2.set_ylabel("event rate" if "woe" in rows.columns else "mean target")
    ax.set_title(title or feature)
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110)
    plt.close(fig)
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, f"bin_chart_{block_id}.png" if block_id else "bin_chart.png"), "wb") as f:
        f.write(buf.getvalue())
    return buf.getvalue()


register_block(
    BlockSpec(
        category="bin_chart",
        block_type="output",
        group="tests",
        display_name="Univariate bin chart",
        inputs=[PortSpec("bins")],
        outputs=[PortSpec("image", type="image")],
        fn=bin_chart,
        metadata_transform=lambda *_a, **_k: {},
    )
)
