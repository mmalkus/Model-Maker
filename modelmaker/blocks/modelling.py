"""Modelling block library: GLM, logistic regression, and weight-of-evidence
(WoE) transform -- the core toolkit for a credit-risk / PD-style workflow.
Same contract as blocks/library.py: each `fn` is a plain function over
pl.DataFrame / literal params, no DataFramePacket. A "model" output port
(see blocks/base.py PortType) carries a plain, JSON-shaped dict describing
the fitted model -- not a pickled estimator -- so it stays inspectable and
never needs a special deserializer downstream.
"""

from __future__ import annotations

import polars as pl

from ..packet import ColumnMeta, ColumnRole
from .base import BlockSpec, PortSpec, register_block


def _predictions_meta(new_col_roles: dict[str, ColumnRole]):
    """Tag each of this block's own new output columns with the given role;
    every other output column is just the input passed through unchanged.
    `predicted` is a unique role (see packet.UNIQUE_ROLES) -- a block that
    emits more than one prediction-shaped column (logistic_regression's
    predicted_proba/predicted_class) can only give PREDICTED to the one
    that's actually the model's score; the rest fall back to FEATURE, same
    as any other derived column."""

    def _fn(input_metas, outputs, params):
        (in_meta,) = input_metas.values()
        (df,) = outputs.values()
        result = dict(in_meta)
        for name, role in new_col_roles.items():
            result[name] = ColumnMeta(dtype=str(df.schema[name]), role=role)
        return {"predictions": {name: result[name] for name in df.columns if name in result}}

    return _fn


def glm_fit(
    df: pl.DataFrame,
    target: str,
    features: list[str],
    family: str = "gaussian",
    alpha: float = 0.0,
) -> tuple[pl.DataFrame, dict]:
    from sklearn.linear_model import TweedieRegressor

    power = {"gaussian": 0.0, "poisson": 1.0, "gamma": 2.0, "inverse_gaussian": 3.0}.get(family)
    if power is None:
        raise ValueError(f"unknown GLM family: {family!r} (use gaussian, poisson, gamma, or inverse_gaussian)")

    x = df.select(features).to_numpy()
    y = df[target].to_numpy()
    model = TweedieRegressor(power=power, alpha=alpha, link="auto", max_iter=500)
    model.fit(x, y)

    predictions = df.with_columns(pl.Series("predicted", model.predict(x)))
    artifact = {
        "kind": "glm",
        "family": family,
        "target": target,
        "features": features,
        "coefficients": dict(zip(features, [float(c) for c in model.coef_])),
        "intercept": float(model.intercept_),
        "alpha": alpha,
    }
    return predictions, artifact


register_block(
    BlockSpec(
        category="glm_fit",
        block_type="standard",
        group="modelling",
        display_name="GLM fit",
        inputs=[PortSpec("df")],
        outputs=[PortSpec("predictions"), PortSpec("model", type="model", required=False)],
        fn=glm_fit,
        metadata_transform=_predictions_meta({"predicted": ColumnRole.PREDICTED}),
    )
)


def logistic_regression(
    df: pl.DataFrame,
    target: str,
    features: list[str],
    C: float = 1.0,
    max_iter: int = 200,
) -> tuple[pl.DataFrame, dict]:
    from sklearn.linear_model import LogisticRegression

    x = df.select(features).to_numpy()
    y = df[target].to_numpy()
    model = LogisticRegression(C=C, max_iter=max_iter)
    model.fit(x, y)

    proba = model.predict_proba(x)[:, 1]
    predictions = df.with_columns(
        pl.Series("predicted_proba", proba),
        pl.Series("predicted_class", model.predict(x)),
    )
    artifact = {
        "kind": "logistic_regression",
        "target": target,
        "features": features,
        "coefficients": dict(zip(features, [float(c) for c in model.coef_[0]])),
        "intercept": float(model.intercept_[0]),
        "C": C,
    }
    return predictions, artifact


register_block(
    BlockSpec(
        category="logistic_regression",
        block_type="standard",
        group="modelling",
        display_name="Logistic regression",
        inputs=[PortSpec("df")],
        outputs=[PortSpec("predictions"), PortSpec("model", type="model", required=False)],
        fn=logistic_regression,
        metadata_transform=_predictions_meta(
            {"predicted_proba": ColumnRole.PREDICTED, "predicted_class": ColumnRole.FEATURE}
        ),
    )
)


def woe_transform(df: pl.DataFrame, col: str, target: str, bins: int = 10) -> pl.DataFrame:
    """Weight-of-evidence encoding of `col` against a binary `target`
    (1 = event/bad, 0 = non-event/good), the standard credit-scoring
    transform: WoE = ln(% of goods in bucket / % of bads in bucket)."""
    out_col = f"{col}_woe"
    is_numeric = df.schema[col].is_numeric()
    bucket_expr = (
        pl.col(col).qcut(bins, allow_duplicates=True).alias("_bucket") if is_numeric else pl.col(col).cast(pl.Utf8).alias("_bucket")
    )
    bucketed = df.with_columns(bucket_expr)
    tagged = bucketed.select("_bucket", pl.col(target).alias("_target"))

    total_goods = (tagged["_target"] == 0).sum()
    total_bads = (tagged["_target"] == 1).sum()
    # Laplace-style smoothing (+0.5) so an all-good or all-bad bucket never
    # produces ln(0) / a divide-by-zero.
    stats = tagged.group_by("_bucket").agg(
        goods=(pl.col("_target") == 0).sum(),
        bads=(pl.col("_target") == 1).sum(),
    )
    stats = stats.with_columns(
        (((pl.col("goods") + 0.5) / (total_goods + 0.5)) / ((pl.col("bads") + 0.5) / (total_bads + 0.5))).log().alias("_woe")
    )

    result = bucketed.join(stats.select("_bucket", "_woe"), on="_bucket", how="left")
    result = result.rename({"_woe": out_col}).drop("_bucket")
    return result


def _woe_meta(input_metas, outputs, params):
    (in_meta,) = input_metas.values()
    (df,) = outputs.values()
    out_col = f"{params['col']}_woe"
    result = dict(in_meta)
    result[out_col] = ColumnMeta(dtype=str(df.schema[out_col]), role=ColumnRole.FEATURE)
    return {"out": {name: result[name] for name in df.columns if name in result}}


def fit_master_scale(
    df: pl.DataFrame,
    score_col: str,
    target_col: str,
    n_grades: int = 10,
    algorithm: str = "quantile",
) -> dict:
    """Derives rating-grade cutoffs over `score_col` -- fit once (e.g. on a
    development sample) and reused by assign_rating_grade on every later
    run, so grade boundaries stay stable rather than shifting with whatever
    data happens to be on hand the way a plain per-run qcut would. Grade 1
    is the lowest score/risk, grade `n_grades` the highest (or fewer, for
    'monotonic_default_rate' -- see below).

    `algorithm`:
      - "quantile": equal population per grade.
      - "equal_width": equal-width bins between the observed min and max.
      - "monotonic_default_rate": pool-adjacent-violators over fine
        quantile bins, merging any pair whose observed default rate would
        otherwise fall from one grade to the next -- a real rating scale's
        default rate must rise monotonically with risk. Can return fewer
        than `n_grades` grades when the data doesn't support that many
        without violating monotonicity; never more.
    """
    import numpy as np

    scores = df[score_col].to_numpy().astype(float)
    targets = df[target_col].to_numpy().astype(float)
    if np.unique(scores).size < 2:
        raise ValueError(f"'{score_col}' has fewer than 2 distinct values -- cannot fit a master scale")
    if n_grades < 1:
        raise ValueError("n_grades must be at least 1")

    if algorithm == "quantile":
        edges = np.unique(np.quantile(scores, np.linspace(0, 1, n_grades + 1)))
    elif algorithm == "equal_width":
        edges = np.unique(np.linspace(scores.min(), scores.max(), n_grades + 1))
    elif algorithm == "monotonic_default_rate":

        def _pava_edges() -> np.ndarray:
            n_fine = min(max(n_grades * 5, 50), np.unique(scores).size)
            fine_edges = np.unique(np.quantile(scores, np.linspace(0, 1, n_fine + 1)))
            bin_idx = np.clip(np.searchsorted(fine_edges, scores, side="right") - 1, 0, len(fine_edges) - 2)

            # Each pool: [sum(target), count, first fine-bin, last fine-bin].
            pools: list[list[float]] = []
            for b in range(len(fine_edges) - 1):
                mask = bin_idx == b
                if mask.any():
                    pools.append([float(targets[mask].sum()), int(mask.sum()), b, b])

            # Pool-adjacent-violators: merge a pool into its predecessor
            # whenever its default rate would otherwise be lower.
            stack: list[list[float]] = []
            for pool in pools:
                stack.append(pool)
                while len(stack) > 1 and (stack[-2][0] / stack[-2][1]) > (stack[-1][0] / stack[-1][1]):
                    b_pool = stack.pop()
                    a_pool = stack.pop()
                    stack.append([a_pool[0] + b_pool[0], a_pool[1] + b_pool[1], a_pool[2], b_pool[3]])

            # Merge down to n_grades if PAVA still left more pools than
            # requested, always combining whichever adjacent pair has the
            # closest default rate -- never split a pool further apart.
            while len(stack) > n_grades:
                rates = [p[0] / p[1] for p in stack]
                gaps = [rates[i + 1] - rates[i] for i in range(len(stack) - 1)]
                i = min(range(len(gaps)), key=lambda j: gaps[j])
                a_pool, b_pool = stack[i], stack[i + 1]
                stack[i : i + 2] = [[a_pool[0] + b_pool[0], a_pool[1] + b_pool[1], a_pool[2], b_pool[3]]]

            return np.array([fine_edges[0]] + [fine_edges[p[3] + 1] for p in stack], dtype=float)

        edges = _pava_edges()
    else:
        raise ValueError(
            f"unknown algorithm {algorithm!r} (expected 'quantile', 'equal_width', or 'monotonic_default_rate')"
        )

    edges = edges.astype(float)
    edges[0], edges[-1] = -np.inf, np.inf
    grades = [{"grade": i + 1, "lower": float(edges[i]), "upper": float(edges[i + 1])} for i in range(len(edges) - 1)]

    return {"kind": "master_scale", "score_col": score_col, "algorithm": algorithm, "grades": grades}


register_block(
    BlockSpec(
        category="fit_master_scale",
        block_type="standard",
        group="modelling",
        display_name="Fit master scale",
        inputs=[PortSpec("df")],
        outputs=[PortSpec("master_scale", type="master_scale")],
        fn=fit_master_scale,
        metadata_transform=lambda *_a, **_k: {},
    )
)


def assign_rating_grade(df: pl.DataFrame, master_scale: dict, score_col: str | None = None) -> pl.DataFrame:
    """Buckets `score_col` (defaults to whichever column the master scale
    was fit on) into the grades `master_scale` describes -- the apply half
    of fit_master_scale, so the same cutoffs are reused run after run
    instead of being re-derived from whatever data happens to be on hand."""
    col = score_col or master_scale["score_col"]
    grades = master_scale["grades"]
    breaks = [g["upper"] for g in grades[:-1]]
    labels = [str(g["grade"]) for g in grades]
    return df.with_columns(pl.col(col).cut(breaks, labels=labels).cast(pl.Utf8).alias("grade"))


def _rating_grade_meta(input_metas, outputs, params):
    (in_meta,) = input_metas.values()
    (df,) = outputs.values()
    result = dict(in_meta)
    result["grade"] = ColumnMeta(dtype=str(df.schema["grade"]), role=ColumnRole.SEGMENT)
    return {"out": {name: result[name] for name in df.columns if name in result}}


register_block(
    BlockSpec(
        category="assign_rating_grade",
        block_type="standard",
        group="modelling",
        display_name="Assign rating grade",
        inputs=[PortSpec("df"), PortSpec("master_scale", type="master_scale")],
        outputs=[PortSpec("out")],
        fn=assign_rating_grade,
        metadata_transform=_rating_grade_meta,
    )
)


register_block(
    BlockSpec(
        category="woe_transform",
        block_type="standard",
        group="modelling",
        display_name="Weight of evidence",
        inputs=[PortSpec("df")],
        outputs=[PortSpec("out")],
        fn=woe_transform,
        metadata_transform=_woe_meta,
    )
)
