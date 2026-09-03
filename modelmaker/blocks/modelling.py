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


def _predictions_meta(input_metas, outputs, params):
    (in_meta,) = input_metas.values()
    (df,) = outputs.values()
    result = {}
    for name in df.columns:
        result[name] = in_meta[name] if name in in_meta else ColumnMeta(dtype=str(df.schema[name]), role=ColumnRole.FEATURE)
    return {"predictions": result}


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
        metadata_transform=_predictions_meta,
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
        metadata_transform=_predictions_meta,
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
    tagged = df.select(bucket_expr, pl.col(target).alias("_target"))

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

    bucketed = df.with_columns(bucket_expr)
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
