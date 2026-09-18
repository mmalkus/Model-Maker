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


def lgd_regression(
    df: pl.DataFrame,
    target: str,
    features: list[str],
    max_iter: int = 100,
    tol: float = 1e-8,
) -> tuple[pl.DataFrame, dict]:
    """Fractional logit (quasi-binomial GLM, Papke & Wooldridge 1996) for a
    continuous target bounded to [0, 1] -- LGD and CCF both take this shape.
    Unlike beta regression, the logit link handles observations sitting
    exactly at 0 or 1 (a full cure, a total loss) natively, which is the
    common case for both targets, so no boundary-value workaround is
    needed. Fit by IRLS: at each step, reweight by the working variance
    mu*(1-mu) of the current fit and solve a weighted least-squares update,
    same iteration logistic regression itself would converge to if `target`
    were binary rather than continuous -- fractional response is the same
    mean model and link, estimated by quasi-likelihood instead of true
    likelihood."""
    import numpy as np

    x = df.select(features).to_numpy()
    y = df[target].to_numpy().astype(float)
    if np.any((y < 0.0) | (y > 1.0)):
        raise ValueError(f"'{target}' must be within [0, 1] for a fractional-response (LGD/CCF) regression")

    n = x.shape[0]
    design = np.column_stack([np.ones(n), x])
    beta = np.zeros(design.shape[1])
    eps = 1e-6
    converged = False
    n_iter = 0
    for n_iter in range(1, max_iter + 1):
        eta = design @ beta
        mu = np.clip(1.0 / (1.0 + np.exp(-eta)), eps, 1.0 - eps)
        weight = mu * (1.0 - mu)
        working_response = eta + (y - mu) / weight
        weighted_design = design.T * weight
        beta_new = np.linalg.solve(weighted_design @ design, weighted_design @ working_response)
        if np.max(np.abs(beta_new - beta)) < tol:
            beta = beta_new
            converged = True
            break
        beta = beta_new

    predicted = 1.0 / (1.0 + np.exp(-(design @ beta)))
    predictions = df.with_columns(pl.Series("predicted", predicted))
    artifact = {
        "kind": "lgd_regression",
        "target": target,
        "features": features,
        "coefficients": dict(zip(features, [float(c) for c in beta[1:]])),
        "intercept": float(beta[0]),
        "n_iter": n_iter,
        "converged": converged,
    }
    return predictions, artifact


register_block(
    BlockSpec(
        category="lgd_regression",
        block_type="standard",
        group="modelling",
        display_name="LGD / CCF regression (fractional logit)",
        inputs=[PortSpec("df")],
        outputs=[PortSpec("predictions"), PortSpec("model", type="model", required=False)],
        fn=lgd_regression,
        metadata_transform=_predictions_meta({"predicted": ColumnRole.PREDICTED}),
    )
)


def predict(df: pl.DataFrame, model: dict) -> pl.DataFrame:
    """Applies a model artifact produced by glm_fit/logistic_regression (see
    their `model` output port) to a different dataframe -- the held-out/test
    half of a train_test_split, typically -- without refitting anything.
    Reads the coefficients straight out of the JSON artifact, so no pickled
    estimator is ever needed."""
    import numpy as np

    features = model["features"]
    missing = [f for f in features if f not in df.columns]
    if missing:
        raise ValueError(f"model expects feature column(s) not present in this data: {missing}")

    x = df.select(features).to_numpy()
    coefs = np.array([model["coefficients"][f] for f in features])
    linear = x @ coefs + model["intercept"]

    kind = model.get("kind")
    if kind == "logistic_regression":
        proba = 1.0 / (1.0 + np.exp(-linear))
        return df.with_columns(
            pl.Series("predicted_proba", proba),
            pl.Series("predicted_class", (proba >= 0.5).astype(int)),
        )
    if kind == "glm":
        # Mirrors TweedieRegressor's link="auto": identity for gaussian,
        # log for every other family (poisson/gamma/inverse_gaussian) -- see
        # glm_fit above.
        pred = linear if model.get("family", "gaussian") == "gaussian" else np.exp(linear)
        return df.with_columns(pl.Series("predicted", pred))
    if kind == "lgd_regression":
        pred = 1.0 / (1.0 + np.exp(-linear))
        return df.with_columns(pl.Series("predicted", pred))
    raise ValueError(f"unsupported model kind for predict: {kind!r}")


def _predict_meta(input_metas, outputs, params):
    in_meta = input_metas.get("df", {})
    (df,) = outputs.values()
    result = dict(in_meta)
    role_by_col = {
        "predicted": ColumnRole.PREDICTED,
        "predicted_proba": ColumnRole.PREDICTED,
        "predicted_class": ColumnRole.FEATURE,
    }
    for name, role in role_by_col.items():
        if name in df.columns:
            result[name] = ColumnMeta(dtype=str(df.schema[name]), role=role)
    return {"predictions": {name: result[name] for name in df.columns if name in result}}


register_block(
    BlockSpec(
        category="predict",
        block_type="standard",
        group="modelling",
        display_name="Predict",
        inputs=[PortSpec("df"), PortSpec("model", type="model")],
        outputs=[PortSpec("predictions")],
        fn=predict,
        metadata_transform=_predict_meta,
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


def compute_lgd(
    df: pl.DataFrame,
    ead_col: str,
    recovered_col: str,
    cost_col: str | None = None,
    floor: float | None = 0.0,
    cap: float | None = 1.0,
) -> pl.DataFrame:
    """LGD = 1 - recovery rate = (EAD - recoveries + workout costs) / EAD,
    the standard definition once recovery cash flows have already been
    aggregated and discounted to the default date upstream (that
    aggregation/discounting is not this block's job). `cost_col` is
    optional -- omit it if workout costs are already netted into
    `recovered_col`. An account with EAD <= 0 has no meaningful recovery
    rate and gets a null LGD rather than a divide-by-zero. `floor`/`cap`
    truncate the result to a plausible range (LGD outside [0, 1] is a data
    issue, not a real observation -- see model-developer-review.md §1.1);
    pass None to skip either truncation."""
    costs = pl.col(cost_col) if cost_col is not None else pl.lit(0.0)
    lgd = pl.when(pl.col(ead_col) > 0).then(
        (pl.col(ead_col) - pl.col(recovered_col) + costs) / pl.col(ead_col)
    ).otherwise(None)
    if floor is not None:
        lgd = lgd.clip(lower_bound=floor)
    if cap is not None:
        lgd = lgd.clip(upper_bound=cap)
    return df.with_columns(lgd.alias("lgd"))


def _compute_lgd_meta(input_metas, outputs, params):
    (in_meta,) = input_metas.values()
    (df,) = outputs.values()
    result = dict(in_meta)
    result["lgd"] = ColumnMeta(dtype=str(df.schema["lgd"]), role=ColumnRole.FEATURE)
    return {"out": {name: result[name] for name in df.columns if name in result}}


register_block(
    BlockSpec(
        category="compute_lgd",
        block_type="standard",
        group="modelling",
        display_name="Compute LGD",
        inputs=[PortSpec("df")],
        outputs=[PortSpec("out")],
        fn=compute_lgd,
        metadata_transform=_compute_lgd_meta,
    )
)


def compute_ccf(
    df: pl.DataFrame,
    limit_col: str,
    balance_ref_col: str,
    balance_default_col: str,
    floor: float | None = 0.0,
    cap: float | None = 1.0,
) -> pl.DataFrame:
    """Credit conversion factor observed on a defaulted account: the
    fraction of the undrawn commitment at a reference date (typically 12
    months, or the cohort start, before default -- which convention is used
    is a modelling decision made upstream, in how `balance_ref_col` was
    picked) that got drawn down by the time of default.

        undrawn_at_reference = max(limit - balance_at_reference, 0)
        ccf = (balance_at_default - balance_at_reference) / undrawn_at_reference

    An account already at or over its limit at the reference date has zero
    undrawn headroom and gets a zero CCF rather than a divide-by-zero.
    `floor`/`cap` truncate the result (a real observation can be negative --
    balance fell before default -- or exceed 1 if the limit itself changed;
    truncating to [0, 1] is the common convention, but pass None to keep the
    raw value and inspect it instead)."""
    undrawn = (pl.col(limit_col) - pl.col(balance_ref_col)).clip(lower_bound=0.0)
    ccf = pl.when(undrawn > 0).then(
        (pl.col(balance_default_col) - pl.col(balance_ref_col)) / undrawn
    ).otherwise(0.0)
    if floor is not None:
        ccf = ccf.clip(lower_bound=floor)
    if cap is not None:
        ccf = ccf.clip(upper_bound=cap)
    return df.with_columns(undrawn.alias("undrawn_at_reference"), ccf.alias("ccf"))


def _compute_ccf_meta(input_metas, outputs, params):
    (in_meta,) = input_metas.values()
    (df,) = outputs.values()
    result = dict(in_meta)
    result["undrawn_at_reference"] = ColumnMeta(dtype=str(df.schema["undrawn_at_reference"]), role=ColumnRole.FEATURE)
    result["ccf"] = ColumnMeta(dtype=str(df.schema["ccf"]), role=ColumnRole.FEATURE)
    return {"out": {name: result[name] for name in df.columns if name in result}}


register_block(
    BlockSpec(
        category="compute_ccf",
        block_type="standard",
        group="modelling",
        display_name="Compute CCF",
        inputs=[PortSpec("df")],
        outputs=[PortSpec("out")],
        fn=compute_ccf,
        metadata_transform=_compute_ccf_meta,
    )
)


def scorecard_scale(model: dict, base_score: float = 600.0, base_odds: float = 50.0, pdo: float = 20.0) -> dict:
    """Turns a fitted logistic_regression model's coefficients into a
    points-based scorecard: the standard points-to-double-odds (PDO)
    scaling used in retail credit scoring. `base_odds` (good:bad) maps to
    `base_score` points, and every `pdo` points doubles the odds.

    Assumes `model`'s features are already in the units you want scored on
    the card (e.g. WoE-transformed categories -- see woe_transform); for a
    raw continuous feature this still gives a correct scaling, just as
    points *per unit* of that feature rather than points per category."""
    import math

    if model.get("kind") != "logistic_regression":
        raise ValueError(f"scorecard_scale needs a logistic_regression model, got {model.get('kind')!r}")

    # Odds of the *good* outcome (not the target's own 1=bad encoding) rise
    # with score, so the model's own bad-odds logit gets negated below.
    factor = pdo / math.log(2)
    offset = base_score - factor * math.log(base_odds)

    coefficients: dict[str, float] = model["coefficients"]
    intercept_points = offset - factor * model["intercept"]
    rows = [
        {"feature": feature, "coefficient": coef, "points_per_unit": -coef * factor}
        for feature, coef in coefficients.items()
    ]
    return {
        "kind": "scorecard_scale",
        "base_score": base_score,
        "base_odds": base_odds,
        "pdo": pdo,
        "factor": factor,
        "offset": offset,
        "intercept_points": intercept_points,
        "rows": rows,
    }


register_block(
    BlockSpec(
        category="scorecard_scale",
        block_type="output",
        group="modelling",
        display_name="Scorecard scaling",
        inputs=[PortSpec("model", type="model")],
        outputs=[PortSpec("metric", type="scalar_metric")],
        fn=scorecard_scale,
        metadata_transform=lambda *_a, **_k: {},
    )
)
