"""Modelling block library: GLM, logistic regression, and weight-of-evidence
(WoE) transform -- the core toolkit for a credit-risk / PD-style workflow.
Same contract as blocks/library.py: each `fn` is a plain function over
pl.DataFrame / literal params, no DataFramePacket. A "model" output port
(see blocks/base.py PortType) carries a plain, JSON-shaped dict describing
the fitted model -- not a pickled estimator -- so it stays inspectable and
never needs a special deserializer downstream.
"""

from __future__ import annotations

from dataclasses import replace

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
    """Fits a generalized linear model of a continuous/count `target` on
    `features` (sklearn TweedieRegressor). Outputs: `predictions` -- the
    input rows plus a `predicted` column (role=predicted) -- and `model`, a
    JSON-shaped artifact (kind "glm": coefficients, intercept, family)
    that the predict block can apply to other data. `family` is
    "gaussian" (default, identity link), or "poisson", "gamma",
    "inverse_gaussian" (log link; gamma/inverse_gaussian need a strictly
    positive target, poisson a non-negative one). `alpha` is the L2
    penalty strength (0 = unpenalized). `features` must be numeric,
    null-free columns. `target` auto-fills from the input's role=target
    column when left unset. Predictions are in-sample (on the fit data)."""
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
        tags=("regression",),
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
    """Fits an L2-regularized logistic regression of a binary `target`
    (0/1, 1 = event/default) on `features` (sklearn LogisticRegression) --
    the standard PD / scorecard model. Outputs: `predictions` -- the input
    rows plus `predicted_proba` (P(target=1), role=predicted) and
    `predicted_class` (0/1 at a 0.5 cutoff) -- and `model`, a JSON-shaped
    artifact (kind "logistic_regression": coefficients, intercept) usable
    by predict and scorecard_scale. `C` is the inverse regularization
    strength (smaller = stronger penalty; default 1.0); `max_iter` caps
    solver iterations. `features` must be numeric, null-free columns (e.g.
    WoE-transformed). `target` auto-fills from the input's role=target
    column when left unset. Predictions are in-sample (on the fit data).

    The artifact also carries `statistics`: per coefficient (and the
    intercept) its standard error, Wald z and two-sided p-value, from the
    inverse of the penalized Hessian -- with the default mild penalty these
    are close to the unpenalized maximum-likelihood ones. With WoE
    features every coefficient should be negative (higher WoE = safer); a
    positive one is a sign of collinearity worth investigating."""
    import math

    import numpy as np
    from sklearn.linear_model import LogisticRegression

    x = df.select(features).to_numpy()
    y = df[target].to_numpy()
    model = LogisticRegression(C=C, max_iter=max_iter)
    model.fit(x, y)

    proba = model.predict_proba(x)[:, 1]
    design = np.column_stack([np.ones(len(x)), x.astype(float)])
    hessian = (design.T * (proba * (1.0 - proba))) @ design
    hessian[1:, 1:] += np.eye(len(features)) / C
    names = ["intercept", *features]
    estimates = [float(model.intercept_[0]), *[float(c) for c in model.coef_[0]]]
    try:
        std_errors = np.sqrt(np.clip(np.diag(np.linalg.inv(hessian)), 0.0, None))
    except np.linalg.LinAlgError:
        std_errors = [None] * len(names)
    statistics = {}
    for name, est, se in zip(names, estimates, std_errors):
        z = est / se if se else None
        statistics[name] = {
            "estimate": est,
            "std_error": float(se) if se is not None else None,
            "z": float(z) if z is not None else None,
            "p_value": float(math.erfc(abs(z) / math.sqrt(2.0))) if z is not None else None,
        }
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
        "statistics": statistics,
    }
    return predictions, artifact


register_block(
    BlockSpec(
        category="logistic_regression",
        block_type="standard",
        group="modelling",
        display_name="Logistic regression",
        tags=("regression", "scorecard", "pd"),
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
    likelihood.

    The artifact carries `statistics` per coefficient: robust (sandwich,
    Papke-Wooldridge) standard errors, z and p-value -- the quasi-likelihood
    model's own variance isn't the true one, so the sandwich form is the
    right basis for dropping or keeping a driver."""
    import math

    import numpy as np

    # Unlike glm_fit/logistic_regression (sklearn rejects these itself), a
    # single null/NaN here would silently turn every coefficient into NaN --
    # and compute_lgd legitimately emits a null LGD for EAD <= 0.
    for col in [target, *features]:
        missing = df[col].is_null().sum() + (df[col].is_nan().sum() if df[col].dtype.is_float() else 0)
        if missing:
            raise ValueError(
                f"'{col}' has {missing} null/NaN value(s); drop those rows upstream (e.g. a filter block) "
                "before fitting a fractional-response (LGD/CCF) regression"
            )

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
    mu = np.clip(predicted, eps, 1.0 - eps)
    bread = (design.T * (mu * (1.0 - mu))) @ design
    meat = (design.T * ((y - mu) ** 2)) @ design
    statistics = {}
    try:
        inv = np.linalg.inv(bread)
        std_errors = np.sqrt(np.clip(np.diag(inv @ meat @ inv), 0.0, None))
    except np.linalg.LinAlgError:
        std_errors = [None] * design.shape[1]
    for name, est, se in zip(["intercept", *features], beta, std_errors):
        z = float(est / se) if se else None
        statistics[name] = {
            "estimate": float(est),
            "std_error": float(se) if se is not None else None,
            "z": z,
            "p_value": float(math.erfc(abs(z) / math.sqrt(2.0))) if z is not None else None,
        }
    predictions = df.with_columns(pl.Series("predicted", predicted))
    artifact = {
        "kind": "lgd_regression",
        "target": target,
        "features": features,
        "coefficients": dict(zip(features, [float(c) for c in beta[1:]])),
        "intercept": float(beta[0]),
        "n_iter": n_iter,
        "converged": converged,
        "statistics": statistics,
    }
    return predictions, artifact


register_block(
    BlockSpec(
        category="lgd_regression",
        block_type="standard",
        group="modelling",
        display_name="LGD / CCF regression (fractional logit)",
        tags=("regression", "lgd", "ccf_ead"),
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
        tags=("regression",),
        inputs=[PortSpec("df"), PortSpec("model", type="model")],
        outputs=[PortSpec("predictions")],
        fn=predict,
        metadata_transform=_predict_meta,
    )
)


def woe_transform(df: pl.DataFrame, col: str, target: str, bins: int = 10) -> pl.DataFrame:
    """Weight-of-evidence encoding of `col` against a binary `target`
    (1 = event/bad, 0 = non-event/good), the standard credit-scoring
    transform: WoE = ln(% of goods in bucket / % of bads in bucket).

    Fits and applies in one go on whatever data it's given, so only use it
    for quick exploration: never run it on a test or out-of-time sample
    (it would derive that sample's WoE from that sample's own target --
    leakage that inflates test Gini). For a real model use fit_binning on
    the train sample and apply_binning everywhere."""
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
    min_grade_share: float = 0.0,
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
        default rate must rise monotonically with risk. Pools left over
        beyond `n_grades` are then merged by closest mean *predicted*
        score (log scale for a PD), not observed rate, so a zero-default
        low-risk tail still spreads over several grades. Can return fewer
        than `n_grades` grades when the data doesn't support that many
        without violating monotonicity; never more.

    `min_grade_share` (e.g. 0.03) then merges any grade holding less than
    that share of the fit population into its closest neighbour (by mean
    predicted score, as above), so no grade is too thin to back-test. Each grade in the
    artifact records its fit-sample `n`, `observed_rate` (mean target) and
    `pd` -- the mean score in the grade, i.e. the grade's PD when the score
    is a (calibrated) probability. The target may be binary (PD) or a
    continuous rate (LGD/CCF pools) -- only its mean per grade is used.
    """
    import numpy as np

    scores = df[score_col].to_numpy().astype(float)
    targets = df[target_col].to_numpy().astype(float)
    if np.unique(scores).size < 2:
        raise ValueError(f"'{score_col}' has fewer than 2 distinct values -- cannot fit a master scale")
    if n_grades < 1:
        raise ValueError("n_grades must be at least 1")

    # How far apart two grades are, for deciding which neighbours to merge:
    # by their mean *predicted* score, never their observed rate -- on a
    # thin low-risk tail every zero-default pool has observed rate 0, so
    # merging on observed rate folds them all into one grade. On the log
    # scale when the score is a probability (a PD master scale is
    # geometric: 0.1% -> 0.2% is as big a step as 10% -> 20%).
    log_scale = bool(scores.min() > 0.0 and scores.max() <= 1.0)

    def _distance(mean_a: float, mean_b: float) -> float:
        if log_scale:
            return abs(np.log(mean_a) - np.log(mean_b))
        return abs(mean_a - mean_b)

    if algorithm == "quantile":
        edges = np.unique(np.quantile(scores, np.linspace(0, 1, n_grades + 1)))
    elif algorithm == "equal_width":
        edges = np.unique(np.linspace(scores.min(), scores.max(), n_grades + 1))
    elif algorithm == "monotonic_default_rate":

        def _pava_edges() -> np.ndarray:
            n_fine = min(max(n_grades * 5, 50), np.unique(scores).size)
            fine_edges = np.unique(np.quantile(scores, np.linspace(0, 1, n_fine + 1)))
            # Right-closed (lo, hi] bins, the same convention assign_rating_grade
            # (pl.cut) and the grade statistics below use -- quantile edges are
            # data values, so with tied scores (a binned/WoE model has few
            # distinct ones) a left-closed pooling would put whole clumps of
            # observations in a different grade than they end up assigned to,
            # breaking the monotonicity this algorithm exists to guarantee.
            bin_idx = np.searchsorted(fine_edges[1:-1], scores, side="left")

            # Each pool: [sum(target), count, first fine-bin, last fine-bin, sum(score)].
            pools: list[list[float]] = []
            for b in range(len(fine_edges) - 1):
                mask = bin_idx == b
                if mask.any():
                    pools.append([float(targets[mask].sum()), int(mask.sum()), b, b, float(scores[mask].sum())])

            def _merge(a_pool: list[float], b_pool: list[float]) -> list[float]:
                return [a_pool[0] + b_pool[0], a_pool[1] + b_pool[1], a_pool[2], b_pool[3], a_pool[4] + b_pool[4]]

            # Pool-adjacent-violators: merge a pool into its predecessor
            # whenever its default rate would otherwise be lower.
            stack: list[list[float]] = []
            for pool in pools:
                stack.append(pool)
                while len(stack) > 1 and (stack[-2][0] / stack[-2][1]) > (stack[-1][0] / stack[-1][1]):
                    b_pool = stack.pop()
                    a_pool = stack.pop()
                    stack.append(_merge(a_pool, b_pool))

            # Merge down to n_grades if PAVA still left more pools than
            # requested, always combining whichever adjacent pair is closest
            # in mean predicted score (see _distance) -- never split a pool
            # further apart. Merging neighbours of a monotonic sequence
            # keeps it monotonic, so this can't undo the PAVA guarantee.
            while len(stack) > n_grades:
                means = [p[4] / p[1] for p in stack]
                gaps = [_distance(means[i], means[i + 1]) for i in range(len(stack) - 1)]
                i = min(range(len(gaps)), key=lambda j: gaps[j])
                stack[i : i + 2] = [_merge(stack[i], stack[i + 1])]

            return np.array([fine_edges[0]] + [fine_edges[p[3] + 1] for p in stack], dtype=float)

        edges = _pava_edges()
    else:
        raise ValueError(
            f"unknown algorithm {algorithm!r} (expected 'quantile', 'equal_width', or 'monotonic_default_rate')"
        )

    edges = edges.astype(float)
    edges[0], edges[-1] = -np.inf, np.inf

    def _grade_index(e: np.ndarray) -> np.ndarray:
        # Right-closed (lower, upper], matching assign_rating_grade's pl.cut.
        return np.searchsorted(e[1:-1], scores, side="left")

    if min_grade_share > 0:
        min_n = min_grade_share * len(scores)
        while len(edges) > 2:
            idx = _grade_index(edges)
            counts = np.bincount(idx, minlength=len(edges) - 1)
            small = [g for g in range(len(counts)) if counts[g] < min_n]
            if not small:
                break
            g = min(small, key=lambda k: counts[k])
            # Neighbour by closest mean predicted score (see _distance).
            means = [scores[idx == k].mean() if counts[k] else scores.mean() for k in range(len(counts))]
            if g == 0:
                drop = 1
            elif g == len(counts) - 1:
                drop = g
            else:
                drop = g if _distance(means[g], means[g - 1]) <= _distance(means[g], means[g + 1]) else g + 1
            edges = np.delete(edges, drop)

    idx = _grade_index(edges)
    grades = []
    for i in range(len(edges) - 1):
        mask = idx == i
        n = int(mask.sum())
        grades.append(
            {
                "grade": i + 1,
                "lower": float(edges[i]),
                "upper": float(edges[i + 1]),
                "n": n,
                "observed_rate": float(targets[mask].mean()) if n else None,
                "pd": float(scores[mask].mean()) if n else None,
            }
        )

    return {"kind": "master_scale", "score_col": score_col, "algorithm": algorithm, "grades": grades}


register_block(
    BlockSpec(
        category="fit_master_scale",
        block_type="standard",
        group="modelling",
        display_name="Fit master scale",
        tags=("rating_scale", "pd"),
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
    instead of being re-derived from whatever data happens to be on hand.
    Also adds `grade_pd`, the grade's PD from the master scale (its mean
    score on the fit sample), when the scale records one -- the value a
    grade-level PD back-test (grade_backtest) compares against."""
    col = score_col or master_scale["score_col"]
    grades = master_scale["grades"]
    breaks = [g["upper"] for g in grades[:-1]]
    labels = [str(g["grade"]) for g in grades]
    out = df.with_columns(pl.col(col).cut(breaks, labels=labels).cast(pl.Utf8).alias("grade"))
    if all(g.get("pd") is not None for g in grades):
        pd_map = {str(g["grade"]): float(g["pd"]) for g in grades}
        out = out.with_columns(pl.col("grade").replace_strict(pd_map, default=None, return_dtype=pl.Float64).alias("grade_pd"))
    return out


def _rating_grade_meta(input_metas, outputs, params):
    (in_meta,) = input_metas.values()
    (df,) = outputs.values()
    result = dict(in_meta)
    result["grade"] = ColumnMeta(dtype=str(df.schema["grade"]), role=ColumnRole.SEGMENT)
    if "grade_pd" in df.columns:
        result["grade_pd"] = ColumnMeta(dtype=str(df.schema["grade_pd"]), role=ColumnRole.FEATURE)
    return {"out": {name: result[name] for name in df.columns if name in result}}


register_block(
    BlockSpec(
        category="assign_rating_grade",
        block_type="standard",
        group="modelling",
        display_name="Assign rating grade",
        tags=("rating_scale", "pd"),
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
        tags=("binning_woe", "scorecard"),
        inputs=[PortSpec("df")],
        outputs=[PortSpec("out")],
        fn=woe_transform,
        metadata_transform=_woe_meta,
    )
)


def forward_default_flag(
    df: pl.DataFrame,
    id_col: str,
    date_col: str,
    default_col: str,
    horizon: int = 12,
    flag_col: str | None = None,
    performing_only: bool = True,
    incomplete: str = "drop",
) -> pl.DataFrame:
    """Builds the PD target from a monthly account panel (one row per
    account per month): `flag_col` (default "default_<horizon>m") is 1 when
    the account is in default (`default_col` truthy) in any month after the
    row's own, up to `horizon` calendar months ahead, else 0. Months are
    calendar months, not rows, so a gap in an account's history doesn't
    stretch the window.

    `date_col` may be a Date/Datetime column, ISO date strings, or integer
    YYYYMM periods. `id_col` + `date_col` must be unique.

    `performing_only` (default on) keeps only rows not already in default
    at observation -- the usual PD population. A window is incomplete when
    the account's history ends before `horizon` months out with no default
    seen (the most recent periods, or accounts that closed or were sold):
    `incomplete` "drop" (default) removes those rows, "null" keeps them
    with a null flag. A default found inside the window always counts, even
    if the history ends right after it."""
    if horizon < 1:
        raise ValueError("horizon must be at least 1 month")
    if incomplete not in ("drop", "null"):
        raise ValueError("incomplete must be 'drop' or 'null'")
    flag_col = flag_col or f"default_{horizon}m"

    dtype = df.schema[date_col]
    if dtype.is_integer():
        month = (pl.col(date_col) // 100) * 12 + pl.col(date_col) % 100
    else:
        if dtype == pl.Utf8:
            d = pl.col(date_col).str.slice(0, 10).str.to_date("%Y-%m-%d", strict=False)
        elif isinstance(dtype, pl.Datetime):
            d = pl.col(date_col).dt.date()
        else:
            d = pl.col(date_col).cast(pl.Date)
        month = d.dt.year().cast(pl.Int64) * 12 + d.dt.month().cast(pl.Int64)

    work = df.with_row_index("__row").with_columns(
        month.cast(pl.Int64).alias("__m"),
        (pl.col(default_col).cast(pl.Float64).fill_null(0) > 0).alias("__d"),
    )
    if work.select(pl.struct(id_col, "__m").is_duplicated().any()).item():
        raise ValueError(f"{id_col} + {date_col} is not unique: expected one row per account per month")

    work = (
        work.sort(id_col, "__m")
        .with_columns(
            # First default month strictly after this row's, per account.
            pl.when(pl.col("__d")).then(pl.col("__m")).shift(-1).fill_null(strategy="backward").over(id_col).alias("__next"),
            pl.col("__m").max().over(id_col).alias("__last"),
        )
        .with_columns((pl.col("__next") <= pl.col("__m") + horizon).fill_null(False).alias("__hit"))
        .with_columns((pl.col("__hit") | (pl.col("__last") >= pl.col("__m") + horizon)).alias("__complete"))
        .sort("__row")
    )
    if performing_only:
        work = work.filter(~pl.col("__d"))
    if incomplete == "drop":
        work = work.filter(pl.col("__complete"))
    flag = pl.when(pl.col("__complete")).then(pl.col("__hit").cast(pl.Int64)).otherwise(None)
    return work.with_columns(flag.alias(flag_col)).drop("__row", "__m", "__d", "__next", "__last", "__hit", "__complete")


def _forward_default_flag_meta(input_metas, outputs, params):
    """The new flag is the modelling target. Target is a unique role, so an
    input column already tagged target gives it up (to excluded), and so does
    `default_col` itself -- the current-month default status is the outcome
    the flag is built from, never a feature."""
    (in_meta,) = input_metas.values()
    (df,) = outputs.values()
    flag_col = params.get("flag_col") or f"default_{params.get('horizon', 12)}m"
    default_col = params.get("default_col")
    result = {}
    for name in df.columns:
        if name == flag_col:
            result[name] = ColumnMeta(dtype=str(df.schema[name]), role=ColumnRole.TARGET)
        elif name in in_meta:
            meta = in_meta[name]
            if name == default_col or meta.role == ColumnRole.TARGET:
                meta = replace(meta, role=ColumnRole.EXCLUDED)
            result[name] = meta
    return {"out": result}


register_block(
    BlockSpec(
        category="forward_default_flag",
        block_type="standard",
        group="modelling",
        display_name="Forward default flag (PD target)",
        tags=("data_prep", "pd"),
        inputs=[PortSpec("df")],
        outputs=[PortSpec("out")],
        fn=forward_default_flag,
        metadata_transform=_forward_default_flag_meta,
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


def _target_role_unless_taken(in_meta: dict, new_col: str) -> ColumnRole:
    """The realised LGD/CCF is the modelling target -- tag it so, so
    target/actual_col params auto-fill downstream -- unless the input
    already has a target column (target is a unique role)."""
    taken = any(m.role == ColumnRole.TARGET for name, m in in_meta.items() if name != new_col)
    return ColumnRole.FEATURE if taken else ColumnRole.TARGET


def _compute_lgd_meta(input_metas, outputs, params):
    (in_meta,) = input_metas.values()
    (df,) = outputs.values()
    result = dict(in_meta)
    result["lgd"] = ColumnMeta(dtype=str(df.schema["lgd"]), role=_target_role_unless_taken(in_meta, "lgd"))
    return {"out": {name: result[name] for name in df.columns if name in result}}


register_block(
    BlockSpec(
        category="compute_lgd",
        block_type="standard",
        group="modelling",
        display_name="Compute LGD",
        tags=("lgd",),
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
    no_headroom: str = "zero",
) -> pl.DataFrame:
    """Credit conversion factor observed on a defaulted account: the
    fraction of the undrawn commitment at a reference date (typically 12
    months, or the cohort start, before default -- which convention is used
    is a modelling decision made upstream, in how `balance_ref_col` was
    picked) that got drawn down by the time of default.

        undrawn_at_reference = max(limit - balance_at_reference, 0)
        ccf = (balance_at_default - balance_at_reference) / undrawn_at_reference

    An account already at or over its limit at the reference date has zero
    undrawn headroom, so its CCF is undefined: `no_headroom` = "zero"
    (default) sets it to 0, "null" leaves it null so it can be excluded
    (apply_exclusions on "ccf IS NOT NULL") or modelled separately -- the
    usual choice, since a 0 there is a convention, not an observation.
    `floor`/`cap` truncate the result (a real observation can be negative --
    balance fell before default -- or exceed 1 if the limit itself changed;
    truncating to [0, 1] is the common convention, but pass None to keep the
    raw value and inspect it instead)."""
    undrawn = (pl.col(limit_col) - pl.col(balance_ref_col)).clip(lower_bound=0.0)
    if no_headroom not in ("zero", "null"):
        raise ValueError("no_headroom must be 'zero' or 'null'")
    ccf = pl.when(undrawn > 0).then(
        (pl.col(balance_default_col) - pl.col(balance_ref_col)) / undrawn
    ).otherwise(0.0 if no_headroom == "zero" else None)
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
    result["ccf"] = ColumnMeta(dtype=str(df.schema["ccf"]), role=_target_role_unless_taken(in_meta, "ccf"))
    return {"out": {name: result[name] for name in df.columns if name in result}}


register_block(
    BlockSpec(
        category="compute_ccf",
        block_type="standard",
        group="modelling",
        display_name="Compute CCF",
        tags=("ccf_ead",),
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
        tags=("scorecard",),
        inputs=[PortSpec("model", type="model")],
        outputs=[PortSpec("metric", type="scalar_metric")],
        fn=scorecard_scale,
        metadata_transform=lambda *_a, **_k: {},
    )
)


def stepwise_selection(
    df: pl.DataFrame,
    target: str,
    features: list[str],
    direction: str = "both",
    p_enter: float = 0.05,
    p_remove: float = 0.10,
    max_features: int | None = None,
) -> dict:
    """Stepwise variable selection by Wald p-value, the classic way to go
    from a univariate shortlist to a final model. Fits an unpenalized logit
    model -- a logistic regression for a binary `target`, a fractional
    logit (robust standard errors) for a continuous one in [0, 1] (LGD,
    CCF) -- and:
      - direction "forward": adds, one at a time, the candidate with the
        lowest p-value below `p_enter`;
      - "backward": starts from all `features` and removes the one with
        the highest p-value above `p_remove`, one at a time;
      - "both" (default): forward, but after each addition re-checks every
        selected feature and drops any whose p-value rose above `p_remove`.
    `max_features` caps the selection. Emits the `selected` feature list
    (copy it into the model block's `features`), every step taken, and the
    final model's coefficient table -- including each coefficient's sign,
    to check it against the univariate direction. `features` must be
    numeric and null-free (e.g. WoE columns)."""
    import math

    import numpy as np

    y = df[target].cast(pl.Float64).to_numpy()
    if np.any((y < 0) | (y > 1)):
        raise ValueError(f"'{target}' must be 0/1 or within [0, 1]")
    binary = bool(np.isin(np.unique(y), [0.0, 1.0]).all())
    if direction not in ("forward", "backward", "both"):
        raise ValueError("direction must be 'forward', 'backward' or 'both'")

    def _fit(cols: list[str]) -> dict[str, float]:
        x = df.select(cols).cast(pl.Float64).to_numpy() if cols else np.empty((len(y), 0))
        design = np.column_stack([np.ones(len(y)), x])
        beta = np.zeros(design.shape[1])
        for _ in range(100):
            mu = np.clip(1.0 / (1.0 + np.exp(-(design @ beta))), 1e-9, 1 - 1e-9)
            w = mu * (1 - mu)
            hess = (design.T * w) @ design + 1e-9 * np.eye(design.shape[1])
            step = np.linalg.solve(hess, design.T @ (y - mu))
            beta = beta + step
            if np.max(np.abs(step)) < 1e-8:
                break
        mu = np.clip(1.0 / (1.0 + np.exp(-(design @ beta))), 1e-9, 1 - 1e-9)
        inv = np.linalg.pinv((design.T * (mu * (1 - mu))) @ design)
        cov = inv if binary else inv @ ((design.T * (y - mu) ** 2) @ design) @ inv
        se = np.sqrt(np.clip(np.diag(cov), 1e-300, None))
        out = {}
        for i, name in enumerate(["intercept", *cols]):
            z = beta[i] / se[i]
            out[name] = {"estimate": float(beta[i]), "std_error": float(se[i]), "p_value": float(math.erfc(abs(z) / math.sqrt(2)))}
        return out

    limit = max_features or len(features)
    selected: list[str] = list(features) if direction == "backward" else []
    steps: list[dict] = []
    for _ in range(4 * len(features) + 4):
        changed = False
        if direction in ("forward", "both") and len(selected) < limit:
            best = None
            for cand in (f for f in features if f not in selected):
                p = _fit([*selected, cand])[cand]["p_value"]
                if p < p_enter and (best is None or p < best[1]):
                    best = (cand, p)
            if best is not None:
                selected.append(best[0])
                steps.append({"step": len(steps) + 1, "action": "add", "feature": best[0], "p_value": best[1]})
                changed = True
        if direction in ("backward", "both") and selected:
            stats = _fit(selected)
            worst = max(selected, key=lambda f: stats[f]["p_value"])
            if stats[worst]["p_value"] > p_remove:
                selected.remove(worst)
                steps.append({"step": len(steps) + 1, "action": "remove", "feature": worst, "p_value": stats[worst]["p_value"]})
                changed = True
        if not changed:
            break
    final = _fit(selected)
    return {
        "kind": "stepwise_selection",
        "target": target,
        "target_type": "binary" if binary else "continuous",
        "direction": direction,
        "selected": selected,
        "dropped": [f for f in features if f not in selected],
        "steps": steps,
        "coefficients": [{"feature": k, **v, "sign": "+" if v["estimate"] >= 0 else "-"} for k, v in final.items()],
    }


register_block(
    BlockSpec(
        category="stepwise_selection",
        block_type="output",
        group="modelling",
        display_name="Stepwise feature selection",
        tags=("feature_selection", "regression"),
        inputs=[PortSpec("df")],
        outputs=[PortSpec("metric", type="scalar_metric")],
        fn=stepwise_selection,
        metadata_transform=lambda *_a, **_k: {},
    )
)


def calibrate_model(df: pl.DataFrame, model: dict, central_tendency: float) -> dict:
    """Calibrates a logit-link model (logistic_regression for PD, or
    lgd_regression for LGD/CCF) to a target level by shifting its
    intercept, so its average prediction over `df` equals
    `central_tendency` -- e.g. the long-run average default rate, when the
    development sample's default rate isn't representative of the cycle,
    or a long-run average LGD. Ranking is unchanged (only the intercept
    moves); the output is a new `model` for predict / scorecard blocks,
    recording the shift and the mean prediction before and after. `df` is
    the calibration sample (usually the development data) and must carry
    the model's feature columns."""
    import copy

    import numpy as np

    if model.get("kind") not in ("logistic_regression", "lgd_regression"):
        raise ValueError(f"calibrate_model needs a logistic_regression or lgd_regression model, got {model.get('kind')!r}")
    if not 0.0 < central_tendency < 1.0:
        raise ValueError("central_tendency must be strictly between 0 and 1")
    feats = model["features"]
    x = df.select(feats).cast(pl.Float64).to_numpy()
    lp = x @ np.array([model["coefficients"][f] for f in feats]) + model["intercept"]

    def mean_at(shift: float) -> float:
        return float(np.mean(1.0 / (1.0 + np.exp(-(lp + shift)))))

    lo, hi = -30.0, 30.0
    for _ in range(200):
        mid = (lo + hi) / 2
        if mean_at(mid) < central_tendency:
            lo = mid
        else:
            hi = mid
    shift = (lo + hi) / 2
    out = copy.deepcopy(model)
    out["intercept"] = float(model["intercept"] + shift)
    out["calibration"] = {
        "method": "intercept_shift",
        "central_tendency": central_tendency,
        "intercept_shift": shift,
        "mean_prediction_before": mean_at(0.0),
        "mean_prediction_after": mean_at(shift),
    }
    return out


register_block(
    BlockSpec(
        category="calibrate_model",
        block_type="standard",
        group="modelling",
        display_name="Calibrate to central tendency",
        tags=("regression", "calibration", "pd", "lgd"),
        inputs=[PortSpec("df"), PortSpec("model", type="model")],
        outputs=[PortSpec("model", type="model")],
        fn=calibrate_model,
        metadata_transform=lambda *_a, **_k: {},
    )
)


def margin_of_conservatism(
    df: pl.DataFrame,
    predicted_col: str,
    add_on: float = 0.0,
    multiplier: float = 1.0,
    floor: float | None = None,
    cap: float | None = 1.0,
    output_col: str = "predicted_moc",
) -> pl.DataFrame:
    """Applies a margin of conservatism (or downturn adjustment) to a PD,
    LGD or CCF estimate: `output_col` = clip(`predicted_col` * multiplier +
    add_on, floor, cap) -- e.g. a +5pp downturn LGD add-on, or a 1.1x PD
    MoC. The new column becomes the role=predicted estimate that validation
    blocks pick up; the original is kept (as a feature) for comparison. A
    regulatory floor (e.g. 0.0003 for PD) goes in `floor`."""
    value = pl.col(predicted_col) * multiplier + add_on
    if floor is not None:
        value = value.clip(lower_bound=floor)
    if cap is not None:
        value = value.clip(upper_bound=cap)
    return df.with_columns(value.alias(output_col))


def _moc_meta(input_metas, outputs, params):
    (in_meta,) = input_metas.values()
    (df,) = outputs.values()
    out_col = params.get("output_col", "predicted_moc")
    result = {
        name: (ColumnMeta(dtype=m.dtype, role=ColumnRole.FEATURE, description=m.description, tags=m.tags) if m.role == ColumnRole.PREDICTED else m)
        for name, m in in_meta.items()
    }
    result[out_col] = ColumnMeta(dtype=str(df.schema[out_col]), role=ColumnRole.PREDICTED)
    return {"out": {name: result[name] for name in df.columns if name in result}}


register_block(
    BlockSpec(
        category="margin_of_conservatism",
        block_type="standard",
        group="modelling",
        display_name="Margin of conservatism / downturn",
        tags=("calibration", "pd", "lgd", "ccf_ead"),
        inputs=[PortSpec("df")],
        outputs=[PortSpec("out")],
        fn=margin_of_conservatism,
        metadata_transform=_moc_meta,
    )
)


def discount_recoveries(
    facilities: pl.DataFrame,
    cashflows: pl.DataFrame,
    id_col: str,
    default_date_col: str,
    cf_date_col: str,
    recovery_col: str,
    cost_col: str | None = None,
    annual_rate: float = 0.05,
    rate_col: str | None = None,
) -> pl.DataFrame:
    """Discounts post-default recovery (and workout cost) cash flows back
    to each facility's default date and totals them per facility -- the
    step before compute_lgd, which expects recoveries already discounted.
    `cashflows` has one row per cash flow (`id_col`, `cf_date_col`,
    `recovery_col`, optionally `cost_col`); `facilities` one row per
    defaulted facility with `id_col` and `default_date_col`. Discount
    factor = (1 + rate) ** -(days since default / 365.25), with `rate` the
    flat `annual_rate` or, when `rate_col` names a facilities column (e.g.
    the contractual rate), that facility's own rate.

    Returns `facilities` with `pv_recovery`, `pv_cost` (0 when no cost
    column), `n_cashflows` and `workout_years` (default date to last cash
    flow) added; a facility without cash flows gets 0 recoveries/costs.
    Cash flows dated before default count at face value."""

    def _as_date(frame: pl.DataFrame, col: str) -> pl.Expr:
        dt = frame.schema[col]
        if dt == pl.Utf8:
            return pl.col(col).str.slice(0, 10).str.to_date("%Y-%m-%d", strict=False)
        if isinstance(dt, pl.Datetime):
            return pl.col(col).dt.date()
        return pl.col(col).cast(pl.Date)

    fac = facilities.select(
        pl.col(id_col),
        _as_date(facilities, default_date_col).alias("_default_date"),
        (pl.col(rate_col).cast(pl.Float64) if rate_col else pl.lit(annual_rate)).alias("_rate"),
    )
    cf = cashflows.select(
        pl.col(id_col),
        _as_date(cashflows, cf_date_col).alias("_cf_date"),
        pl.col(recovery_col).cast(pl.Float64).fill_null(0.0).alias("_rec"),
        (pl.col(cost_col).cast(pl.Float64).fill_null(0.0) if cost_col else pl.lit(0.0)).alias("_cost"),
    ).join(fac, on=id_col, how="inner")
    years = ((pl.col("_cf_date") - pl.col("_default_date")).dt.total_days() / 365.25).clip(lower_bound=0.0)
    factor = (1.0 + pl.col("_rate")) ** (-years)
    per_facility = cf.group_by(id_col).agg(
        (pl.col("_rec") * factor).sum().alias("pv_recovery"),
        (pl.col("_cost") * factor).sum().alias("pv_cost"),
        pl.len().cast(pl.Int64).alias("n_cashflows"),
        years.max().alias("workout_years"),
    )
    out = facilities.join(per_facility, on=id_col, how="left")
    return out.with_columns(
        pl.col("pv_recovery").fill_null(0.0),
        pl.col("pv_cost").fill_null(0.0),
        pl.col("n_cashflows").fill_null(0),
    )


def _discount_recoveries_meta(input_metas, outputs, params):
    in_meta = input_metas.get("facilities", {})
    (df,) = outputs.values()
    result = dict(in_meta)
    for name in ("pv_recovery", "pv_cost", "n_cashflows", "workout_years"):
        result[name] = ColumnMeta(dtype=str(df.schema[name]), role=ColumnRole.FEATURE)
    return {"out": {name: result[name] for name in df.columns if name in result}}


register_block(
    BlockSpec(
        category="discount_recoveries",
        block_type="standard",
        group="modelling",
        display_name="Discount recoveries",
        tags=("lgd",),
        inputs=[PortSpec("facilities"), PortSpec("cashflows")],
        outputs=[PortSpec("out")],
        fn=discount_recoveries,
        metadata_transform=_discount_recoveries_meta,
    )
)


def compute_ead(
    df: pl.DataFrame,
    balance_col: str,
    limit_col: str,
    predicted_col: str,
    output_col: str = "ead_predicted",
) -> pl.DataFrame:
    """Exposure at default from a CCF estimate -- what a CCF model is
    for: EAD = drawn balance + CCF * max(limit - balance, 0). Adds
    `output_col` (default `ead_predicted`). `predicted_col` is the CCF
    (auto-fills from the role=predicted column, e.g. the CCF model's
    `predicted`); compare against the realised balance at default with
    continuous_accuracy."""
    undrawn = (pl.col(limit_col) - pl.col(balance_col)).clip(lower_bound=0.0)
    return df.with_columns((pl.col(balance_col) + pl.col(predicted_col) * undrawn).alias(output_col))


def _compute_ead_meta(input_metas, outputs, params):
    (in_meta,) = input_metas.values()
    (df,) = outputs.values()
    out_col = params.get("output_col", "ead_predicted")
    result = dict(in_meta)
    result[out_col] = ColumnMeta(dtype=str(df.schema[out_col]), role=ColumnRole.FEATURE)
    return {"out": {name: result[name] for name in df.columns if name in result}}


register_block(
    BlockSpec(
        category="compute_ead",
        block_type="standard",
        group="modelling",
        display_name="Compute EAD from CCF",
        tags=("ccf_ead",),
        inputs=[PortSpec("df")],
        outputs=[PortSpec("out")],
        fn=compute_ead,
        metadata_transform=_compute_ead_meta,
    )
)


def long_run_average(
    df: pl.DataFrame,
    target_col: str,
    date_col: str,
    weight_col: str | None = None,
    period: str = "year",
) -> tuple[pl.DataFrame, dict]:
    """Long-run average of a realised risk parameter over a full cycle:
    the average LGD (or CCF, or default rate) per `period` ("year",
    "quarter" or "month") of `date_col` (the default date), plus the
    portfolio-level averages a calibration target is chosen from:
      - default_weighted: the mean over all defaults;
      - exposure_weighted: weighted by `weight_col` (e.g. EAD), when set;
      - time_weighted: the plain mean of the per-period averages, so every
        period counts equally however many defaults it had.
    `target_col` auto-fills from the role=target column (compute_lgd tags
    `lgd` as the target). Outputs the per-period `table` and the `metric`."""
    if period not in ("month", "quarter", "year"):
        raise ValueError("period must be 'month', 'quarter' or 'year'")
    dt = df.schema[date_col]
    if dt == pl.Utf8:
        d = pl.col(date_col).str.slice(0, 10).str.to_date("%Y-%m-%d", strict=False)
    elif isinstance(dt, pl.Datetime):
        d = pl.col(date_col).dt.date()
    else:
        d = pl.col(date_col).cast(pl.Date)
    label = {"month": d.dt.strftime("%Y-%m"), "quarter": pl.format("{}-Q{}", d.dt.year(), d.dt.quarter()), "year": d.dt.year().cast(pl.Utf8)}[period]
    data = df.filter(pl.col(target_col).is_not_null()).with_columns(label.alias("period"))
    aggs = [pl.len().alias("n"), pl.col(target_col).mean().alias("mean")]
    if weight_col:
        aggs += [
            ((pl.col(target_col) * pl.col(weight_col)).sum() / pl.col(weight_col).sum()).alias("exposure_weighted_mean"),
            pl.col(weight_col).sum().alias("total_weight"),
        ]
    table = data.group_by("period").agg(aggs).sort("period")
    metric = {
        "kind": "long_run_average",
        "target_col": target_col,
        "period": period,
        "n_periods": table.height,
        "n": data.height,
        "default_weighted": float(data[target_col].mean()) if data.height else None,
        "time_weighted": float(table["mean"].mean()) if table.height else None,
    }
    if weight_col and data.height:
        metric["exposure_weighted"] = float((data[target_col] * data[weight_col]).sum() / data[weight_col].sum())
    return table, metric


register_block(
    BlockSpec(
        category="long_run_average",
        block_type="output",
        group="modelling",
        display_name="Long-run average (LGD/CCF/DR)",
        tags=("calibration", "pd", "lgd", "ccf_ead"),
        inputs=[PortSpec("df")],
        outputs=[PortSpec("table"), PortSpec("metric", type="scalar_metric")],
        aggregate_outputs=("table",),
        fn=long_run_average,
        metadata_transform=lambda _im, outputs, _p: {k: {c: ColumnMeta(dtype=str(v.schema[c])) for c in v.columns} for k, v in outputs.items()},
    )
)
