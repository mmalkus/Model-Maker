"""Stochastic engine block library (see /stochastic-engine-proposal.md):
distribution fitting, a vectorised compound-Poisson simulation block,
copula-based dependency/aggregation, and the curve-fitting / var-covar
proxy-function family. Same contract as every other block module: each
`fn` is a plain function over pl.DataFrame / literal params / plain dicts,
never a DataFramePacket -- the engine (runner.run_block) does the
packet unwrap/rewrap. All the actual numerical work lives in
`modelmaker.stochastic.*`; this module is the thin pl.DataFrame <-> numpy
edge plus BlockSpec registration.

Every `fn` below imports numpy and its `modelmaker.stochastic.*` helpers
*inside* its own body, by absolute path, never at this module's top level
-- the same convention every other block module uses for numpy/sklearn
(see e.g. blocks/modelling.py's `from sklearn.linear_model import ...`
inside glm_fit). compiler.compile_graph inlines a block's compiled call
site via `inspect.getsource(fn)`, which captures only the function's own
body, not this module's imports or any other module-level helper -- a
module-level import or a shared private helper called from inside `fn`
would work live but raise NameError the moment a graph using it got
compiled to a script. That's also why there's some deliberate small
duplication below (each simulation-shaped block builds its own quantile
table inline) rather than a shared `_quantile_table` helper. This does
mean a compiled script using a stochastic block depends on `modelmaker`
itself being installed wherever that script runs (unlike the rest of the
block library, which only depends on third-party packages) -- a narrower
self-containment guarantee, worth it against duplicating this module's
numerical machinery inline in every function.

Non-dataframe outputs (distribution/dependency/proxy_function/
simulation_result) are plain JSON-shaped dicts, exactly like the existing
"model" port (see blocks/modelling.py) -- inspectable over the API's
generic /value endpoint, no special deserializer, and they travel through
the normal pickle cache like any other Python value.

Deferred (see stochastic-engine-proposal.md S10-S11, out of scope for this
build): the sub-graph fan-out engine primitive (phase 4 -- bootstrap CIs
here use a local numpy resample instead), LSMC/replicating-portfolio proxy
methods, Shapley allocation, and disk-backed raw-path persistence (paths
are an optional plain dataframe output here, gated by `keep_paths`, rather
than a parquet-file handle).
"""

from __future__ import annotations

import polars as pl

from ..metadata_transforms import infer_dtypes, passthrough
from .base import BlockSpec, PortSpec, register_block

# ---------------------------------------------------------------------------
# Distribution fitting
# ---------------------------------------------------------------------------


def fit_distribution(
    df: pl.DataFrame,
    column: str,
    families: list[str] | None = None,
    spliced_tail: bool = False,
    threshold_quantile: float = 0.9,
    body_family: str = "lognorm",
) -> tuple[dict, pl.DataFrame]:
    from modelmaker.stochastic import distributions

    x = df[column].drop_nulls().to_numpy()
    if spliced_tail:
        dist = distributions.fit_spliced_gpd(x, threshold_quantile=threshold_quantile, body_family=body_family)
        table = pl.DataFrame(
            [{"family": "spliced_gpd", "n_exceedances": dist["fit_stats"]["n_exceedances"], "threshold": dist["fit_stats"]["threshold"]}]
        )
    else:
        dist = distributions.fit_distribution(x, families)
        table = pl.DataFrame(
            [{k: c.get(k) for k in ("family", "aic", "bic", "ks_stat", "ks_p", "ad_stat", "error")} for c in dist["candidates"]]
        )
    return dist, table


register_block(
    BlockSpec(
        category="fit_distribution",
        block_type="standard",
        group="stochastic",
        display_name="Fit distribution",
        inputs=[PortSpec("df")],
        outputs=[PortSpec("distribution", type="distribution"), PortSpec("fit_table")],
        fn=fit_distribution,
        metadata_transform=infer_dtypes,
    )
)


def sample_distribution(distribution: dict, n: int = 100_000, seed: int = 0, block_id: str | None = None) -> pl.DataFrame:
    from modelmaker.stochastic import distributions
    from modelmaker.stochastic import seed as seed_mod

    rng = seed_mod.spawn_rng(seed, [block_id or "sample_distribution"])
    return pl.DataFrame({"sample": distributions.sample(distribution, n, rng)})


register_block(
    BlockSpec(
        category="sample_distribution",
        block_type="standard",
        group="stochastic",
        display_name="Sample distribution",
        inputs=[PortSpec("distribution", type="distribution")],
        outputs=[PortSpec("samples")],
        fn=sample_distribution,
        metadata_transform=infer_dtypes,
    )
)


# ---------------------------------------------------------------------------
# Dependency / copulas
# ---------------------------------------------------------------------------


def build_dependency(df: pl.DataFrame, columns: list[str], copula_type: str = "gaussian", dof: float = 5.0, theta: float = 2.0) -> dict:
    import numpy as np

    from modelmaker.stochastic import dependency as dependency_mod

    x = df.select(columns).to_numpy()
    empirical_corr = np.atleast_2d(np.corrcoef(x, rowvar=False))
    repaired = dependency_mod.nearest_psd_correlation(empirical_corr)
    stds = {c: float(np.std(x[:, i], ddof=1)) for i, c in enumerate(columns)}
    return {
        "kind": "dependency",
        "type": copula_type,
        "labels": list(columns),
        "corr": repaired.tolist(),
        "stds": stds,
        "dof": dof,
        "theta": theta,
        "psd_repaired": bool(not np.allclose(repaired, empirical_corr, atol=1e-8)),
    }


register_block(
    BlockSpec(
        category="build_dependency",
        block_type="standard",
        group="stochastic",
        display_name="Build dependency",
        inputs=[PortSpec("df")],
        outputs=[PortSpec("dependency", type="dependency")],
        fn=build_dependency,
        metadata_transform=infer_dtypes,
    )
)


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------


def simulate_op_risk_lda(
    severity_dist: dict,
    n_paths: int = 1_000_000,
    chunk_size: int = 100_000,
    frequency_family: str = "poisson",
    frequency_mean: float = 10.0,
    frequency_dispersion: float = 1.5,
    alpha_levels: list[float] | None = None,
    n_bootstrap: int = 200,
    keep_paths: bool = False,
    seed: int = 0,
    block_id: str | None = None,
) -> tuple[dict, pl.DataFrame, pl.DataFrame]:
    """Compound-Poisson (or negative-binomial) frequency x severity op-risk
    LDA, aggregated by Monte Carlo (proposal S3.3). Draws are generated
    `chunk_size` paths at a time so peak memory never depends on `n_paths`
    -- the per-chunk claim count total is the only thing sized to the
    chunk, and per-path losses accumulate via np.bincount rather than a
    path x claim matrix."""
    import numpy as np

    from modelmaker.stochastic import accumulate, distributions
    from modelmaker.stochastic import seed as seed_mod

    levels = list(alpha_levels) if alpha_levels else [0.95, 0.99, 0.999]
    rng = seed_mod.spawn_rng(seed, [block_id or "simulate_op_risk_lda"])

    chunks: list[np.ndarray] = []
    remaining = n_paths
    while remaining > 0:
        this_chunk = min(chunk_size, remaining)
        if frequency_family == "poisson":
            counts = rng.poisson(frequency_mean, size=this_chunk)
        elif frequency_family == "negbinom":
            p = frequency_dispersion / (frequency_dispersion + frequency_mean)
            counts = rng.negative_binomial(frequency_dispersion, p, size=this_chunk)
        else:
            raise ValueError(f"unknown frequency family: {frequency_family!r} (use poisson or negbinom)")
        total_claims = int(counts.sum())
        path_losses = np.zeros(this_chunk)
        if total_claims:
            severities = distributions.sample(severity_dist, total_claims, rng)
            path_ids = np.repeat(np.arange(this_chunk), counts)
            path_losses = np.bincount(path_ids, weights=severities, minlength=this_chunk)
        chunks.append(path_losses)
        remaining -= this_chunk

    losses = np.concatenate(chunks)
    rm = accumulate.risk_measures(losses, levels)
    ci = {a: accumulate.bootstrap_quantile_ci(losses, a, rng, n_boot=n_bootstrap) for a in levels}
    convergence = accumulate.running_estimate(losses, max(levels))

    result = {
        "kind": "simulation_result",
        "n_paths": n_paths,
        **rm,
        "quantile_ci": {a: list(ci[a]) for a in levels},
        "convergence": convergence,
        "seed_lineage": {"seed": seed, "path": [block_id or "simulate_op_risk_lda"]},
    }
    quantile_table = pl.DataFrame(
        {
            "alpha": levels,
            "var": [rm["quantiles"][a] for a in levels],
            "tvar": [rm["tvar"][a] for a in levels],
            "unexpected_loss": [rm["unexpected_loss"][a] for a in levels],
            "ci_low": [ci[a][0] for a in levels],
            "ci_high": [ci[a][1] for a in levels],
        }
    )
    if keep_paths:
        paths_table = pl.DataFrame({"path_id": np.arange(n_paths), "loss": losses})
    else:
        paths_table = pl.DataFrame({"path_id": pl.Series([], dtype=pl.Int64), "loss": pl.Series([], dtype=pl.Float64)})
    return result, quantile_table, paths_table


register_block(
    BlockSpec(
        category="simulate_op_risk_lda",
        block_type="standard",
        group="stochastic",
        display_name="Simulate op-risk LDA",
        inputs=[PortSpec("severity_dist", type="distribution")],
        outputs=[
            PortSpec("simulation_result", type="simulation_result"),
            PortSpec("quantile_table"),
            PortSpec("paths", required=False),
        ],
        fn=simulate_op_risk_lda,
        metadata_transform=infer_dtypes,
    )
)


def risk_measures(
    df: pl.DataFrame, column: str, alpha_levels: list[float] | None = None, n_bootstrap: int = 200, seed: int = 0
) -> tuple[dict, pl.DataFrame]:
    """Standalone VaR/TVaR + bootstrap-CI block over any dataframe column of
    per-path losses -- the same math simulate_op_risk_lda and
    aggregate_simulation use internally, usable directly on a loss column
    produced elsewhere in the graph."""
    import numpy as np

    from modelmaker.stochastic import accumulate

    levels = list(alpha_levels) if alpha_levels else [0.95, 0.99, 0.999]
    losses = df[column].drop_nulls().to_numpy()
    rng = np.random.default_rng(seed)
    rm = accumulate.risk_measures(losses, levels)
    ci = {a: accumulate.bootstrap_quantile_ci(losses, a, rng, n_boot=n_bootstrap) for a in levels}
    result = {"kind": "simulation_result", "n_paths": len(losses), **rm, "quantile_ci": {a: list(ci[a]) for a in levels}}
    table = pl.DataFrame(
        {
            "alpha": levels,
            "var": [rm["quantiles"][a] for a in levels],
            "tvar": [rm["tvar"][a] for a in levels],
            "unexpected_loss": [rm["unexpected_loss"][a] for a in levels],
            "ci_low": [ci[a][0] for a in levels],
            "ci_high": [ci[a][1] for a in levels],
        }
    )
    return result, table


register_block(
    BlockSpec(
        category="risk_measures",
        block_type="standard",
        group="stochastic",
        display_name="Risk measures (VaR / TVaR)",
        inputs=[PortSpec("df")],
        outputs=[PortSpec("simulation_result", type="simulation_result"), PortSpec("quantile_table")],
        fn=risk_measures,
        metadata_transform=infer_dtypes,
    )
)


def aggregate_simulation(
    components: pl.DataFrame,
    dependency: dict,
    alpha_levels: list[float] | None = None,
    n_bootstrap: int = 200,
    seed: int = 0,
    contributions_method: str = "euler",
) -> tuple[dict, pl.DataFrame, pl.DataFrame]:
    """Combine independently-simulated component loss vectors (one column
    per component in `components`) into a total loss distribution under a
    target dependency structure, via rank reordering (Iman-Conover): each
    component's already-simulated marginal is kept exactly, but reordered
    so the joint ranks match a sample from `dependency`'s copula. Avoids
    re-simulating every component jointly -- the practical technique for
    aggregating risk modules that were (or had to be) simulated separately.

    `contributions_method`: "euler" (default -- ES-conditional-expectation
    contributions, stable at any component count) or "shapley" (exact,
    symmetric, efficient allocation by coalition enumeration -- capped at
    stochastic.accumulate.MAX_SHAPLEY_COMPONENTS components, see S7.3)."""
    import numpy as np

    from modelmaker.stochastic import accumulate
    from modelmaker.stochastic import dependency as dependency_mod

    levels = list(alpha_levels) if alpha_levels else [0.95, 0.99, 0.999]
    names = dependency["labels"]
    missing = [n for n in names if n not in components.columns]
    if missing:
        raise ValueError(f"dependency labels not found as columns in components: {missing}")
    n = components.height
    rng = np.random.default_rng(seed)
    u = dependency_mod.sample_uniforms(dependency, n, rng)
    target_ranks = np.argsort(np.argsort(u, axis=0), axis=0)

    reordered: dict[str, "np.ndarray"] = {}
    for i, name in enumerate(names):
        sorted_values = np.sort(components[name].to_numpy())
        reordered[name] = sorted_values[target_ranks[:, i]]

    total = sum(reordered.values())
    rm = accumulate.risk_measures(total, levels)
    ci = {a: accumulate.bootstrap_quantile_ci(total, a, rng, n_boot=n_bootstrap) for a in levels}
    if contributions_method == "euler":
        contributions = accumulate.euler_contributions(reordered, max(levels))
    elif contributions_method == "shapley":
        contributions = accumulate.shapley_contributions(reordered, max(levels), measure="var")
    else:
        raise ValueError(f"unknown contributions_method: {contributions_method!r} (use euler or shapley)")

    result = {
        "kind": "simulation_result",
        "n_paths": n,
        **rm,
        "quantile_ci": {a: list(ci[a]) for a in levels},
        "contributions": contributions,
    }
    quantile_table = pl.DataFrame(
        {
            "alpha": levels,
            "var": [rm["quantiles"][a] for a in levels],
            "tvar": [rm["tvar"][a] for a in levels],
            "unexpected_loss": [rm["unexpected_loss"][a] for a in levels],
            "ci_low": [ci[a][0] for a in levels],
            "ci_high": [ci[a][1] for a in levels],
        }
    )
    contributions_table = pl.DataFrame({"component": list(contributions), "contribution": list(contributions.values())})
    return result, quantile_table, contributions_table


register_block(
    BlockSpec(
        category="aggregate_simulation",
        block_type="standard",
        group="stochastic",
        display_name="Aggregate simulation (copula)",
        inputs=[PortSpec("components"), PortSpec("dependency", type="dependency")],
        outputs=[
            PortSpec("simulation_result", type="simulation_result"),
            PortSpec("quantile_table"),
            PortSpec("contributions"),
        ],
        fn=aggregate_simulation,
        metadata_transform=infer_dtypes,
    )
)


# ---------------------------------------------------------------------------
# Curve fitting / proxy functions
# ---------------------------------------------------------------------------


def fit_proxy(
    fitting: pl.DataFrame,
    risk_factors: list[str],
    value_col: str | None = None,
    method: str = "polynomial",
    degree: int = 2,
    regressor: str = "ridge",
    alpha: float = 1.0,
    expr: str | None = None,
) -> dict:
    from modelmaker.stochastic import proxy as proxy_mod

    if method == "closed_form":
        if not expr:
            raise ValueError("closed_form proxy requires the 'expr' param")
        return proxy_mod.fit_closed_form(expr, risk_factors)
    if method == "polynomial":
        if not value_col:
            raise ValueError("polynomial proxy requires the 'value_col' param")
        return proxy_mod.fit_polynomial(fitting, risk_factors, value_col, degree=degree, regressor=regressor, alpha=alpha)
    raise ValueError(f"unknown proxy method: {method!r} (use closed_form or polynomial)")


register_block(
    BlockSpec(
        category="fit_proxy",
        block_type="standard",
        group="stochastic",
        display_name="Fit proxy function",
        inputs=[PortSpec("fitting")],
        outputs=[PortSpec("proxy", type="proxy_function")],
        fn=fit_proxy,
        metadata_transform=infer_dtypes,
    )
)


def evaluate_proxy(proxy: dict, scenarios: pl.DataFrame) -> pl.DataFrame:
    from modelmaker.stochastic import proxy as proxy_mod

    values = proxy_mod.evaluate_proxy(proxy, scenarios)
    return scenarios.with_columns(pl.Series("value", values))


register_block(
    BlockSpec(
        category="evaluate_proxy",
        block_type="standard",
        group="stochastic",
        display_name="Evaluate proxy function",
        inputs=[PortSpec("proxy", type="proxy_function"), PortSpec("scenarios")],
        outputs=[PortSpec("valued")],
        fn=evaluate_proxy,
        metadata_transform=passthrough,
    )
)


def validate_proxy(proxy: dict, validation: pl.DataFrame, value_col: str) -> tuple[dict, pl.DataFrame]:
    import numpy as np

    from modelmaker.stochastic import proxy as proxy_mod

    actual = validation[value_col].to_numpy()
    predicted = proxy_mod.evaluate_proxy(proxy, validation)
    diagnostics = {"kind": "scalar_metric", **proxy_mod.validate_proxy(proxy, actual, predicted)}
    error = predicted - actual
    table = validation.with_columns(
        pl.Series("predicted", predicted), pl.Series("error", error), pl.Series("abs_error", np.abs(error))
    )
    return diagnostics, table


register_block(
    BlockSpec(
        category="validate_proxy",
        block_type="standard",
        group="stochastic",
        display_name="Validate proxy function",
        inputs=[PortSpec("proxy", type="proxy_function"), PortSpec("validation")],
        outputs=[PortSpec("diagnostics", type="scalar_metric"), PortSpec("error_table")],
        fn=validate_proxy,
        metadata_transform=passthrough,
    )
)


# ---------------------------------------------------------------------------
# Var-covar aggregation
# ---------------------------------------------------------------------------


def var_covar_aggregate(
    proxy: dict,
    base: pl.DataFrame,
    dependency: dict,
    alpha_levels: list[float] | None = None,
    bump_size: float = 0.01,
    method: str = "normal",
    n_mc: int = 50_000,
    seed: int = 0,
) -> tuple[dict, pl.DataFrame]:
    """Var-covar proxy method (review S3.6 case 5): sensitivities via
    bump-and-reval on `proxy` (itself possibly a curve-fit polynomial --
    see module docstring), aggregated under `dependency`'s correlation
    matrix by one of four methods (see stochastic.var_covar)."""
    import numpy as np

    from modelmaker.stochastic import var_covar as var_covar_mod

    levels = list(alpha_levels) if alpha_levels else [0.95, 0.99, 0.999]
    if base.height != 1:
        raise ValueError(f"'base' must be exactly one row (the base scenario), got {base.height}")
    base_row = base.row(0, named=True)
    risk_factors = proxy["risk_factors"]
    missing = [f for f in risk_factors if f not in dependency["labels"]]
    if missing:
        raise ValueError(f"proxy risk factor(s) not found in dependency labels: {missing}")
    base_values = {f: float(base_row[f]) for f in risk_factors}

    sensitivities = var_covar_mod.bump_and_reval(proxy, base_values, risk_factors, bump_size)
    corr = np.array(dependency["corr"])
    result = var_covar_mod.aggregate(
        sensitivities,
        dependency["stds"],
        corr,
        dependency["labels"],
        levels,
        method=method,
        n_mc=n_mc,
        seed=seed,
        copula=dependency if method == "delta_gamma_copula" else None,
    )
    factor_order = dependency["labels"]
    contributions_table = pl.DataFrame(
        {
            "factor": factor_order,
            "delta": [sensitivities["delta"].get(f, 0.0) for f in factor_order],
            "gamma": [sensitivities["gamma"].get(f, 0.0) for f in factor_order],
            "euler_contribution": [result["euler_contributions"][f] for f in factor_order],
        }
    )
    return result, contributions_table


register_block(
    BlockSpec(
        category="var_covar_aggregate",
        block_type="standard",
        group="stochastic",
        display_name="Var-covar aggregate",
        inputs=[PortSpec("proxy", type="proxy_function"), PortSpec("base"), PortSpec("dependency", type="dependency")],
        outputs=[PortSpec("aggregation_result", type="simulation_result"), PortSpec("contributions")],
        fn=var_covar_aggregate,
        metadata_transform=infer_dtypes,
    )
)


# ---------------------------------------------------------------------------
# Graph fan-out (iterate / collect) -- stochastic-engine-proposal.md S4
# ---------------------------------------------------------------------------
#
# An 'iterate' block and a matching 'collect' block (paired by the
# collect's `iterate_block` param, naming the iterate block's id -- there's
# no dedicated wire for the pairing itself, since a plain param is the same
# convention every other block-execution modifier in this tool already
# uses, e.g. group_by naming a column) bound a fan-out *region*: every
# block on some path between them (see runner.Runner._iterate_region). The
# region is re-run once per iteration, each in its own subprocess (see
# Runner._run_region_iterations/_dispatch_iterations) -- the graph
# fan-out engine primitive, as opposed to a vectorised simulation block's
# in-process chunking (S5). Practical for N ~ 10^2-10^4 (bootstrap CIs,
# scenario expansion); not the tool for N ~ 10^6+, which is what
# simulate_op_risk_lda-style vectorised blocks are for.
#
# Both blocks are ordinary registry blocks with ordinary fn's -- there is
# deliberately no engine magic in `iterate`/`collect` themselves. Run
# outside a fan-out context (e.g. previewing the 'iterate' block on its
# own), `iterate` just computes what iteration 0 looks like; the N-times
# re-execution only happens when a 'collect' block wired to it actually
# runs (see run_block's `block.category == "collect"` branch).
#
# Not yet supported (see stochastic-engine-proposal.md's Implementation
# status): compiling a graph containing this pair to a standalone script
# (compiler.compile_graph raises CompileError -- the seed hierarchy and
# the loop shape both need their own compiled form, deferred); a
# `group_by` block inside the region; an `iterator` other than
# bootstrap_resample/scenario_row (a bare "reseed and rerun" iterator
# doesn't fit this tool's params-are-literal-design-time-config model
# without also inventing a way to wire a scalar into a param -- see the
# proposal file for why that was scoped out).


def iterate(
    df: pl.DataFrame,
    n_iterations: int = 100,
    iterator: str = "bootstrap_resample",
    seed: int = 0,
    _iteration_index: int | None = None,
    block_id: str | None = None,
) -> pl.DataFrame:
    from modelmaker.stochastic import seed as seed_mod

    i = _iteration_index if _iteration_index is not None else 0
    rng = seed_mod.spawn_rng(seed, [block_id or "iterate", f"iter_{i}"])
    if iterator == "bootstrap_resample":
        idx = rng.integers(0, df.height, size=df.height)
        return df[idx]
    if iterator == "scenario_row":
        if i >= df.height:
            raise ValueError(f"iteration index {i} is out of range for {df.height} scenario row(s) -- set n_iterations <= the scenario set's row count")
        return df[i : i + 1]
    raise ValueError(f"unknown iterator: {iterator!r} (use bootstrap_resample or scenario_row)")


register_block(
    BlockSpec(
        category="iterate",
        block_type="standard",
        group="stochastic",
        display_name="Iterate (fan-out)",
        inputs=[PortSpec("df")],
        outputs=[PortSpec("out")],
        fn=iterate,
        metadata_transform=passthrough,
    )
)


def collect(
    value: pl.DataFrame,
    reducer: str = "concat",
    value_col: str | None = None,
    alpha_levels: list[float] | None = None,
    n_bootstrap: int = 200,
    seed: int = 0,
    iterate_block: str = "",
) -> tuple[pl.DataFrame, dict | None]:
    """`value` is every iteration's collected value already concatenated,
    with an `iteration` column prepended (see
    Runner._run_region_iterations) -- this fn itself has no iteration
    logic, it's an ordinary reducer over an ordinary dataframe.
    `iterate_block` isn't read here (the Runner reads it directly off
    block.params to find the paired 'iterate' block before this ever
    runs) -- accepted only so it doesn't land as an unexpected kwarg."""
    if reducer == "concat":
        return value, None
    if reducer == "risk_measures":
        import numpy as np

        from modelmaker.stochastic import accumulate

        if not value_col:
            raise ValueError("reducer='risk_measures' requires the 'value_col' param")
        # [0.05, 0.5, 0.95] (median + a 90% band), not simulate_op_risk_lda's
        # deep-tail defaults -- this reduces a *bootstrap estimate*'s spread
        # across iterations, not a loss distribution's tail.
        levels = list(alpha_levels) if alpha_levels else [0.05, 0.5, 0.95]
        estimates = value[value_col].drop_nulls().to_numpy()
        rng = np.random.default_rng(seed)
        rm = accumulate.risk_measures(estimates, levels)
        ci = {a: accumulate.bootstrap_quantile_ci(estimates, a, rng, n_boot=n_bootstrap) for a in levels}
        result = {"kind": "simulation_result", "n_paths": len(estimates), **rm, "quantile_ci": {a: list(ci[a]) for a in levels}}
        return value, result
    raise ValueError(f"unknown reducer: {reducer!r} (use concat or risk_measures)")


register_block(
    BlockSpec(
        category="collect",
        block_type="standard",
        group="stochastic",
        display_name="Collect (fan-out)",
        inputs=[PortSpec("value")],
        outputs=[PortSpec("table"), PortSpec("simulation_result", type="simulation_result", required=False)],
        fn=collect,
        metadata_transform=infer_dtypes,
    )
)
