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

BLOCKS = {
    "fit_distribution": {
        "display_name": "Fit distribution",
        "version": 1,
        "block_type": "standard",
        "group": "stochastic",
        "tags": ["distributions"],
        "inputs": [{"name": "df"}],
        "outputs": [{"name": "distribution", "type": "distribution"}, {"name": "fit_table", "aggregate": True}],
        "form": [{"key": "column", "label": "Column to fit", "kind": "column"}],
        "meta": {"inherit": "none"},
        "hints": "fit_distribution",
    },
    "sample_distribution": {
        "display_name": "Sample distribution",
        "version": 1,
        "block_type": "standard",
        "group": "stochastic",
        "tags": ["distributions"],
        "inputs": [{"name": "distribution", "type": "distribution"}],
        "outputs": [{"name": "samples"}],
        "form": [
            {"key": "n", "label": "Number of samples", "kind": "number"},
            {"key": "seed", "label": "Random seed", "kind": "number"},
        ],
        "meta": {"inherit": "none"},
    },
    "build_dependency": {
        "display_name": "Build dependency",
        "version": 1,
        "block_type": "standard",
        "group": "stochastic",
        "tags": ["capital_simulation", "distributions"],
        "inputs": [{"name": "df"}],
        "outputs": [{"name": "dependency", "type": "dependency"}],
        "form": [
            {"key": "columns", "label": "Columns", "kind": "columns"},
            {
                "key": "copula_type",
                "label": "Copula",
                "kind": "select",
                "options": ["gaussian", "t", "clayton", "gumbel"],
            },
            {"key": "dof", "label": "Degrees of freedom (t copula)", "kind": "number"},
            {"key": "theta", "label": "Theta (Clayton / Gumbel)", "kind": "number", "step": 0.1},
        ],
        "meta": {"inherit": "none"},
    },
    "simulate_op_risk_lda": {
        "display_name": "Simulate op-risk LDA",
        "version": 1,
        "block_type": "standard",
        "group": "stochastic",
        "tags": ["capital_simulation"],
        "inputs": [{"name": "severity_dist", "type": "distribution"}],
        "outputs": [
            {"name": "simulation_result", "type": "simulation_result"},
            {"name": "quantile_table", "aggregate": True},
            {"name": "paths", "required": False},
        ],
        "form": [
            {"key": "n_paths", "label": "Number of paths", "kind": "number"},
            {"key": "chunk_size", "label": "Chunk size", "kind": "number"},
            {
                "key": "frequency_family",
                "label": "Frequency family",
                "kind": "select",
                "options": ["poisson", "negbinom"],
            },
            {"key": "frequency_mean", "label": "Frequency mean", "kind": "number", "step": 0.1},
            {
                "key": "frequency_dispersion",
                "label": "Frequency dispersion (negbinom)",
                "kind": "number",
                "step": 0.1,
            },
            {"key": "n_bootstrap", "label": "Bootstrap replicates", "kind": "number"},
            {"key": "seed", "label": "Random seed", "kind": "number"},
        ],
        "meta": {"inherit": "none"},
    },
    "risk_measures": {
        "display_name": "Risk measures (VaR / TVaR)",
        "version": 1,
        "block_type": "standard",
        "group": "stochastic",
        "tags": ["capital_simulation", "distributions"],
        "inputs": [{"name": "df"}],
        "outputs": [
            {"name": "simulation_result", "type": "simulation_result"},
            {"name": "quantile_table", "aggregate": True},
        ],
        "form": [
            {"key": "column", "label": "Loss column", "kind": "column"},
            {"key": "n_bootstrap", "label": "Bootstrap replicates", "kind": "number"},
            {"key": "seed", "label": "Random seed", "kind": "number"},
        ],
        "meta": {"inherit": "none"},
    },
    "aggregate_simulation": {
        "display_name": "Aggregate simulation (copula)",
        "version": 1,
        "block_type": "standard",
        "group": "stochastic",
        "tags": ["capital_simulation"],
        "inputs": [{"name": "components"}, {"name": "dependency", "type": "dependency"}],
        "outputs": [
            {"name": "simulation_result", "type": "simulation_result"},
            {"name": "quantile_table", "aggregate": True},
            {"name": "contributions", "aggregate": True},
        ],
        "form": [
            {"key": "n_bootstrap", "label": "Bootstrap replicates", "kind": "number"},
            {"key": "seed", "label": "Random seed", "kind": "number"},
        ],
        "meta": {"inherit": "none"},
    },
    "asrf_economic_capital": {
        "display_name": "ASRF economic capital (closed form)",
        "version": 1,
        "block_type": "standard",
        "group": "stochastic",
        "tags": ["capital_simulation"],
        "inputs": [{"name": "df"}],
        "outputs": [{"name": "simulation_result", "type": "simulation_result"}, {"name": "detail"}],
        "form": [
            {"key": "pd_col", "label": "PD column", "kind": "column"},
            {"key": "lgd_col", "label": "LGD column", "kind": "column"},
            {"key": "ead_col", "label": "EAD column", "kind": "column"},
            {"key": "confidence", "label": "Confidence level", "kind": "number", "step": 0.001},
        ],
    },
    "simulate_credit_portfolio": {
        "display_name": "Simulate credit portfolio (multi-factor)",
        "version": 1,
        "block_type": "standard",
        "group": "stochastic",
        "tags": ["capital_simulation"],
        "inputs": [{"name": "df"}, {"name": "dependency", "type": "dependency"}],
        "outputs": [
            {"name": "simulation_result", "type": "simulation_result"},
            {"name": "quantile_table", "aggregate": True},
            {"name": "contributions"},
        ],
        "form": [
            {"key": "pd_col", "label": "PD column", "kind": "column"},
            {"key": "lgd_col", "label": "LGD column", "kind": "column"},
            {"key": "ead_col", "label": "EAD column", "kind": "column"},
            {
                "key": "sector_col",
                "label": "Sector column (must match the dependency labels)",
                "kind": "column",
            },
            {"key": "name_col", "label": "Segment name column (optional)", "kind": "column"},
            {"key": "n_paths", "label": "Number of paths", "kind": "number"},
            {"key": "chunk_size", "label": "Chunk size", "kind": "number"},
            {"key": "n_bootstrap", "label": "Bootstrap replicates", "kind": "number"},
            {"key": "seed", "label": "Random seed", "kind": "number"},
        ],
        "meta": {"inherit": "none"},
    },
    "fit_proxy": {
        "display_name": "Fit proxy function",
        "version": 1,
        "block_type": "standard",
        "group": "stochastic",
        "tags": ["proxy_models"],
        "inputs": [{"name": "fitting"}],
        "outputs": [{"name": "proxy", "type": "proxy_function"}],
        "form": [
            {"key": "value_col", "label": "Value column (polynomial)", "kind": "column"},
            {
                "key": "method",
                "label": "Method",
                "kind": "select",
                "options": ["closed_form", "polynomial"],
            },
            {
                "key": "expr",
                "label": "Expression (closed_form)",
                "kind": "text",
                "placeholder": "x + 0.5 * y",
            },
            {"key": "degree", "label": "Degree (polynomial)", "kind": "number"},
            {
                "key": "regressor",
                "label": "Regressor (polynomial)",
                "kind": "select",
                "options": ["ridge", "lasso", "ols"],
            },
            {
                "key": "alpha",
                "label": "Regularization (ridge / lasso)",
                "kind": "number",
                "step": 0.1,
            },
        ],
        "meta": {"inherit": "none"},
    },
    "evaluate_proxy": {
        "display_name": "Evaluate proxy function",
        "version": 1,
        "block_type": "standard",
        "group": "stochastic",
        "tags": ["proxy_models"],
        "inputs": [{"name": "proxy", "type": "proxy_function"}, {"name": "scenarios"}],
        "outputs": [{"name": "valued"}],
    },
    "validate_proxy": {
        "display_name": "Validate proxy function",
        "version": 1,
        "block_type": "standard",
        "group": "stochastic",
        "tags": ["proxy_models"],
        "inputs": [{"name": "proxy", "type": "proxy_function"}, {"name": "validation"}],
        "outputs": [{"name": "diagnostics", "type": "scalar_metric"}, {"name": "error_table"}],
        "form": [{"key": "value_col", "label": "Actual value column", "kind": "column"}],
        "hints": "validate_proxy",
    },
    "var_covar_aggregate": {
        "display_name": "Var-covar aggregate",
        "version": 1,
        "block_type": "standard",
        "group": "stochastic",
        "tags": ["capital_simulation", "proxy_models"],
        "inputs": [
            {"name": "proxy", "type": "proxy_function"},
            {"name": "base"},
            {"name": "dependency", "type": "dependency"},
        ],
        "outputs": [{"name": "aggregation_result", "type": "simulation_result"}, {"name": "contributions"}],
        "form": [
            {
                "key": "method",
                "label": "Method",
                "kind": "select",
                "options": ["normal", "cornish_fisher", "moment_matching", "delta_gamma_copula"],
            },
            {"key": "bump_size", "label": "Bump size", "kind": "number", "step": 0.001},
            {"key": "n_mc", "label": "Monte Carlo draws (non-normal methods)", "kind": "number"},
            {"key": "seed", "label": "Random seed", "kind": "number"},
        ],
        "meta": {"inherit": "none"},
    },
    "iterate": {
        "display_name": "Iterate (fan-out)",
        "version": 1,
        "block_type": "standard",
        "group": "stochastic",
        "inputs": [{"name": "df"}],
        "outputs": [{"name": "out"}],
        "form": [
            {"key": "n_iterations", "label": "Iterations", "kind": "number"},
            {
                "key": "iterator",
                "label": "Iterator",
                "kind": "select",
                "options": ["bootstrap_resample", "scenario_row"],
            },
            {"key": "seed", "label": "Random seed", "kind": "number"},
        ],
    },
    "collect": {
        "display_name": "Collect (fan-out)",
        "version": 1,
        "block_type": "standard",
        "group": "stochastic",
        "inputs": [{"name": "value"}],
        "outputs": [
            {"name": "table"},
            {"name": "simulation_result", "type": "simulation_result", "required": False},
        ],
        "form": [
            {
                "key": "iterate_block",
                "label": "Paired 'iterate' block id",
                "kind": "text",
                "placeholder": "b_042",
            },
            {
                "key": "reducer",
                "label": "Reducer",
                "kind": "select",
                "options": ["concat", "risk_measures"],
            },
            {"key": "value_col", "label": "Value column (risk_measures)", "kind": "column"},
            {
                "key": "n_bootstrap",
                "label": "Bootstrap replicates (risk_measures)",
                "kind": "number",
            },
            {"key": "seed", "label": "Random seed (risk_measures)", "kind": "number"},
        ],
        "meta": {"inherit": "none"},
    },
}


def fit_distribution(
    df: pl.DataFrame,
    column: str,
    families: list[str] | None = None,
    spliced_tail: bool = False,
    threshold_quantile: float = 0.9,
    body_family: str = "lognorm",
) -> tuple[dict, pl.DataFrame]:
    """Fits a parametric distribution to the values of `column` in the `df`
    input (nulls dropped) by maximum likelihood. Outputs: `distribution`, a
    JSON-shaped dict (family, params, fit_stats) consumed by
    sample_distribution / simulate_op_risk_lda, and `fit_table`, a
    dataframe comparing candidates.

    Default mode: tries each of `families` (default ["norm", "lognorm",
    "gamma", "expon"]; also accepted: "genpareto", "t") and picks the
    lowest AIC; `fit_table` has one row per candidate (aic, bic, ks_stat,
    ks_p, ad_stat, or an error for a family that couldn't fit). Raises if
    no family fits. With `spliced_tail=True`, `families` is ignored and a
    spliced body-tail model is fitted instead: `body_family` (default
    "lognorm") below the `threshold_quantile` quantile of the data
    (default 0.9), a generalized Pareto tail above it -- needs >= 10
    observations above the threshold. Use this for heavy-tailed loss
    severities.
    """
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


def sample_distribution(distribution: dict, n: int = 100_000, seed: int = 0, block_id: str | None = None) -> pl.DataFrame:
    """Draws `n` random values (default 100,000) from the `distribution`
    input (a fit_distribution output) and emits them on the `samples` port
    as a one-column dataframe, column "sample". `seed` (default 0) makes
    the draw reproducible; the RNG stream is keyed on (seed, block id), so
    two sample blocks with the same seed still draw independent streams.
    `block_id` is engine-injected; leave it unset."""
    from modelmaker.stochastic import distributions
    from modelmaker.stochastic import seed as seed_mod

    rng = seed_mod.spawn_rng(seed, [block_id or "sample_distribution"])
    return pl.DataFrame({"sample": distributions.sample(distribution, n, rng)})


def build_dependency(df: pl.DataFrame, columns: list[str], copula_type: str = "gaussian", dof: float = 5.0, theta: float = 2.0) -> dict:
    """Estimates a dependency structure between the numeric `columns` of
    the `df` input (e.g. historical risk-factor moves or per-component
    losses, one row per observation) and emits it on the `dependency` port
    as a JSON-shaped dict: the Pearson correlation matrix (projected to the
    nearest positive semi-definite matrix if needed -- `psd_repaired`
    flags that), each column's sample std dev, and the copula settings.
    Consumed by aggregate_simulation and var_covar_aggregate, which match
    it to their own inputs by column label. `copula_type` is "gaussian"
    (default), "t" (uses `dof`, degrees of freedom, default 5), "clayton"
    or "gumbel" (use `theta`, default 2); it isn't validated here -- an
    unknown value only fails when a consumer samples from it. Columns
    should be null-free."""
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


def asrf_economic_capital(
    df: pl.DataFrame,
    pd_col: str,
    lgd_col: str,
    ead_col: str,
    confidence: float = 0.999,
    correlation: float | str = "basel_corporate",
) -> tuple[dict, pl.DataFrame]:
    """`correlation` is either a fixed asset correlation applied to every
    row, or the string "basel_corporate" for the Basel II IRB regulatory
    PD-dependent formula (see stochastic.credit.basel_corporate_correlation)."""
    import numpy as np

    from modelmaker.stochastic import credit

    pd_arr = df[pd_col].to_numpy().astype(float)
    lgd_arr = df[lgd_col].to_numpy().astype(float)
    ead_arr = df[ead_col].to_numpy().astype(float)
    if correlation == "basel_corporate":
        corr_arr = credit.basel_corporate_correlation(pd_arr)
    else:
        corr_arr = np.full_like(pd_arr, float(correlation))

    summary, ec_per_obligor, el_per_obligor = credit.asrf_economic_capital(pd_arr, lgd_arr, ead_arr, corr_arr, confidence)
    detail = df.with_columns(
        pl.Series("correlation", corr_arr),
        pl.Series("expected_loss", el_per_obligor),
        pl.Series("economic_capital", ec_per_obligor),
    )
    return summary, detail


def simulate_credit_portfolio(
    df: pl.DataFrame,
    dependency: dict,
    pd_col: str,
    lgd_col: str,
    ead_col: str,
    sector_col: str,
    name_col: str | None = None,
    correlation: float | str = "basel_corporate",
    n_paths: int = 100_000,
    chunk_size: int = 20_000,
    alpha_levels: list[float] | None = None,
    n_bootstrap: int = 200,
    compute_contributions: bool = True,
    seed: int = 0,
    block_id: str | None = None,
) -> tuple[dict, pl.DataFrame, pl.DataFrame]:
    """Multi-factor obligor/segment-level credit EC by Monte Carlo (review
    S3.2): `dependency` (from build_dependency) carries the *sector-level*
    correlation matrix, its `labels` naming the sectors `sector_col`'s
    values must match; each row of `df` is a segment (an individual
    obligor for a small commercial book, or a homogeneous pool/rating
    grade for a retail one -- see stochastic.credit's module docstring for
    why this scopes to that granularity rather than true hundreds-of-
    thousands-of-obligors scale). Draws are chunked over paths, not
    segments, so peak memory is `chunk_size x n_segments`, not
    `n_paths x n_segments` -- fine at the segment counts this targets
    (tens to a few hundred), the same discipline simulate_op_risk_lda uses
    for its own chunking.

    With a single sector and matching correlation assumptions, this
    converges (as n_paths grows) to asrf_economic_capital's closed-form
    capital_requirement exactly -- see stochastic-engine-proposal.md and
    tests/test_stochastic_credit.py's cross-validation test, the reason
    the closed-form block was built first."""
    import numpy as np

    from modelmaker.stochastic import accumulate, credit
    from modelmaker.stochastic import dependency as dependency_mod
    from modelmaker.stochastic import seed as seed_mod

    levels = list(alpha_levels) if alpha_levels else [0.95, 0.99, 0.999]
    pd_arr = df[pd_col].to_numpy().astype(float)
    lgd_arr = df[lgd_col].to_numpy().astype(float)
    ead_arr = df[ead_col].to_numpy().astype(float)
    sectors = df[sector_col].to_list()
    names = df[name_col].to_list() if name_col else [f"segment_{i}" for i in range(len(pd_arr))]

    sector_labels = dependency["labels"]
    missing = sorted(set(sectors) - set(sector_labels))
    if missing:
        raise ValueError(f"sector(s) {missing} not found in dependency labels {sector_labels}")
    sector_index = np.array([sector_labels.index(s) for s in sectors])
    corr_matrix = np.array(dependency["corr"])

    if correlation == "basel_corporate":
        corr_arr = credit.basel_corporate_correlation(pd_arr)
    else:
        corr_arr = np.full_like(pd_arr, float(correlation))

    rng = seed_mod.spawn_rng(seed, [block_id or "simulate_credit_portfolio"])
    portfolio_chunks: list[np.ndarray] = []
    segment_chunks: list[np.ndarray] = []
    remaining = n_paths
    while remaining > 0:
        this_chunk = min(chunk_size, remaining)
        factors = dependency_mod.sample_correlated_normal(corr_matrix, this_chunk, rng)
        segment_losses = credit.multi_factor_conditional_mean_losses(pd_arr, lgd_arr, ead_arr, corr_arr, sector_index, factors)
        portfolio_chunks.append(segment_losses.sum(axis=1))
        if compute_contributions:
            segment_chunks.append(segment_losses)
        remaining -= this_chunk

    losses = np.concatenate(portfolio_chunks)
    rm = accumulate.risk_measures(losses, levels)
    ci = {a: accumulate.bootstrap_quantile_ci(losses, a, rng, n_boot=n_bootstrap) for a in levels}
    convergence = accumulate.running_estimate(losses, max(levels))

    result = {
        "kind": "simulation_result",
        "n_paths": n_paths,
        "n_segments": len(pd_arr),
        **rm,
        "quantile_ci": {a: list(ci[a]) for a in levels},
        "convergence": convergence,
        "economic_capital": rm["quantiles"][max(levels)] - rm["mean"],
        "seed_lineage": {"seed": seed, "path": [block_id or "simulate_credit_portfolio"]},
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

    if compute_contributions:
        segment_matrix = np.concatenate(segment_chunks, axis=0)
        component_losses = {names[j]: segment_matrix[:, j] for j in range(len(pd_arr))}
        contributions = accumulate.euler_contributions(component_losses, max(levels))
        contributions_table = pl.DataFrame({"segment": list(contributions), "euler_contribution": list(contributions.values())})
    else:
        contributions_table = pl.DataFrame({"segment": pl.Series([], dtype=pl.Utf8), "euler_contribution": pl.Series([], dtype=pl.Float64)})

    return result, quantile_table, contributions_table


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
    """Builds a proxy (surrogate valuation) function of the `risk_factors`
    columns, emitted on the `proxy` port as a JSON-shaped dict that
    evaluate_proxy, validate_proxy and var_covar_aggregate consume.
    `method`:
      - "polynomial" (default): fits a polynomial of `degree` (default 2)
        in the risk factors to `value_col` (required) over the `fitting`
        input's rows, using `regressor` "ridge" (default), "lasso" or
        "ols"; `alpha` (default 1.0) is the ridge/lasso penalty. Needs
        more fitting rows than risk factors. Records in-sample R^2.
      - "closed_form": no fitting; `expr` (required) is a Polars SQL
        expression over the risk-factor columns, e.g. "notional * rate",
        evaluated directly per scenario. The `fitting` port must still be
        wired, but its data is unused.
    """
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


def evaluate_proxy(proxy: dict, scenarios: pl.DataFrame) -> pl.DataFrame:
    """Applies the `proxy` input (a fit_proxy output) to every row of the
    `scenarios` input, emitting the scenarios plus a new "value" column on
    the `valued` port. `scenarios` must contain every risk-factor column
    the proxy was built on (or that its closed_form expression uses). No
    params."""
    from modelmaker.stochastic import proxy as proxy_mod

    values = proxy_mod.evaluate_proxy(proxy, scenarios)
    return scenarios.with_columns(pl.Series("value", values))


def validate_proxy(proxy: dict, validation: pl.DataFrame, value_col: str) -> tuple[dict, pl.DataFrame]:
    """Out-of-sample check of the `proxy` input against true values: the
    `validation` input holds scenarios (risk-factor columns) not used for
    fitting, with their full-revaluation result in `value_col` (required).
    Outputs: `diagnostics`, a scalar_metric dict (out_of_sample_r2, mae,
    rmse, max_abs_error, and worst_indices -- row positions of the 10
    largest absolute errors), and `error_table`, the validation rows plus
    "predicted", "error" (predicted - actual) and "abs_error" columns."""
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


def iterate(
    df: pl.DataFrame,
    n_iterations: int = 100,
    iterator: str = "bootstrap_resample",
    seed: int = 0,
    _iteration_index: int | None = None,
    block_id: str | None = None,
) -> pl.DataFrame:
    """Opens a fan-out region: every block between this one and a
    'collect' block (whose `iterate_block` param names this block's id)
    is re-run once per iteration, and collect gathers the results. Emits
    this iteration's slice of the `df` input on `out`. `iterator`:
    "bootstrap_resample" (default; `df.height` rows resampled with
    replacement, a fresh draw per iteration) or "scenario_row" (iteration
    i emits only row i -- set `n_iterations` <= the row count).
    `n_iterations` (default 100, max 10,000) is read by the engine, not by
    this function. `seed` (default 0) makes the resamples reproducible.
    Run on its own (no collect), it just emits iteration 0.
    `_iteration_index` and `block_id` are engine-injected; leave them
    unset. The region may not contain input blocks or group_by blocks,
    and graphs using iterate/collect can't yet be compiled to a script."""
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
