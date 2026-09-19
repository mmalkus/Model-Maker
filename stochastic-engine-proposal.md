# Stochastic engine — build proposal

Concrete proposal for the item tracked as "stochastic engine" in
`model-developer-review.md` §3/§7.1/§7.2. That document maps the *option
space*; this one commits to a design, sequenced so each phase is shippable
and useful on its own, and is written against the codebase as it stands
today (`packet.py`, `cache.py`, `runner.py`, `blocks/base.py`).

Scope for this proposal, per the task: the simulation engine itself,
curve fitting (the proxy-function / scenario-valuation layer), and
aggregation with a var-covar path. Distribution/copula catalogue depth and
the insurance- and credit-specific blocks on top of the spine are follow-on
work once the spine lands.

## Implementation status

Phases 1-3 and the core of phase 5 (§10) are built, on
`modelmaker/stochastic/` (seed, distributions, dependency, accumulate,
proxy, var_covar) and `modelmaker/blocks/stochastic.py`: distribution
fitting incl. a spliced body-GPD tail, hand-rolled Gaussian/t/Clayton/
Gumbel copulas with PSD repair, a chunked compound-Poisson/negative-binomial
op-risk LDA simulation block, the closed-form/polynomial curve-fitting
proxy family with a validation block, var-covar aggregation across all four
methods (normal/Cornish-Fisher/moment-matching/delta-gamma-copula) with
Euler contributions, and copula-based aggregation of independently
simulated components via Iman-Conover rank reordering. New port types
(`distribution`/`dependency`/`proxy_function`/`simulation_result`) follow
the existing "model" port convention (a plain JSON-shaped dict) end to end,
including on the frontend (`PORT_BADGE`, palette group, `PARAM_SPECS`
entries) — see `tests/test_stochastic_core.py`,
`tests/test_stochastic_blocks.py`, and `tests/test_stochastic_runner.py`.

**Deliberately deferred**, consistent with §10's own sequencing (each is
called out at the point above where it would bite):
- **Phase 4, the graph fan-out engine primitive** (`IterateBlock`/
  `CollectBlock`, §4) — the one change to the core run/cache-key/compiler
  machinery, correspondingly the highest blast-radius one to get wrong.
  Bootstrap CIs here use a local numpy resample (§6) instead of fan-out.
- LSMC and replicating-portfolio proxy methods (§8) — increments on the
  `ProxyFunctionPacket` interface once a real nested-simulation use case
  (CVA/XVA, insurance guarantees) pulls for them.
- Shapley allocation (§7.3) — Euler/ES contributions ship; Shapley is a
  follow-on for the risk-module level.
- Disk-backed raw-path persistence (§9) — `simulate_op_risk_lda`'s
  `keep_paths` param emits paths as a plain dataframe output instead of a
  parquet-file handle; fine at today's scale, worth revisiting if raw-path
  retention becomes a common request at N well beyond 10⁶.
- Quasi-MC (Sobol) and importance sampling (§3.1(7)) — antithetic/
  stratified variance reduction wasn't built either; not yet needed at the
  path counts the shipped blocks target.

One correctness note for anyone extending this: `runner.RunPlan.basis`
now folds a registry block's own id into its cache-key basis whenever its
`fn` accepts a `block_id` param (previously that injection was output-block
-only and never affected the key) — needed because a stochastic block's
output genuinely depends on its identity via `stochastic.seed.spawn_rng`,
not just on category/params/upstream. Any new block that accepts
`block_id` inherits this automatically; nothing else changes.

---

## 1. Design principles

1. **One engine change, not one per use case.** Sub-graph fan-out serves
   bootstrap CIs, k-fold CV, IFRS 9 scenario expansion, and Monte Carlo
   alike. Build it once against an abstract "iterate this input, collect
   these outputs" contract.
2. **Two execution shapes stay separate**, because they have opposite
   performance profiles (review §3.5):
   - **Graph fan-out** — re-run a sub-graph N ≈ 10²–10⁴ times with a
     varying input. Each iteration pays graph/process overhead, so N is
     bounded. This is a genuine `Runner` engine feature.
   - **Vectorised simulation** — one block draws an `(N, k)` NumPy array
     with N ≥ 10⁶ internally. This is *not* an engine feature — it's an
     ordinary block that happens to use NumPy instead of Polars, executed
     through the existing process-isolated `run_block`. No new engine
     machinery, just new block-authoring conventions (§4) and a caching
     rule for large outputs (§7).

   Building fan-out to also cover the N ≥ 10⁶ case would mean either
   accepting per-iteration graph overhead a million times, or building a
   second, secretly-different fast path underneath a fan-out UI — worse
   than having two honestly different mechanisms.
3. **Reuse what the runner already has.** Process isolation via
   `MP_CONTEXT = multiprocessing.get_context("spawn")`, the `max_workers`
   ceiling, and the `group_by` fan-out (`_dispatch`/`_combine_group_results`
   in `runner.py`) are exactly the mechanics graph fan-out needs — one
   block already runs itself once per group value, in parallel, isolated,
   and merges results. Sub-graph fan-out generalises that from "one block,
   split by column value" to "a subgraph, split by iteration index."
4. **Determinism is a correctness requirement, not a nice-to-have.** Every
   stochastic block takes its randomness from a seed derived
   deterministically from (project seed, block id, iteration index) —
   never from ambient `numpy.random` global state — so results are
   identical regardless of worker count or scheduling order.
5. **Don't cache what you can't afford to hash-and-stash.** The existing
   `CacheStore` is lineage-keyed (`Runner.compute_key`), not data-hash-keyed
   — cheap regardless of DataFrame size — but it still pickles the full
   `outputs` dict to disk. A million-path result needs a different
   materialisation rule (§7), or the cache design that makes the tool
   pleasant today becomes the thing that falls over here.

---

## 2. New packet types (`packet.py`)

Today `DataFramePacket` is the only packet shape; `PortType` (`blocks/base.py`)
already anticipates more (`"model"`, `"scalar_metric"`, `"any"` exist but
nothing typed flows over them yet as first-class citizens with their own
metadata/preview/cache story). Add:

```python
@dataclass
class DistributionPacket:
    family: str                        # "norm", "genpareto", "spliced", ...
    params: dict[str, float]            # fitted parameters
    fit_method: str                     # "mle" | "mom"
    fit_stats: dict[str, float]         # AIC, BIC, KS stat/p, AD stat/p
    domain: tuple[float, float] | None  # support, for spliced/GPD tails
    lineage: list[str] = field(default_factory=list)

    def sample(self, n: int, rng: np.random.Generator) -> np.ndarray: ...
    def ppf(self, q: np.ndarray) -> np.ndarray: ...          # for VaR/TVaR
    def cdf(self, x: np.ndarray) -> np.ndarray: ...


@dataclass
class ScenarioSetPacket:
    """A scenario set is not 'a dataframe I hope is the right one' --
    see review §3.6. Carries the metadata that turns 'valued under the
    wrong measure' into a wire-validity check instead of a review finding."""
    values: pl.DataFrame                # rows = scenarios, cols = risk factors
    measure: Literal["real_world", "risk_neutral"]
    horizon: float
    purpose: Literal["fitting", "validation", "capital"]
    seed: int | None
    generator_ref: str | None           # what produced/calibrated this set
    weights: np.ndarray | None = None   # None = equal weight (pure MC);
                                         # set for macro-scenario N=3 style use
    lineage: list[str] = field(default_factory=list)


@dataclass
class ProxyFunctionPacket:
    """Common contract for every scenario-valuation escape in review §3.6
    (closed-form / curve fit / LSMC / replicating portfolio / var-covar).
    A block that fits one of these can be swapped for a block that fits
    another without touching anything downstream that just calls evaluate()."""
    method: str                         # "closed_form" | "curve_fit" | "var_covar" | ...
    payload: Any                        # method-specific: coefficients, sensitivities, ...
    risk_factors: list[str]             # names, in the order evaluate() expects
    fit_diagnostics: dict[str, Any]     # R^2 / tail error / worst-case scenarios, per §3.6
    lineage: list[str] = field(default_factory=list)

    def evaluate(self, scenarios: pl.DataFrame) -> np.ndarray: ...


@dataclass
class SimulationResultPacket:
    """Distilled by default -- see §7. `paths_handle` is a lazy reference
    to persisted raw paths, not the array itself."""
    n_paths: int
    quantiles: dict[float, float]       # e.g. {0.995: ..., 0.999: ...}
    quantile_ci: dict[float, tuple[float, float]]  # bootstrap CI per quantile, §6
    moments: dict[str, float]           # mean, std, skew, kurtosis
    convergence: dict[str, Any]         # running-estimate series, se(quantile)
    contributions: pl.DataFrame | None  # component -> risk contribution, §5
    paths_handle: PathsHandle | None    # None unless raw paths were persisted
    seed_lineage: SeedLineage
    lineage: list[str] = field(default_factory=list)
```

`PortType` gains `"distribution"`, `"scenario_set"`, `"proxy_function"`,
`"simulation_result"` alongside the existing five. Each needs its own
inspector preview (PPF/CDF plot for a distribution; quantile table +
convergence plot for a simulation result) — same pattern as today's
dataframe preview, new renderer per type.

---

## 3. Seed hierarchy

```python
@dataclass
class SeedLineage:
    project_seed: int
    path: list[str]          # e.g. ["b_017", "iter_00042"]

    def spawn_key(self) -> tuple[int, ...]:
        return (self.project_seed, *(zlib.crc32(p.encode()) for p in self.path))
```

Every stochastic block resolves its `rng` via
`np.random.SeedSequence(entropy=project_seed, spawn_key=...).generate_state(...)`
→ `np.random.Generator(np.random.PCG64(seed_sequence))`, keyed on
`(project_seed, block_id, iteration_index)`, **never** on worker PID, wall
clock, or dict/set iteration order. `project_seed` lives in project JSON
next to the graph (versioned with it, per §8 of the review). This is what
makes graph fan-out and vectorised simulation *both* reproducible under
arbitrary parallelism — the seed derivation doesn't know or care how many
workers ran it.

`SeedLineage` travels inside `SimulationResultPacket` and gets written into
the compiled script's provenance header (review §7 compilation concern) —
a capital number without a reproducible seed path is an audit finding, so
the seed derivation has to compile out, not just run out.

---

## 4. Graph fan-out — the engine primitive (`runner.py`)

New graph construct: an **iterate/collect pair**, not a canvas loop (per
review §3.5's UX note — a container that visually encloses the iterated
sub-graph, matching how `group_by` already reads on a single block).

- `IterateBlock`: params `{n_iterations, iterator: "seed" | "scenario_row" |
  "bootstrap_resample" | "fold"}`. Declares the sub-graph between it and its
  matching `CollectBlock` as the fan-out region.
- Reuses `_dispatch`'s subprocess-per-unit machinery almost verbatim: today
  `_dispatch` takes `{group_value: block_fn}` and returns `{group_value:
  result}` across up to `max_workers` spawned processes. Fan-out generalises
  the value being dispatched from "one block's group slice" to "one full
  run of the enclosed sub-graph," keyed by iteration index instead of group
  value. The existing `MAX_WORKERS` ceiling, spawn-context isolation, and
  "worker process exited unexpectedly" error handling all carry over
  unchanged.
- `CollectBlock` merges per-iteration outputs the same way
  `_combine_group_results` already merges per-group outputs today (concat
  for dataframe outputs, dict-keyed for scalar outputs) — extended with a
  reducer appropriate to a `SimulationResultPacket` (streaming
  moments/quantile accumulation, not a plain concat — see §6).
- Cache key: `compute_key` extends to include the iteration index in the
  lineage hash for blocks inside the fan-out region, so caching/`orange`
  propagation works exactly as today per-iteration.
- Bound: this path is for N ≈ 10²–10⁴ (bootstrap, CV, IFRS-9 scenario
  count, sensitivity sweeps). Above that, it's the wrong tool — see design
  principle 2.

---

## 5. Vectorised simulation blocks — the N ≥ 10⁶ path

No engine change. A convention for how these blocks are written, since
performance here is entirely about not fighting NumPy:

- **Chunked accumulation, never a full `(N, k)` materialisation for large
  k.** A simulation block takes a `chunk_size` param (default tuned to
  keep one chunk comfortably under ~200MB), loops chunks inside the block
  body, and accumulates a running histogram/moment/order-statistic
  structure rather than concatenating chunks into one array. For portfolio
  credit EC specifically: accumulate **portfolio loss per path**
  (a 1-D running array of length N) and never materialise obligor × path;
  keep per-obligor detail only inside the chunk that's currently live, and
  derive contributions from conditional expectations accumulated
  chunk-by-chunk (§6), exactly as review §3.5 specifies.
- **Vectorise the inner loop.** Draws via `Generator.standard_normal((chunk,
  k))` / `Generator.gamma(...)` etc., not `for path in range(n): ...`. Any
  block proposing a Python-level per-path loop over N ≥ 10⁵ should be
  treated as a design smell in code review, the same way an unindexed
  join-in-a-loop would be for a dataframe block.
- **Parallel across cores by chunk**, using the same `run_block` subprocess
  isolation and `max_workers` ceiling as everything else — chunks are
  embarrassingly parallel and combine via the streaming accumulator, so
  this is a `_dispatch`-style map over chunk index, not new infrastructure.
- **Variance reduction as block params, not separate blocks**: antithetic
  variates and Latin hypercube stratification are draw-time transforms
  (cheap, always worth defaulting on for symmetric risk factors); quasi-MC
  (Sobol) and importance sampling on the systematic factor are opt-in
  params on the simulation block since they change the estimator's
  variance/bias tradeoff and need the convergence diagnostics (§6) to
  reflect which one was used.
- **Semi-analytic short-circuit where it exists.** For credit portfolio EC
  specifically, conditional-independence means the loss distribution given
  the systematic factor draw is analytic/FFT — so the "simulation" is a
  factor-draw loop (small N, cheap) with an analytic or FFT evaluation per
  draw, not obligor-level Monte Carlo at all. Ship this (single-factor
  ASRF closed form + FFT/saddlepoint conditional-independence) *before*
  full obligor-level MC — it's the performance win and the validation
  benchmark in one block (review §3.2).

---

## 6. Convergence diagnostics and risk measures

Every `SimulationResultPacket` carries these unconditionally — a capital
number without them is, per the review, "a finding waiting to happen":

- **Running-estimate series**: quantile estimate recomputed at geometric
  checkpoints (N/32, N/16, ..., N) during the same pass that accumulates
  the histogram, so this is ~free (no second pass over paths).
- **Standard error of the quantile estimator**: order-statistic-based SE
  (Maritz-Jarrett or the simpler binomial approximation around the
  order statistic) computed directly from the accumulated histogram —
  avoids a second full bootstrap just to report an SE.
- **Bootstrap CI on VaR/TVaR**: resample path-level losses (or, for the
  chunked case, resample chunk-level sub-losses as an approximation when
  raw paths aren't retained) — this is itself an instance of graph fan-out
  from §4 at small N (≈500-1000 resamples on a coarse array), reusing the
  same primitive rather than a bespoke bootstrap implementation.
- **VaR(α) = quantile**, **TVaR/ES(α) = mean of tail beyond VaR**, both from
  the same accumulated structure. Full quantile table (not just the one
  headline α) so a validator's "show me 99.0 through 99.97" doesn't need a
  re-run.

---

## 7. Aggregation and allocation — including the var-covar path

This is the piece the task calls out explicitly, so it gets its own
section rather than living inside "risk measures."

### 7.1 Two aggregation regimes, one output contract

Both regimes produce the same shape of answer — total capital, a
diversification benefit, and a per-component allocation — so downstream
reporting blocks don't need to know which one ran:

```python
def aggregate(components: list[DistributionPacket | ProxyFunctionPacket],
              dependency: DependencyPacket,
              alpha: float) -> AggregationResultPacket: ...
```

**(a) Simulation-based aggregation** (full MC or graph fan-out): draw
correlated/copula-linked factors (§7.2 below), evaluate each component
under the joint draw, sum, take the quantile of the sum. Exact for
whatever the dependency structure actually is, expensive, and what you fall
back to when components are strongly non-normal or path-dependent.

**(b) Var-covar aggregation** (the analytic/proxy path — this is where
curve fitting and var-covar meet, per review §3.6 method 5): each
component is represented as a **local expansion around the base case**
rather than a full distribution:

1. **Sensitivities.** Delta (first derivative) and gamma (second
   derivative, plus cross-gamma between risk factors) per component, via
   either analytic formulas where the component has one, or bump-and-reval
   — which is itself literally the **curve-fitting proxy** from §3.6/§8
   below at a minimal fitting-scenario set (base ± bump per factor, base ±
   bump per factor pair for cross-gamma). This is the concrete link between
   "curve fitting" and "var-covar": var-covar *is* a curve fit, just one
   truncated to a quadratic Taylor expansion instead of a general basis.
2. **Risk-factor distribution.** A correlation/covariance matrix across
   risk factors (with PSD repair / nearest-correlation projection — real
   covariance matrices built from mismatched history windows are routinely
   not PSD and this must not silently produce garbage or crash deep inside
   an eigendecomposition).
3. **Analytic aggregation under normality**: portfolio variance is the
   quadratic form `wᵀΣw` over the delta vector and covariance matrix;
   portfolio VaR is a closed-form multiple of that std dev. Fast,
   transparent, and the right default when it's defensible.
4. **Non-normal correction where normality fails** — which review §3.4
   correctly flags as *exactly* where var-covar is weakest, so it's not
   optional to support:
   - **Cornish-Fisher expansion**: adjust the normal quantile using the
     portfolio's skewness/kurtosis (computed from the component moments and
     the dependency structure) — cheap, standard, and the natural default
     upgrade from pure normal var-covar.
   - **Delta-gamma-copula**: push the gamma term through a Gaussian or t
     copula on the risk factors rather than assuming joint normality of the
     factors themselves — needed once cross-gamma / tail dependence
     actually matters (FX/rates books, deep OTM optionality).
   - **Moment-matching**: fit a shifted lognormal/Johnson distribution to
     the first 3-4 moments of the portfolio, then read the quantile off
     that instead of off a normal — cheapest non-normal upgrade, worth
     offering as the middle option between plain normal and full
     delta-gamma-copula.

   These are params on the *same* var-covar aggregation block (a `method:
   "normal" | "cornish_fisher" | "delta_gamma_copula" | "moment_matching"`
   dropdown), not four separate blocks — they share the sensitivity/
   covariance inputs and differ only in the final quantile step.

### 7.2 Dependency layer

Shared by both regimes: correlation/covariance matrix input (with the PSD
repair above), Cholesky decomposition for correlated normal draws, Gaussian
and Student-t copulas for the simulation path, Archimedean (Clayton/Gumbel)
copulas where asymmetric tail dependence matters. One `DependencyPacket`
port type feeding both the simulation-based and var-covar aggregation
blocks — the dependency assumption is what actually drives the answer
(review §3.1.5), so it's a visible wire on the canvas, not a hidden param
buried inside an aggregation block.

### 7.3 Allocation

Given the total and the components, allocate the diversified total back:

- **Euler / expected-shortfall contributions** — the default, since ES
  contributions are far more stable than VaR contributions at realistic N
  (review §3.2). For simulation-based aggregation: contribution of
  component i = mean of component i's loss, conditional on total loss
  exceeding the VaR threshold (a conditional expectation over paths already
  in memory from the chunked accumulation in §5 — no extra simulation
  needed). For var-covar: Euler contributions have a closed form directly
  from the sensitivity vector and covariance matrix (`(Σw)_i · w_i / σ_p`
  scaled to the reported risk measure) — another place the analytic path is
  cheap.
- **Shapley allocation** as an opt-in alternative for the "what does my
  largest exposure cost me" question, at the cost of evaluating the
  aggregate over component subsets (2^k for k components/modules — fine for
  ~10-20 risk modules, not for obligor-level allocation, so it's offered at
  the risk-module level, not the obligor level).

---

## 8. Curve fitting / proxy functions — general form

Section 7.1 above is var-covar specifically; this is the general
proxy-fitting block family per review §3.6, sharing the
`ProxyFunctionPacket` contract from §2:

- **`fit_proxy` block**: inputs are a `ScenarioSetPacket` (purpose=
  `"fitting"`) plus either heavily-simulated/exact values at each fitting
  scenario, or an analytic spec. Params: `method` (`"polynomial"` |
  `"spline"` | `"replicating_portfolio"` | `"lsmc"`), and method-specific
  knobs (basis degree, regressor selection strategy for LSMC's stepwise/AIC
  selection). Implementation for the regression-based methods sits directly
  on `scikit-learn` (already a dependency) — polynomial features + Ridge/
  Lasso for basis selection, no new numerical dependency needed for the
  entry-level version.
- **`evaluate_proxy` block**: `ProxyFunctionPacket` + a `ScenarioSetPacket`
  (purpose=`"capital"` or `"validation"`) → value per scenario. Same block
  regardless of which method fitted the proxy — this is the swap point
  review §3.6 calls out as the actual point of the abstraction.
- **`validate_proxy` block** (ships alongside, not deferred — review §3.6
  calls this "mandatory, and a real product opportunity"): takes a
  validation `ScenarioSetPacket` fully/exactly valued, plus the proxy,
  reports error overall and **in the region that matters** (near the
  target quantile), per-risk-factor error curves, worst-case scenarios,
  and out-of-sample R². This is what makes "swap curve-fit for var-covar
  and compare" an actual one-block operation instead of a manual exercise,
  and it's the artifact a validator/supervisor will ask for regardless of
  which proxy method was used.
- **Entry point**: ship `method="closed_form"` (identity proxy — no fitting,
  used where an analytic valuation function exists, which is all IFRS 9
  needs per review §3.6) and `method="polynomial"` first. LSMC and
  replicating portfolios are increments on the same interface once the
  contract is proven, not a bigger initial scope.

---

## 9. Caching and memory (`cache.py`)

`CacheStore.set` today pickles the whole `outputs` dict to
`{cache_dir}/{key}.pkl` unconditionally — fine for dataframes, wrong for a
10⁶-path result.

- `SimulationResultPacket` is what gets cached normally (pickled like
  today) — it's already distilled (quantiles, moments, convergence,
  contributions), typically kilobytes.
- Raw paths are **not** part of the cached packet. When a block/user
  explicitly asks to retain them (debugging, an unusual downstream need),
  they're written to `{cache_dir}/paths/{key}.parquet` (or `.npy` for a
  plain array) on request, and `SimulationResultPacket.paths_handle`
  becomes a `PathsHandle(path=..., shape=..., dtype=...)` reference rather
  than the array — lazily loadable, never eagerly deserialized as part of
  a normal cache hit.
- `CacheStore` needs no structural change for this — it's a discipline on
  *what* gets put in `outputs` for these block types, enforced at the block
  layer (the vectorised simulation block returns the distilled packet plus
  an optional handle, never the raw array, as its actual return value).

---

## 10. Phased delivery

Each phase is independently useful, in dependency order:

1. **Seed hierarchy + `DistributionPacket` + GoF battery.** No engine
   change. Unlocks: distribution fitting/inspection blocks, and is a
   prerequisite for everything after.
2. **Vectorised simulation block convention + chunked accumulator +
   `SimulationResultPacket`.** No engine change (§5 runs through existing
   `run_block`/subprocess isolation). Unlocks: single-factor ASRF closed
   form, then a first real Monte Carlo block (e.g. compound-Poisson op-risk
   LDA — smallest real vectorised simulation, good proving ground before
   portfolio credit EC's memory-chunking complexity).
3. **Dependency layer** (correlation/PSD repair, Cholesky, Gaussian/t
   copula) + **`ProxyFunctionPacket`/`fit_proxy`/`evaluate_proxy`/
   `validate_proxy`** with `closed_form` + `polynomial` methods. No engine
   change. Unlocks: IFRS 9 macro-scenario valuation (closed-form, small N)
   end to end, and the var-covar aggregation block from §7.1 (which
   consumes the dependency layer + a bump-and-reval proxy fit directly).
4. **Graph fan-out engine primitive** (`IterateBlock`/`CollectBlock`, §4).
   The one real engine change. Unlocks: bootstrap CIs, k-fold CV, and
   multi-factor obligor-level credit EC simulation (now that fan-out and
   the vectorised path both exist, obligor-level MC can be built as a
   vectorised block *inside* a small-N fan-out over stress/reverse-stress
   parameter sets).
5. **Full aggregation/allocation block** (§7.3: Euler/ES contributions,
   Shapley) + cache/memory handling for retained raw paths (§9).
   LSMC and replicating-portfolio proxy methods as increments on step 3's
   interface, once there's a real nested-simulation use case pulling for
   them (CVA/XVA, insurance liabilities with options and guarantees).

Step 3 before step 4 is deliberate: it gets curve fitting and var-covar
aggregation — the two things this proposal was specifically asked to cover
— shipped without waiting on the larger engine change, since neither one
actually needs sub-graph fan-out to be useful.

---

## 11. Open decisions

1. **Which vectorised simulation ships first** as the proving ground for
   §5's chunking discipline — op-risk LDA (simpler, lower stakes) vs.
   single-factor credit ASRF (closed form, no chunking needed, but doesn't
   exercise the memory strategy) vs. jumping straight to multi-factor
   obligor-level EC (exercises everything, highest effort). Recommend
   op-risk LDA, per §10 step 2.
2. **Copula library**: hand-roll Gaussian/t/Clayton/Gumbel over NumPy/SciPy
   (small surface, full control, no new dependency) vs. adopt an existing
   package (e.g. `copulas`). Recommend hand-rolling initially — the four
   families needed are individually simple, and a dependency here would be
   the tool's first non-`scipy`/`sklearn` numerical library.
3. **`chunk_size` default and the memory budget it targets** — needs a
   number tied to the actual deployment target (single laptop vs. a beefier
   server), since it changes the practical N ceiling before someone hits
   swap.
