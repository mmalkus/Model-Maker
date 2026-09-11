# Model developer review — gaps, options, open questions

A working-through of what Model-Maker would need to be a tool a *model
developer* (credit risk / IFRS 9, and insurance internal models) actually
builds production-relevant models in, rather than a nice pipeline toy.
Nothing here is a commitment — it's the full option set so we can pick.

Written against the tool as it stands at the time of writing (see §0).

---

## 0. Baseline — what exists today

| Area | Today |
|---|---|
| Data in/out | `read_csv`, `write_csv`, `display_table`, `display_value`, `generate_image` |
| Transforms | `filter`, `select`, `groupby_agg`, `join`, `train_test_split` |
| Features | `woe_transform` (fixed-bin WoE on one column) |
| Estimation | `glm_fit`, `logistic_regression` |
| Validation | `ks_test`, `auc_gini`, `psi_test` |
| Metadata | `ColumnRole` = id / target / predicted / weight / feature / date / segment / excluded / unassigned, plus free-text description + tags |
| Execution | Cache keyed on `(block_id, inputs_hash, params_hash, code_version)`, grey/green/orange/red states, process-isolated block runs, cancellation, `group_by` execution |
| Export | Single-file Python compile, provenance header, parseable block markers |
| Project | JSON graph + sidecar `.py` for custom blocks, git-based versioning |
| AI | Pluggable LLM providers, "draft a block", "suggest a fix" |

That's a credible **scorecard-development skeleton**. The gaps below are
mostly (a) the rest of the credit lifecycle, (b) anything stochastic /
actuarial, and (c) everything that makes output defensible to a validator
or a regulator.

---

## 1. Cross-cutting gaps (hit every model type)

### 1.1 Data quality & profiling

Nothing today tells you your data is fit to model on. A validator will ask
for this before looking at a single coefficient.

- **Data profile block** — per column: fill rate, distinct count, min/max,
  percentiles, dominant value share, dtype vs. expected dtype. `ColumnStats`
  already computes a subset on demand; promote it to a block whose output is
  a table you can wire onward (and put in the doc pack).
- **Data quality rules block** — declarative assertions (`not null`,
  `in set`, `between`, `unique`, `referential: exists in other frame`,
  `row count within X% of prior run`). Output: a pass/fail table + a
  severity. Should be able to *fail the run* on a breach, or just flag.
- **Duplicate / key-integrity check** — duplicate account-months are the
  single most common silent killer in IFRS 9 datasets.
- **Outlier & implausible-value detection** — IQR, z-score, percentile caps,
  plus a "domain rule" variant (negative exposure, LGD outside [0, 1], PD of
  exactly 0 or 1, maturity before origination).
- **Missing-data treatment block** — currently absent entirely. Needs
  explicit strategies (drop / constant / mean / median / mode / separate WoE
  bin / model-based) and must **record which strategy was used per column in
  the metadata**, because that is a documented modelling assumption.
- **Reconciliation block** — totals vs. a control figure (e.g. exposure in
  the modelling sample vs. the finance ledger). Regulators ask "does your
  modelling data tie to the books"; right now there's no way to evidence it.
- **Exclusion tracking / waterfall block** — the single most-requested table
  in any model doc: start population N → each exclusion rule → final
  modelling population, with counts and exposure at each step. Today
  `filter` silently drops rows and the count is lost. Proposal: make `filter`
  emit an optional second output port carrying the exclusion summary, or add
  a dedicated `apply_exclusions` block taking a list of named rules.

### 1.2 Feature engineering & variable selection

- **Binning that's actually usable**: optimal/supervised binning (tree-based,
  chi-merge, monotonic constraint), manual bin override, min-bin-size and
  min-event-count constraints, special handling for missing/exceptional
  values as their own bin. Today's `woe_transform` is equal-ish bins on one
  column only.
- **Information Value / Gini per variable** table across all candidates at
  once (the univariate analysis step), with the standard IV interpretation
  bands.
- **Correlation matrix + VIF / multicollinearity screen**, with a
  cluster-based variable reduction option.
- **Stepwise / LASSO / forward-backward selection block** with the selection
  path recorded (which variables entered, at which step, on what criterion)
  — that path *is* documentation.
- **Monotonicity enforcement & check** — for credit, non-monotonic WoE on a
  risk driver is a review finding. Need both "enforce during binning" and
  "test and report" variants.
- **Transformations**: log/sqrt/Box-Cox, capping/flooring, ratio builders,
  time-since/vintage derivations, lagged and rolling-window features on
  panel data (crucial for behavioural models — and currently impossible
  without custom code).
- **Categorical handling**: rare-level grouping, target/WoE encoding,
  one-hot with reference level.
- **Feature dictionary export** — name, definition, source, derivation,
  role, treatment applied. Falls out of `ColumnMeta` almost free and is a
  mandatory appendix in most model docs.

### 1.3 Estimation coverage

- **Credit**: logistic with weights and offsets, scorecard scaling
  (points-to-double-odds, base score/odds — turning coefficients into an
  actual points-based scorecard table is the deliverable in retail credit
  and is missing), constrained regression, GBM/XGBoost/LightGBM as
  challengers, decision trees for segmentation.
- **Beta regression / fractional logit / Tobit** for LGD (bounded [0,1]).
- **Two-stage / hurdle models** (cure-rate + loss-given-non-cure for LGD;
  zero-inflated frequency in insurance).
- **Survival models** (Cox, discrete-time hazard, Kaplan–Meier) — needed for
  lifetime PD term structures, prepayment, and claims run-off timing.
- **Panel/GEE and mixed models** with clustered standard errors — credit
  panel data violates independence and nobody's standard errors are right
  without this.
- **GLM completeness**: full family/link set (Poisson, Gamma, Tweedie,
  binomial, negative binomial), offsets, exposure weights, dispersion
  estimate, deviance/AIC/BIC, type-III tests. The current `glm_fit` is the
  right hook; it's the surrounding statistics that are thin.
- **Time series / macro models**: ARIMA/VAR/OLS with lags for macro-factor
  models (needed for IFRS 9 forward-looking and for insurance ESG-lite work).
- **Calibration blocks**: Platt scaling, isotonic regression, central
  tendency anchoring / scaling to a long-run average default rate, and
  "shift the intercept to hit target ODR" — the last one is a daily job in
  credit and has no equivalent today.

### 1.4 Validation & performance testing

Current set (KS, AUC/Gini, PSI) covers discrimination and stability
*shallowly*. Missing:

- **Discrimination**: ROC curve (not just the scalar), CAP/Lorenz, Somers' D,
  AUC confidence intervals (DeLong), per-segment and out-of-time Gini,
  Gini *decay* over time.
- **Calibration / accuracy**: Hosmer–Lemeshow, binomial / Jeffreys test per
  grade, Brier score, observed-vs-expected by bucket with CIs, traffic-light
  test.
- **Stability**: characteristic stability (CSI) alongside PSI, migration
  matrices, concentration (Herfindahl) of the rating grade distribution.
- **Robustness**: k-fold / stratified CV, bootstrap CIs on coefficients and
  metrics, out-of-time and out-of-sample holdout as *first-class* rather
  than a manual split, sensitivity to sample period, stress on input
  distributions.
- **Benchmarking**: champion vs. challenger comparison block — take N model
  outputs, produce one comparison table + overlay plots. This is also the
  natural home for "did v4 beat v3".
- **Explainability**: SHAP / permutation importance / partial dependence /
  coefficient contribution waterfall. Increasingly not optional (EU AI Act
  treats consumer creditworthiness scoring as high-risk; adverse-action
  reasons need per-decision drivers).
- **Fairness / discrimination testing**: outcome rates and model performance
  by protected-ish characteristic, proxy-correlation screening. Depends on
  jurisdiction whether it's required, but the *capability* to run it is
  cheap and the reputational downside of not having it is not.
- **A "validation suite" macro-block** — one block that fans out into the
  standard battery for a given model family and emits a single structured
  results object. This is what makes documentation generation (§5) tractable.

### 1.5 Reproducibility & determinism

- **Global seed / RNG policy**: `train_test_split` takes a seed, but there's
  no project-level seed and nothing forces stochastic blocks to declare one.
  Any block whose output depends on an RNG must take a seed param and hash
  it into the cache key.
- **Environment capture**: compile currently emits a self-contained `.py`
  but not the package versions it was run under. Emit a `requirements.txt` /
  lock alongside, and record Python + key library versions in the provenance
  header. "It reproduces" is a regulatory expectation, not a nicety.
- **Data snapshot identity**: hash / row count / schema hash of each input
  source at run time, stored with the run. Today's `_probe_read_csv` is the
  seed of this — extend to a full "source fingerprint" that the doc pack and
  the compiled header both quote.
- **Determinism flag per block** — mark blocks that are not deterministic
  (wall-clock, RNG without seed, external API) so the UI can warn.

---

## 2. Credit risk & IFRS 9

Covers the account/obligor-level estimation stack. **Portfolio-level credit
risk — economic capital, concentration, risk contributions — is in §3.2**,
because it shares its machinery with insurance capital rather than with
scorecard development.

### 2.1 PD

- Rating/grade assignment block (score → masterscale bands, with a grade
  boundary editor).
- Long-run average PD / central tendency calculation and calibration to it.
- Through-the-cycle vs. point-in-time conversion (scalar, or a variable
  scalar approach).
- Low-default-portfolio treatment (most prudent estimate / confidence-level
  based PD floors).
- Reject inference (parcelling, augmentation, fuzzy) for application
  scorecards.
- Definition-of-default flagging block (90dpd + unlikeliness-to-pay markers,
  cure periods, re-default) — the target variable construction is itself a
  modelling decision that needs to be visible and versioned, not buried in a
  SQL query upstream of the tool.
- Observation/outcome window construction on panel data (snapshot date +
  12-month outcome window, with proper exclusion of incomplete windows).

### 2.2 LGD

- Recovery cash-flow aggregation and discounting to default date (with the
  discount-rate choice as an explicit, documented param).
- Workout-period handling, incomplete-workout treatment.
- Cure-rate model + loss-given-non-cure model, combined.
- Downturn LGD add-on / downturn identification.
- Collateral haircut and time-to-realisation modelling.
- LGD bounded-outcome modelling (see §1.3) and the standard LGD validation
  battery (which is *not* Gini — it's predictive accuracy on a continuous
  bounded target: MAE, MSE, bucketed O vs. E, transition analysis).

### 2.3 EAD / CCF

- CCF / credit-conversion-factor computation from limit and drawn balance at
  reference and default dates (fixed-horizon, cohort, variable time
  approaches — each is a distinct method that should be a param, not a
  rewrite).
- Undrawn-limit handling, limit changes, negative CCF truncation.
- EAD validation battery (same continuous-target flavour as LGD).

### 2.4 IFRS 9 machinery

This is where the tool is furthest from being usable, and also where a
visual pipeline tool has the most obvious value — the ECL calculation is
*exactly* a DAG of joins and multiplications that people currently run in
unauditable spreadsheets.

- **Lifetime PD term structure**: survival/hazard-based or migration-matrix
  based, producing marginal and cumulative PD by period out to maturity.
- **Macroeconomic scenario blocks**: scenario input (base/up/down with
  probability weights), macro-factor model linking macro variables to PD/LGD,
  scenario-conditional PD term structures, and probability-weighted
  aggregation across scenarios. Needs the engine to support **running the
  same sub-graph once per scenario** (see §7.2) — today you'd have to
  duplicate the branch three times by hand.
- **SICR / staging block**: stage 1/2/3 assignment with configurable
  quantitative trigger (relative/absolute PD deterioration vs. origination),
  qualitative triggers, 30dpd backstop, low-credit-risk exemption, cure
  criteria. Output: stage per account + a **staging migration matrix**,
  which is a standing audit ask.
- **ECL engine block**: EAD × PD × LGD × discount, summed over periods,
  12-month for stage 1 and lifetime for stage 2/3, with an explicit
  effective-interest-rate discounting param.
- **Post-model adjustments / overlays register**: named, quantified,
  rationale-carrying, approver-carrying overlays applied on top of modelled
  ECL. Auditors currently chase these through email. Making them a
  first-class block with mandatory metadata is a genuine selling point.
- **ECL attribution / walk**: movement of ECL between two reporting dates
  decomposed into stage transfers, new business, derecognition, model
  changes, macro-scenario changes, overlay changes. This is a *required*
  disclosure and it's pure DAG arithmetic — a strong candidate for a
  flagship block.
- **Sensitivity disclosure block**: ECL under 100% weight on each scenario,
  and under +/- shifts in key assumptions (IFRS 7 disclosure).
- **Disclosure tables**: gross carrying amount and ECL by stage, by segment,
  by rating grade, plus the reconciliation table — as export-ready outputs.

### 2.5 Ongoing monitoring (the part that never gets built, then gets audited)

- Scheduled/repeatable monitoring run over a new reference period producing:
  PSI/CSI, Gini decay, O vs. E by grade, override rates, stage migration,
  data quality breaches.
- **Thresholds with RAG status** (green/amber/red per metric) — configurable
  per metric, with breach history.
- **Time-series of monitoring metrics** across runs — this requires a
  persistent run/result store (§7.3), which doesn't exist yet; the cache is
  keyed for correctness, not for history.
- Automatic "model requires re-development" flagging when N metrics breach.

---

## 3. Stochastic modelling — economic capital and insurance internal models

**These are one build, not two.** Bank economic capital and an insurance
internal model are the same machine pointed at different portfolios: fit
distributions, impose a dependency structure, simulate, read a quantile off
the loss distribution, allocate it back. The regulatory vocabulary differs
(EC/ICAAP vs. SCR), the confidence level differs (99.9% vs. 99.5%), and the
granularity differs by four orders of magnitude — but the primitives are
identical.

That materially changes the cost case. Building the stochastic core is not
"the price of entering insurance"; it is a capability banking needs anyway
for EC, ICAAP, op risk, concentration risk and stress testing, and which
also pays for bootstrap confidence intervals and IFRS 9 scenario expansion
in the credit work we already have. Once it exists, insurance is mostly
*blocks* (triangles, LoB structures, risk margin) rather than engine work.

### 3.1 The shared spine

What "we support Monte Carlo" actually decomposes into:

1. **Fan-out / collect** — run a sub-graph N times with a varying input
   (seed, scenario, fold, bootstrap resample) and gather the results. The
   §7.2 engine change; everything else here depends on it.
2. **RNG and seed hierarchy** — a project seed that spawns independent,
   reproducible sub-streams per iteration (`numpy.random.SeedSequence.spawn`
   / Philox counter-based streams), so results are identical regardless of
   execution order or parallelism. Non-negotiable: an unreproducible capital
   number is an audit finding.
3. **Distribution objects as a port type** — fit (MLE/MoM across candidate
   families), inspect, sample, and pass over a wire. Includes EVT/GPD tail
   fitting and spliced body-tail distributions.
4. **Goodness-of-fit battery** — KS, Anderson–Darling (tail-weighted, which
   is what you actually want here), chi-square, QQ/PP plots, and a
   candidate-comparison table with AIC/BIC.
5. **Dependency layer** — correlation matrices (with PSD repair /
   nearest-correlation), Cholesky, Gaussian and t copulas, Archimedean
   (Clayton/Gumbel) for asymmetric tail dependence, and tail-dependence
   diagnostics. The dependency assumption drives the answer more than the
   marginals do, so it needs to be visible and documented, not buried.
6. **Simulation execution** — N draws, chunked/streaming accumulation,
   parallel across cores, progress and cancellation (the runner already has
   process isolation and cancellation, which helps).
7. **Variance reduction** — antithetic variates, control variates,
   stratification / Latin hypercube, quasi-MC (Sobol), and importance
   sampling. Not a luxury: naive MC at 99.9% wastes almost all its paths,
   and importance sampling on the systematic factor is the standard trick in
   credit portfolio models.
8. **Convergence diagnostics** — standard error of the quantile estimator,
   bootstrap CI on the reported VaR/TVaR, running-estimate plots vs. N. A
   capital number quoted without its simulation error is a finding waiting
   to happen, and "how many paths is enough" is the first question a
   validator asks.
9. **Risk measures on a loss distribution** — VaR and TVaR/ES at arbitrary
   confidence, expected loss, unexpected loss, full quantile table,
   with CIs from (8).
10. **Aggregation and allocation** — combine risk modules / sub-portfolios,
    quantify the diversification benefit, and allocate the total back to
    components (Euler / expected-shortfall contributions, Shapley). This is
    the shared answer to both "capital by business line and RAROC" and
    "diversification by risk module".

### 3.2 Consumer: bank economic capital / credit portfolio model

Missing from §2 entirely today, and arguably as valuable as the IFRS 9 work:

- Factor models: single-factor ASRF (closed form — useful as a benchmark to
  validate the simulation against), multi-factor Merton/CreditMetrics-style
  latent-variable simulation, CreditRisk+ style actuarial variant.
- Asset correlation calibration and sector/region factor structure.
- Obligor-level simulation → portfolio loss distribution → EC as
  VaR(α) − EL, with α set by the target rating (99.9% Basel-aligned, or
  99.95–99.97% for a AA-equivalent target).
- Risk contributions per obligor / sector / business line (ES contributions
  are far more stable than VaR contributions at these N — worth defaulting
  to them).
- Concentration risk: single-name and sector, HHI, granularity adjustment,
  and the "what does my largest exposure cost me in capital" question.
- Migration-based (mark-to-model) as well as default-only loss definitions.
- Stress and reverse stress on the factor draws; linking EC output to risk
  appetite thresholds.

### 3.3 Other consumers of the same spine

Each of these is mostly *configuration* of §3.1 rather than a new build,
which is what makes the core worth paying for:

- **Operational risk LDA** — compound Poisson frequency × spliced
  lognormal/GPD severity, aggregated by MC to 99.9%. No longer regulatory
  capital under the standardised approach, but alive and well in ICAAP and
  EC, and it exists in both banks and insurers.
- **Market risk / ALM** — MC and historical VaR, ES at 97.5%, backtesting
  exceptions, IRRBB scenario grids.
- **Reserving bootstrap** — ODP bootstrap and Mack are literally fan-out +
  quantiles (§3.8).
- **IFRS 9 macro scenarios** (§2.4) — the same fan-out at N=3 instead of
  N=1,000,000, with probability weights instead of equal weights.
- **Validation** — k-fold CV, bootstrap CIs on Gini and on coefficients
  (§1.4) are fan-out + collect with a different iterator.
- **Sensitivity / stress testing** generally — sweep a parameter, collect the
  output curve. Also gives us reverse stress testing: search the input space
  for scenarios producing a given adverse outcome.

### 3.4 Where the overlap actually stops

Worth being honest about, because it affects the design rather than just the
sales pitch:

- **Tail depth.** 99.5% (SII) vs. 99.9%–99.97% (EC). Deeper tail → more
  paths, or mandatory variance reduction. The API should make the confidence
  level a parameter and the path count a consequence, with convergence
  checked automatically.
- **Dimensionality and compute profile.** Credit EC simulates hundreds of
  thousands of obligors × ~1M paths — matrix-heavy, memory-bound, needs
  vectorised NumPy (not Polars, not a per-iteration graph fan-out). An
  insurance internal model aggregates ~20–100 risk nodes with a rich
  dependency structure — trivial compute, complex structure. Same API, wildly
  different execution strategy underneath; see §3.5.
- **Severity of the nesting problem.** Both domains have it; insurance has it
  worst. A one-year view of a multi-year liability with options and
  guarantees is stochastic-within-stochastic and unavoidable; the banking
  twin is CVA/XVA exposure simulation, while credit EC largely escapes via
  semi-analytic conditional independence. So this is a difference of
  *degree and of which escape you reach for*, not a clean insurance-only
  boundary — see §3.6, which is where the actual design decision sits.
- **Benchmarks.** Credit has closed-form ASRF to sanity-check against;
  insurance has the standard formula as a rough comparator but no analytic
  truth. Affects how much validation tooling each needs.
- **Allocation vs. attribution.** Banks want capital allocation and RAROC;
  insurers additionally need P&L attribution as a standing regulatory test.
  Related but not the same computation.

### 3.5 Two execution shapes (an early design decision)

Monte Carlo shows up in the tool in two forms that should not be conflated:

- **(a) Graph fan-out** — run a sub-graph N times. General, visible on the
  canvas, works for scenarios, folds, bootstrap, sensitivity sweeps.
  Practical to roughly N ≈ 10³–10⁴; each iteration carries graph overhead.
- **(b) Vectorised simulation block** — one block internally draws an
  (N × k) array with N ≥ 10⁶. The only workable shape for credit portfolio
  EC. This is mostly a *block-authoring* concern, but it needs array-shaped
  packets (§7.1) and a memory strategy, not the fan-out engine.

Both are probably needed, and they have different costs. Related decisions:

- **Caching.** A million-path result is not something to hash and stash like
  a dataframe. Proposal: cache the *distilled* output (quantile table,
  moments, convergence stats, allocation vector) in the normal cache, and
  persist raw paths to disk (parquet/npy) only on request, referenced by
  handle. Otherwise the cache design that makes the tool pleasant becomes
  the thing that makes it fall over.
- **Memory.** Don't materialise obligor × path. Accumulate portfolio loss per
  path in chunks; keep per-obligor detail only where allocation needs it, or
  derive contributions from conditional expectations.
- **Canvas UX.** How do you draw a loop in a DAG tool? Options: a container
  /"fan-out lane" that visually encloses the iterated sub-graph; an explicit
  `iterate` block paired with a `collect` block; or marking a wire as
  carrying N replicates with a badge. Worth prototyping on paper before
  committing — this is the part users will either immediately understand or
  never trust.
- **Determinism under parallelism** — see §3.1(2); it has to hold whether
  the run is serial, threaded or across processes.

### 3.6 Scenario valuation — the nested-simulation problem and its escapes

The outer loop hands you N real-world scenarios at the horizon. At each one
you need a **value** (own funds, portfolio value, ECL, exposure). If valuing
itself requires simulation, you have N × M nested paths — 10⁶ × 10⁴ is not a
compute problem, it's an impossibility. Every practical method is a way of
*not doing that*, and they are all instances of one abstraction:

> a **proxy function** (valuation surrogate) mapping risk factors → value,
> fitted or derived once, then evaluated cheaply N times.

This is not a corner of the design — for scenario-based capital it *is* the
design. Correcting what I wrote earlier: this isn't deferrable. What's
deferrable is how many of the escapes below we support beyond the cheapest.

**The family:**

1. **Closed-form / semi-analytic** — the value function is known, so evaluate
   it directly per scenario. No fitting, no nesting, exact where it applies,
   and it should be the default wherever the payoff permits. In credit this
   is the conditional-independence trick: conditional on the systematic
   factor draw the loss distribution is analytic (or FFT / saddlepoint), so
   you integrate over factors instead of nesting — Vasicek/ASRF is its
   limiting case, CreditRisk+ its FFT form. In insurance it covers simple
   liabilities and vanilla market instruments.
2. **Curve fitting** — value accurately at a *small number of deliberately
   chosen* fitting scenarios (heavy inner simulation or exact valuation at
   each), then fit a proxy function through those points. Few points, each
   precise. Scenario selection is the craft.
3. **LSMC** — the mirror image: very many fitting scenarios with very few
   inner paths each (often 1–2), so every point is extremely noisy, and
   least-squares regression on a basis expansion averages the noise out.
   Same compute budget, spent the opposite way. Design params that matter:
   basis family and degree, regressor selection (stepwise/AIC), the risk
   factor set, and the fitting-scenarios ÷ inner-paths split.
4. **Replicating portfolio** — fit a portfolio of instruments with known
   closed-form values (zeros, swaps, swaptions, equity options) to match
   liability cash flows or values across a calibration set, then value the
   replicating portfolio under the full scenario set. The fitted object is a
   *portfolio*, not a polynomial: interpretable, hedge-relevant, reusable.
   Limited by the instrument universe and weak on non-market risks
   (biometric, lapse).
5. **Var-covar / sensitivity-based** — local expansion around the base:
   delta, gamma, cross-gamma plus an assumed risk-factor distribution.
   Analytic aggregation if everything is normal — but the tail is precisely
   where normality fails, hence the non-normal variants worth supporting:
   fat-tailed marginals and a copula pushed through the delta-gamma
   expansion, Cornish–Fisher or moment-based quantile corrections, Johnson
   transformations, moment-matching on the aggregate. Fast and transparent;
   poor for path-dependent or strongly convex liabilities.

**The unifying contract.** All five reduce to:

```
fit_proxy(fitting_scenarios, values | analytic_spec) -> ProxyFunction
evaluate(ProxyFunction, scenario_set)               -> value per scenario
```

So `ProxyFunction` belongs as a first-class port type next to `distribution`
and `model` (§7.1), and the choice of method becomes a **swappable block
against a common interface**. That matters practically, because real
balance sheets use different methods for different sub-portfolios and
aggregate the results — and because being able to swap one for another and
compare *is* a large part of how you validate the proxy.

**Scenario sets as typed artefacts.** A scenario set is not "a dataframe I
hope is the right one". It should carry: measure (**real-world vs.
risk-neutral**), horizon, risk factors, N, generator/calibration reference,
seed, and its purpose (fitting / validation / capital). Valuing under the
wrong measure is a classic and expensive error, and with this metadata it
becomes a **wire-validity check** rather than something caught in review —
which is exactly the trick the tool already plays with `ColumnRole`, raised
from the column to the object level. Real-world outer / risk-neutral inner
is the standard structure and should be visible on the canvas. ESG
integration is either a simple built-in generator or (more realistically)
import from whatever external ESG the shop already runs, with that metadata
attached on import.

**Proxy validation — mandatory, and a real product opportunity.** The proxy
is a model approximating a model, and supervisors scrutinise it hard:

- Out-of-sample validation scenarios, fully and accurately valued, compared
  against the proxy — with error measured not just overall but **in the
  region that matters**, around the SCR quantile and the biting scenarios.
- Diagnostics: error vs. each risk factor, tail error, the worst-case
  scenarios where the proxy breaks down, stability of fitted coefficients
  across refits, out-of-sample R².
- Validation on deliberately chosen stress scenarios, not only random ones.
- Refit cadence and drift: when is the proxy stale? Ties to §7.3.

This pack is currently built in spreadsheets almost everywhere. A block that
produces it is a strong candidate for a flagship feature.

**Where this bites outside insurance** — reinforcing §3's thesis rather than
undercutting it:

- **CVA / XVA** is nested Monte Carlo in a bank: simulate exposure paths,
  value the book at each path and time step. Solved with regression
  proxies — and Longstaff–Schwartz American Monte Carlo is the direct
  ancestor of LSMC. Same technique, different desk.
- **Market risk**: full revaluation vs. grid vs. Taylor expansion is the
  same trade-off; FRTB's sensitivities-based approach is a prescribed
  var-covar.
- **Credit EC**: semi-analytic conditional-independence (above) is the
  escape from nesting; importance sampling handles the deep tail.
- **IFRS 9**: ECL is a *deterministic* function of the macro scenario, so
  it's case (1) at small N — no proxy fitting needed. Worth stating plainly,
  because it means the scenario machinery serves the IFRS 9 work we already
  care about without any of the fitting complexity.
- Any what-if or stress-testing workflow where full revaluation is too slow.

### 3.7 Insurance pricing / underwriting

- Frequency–severity structure: Poisson/NB frequency with exposure offset ×
  Gamma/Lognormal/Tweedie severity, combined to pure premium.
- Exposure handling (earned exposure, policy-year vs. accident-year).
- Large-loss capping and excess-layer treatment, with the cap as an explicit
  assumption.
- Credibility weighting (Bühlmann–Straub, limited fluctuation).
- Rating-factor table output — the insurance analogue of the scorecard table:
  relativities per level, base rate, and the resulting tariff.
- Lift/gains charts, double-lift charts, actual-vs-expected by factor level,
  and Gini for pricing models (the "Lorenz curve of premium vs. loss" flavour,
  which is not the same construction as credit Gini).
- One-way vs. multi-way analysis views.

### 3.8 Insurance reserving

- Claims triangle construction from transactional data (accident/underwriting
  period × development period), incremental and cumulative.
- Chain ladder (volume-weighted, simple average, selected factors with manual
  override), Bornhuetter–Ferguson, Cape Cod, Mack (with its variance
  estimate), Over-Dispersed Poisson bootstrap for a reserve distribution.
- Tail factor selection (curve fitting + manual).
- IBNR split, case-reserve interaction, run-off / back-testing of prior
  estimates.
- Diagnostics: residual plots by origin/development/calendar period, calendar-
  year trend tests.
- **Manual selection with audit trail** — actuaries override factors, and the
  override plus its rationale must be captured. Architecturally this is the
  same feature as the IFRS 9 overlay register (§2.4): a block whose params
  are human judgements carrying a justification and an approver.

### 3.9 Insurance capital specifics, on top of the shared spine

Distribution fitting, copulas, the simulation engine, risk measures and
aggregation/allocation all come from §3.1 — they are *not* insurance line
items. What's genuinely additional here:

- Risk-module structure and the SCR as a 99.5% VaR of the one-year change in
  basic own funds (vs. EC's loss-distribution framing — same quantile
  machinery, different definition of the random variable, and that definition
  needs to be an explicit, documented modelling choice).
- Loss-absorbing capacity of technical provisions and deferred taxes.
- Risk margin (cost-of-capital on projected future SCRs) — which needs the
  SCR projected forward, i.e. the nested problem from §3.6.
- Standard-formula comparison as a benchmark/sanity view alongside the
  internal model result.
- **P&L attribution** — mandatory annual test: attribute realised P&L to the
  risk drivers in the model. Again a DAG of decompositions.
- **Validation tests required by the regime**: statistical quality test, use
  test evidence, calibration test, reverse stress testing, stability of
  outputs, sensitivity to key assumptions. Several of these are *process*
  artefacts, which makes §4/§5 (governance + docs) the actual deliverable
  rather than a new block.
- **Model change policy support**: quantify the impact of a model change on
  the SCR, classify it major vs. minor against thresholds, and produce the
  change-log entry. This is the insurance twin of §4's change-materiality
  feature and the same machinery serves both.

### 3.10 IFRS 17 adjacency (probably out of scope, worth naming)

Building blocks for fulfilment cash flows, risk adjustment (confidence-level
or cost-of-capital), CSM roll-forward, and cohort/grouping. Very large scope;
mention only so we can explicitly say "not now".

### 3.11 What the engine lacks for any of §3

- No `model` / `distribution` / `simulation` port type flowing over a wire
  with the same richness as `DataFramePacket` (there's a typed-port notion in
  the plan, but the packet is dataframe-shaped — and simulation wants NumPy
  arrays, not Polars frames).
- No way to run a sub-graph N times (scenarios, simulations, bootstrap folds)
  — see §7.2. Without it, every stochastic method is a hand-written block.
- No project-level RNG/seed discipline (§3.1(2), §1.5).
- Cache and memory model assume a materialisable in-memory frame (§3.5).
- No convergence/simulation-error concept anywhere, so nothing would stop a
  user quoting a capital number from 1,000 paths.

---

## 4. Model risk management & governance

This is the part that decides whether a bank/insurer can *use* the tool at
all, and it's largely orthogonal to the modelling blocks.

- **Model lifecycle state** on the project (draft → development complete →
  under validation → approved → in production → under monitoring → retired),
  distinct from block run state. Transitions recorded with who and when.
- **Sign-off / approval capture** — developer, reviewer, validator, approver,
  with date and comment. Even lightweight (a signed JSON block in the project
  file + git commit identity) beats what most teams have.
- **Immutability of an approved version** — tag/lock a commit as the approved
  model; any further edit branches. Git gives us 90% of this; what's missing
  is the tool *knowing* about it and displaying it.
- **Audit trail of edits** — git log is the substrate, but a human-readable
  "what changed between v3 and v4" that understands *blocks* (block added /
  param changed from X to Y / code edited) rather than JSON lines. A
  semantic project diff is a genuinely valuable, self-contained feature and
  feeds both docs (§5) and the dashboard (§6).
- **Change materiality assessment** — for a given change, re-run and report
  the impact on key outputs (ECL, SCR, Gini, PD distribution) and classify
  against thresholds. Needs the run-history store (§7.3).
- **Assumptions & limitations register** — structured, block-attached fields
  (`assumption`, `rationale`, `limitation`, `compensating control`,
  `owner`, `review date`). `ColumnMeta.description`/`tags` show the pattern;
  blocks need the equivalent. Every model doc has this section and it's
  always reconstructed from memory at the end.
- **Expert judgement / override log** — shared mechanism with §2.4 overlays
  and §3.8 actuarial selections.
- **Data lineage back to source systems** — `lineage` currently lists block
  ids. Extend to: source system, extract date, query/file, owner. The
  question "where did this field come from" must be answerable end to end.
- **Environment segregation** — dev vs. UAT vs. production runs, with
  different data paths and a promotion step. Today a path is a string param
  in a block.
- **Access control** — read/edit/approve roles. Probably out of scope for a
  local tool, but relevant the moment it's shared.
- **Regulatory context to keep in view**: SR 11-7 / SS1/23 (model risk
  management), CRR/EBA GL on PD/LGD estimation and on IRB, IFRS 9 + IFRS 7
  disclosure, EIOPA internal model requirements, EU AI Act (credit scoring =
  high-risk: logging, human oversight, technical documentation, accuracy and
  robustness evidence), GDPR Art. 22 (automated decision explanation).
  Most of these convert into "capture this metadata and print it".

---

## 5. Documentation

The strongest argument for this tool over a pile of notebooks: **the graph
already contains most of the model documentation**, it's just not being
printed. Options, roughly increasing in ambition:

### 5.1 Evidence pack (mechanical, high value, low risk)
Export a folder/zip containing: the graph diagram (SVG/PNG), the compiled
`.py`, every output table as CSV/XLSX, every plot as PNG, the run
provenance (source fingerprints, timestamps, versions), and an index. No
prose, no judgement. Useful immediately, and it's the substrate everything
below sits on.

### 5.2 Auto-generated Model Development Document
A templated document assembled from graph metadata + block outputs:

- Cover: model name, version, owner, lifecycle state, approval status.
- Purpose, scope, materiality (from project-level fields we'd add).
- Data: sources, extract dates, fingerprints, profile tables, quality-check
  results, exclusion waterfall, final sample description.
- Methodology: per-block narrative walk in topological order — block name,
  phase/lane, what it does, params used, rationale field, code (optionally
  appendixed).
- Variable analysis: feature dictionary, univariate IV table, correlation,
  selection path.
- Estimation: model form, coefficients with SEs and p-values, fit stats,
  scorecard/rating-factor table.
- Calibration: method, targets, resulting mapping.
- Validation: the full battery from §1.4 with tables and charts.
- Assumptions, limitations, compensating controls (from §4's register).
- Appendices: full code, full metadata, environment.

Templates should be **swappable per regime** (IRB PD, IFRS 9, Solvency II
internal model component, generic) and ideally per-institution, since every
bank has a house model-doc template and will want their headings.

### 5.3 Section-level narrative drafting with the LLM
The tool already has a pluggable LLM. Use it to draft the *prose* around the
mechanical content — "explain what this block does and why, given its code,
params and upstream metadata" — as a **draft that a human edits and the
project stores**, never as something regenerated silently at export time.
Same review-before-accept pattern as the existing block drafting. Keep the
edited text in the project JSON so it diffs, and mark any section whose
underlying block changed since the text was written as stale (the same
grey/orange idea, applied to documentation). That staleness marker is
probably the single most useful documentation feature here: model docs rot
because nobody knows which paragraphs a change invalidated.

### 5.4 Validation and monitoring reports
Separate templates from the development doc: a validation report (independent
review findings against the battery, with pass/fail and findings log) and a
periodic monitoring report (§2.5 metrics, RAG status, trend charts,
recommendation). Both mostly reuse §5.2 machinery.

### 5.5 Change / version documentation
Given the semantic project diff (§4), auto-produce the "changes since
approved version" section: blocks added/removed/edited, params changed,
impact on key metrics, materiality classification. Feeds the model change
log required in both regimes.

### 5.6 Formats
Markdown as the canonical intermediate (diffs well, lives in git), rendered
to HTML for reading, DOCX for the people who will demand tracked changes,
PDF for the submission. XLSX for the tables that get re-cut. A repo skill
set for DOCX/PDF/XLSX already exists in this environment, so the render
step is not a from-scratch build.

### 5.7 Traceability matrix
Requirement/regulatory-clause → where it's evidenced (block, doc section,
test result). Tedious but it's exactly what a submission needs, and with
structured metadata it's generatable rather than hand-maintained.

---

## 6. Stakeholder dashboard

Three distinct audiences with genuinely different needs — worth not
collapsing them into one screen.

**A. Executive / model owner** — "is this model good, and can I sign it?"
Model name, version, lifecycle state, approval status, headline metrics with
RAG (Gini, PSI, O vs. E, ECL/SCR impact), a one-paragraph summary, key
assumptions and limitations, open findings, and the top 3 charts. No graph,
no code. This is the "present to the committee" view.

**B. Validator / auditor** — "show me the evidence". Full validation
battery with pass/fail against thresholds, the data quality and
reconciliation results, the exclusion waterfall, lineage from output back to
source, the assumption register, the change log vs. the last approved
version, and one-click drill into the block that produced any number.

**C. Monitoring / BAU** — "is it still good?" Metric time series across
monitoring runs, breach history, stage migration and portfolio drift,
override rates, and a "next review due" indicator.

### Implementation options

1. **In-app route** (`/dashboard` in the existing React frontend, fed by the
   API). Interactive, drill-down into blocks, always current. Most work, best
   result, and it's where "present-mode" belongs — a toggle that hides the
   canvas chrome for a meeting.
2. **Exported static HTML** — self-contained file (data inlined), emailable,
   attachable to a committee pack, no server needed. Probably the highest
   ratio of stakeholder value to effort, and it doubles as an archivable
   artefact for the evidence pack.
3. **Published artifact / shareable link** — same content, hosted, shareable
   with people who'll never install the tool. Good for the "show the risk
   committee" case; needs a view on data sensitivity before anything leaves
   the machine.
4. **Slide export** (PPTX) for the governance committee pack — the format
   the actual decision meeting runs on, whether we like it or not.

### Content mechanics
- **Dashboard blocks in the graph**: tag which block outputs are "headline"
  and the dashboard assembles them, rather than a dashboard hard-coding
  metric names. Keeps it model-agnostic (credit *and* insurance) and means a
  custom block can appear on the dashboard without a code change.
- **Thresholds/RAG config** stored per project, versioned with it.
- **Comparison mode** — two versions (or champion vs. challenger) side by
  side. The committee question is almost always "versus what".
- **Narrative slot** per panel, drafted by the LLM, edited and stored (same
  pattern as §5.3).
- **Scenario toggle** for IFRS 9 (view ECL under each macro scenario) and
  segment filters — both imply the dashboard reads a *result store*, not
  just the last run in memory (§7.3).

---

## 7. Engine / architecture implications

Things that would need to change under the hood for a lot of the above.
These are the decisions with the longest lead time, so worth settling early.

1. **Non-dataframe packets as first-class citizens** — fitted model objects,
   distributions, **proxy functions** (§3.6), **scenario sets** with their
   measure/horizon/purpose metadata, simulation results (NumPy arrays, not
   frames), metric bundles. Currently the packet is dataframe-shaped; ports
   are typed in the plan but the richness (metadata, caching, preview,
   compilation) is dataframe-specific. Prerequisite for all of §3 — and the
   scenario-set metadata is what turns "valued under the wrong measure" from
   a review finding into a wire-validity error.
2. **Sub-graph iteration** — run a branch once per scenario / per fold / per
   simulation / per segment, then collect. Needed for macro scenarios (§2.4),
   CV and bootstrap (§1.4), the whole of §3, and per-segment models. This
   is the single biggest structural gap: it has knock-on effects on the cache
   key, the compiler, and the UI (how do you draw a loop on a canvas?), and
   it is the one item that unlocks capability in *both* domains at once.
   Worth a design note of its own — see §3.5 for the fan-out vs. vectorised
   split, which should be settled in the same pass.
3. **Run history / result store** — persist run results (metrics, not full
   frames) with timestamp, graph version, data fingerprint. Prerequisite for
   monitoring trends (§2.5), change materiality (§4), version comparison
   (§6), and "did this change move the number" generally. The current cache
   is keyed for *correctness* and is deliberately disposable; this is a
   different, additive store.
4. **Larger-than-memory data** — Polars lazy frames / streaming, or a DuckDB
   backend. Real IFRS 9 panel data is tens of millions of account-months.
   The current eager in-memory model will hit a wall, and retrofitting lazy
   evaluation later is painful.
5. **More connectors** — parquet, ODBC/SQL, Snowflake/Databricks, Excel. CSV
   only is a blocker for anyone whose data lives where real data lives. Plus
   credential handling that isn't a path in a param (there's already a
   pattern for LLM keys to follow).
6. **Typed, validated params** — param schema per block with types, ranges,
   enums, and role-filtered column pickers, validated at edit time rather
   than surfacing as a Python traceback on run.
7. **Block-level unit tests** — let a block carry example input/expected
   output; run them on edit. Cheap way to make LLM-drafted blocks trustworthy,
   and it's evidence for the "model implementation testing" section every
   validator asks for.
8. **Scheduled / headless runs** — CLI entry to run a project against new
   data, for monitoring. The compiled script half-does this; a supported
   `modelmaker run project.json --params ...` is cleaner.
9. **Performance visibility** — per-block timing and row counts, already
   partly there via run state; surface it so people can find the slow join.

---

## 8. Quality of life

Sorted by the moment in the workflow where the friction actually bites,
rather than as a flat list. These are cheap relative to §3 and they are what
decides whether someone reaches for this tool or goes back to a notebook —
a model developer spends far more hours in the inner loop than in the parts
§1–§3 are about. ⭐ marks the ones I'd argue hardest for.

### 8.1 The inner loop: edit → run → look → fix

- ⭐ **Sample mode.** One global toggle that runs the whole pipeline on the
  first N rows / an X% sample, then a switch back to full. This is the single
  biggest time saver in any data pipeline tool — it's what `obs=` is to SAS
  users — and right now every iteration on a tiny code change pays full data
  cost. Needs care in one respect: sample mode must be unmistakable in the UI
  and must never be what a compile or a documented run is based on.
- ⭐ **"Why is this orange?"** A stale block should say what staled it:
  *your code changed*, *param `bins` changed*, *upstream `b_003` re-ran*.
  The runner already knows — it's a cache-key comparison — it just isn't
  surfaced, and without it a big orange cascade is a mystery.
- ⭐ **Eject to a REPL/notebook.** "Open this block's inputs in an IPython
  session / a scratch notebook with `inputs` bound." The one thing notebooks
  genuinely do better is poke at an intermediate frame ad hoc; rather than
  compete with that, give people a door. Probably the highest
  goodwill-per-line-of-code item on this list.
- **Error surfaces**: the failing row/column and the offending value, not
  just a traceback; click the traceback line to jump to it in the editor.
- **Packet diff in the inspector** — for any block, what it did to the data:
  columns added/removed/retyped, row count delta, null-count delta per
  column. Answers "did that join blow up my row count" without wiring a
  display block, and it's the question you ask after every single run.
- **Run log pane** you can scroll back through, with per-block timing and row
  counts (§7.9), plus a browser notification when a long run finishes.
- **Keep editing during a run** — long runs shouldn't freeze the canvas.

### 8.2 The code editor

Block code is currently a plain `<textarea>`. For a tool whose whole premise
is "blocks contain real Python", that's the weakest link in the experience.

- ⭐ Swap in CodeMirror or Monaco: syntax highlighting, bracket matching,
  auto-indent, find/replace, and a sane tab/indent story.
- Autocomplete against `pl.` and the packet/`ColumnMeta` API — even a static
  stub list helps enormously, and it doubles as the fastest way to teach
  Polars to someone who only knows pandas.
- **Column-name autocomplete from the actual upstream packet** — the tool
  knows the incoming schema, so offer it. Kills the most common typo class.
- Lint/format on save (ruff), and flag a syntax error *before* a run rather
  than surfacing it as a red block.
- Error line markers from the last traceback, in the gutter.
- Diff view: against the previous version of the block, and against an AI
  suggestion (§8.8).

### 8.3 Not losing work

There is currently no autosave, no dirty indicator, no `beforeunload` guard
and no undo — and the delete-block dialog literally says "this can't be
undone". That combination will eventually cost someone an afternoon.

- ⭐ **Autosave + crash recovery**, with a visible saved/unsaved indicator
  and a warning on navigating away dirty.
- ⭐ **Undo/redo across graph edits** (add/delete/move/rewire/param change).
- **Delete guard that shows the blast radius** — "3 downstream blocks will
  lose an input", listing them, instead of a generic confirm.
- **Preserve in-progress AI drafts** and unsaved code edits across a reload;
  losing a half-reviewed LLM draft is infuriating.

### 8.4 Canvas at scale

Fine at 10 blocks; a real PD pipeline is 40–80 and an IFRS 9 chain more.

- ⭐ **Dim-to-path highlight** — select a block, dim everything that isn't
  upstream or downstream of it. The fastest way to comprehend someone else's
  graph, and it directly serves the validator's "what feeds this number"
  question (§6B).
- Search / command palette (Cmd-K) over blocks, params and columns;
  go-to-block by name.
- Minimap — React Flow ships one and it isn't enabled today.
- Auto-layout / tidy, snap to grid, align and distribute.
- Follow-a-wire: click a long wire to highlight both ends and jump between
  them.
- Zoom to fit / zoom to selection; named views or bookmarks on big graphs.
- Lane collapse, and **group a selection into a reusable composite block**
  (huge for "our standard validation battery").
- Copy/paste and duplicate blocks, including across projects.
- Colour/tag blocks; sticky notes for reviewers (§4 review workflow).

### 8.5 Params and configuration

- Column pickers filtered by role and dtype instead of free-text names
  (§7.6), with multi-select for feature lists.
- Expression fields (`filter`'s `expr` especially) with column autocomplete
  and validate-as-you-type rather than failing at run time.
- ⭐ **Effect preview before running** — "this filter drops 1,243 rows
  (4.2%)", "this join matches 98.7% of left rows". Cheap to compute on a
  sample, and it catches the errors that otherwise survive to validation.
  It also happens to be exactly the content the exclusion waterfall (§1.1)
  and the join-quality diagnostics need, so the work is shared.
- Param presets — save a configured block as a named preset to reuse.
- Show which params were auto-filled by role inference vs. set by hand.

### 8.6 Data preview

- Sort, filter and search within the preview; pin/reorder/resize columns.
- ⭐ **Render null distinctly from empty string and from `NaN`.** Sounds
  trivial; it is a genuine and recurring source of silent modelling errors.
- Sensible formatting: thousands separators, aligned decimals, readable
  dates, monospaced numerics, and full precision on demand.
- Per-column mini-histogram/sparkline and fill-rate in the header.
- Copy selection as TSV/markdown/Excel; export the visible view.
- Row count with "showing X of N", and jump-to-row.
- Quick chart from the preview without wiring a plot block.

### 8.7 Roles and metadata ergonomics

- ⭐ **Do role tags survive a source refresh or schema change?** If tagging
  40 features is lost whenever a CSV gains a column, nobody will tag
  anything. Worth treating as correctness, not polish.
- Bulk tagging: multi-select columns, tag in one action; tag by pattern
  (`*_ratio` → feature).
- Role inference suggestions from names, dtypes and sample values, offered
  as a reviewable batch rather than silently applied.
- A project-wide metadata table: every column, role, description, where it
  was created — editable in one place instead of block by block. Doubles as
  the feature dictionary export (§1.2).

### 8.8 The AI loop

It's a differentiator, so its ergonomics matter more than average.

- ⭐ **Show the prompt that will be sent**, and let it be edited before
  sending. Trust, debuggability, and it's how people learn to get good
  results.
- Stream the response — a local model with no feedback for 40 seconds reads
  as a hang.
- **Conversational iteration on a draft** rather than one-shot: "good, but
  use Polars expressions instead of `map_elements`". Currently a reject and
  a full retry.
- Diff + per-hunk accept/reject on a suggested fix, not all-or-nothing.
- **"Explain this block"** — for inherited projects and for reviewers, and
  it feeds §5.3's documentation drafting from the same plumbing.
- Token/cost and latency indicator per call; a running total per session.

### 8.9 Onboarding and reuse

- ⭐ **A demo project that actually runs**, shipped against `sample_data/`.
  The repo has the CSVs but no project JSON — so the first-run experience is
  an empty canvas, which is the hardest possible start.
- **Project templates** — "standard PD development", "IFRS 9 skeleton" — as
  both onboarding and house-standard enforcement.
- Recent files list; drag-and-drop a CSV onto the canvas to create a read
  block pre-configured.
- Empty-state guidance on the canvas and in the inspector.

### 8.10 Output and handoff

- Copy the compiled script to clipboard; syntax-highlight it; diff it
  against the last compile.
- ⭐ Export the canvas as an image (needed by §5 anyway, so it pays twice).
- Copy any result table as markdown for pasting into a review comment or a
  document.

### If I had to pick five

Sample mode (§8.1), autosave + undo (§8.3), a real code editor (§8.2),
"why is this orange" (§8.1), and a runnable demo project (§8.9). Together
they're a fraction of the cost of one item in §3 and they change how the
tool feels to use every single day.

---

## 9. Rough shaping

Not a plan, just my read on where the value/effort ratio sits.

**High value, low effort — good first moves**
- Data profile + data quality rules + exclusion waterfall blocks (§1.1)
- Missing-value treatment block (§1.1) — a real hole today
- Validation battery expansion: ROC/calibration/HL/O-vs-E/CV (§1.4)
- IV / univariate analysis table and correlation/VIF (§1.2)
- Scorecard scaling + scorecard table output (§1.3)
- Evidence pack export (§5.1) and canvas image export (§8)
- Block-level assumption/rationale metadata fields (§4) — cheap now,
  expensive to retrofit once projects exist
- Static HTML dashboard export (§6, option 2)
- **The five at the end of §8** — sample mode, autosave + undo, a real code editor,
  "why is this orange", a runnable demo project. Cheapest items on this
  whole list per unit of daily benefit, and two of them (autosave, undo)
  are really defect-prevention rather than polish

**High value, medium effort**
- Auto-generated model development document (§5.2) + LLM narrative with
  staleness tracking (§5.3)
- Run history / result store (§7.3) and the monitoring view it unlocks
- Semantic project diff + change log (§4)
- In-app dashboard route with present-mode (§6, option 1)
- Optimal/monotonic binning (§1.2)
- Parquet + SQL connectors (§7.5)

**High value, high effort — needs a design decision first**
- **The stochastic core** (§3.1 + §7.1 + §7.2): non-dataframe packets,
  sub-graph iteration, seed discipline, distributions, copulas, risk
  measures. One build that serves bank EC, insurance SCR, op risk, reserving
  bootstrap, IFRS 9 scenarios, CV and bootstrap CIs, and stress testing.
  Because it is shared, it should be costed as core infrastructure rather
  than as a domain feature — the per-domain work on top is comparatively
  thin.
- **The scenario/proxy layer** (§3.6): typed scenario sets, a `ProxyFunction`
  port type, and at least the closed-form and one fitted method behind a
  common interface, plus the proxy validation pack. Not separable from the
  above for any scenario-based capital use case — the nesting problem has to
  be answered the moment you do that work at all. The cheap entry point is
  closed-form/semi-analytic only (which is all IFRS 9 needs); LSMC,
  replicating portfolios and non-normal var-covar are increments on the same
  interface.
- Full IFRS 9 chain: lifetime PD → staging → ECL → attribution (§2.4)
- Lazy/streaming or DuckDB execution (§7.4)

**Explicit "decide whether we care" list**
- *Which* proxy methods beyond closed-form to support, and in what order —
  LSMC, replicating portfolios, non-normal var-covar (§3.6). The interface
  is the commitment; each method is then an increment
- IFRS 17 (§3.10) — recommend no, for now
- Multi-user / access control / server deployment (§4)
- Fairness testing and EU AI Act positioning (§1.4, §4)

**Sequencing note.** Given the overlap, the natural order is: (1) build the
spine, (2) land bank credit EC on it — it's the domain we already have data
and vocabulary for, and closed-form ASRF gives a benchmark to validate the
simulation against, (3) reuse it for IFRS 9 scenarios and bootstrap CIs,
which is nearly free once (1) exists, (4) add insurance structures
(triangles, risk modules, LoBs, risk margin) as blocks. Insurance stops
being a separate programme and becomes an increment.

**Open questions for you**
1. ~~Credit-first or insurance too?~~ Largely dissolved: the shared spine
   means the real question is **do we do stochastic capital work at all?**
   If yes, the build is common and the domain order is a sequencing choice,
   not an architectural fork. If no, §3 collapses and §7.1/§7.2 drop down
   the list (though §7.2 still earns its place for CV, bootstrap and IFRS 9
   scenarios alone).
2. Which capital use case pays first — bank EC/ICAAP, or an insurance
   internal model? Same spine, but it decides which set of blocks and which
   validation benchmarks get built alongside it.
3. Is the tool for *development* only, or also for *production/monitoring
   runs*? Monitoring implies scheduling, result stores and connectors.
4. Single-user local tool, or shared/server? Governance features (§4) only
   half-make-sense locally.
5. Is the compiled `.py` the deliverable handed to an implementation team,
   or is Model-Maker itself meant to be the production runtime? Note this
   gets harder with §3: compiling a reproducible simulation to a standalone
   script means the seed hierarchy has to compile out too.
6. How much does the documentation output need to match a specific existing
   house template — i.e. is template configurability a v1 requirement?
