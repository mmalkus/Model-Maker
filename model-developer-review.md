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

## 3. Insurance internal models

The shape of the work is different enough from credit that some engine
changes (§7) are prerequisites rather than nice-to-haves. Worth deciding
early whether insurance is a *target* or a *maybe*, because the
"distribution + simulation + aggregation" axis is a real build.

### 3.1 Pricing / underwriting

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

### 3.2 Reserving

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

### 3.3 Capital / internal model specifics (Solvency II flavour)

- Distribution fitting block (MLE across candidate distributions, GoF tests:
  KS, AD, chi-square, QQ plots, and a comparison table) — including EVT /
  GPD tail fitting.
- Dependency structure: correlation matrices, copulas (Gaussian, t, Clayton,
  Gumbel), tail-dependence diagnostics.
- Monte Carlo simulation engine block: N simulations, seed, antithetic /
  variance reduction, convergence diagnostics.
- Aggregation: risk-module aggregation with correlation matrix, diversification
  benefit calculation and allocation back to modules/LoBs (Euler, Shapley).
- Risk measures: VaR 99.5%, TVaR, with simulation error / confidence bands.
- Loss-absorbing capacity, risk-margin calculation.
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

### 3.4 IFRS 17 adjacency (probably out of scope, worth naming)

Building blocks for fulfilment cash flows, risk adjustment (confidence-level
or cost-of-capital), CSM roll-forward, and cohort/grouping. Very large scope;
mention only so we can explicitly say "not now".

### 3.5 What the engine lacks for this work

- No `model` / `distribution` / `simulation` port type flowing over a wire
  with the same richness as `DataFramePacket` (there's a typed-port notion in
  the plan, but the packet is dataframe-shaped).
- No way to run a sub-graph N times (scenarios, simulations, bootstrap folds)
  — see §7.2. Without it, every stochastic method is a hand-written block.
- Memory model: simulation output is wide/long and may exceed RAM; today
  everything is an in-memory Polars frame.

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
  and §3.2 actuarial selections.
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
   distributions, simulation results, metric bundles. Currently the packet is
   dataframe-shaped; ports are typed in the plan but the richness (metadata,
   caching, preview, compilation) is dataframe-specific.
2. **Sub-graph iteration** — run a branch once per scenario / per fold / per
   simulation / per segment, then collect. Needed for macro scenarios (§2.4),
   CV and bootstrap (§1.4), simulation (§3.3), and per-segment models. This
   is the single biggest structural gap and it has knock-on effects on the
   cache key, the compiler, and the UI (how do you draw a loop on a canvas?).
   Worth a design note of its own.
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

## 8. Smaller developer-experience items

- Search / palette over blocks on a large canvas; go-to-block by name.
- Copy/paste and duplicate blocks; group a selection into a reusable
  **composite block** (huge for "our standard validation battery").
- **Reusable block library / templates** shared across projects — a
  "standard IFRS 9 skeleton" or "standard PD dev pipeline" project template
  is a strong onboarding story.
- Notes/annotations on the canvas (sticky notes for reviewers).
- Undo/redo across graph edits.
- Diff view for a block's code between versions.
- Better error surfaces: the failing row/column, not just the traceback.
- Data preview improvements: filter/sort in the preview, value distribution
  sparkline per column, quick chart from the preview.
- Column-role bulk tagging (tag 40 features without 40 clicks) and role
  inference suggestions from names/dtypes.
- Export the canvas as an image for documents (needed by §5 anyway).

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

**High value, medium effort**
- Auto-generated model development document (§5.2) + LLM narrative with
  staleness tracking (§5.3)
- Run history / result store (§7.3) and the monitoring view it unlocks
- Semantic project diff + change log (§4)
- In-app dashboard route with present-mode (§6, option 1)
- Optimal/monotonic binning (§1.2)
- Parquet + SQL connectors (§7.5)

**High value, high effort — needs a design decision first**
- Sub-graph iteration / scenario loops (§7.2) → unlocks IFRS 9 macro
  scenarios, CV, bootstrap, simulation, per-segment models
- Full IFRS 9 chain: lifetime PD → staging → ECL → attribution (§2.4)
- Non-dataframe packet types (§7.1) → prerequisite for most of insurance
- Lazy/streaming or DuckDB execution (§7.4)

**Explicit "decide whether we care" list**
- Insurance internal models as a target at all (§3) — big, and the engine
  work is real
- IFRS 17 (§3.4) — recommend no, for now
- Multi-user / access control / server deployment (§4)
- Fairness testing and EU AI Act positioning (§1.4, §4)

**Open questions for you**
1. Primary audience: credit-risk-first with insurance later, or both from
   the start? It changes §7.1/§7.2 priority a lot.
2. Is the tool for *development* only, or also for *production/monitoring
   runs*? Monitoring implies scheduling, result stores and connectors.
3. Single-user local tool, or shared/server? Governance features (§4) only
   half-make-sense locally.
4. Is the compiled `.py` the deliverable handed to an implementation team,
   or is Model-Maker itself meant to be the production runtime?
5. How much does the documentation output need to match a specific existing
   house template — i.e. is template configurability a v1 requirement?
