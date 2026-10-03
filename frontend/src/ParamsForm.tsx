import type { CSSProperties } from 'react'
import type { SchemaColumn } from './types'

export type FieldSpec =
  | { key: string; label: string; kind: 'text'; placeholder?: string }
  | { key: string; label: string; kind: 'number'; step?: number }
  | { key: string; label: string; kind: 'select'; options: string[] }
  // autoRole marks a field that should default to whichever input column
  // currently carries that role (see packet.resolve_role_column) when the
  // param is left out of `params` entirely -- the field then offers an
  // "Auto" option that clears the key rather than setting it to '', putting
  // it back in that dynamically-resolved state. 'target' picks up the
  // role=target column; 'predicted' picks up whichever column a modelling
  // block upstream tagged role=predicted (see blocks/modelling.py).
  | { key: string; label: string; kind: 'column'; autoRole?: 'target' | 'predicted' }
  | { key: string; label: string; kind: 'columns' }

// Declarative field lists for the block categories common enough to be
// worth dropdowns/inputs over raw JSON. Anything not listed here (custom
// AI-authored blocks, or a standard block like groupby_agg whose params
// don't map cleanly to flat fields) falls back to the JSON textarea.
export const PARAM_SPECS: Record<string, FieldSpec[]> = {
  filter: [{ key: 'expr', label: 'Filter expression (SQL)', kind: 'text', placeholder: 'age > 30 and region = \'West\'' }],
  select: [{ key: 'cols', label: 'Columns to keep', kind: 'columns' }],
  join: [
    { key: 'on', label: 'Join on', kind: 'columns' },
    { key: 'how', label: 'How', kind: 'select', options: ['inner', 'left', 'right', 'outer', 'semi', 'anti', 'cross'] },
  ],
  train_test_split: [
    { key: 'test_size', label: 'Test size (fraction)', kind: 'number', step: 0.05 },
    { key: 'seed', label: 'Random seed', kind: 'number' },
    { key: 'stratify_col', label: "Stratify by (optional, e.g. the default flag)", kind: 'column' },
  ],
  write_csv: [{ key: 'filename', label: 'Output filename', kind: 'text', placeholder: 'output.csv' }],
  read_excel: [{ key: 'sheet', label: 'Sheet name (optional -- defaults to the first sheet)', kind: 'text' }],
  read_sql: [
    {
      key: 'connection_env',
      label: 'Connection env var (holds the DB URI, e.g. postgresql://user:pass@host/db)',
      kind: 'text',
      placeholder: 'WAREHOUSE_DB_URL',
    },
    { key: 'query', label: 'SQL query', kind: 'text', placeholder: 'SELECT * FROM loans' },
    {
      key: 'probe_query',
      label: 'Check-for-changes query (optional, e.g. SELECT COUNT(*) FROM loans)',
      kind: 'text',
    },
  ],
  generate_image: [
    { key: 'kind', label: 'Chart type', kind: 'select', options: ['hist', 'bar', 'scatter', 'line'] },
    { key: 'x', label: 'X column', kind: 'column' },
    { key: 'y', label: 'Y column (bar / scatter / line)', kind: 'column' },
    { key: 'bins', label: 'Bins (hist)', kind: 'number' },
    { key: 'title', label: 'Title', kind: 'text' },
  ],
  glm_fit: [
    { key: 'target', label: 'Target column', kind: 'column', autoRole: 'target' },
    { key: 'features', label: 'Feature columns', kind: 'columns' },
    { key: 'family', label: 'Family', kind: 'select', options: ['gaussian', 'poisson', 'gamma', 'inverse_gaussian'] },
    { key: 'alpha', label: 'Regularization (alpha)', kind: 'number', step: 0.01 },
  ],
  logistic_regression: [
    { key: 'target', label: 'Target column (binary)', kind: 'column', autoRole: 'target' },
    { key: 'features', label: 'Feature columns', kind: 'columns' },
    { key: 'C', label: 'Inverse regularization (C)', kind: 'number', step: 0.1 },
    { key: 'max_iter', label: 'Max iterations', kind: 'number' },
  ],
  woe_transform: [
    { key: 'col', label: 'Column to transform', kind: 'column' },
    { key: 'target', label: 'Target column (binary)', kind: 'column', autoRole: 'target' },
    { key: 'bins', label: 'Bins (numeric columns)', kind: 'number' },
  ],
  ks_test: [
    { key: 'score_col', label: 'Score column', kind: 'column', autoRole: 'predicted' },
    { key: 'target_col', label: 'Target column (binary)', kind: 'column', autoRole: 'target' },
  ],
  auc_gini: [
    { key: 'score_col', label: 'Score column', kind: 'column', autoRole: 'predicted' },
    { key: 'target_col', label: 'Target column (binary)', kind: 'column', autoRole: 'target' },
  ],
  psi_test: [
    { key: 'col', label: 'Column to compare', kind: 'column' },
    { key: 'bins', label: 'Bins', kind: 'number' },
  ],
  fit_master_scale: [
    { key: 'score_col', label: 'Score column', kind: 'column', autoRole: 'predicted' },
    { key: 'target_col', label: 'Target column (binary)', kind: 'column', autoRole: 'target' },
    { key: 'n_grades', label: 'Number of grades', kind: 'number' },
    {
      key: 'algorithm',
      label: 'Algorithm',
      kind: 'select',
      options: ['quantile', 'equal_width', 'monotonic_default_rate'],
    },
    { key: 'min_grade_share', label: "Minimum grade share (e.g. 0.03)", kind: 'number', step: 0.01 },
  ],
  assign_rating_grade: [{ key: 'score_col', label: 'Score column (defaults to the scale’s own)', kind: 'column', autoRole: 'predicted' }],
  rating_summary: [
    { key: 'grade_col', label: 'Grade column', kind: 'column' },
    { key: 'target_col', label: 'Target column (binary)', kind: 'column', autoRole: 'target' },
  ],
  lgd_regression: [
    { key: 'target', label: 'Target column (LGD or CCF, in [0, 1])', kind: 'column', autoRole: 'target' },
    { key: 'features', label: 'Feature columns', kind: 'columns' },
    { key: 'max_iter', label: 'Max iterations', kind: 'number' },
    { key: 'tol', label: 'Convergence tolerance', kind: 'number', step: 1e-8 },
  ],
  compute_lgd: [
    { key: 'ead_col', label: 'EAD column', kind: 'column' },
    { key: 'recovered_col', label: 'Recovered amount column', kind: 'column' },
    { key: 'cost_col', label: 'Workout cost column (optional)', kind: 'column' },
    { key: 'floor', label: 'Floor', kind: 'number', step: 0.05 },
    { key: 'cap', label: 'Cap', kind: 'number', step: 0.05 },
  ],
  // performing_only (boolean) stays in the JSON textarea, see the note above fit_distribution.
  forward_default_flag: [
    { key: 'id_col', label: 'Account id column', kind: 'column' },
    { key: 'date_col', label: 'Month column (date or YYYYMM)', kind: 'column' },
    { key: 'default_col', label: 'Current default status column', kind: 'column' },
    { key: 'horizon', label: 'Horizon (months)', kind: 'number' },
    { key: 'flag_col', label: 'Flag column name (optional)', kind: 'text', placeholder: 'default_12m' },
    { key: 'incomplete', label: 'Incomplete outcome windows', kind: 'select', options: ['drop', 'null'] },
  ],
  compute_ccf: [
    { key: 'limit_col', label: 'Limit column', kind: 'column' },
    { key: 'balance_ref_col', label: 'Balance at reference date column', kind: 'column' },
    { key: 'balance_default_col', label: 'Balance at default column', kind: 'column' },
    { key: 'floor', label: 'Floor', kind: 'number', step: 0.05 },
    { key: 'cap', label: 'Cap', kind: 'number', step: 0.05 },
    { key: 'no_headroom', label: "No undrawn headroom at reference", kind: 'select', options: ['zero', 'null'] },
  ],
  continuous_accuracy: [
    { key: 'actual_col', label: 'Actual column', kind: 'column', autoRole: 'target' },
    { key: 'predicted_col', label: 'Predicted column', kind: 'column', autoRole: 'predicted' },
  ],
  bucketed_calibration: [
    { key: 'actual_col', label: 'Actual column', kind: 'column', autoRole: 'target' },
    { key: 'predicted_col', label: 'Predicted column', kind: 'column', autoRole: 'predicted' },
    { key: 'bins', label: 'Bins', kind: 'number' },
  ],
  // Stochastic engine blocks (see /stochastic-engine-proposal.md and
  // modelmaker/blocks/stochastic.py). Boolean params (spliced_tail,
  // keep_paths) and list-of-float params (alpha_levels) are deliberately
  // left out of these specs -- there's no FieldSpec 'boolean' kind, and
  // modelling one as a 'select' of the strings 'true'/'false' would set a
  // truthy *string* value ('false' is still truthy in Python), so both
  // fall back to the JSON textarea where they can be real JSON literals.
  fit_distribution: [{ key: 'column', label: 'Column to fit', kind: 'column' }],
  sample_distribution: [
    { key: 'n', label: 'Number of samples', kind: 'number' },
    { key: 'seed', label: 'Random seed', kind: 'number' },
  ],
  build_dependency: [
    { key: 'columns', label: 'Columns', kind: 'columns' },
    { key: 'copula_type', label: 'Copula', kind: 'select', options: ['gaussian', 't', 'clayton', 'gumbel'] },
    { key: 'dof', label: 'Degrees of freedom (t copula)', kind: 'number' },
    { key: 'theta', label: 'Theta (Clayton / Gumbel)', kind: 'number', step: 0.1 },
  ],
  simulate_op_risk_lda: [
    { key: 'n_paths', label: 'Number of paths', kind: 'number' },
    { key: 'chunk_size', label: 'Chunk size', kind: 'number' },
    { key: 'frequency_family', label: 'Frequency family', kind: 'select', options: ['poisson', 'negbinom'] },
    { key: 'frequency_mean', label: 'Frequency mean', kind: 'number', step: 0.1 },
    { key: 'frequency_dispersion', label: 'Frequency dispersion (negbinom)', kind: 'number', step: 0.1 },
    { key: 'n_bootstrap', label: 'Bootstrap replicates', kind: 'number' },
    { key: 'seed', label: 'Random seed', kind: 'number' },
  ],
  risk_measures: [
    { key: 'column', label: 'Loss column', kind: 'column' },
    { key: 'n_bootstrap', label: 'Bootstrap replicates', kind: 'number' },
    { key: 'seed', label: 'Random seed', kind: 'number' },
  ],
  asrf_economic_capital: [
    { key: 'pd_col', label: 'PD column', kind: 'column' },
    { key: 'lgd_col', label: 'LGD column', kind: 'column' },
    { key: 'ead_col', label: 'EAD column', kind: 'column' },
    { key: 'confidence', label: 'Confidence level', kind: 'number', step: 0.001 },
    // 'correlation' is a float OR the string "basel_corporate" -- a
    // 'select' field can't offer "pick a fixed number, or this preset", so
    // it's left to the JSON textarea (e.g. "basel_corporate" or 0.15).
  ],
  simulate_credit_portfolio: [
    { key: 'pd_col', label: 'PD column', kind: 'column' },
    { key: 'lgd_col', label: 'LGD column', kind: 'column' },
    { key: 'ead_col', label: 'EAD column', kind: 'column' },
    { key: 'sector_col', label: 'Sector column (must match the dependency labels)', kind: 'column' },
    { key: 'name_col', label: 'Segment name column (optional)', kind: 'column' },
    { key: 'n_paths', label: 'Number of paths', kind: 'number' },
    { key: 'chunk_size', label: 'Chunk size', kind: 'number' },
    { key: 'n_bootstrap', label: 'Bootstrap replicates', kind: 'number' },
    { key: 'seed', label: 'Random seed', kind: 'number' },
    // 'correlation' -- see asrf_economic_capital above, same reason it's
    // left to the JSON textarea.
  ],
  aggregate_simulation: [
    { key: 'n_bootstrap', label: 'Bootstrap replicates', kind: 'number' },
    { key: 'seed', label: 'Random seed', kind: 'number' },
    {
      key: 'contributions_method',
      label: 'Contributions',
      kind: 'select',
      options: ['euler', 'shapley'],
    },
  ],
  fit_proxy: [
    { key: 'value_col', label: 'Value column (polynomial)', kind: 'column' },
    { key: 'method', label: 'Method', kind: 'select', options: ['closed_form', 'polynomial'] },
    { key: 'expr', label: 'Expression (closed_form)', kind: 'text', placeholder: 'x + 0.5 * y' },
    { key: 'degree', label: 'Degree (polynomial)', kind: 'number' },
    { key: 'regressor', label: 'Regressor (polynomial)', kind: 'select', options: ['ridge', 'lasso', 'ols'] },
    { key: 'alpha', label: 'Regularization (ridge / lasso)', kind: 'number', step: 0.1 },
  ],
  validate_proxy: [{ key: 'value_col', label: 'Actual value column', kind: 'column' }],
  var_covar_aggregate: [
    { key: 'method', label: 'Method', kind: 'select', options: ['normal', 'cornish_fisher', 'moment_matching', 'delta_gamma_copula'] },
    { key: 'bump_size', label: 'Bump size', kind: 'number', step: 0.001 },
    { key: 'n_mc', label: 'Monte Carlo draws (non-normal methods)', kind: 'number' },
    { key: 'seed', label: 'Random seed', kind: 'number' },
  ],
  // Graph fan-out (see /stochastic-engine-proposal.md S4). There's no
  // FieldSpec kind for "pick another block on the canvas", so `collect`'s
  // `iterate_block` param -- the id of its paired `iterate` block -- stays
  // a plain text field; find the id from the `iterate` block's inspector
  // panel. A visually distinguished fan-out region on the canvas itself is
  // a follow-up (see the proposal's Implementation status section).
  iterate: [
    { key: 'n_iterations', label: 'Iterations', kind: 'number' },
    { key: 'iterator', label: 'Iterator', kind: 'select', options: ['bootstrap_resample', 'scenario_row'] },
    { key: 'seed', label: 'Random seed', kind: 'number' },
  ],
  collect: [
    { key: 'iterate_block', label: "Paired 'iterate' block id", kind: 'text', placeholder: 'b_042' },
    { key: 'reducer', label: 'Reducer', kind: 'select', options: ['concat', 'risk_measures'] },
    { key: 'value_col', label: 'Value column (risk_measures)', kind: 'column' },
    { key: 'n_bootstrap', label: 'Bootstrap replicates (risk_measures)', kind: 'number' },
    { key: 'seed', label: 'Random seed (risk_measures)', kind: 'number' },
  ],
  // Binning / univariate, calibration, LGD/CCF/EAD and back-testing blocks.
  time_split: [
    { key: 'date_col', label: "Date column", kind: 'column' },
    { key: 'cutoff', label: "Cut-off date (first out-of-time day)", kind: 'text', placeholder: "2025-01-01" },
    { key: 'oot_end', label: "Out-of-time end date (optional, exclusive)", kind: 'text', placeholder: "2026-01-01" },
  ],
  apply_binning: [
    { key: 'output', label: "Output", kind: 'select', options: ['woe', 'target_mean', 'bin', 'both'] },
    { key: 'features', label: "Features (optional -- defaults to every binned feature)", kind: 'columns' },
  ],
  scorecard_table: [
    { key: 'base_score', label: "Base score", kind: 'number' },
    { key: 'base_odds', label: "Base odds (good:bad)", kind: 'number' },
    { key: 'pdo', label: "Points to double the odds", kind: 'number' },
  ],
  characteristic_stability: [
    { key: 'features', label: "Features (optional -- defaults to all shared columns)", kind: 'columns' },
    { key: 'bins', label: "Bins (numeric features)", kind: 'number' },
  ],
  target_trend: [
    { key: 'date_col', label: "Date column", kind: 'column' },
    { key: 'target_col', label: "Target column", kind: 'column', autoRole: 'target' },
    { key: 'period', label: "Period", kind: 'select', options: ['month', 'quarter', 'year'] },
    { key: 'features', label: "Features to trend (optional)", kind: 'columns' },
  ],
  bin_chart: [
    { key: 'feature', label: "Feature", kind: 'text', placeholder: "credit_score" },
    { key: 'title', label: "Title (optional)", kind: 'text' },
  ],
  stepwise_selection: [
    { key: 'target', label: "Target column (binary, or LGD/CCF in [0, 1])", kind: 'column', autoRole: 'target' },
    { key: 'features', label: "Candidate features", kind: 'columns' },
    { key: 'direction', label: "Direction", kind: 'select', options: ['both', 'forward', 'backward'] },
    { key: 'p_enter', label: "p-value to enter", kind: 'number', step: 0.01 },
    { key: 'p_remove', label: "p-value to remove", kind: 'number', step: 0.01 },
    { key: 'max_features', label: "Max features (optional)", kind: 'number' },
  ],
  calibrate_model: [
    { key: 'central_tendency', label: "Central tendency (target mean, e.g. long-run default rate)", kind: 'number', step: 0.001 },
  ],
  margin_of_conservatism: [
    { key: 'predicted_col', label: "Estimate column", kind: 'column', autoRole: 'predicted' },
    { key: 'add_on', label: "Add-on", kind: 'number', step: 0.01 },
    { key: 'multiplier', label: "Multiplier", kind: 'number', step: 0.05 },
    { key: 'floor', label: "Floor (optional)", kind: 'number', step: 0.0001 },
    { key: 'cap', label: "Cap (optional)", kind: 'number', step: 0.05 },
    { key: 'output_col', label: "Output column", kind: 'text', placeholder: "predicted_moc" },
  ],
  discount_recoveries: [
    { key: 'id_col', label: "Facility id column (both inputs)", kind: 'column' },
    { key: 'default_date_col', label: "Default date column (facilities)", kind: 'column' },
    { key: 'cf_date_col', label: "Cash-flow date column (cash flows)", kind: 'column' },
    { key: 'recovery_col', label: "Recovery amount column", kind: 'column' },
    { key: 'cost_col', label: "Workout cost column (optional)", kind: 'column' },
    { key: 'annual_rate', label: "Annual discount rate", kind: 'number', step: 0.005 },
    { key: 'rate_col', label: "Per-facility rate column (optional, overrides the rate)", kind: 'column' },
  ],
  compute_ead: [
    { key: 'balance_col', label: "Drawn balance column", kind: 'column' },
    { key: 'limit_col', label: "Limit column", kind: 'column' },
    { key: 'predicted_col', label: "CCF column", kind: 'column', autoRole: 'predicted' },
    { key: 'output_col', label: "Output column", kind: 'text', placeholder: "ead_predicted" },
  ],
  long_run_average: [
    { key: 'target_col', label: "Realised value column (LGD, CCF or default flag)", kind: 'column', autoRole: 'target' },
    { key: 'date_col', label: "Default date column", kind: 'column' },
    { key: 'weight_col', label: "Exposure weight column (optional, e.g. EAD)", kind: 'column' },
    { key: 'period', label: "Period", kind: 'select', options: ['year', 'quarter', 'month'] },
  ],
  grade_backtest: [
    { key: 'grade_col', label: "Grade column", kind: 'column' },
    { key: 'target_col', label: "Default flag column", kind: 'column', autoRole: 'target' },
    { key: 'pd_col', label: "Grade PD column (e.g. grade_pd)", kind: 'column' },
    { key: 'confidence', label: "Confidence level", kind: 'number', step: 0.005 },
  ],
}

function prettifyColumnParamLabel(key: string): string {
  const base = key.replace(/_col$/, '').replace(/_/g, ' ')
  return `${base.charAt(0).toUpperCase()}${base.slice(1)} column`
}

// AI-drafted (custom) blocks name every existing input column they read as a
// `<name>_col` string parameter defaulting to that column's real name (see
// llm/prompts.CONTRACT) instead of hardcoding it into the function body --
// that's what lets a block get re-pointed at a differently-named upstream
// column without touching generated code. Turn each such param into a
// `column` field (a dropdown of the block's real input columns) instead of
// leaving it in the raw JSON textarea, so re-wiring it is a pick from a list
// rather than free-text guesswork -- and a stale value (e.g. after an
// upstream rename) shows up as an extra, clearly-off option rather than
// silently failing at run time.
export function deriveColumnFieldSpecs(params: Record<string, unknown>): FieldSpec[] {
  return Object.keys(params)
    .filter((key) => key.endsWith('_col') && (params[key] === null || typeof params[key] === 'string'))
    .sort()
    .map((key) => ({ key, label: prettifyColumnParamLabel(key), kind: 'column' as const }))
}

export function ParamsForm({
  spec,
  paramsText,
  setParamsText,
  columns,
  disabled,
}: {
  spec: FieldSpec[]
  paramsText: string
  setParamsText: (text: string) => void
  columns: SchemaColumn[]
  disabled: boolean
}) {
  let params: Record<string, unknown> = {}
  try {
    params = JSON.parse(paramsText || '{}')
  } catch {
    params = {}
  }

  const update = (key: string, value: unknown) => {
    setParamsText(JSON.stringify({ ...params, [key]: value }, null, 2))
  }

  // Removes the key entirely (as opposed to setting it to '' / undefined),
  // putting an autoRole field back into "follow the tagged column" mode --
  // see FieldSpec.autoRole and Runner.run_block's dynamic target resolution.
  const clear = (key: string) => {
    const rest = { ...params }
    delete rest[key]
    setParamsText(JSON.stringify(rest, null, 2))
  }

  const fieldStyle: CSSProperties = { width: '100%', fontSize: 11, boxSizing: 'border-box' }
  const AUTO = '__auto__'

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
      {spec.map((f) => {
        const value = params[f.key]
        return (
          <div key={f.key}>
            <div style={{ color: '#374151', marginBottom: 2 }}>{f.label}</div>
            {f.kind === 'text' && (
              <input
                disabled={disabled}
                value={typeof value === 'string' ? value : ''}
                placeholder={f.placeholder}
                onChange={(e) => update(f.key, e.target.value)}
                style={fieldStyle}
              />
            )}
            {f.kind === 'number' && (
              <input
                type="number"
                step={f.step ?? 'any'}
                disabled={disabled}
                value={typeof value === 'number' ? value : ''}
                onChange={(e) => update(f.key, e.target.value === '' ? undefined : Number(e.target.value))}
                style={fieldStyle}
              />
            )}
            {f.kind === 'select' && (
              <select
                disabled={disabled}
                value={typeof value === 'string' ? value : ''}
                onChange={(e) => update(f.key, e.target.value)}
                style={fieldStyle}
              >
                <option value="" disabled>
                  (choose)
                </option>
                {f.options.map((o) => (
                  <option key={o} value={o}>
                    {o}
                  </option>
                ))}
              </select>
            )}
            {f.kind === 'column' && f.autoRole && (() => {
              const autoResolved = columns.find((c) => c.role === f.autoRole)?.name
              const isAuto = !(f.key in params)
              return (
                <select
                  disabled={disabled}
                  value={isAuto ? AUTO : typeof value === 'string' ? value : ''}
                  onChange={(e) => (e.target.value === AUTO ? clear(f.key) : update(f.key, e.target.value))}
                  style={fieldStyle}
                >
                  <option value={AUTO}>{autoResolved ? `Auto (${autoResolved})` : 'Auto (no target tagged upstream)'}</option>
                  {columns.map((c) => (
                    <option key={c.name} value={c.name}>
                      {c.name}
                    </option>
                  ))}
                  {typeof value === 'string' && value && !columns.some((c) => c.name === value) && (
                    <option value={value}>{value}</option>
                  )}
                </select>
              )
            })()}
            {f.kind === 'column' && !f.autoRole && (
              <select
                disabled={disabled}
                value={typeof value === 'string' ? value : ''}
                onChange={(e) => update(f.key, e.target.value)}
                style={fieldStyle}
              >
                <option value="">(none)</option>
                {columns.map((c) => (
                  <option key={c.name} value={c.name}>
                    {c.name}
                  </option>
                ))}
                {typeof value === 'string' && value && !columns.some((c) => c.name === value) && (
                  <option value={value}>{value}</option>
                )}
              </select>
            )}
            {f.kind === 'columns' &&
              (columns.length > 0 ? (
                <div style={{ display: 'flex', flexWrap: 'wrap', gap: 8 }}>
                  {columns.map((c) => {
                    const selected = Array.isArray(value) && value.includes(c.name)
                    return (
                      <label key={c.name} style={{ display: 'flex', alignItems: 'center', gap: 3, fontSize: 11 }}>
                        <input
                          type="checkbox"
                          disabled={disabled}
                          checked={selected}
                          onChange={(e) => {
                            const cur = Array.isArray(value) ? (value as string[]) : []
                            update(f.key, e.target.checked ? [...cur, c.name] : cur.filter((v) => v !== c.name))
                          }}
                        />
                        {c.name}
                      </label>
                    )
                  })}
                </div>
              ) : (
                <input
                  disabled={disabled}
                  placeholder="comma-separated column names (input schema not available yet)"
                  value={Array.isArray(value) ? (value as string[]).join(', ') : ''}
                  onChange={(e) =>
                    update(
                      f.key,
                      e.target.value
                        .split(',')
                        .map((s) => s.trim())
                        .filter(Boolean),
                    )
                  }
                  style={fieldStyle}
                />
              ))}
          </div>
        )
      })}
    </div>
  )
}
