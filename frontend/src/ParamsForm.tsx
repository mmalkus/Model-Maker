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
  ],
  write_csv: [{ key: 'filename', label: 'Output filename', kind: 'text', placeholder: 'output.csv' }],
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
  ],
  assign_rating_grade: [{ key: 'score_col', label: 'Score column (defaults to the scale’s own)', kind: 'column', autoRole: 'predicted' }],
  rating_summary: [
    { key: 'grade_col', label: 'Grade column', kind: 'column' },
    { key: 'target_col', label: 'Target column (binary)', kind: 'column', autoRole: 'target' },
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
