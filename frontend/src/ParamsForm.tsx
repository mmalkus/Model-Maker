import type { CSSProperties } from 'react'

type FieldSpec =
  | { key: string; label: string; kind: 'text'; placeholder?: string }
  | { key: string; label: string; kind: 'number'; step?: number }
  | { key: string; label: string; kind: 'select'; options: string[] }
  | { key: string; label: string; kind: 'column' }
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
    { key: 'target', label: 'Target column', kind: 'column' },
    { key: 'features', label: 'Feature columns', kind: 'columns' },
    { key: 'family', label: 'Family', kind: 'select', options: ['gaussian', 'poisson', 'gamma', 'inverse_gaussian'] },
    { key: 'alpha', label: 'Regularization (alpha)', kind: 'number', step: 0.01 },
  ],
  logistic_regression: [
    { key: 'target', label: 'Target column (binary)', kind: 'column' },
    { key: 'features', label: 'Feature columns', kind: 'columns' },
    { key: 'C', label: 'Inverse regularization (C)', kind: 'number', step: 0.1 },
    { key: 'max_iter', label: 'Max iterations', kind: 'number' },
  ],
  woe_transform: [
    { key: 'col', label: 'Column to transform', kind: 'column' },
    { key: 'target', label: 'Target column (binary)', kind: 'column' },
    { key: 'bins', label: 'Bins (numeric columns)', kind: 'number' },
  ],
  ks_test: [
    { key: 'score_col', label: 'Score column', kind: 'column' },
    { key: 'target_col', label: 'Target column (binary)', kind: 'column' },
  ],
  auc_gini: [
    { key: 'score_col', label: 'Score column', kind: 'column' },
    { key: 'target_col', label: 'Target column (binary)', kind: 'column' },
  ],
  psi_test: [
    { key: 'col', label: 'Column to compare', kind: 'column' },
    { key: 'bins', label: 'Bins', kind: 'number' },
  ],
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
  columns: string[]
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

  const fieldStyle: CSSProperties = { width: '100%', fontSize: 11, boxSizing: 'border-box' }

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
            {f.kind === 'column' && (
              <select
                disabled={disabled}
                value={typeof value === 'string' ? value : ''}
                onChange={(e) => update(f.key, e.target.value)}
                style={fieldStyle}
              >
                <option value="">(none)</option>
                {columns.map((c) => (
                  <option key={c} value={c}>
                    {c}
                  </option>
                ))}
                {typeof value === 'string' && value && !columns.includes(value) && <option value={value}>{value}</option>}
              </select>
            )}
            {f.kind === 'columns' &&
              (columns.length > 0 ? (
                <div style={{ display: 'flex', flexWrap: 'wrap', gap: 8 }}>
                  {columns.map((c) => {
                    const selected = Array.isArray(value) && value.includes(c)
                    return (
                      <label key={c} style={{ display: 'flex', alignItems: 'center', gap: 3, fontSize: 11 }}>
                        <input
                          type="checkbox"
                          disabled={disabled}
                          checked={selected}
                          onChange={(e) => {
                            const cur = Array.isArray(value) ? (value as string[]) : []
                            update(f.key, e.target.checked ? [...cur, c] : cur.filter((v) => v !== c))
                          }}
                        />
                        {c}
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
