import type { CSSProperties } from 'react'
import type { FormField, SchemaColumn } from './types'

// A registry block's fields come from the API with the block (block.form,
// built from its BlockSpec and signature -- see modelmaker/forms.py), so
// there's no per-block list to keep here.

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
export function deriveColumnFieldSpecs(params: Record<string, unknown>): FormField[] {
  return Object.keys(params)
    .filter((key) => key.endsWith('_col') && (params[key] === null || typeof params[key] === 'string'))
    .sort()
    .map((key) => ({
      key,
      label: prettifyColumnParamLabel(key),
      kind: 'column' as const,
      placeholder: '',
      step: null,
      options: [],
      auto_role: null,
      default: null,
    }))
}

export function ParamsForm({
  spec,
  paramsText,
  setParamsText,
  columns,
  disabled,
}: {
  spec: FormField[]
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
                placeholder={f.placeholder || (typeof f.default === 'string' ? f.default : '')}
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
                placeholder={typeof f.default === 'number' ? String(f.default) : ''}
                onChange={(e) => update(f.key, e.target.value === '' ? undefined : Number(e.target.value))}
                style={fieldStyle}
              />
            )}
            {f.kind === 'checkbox' && (
              <input
                type="checkbox"
                disabled={disabled}
                checked={typeof value === 'boolean' ? value : Boolean(f.default)}
                onChange={(e) => update(f.key, e.target.checked)}
              />
            )}
            {f.kind === 'select' && (
              <select
                disabled={disabled}
                value={typeof value === 'string' ? value : typeof f.default === 'string' ? f.default : ''}
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
            {f.kind === 'column' && f.auto_role && (() => {
              const autoResolved = columns.find((c) => c.role === f.auto_role)?.name
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
            {f.kind === 'column' && !f.auto_role && (
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
