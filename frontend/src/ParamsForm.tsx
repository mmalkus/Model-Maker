import type { CSSProperties } from 'react'
import { EnvVarField } from './EnvVarField'
import type { FormField, SchemaColumn } from './types'

// A block's fields come from the API with the block (block.form, built from
// its BLOCKS "form" and its signature -- see modelmaker/forms.py), so
// there's no per-block list to keep here.

export function ParamsForm({
  spec,
  paramsText,
  setParamsText,
  columns,
  disabled,
  onBrowse,
}: {
  spec: FormField[]
  paramsText: string
  setParamsText: (text: string) => void
  columns: SchemaColumn[]
  disabled: boolean
  // A `path` field's Browse... button: pick a file (filtered to the field's
  // extensions) and set it as the param.
  onBrowse?: (key: string, ext: string) => void
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
            <div style={{ color: 'var(--mm-fg-3)', marginBottom: 2 }}>{f.label}</div>
            {f.kind === 'path' && (
              <div style={{ display: 'flex', gap: 6 }}>
                <input
                  disabled={disabled}
                  value={typeof value === 'string' ? value : ''}
                  placeholder={f.placeholder || '(no file selected)'}
                  onChange={(e) => update(f.key, e.target.value)}
                  style={{ ...fieldStyle, fontFamily: 'monospace' }}
                />
                {onBrowse && (
                  <button disabled={disabled} onClick={() => onBrowse(f.key, f.options[0] ?? '')}>
                    Browse...
                  </button>
                )}
              </div>
            )}
            {f.kind === 'sql' && (
              <textarea
                disabled={disabled}
                rows={4}
                value={typeof value === 'string' ? value : ''}
                placeholder={f.placeholder}
                onChange={(e) => update(f.key, e.target.value)}
                style={{ ...fieldStyle, fontFamily: 'monospace' }}
              />
            )}
            {f.kind === 'env' && (
              <>
                <input
                  disabled={disabled}
                  value={typeof value === 'string' ? value : ''}
                  placeholder={f.placeholder}
                  onChange={(e) => update(f.key, e.target.value)}
                  style={{ ...fieldStyle, fontFamily: 'monospace' }}
                />
                {typeof value === 'string' && value && <EnvVarField name={value} />}
              </>
            )}
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
