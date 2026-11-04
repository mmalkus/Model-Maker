import type { CSSProperties } from 'react'
import { useEffect, useState } from 'react'
import { api } from './api'

// The picker of a 'dataset' or 'source' field: the workspace's datasets
// (GET /api/datasets) or the administrator's data sources (GET
// /api/sources), by name. The embedding application provides both (see
// modelmaker/datasets.py); a workspace without any shows an empty list.
export function DataPicker({
  kind,
  value,
  disabled,
  onChange,
  style,
}: {
  kind: 'dataset' | 'source'
  value: string
  disabled: boolean
  onChange: (name: string) => void
  style: CSSProperties
}) {
  const [names, setNames] = useState<{ name: string; detail: string }[] | null>(null)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    let live = true
    const load =
      kind === 'dataset'
        ? api.datasets().then((ds) => ds.map((d) => ({ name: d.name, detail: `${d.format}, ${formatSize(d.size)}` })))
        : api.sources().then((ss) => ss.map((s) => ({ name: s.name, detail: s.description })))
    load
      .then((n) => live && setNames(n))
      .catch((e) => live && setError((e as Error).message))
    return () => {
      live = false
    }
  }, [kind])

  const known = names?.some((n) => n.name === value) ?? true
  return (
    <>
      <select disabled={disabled} value={value} onChange={(e) => onChange(e.target.value)} style={style}>
        <option value="" disabled>
          {names === null ? '(loading...)' : names.length ? `(choose a ${kind})` : `(no ${kind}s available)`}
        </option>
        {!known && value && <option value={value}>{value} (not found)</option>}
        {(names ?? []).map((n) => (
          <option key={n.name} value={n.name}>
            {n.detail ? `${n.name} (${n.detail})` : n.name}
          </option>
        ))}
      </select>
      {error && <div style={{ color: 'var(--mm-error)', fontSize: 11 }}>{error}</div>}
    </>
  )
}

function formatSize(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`
  return `${(bytes / 1024 / 1024).toFixed(1)} MB`
}
