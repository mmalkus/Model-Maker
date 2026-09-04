import { useEffect, useState } from 'react'
import { api } from './api'
import { DataModal } from './DataModal'
import type { GraphOut, PortType, PreviewOut, WireOut } from './types'

export function WireInspector({
  wireId,
  graph,
  onClose,
}: {
  wireId: string
  graph: GraphOut
  onClose: () => void
}) {
  const wire: WireOut | undefined = graph.wires[wireId]
  const fromBlock = wire ? graph.blocks[wire.from_block] : undefined
  const portType: PortType | undefined = fromBlock?.outputs.find((p) => p.name === wire?.from_port)?.type

  const [preview, setPreview] = useState<PreviewOut | null>(null)
  const [value, setValue] = useState<{ v: unknown } | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [showTable, setShowTable] = useState(false)

  useEffect(() => {
    setPreview(null)
    setValue(null)
    setError(null)
    setShowTable(false)
    if (!wire || portType === 'image') return
    let cancelled = false
    const load =
      portType === 'dataframe'
        ? api.preview(wire.from_block, { port: wire.from_port, rows: 200, summary: true }).then((p) => {
            if (!cancelled) setPreview(p)
          })
        : api.value(wire.from_block, wire.from_port).then((v) => {
            if (!cancelled) setValue({ v })
          })
    load.catch((e) => {
      if (!cancelled) setError((e as Error).message)
    })
    return () => {
      cancelled = true
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [wireId, portType])

  if (!wire) {
    return (
      <div style={{ width: 340, borderLeft: '1px solid #e5e7eb', padding: 12, color: '#9ca3af', fontSize: 12 }}>
        This wire no longer exists.
      </div>
    )
  }

  const title = wire.name || fromBlock?.name || `${wire.from_block}:${wire.from_port}`

  const rename = () => {
    const name = prompt('Name this data:', wire.name ?? '')
    if (name === null) return
    api.renameWire(wireId, name.trim() || null).catch((err) => alert((err as Error).message))
  }

  return (
    <div style={{ width: 340, borderLeft: '1px solid #e5e7eb', padding: 12, overflowY: 'auto', fontSize: 12 }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'baseline', marginBottom: 4 }}>
        <h3 style={{ fontSize: 14, margin: 0, cursor: 'text' }} title="Click to name this data" onClick={rename}>
          {title}
        </h3>
        <button onClick={onClose}>Close</button>
      </div>
      <div style={{ color: '#6b7280', marginBottom: 12 }}>
        {wire.from_block}:{wire.from_port} &rarr; {wire.to_block}:{wire.to_port}
        {!wire.valid && <div style={{ color: '#ef4444', marginTop: 2 }}>This wire is no longer valid.</div>}
      </div>

      {portType === 'dataframe' &&
        (preview ? (
          <>
            <div style={{ color: '#6b7280', marginBottom: 8 }}>
              {preview.row_count} rows &middot; lineage: {preview.lineage.join(' -> ') || '(none)'}
            </div>
            <button onClick={() => setShowTable(true)}>View full table</button>
          </>
        ) : (
          <div style={{ color: '#9ca3af' }}>{error ?? 'Loading...'}</div>
        ))}

      {portType === 'image' && (
        <img
          src={api.imageUrl(wire.from_block, wire.from_port)}
          alt={title}
          style={{ maxWidth: '100%', border: '1px solid #e5e7eb', borderRadius: 6 }}
        />
      )}

      {portType !== 'dataframe' && portType !== 'image' && (
        <div>
          {error ? (
            <span style={{ color: '#b91c1c' }}>{error}</span>
          ) : (
            <pre style={{ margin: 0, fontSize: 11, whiteSpace: 'pre-wrap' }}>
              {value ? JSON.stringify(value.v, null, 2) : 'Loading...'}
            </pre>
          )}
        </div>
      )}

      {showTable && preview && <DataModal blockName={title} preview={preview} onClose={() => setShowTable(false)} />}
    </div>
  )
}
