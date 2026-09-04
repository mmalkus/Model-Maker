import { useEffect, useState } from 'react'
import { api } from './api'
import { DataModal } from './DataModal'
import type { PortType, PreviewOut } from './types'

// The actual look at a port's data (dataframe preview, image, or plain
// value) -- rendered by PortInspector for whichever output port badge the
// user clicked on a block.
export function PortDataView({
  blockId,
  port,
  portType,
  title,
  onChanged,
}: {
  blockId: string
  port: string
  portType: PortType | undefined
  title: string
  onChanged?: () => void
}) {
  const [preview, setPreview] = useState<PreviewOut | null>(null)
  const [value, setValue] = useState<{ v: unknown } | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [showTable, setShowTable] = useState(false)

  useEffect(() => {
    setPreview(null)
    setValue(null)
    setError(null)
    setShowTable(false)
    if (portType === 'image') return
    let cancelled = false
    const load =
      portType === 'dataframe'
        ? api.preview(blockId, { port, rows: 200, summary: true }).then((p) => {
            if (!cancelled) setPreview(p)
          })
        : api.value(blockId, port).then((v) => {
            if (!cancelled) setValue({ v })
          })
    load.catch((e) => {
      if (!cancelled) setError((e as Error).message)
    })
    return () => {
      cancelled = true
    }
  }, [blockId, port, portType])

  return (
    <>
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
          src={api.imageUrl(blockId, port)}
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

      {showTable && preview && (
        <DataModal
          blockName={title}
          blockId={blockId}
          preview={preview}
          onClose={() => setShowTable(false)}
          onChanged={onChanged}
        />
      )}
    </>
  )
}
