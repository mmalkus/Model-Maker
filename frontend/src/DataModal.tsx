import type { PreviewOut } from './types'

export function DataModal({
  blockName,
  preview,
  onClose,
}: {
  blockName: string
  preview: PreviewOut
  onClose: () => void
}) {
  return (
    <div
      // "nodrag nopan" + pointer-events: all matter when this is opened from
      // a wire's mid-line data view (DataWireEdge): it's rendered inside
      // ReactFlow's EdgeLabelRenderer portal, whose container has
      // pointer-events: none by default so it doesn't block the canvas --
      // without opting back in here, clicks would fall through to whatever
      // canvas node sits behind the modal.
      className="nodrag nopan"
      style={{
        position: 'fixed',
        inset: 24,
        background: '#fff',
        border: '1px solid #d1d5db',
        borderRadius: 8,
        padding: 16,
        zIndex: 50,
        pointerEvents: 'all',
        display: 'flex',
        flexDirection: 'column',
        boxShadow: '0 10px 30px rgba(0,0,0,0.25)',
      }}
    >
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'baseline', marginBottom: 8 }}>
        <div>
          <strong style={{ fontSize: 14 }}>{blockName}</strong>
          <span style={{ color: '#6b7280', fontSize: 12, marginLeft: 8 }}>
            {preview.row_count} rows shown &middot; lineage: {preview.lineage.join(' -> ') || '(none)'}
          </span>
        </div>
        <button onClick={onClose}>Close</button>
      </div>
      <div style={{ overflow: 'auto', flex: 1, border: '1px solid #e5e7eb', borderRadius: 6 }}>
        <table style={{ borderCollapse: 'collapse', fontSize: 12, width: '100%' }}>
          <thead style={{ position: 'sticky', top: 0, background: '#f9fafb', zIndex: 1 }}>
            <tr>
              {preview.columns.map((c) => {
                const s = preview.summary?.[c.name]
                return (
                  <th key={c.name} style={{ textAlign: 'left', padding: 6, borderBottom: '1px solid #e5e7eb' }}>
                    {c.name}
                    <div style={{ fontWeight: 400, color: '#9ca3af' }}>
                      {c.dtype} / {c.role}
                    </div>
                    {s && (
                      <div style={{ fontWeight: 400, color: '#9ca3af' }}>
                        {s.null_count > 0 && `${s.null_count} null `}
                        {s.n_unique != null && `${s.n_unique} unique`}
                        {s.mean != null && ` · mean ${s.mean.toFixed(2)}`}
                      </div>
                    )}
                  </th>
                )
              })}
            </tr>
          </thead>
          <tbody>
            {preview.rows.map((row, i) => (
              <tr key={i}>
                {preview.columns.map((c) => (
                  <td key={c.name} style={{ padding: 6, borderBottom: '1px solid #f3f4f6', whiteSpace: 'nowrap' }}>
                    {String(row[c.name])}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  )
}
