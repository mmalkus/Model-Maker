import { useState } from 'react'
import { api } from './api'
import { ASSIGNABLE_ROLES, ROLE_LABELS } from './roles'
import type { PreviewOut } from './types'

export function DataModal({
  blockName,
  blockId,
  preview,
  onClose,
  onChanged,
}: {
  blockName: string
  // The block whose output this preview shows -- present only when the
  // caller has one to hand (every caller does today), so a column's role
  // can be tagged right here where the data is visible. onChanged, if
  // given, reloads the graph afterward since retagging invalidates this
  // block and everything downstream (see runner.compute_key).
  blockId?: string
  preview: PreviewOut
  onClose: () => void
  onChanged?: () => void
}) {
  // The tag itself is stored immediately, but `preview` is a snapshot of
  // the block's *last run* -- role tagging invalidates the cache (see
  // runner.compute_key) without recomputing it, so the new role won't show
  // up in `preview.columns[].role` until the block is re-run. Without this,
  // the <select> (controlled straight off that stale prop) would visibly
  // snap back to the old value right after picking a new one, looking like
  // the click did nothing. Tracked here rather than bumping preview itself,
  // since there's nothing valid to bump it to before that re-run happens.
  const [pendingRoles, setPendingRoles] = useState<Record<string, string>>({})
  // Same staleness problem as pendingRoles above: tags apply immediately
  // server-side, but `preview` won't reflect them until this block is
  // re-run, so the just-applied tags are tracked here for instant feedback.
  const [pendingTags, setPendingTags] = useState<Record<string, string[]> | null>(null)
  const [analyzing, setAnalyzing] = useState(false)
  const [analysis, setAnalysis] = useState<{ document: string; documentPath: string | null } | null>(null)
  const [showDocument, setShowDocument] = useState(false)

  const analyzeData = () => {
    if (!blockId) return
    setAnalyzing(true)
    api
      .analyzeData(blockId)
      .then((result) => {
        setPendingTags(result.tags)
        setAnalysis({ document: result.document, documentPath: result.document_path })
        setShowDocument(true)
        onChanged?.()
      })
      .catch((e) => alert((e as Error).message))
      .finally(() => setAnalyzing(false))
  }

  const setRole = (column: string, role: string) => {
    if (!blockId) return
    setPendingRoles((prev) => ({ ...prev, [column]: role }))
    api
      .setColumnRole(blockId, column, role)
      .then(onChanged)
      .catch((e) => {
        setPendingRoles((prev) => {
          const next = { ...prev }
          delete next[column]
          return next
        })
        alert((e as Error).message)
      })
  }
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
        <div style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
          {blockId && (
            <button
              disabled={analyzing}
              onClick={analyzeData}
              title="Look at this data's column names and summary statistics, tag columns worth flagging, and write a short description -- see the Data Analysis panel below once it's done."
            >
              {analyzing ? 'Analyzing…' : 'AI analyze data'}
            </button>
          )}
          <button onClick={onClose}>Close</button>
        </div>
      </div>
      {analysis && (
        <div style={{ marginBottom: 8, border: '1px solid #e5e7eb', borderRadius: 6, background: '#f9fafb' }}>
          <button
            onClick={() => setShowDocument((s) => !s)}
            style={{
              width: '100%',
              textAlign: 'left',
              border: 'none',
              background: 'transparent',
              padding: 8,
              fontSize: 12,
              fontWeight: 600,
            }}
          >
            {showDocument ? '▾' : '▸'} Data analysis
            {analysis.documentPath && (
              <span style={{ fontWeight: 400, color: '#6b7280' }}> — saved to {analysis.documentPath}</span>
            )}
          </button>
          {showDocument && (
            <pre
              style={{
                margin: 0,
                padding: '0 8px 8px',
                fontSize: 12,
                whiteSpace: 'pre-wrap',
                fontFamily: 'inherit',
                maxHeight: 240,
                overflow: 'auto',
              }}
            >
              {analysis.document}
            </pre>
          )}
        </div>
      )}
      <div style={{ overflow: 'auto', flex: 1, border: '1px solid #e5e7eb', borderRadius: 6 }}>
        <table style={{ borderCollapse: 'collapse', fontSize: 12, width: '100%' }}>
          <thead style={{ position: 'sticky', top: 0, background: '#f9fafb', zIndex: 1 }}>
            <tr>
              {preview.columns.map((c) => {
                const s = preview.summary?.[c.name]
                const pending = pendingRoles[c.name]
                const displayedRole = pending ?? c.role
                const displayedTags = pendingTags ? (pendingTags[c.name] ?? []) : c.tags
                return (
                  <th key={c.name} style={{ textAlign: 'left', padding: 6, borderBottom: '1px solid #e5e7eb' }}>
                    {c.name}
                    <div style={{ fontWeight: 400, color: '#9ca3af', display: 'flex', alignItems: 'center', gap: 2 }}>
                      {c.dtype} /
                      {blockId && c.role !== 'predicted' ? (
                        <select
                          value={displayedRole}
                          onChange={(e) => setRole(c.name, e.target.value)}
                          title="Tag this column's role"
                          style={{ fontSize: 10, color: '#9ca3af', border: 'none', background: 'transparent', padding: 0 }}
                        >
                          <option value="unassigned">{ROLE_LABELS.unassigned}</option>
                          {ASSIGNABLE_ROLES.map((r) => (
                            <option key={r} value={r}>
                              {ROLE_LABELS[r]}
                            </option>
                          ))}
                        </select>
                      ) : (
                        <span>{c.role}</span>
                      )}
                    </div>
                    {pending !== undefined && pending !== c.role && (
                      <div style={{ fontWeight: 400, color: 'var(--brand-dark, #b45309)' }}>re-run to apply</div>
                    )}
                    {displayedTags.length > 0 && (
                      <div style={{ fontWeight: 400, color: '#7c3aed', fontSize: 10 }}>{displayedTags.join(', ')}</div>
                    )}
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
