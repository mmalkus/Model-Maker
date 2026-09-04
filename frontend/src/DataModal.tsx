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
        <button onClick={onClose}>Close</button>
      </div>
      <div style={{ overflow: 'auto', flex: 1, border: '1px solid #e5e7eb', borderRadius: 6 }}>
        <table style={{ borderCollapse: 'collapse', fontSize: 12, width: '100%' }}>
          <thead style={{ position: 'sticky', top: 0, background: '#f9fafb', zIndex: 1 }}>
            <tr>
              {preview.columns.map((c) => {
                const s = preview.summary?.[c.name]
                const pending = pendingRoles[c.name]
                const displayedRole = pending ?? c.role
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
