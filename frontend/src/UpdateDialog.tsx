import { useEffect, useState } from 'react'
import { api } from './api'
import type { BlockOut, UpdatePreview } from './types'

/** Moving a placed block to its library's newer version (or accepting a
 * changed built-in): the code diff plus the structural impact -- params
 * carried, renamed, dropped or newly required, and wires that stay, get
 * renamed or would be disconnected -- confirmed before anything changes. */
export function UpdateDialog({ block, onClose, onApplied }: { block: BlockOut; onClose: () => void; onApplied: () => void }) {
  const [preview, setPreview] = useState<UpdatePreview | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [busy, setBusy] = useState(false)
  const builtin = block.definition_info?.source === 'builtin'

  useEffect(() => {
    api
      .updatePreview(block.id)
      .then(setPreview)
      .catch((e) => setError((e as Error).message))
  }, [block.id])

  const apply = () => {
    setBusy(true)
    api
      .applyUpdate(block.id)
      .then(() => {
        onApplied()
        onClose()
      })
      .catch((e) => setError((e as Error).message))
      .finally(() => setBusy(false))
  }

  const item = (ok: boolean | 'warn', text: string) => (
    <li style={{ color: ok === true ? '#3f6259' : ok === 'warn' ? '#92400e' : '#b91c1c' }}>{text}</li>
  )

  return (
    <div style={{ position: 'fixed', inset: 0, background: 'rgba(11,35,64,0.35)', zIndex: 55, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
      <div role="dialog" aria-label="Update block" style={{ width: 760, maxHeight: '85vh', background: '#fff', borderRadius: 12, display: 'flex', flexDirection: 'column', overflow: 'hidden' }}>
        <div style={{ padding: '14px 18px 8px' }}>
          <h2 style={{ margin: 0, fontFamily: 'var(--font-heading)', fontSize: 19, color: 'var(--brand-ink)' }}>
            {builtin ? `Accept the changed built-in: ${block.category}` : `Update ${block.name}`}
          </h2>
          {preview && (
            <div style={{ color: '#6b7280', fontSize: 12 }}>
              v{preview.from_version} → v{preview.to_version}
              {builtin && ' · the installed Model-Maker changed this block since the model was saved'}
            </div>
          )}
        </div>
        {error && <div style={{ background: '#fef2f2', color: '#b91c1c', padding: '6px 18px', fontSize: 12 }}>{error}</div>}
        {preview && (
          <div style={{ padding: '0 18px', overflowY: 'auto', fontSize: 12 }}>
            <ul style={{ margin: '8px 0', paddingLeft: 18 }}>
              {preview.params_carried.length > 0 && item(true, `Params carried over: ${preview.params_carried.join(', ')}`)}
              {Object.entries(preview.params_renamed).map(([a, b]) => item(true, `Param ${a} is now ${b} -- its value moves with it`))}
              {preview.params_dropped.length > 0 && item('warn', `Params dropped (no longer exist): ${preview.params_dropped.join(', ')}`)}
              {preview.params_newly_required.length > 0 && item(false, `New required params to fill in: ${preview.params_newly_required.join(', ')}`)}
              {preview.wires_kept.length > 0 && item(true, `${preview.wires_kept.length} wire(s) stay connected`)}
              {Object.keys(preview.wires_renamed).length > 0 && item(true, `${Object.keys(preview.wires_renamed).length} wire(s) follow a renamed port`)}
              {preview.wires_disconnected.map((w) => item(false, `A wire on ${w.side} port ${w.port} will be disconnected: ${w.reason}`))}
              {preview.meta_changed && !preview.code_diff && item('warn', 'Its description (ports, form or metadata rules) changed')}
            </ul>
            {preview.code_diff ? (
              <pre style={{ fontFamily: 'ui-monospace, monospace', fontSize: 11, background: '#f9fafb', border: '1px solid #e5e7eb', borderRadius: 6, padding: 8, margin: '0 0 12px', overflowX: 'auto' }}>
                {preview.code_diff.split('\n').map((l, i) => (
                  <div key={i} style={{ color: l.startsWith('+') ? '#166534' : l.startsWith('-') ? '#b91c1c' : undefined }}>
                    {l}
                  </div>
                ))}
              </pre>
            ) : (
              <div style={{ color: '#6b7280', marginBottom: 12 }}>The code is unchanged.</div>
            )}
          </div>
        )}
        <div style={{ display: 'flex', gap: 8, justifyContent: 'flex-end', padding: '12px 18px', borderTop: '1px solid #e5e7eb', background: '#fcfbf8' }}>
          <button onClick={onClose}>Cancel</button>
          <button className="brand-primary" disabled={!preview || busy} onClick={apply}>
            {builtin ? 'Accept' : `Update to v${preview?.to_version ?? ''}`}
          </button>
        </div>
      </div>
    </div>
  )
}
