import { useEffect, useState } from 'react'
import { api } from './api'
import type { BlockOut, DraftOut, PreviewOut } from './types'

export function Inspector({ block, onChanged }: { block: BlockOut | null; onChanged: () => void }) {
  const [paramsText, setParamsText] = useState('')
  const [paramsError, setParamsError] = useState<string | null>(null)
  const [codeText, setCodeText] = useState('')
  const [metaText, setMetaText] = useState('')
  const [metaError, setMetaError] = useState<string | null>(null)
  const [preview, setPreview] = useState<PreviewOut | null>(null)
  const [busy, setBusy] = useState(false)
  const [instruction, setInstruction] = useState('')
  const [draftExplanation, setDraftExplanation] = useState<string | null>(null)

  useEffect(() => {
    setParamsText(block ? JSON.stringify(block.params, null, 2) : '')
    setParamsError(null)
    setCodeText(block?.code ?? '')
    setMetaText(block ? JSON.stringify(block.metadata_transform ?? { kind: 'passthrough' }, null, 2) : '')
    setMetaError(null)
    setPreview(null)
    setDraftExplanation(null)
  }, [block?.id])

  if (!block) {
    return (
      <div style={{ width: 320, borderLeft: '1px solid #e5e7eb', padding: 12, color: '#9ca3af', fontSize: 12 }}>
        Select a block to inspect it.
      </div>
    )
  }

  const run = async (fn: () => Promise<unknown>) => {
    setBusy(true)
    try {
      await fn()
      onChanged()
    } catch (e) {
      alert((e as Error).message)
    } finally {
      setBusy(false)
    }
  }

  const saveParams = () => {
    try {
      const parsed = JSON.parse(paramsText || '{}')
      setParamsError(null)
      run(() => api.updateBlock(block.id, { params: parsed }))
    } catch {
      setParamsError('invalid JSON')
    }
  }

  const loadPreview = () => run(async () => setPreview(await api.preview(block.id, { rows: 20, summary: true })))

  const applyDraft = (draft: DraftOut) => {
    setCodeText(draft.code)
    setMetaText(JSON.stringify(draft.metadata_transform, null, 2))
    setMetaError(null)
    if (draft.params && Object.keys(draft.params).length > 0) {
      let merged: Record<string, unknown> = draft.params
      try {
        merged = { ...JSON.parse(paramsText || '{}'), ...draft.params }
      } catch {
        // current params text isn't valid JSON -- fall back to the draft's params alone
      }
      setParamsText(JSON.stringify(merged, null, 2))
      setParamsError(null)
    }
    setDraftExplanation(draft.explanation)
  }

  const draftWithAI = () => run(async () => applyDraft(await api.draftBlock(block.id, instruction)))

  const suggestFix = () => run(async () => applyDraft(await api.suggestFix(block.id)))

  const saveCode = () => run(() => api.updateBlock(block.id, { code: codeText }))

  const saveMeta = () => {
    try {
      const parsed = JSON.parse(metaText || '{}')
      setMetaError(null)
      run(() => api.updateBlock(block.id, { metadata_transform: parsed }))
    } catch {
      setMetaError('invalid JSON')
    }
  }

  return (
    <div style={{ width: 340, borderLeft: '1px solid #e5e7eb', padding: 12, overflowY: 'auto', fontSize: 12 }}>
      <h3 style={{ fontSize: 14, margin: '0 0 4px' }}>{block.name}</h3>
      <div style={{ color: '#6b7280', marginBottom: 8 }}>
        {block.category} &middot; {block.block_type} &middot; status: <strong>{block.status}</strong>
      </div>

      {block.last_error && (
        <div style={{ background: '#fef2f2', color: '#b91c1c', padding: 8, borderRadius: 6, marginBottom: 8 }}>
          {block.last_error}
        </div>
      )}

      <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap', marginBottom: 12 }}>
        <button disabled={busy} onClick={() => run(() => api.run(block.id))}>
          Run
        </button>
        <button disabled={busy} onClick={() => run(() => api.runToHere(block.id))}>
          Run to here
        </button>
        {block.block_type === 'input' && (
          <>
            <button disabled={busy} onClick={() => run(() => api.refresh(block.id))}>
              Refresh
            </button>
            <button
              disabled={busy}
              onClick={() =>
                run(async () => {
                  const { changed } = await api.checkChanges(block.id)
                  alert(changed ? 'Source has changed since last read.' : 'No change detected.')
                })
              }
            >
              Check for changes
            </button>
          </>
        )}
      </div>

      <div style={{ marginBottom: 12 }}>
        <div style={{ fontWeight: 600, marginBottom: 4 }}>Params</div>
        <textarea
          value={paramsText}
          onChange={(e) => setParamsText(e.target.value)}
          rows={8}
          style={{ width: '100%', fontFamily: 'monospace', fontSize: 11, boxSizing: 'border-box' }}
        />
        {paramsError && <div style={{ color: '#b91c1c' }}>{paramsError}</div>}
        <button disabled={busy} onClick={saveParams} style={{ marginTop: 4 }}>
          Save params
        </button>
      </div>

      {block.block_type === 'llm_authored' && (
        <>
          <div style={{ marginBottom: 12, background: '#faf5ff', border: '1px solid #e9d5ff', borderRadius: 6, padding: 8 }}>
            <div style={{ fontWeight: 600, marginBottom: 4 }}>Draft with AI</div>
            <textarea
              value={instruction}
              onChange={(e) => setInstruction(e.target.value)}
              rows={3}
              placeholder="Describe what this block should do..."
              style={{ width: '100%', fontSize: 11, boxSizing: 'border-box' }}
            />
            <div style={{ display: 'flex', gap: 6, marginTop: 4 }}>
              <button disabled={busy || !instruction.trim()} onClick={draftWithAI}>
                Draft
              </button>
              {block.status === 'red' && (
                <button disabled={busy} onClick={suggestFix}>
                  Suggest fix
                </button>
              )}
            </div>
            {draftExplanation && (
              <div style={{ marginTop: 6, color: '#6b21a8' }}>
                {draftExplanation}
                <div style={{ color: '#9ca3af', marginTop: 2 }}>
                  Draft filled into Code/Params/Metadata transform below -- review, then Save to apply.
                </div>
              </div>
            )}
          </div>

          <div style={{ marginBottom: 12 }}>
            <div style={{ fontWeight: 600, marginBottom: 4 }}>Code (v{block.code_version})</div>
            <textarea
              value={codeText}
              onChange={(e) => setCodeText(e.target.value)}
              rows={8}
              spellCheck={false}
              style={{ width: '100%', fontFamily: 'monospace', fontSize: 11, boxSizing: 'border-box' }}
            />
            <button disabled={busy} onClick={saveCode} style={{ marginTop: 4 }}>
              Save code
            </button>
            <div style={{ color: '#9ca3af', marginTop: 2 }}>
              Saving edits bumps the code version and cascades this block and its downstream to grey/orange.
            </div>
          </div>

          <div style={{ marginBottom: 12 }}>
            <div style={{ fontWeight: 600, marginBottom: 4 }}>Metadata transform</div>
            <textarea
              value={metaText}
              onChange={(e) => setMetaText(e.target.value)}
              rows={4}
              style={{ width: '100%', fontFamily: 'monospace', fontSize: 11, boxSizing: 'border-box' }}
            />
            {metaError && <div style={{ color: '#b91c1c' }}>{metaError}</div>}
            <button disabled={busy} onClick={saveMeta} style={{ marginTop: 4 }}>
              Save metadata transform
            </button>
          </div>
        </>
      )}

      {(block.status === 'green' || block.status === 'orange') && (
        <div>
          <button disabled={busy} onClick={loadPreview}>
            Load preview
          </button>
          {preview && (
            <div style={{ marginTop: 8 }}>
              <div style={{ color: '#6b7280', marginBottom: 4 }}>
                {preview.row_count} rows &middot; lineage: {preview.lineage.join(' -> ') || '(none)'}
              </div>
              <div style={{ overflowX: 'auto', border: '1px solid #e5e7eb', borderRadius: 6 }}>
                <table style={{ borderCollapse: 'collapse', fontSize: 11, width: '100%' }}>
                  <thead>
                    <tr>
                      {preview.columns.map((c) => (
                        <th key={c.name} style={{ textAlign: 'left', padding: 4, borderBottom: '1px solid #e5e7eb' }}>
                          {c.name}
                          <div style={{ fontWeight: 400, color: '#9ca3af' }}>
                            {c.dtype} / {c.role}
                          </div>
                        </th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {preview.rows.map((row, i) => (
                      <tr key={i}>
                        {preview.columns.map((c) => (
                          <td key={c.name} style={{ padding: 4, borderBottom: '1px solid #f3f4f6' }}>
                            {String(row[c.name])}
                          </td>
                        ))}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  )
}
