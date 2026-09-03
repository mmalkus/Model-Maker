import { useEffect, useRef, useState } from 'react'
import { api } from './api'
import { DataModal } from './DataModal'
import { FileBrowser } from './FileBrowser'
import { PARAM_SPECS, ParamsForm } from './ParamsForm'
import type { BlockOut, DraftOut, PreviewOut } from './types'

export function Inspector({ block, onChanged }: { block: BlockOut | null; onChanged: () => void }) {
  const [paramsText, setParamsText] = useState('')
  const [paramsError, setParamsError] = useState<string | null>(null)
  const [codeText, setCodeText] = useState('')
  const [metaText, setMetaText] = useState('')
  const [metaError, setMetaError] = useState<string | null>(null)
  const [preview, setPreview] = useState<PreviewOut | null>(null)
  const [showData, setShowData] = useState(false)
  const [busy, setBusy] = useState(false)
  const [instruction, setInstruction] = useState('')
  const [draftExplanation, setDraftExplanation] = useState<string | null>(null)
  const [nameText, setNameText] = useState('')
  const [editingName, setEditingName] = useState(false)
  const [showBrowser, setShowBrowser] = useState(false)
  const [inputColumns, setInputColumns] = useState<string[]>([])
  const [drafting, setDrafting] = useState(false)
  const [draftElapsed, setDraftElapsed] = useState(0)
  const [llmProvider, setLlmProvider] = useState<string | null>(null)
  const [metricValues, setMetricValues] = useState<Record<string, unknown>>({})
  const [dynamicDataframePorts, setDynamicDataframePorts] = useState<Set<string>>(new Set())
  const [dynamicImagePorts, setDynamicImagePorts] = useState<Set<string>>(new Set())
  const instructionRef = useRef<HTMLTextAreaElement | null>(null)

  useEffect(() => {
    api
      .llmProviders()
      .then((p) => setLlmProvider(p.active))
      .catch(() => setLlmProvider(null))
  }, [])

  useEffect(() => {
    if (!drafting) return
    setDraftElapsed(0)
    const id = setInterval(() => setDraftElapsed((s) => s + 1), 1000)
    return () => clearInterval(id)
  }, [drafting])

  useEffect(() => {
    if (!block) {
      setInputColumns([])
      return
    }
    api
      .inputSchema(block.id)
      .then((schema) => {
        const cols = new Set<string>()
        Object.values(schema).forEach((list) => list.forEach((c) => cols.add(c.name)))
        setInputColumns([...cols])
      })
      .catch(() => setInputColumns([]))
    // Refetch on every graph reload (App hands down a fresh `block` object
    // each time), not just when this block's own id/status changes -- an
    // upstream block turning green is what usually makes columns available,
    // and that only shows up as this block's *reference* changing.
  }, [block])

  useEffect(() => {
    if (!block || !(block.status === 'green' || block.status === 'orange')) {
      setMetricValues({})
      setDynamicDataframePorts(new Set())
      setDynamicImagePorts(new Set())
      return
    }
    // scalar_metric/model ports have a known non-viewable type; "any" ports
    // (the generic View value block) could turn out to hold any of the
    // four kinds, so probe them the same way and sort by what /value says.
    const candidates = block.outputs.filter((p) => p.type === 'scalar_metric' || p.type === 'model' || p.type === 'any')
    if (candidates.length === 0) {
      setMetricValues({})
      setDynamicDataframePorts(new Set())
      setDynamicImagePorts(new Set())
      return
    }
    let cancelled = false
    Promise.all(
      candidates.map(async (p) => {
        try {
          return { name: p.name, kind: 'value' as const, value: await api.value(block.id, p.name) }
        } catch (e) {
          const msg = (e as Error).message || ''
          if (msg.includes('is a dataframe')) return { name: p.name, kind: 'dataframe' as const }
          if (msg.includes('binary')) return { name: p.name, kind: 'image' as const }
          return { name: p.name, kind: 'error' as const }
        }
      }),
    ).then((results) => {
      if (cancelled) return
      setMetricValues(
        Object.fromEntries(results.filter((r): r is { name: string; kind: 'value'; value: unknown } => r.kind === 'value').map((r) => [r.name, r.value])),
      )
      setDynamicDataframePorts(new Set(results.filter((r) => r.kind === 'dataframe').map((r) => r.name)))
      setDynamicImagePorts(new Set(results.filter((r) => r.kind === 'image').map((r) => r.name)))
    })
    return () => {
      cancelled = true
    }
  }, [block])

  useEffect(() => {
    setParamsText(block ? JSON.stringify(block.params, null, 2) : '')
    setParamsError(null)
    setCodeText(block?.code ?? '')
    setMetaText(block ? JSON.stringify(block.metadata_transform ?? { kind: 'passthrough' }, null, 2) : '')
    setMetaError(null)
    setPreview(null)
    setShowData(false)
    setDraftExplanation(null)
    setInstruction('')
    setNameText(block?.name ?? '')
    setEditingName(false)
    setShowBrowser(false)
    setDrafting(false)

    // A block that's never been touched (still v1, still the default
    // scaffold body) is almost certainly one the user just created to hand
    // straight to AI -- put the cursor where they'll type next.
    if (block?.is_custom && block.code_version === 1) {
      requestAnimationFrame(() => instructionRef.current?.focus())
    }
  }, [block?.id])

  if (!block) {
    return (
      <div style={{ width: 340, borderLeft: '1px solid #e5e7eb', padding: 12, color: '#9ca3af', fontSize: 12 }}>
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

  const saveName = () => {
    setEditingName(false)
    if (nameText.trim() && nameText !== block.name) run(() => api.updateBlock(block.id, { name: nameText.trim() }))
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

  const dataframePort = block.outputs.find((p) => p.type === 'dataframe' || dynamicDataframePorts.has(p.name))?.name

  const viewData = () =>
    run(async () => {
      setPreview(await api.preview(block.id, { port: dataframePort, rows: 200, summary: true }))
      setShowData(true)
    })

  const applyDraft = (draft: DraftOut) => {
    if (block.is_custom) {
      setCodeText(draft.code)
      setMetaText(JSON.stringify(draft.metadata_transform, null, 2))
      setMetaError(null)
    }
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

  const pickPath = (path: string) => {
    setShowBrowser(false)
    let parsed: Record<string, unknown>
    try {
      parsed = JSON.parse(paramsText || '{}')
    } catch {
      parsed = { ...block.params }
    }
    const merged = { ...parsed, path }
    setParamsText(JSON.stringify(merged, null, 2))
    setParamsError(null)
    run(() => api.updateBlock(block.id, { params: merged }))
  }

  const draftWithAI = () =>
    run(async () => {
      setDrafting(true)
      try {
        applyDraft(await api.draftBlock(block.id, instruction))
      } finally {
        setDrafting(false)
      }
    })

  const suggestFix = () =>
    run(async () => {
      setDrafting(true)
      try {
        applyDraft(await api.suggestFix(block.id))
      } finally {
        setDrafting(false)
      }
    })

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
      {editingName ? (
        <input
          autoFocus
          value={nameText}
          onChange={(e) => setNameText(e.target.value)}
          onBlur={saveName}
          onKeyDown={(e) => {
            if (e.key === 'Enter') saveName()
            if (e.key === 'Escape') {
              setNameText(block.name)
              setEditingName(false)
            }
          }}
          style={{ fontSize: 14, fontWeight: 600, width: '100%', boxSizing: 'border-box', marginBottom: 4 }}
        />
      ) : (
        <h3
          style={{ fontSize: 14, margin: '0 0 4px', cursor: 'text' }}
          title="Click to rename"
          onClick={() => setEditingName(true)}
        >
          {block.name}
        </h3>
      )}
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
        {(block.status === 'green' || block.status === 'orange') && dataframePort && (
          <button disabled={busy} onClick={viewData}>
            View data
          </button>
        )}
        <button
          disabled={busy}
          onClick={() => {
            if (confirm(`Delete block "${block.name}"? Its wires go with it. This can't be undone.`)) {
              run(() => api.deleteBlock(block.id))
            }
          }}
          style={{ marginLeft: 'auto', color: '#b91c1c', borderColor: '#fecaca' }}
        >
          Delete block
        </button>
      </div>

      {(block.status === 'green' || block.status === 'orange') &&
        block.outputs
          .filter((p) => p.type === 'image' || dynamicImagePorts.has(p.name))
          .map((p) => (
            <div key={p.name} style={{ marginBottom: 12 }}>
              <img
                src={api.imageUrl(block.id, p.name, block.last_attempt_at)}
                alt={`${block.name} (${p.name})`}
                style={{ maxWidth: '100%', border: '1px solid #e5e7eb', borderRadius: 6 }}
              />
            </div>
          ))}

      {Object.entries(metricValues).map(([name, value]) => (
        <MetricView key={name} name={name} value={value} />
      ))}

      {block.category === 'read_csv' && (
        <div style={{ marginBottom: 12 }}>
          <div style={{ fontWeight: 600, marginBottom: 4 }}>CSV file</div>
          <div style={{ display: 'flex', gap: 6 }}>
            <input
              readOnly
              value={(block.params.path as string) ?? ''}
              placeholder="(no file selected)"
              style={{ flex: 1, fontFamily: 'monospace', fontSize: 11, boxSizing: 'border-box' }}
            />
            <button disabled={busy} onClick={() => setShowBrowser(true)}>
              Browse...
            </button>
          </div>
        </div>
      )}

      {block.category !== 'display_table' && (
        <div style={{ marginBottom: 12 }}>
          <div style={{ fontWeight: 600, marginBottom: 4 }}>Params</div>
          {PARAM_SPECS[block.category] ? (
            <ParamsForm
              spec={PARAM_SPECS[block.category]}
              paramsText={paramsText}
              setParamsText={setParamsText}
              columns={inputColumns}
              disabled={busy}
            />
          ) : (
            <textarea
              value={paramsText}
              onChange={(e) => setParamsText(e.target.value)}
              rows={8}
              style={{ width: '100%', fontFamily: 'monospace', fontSize: 11, boxSizing: 'border-box' }}
            />
          )}
          {paramsError && <div style={{ color: '#b91c1c' }}>{paramsError}</div>}
          <button disabled={busy} onClick={saveParams} style={{ marginTop: 4 }}>
            Save params
          </button>
        </div>
      )}

      <div style={{ marginBottom: 12, background: '#faf5ff', border: '1px solid #e9d5ff', borderRadius: 6, padding: 8 }}>
        <div style={{ fontWeight: 600, marginBottom: 4 }}>
          Draft with AI
          {llmProvider && <span style={{ fontWeight: 400, color: '#9ca3af' }}> (via {llmProvider})</span>}
        </div>
        <textarea
          ref={instructionRef}
          value={instruction}
          onChange={(e) => setInstruction(e.target.value)}
          rows={3}
          placeholder={
            block.is_custom
              ? 'Describe what this block should do...'
              : "Describe what to set, e.g. \"only keep rows where credit_score > 700\"..."
          }
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
        {drafting && (
          <div style={{ marginTop: 6, color: '#6b21a8' }}>
            {draftElapsed < 5
              ? `Asking ${llmProvider ?? 'the LLM provider'}...`
              : llmProvider === 'claude_cli'
                ? `Still waiting on the local claude CLI (${draftElapsed}s)... it can take a while, or hang if the configured model isn't available on your plan.`
                : `Still waiting (${draftElapsed}s)...`}
          </div>
        )}
        {!block.is_custom && (
          <div style={{ color: '#9ca3af', marginTop: 4 }}>
            This block's code is fixed -- AI can only choose values for its params, shown above.
          </div>
        )}
        {draftExplanation && (
          <div style={{ marginTop: 6, color: '#6b21a8' }}>
            {draftExplanation}
            <div style={{ color: '#9ca3af', marginTop: 2 }}>
              {block.is_custom
                ? 'Draft filled into Code/Params/Metadata transform below -- review, then Save to apply.'
                : 'Draft filled into Params above -- review, then Save params to apply.'}
            </div>
          </div>
        )}
      </div>

      {block.is_custom ? (
        <>
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
      ) : (
        block.source && (
          <div style={{ marginBottom: 12 }}>
            <div style={{ fontWeight: 600, marginBottom: 4 }}>Code</div>
            <pre
              style={{
                background: '#f9fafb',
                border: '1px solid #e5e7eb',
                borderRadius: 6,
                padding: 8,
                fontSize: 11,
                overflow: 'auto',
                maxHeight: 320,
                margin: 0,
              }}
            >
              {block.source}
            </pre>
            <div style={{ color: '#9ca3af', marginTop: 2 }}>
              Fixed code from the block library -- not editable. Configure this block via its params above.
            </div>
          </div>
        )
      )}

      {showData && preview && (
        <DataModal blockName={block.name} preview={preview} onClose={() => setShowData(false)} />
      )}

      {showBrowser && <FileBrowser ext=".csv" onPick={pickPath} onClose={() => setShowBrowser(false)} />}
    </div>
  )
}

function formatMetricScalar(value: unknown): string {
  if (typeof value === 'number') return Number.isInteger(value) ? String(value) : value.toFixed(4)
  if (value === null || value === undefined) return String(value)
  if (typeof value === 'object') return JSON.stringify(value)
  return String(value)
}

// Renders a scalar_metric/model artifact (or whatever a "View value" block's
// "any" port turned out to hold, once it's not a dataframe or an image) as a
// small key/value table -- these are always plain flat-ish dicts (KS/AUC/
// PSI results, GLM coefficients, ...), never something that needs a chart.
function MetricView({ name, value }: { name: string; value: unknown }) {
  const entries = value && typeof value === 'object' && !Array.isArray(value) ? Object.entries(value as Record<string, unknown>) : null
  return (
    <div style={{ marginBottom: 12 }}>
      <div style={{ fontWeight: 600, marginBottom: 4 }}>{name}</div>
      {entries ? (
        <table style={{ fontSize: 11, borderCollapse: 'collapse', width: '100%' }}>
          <tbody>
            {entries.map(([k, v]) => (
              <tr key={k}>
                <td style={{ padding: '2px 8px 2px 0', color: '#6b7280', verticalAlign: 'top', whiteSpace: 'nowrap' }}>{k}</td>
                <td style={{ padding: '2px 0', wordBreak: 'break-word' }}>{formatMetricScalar(v)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      ) : (
        <pre style={{ background: '#f9fafb', border: '1px solid #e5e7eb', borderRadius: 6, padding: 8, fontSize: 11, margin: 0 }}>
          {JSON.stringify(value, null, 2)}
        </pre>
      )}
    </div>
  )
}
