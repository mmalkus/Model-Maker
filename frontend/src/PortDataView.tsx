import { useEffect, useState } from 'react'
import { api } from './api'
import { DataModal } from './DataModal'
import { ASSIGNABLE_ROLES, ROLE_LABELS } from './roles'
import type { Artifact, PortType, PreviewOut } from './types'

// The actual look at a port's data (dataframe preview, image, or plain
// value) -- rendered by PortInspector for whichever output port badge the
// user clicked on a block. For a dataframe, this is also where AI analysis
// and column-role tagging live: both are one click away here rather than
// gated behind opening the full table, since tagging a column or kicking
// off an analysis doesn't need to see every row. The AI analysis write-up
// itself is a persistent Artifact (see api.listArtifacts/getArtifact) --
// this is also where you reopen, rename, or delete this port's own one,
// rather than hunting for it in a separate project-wide list.
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

  // Same staleness problem DataModal solves: a role/tag change applies
  // immediately server-side, but `preview` is a snapshot of the block's
  // last run and won't reflect it until the block is re-run (see
  // runner.compute_key) -- tracked here for instant feedback instead of
  // waiting on a re-run or looking like the click did nothing.
  const [pendingRoles, setPendingRoles] = useState<Record<string, string>>({})
  const [pendingTags, setPendingTags] = useState<Record<string, string[]> | null>(null)
  const [analyzing, setAnalyzing] = useState(false)

  // This port's own data-analysis artifact, if one exists -- looked up on
  // open so a previously-generated write-up shows up again without
  // re-running the analysis, not just right after generating it.
  const [artifact, setArtifact] = useState<Artifact | null>(null)
  const [artifactError, setArtifactError] = useState<string | null>(null)
  const [showDocument, setShowDocument] = useState(false)
  const [renaming, setRenaming] = useState(false)
  const [titleInput, setTitleInput] = useState('')

  useEffect(() => {
    setPreview(null)
    setValue(null)
    setError(null)
    setShowTable(false)
    setPendingRoles({})
    setPendingTags(null)
    setArtifact(null)
    setArtifactError(null)
    setShowDocument(false)
    setRenaming(false)
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
    if (portType === 'dataframe') {
      api
        .listArtifacts()
        .then((all) => all.find((a) => a.kind === 'data_analysis' && a.block_id === blockId && a.port === port))
        .then((match) => (match ? api.getArtifact(match.id) : null))
        .then((a) => {
          if (!cancelled && a) setArtifact(a)
        })
        .catch(() => {
          // No existing artifact, or it failed to load -- not fatal, "AI
          // analyze data" below still works to generate a fresh one.
        })
    }
    return () => {
      cancelled = true
    }
  }, [blockId, port, portType])

  const setRole = (column: string, role: string) => {
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

  const analyzeData = () => {
    setAnalyzing(true)
    api
      .analyzeData(blockId, port)
      .then((result) => {
        setPendingTags(result.tags)
        onChanged?.()
        return api.getArtifact(result.artifact_id)
      })
      .then((a) => {
        setArtifact(a)
        setTitleInput(a.title)
        setShowDocument(true)
      })
      .catch((e) => alert((e as Error).message))
      .finally(() => setAnalyzing(false))
  }

  const saveRename = () => {
    if (!artifact) return
    const nextTitle = titleInput.trim()
    if (!nextTitle || nextTitle === artifact.title) {
      setRenaming(false)
      return
    }
    api
      .renameArtifact(artifact.id, nextTitle)
      .then((a) => {
        setArtifact(a)
        setRenaming(false)
      })
      .catch((e) => {
        setArtifactError((e as Error).message)
        setRenaming(false)
      })
  }

  const deleteArtifact = () => {
    if (!artifact) return
    if (!confirm('Delete this data analysis? This does not affect the block or its tags.')) return
    api
      .deleteArtifact(artifact.id)
      .then(() => setArtifact(null))
      .catch((e) => setArtifactError((e as Error).message))
  }

  return (
    <>
      {portType === 'dataframe' &&
        (preview ? (
          <>
            <div style={{ color: '#6b7280', marginBottom: 8 }}>
              {preview.row_count} rows &middot; lineage: {preview.lineage.join(' -> ') || '(none)'}
            </div>
            <div style={{ display: 'flex', gap: 8, marginBottom: 8 }}>
              <button
                disabled={analyzing}
                onClick={analyzeData}
                title="Look at this data's column names and summary statistics, tag columns worth flagging, and write a short description, kept here as this port's data-analysis artifact."
              >
                {analyzing ? 'Analyzing…' : artifact ? 'Re-analyze' : 'AI analyze data'}
              </button>
              <button onClick={() => setShowTable(true)}>View full table</button>
            </div>

            {artifact && (
              <div style={{ marginBottom: 8, border: '1px solid #e5e7eb', borderRadius: 6, background: '#f9fafb' }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: 6, padding: '6px 6px 0' }}>
                  <button
                    onClick={() => setShowDocument((s) => !s)}
                    style={{ flex: 1, textAlign: 'left', border: 'none', background: 'transparent', padding: 0, fontSize: 11, fontWeight: 600 }}
                  >
                    {showDocument ? '▾' : '▸'}{' '}
                    {renaming ? (
                      <input
                        autoFocus
                        value={titleInput}
                        onClick={(e) => e.stopPropagation()}
                        onChange={(e) => setTitleInput(e.target.value)}
                        onBlur={saveRename}
                        onKeyDown={(e) => {
                          if (e.key === 'Enter') saveRename()
                          if (e.key === 'Escape') {
                            setTitleInput(artifact.title)
                            setRenaming(false)
                          }
                        }}
                        style={{ fontSize: 11, fontWeight: 600 }}
                      />
                    ) : (
                      <span
                        title="Click to rename"
                        onClick={(e) => {
                          e.stopPropagation()
                          setTitleInput(artifact.title)
                          setRenaming(true)
                        }}
                      >
                        {artifact.title}
                      </span>
                    )}
                  </button>
                  <button onClick={deleteArtifact} style={{ fontSize: 10, color: '#b91c1c' }}>
                    Delete
                  </button>
                </div>
                {artifact.stale && (
                  <div style={{ padding: '0 6px', fontSize: 10, color: 'var(--brand-dark, #b45309)' }}>
                    source data has changed since this was written
                  </div>
                )}
                {artifactError && <div style={{ padding: '0 6px', fontSize: 10, color: '#b91c1c' }}>{artifactError}</div>}
                {showDocument && (
                  <pre
                    style={{ margin: 0, padding: '4px 6px 6px', fontSize: 11, whiteSpace: 'pre-wrap', fontFamily: 'inherit', maxHeight: 200, overflow: 'auto' }}
                  >
                    {artifact.document}
                  </pre>
                )}
              </div>
            )}

            <div style={{ border: '1px solid #e5e7eb', borderRadius: 6, marginBottom: 8 }}>
              {preview.columns.map((c, i) => {
                const pending = pendingRoles[c.name]
                const displayedRole = pending ?? c.role
                const displayedTags = pendingTags ? (pendingTags[c.name] ?? []) : c.tags
                return (
                  <div
                    key={c.name}
                    style={{
                      display: 'flex',
                      alignItems: 'center',
                      gap: 6,
                      padding: '4px 6px',
                      borderBottom: i < preview.columns.length - 1 ? '1px solid #f3f4f6' : undefined,
                    }}
                  >
                    <div style={{ flex: 1, minWidth: 0, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                      <span title={c.name}>{c.name}</span>
                      <span style={{ color: '#9ca3af' }}> ({c.dtype})</span>
                      {displayedTags.length > 0 && (
                        <div style={{ color: '#7c3aed', fontSize: 10 }}>{displayedTags.join(', ')}</div>
                      )}
                    </div>
                    {c.role !== 'predicted' ? (
                      <select
                        value={displayedRole}
                        onChange={(e) => setRole(c.name, e.target.value)}
                        title="Tag this column's role"
                        style={{ fontSize: 11 }}
                      >
                        <option value="unassigned">{ROLE_LABELS.unassigned}</option>
                        {ASSIGNABLE_ROLES.map((r) => (
                          <option key={r} value={r}>
                            {ROLE_LABELS[r]}
                          </option>
                        ))}
                      </select>
                    ) : (
                      <span style={{ color: '#9ca3af', fontSize: 11 }}>{c.role}</span>
                    )}
                  </div>
                )
              })}
            </div>
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
