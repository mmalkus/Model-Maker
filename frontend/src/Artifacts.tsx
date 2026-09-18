import { useEffect, useState } from 'react'
import { api } from './api'
import type { Artifact, ArtifactSummary } from './types'

// Dropdown panel listing every generated Artifact (today: AI data-analysis
// write-ups -- see api.analyzeData) across the whole project, newest first
// -- the project-wide counterpart to the per-port "Data analysis" section
// in PortDataView, so a write-up stays findable after you've navigated
// away from the block that produced it.
export function Artifacts({ onClose }: { onClose: () => void }) {
  const [summaries, setSummaries] = useState<ArtifactSummary[] | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [open, setOpen] = useState<Artifact | null>(null)
  const [openError, setOpenError] = useState<string | null>(null)
  const [renaming, setRenaming] = useState(false)
  const [titleInput, setTitleInput] = useState('')

  const reload = () => api.listArtifacts().then(setSummaries).catch((e) => setError((e as Error).message))

  useEffect(() => {
    reload()
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  const openArtifact = (id: string) => {
    setOpenError(null)
    api
      .getArtifact(id)
      .then((a) => {
        setOpen(a)
        setTitleInput(a.title)
        setRenaming(false)
      })
      .catch((e) => setOpenError((e as Error).message))
  }

  const saveRename = () => {
    if (!open) return
    const title = titleInput.trim()
    if (!title || title === open.title) {
      setRenaming(false)
      return
    }
    api
      .renameArtifact(open.id, title)
      .then((a) => {
        setOpen(a)
        setRenaming(false)
        reload()
      })
      .catch((e) => alert((e as Error).message))
  }

  const doDelete = (id: string) => {
    if (!confirm('Delete this artifact? This does not affect the block that produced it.')) return
    api
      .deleteArtifact(id)
      .then(() => {
        if (open?.id === id) setOpen(null)
        reload()
      })
      .catch((e) => alert((e as Error).message))
  }

  return (
    <div
      style={{
        position: 'absolute',
        top: '100%',
        right: 0,
        marginTop: 4,
        background: '#fff',
        border: '1px solid #d1d5db',
        borderRadius: 8,
        padding: 12,
        zIndex: 50,
        width: open ? 420 : 320,
        maxHeight: 480,
        overflowY: 'auto',
        boxShadow: '0 10px 30px rgba(0,0,0,0.25)',
        fontSize: 12,
      }}
    >
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'baseline', marginBottom: 8 }}>
        <strong style={{ fontSize: 13 }}>Artifacts</strong>
        <button onClick={onClose}>Close</button>
      </div>

      {error && <div style={{ color: '#b91c1c', marginBottom: 8 }}>{error}</div>}

      {open ? (
        <div>
          <button onClick={() => setOpen(null)} style={{ marginBottom: 8 }}>
            ← Back to list
          </button>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'baseline', marginBottom: 4, gap: 6 }}>
            {renaming ? (
              <input
                autoFocus
                value={titleInput}
                onChange={(e) => setTitleInput(e.target.value)}
                onBlur={saveRename}
                onKeyDown={(e) => {
                  if (e.key === 'Enter') saveRename()
                  if (e.key === 'Escape') {
                    setTitleInput(open.title)
                    setRenaming(false)
                  }
                }}
                style={{ flex: 1, fontSize: 13, fontWeight: 600, boxSizing: 'border-box' }}
              />
            ) : (
              <strong style={{ fontSize: 13, cursor: 'text' }} title="Click to rename" onClick={() => setRenaming(true)}>
                {open.title}
              </strong>
            )}
            <button onClick={() => doDelete(open.id)} style={{ color: '#b91c1c' }}>
              Delete
            </button>
          </div>
          <div style={{ color: '#6b7280', marginBottom: 6 }}>
            {open.block_name ?? '(deleted block)'} :: {open.port}
            {open.stale && (
              <span style={{ color: 'var(--brand-dark, #b45309)' }}> · source data has changed since this was written</span>
            )}
          </div>
          {openError && <div style={{ color: '#b91c1c', marginBottom: 6 }}>{openError}</div>}
          <pre
            style={{
              margin: 0,
              padding: 8,
              fontSize: 12,
              whiteSpace: 'pre-wrap',
              fontFamily: 'inherit',
              background: '#f9fafb',
              border: '1px solid #e5e7eb',
              borderRadius: 6,
              maxHeight: 320,
              overflow: 'auto',
            }}
          >
            {open.document}
          </pre>
        </div>
      ) : summaries === null ? (
        <div style={{ color: '#9ca3af' }}>Loading...</div>
      ) : summaries.length === 0 ? (
        <div style={{ color: '#9ca3af' }}>
          No artifacts yet -- click "AI analyze data" on a data port to generate one.
        </div>
      ) : (
        <div>
          {summaries.map((a) => (
            <div
              key={a.id}
              onClick={() => openArtifact(a.id)}
              style={{ padding: '6px 4px', cursor: 'pointer', borderBottom: '1px solid #f3f4f6' }}
              onMouseEnter={(ev) => (ev.currentTarget.style.background = '#f9fafb')}
              onMouseLeave={(ev) => (ev.currentTarget.style.background = 'transparent')}
            >
              <div style={{ fontWeight: 600 }}>{a.title}</div>
              <div style={{ color: '#6b7280' }}>
                {a.block_name ?? '(deleted block)'} :: {a.port} &middot; {new Date(a.updated_at).toLocaleString()}
                {a.stale && <span style={{ color: 'var(--brand-dark, #b45309)' }}> · stale</span>}
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}
