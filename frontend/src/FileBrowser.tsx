import { useEffect, useState } from 'react'
import { api } from './api'
import type { BrowseOut } from './types'

export function FileBrowser({
  startPath,
  ext,
  mode = 'open',
  defaultName,
  onPick,
  onClose,
}: {
  startPath?: string | null
  ext?: string
  // 'open' picks an existing file immediately on click (e.g. choosing a CSV
  // to read). 'save' instead fills a filename field the user can edit --
  // clicking an existing file just reuses its name (to overwrite it) --
  // and a Save button combines it with the current directory into the path
  // handed to onPick, since the target file need not exist yet.
  mode?: 'open' | 'save'
  defaultName?: string
  onPick: (path: string) => void
  onClose: () => void
}) {
  const [listing, setListing] = useState<BrowseOut | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [fileName, setFileName] = useState(defaultName ?? '')

  const load = (path?: string) => {
    api
      .browse(path, ext)
      .then((l) => {
        setListing(l)
        setError(null)
      })
      .catch((e) => setError((e as Error).message))
  }

  useEffect(() => {
    load(startPath ?? undefined)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  const joinPath = (dir: string, name: string) => {
    const sep = dir.includes('\\') && !dir.includes('/') ? '\\' : '/'
    return dir.endsWith(sep) ? `${dir}${name}` : `${dir}${sep}${name}`
  }

  const doSave = () => {
    if (!listing || !fileName.trim()) return
    const name = ext && !fileName.trim().toLowerCase().endsWith(ext.toLowerCase()) ? `${fileName.trim()}${ext}` : fileName.trim()
    onPick(joinPath(listing.path, name))
  }

  return (
    <div
      style={{
        position: 'fixed',
        inset: 0,
        background: 'rgba(0,0,0,0.2)',
        zIndex: 60,
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
      }}
      onClick={onClose}
    >
      <div
        onClick={(e) => e.stopPropagation()}
        style={{
          width: 480,
          maxHeight: '70vh',
          background: '#fff',
          border: '1px solid #d1d5db',
          borderRadius: 8,
          boxShadow: '0 10px 30px rgba(0,0,0,0.25)',
          display: 'flex',
          flexDirection: 'column',
          fontSize: 12,
        }}
      >
        <div style={{ padding: 10, borderBottom: '1px solid #e5e7eb', display: 'flex', alignItems: 'center', gap: 6 }}>
          <button disabled={!listing?.parent} onClick={() => listing?.parent && load(listing.parent)}>
            Up
          </button>
          <div style={{ flex: 1, color: '#374151', fontFamily: 'monospace', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
            {listing?.path ?? ''}
          </div>
          <button onClick={onClose}>Close</button>
        </div>
        <div style={{ overflowY: 'auto', flex: 1 }}>
          {error && <div style={{ padding: 10, color: '#b91c1c' }}>{error}</div>}
          {listing?.entries.length === 0 && <div style={{ padding: 10, color: '#9ca3af' }}>(empty)</div>}
          {listing?.entries.map((e) => (
            <div
              key={e.path}
              onClick={() => (e.is_dir ? load(e.path) : mode === 'save' ? setFileName(e.name) : onPick(e.path))}
              style={{
                padding: '6px 10px',
                cursor: 'pointer',
                display: 'flex',
                gap: 6,
                borderBottom: '1px solid #f3f4f6',
              }}
              onMouseEnter={(ev) => (ev.currentTarget.style.background = '#f9fafb')}
              onMouseLeave={(ev) => (ev.currentTarget.style.background = 'transparent')}
            >
              <span>{e.is_dir ? '\u{1F4C1}' : '\u{1F4C4}'}</span>
              <span>{e.name}</span>
            </div>
          ))}
        </div>
        {mode === 'save' && (
          <div style={{ padding: 10, borderTop: '1px solid #e5e7eb', display: 'flex', gap: 6 }}>
            <input
              autoFocus
              value={fileName}
              onChange={(e) => setFileName(e.target.value)}
              onKeyDown={(e) => e.key === 'Enter' && doSave()}
              placeholder="filename"
              style={{ flex: 1, fontFamily: 'monospace', fontSize: 12, boxSizing: 'border-box' }}
            />
            <button disabled={!fileName.trim()} onClick={doSave}>
              Save
            </button>
          </div>
        )}
      </div>
    </div>
  )
}
