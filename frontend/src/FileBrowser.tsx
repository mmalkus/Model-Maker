import { useEffect, useState } from 'react'
import { api } from './api'
import type { BrowseOut } from './types'

export function FileBrowser({
  startPath,
  ext,
  mode = 'open',
  folders = false,
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
  // Project-folder picker instead of a file picker: only directories are
  // listed, and a folder that already holds a project (BrowseEntry.is_project
  // -- see project.PROJECT_FILENAME) is a pickable/reusable leaf rather than
  // something to navigate into, the way a file is in the ext-filtered picker.
  folders?: boolean
  defaultName?: string
  onPick: (path: string) => void
  onClose: () => void
}) {
  const [listing, setListing] = useState<BrowseOut | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [fileName, setFileName] = useState(defaultName ?? '')

  const load = (path?: string) => {
    api
      .browse(path, folders ? undefined : ext)
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
    const trimmed = fileName.trim()
    const name = !folders && ext && !trimmed.toLowerCase().endsWith(ext.toLowerCase()) ? `${trimmed}${ext}` : trimmed
    onPick(joinPath(listing.path, name))
  }

  const entries = folders ? (listing?.entries.filter((e) => e.is_dir) ?? []) : (listing?.entries ?? [])

  const onEntryClick = (e: BrowseOut['entries'][number]) => {
    if (folders) {
      // A folder that's already a project is the pickable leaf here (like a
      // file in the ext-filtered picker); a plain folder is just somewhere
      // to browse through on the way to one.
      if (e.is_project) {
        if (mode === 'save') setFileName(e.name)
        else onPick(e.path)
      } else {
        load(e.path)
      }
      return
    }
    if (e.is_dir) load(e.path)
    else if (mode === 'save') setFileName(e.name)
    else onPick(e.path)
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
          {entries.length === 0 && <div style={{ padding: 10, color: '#9ca3af' }}>(empty)</div>}
          {entries.map((e) => (
            <div
              key={e.path}
              onClick={() => onEntryClick(e)}
              title={folders && e.is_project ? 'Project folder' : undefined}
              style={{
                padding: '6px 10px',
                cursor: 'pointer',
                display: 'flex',
                gap: 6,
                borderBottom: '1px solid #f3f4f6',
                fontWeight: folders && e.is_project ? 600 : undefined,
              }}
              onMouseEnter={(ev) => (ev.currentTarget.style.background = '#f9fafb')}
              onMouseLeave={(ev) => (ev.currentTarget.style.background = 'transparent')}
            >
              <span>{folders ? (e.is_project ? '\u{1F4E6}' : '\u{1F4C1}') : e.is_dir ? '\u{1F4C1}' : '\u{1F4C4}'}</span>
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
              placeholder={folders ? 'folder name' : 'filename'}
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
