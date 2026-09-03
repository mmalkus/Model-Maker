import { useEffect, useState } from 'react'
import { api } from './api'
import type { BrowseOut } from './types'

export function FileBrowser({
  startPath,
  ext,
  onPick,
  onClose,
}: {
  startPath?: string | null
  ext?: string
  onPick: (path: string) => void
  onClose: () => void
}) {
  const [listing, setListing] = useState<BrowseOut | null>(null)
  const [error, setError] = useState<string | null>(null)

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
              onClick={() => (e.is_dir ? load(e.path) : onPick(e.path))}
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
      </div>
    </div>
  )
}
