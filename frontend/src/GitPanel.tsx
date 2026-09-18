import { useEffect, useState } from 'react'
import { api } from './api'
import type { GitStatusOut } from './types'

export function GitPanel({
  projectPath,
  onClose,
}: {
  projectPath: string | null
  onClose: () => void
}) {
  const [status, setStatus] = useState<GitStatusOut | null>(null)
  const [remoteInput, setRemoteInput] = useState('')
  const [message, setMessage] = useState('')
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const reload = () => api.gitStatus().then(setStatus).catch((e) => setError((e as Error).message))

  useEffect(() => {
    reload()
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  useEffect(() => {
    setRemoteInput(status?.remote ?? '')
  }, [status?.remote])

  const run = async (fn: () => Promise<GitStatusOut>) => {
    setBusy(true)
    setError(null)
    try {
      setStatus(await fn())
    } catch (e) {
      setError((e as Error).message)
    } finally {
      setBusy(false)
    }
  }

  const doInit = () => run(() => api.gitInit())
  const doSetRemote = () => remoteInput.trim() && run(() => api.gitSetRemote(remoteInput.trim()))
  const doCommit = () => {
    if (!message.trim()) return
    run(() => api.gitCommit(message.trim())).then(() => setMessage(''))
  }
  const doPush = () => run(() => api.gitPush())
  const doPull = () => run(() => api.gitPull())

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
        width: 320,
        boxShadow: '0 10px 30px rgba(0,0,0,0.25)',
        fontSize: 12,
      }}
    >
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'baseline', marginBottom: 8 }}>
        <strong style={{ fontSize: 13 }}>Git</strong>
        <button onClick={onClose}>Close</button>
      </div>

      {!projectPath ? (
        <div style={{ color: '#9ca3af' }}>Save the project first -- git works on the project's own folder.</div>
      ) : !status ? (
        <div style={{ color: '#9ca3af' }}>Loading...</div>
      ) : !status.is_repo ? (
        <>
          <div style={{ color: '#9ca3af', marginBottom: 8 }}>This project's folder isn't a git repo yet.</div>
          <button disabled={busy} onClick={doInit}>
            Initialize repo
          </button>
        </>
      ) : (
        <>
          <div style={{ marginBottom: 8, color: '#374151' }}>
            Branch <strong>{status.branch ?? '(none)'}</strong>
            {status.remote && (status.ahead > 0 || status.behind > 0) && (
              <span style={{ color: '#9ca3af' }}>
                {' '}
                ({status.ahead > 0 ? `${status.ahead} ahead` : ''}
                {status.ahead > 0 && status.behind > 0 ? ', ' : ''}
                {status.behind > 0 ? `${status.behind} behind` : ''})
              </span>
            )}
          </div>

          <label style={{ display: 'block', marginBottom: 8 }}>
            <div style={{ color: '#6b7280', marginBottom: 2 }}>Remote (GitHub) URL</div>
            <div style={{ display: 'flex', gap: 6 }}>
              <input
                value={remoteInput}
                onChange={(e) => setRemoteInput(e.target.value)}
                onKeyDown={(e) => e.key === 'Enter' && doSetRemote()}
                placeholder="https://github.com/you/repo.git"
                disabled={busy}
                style={{ flex: 1, fontFamily: 'monospace', fontSize: 11, boxSizing: 'border-box', minWidth: 0 }}
              />
              <button disabled={busy || !remoteInput.trim() || remoteInput.trim() === status.remote} onClick={doSetRemote}>
                Set
              </button>
            </div>
          </label>

          <div style={{ marginBottom: 8 }}>
            <div style={{ color: '#6b7280', marginBottom: 2 }}>
              Changes {status.changes.length > 0 ? `(${status.changes.length})` : '(none)'}
            </div>
            {status.changes.length > 0 && (
              <div style={{ maxHeight: 100, overflowY: 'auto', border: '1px solid #e5e7eb', borderRadius: 4, padding: 4 }}>
                {status.changes.map((c) => (
                  <div key={c.path} style={{ fontFamily: 'monospace', fontSize: 11, display: 'flex', gap: 6 }}>
                    <span style={{ color: '#9ca3af', width: 20 }}>{c.status}</span>
                    <span>{c.path}</span>
                  </div>
                ))}
              </div>
            )}
          </div>

          <label style={{ display: 'block', marginBottom: 8 }}>
            <div style={{ color: '#6b7280', marginBottom: 2 }}>Commit message</div>
            <div style={{ display: 'flex', gap: 6 }}>
              <input
                value={message}
                onChange={(e) => setMessage(e.target.value)}
                onKeyDown={(e) => e.key === 'Enter' && doCommit()}
                placeholder="Describe the change..."
                disabled={busy}
                style={{ flex: 1, fontSize: 11, boxSizing: 'border-box', minWidth: 0 }}
              />
              <button disabled={busy || !message.trim() || status.changes.length === 0} onClick={doCommit}>
                Commit
              </button>
            </div>
          </label>

          <div style={{ display: 'flex', gap: 6 }}>
            <button disabled={busy || !status.remote} onClick={doPush} style={{ flex: 1 }}>
              Push
            </button>
            <button disabled={busy || !status.remote} onClick={doPull} style={{ flex: 1 }}>
              Pull
            </button>
          </div>
        </>
      )}

      {error && <div style={{ color: '#b91c1c', marginTop: 8 }}>{error}</div>}
    </div>
  )
}
