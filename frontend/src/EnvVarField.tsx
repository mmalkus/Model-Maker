import { useEffect, useState } from 'react'
import { api } from './api'
import type { EnvVarStatus } from './types'

// Lets the user set an env var's value straight from the UI -- e.g. a Read
// SQL block's connection_env -- instead of having to edit a .env file or
// export it in their shell before starting the server. The value itself
// never comes back from the API (see types.EnvVarStatus); only whether
// it's set, and whether that came from here ("override", session-only) or
// was already present in the environment ("env").
export function EnvVarField({ name }: { name: string }) {
  const [status, setStatus] = useState<EnvVarStatus | null>(null)
  const [value, setValue] = useState('')
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    setStatus(null)
    setValue('')
    setError(null)
    const trimmed = name.trim()
    if (!trimmed) return
    api
      .envVar(trimmed)
      .then(setStatus)
      .catch((e) => setError((e as Error).message))
  }, [name])

  const trimmedName = name.trim()
  if (!trimmedName) return null

  const save = () => {
    if (!value.trim()) return
    setBusy(true)
    api
      .setEnvVar(trimmedName, value)
      .then((s) => {
        setStatus(s)
        setValue('')
        setError(null)
      })
      .catch((e) => setError((e as Error).message))
      .finally(() => setBusy(false))
  }

  const clear = () => {
    setBusy(true)
    api
      .clearEnvVar(trimmedName)
      .then(setStatus)
      .catch((e) => setError((e as Error).message))
      .finally(() => setBusy(false))
  }

  return (
    <div style={{ marginBottom: 12 }}>
      <div style={{ fontWeight: 600, marginBottom: 4 }}>Env var: {trimmedName}</div>
      <div style={{ fontSize: 11, color: '#6b7280', marginBottom: 4 }}>
        {status === null
          ? '…'
          : status.is_set
            ? status.source === 'override'
              ? 'Set for this session only -- not saved to disk, gone on restart.'
              : 'Already set from your shell environment or .env file.'
            : 'Not set -- the block will fail to read this until it is.'}
      </div>
      <div style={{ display: 'flex', gap: 6 }}>
        <input
          type="password"
          value={value}
          onChange={(e) => setValue(e.target.value)}
          onKeyDown={(e) => e.key === 'Enter' && save()}
          placeholder="paste value, e.g. postgresql://user:pass@host/db"
          disabled={busy}
          style={{ flex: 1, fontFamily: 'monospace', fontSize: 11, boxSizing: 'border-box' }}
        />
        <button disabled={busy || !value.trim()} onClick={save}>
          Set
        </button>
        {status?.source === 'override' && (
          <button disabled={busy} onClick={clear}>
            Clear
          </button>
        )}
      </div>
      {error && <div style={{ color: '#b91c1c', fontSize: 11 }}>{error}</div>}
    </div>
  )
}
