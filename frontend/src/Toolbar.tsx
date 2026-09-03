import { useState } from 'react'
import { api } from './api'

export function Toolbar({ onChanged, projectPath }: { onChanged: () => void; projectPath: string | null }) {
  const [busy, setBusy] = useState(false)
  const [compiled, setCompiled] = useState<string | null>(null)

  const run = async (fn: () => Promise<unknown>) => {
    setBusy(true)
    try {
      const result = await fn()
      onChanged()
      return result
    } catch (e) {
      alert((e as Error).message)
    } finally {
      setBusy(false)
    }
  }

  const doCompile = () =>
    run(async () => {
      const { source } = await api.compile()
      setCompiled(source)
    })

  const doSave = () =>
    run(() => api.save(projectPath ?? prompt('Save project to path:', 'project.json') ?? undefined))

  const doLoad = () => {
    const path = prompt('Load project from path:', projectPath ?? 'project.json')
    if (path) run(() => api.load(path))
  }

  return (
    <div style={{ display: 'flex', gap: 8, padding: 8, borderBottom: '1px solid #e5e7eb', alignItems: 'center' }}>
      <strong style={{ marginRight: 8 }}>Model-Maker</strong>
      <button disabled={busy} onClick={() => run(() => api.runAll())}>
        Run all
      </button>
      <button disabled={busy} onClick={() => run(() => api.forceRunAll())}>
        Force run all
      </button>
      <button disabled={busy} onClick={() => run(() => api.refreshAll())}>
        Refresh sources
      </button>
      <button disabled={busy} onClick={() => run(() => api.checkAllSources())}>
        Check all sources
      </button>
      <button disabled={busy} onClick={doCompile}>
        Compile
      </button>
      <span style={{ flex: 1 }} />
      <button disabled={busy} onClick={doSave}>
        Save
      </button>
      <button disabled={busy} onClick={doLoad}>
        Load
      </button>
      <span style={{ fontSize: 12, color: '#6b7280' }}>{projectPath ?? '(unsaved)'}</span>

      {compiled !== null && (
        <div
          style={{
            position: 'fixed',
            inset: 40,
            background: '#fff',
            border: '1px solid #d1d5db',
            borderRadius: 8,
            padding: 16,
            zIndex: 50,
            display: 'flex',
            flexDirection: 'column',
            boxShadow: '0 10px 30px rgba(0,0,0,0.25)',
          }}
        >
          <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 8 }}>
            <strong>Compiled script</strong>
            <button onClick={() => setCompiled(null)}>Close</button>
          </div>
          <pre style={{ overflow: 'auto', flex: 1, fontSize: 12, background: '#f9fafb', padding: 12, margin: 0 }}>
            {compiled}
          </pre>
        </div>
      )}
    </div>
  )
}
