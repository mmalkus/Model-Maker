import { useState } from 'react'
import { api } from './api'
import { LlmSettingsPanel } from './LlmSettingsPanel'

export function Toolbar({ onChanged, projectPath }: { onChanged: () => void; projectPath: string | null }) {
  const [busy, setBusy] = useState(false)
  const [compiled, setCompiled] = useState<string | null>(null)
  const [showLlmSettings, setShowLlmSettings] = useState(false)

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
    <div
      style={{
        display: 'flex',
        gap: 8,
        padding: '8px 12px',
        alignItems: 'center',
        background: 'var(--brand-light)',
        borderBottom: '1px solid rgba(11,35,64,0.1)',
      }}
    >
      <img src="/brand-quantology/icon-mark-light.svg" alt="" width={24} height={24} style={{ display: 'block' }} />
      <div style={{ display: 'flex', alignItems: 'baseline', gap: 6, marginRight: 12 }}>
        <strong style={{ color: 'var(--brand-ink)', fontFamily: 'var(--font-heading)', fontSize: 16, fontWeight: 600 }}>
          Quantology
        </strong>
        <span style={{ color: 'var(--brand-border)' }}>|</span>
        <span style={{ color: 'var(--brand-charcoal)', fontWeight: 500 }}>Model Maker</span>
      </div>
      <button className="brand-primary" disabled={busy} onClick={() => run(() => api.runAll())}>
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
      <button disabled={busy} onClick={() => setShowLlmSettings(true)} title="LLM provider, model, and prompt settings">
        LLM settings
      </button>
      <span style={{ fontSize: 12, color: '#6b7280' }}>{projectPath ?? '(unsaved)'}</span>

      {showLlmSettings && <LlmSettingsPanel onClose={() => setShowLlmSettings(false)} />}

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
