import { useEffect, useState } from 'react'
import { api } from './api'
import type { LlmSettings } from './types'

export function LlmSettingsPanel({ onClose }: { onClose: () => void }) {
  const [providers, setProviders] = useState<string[]>([])
  const [settings, setSettings] = useState<LlmSettings | null>(null)
  const [saving, setSaving] = useState(false)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    Promise.all([api.llmProviders(), api.llmSettings()])
      .then(([p, s]) => {
        setProviders(p.providers)
        setSettings(s)
      })
      .catch((e) => setError((e as Error).message))
  }, [])

  const save = async () => {
    if (!settings) return
    setSaving(true)
    setError(null)
    try {
      const saved = await api.setLlmSettings(settings)
      setSettings(saved)
      onClose()
    } catch (e) {
      setError((e as Error).message)
    } finally {
      setSaving(false)
    }
  }

  return (
    <div
      style={{
        position: 'fixed',
        inset: 0,
        background: 'rgba(0,0,0,0.3)',
        zIndex: 60,
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
      }}
      onClick={onClose}
    >
      <div
        style={{
          background: '#fff',
          borderRadius: 8,
          padding: 20,
          width: 380,
          boxShadow: '0 10px 30px rgba(0,0,0,0.25)',
        }}
        onClick={(e) => e.stopPropagation()}
      >
        <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 12 }}>
          <strong>LLM settings</strong>
          <button onClick={onClose}>Close</button>
        </div>

        {!settings ? (
          <div>Loading...</div>
        ) : (
          <>
            <label style={{ display: 'block', marginBottom: 10 }}>
              <div style={{ fontWeight: 600, marginBottom: 4 }}>Provider</div>
              <select
                value={settings.provider ?? ''}
                onChange={(e) => setSettings({ ...settings, provider: e.target.value || null })}
                style={{ width: '100%', boxSizing: 'border-box' }}
              >
                <option value="">(default -- MODELMAKER_LLM_PROVIDER env var)</option>
                {providers.map((p) => (
                  <option key={p} value={p}>
                    {p}
                  </option>
                ))}
              </select>
            </label>

            <label style={{ display: 'block', marginBottom: 10 }}>
              <div style={{ fontWeight: 600, marginBottom: 4 }}>Model</div>
              <input
                value={settings.model ?? ''}
                onChange={(e) => setSettings({ ...settings, model: e.target.value || null })}
                placeholder="(default for the chosen provider)"
                style={{ width: '100%', boxSizing: 'border-box' }}
              />
              <div style={{ fontSize: 11, color: '#6b7280', marginTop: 2 }}>
                E.g. a Claude model name, or a local model tag if the provider talks to a local runtime (Ollama, etc).
              </div>
            </label>

            <label style={{ display: 'flex', alignItems: 'flex-start', gap: 6, marginBottom: 4 }}>
              <input
                type="checkbox"
                checked={settings.include_reference}
                onChange={(e) => setSettings({ ...settings, include_reference: e.target.checked })}
                style={{ marginTop: 2 }}
              />
              <span>
                <div style={{ fontWeight: 600 }}>Include Polars reference in prompt</div>
                <div style={{ fontSize: 11, color: '#6b7280' }}>
                  Adds a condensed Polars API cheat sheet + worked example to every draft/fix call. Costs extra
                  tokens on every call -- off by default. Mainly helps smaller or local models that don't reliably
                  recall Polars's exact API.
                </div>
              </span>
            </label>

            {error && <div style={{ color: '#b91c1c', marginTop: 8 }}>{error}</div>}

            <div style={{ display: 'flex', justifyContent: 'flex-end', gap: 6, marginTop: 12 }}>
              <button disabled={saving} onClick={save} className="brand-primary">
                Save
              </button>
            </div>
          </>
        )}
      </div>
    </div>
  )
}
