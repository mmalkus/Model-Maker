import { useEffect, useState } from 'react'
import { api } from './api'
import type { LLMSettingsOut } from './types'

const PROVIDER_HELP: Record<string, string> = {
  lmstudio: 'Local LM Studio server -- start it from Developer > Start Server, then Fetch models below.',
  claude_cli: 'Shells out to the locally installed `claude` CLI, using its existing login.',
  anthropic: 'Calls the Anthropic API directly -- needs an API key below (or ANTHROPIC_API_KEY / `ant auth login`).',
  openai:
    'Calls the OpenAI API, or any OpenAI-compatible endpoint (Groq, Together, OpenRouter, Fireworks, a self-hosted server, ...) -- set a custom Base URL below to point elsewhere.',
  gemini: 'Calls the Google Gemini API directly -- needs an API key below.',
  stub: 'No LLM calls -- returns the input unchanged. Useful for testing.',
}

// Providers that take an API key, and the endpoint field each one accepts a
// custom Base URL for (openai only -- to support OpenAI-compatible services).
const KEY_PROVIDERS = new Set(['anthropic', 'openai', 'gemini'])
const BASE_URL_PROVIDERS = new Set(['lmstudio', 'openai'])

export function SettingsPanel({
  settings,
  onChange,
  onClose,
}: {
  settings: LLMSettingsOut | null
  onChange: (s: LLMSettingsOut) => void
  onClose: () => void
}) {
  const provider = settings?.active_provider ?? ''
  const providerSettings = settings?.settings[provider] ?? {}
  const [baseUrl, setBaseUrl] = useState(providerSettings.base_url ?? '')
  const [model, setModel] = useState(providerSettings.model ?? '')
  const [apiKeyInput, setApiKeyInput] = useState('')
  const [modelOptions, setModelOptions] = useState<string[]>([])
  const [fetchingModels, setFetchingModels] = useState(false)
  const [modelsError, setModelsError] = useState<string | null>(null)
  const [busy, setBusy] = useState(false)

  // Reset local text fields whenever the active provider (or the settings
  // object itself, e.g. after a save) changes underneath us. The API key
  // field always starts blank -- its value is never sent back by the
  // server, so there is nothing to prefill.
  useEffect(() => {
    setBaseUrl(providerSettings.base_url ?? '')
    setModel(providerSettings.model ?? '')
    setApiKeyInput('')
    setModelOptions([])
    setModelsError(null)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [provider, providerSettings.base_url, providerSettings.model])

  const update = async (body: Parameters<typeof api.updateLlmSettings>[0]) => {
    setBusy(true)
    try {
      onChange(await api.updateLlmSettings(body))
    } catch (e) {
      alert((e as Error).message)
    } finally {
      setBusy(false)
    }
  }

  const fetchModels = async () => {
    setFetchingModels(true)
    setModelsError(null)
    try {
      const { models } = await api.llmModels(provider, BASE_URL_PROVIDERS.has(provider) ? baseUrl.trim() || undefined : undefined)
      setModelOptions(models)
    } catch (e) {
      setModelsError((e as Error).message)
    } finally {
      setFetchingModels(false)
    }
  }

  const saveApiKey = () => {
    const trimmed = apiKeyInput.trim()
    if (!trimmed) return
    update({ settings: { [provider]: { api_key: trimmed } } }).then(() => setApiKeyInput(''))
  }

  const clearApiKey = () => update({ settings: { [provider]: { api_key: null } } })

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
        width: 300,
        boxShadow: '0 10px 30px rgba(0,0,0,0.25)',
      }}
    >
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'baseline', marginBottom: 8 }}>
        <strong style={{ fontSize: 13 }}>Settings</strong>
        <button onClick={onClose}>Close</button>
      </div>

      {!settings ? (
        <div style={{ color: '#9ca3af', fontSize: 12 }}>Loading...</div>
      ) : (
        <>
          <label style={{ display: 'block', fontSize: 12, marginBottom: 8 }}>
            <div style={{ color: '#6b7280', marginBottom: 2 }}>AI provider (used to draft/fix block code)</div>
            <select
              value={provider}
              onChange={(e) => update({ active_provider: e.target.value })}
              disabled={busy || settings.providers.length === 0}
              style={{ width: '100%', fontSize: 12 }}
            >
              {settings.providers.length === 0 && <option value="">(unavailable)</option>}
              {settings.providers.map((p) => (
                <option key={p} value={p}>
                  {p}
                </option>
              ))}
            </select>
            {PROVIDER_HELP[provider] && <div style={{ color: '#9ca3af', marginTop: 2 }}>{PROVIDER_HELP[provider]}</div>}
          </label>

          {BASE_URL_PROVIDERS.has(provider) && (
            <label style={{ display: 'block', fontSize: 12, marginBottom: 8 }}>
              <div style={{ color: '#6b7280', marginBottom: 2 }}>
                {provider === 'openai' ? 'Base URL (optional -- for OpenAI-compatible endpoints)' : 'Base URL (address:port)'}
              </div>
              <input
                value={baseUrl}
                onChange={(e) => setBaseUrl(e.target.value)}
                onBlur={() => {
                  const trimmed = baseUrl.trim()
                  if (trimmed !== (providerSettings.base_url ?? '')) {
                    update({ settings: { [provider]: { base_url: trimmed || null } } })
                  }
                }}
                placeholder={provider === 'openai' ? 'https://api.openai.com/v1' : 'http://localhost:1234/v1'}
                disabled={busy}
                style={{ width: '100%', fontSize: 12, fontFamily: 'monospace', boxSizing: 'border-box' }}
              />
            </label>
          )}

          {KEY_PROVIDERS.has(provider) && (
            <label style={{ display: 'block', fontSize: 12, marginBottom: 8 }}>
              <div style={{ color: '#6b7280', marginBottom: 2 }}>API key</div>
              <div style={{ display: 'flex', gap: 6 }}>
                <input
                  type="password"
                  value={apiKeyInput}
                  onChange={(e) => setApiKeyInput(e.target.value)}
                  onBlur={saveApiKey}
                  onKeyDown={(e) => e.key === 'Enter' && saveApiKey()}
                  placeholder={
                    providerSettings.api_key_set
                      ? providerSettings.api_key_source === 'env'
                        ? 'set via environment variable'
                        : 'set -- type to replace'
                      : 'not set'
                  }
                  disabled={busy}
                  autoComplete="off"
                  style={{ flex: 1, fontSize: 12, fontFamily: 'monospace', boxSizing: 'border-box', minWidth: 0 }}
                />
                {providerSettings.api_key_source === 'override' && (
                  <button disabled={busy} onClick={clearApiKey} style={{ fontSize: 11 }}>
                    Clear
                  </button>
                )}
              </div>
              <div style={{ color: '#9ca3af', marginTop: 2 }}>
                Kept in server memory only for this run -- never written to disk, and never sent back to the browser.
              </div>
            </label>
          )}

          {provider && provider !== 'stub' && (
            <label style={{ display: 'block', fontSize: 12, marginBottom: 8 }}>
              <div style={{ color: '#6b7280', marginBottom: 2 }}>
                Model{provider === 'lmstudio' ? ' (blank = auto-detect loaded model)' : ''}
              </div>
              <div style={{ display: 'flex', gap: 6 }}>
                <input
                  value={model}
                  onChange={(e) => setModel(e.target.value)}
                  onBlur={() => {
                    const trimmed = model.trim()
                    if (trimmed !== (providerSettings.model ?? '')) {
                      update({ settings: { [provider]: { model: trimmed || null } } })
                    }
                  }}
                  placeholder="(default)"
                  disabled={busy}
                  list="llm-model-options"
                  style={{ flex: 1, fontSize: 12, fontFamily: 'monospace', boxSizing: 'border-box', minWidth: 0 }}
                />
                <button disabled={busy || fetchingModels} onClick={fetchModels} style={{ fontSize: 11 }}>
                  {fetchingModels ? '...' : 'Fetch'}
                </button>
              </div>
              <datalist id="llm-model-options">
                {modelOptions.map((m) => (
                  <option key={m} value={m} />
                ))}
              </datalist>
              {modelOptions.length > 0 && (
                <select
                  value=""
                  onChange={(e) => {
                    if (!e.target.value) return
                    setModel(e.target.value)
                    update({ settings: { [provider]: { model: e.target.value } } })
                  }}
                  disabled={busy}
                  style={{ width: '100%', fontSize: 12, marginTop: 4 }}
                >
                  <option value="">{`${modelOptions.length} model(s) found -- pick one...`}</option>
                  {modelOptions.map((m) => (
                    <option key={m} value={m}>
                      {m}
                    </option>
                  ))}
                </select>
              )}
              {modelsError && <div style={{ color: '#b91c1c', marginTop: 2 }}>{modelsError}</div>}
            </label>
          )}

          <label style={{ display: 'flex', alignItems: 'center', gap: 6, fontSize: 12, marginTop: 4 }}>
            <input
              type="checkbox"
              checked={settings.include_reference ?? true}
              disabled={busy}
              onChange={(e) => update({ include_reference: e.target.checked })}
            />
            <span>
              Include Polars reference examples in prompts
              <div style={{ color: '#9ca3af', fontWeight: 400 }}>
                Helps smaller/local models get exact Polars syntax right, at the cost of extra tokens per call.
              </div>
            </span>
          </label>
        </>
      )}
    </div>
  )
}
