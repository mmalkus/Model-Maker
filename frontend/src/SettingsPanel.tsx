import { useEffect, useState } from 'react'
import { api } from './api'
import type { LLMSettingsOut, LocalModelStatus } from './types'

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

          {settings.agent && (
            <div style={{ borderTop: '1px solid #e5e7eb', marginTop: 10, paddingTop: 8, fontSize: 12 }}>
              <div style={{ fontWeight: 600, marginBottom: 2 }}>AI builder</div>
              <div style={{ color: '#9ca3af', marginBottom: 6 }}>
                Which LLMs plan and build models with "Build with AI". Planning is short but benefits from the strongest
                model; building is many tool calls. Each can be overridden per build.
              </div>
              <LocalModelCard />
              <AgentLlmRow
                key={`plan-${settings.agent.plan.provider}-${settings.agent.plan.model}`}
                label="Plan"
                choice={settings.agent.plan}
                capable={settings.agent.capable_providers}
                disabled={busy}
                onSave={(v) => update({ agent_plan: v })}
              />
              <AgentLlmRow
                key={`build-${settings.agent.build.provider}-${settings.agent.build.model}`}
                label="Build"
                choice={settings.agent.build}
                capable={settings.agent.capable_providers}
                disabled={busy}
                onSave={(v) => update({ agent_build: v })}
              />
              <label style={{ display: 'flex', alignItems: 'flex-start', gap: 6, marginTop: 6 }}>
                <select
                  value={settings.agent.guided == null ? 'auto' : settings.agent.guided ? 'on' : 'off'}
                  disabled={busy}
                  onChange={(e) => update({ agent_guided: e.target.value === 'auto' ? 'auto' : e.target.value === 'on' })}
                  style={{ fontSize: 12 }}
                >
                  <option value="auto">auto</option>
                  <option value="on">on</option>
                  <option value="off">off</option>
                </select>
                <span>
                  Guided{' '}
                  <input
                    type="number"
                    min={0}
                    placeholder="default"
                    title="Thinking budget per question, in tokens (local models; 0 = no thinking)"
                    defaultValue={settings.agent.think_tokens ?? ''}
                    disabled={busy}
                    onBlur={(e) => update({ agent_think_tokens: e.target.value === '' ? 'default' : Number(e.target.value) })}
                    style={{ width: 70, fontSize: 11 }}
                  />{' '}
                  thinking tokens
                  <div style={{ color: '#9ca3af', fontWeight: 400 }}>
                    The app asks one narrow question at a time -- the plan in one prompt, then per stage which block
                    to place next -- with the data profiled for it. For small models. Auto turns it on for local
                    (LM Studio) models.
                  </div>
                </span>
              </label>
              <label style={{ display: 'flex', alignItems: 'flex-start', gap: 6, marginTop: 6 }}>
                <select
                  value={settings.agent.small_context == null ? 'auto' : settings.agent.small_context ? 'on' : 'off'}
                  disabled={busy}
                  onChange={(e) =>
                    update({ agent_small_context: e.target.value === 'auto' ? 'auto' : e.target.value === 'on' })
                  }
                  style={{ fontSize: 12 }}
                >
                  <option value="auto">auto</option>
                  <option value="on">on</option>
                  <option value="off">off</option>
                </select>
                <span>
                  Small context
                  <div style={{ color: '#9ca3af', fontWeight: 400 }}>
                    Start each build stage in a fresh conversation, carrying only the outline, what's built and the
                    decisions so far. Auto turns it on for a build LLM with a context window under 32k tokens.
                  </div>
                </span>
              </label>
              <label style={{ display: 'flex', alignItems: 'flex-start', gap: 6, marginTop: 6 }}>
                <input
                  type="checkbox"
                  checked={settings.agent.decision_hints ?? false}
                  disabled={busy}
                  onChange={(e) => update({ agent_decision_hints: e.target.checked })}
                />
                <span>
                  Decision hints
                  <div style={{ color: '#9ca3af', fontWeight: 400 }}>
                    Add a short, rule-based "what this means and what to do next" to the build's results (binning,
                    coefficients, Gini, PSI...), so the model reads a recommendation rather than a statistics table.
                    For small models.
                  </div>
                </span>
              </label>
            </div>
          )}
        </>
      )}
    </div>
  )
}

function AgentLlmRow({
  label,
  choice,
  capable,
  disabled,
  onSave,
}: {
  label: string
  choice: { provider: string; model: string | null; provider_set?: boolean; model_set?: boolean }
  capable: string[]
  disabled: boolean
  onSave: (v: { provider?: string | null; model?: string | null }) => void
}) {
  // The caller remounts this row (via `key`) when the saved choice changes,
  // so the local text starts from it without syncing in an effect.
  const [model, setModel] = useState(choice.model_set ? (choice.model ?? '') : '')
  return (
    <div style={{ display: 'flex', gap: 4, alignItems: 'center', marginBottom: 4 }}>
      <span style={{ width: 36, color: '#6b7280' }}>{label}</span>
      <select
        value={choice.provider_set ? choice.provider : ''}
        disabled={disabled}
        onChange={(e) => onSave({ provider: e.target.value || null })}
        style={{ fontSize: 12 }}
      >
        <option value="">active ({choice.provider_set ? '…' : choice.provider})</option>
        {capable.map((p) => (
          <option key={p} value={p}>
            {p}
          </option>
        ))}
      </select>
      <input
        value={model}
        disabled={disabled}
        onChange={(e) => setModel(e.target.value)}
        onBlur={() => {
          const trimmed = model.trim()
          if (trimmed !== (choice.model_set ? (choice.model ?? '') : '')) onSave({ model: trimmed || null })
        }}
        placeholder={choice.model ?? '(default)'}
        style={{ flex: 1, minWidth: 0, fontSize: 12, fontFamily: 'monospace' }}
      />
    </div>
  )
}

// The built-in local model (provider "local"): one-click download, polled
// while it runs. The model itself is never part of the package.
function LocalModelCard() {
  const [status, setStatus] = useState<LocalModelStatus | null>(null)
  const state = status?.download?.state
  const busy = state === 'downloading' || state === 'verifying'
  useEffect(() => {
    api.localModel().then(setStatus, () => {})
  }, [])
  useEffect(() => {
    if (!busy) return
    const timer = setInterval(() => api.localModel().then(setStatus, () => {}), 1000)
    return () => clearInterval(timer)
  }, [busy])
  if (!status) return null
  const gb = (n: number) => `${(n / 1e9).toFixed(2)} GB`
  return (
    <div style={{ border: '1px solid #e5e7eb', borderRadius: 4, padding: 6, marginBottom: 8 }}>
      <div style={{ fontWeight: 600 }}>Local model: {status.name}</div>
      <div style={{ color: '#9ca3af' }}>
        Runs on this computer's CPU as the <code>local</code> provider (guided).{' '}
        <a href={status.license_url} target="_blank" rel="noreferrer">Licence</a>
      </div>
      {!status.runtime_available && <div style={{ color: '#b45309' }}>Needs <code>pip install modelmaker[local]</code>.</div>}
      {status.downloaded ? (
        <div style={{ color: '#15803d' }}>Downloaded -- used for AI builds unless you pick another provider.</div>
      ) : busy && status.download ? (
        <div>
          <progress value={status.download.bytes} max={status.download.total} style={{ width: '100%' }} />
          {state === 'verifying' ? 'Checking...' : `${gb(status.download.bytes)} of ${gb(status.download.total)} `}
          <button onClick={() => api.localModelCancel().then(setStatus)}>Cancel</button>
        </div>
      ) : (
        <div>
          <button onClick={() => api.localModelDownload().then(setStatus)}>Download ({gb(status.size)})</button>
          {state === 'failed' && <span style={{ color: '#b91c1c' }}> {status.download?.error}</span>}
        </div>
      )}
    </div>
  )
}
