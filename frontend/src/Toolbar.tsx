import { useEffect, useState } from 'react'
import { api } from './api'
import { CodeEditor } from './CodeEditor'
import { FileBrowser } from './FileBrowser'
import { SettingsPanel } from './SettingsPanel'
import type { LLMSettingsOut } from './types'

// Splits a server-side path into (directory, filename), tolerating both
// '/' (POSIX) and '\' (Windows) separators, whichever the path was given
// with -- used to default the Save/Load file browser to wherever the
// currently-open project already lives.
function splitPath(path: string): { dir: string; name: string } {
  const idx = Math.max(path.lastIndexOf('/'), path.lastIndexOf('\\'))
  return idx === -1 ? { dir: '', name: path } : { dir: path.slice(0, idx), name: path.slice(idx + 1) }
}

// Default cap offered when sample mode is switched on -- small enough that
// a pipeline over millions of rows becomes interactive, large enough that a
// train/test split or a grouped metric still has something to work with.
const DEFAULT_SAMPLE_ROWS = 1000

export function Toolbar({
  onChanged,
  projectPath,
  llmSettings,
  onLlmSettingsChange,
  running,
  dirty,
  canUndo,
  canRedo,
  onUndo,
  onRedo,
  sampleRows,
}: {
  onChanged: () => void
  projectPath: string | null
  llmSettings: LLMSettingsOut | null
  onLlmSettingsChange: (settings: LLMSettingsOut) => void
  // Whether any block is currently mid-run on the server (see
  // BlockOut.status === 'running', polled by App) -- distinct from the
  // local `busy` below, which only covers this component's own in-flight
  // fetch (a run's own POST resolves quickly now that runs execute in the
  // background; `running` is what stays true for the run's actual duration).
  running: boolean
  dirty: boolean
  canUndo: boolean
  canRedo: boolean
  onUndo: () => void
  onRedo: () => void
  sampleRows: number | null
}) {
  const [busy, setBusy] = useState(false)
  const [compiled, setCompiled] = useState<string | null>(null)
  const [compileStreaming, setCompileStreaming] = useState(false)
  const [showSettings, setShowSettings] = useState(false)
  const [showSave, setShowSave] = useState(false)
  const [showLoad, setShowLoad] = useState(false)
  const [defaultProjectsDir, setDefaultProjectsDir] = useState<string | null>(null)

  useEffect(() => {
    api
      .projectDefaultDir()
      .then((d) => setDefaultProjectsDir(d.path))
      .catch(() => setDefaultProjectsDir(null))
  }, [])

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
      const { source } = await api.compile(undefined, compileStreaming)
      setCompiled(source)
    })

  const doSavePick = (path: string) => {
    setShowSave(false)
    run(() => api.save(path))
  }

  const doLoadPick = (path: string) => {
    setShowLoad(false)
    run(() => api.load(path))
  }

  // Both dialogs default to wherever the currently-open project lives; with
  // no project open yet they fall back to the server's default projects
  // subdirectory (see /api/project/default_dir), so Save always lands
  // somewhere sensible instead of the server's raw working directory.
  const { dir: openDir, name: openName } = projectPath ? splitPath(projectPath) : { dir: '', name: 'project.json' }
  const browseStartPath = openDir || defaultProjectsDir

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
      <a
        href="https://www.quantology.nl/"
        target="_blank"
        rel="noreferrer"
        title="Quantology"
        style={{ display: 'flex', alignItems: 'baseline', gap: 6, marginRight: 12, textDecoration: 'none' }}
      >
        <img src="/brand-quantology/icon-mark-light.svg" alt="" width={24} height={24} style={{ display: 'block' }} />
        <strong style={{ color: 'var(--brand-ink)', fontFamily: 'var(--font-heading)', fontSize: 16, fontWeight: 600 }}>
          Quantology
        </strong>
        <span style={{ color: 'var(--brand-border)' }}>|</span>
        <span style={{ color: 'var(--brand-charcoal)', fontWeight: 500 }}>Model Maker</span>
      </a>
      <button className="brand-primary" disabled={busy || running} onClick={() => run(() => api.runAll())}>
        Run all
      </button>
      <button disabled={busy || running} onClick={() => run(() => api.forceRunAll())}>
        Force run all
      </button>
      <button
        disabled={busy || running}
        title="Fuse compatible blocks (filter/select/group-by/join, read from source) into one streaming query per group, so a source larger than memory doesn't fully load at every step. Fused mid-chain blocks won't individually turn green -- only the block ending each fused stretch does. For a full, production-scale run; use sample mode instead for cheap interactive iteration."
        onClick={() => run(() => api.runAllStreaming())}
      >
        Run all (streaming)
      </button>
      <button disabled={busy || running} onClick={() => run(() => api.refreshAll())}>
        Refresh sources
      </button>
      <button disabled={busy || running} onClick={() => run(() => api.checkAllSources())}>
        Check all sources
      </button>
      {running && (
        <button
          onClick={() => run(() => api.cancelRun())}
          style={{ color: '#b91c1c', borderColor: '#fecaca', display: 'inline-flex', alignItems: 'center', gap: 5 }}
          title="Stop the run in progress -- the block currently executing is interrupted immediately"
        >
          <span
            style={{
              width: 8,
              height: 8,
              borderRadius: 999,
              background: '#ef4444',
              display: 'inline-block',
              animation: 'mm-pulse 1s ease-in-out infinite',
            }}
          />
          Stop
        </button>
      )}
      <button disabled={busy} onClick={doCompile}>
        Compile
      </button>
      <label
        style={{ display: 'inline-flex', alignItems: 'center', gap: 4, fontSize: 12, color: 'var(--brand-charcoal)' }}
        title="Compile fusable stretches (filter/select/group-by/join, read from source) into a single streaming polars query each, the same way 'Run all (streaming)' does for the live engine -- for a compiled script meant to run on data larger than memory."
      >
        <input
          type="checkbox"
          checked={compileStreaming}
          disabled={busy}
          onChange={(e) => setCompileStreaming(e.target.checked)}
        />
        Streaming
      </label>

      <span style={{ width: 1, alignSelf: 'stretch', background: 'rgba(11,35,64,0.15)', margin: '0 2px' }} />
      <button disabled={busy || !canUndo} onClick={onUndo} title="Undo the last graph edit (Ctrl/Cmd-Z)">
        ↶
      </button>
      <button disabled={busy || !canRedo} onClick={onRedo} title="Redo (Ctrl/Cmd-Shift-Z)">
        ↷
      </button>

      <label
        style={{ display: 'inline-flex', alignItems: 'center', gap: 5, fontSize: 12, color: 'var(--brand-charcoal)' }}
        title="Run the whole pipeline on the first N rows of every source, for fast iteration. Sources need re-reading when this changes."
      >
        <input
          type="checkbox"
          checked={sampleRows != null}
          disabled={busy || running}
          onChange={(e) => run(() => api.setSampleMode(e.target.checked ? DEFAULT_SAMPLE_ROWS : null))}
        />
        Sample
        {sampleRows != null && (
          <input
            type="number"
            min={1}
            value={sampleRows}
            disabled={busy || running}
            onChange={(e) => {
              const rows = Number(e.target.value)
              if (Number.isFinite(rows) && rows >= 1) run(() => api.setSampleMode(rows))
            }}
            style={{ width: 72, fontSize: 11 }}
          />
        )}
      </label>

      <span style={{ flex: 1 }} />
      <button disabled={busy} onClick={() => setShowSave(true)}>
        Save{dirty ? ' •' : ''}
      </button>
      <button disabled={busy} onClick={() => setShowLoad(true)}>
        Load
      </button>
      <span style={{ fontSize: 12, color: dirty ? '#92400e' : '#6b7280' }}>
        {projectPath ?? '(unsaved)'}
        {dirty && <span title="Edits not yet written to the project file (they are snapshotted for recovery)"> · unsaved changes</span>}
      </span>

      <div style={{ position: 'relative' }}>
        <button onClick={() => setShowSettings((s) => !s)}>Settings</button>
        {showSettings && (
          <SettingsPanel settings={llmSettings} onChange={onLlmSettingsChange} onClose={() => setShowSettings(false)} />
        )}
      </div>

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
          <div style={{ overflow: 'auto', flex: 1 }}>
            <CodeEditor value={compiled} readOnly maxHeight={10_000} />
          </div>
        </div>
      )}

      {showSave && (
        <FileBrowser
          mode="save"
          ext=".json"
          startPath={browseStartPath}
          defaultName={openName}
          onPick={doSavePick}
          onClose={() => setShowSave(false)}
        />
      )}
      {showLoad && (
        <FileBrowser ext=".json" startPath={browseStartPath} onPick={doLoadPick} onClose={() => setShowLoad(false)} />
      )}
    </div>
  )
}
