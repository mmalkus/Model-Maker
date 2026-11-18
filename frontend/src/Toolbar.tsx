import { useEffect, useState } from 'react'
import type { AutosaveStatus } from './App'
import { api } from './api'
import { CodeEditor } from './CodeEditor'
import { FileBrowser } from './FileBrowser'
import { GitPanel } from './GitPanel'
import { SettingsPanel } from './SettingsPanel'
import type { LLMSettingsOut } from './types'
import { host } from './host'
import { has } from './features'

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
  autosave,
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
  autosave: AutosaveStatus
}) {
  const [busy, setBusy] = useState(false)
  const [compiled, setCompiled] = useState<string | null>(null)
  const [compileStreaming, setCompileStreaming] = useState(false)
  const [showSettings, setShowSettings] = useState(false)
  const [showGit, setShowGit] = useState(false)
  const [showSave, setShowSave] = useState(false)
  const [showLoad, setShowLoad] = useState(false)
  const [showMore, setShowMore] = useState(false)
  const [defaultProjectsDir, setDefaultProjectsDir] = useState<string | null>(null)

  useEffect(() => {
    api
      .projectDefaultDir()
      .then((d) => setDefaultProjectsDir(d.path))
      .catch(() => setDefaultProjectsDir(null))
  }, [])

  // Run all/Force run all/etc. POST and wait server-side for the whole
  // sweep to finish (up to 20s -- see api.py's _start_background_run)
  // before their promise resolves at all, so `run()`'s own onChanged()
  // below is too late to show anything mid-sweep: for any sweep that
  // finishes inside that window (i.e. almost all of them), the frontend
  // would otherwise jump straight from all-grey to done with nothing
  // in between. Polling for as long as *any* run() is in flight -- not
  // just once App's own graph state says a block is "running" -- closes
  // that gap: it starts the instant the button is clicked, well before the
  // first real status update could possibly come back.
  useEffect(() => {
    if (!busy) return
    const id = setInterval(onChanged, 200)
    return () => clearInterval(id)
  }, [busy, onChanged])

  const run = async (fn: () => Promise<unknown>) => {
    setBusy(true)
    try {
      const result = await fn()
      onChanged()
      return result
    } catch (e) {
      host.notify((e as Error).message)
    } finally {
      setBusy(false)
    }
  }

  const doCompile = (streaming = compileStreaming && host.kind !== 'vscode') =>
    run(async () => {
      const { source } = await api.compile(undefined, streaming)
      // In VS Code the script opens as a real editor tab (save, diff, run).
      if (host.openText) host.openText(source, 'python')
      else setCompiled(source)
    })

  const doNew = async () => {
    if (dirty && !(await host.confirm('Discard unsaved changes and start a new project?'))) return
    run(() => api.new())
  }

  // VS Code's native dialogs where the host has them, else the in-app browser.
  const openSave = () => {
    if (!host.pickPath) return setShowSave(true)
    host
      .pickPath({ folders: true, mode: 'save', startPath: browseStartPath, defaultName: openName, title: 'Save project as' })
      .then((path) => path && doSavePick(path))
  }
  const openLoad = () => {
    if (!host.pickPath) return setShowLoad(true)
    host.pickPath({ folders: true, startPath: browseStartPath, title: 'Load project' }).then((path) => path && doLoadPick(path))
  }

  const doSavePick = (path: string) => {
    setShowSave(false)
    run(() => api.save(path))
  }

  const doLoadPick = (path: string) => {
    setShowLoad(false)
    run(() => api.load(path))
  }

  // Both dialogs default to wherever the currently-open project folder
  // lives (one level up, so its siblings are what's listed); with no
  // project open yet they fall back to the server's default projects
  // subdirectory (see /api/project/default_dir), so Save always lands
  // somewhere sensible instead of the server's raw working directory.
  // Inside VS Code: a compact bar -- the tab already names the project,
  // VS Code saves it, and the rarer run modes sit under "More".
  const lean = host.kind === 'vscode'
  const { dir: openDir, name: openName } = projectPath ? splitPath(projectPath) : { dir: '', name: '' }
  const browseStartPath = openDir || defaultProjectsDir

  return (
    <div
      style={{
        display: 'flex',
        gap: 8,
        padding: lean ? '4px 8px' : '8px 12px',
        alignItems: 'center',
        flexWrap: lean ? 'wrap' : undefined,
        background: 'var(--brand-light)',
        borderBottom: '1px solid rgba(11,35,64,0.1)',
      }}
    >
      {!lean && (
        <a
          href="https://www.quantology.nl/"
          target="_blank"
          rel="noreferrer"
          title="Quantology"
          style={{ display: 'flex', alignItems: 'baseline', gap: 6, marginRight: 12, textDecoration: 'none' }}
        >
          <img src={`${import.meta.env.BASE_URL}brand-quantology/icon-mark-light.svg`} alt="" width={24} height={24} style={{ display: 'block' }} />
          <strong style={{ color: 'var(--brand-ink)', fontFamily: 'var(--font-heading)', fontSize: 16, fontWeight: 600 }}>
            Quantology
          </strong>
          <span style={{ color: 'var(--brand-border)' }}>|</span>
          <span style={{ color: 'var(--brand-charcoal)', fontWeight: 500 }}>Model Maker</span>
        </a>
      )}
      <button className="brand-primary" disabled={busy || running} onClick={() => run(() => api.runAll())}>
        Run all
      </button>
      {!lean && (
        <>
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
        </>
      )}
      {running && (
        <button
          onClick={() => run(() => api.cancelRun())}
          style={{ color: 'var(--mm-error)', borderColor: 'var(--mm-error-border)', display: 'inline-flex', alignItems: 'center', gap: 5 }}
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
      <button disabled={busy} onClick={() => doCompile()} title={lean ? 'Compile the pipeline to one Python script, opened in an editor tab' : undefined}>
        Compile
      </button>
      {lean ? (
        <div style={{ position: 'relative' }}>
          <button disabled={busy} onClick={() => setShowMore((m) => !m)} title="More run and compile options">
            More ▾
          </button>
          {showMore && (
            <MoreMenu
              onClose={() => setShowMore(false)}
              items={[
                { label: 'Force run all', disabled: running, onClick: () => run(() => api.forceRunAll()) },
                { label: 'Run all (streaming)', disabled: running, onClick: () => run(() => api.runAllStreaming()) },
                { label: 'Refresh sources', disabled: running, onClick: () => run(() => api.refreshAll()) },
                { label: 'Check all sources', disabled: running, onClick: () => run(() => api.checkAllSources()) },
                { label: 'Compile (streaming)', onClick: () => doCompile(true) },
              ]}
            />
          )}
        </div>
      ) : (
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
      )}

      <span style={{ width: 1, alignSelf: 'stretch', background: 'rgba(11,35,64,0.15)', margin: '0 2px' }} />
      <button disabled={busy || !canUndo} onClick={onUndo} title="Undo the last graph edit (Ctrl/Cmd-Z)">
        ↶
      </button>
      <button disabled={busy || !canRedo} onClick={onRedo} title="Redo (Ctrl/Cmd-Shift-Z)">
        ↷
      </button>

      <label
        style={{ display: 'inline-flex', alignItems: 'center', gap: 5, fontSize: 12, color: 'var(--brand-charcoal)' }}
        title="Run the whole pipeline on the first N rows of every source, for fast iteration. The next Run all re-reads the sources."
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
      {lean ? (
        // The editor tab is the project: Ctrl/Cmd-S saves it and edits
        // autosave, so only say when something needs attention.
        <span style={{ fontSize: 12, color: autosave.state === 'error' ? 'var(--mm-error)' : 'var(--mm-muted)' }}>
          {autosave.state === 'error' ? (
            <span title={autosave.message}>autosave failed</span>
          ) : dirty ? (
            <span title="Edits not yet written to model.json (Ctrl/Cmd-S saves now; autosaves shortly)">
              {autosave.state === 'saving' ? 'saving…' : '● unsaved'}
            </span>
          ) : null}
        </span>
      ) : (
        <>
          <button disabled={busy} onClick={doNew} title="Start a new, empty project">
            New
          </button>
          <button disabled={busy} onClick={openSave}>
            Save{dirty ? ' •' : ''}
          </button>
          <button disabled={busy} onClick={openLoad}>
            Load
          </button>
          <span style={{ fontSize: 12, color: dirty ? 'var(--mm-warn)' : 'var(--mm-muted)' }}>
            {projectPath ?? '(unsaved)'}
            {dirty && (
              <span title="Edits not yet written to the project file (always snapshotted for crash recovery; autosaves shortly if the project has a save location)">
                {' '}
                · unsaved changes{autosave.state === 'saving' ? ' — autosaving…' : ''}
              </span>
            )}
            {!dirty && autosave.state === 'saved' && (
              <span title="Automatically written to the project file after your last edit" style={{ color: 'var(--mm-muted)' }}>
                {' '}
                · autosaved {new Date(autosave.at).toLocaleTimeString()}
              </span>
            )}
            {autosave.state === 'error' && (
              <span title={autosave.message} style={{ color: 'var(--mm-error)' }}>
                {' '}
                · autosave failed
              </span>
            )}
          </span>
        </>
      )}

      {/* VS Code's own Source Control covers git there. */}
      {host.kind !== 'vscode' && has('git') && (
      <div style={{ position: 'relative' }}>
        <button
          onClick={() => setShowGit((s) => !s)}
          disabled={!projectPath}
          title={projectPath ? 'Version this project with git' : 'Save the project first to use git'}
        >
          Git
        </button>
        {showGit && <GitPanel projectPath={projectPath} onClose={() => setShowGit(false)} />}
      </div>
      )}

      {/* Restricted mode: LLM settings belong to whoever runs this workspace. */}
      {has('llm_settings') && (
        <div style={{ position: 'relative' }}>
          <button onClick={() => setShowSettings((s) => !s)}>Settings</button>
          {showSettings && (
            <SettingsPanel settings={llmSettings} onChange={onLlmSettingsChange} onClose={() => setShowSettings(false)} />
          )}
        </div>
      )}

      {compiled !== null && (
        <div
          style={{
            position: 'fixed',
            inset: 40,
            background: 'var(--mm-bg)',
            border: '1px solid var(--mm-border-strong)',
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
          <div style={{ flex: 1, minHeight: 0 }}>
            <CodeEditor value={compiled} readOnly fill />
          </div>
        </div>
      )}

      {showSave && (
        <FileBrowser
          mode="save"
          folders
          startPath={browseStartPath}
          defaultName={openName}
          onPick={doSavePick}
          onClose={() => setShowSave(false)}
        />
      )}
      {showLoad && (
        <FileBrowser folders startPath={browseStartPath} onPick={doLoadPick} onClose={() => setShowLoad(false)} />
      )}
    </div>
  )
}

type MenuItem = { label: string; onClick: () => void; disabled?: boolean }

function MoreMenu({ items, onClose }: { items: MenuItem[]; onClose: () => void }) {
  return (
    <>
      {/* Click anywhere else to close. */}
      <div style={{ position: 'fixed', inset: 0, zIndex: 40 }} onClick={onClose} />
      <div
        role="menu"
        style={{
          position: 'absolute',
          top: '100%',
          left: 0,
          marginTop: 4,
          zIndex: 41,
          display: 'flex',
          flexDirection: 'column',
          gap: 2,
          minWidth: 180,
          padding: 4,
          background: 'var(--mm-bg)',
          border: '1px solid var(--mm-border-strong)',
          borderRadius: 6,
          boxShadow: '0 6px 18px rgba(0,0,0,0.2)',
        }}
      >
        {items.map((item) => (
          <button
            key={item.label}
            role="menuitem"
            disabled={item.disabled}
            onClick={() => {
              onClose()
              item.onClick()
            }}
            style={{ textAlign: 'left', border: 'none', background: 'transparent' }}
          >
            {item.label}
          </button>
        ))}
      </div>
    </>
  )
}
