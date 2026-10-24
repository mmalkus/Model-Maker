import { Fragment, useCallback, useEffect, useMemo, useState, type CSSProperties, type ReactNode } from 'react'
import { api } from './api'
import { CodeEditor } from './CodeEditor'
import type {
  AIDraftOut,
  BlockTag,
  ExampleResult,
  GraphOut,
  ImportPreview,
  LibraryBlock,
  LibraryCheck,
  LibraryFile,
  LibraryOut,
  TryOut,
} from './types'

// Where Create with AI starts from: a fresh block, an existing library
// block (Edit with AI), or a red placed block (Fix with AI).
export type AIStart = { file?: string; block?: string; fixBlock?: string; contextBlock?: string; addToCanvas?: boolean }

export type LibraryView =
  | { kind: 'browse'; selected?: { file: string | null; id: string } }
  | { kind: 'edit'; file: string | null }
  | { kind: 'ai'; start: AIStart }
  | { kind: 'import' }

const mono: CSSProperties = { fontFamily: 'ui-monospace, SFMono-Regular, Menlo, Consolas, monospace' }
const muted: CSSProperties = { color: '#6b7280' }
const sectionTitle: CSSProperties = { fontSize: 11, color: '#6b7280', textTransform: 'uppercase', margin: '12px 0 4px' }
const box: CSSProperties = { border: '1px solid #e5e7eb', borderRadius: 8, padding: 10, background: '#fff' }

function Badge({ children, tone = 'neutral', title }: { children: ReactNode; tone?: 'neutral' | 'brand' | 'warn' | 'error' | 'ai'; title?: string }) {
  const tones: Record<string, CSSProperties> = {
    neutral: { background: '#f3f4f6', color: '#374151' },
    brand: { background: 'var(--brand-light)', color: 'var(--brand)', border: '1px solid var(--brand-border)' },
    warn: { background: '#fffbeb', color: '#92400e' },
    error: { background: '#fef2f2', color: '#b91c1c' },
    ai: { background: '#ede9fe', color: '#5b21b6' },
  }
  return (
    <span title={title} style={{ fontSize: 10, fontWeight: 600, borderRadius: 4, padding: '1px 5px', whiteSpace: 'nowrap', ...tones[tone] }}>
      {children}
    </span>
  )
}

function useDebounced<T>(value: T, ms: number): T {
  const [v, setV] = useState(value)
  useEffect(() => {
    const id = setTimeout(() => setV(value), ms)
    return () => clearTimeout(id)
  }, [value, ms])
  return v
}

function download(url: string) {
  const a = document.createElement('a')
  a.href = url
  a.download = ''
  document.body.appendChild(a)
  a.click()
  a.remove()
}

function fileToBase64(file: File): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader()
    reader.onload = () => resolve(String(reader.result).split(',', 2)[1] ?? '')
    reader.onerror = () => reject(reader.error)
    reader.readAsDataURL(file)
  })
}

/** The block library: browse files and blocks (built-ins read-only), edit a
 * file, create or fix a block with AI, import and export. A full-screen
 * overlay over the canvas. */
export function LibraryPanel({
  initialView,
  graph,
  provider,
  onClose,
  onLibraryChanged,
  onPlace,
  onPlaceBuiltin,
  onOpenUpdate,
}: {
  initialView: LibraryView
  graph: GraphOut | null
  provider: string | null
  onClose: () => void
  onLibraryChanged: () => void
  onPlace: (file: string, id: string) => Promise<void>
  onPlaceBuiltin: (category: string) => void
  // Open the update dialog for a placed block (after Fix with AI changed
  // its structure).
  onOpenUpdate: (blockId: string) => void
}) {
  const [view, setView] = useState<LibraryView>(initialView)
  const [library, setLibrary] = useState<LibraryOut | null>(null)
  const [error, setError] = useState<string | null>(null)

  const load = useCallback(
    (refresh = false) =>
      api
        .library(refresh)
        .then(setLibrary)
        .catch((e) => setError((e as Error).message)),
    [],
  )
  useEffect(() => {
    load()
  }, [load])

  const changed = useCallback(() => {
    load(true)
    onLibraryChanged()
  }, [load, onLibraryChanged])

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape' && view.kind === 'browse') onClose()
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [view.kind, onClose])

  const title =
    view.kind === 'browse'
      ? 'Block library'
      : view.kind === 'edit'
        ? view.file ?? 'New block file'
        : view.kind === 'ai'
          ? view.start.fixBlock
            ? 'Fix with AI'
            : view.start.block
              ? `Edit ${view.start.block} with AI`
              : 'Create with AI'
          : 'Import blocks'

  return (
    <div style={{ position: 'fixed', inset: 0, background: 'rgba(11,35,64,0.35)', zIndex: 50, display: 'flex', padding: 24 }}>
      <div
        role="dialog"
        aria-label={title}
        style={{ flex: 1, background: '#fcfbf8', borderRadius: 12, display: 'flex', flexDirection: 'column', overflow: 'hidden', boxShadow: '0 20px 50px rgba(11,35,64,0.3)' }}
      >
        <div style={{ display: 'flex', alignItems: 'center', gap: 8, padding: '12px 16px', borderBottom: '1px solid #e5e7eb', background: '#fff' }}>
          {view.kind !== 'browse' && (
            <button onClick={() => setView({ kind: 'browse' })} style={{ fontSize: 12 }}>
              ← Library
            </button>
          )}
          <h2 style={{ margin: 0, fontFamily: 'var(--font-heading)', fontSize: 19, color: 'var(--brand-ink)' }}>{title}</h2>
          {view.kind === 'browse' && library && (
            <span style={{ ...muted, ...mono, fontSize: 11 }} title="Set MODELMAKER_BLOCK_LIBRARY to use another folder">
              {library.dir}
            </span>
          )}
          <span style={{ flex: 1 }} />
          {view.kind === 'browse' && (
            <>
              <button className="brand-primary" onClick={() => setView({ kind: 'ai', start: { addToCanvas: false } })}>
                ✦ Create with AI
              </button>
              <button onClick={() => setView({ kind: 'edit', file: null })}>New block file</button>
              <button onClick={() => setView({ kind: 'import' })}>Import…</button>
              <button onClick={() => download(api.libraryExportUrl({}))} disabled={!library?.files.some((f) => !f.is_draft)}>
                Export all
              </button>
              <button onClick={() => load(true)} title="Re-read the library folder (after a git pull, or a colleague's save)">
                Reload
              </button>
            </>
          )}
          <button aria-label="Close" onClick={onClose}>
            ✕
          </button>
        </div>
        {error && (
          <div style={{ background: '#fef2f2', color: '#b91c1c', padding: '6px 16px', fontSize: 12 }}>
            {error}{' '}
            <button style={{ fontSize: 11 }} onClick={() => setError(null)}>
              dismiss
            </button>
          </div>
        )}
        <div style={{ flex: 1, minHeight: 0, display: 'flex' }}>
          {view.kind === 'browse' && library && (
            <Browser
              library={library}
              graph={graph}
              selected={view.selected}
              onSelect={(sel) => setView({ kind: 'browse', selected: sel })}
              setView={setView}
              onChanged={changed}
              onError={setError}
              onPlace={async (file, id) => {
                await onPlace(file, id)
                onClose()
              }}
              onPlaceBuiltin={(c) => {
                onPlaceBuiltin(c)
                onClose()
              }}
            />
          )}
          {view.kind === 'edit' && (
            <FileEditor
              key={view.file ?? '(new)'}
              file={view.file}
              library={library}
              onSaved={(name) => {
                changed()
                setView({ kind: 'edit', file: name })
              }}
              onError={setError}
            />
          )}
          {view.kind === 'ai' && (
            <CreateWithAI
              start={view.start}
              library={library}
              graph={graph}
              provider={provider}
              onError={setError}
              onDone={async (file, blockId, addToCanvas) => {
                changed()
                if (view.start.fixBlock) {
                  const preview = await api.updatePreview(view.start.fixBlock)
                  const structural =
                    preview.wires_disconnected.length > 0 || preview.params_dropped.length > 0 || preview.params_newly_required.length > 0
                  if (structural) onOpenUpdate(view.start.fixBlock)
                  else await api.applyUpdate(view.start.fixBlock)
                  onClose()
                } else if (addToCanvas) {
                  await onPlace(file, blockId)
                  onClose()
                } else {
                  setView({ kind: 'browse', selected: { file, id: blockId } })
                }
              }}
            />
          )}
          {view.kind === 'import' && (
            <ImportView
              onDone={() => {
                changed()
                setView({ kind: 'browse' })
              }}
              onError={setError}
            />
          )}
        </div>
      </div>
    </div>
  )
}

// ---- browse ------------------------------------------------------------------

function Browser({
  library,
  graph,
  selected,
  onSelect,
  setView,
  onChanged,
  onError,
  onPlace,
  onPlaceBuiltin,
}: {
  library: LibraryOut
  graph: GraphOut | null
  selected?: { file: string | null; id: string }
  onSelect: (sel: { file: string | null; id: string }) => void
  setView: (v: LibraryView) => void
  onChanged: () => void
  onError: (msg: string) => void
  onPlace: (file: string, id: string) => Promise<void>
  onPlaceBuiltin: (category: string) => void
}) {
  const [search, setSearch] = useState('')
  const [tag, setTag] = useState('')
  const [tags, setTags] = useState<BlockTag[]>([])
  const [showBuiltins, setShowBuiltins] = useState(false)
  useEffect(() => {
    api.registryTags().then(setTags).catch(() => setTags([]))
  }, [])

  const matches = (b: LibraryBlock) => {
    if (tag && !b.tags.includes(tag)) return false
    const q = search.trim().toLowerCase()
    return !q || b.id.includes(q) || b.display_name.toLowerCase().includes(q) || b.summary.toLowerCase().includes(q)
  }
  const userFiles = library.files.filter((f) => !f.is_draft)
  const drafts = library.files.filter((f) => f.is_draft)

  let detail: { block: LibraryBlock; file: LibraryFile | null } | null = null
  if (selected) {
    if (selected.file === null) {
      for (const bf of library.builtin_files) {
        const b = bf.blocks.find((x) => x.id === selected.id)
        if (b) detail = { block: b, file: null }
      }
    } else {
      const f = library.files.find((x) => x.name === selected.file)
      const b = f?.blocks.find((x) => x.id === selected.id)
      if (f && b) detail = { block: b, file: f }
    }
  }

  const row = (b: LibraryBlock, file: string | null, f?: LibraryFile) => {
    const active = selected?.id === b.id && selected?.file === file
    const problems = [...(f?.file_errors ?? []), ...b.errors]
    return (
      <button
        key={`${file}:${b.id}`}
        onClick={() => onSelect({ file, id: b.id })}
        style={{
          display: 'flex',
          alignItems: 'center',
          gap: 6,
          width: '100%',
          textAlign: 'left',
          fontSize: 12,
          padding: '5px 8px',
          marginBottom: 2,
          border: 'none',
          borderRadius: 6,
          background: active ? 'var(--brand-light)' : 'transparent',
          opacity: problems.length || b.unmet_requirements.length ? 0.75 : 1,
        }}
      >
        <span style={{ flex: 1, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{b.display_name}</span>
        {problems.length > 0 && <Badge tone="error">error</Badge>}
        {b.unmet_requirements.length > 0 && <Badge tone="warn">needs pkg</Badge>}
        {(library.usage[b.id]?.length ?? 0) > 0 && <Badge title="Placed in this project">{library.usage[b.id].length}×</Badge>}
        <span style={{ ...muted, fontSize: 10 }}>v{b.version}</span>
      </button>
    )
  }

  return (
    <>
      <div style={{ width: 320, borderRight: '1px solid #e5e7eb', overflowY: 'auto', padding: 12, background: '#fff' }}>
        <input
          value={search}
          onChange={(e) => setSearch(e.target.value)}
          placeholder="Search blocks…"
          aria-label="Search blocks"
          style={{ width: '100%', fontSize: 12, padding: '6px 8px', border: '1px solid #d1d5db', borderRadius: 6, marginBottom: 6 }}
        />
        <select value={tag} onChange={(e) => setTag(e.target.value)} aria-label="Filter by tag" style={{ width: '100%', fontSize: 12, marginBottom: 6 }}>
          <option value="">All tags</option>
          {tags.map((t) => (
            <option key={t.tag} value={t.tag}>
              {t.tag.replace(/_/g, ' ')}
            </option>
          ))}
        </select>

        <div style={sectionTitle}>Your library</div>
        {userFiles.length === 0 && <div style={{ ...muted, fontSize: 12 }}>Empty -- create a block with AI, start a new file, or import one.</div>}
        {userFiles.map((f) => {
          const blocks = f.blocks.filter(matches)
          if (!blocks.length && (search || tag)) return null
          return (
            <div key={f.name} style={{ marginBottom: 8 }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
                <span style={{ ...mono, fontSize: 11, color: 'var(--brand-ink)', flex: 1 }} title={f.doc}>
                  {f.name}
                </span>
                <button style={{ fontSize: 10, padding: '1px 6px' }} onClick={() => setView({ kind: 'edit', file: f.name })}>
                  Edit file
                </button>
                <button style={{ fontSize: 10, padding: '1px 6px' }} onClick={() => setView({ kind: 'ai', start: { file: f.name } })} title="Create a new block in this file with AI">
                  + ✦
                </button>
              </div>
              {f.doc && <div style={{ ...muted, fontSize: 11, margin: '1px 0 3px' }}>{f.doc.split('\n')[0]}</div>}
              {f.file_errors.map((e) => (
                <div key={e} style={{ color: '#b91c1c', fontSize: 11 }}>
                  {e}
                </div>
              ))}
              {blocks.map((b) => row(b, f.name, f))}
            </div>
          )
        })}

        {drafts.length > 0 && (
          <>
            <div style={{ ...sectionTitle, color: '#6d28d9' }}>AI drafts</div>
            <div style={{ ...muted, fontSize: 11, marginBottom: 4 }}>Blocks AI builds created. Keep the ones you want; discarding never breaks a model.</div>
            {drafts.map((f) => (
              <div key={f.name} style={{ marginBottom: 8 }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
                  <span style={{ ...mono, fontSize: 11, color: '#6d28d9', flex: 1 }}>{f.name.slice(7)}</span>
                  <button
                    style={{ fontSize: 10, padding: '1px 6px', color: '#b91c1c' }}
                    onClick={() => {
                      if (!confirm(`Discard ${f.name}? Placed blocks keep working: they embed their code.`)) return
                      api.libraryDiscardDraft(f.name).then(onChanged).catch((e) => onError(e.message))
                    }}
                  >
                    Discard
                  </button>
                </div>
                {f.blocks.filter(matches).map((b) => row(b, f.name, f))}
              </div>
            ))}
          </>
        )}

        <button
          onClick={() => setShowBuiltins((v) => !v)}
          style={{ ...sectionTitle, background: 'none', border: 'none', padding: 0, cursor: 'pointer', display: 'block' }}
        >
          {showBuiltins || search || tag ? '▾' : '▸'} Built-in (read-only)
        </button>
        {(showBuiltins || search || tag) &&
          library.builtin_files.map((bf) => {
            const blocks = bf.blocks.filter(matches)
            if (!blocks.length) return null
            return (
              <div key={bf.name} style={{ marginBottom: 8 }}>
                <div style={{ ...mono, fontSize: 11, ...muted }}>{bf.name.replace('modelmaker/blocks/', '')}</div>
                {blocks.map((b) => row(b, null))}
              </div>
            )
          })}
      </div>
      <div style={{ flex: 1, overflowY: 'auto', padding: 16 }}>
        {detail ? (
          <BlockDetail
            key={`${detail.file?.name}:${detail.block.id}`}
            block={detail.block}
            file={detail.file}
            usage={library.usage[detail.block.id] ?? []}
            graph={graph}
            userFiles={userFiles}
            setView={setView}
            onChanged={onChanged}
            onError={onError}
            onPlace={onPlace}
            onPlaceBuiltin={onPlaceBuiltin}
          />
        ) : (
          <div style={{ ...muted, fontSize: 13, maxWidth: 520 }}>
            <p>Pick a block to see what it does, its ports and params, and what the app does around it (change detection, streaming, hints).</p>
            <p>
              Blocks are plain Python files in your library folder -- one or more blocks per file. Placed blocks keep their own copy, so
              editing the library never changes a model until you choose <em>Update</em> on the block.
            </p>
          </div>
        )}
      </div>
    </>
  )
}

function BlockDetail({
  block,
  file,
  usage,
  graph,
  userFiles,
  setView,
  onChanged,
  onError,
  onPlace,
  onPlaceBuiltin,
}: {
  block: LibraryBlock
  file: LibraryFile | null
  usage: string[]
  graph: GraphOut | null
  userFiles: LibraryFile[]
  setView: (v: LibraryView) => void
  onChanged: () => void
  onError: (msg: string) => void
  onPlace: (file: string, id: string) => Promise<void>
  onPlaceBuiltin: (category: string) => void
}) {
  const [tryResult, setTryResult] = useState<TryOut | null>(null)
  const [trying, setTrying] = useState(false)
  const problems = [...(file?.file_errors ?? []), ...block.errors]
  const placeable = problems.length === 0 && block.unmet_requirements.length === 0
  const arch = block.architecture

  const duplicate = () => {
    const newId = prompt('Id for the copy (a Python function name):', `${block.id}_copy`)
    if (!newId) return
    const to = prompt('Library file to put it in:', userFiles[0]?.name ?? 'my_blocks.py')
    if (!to) return
    api
      .libraryDuplicate({ id: block.id, file: file?.name ?? null, new_id: newId, to })
      .then(() => {
        onChanged()
        setView({ kind: 'browse', selected: { file: to, id: newId } })
      })
      .catch((e) => onError(e.message))
  }

  const tryIt = () => {
    if (!file) return
    setTrying(true)
    api
      .libraryTry(file.name, { blocks: [block.id] })
      .then(setTryResult)
      .catch((e) => onError(e.message))
      .finally(() => setTrying(false))
  }

  return (
    <div style={{ maxWidth: 760 }}>
      <div style={{ display: 'flex', alignItems: 'baseline', gap: 8, flexWrap: 'wrap' }}>
        <h3 style={{ margin: 0, fontFamily: 'var(--font-heading)', fontSize: 20, color: 'var(--brand-ink)' }}>{block.display_name}</h3>
        <span style={{ ...mono, ...muted, fontSize: 12 }}>
          {block.id} · v{block.version} · {file ? file.name : (block.origin ?? 'built-in')}
        </span>
        {!file && <Badge>built-in</Badge>}
        {file?.is_draft && <Badge tone="ai">AI draft</Badge>}
      </div>
      {problems.length > 0 && (
        <div style={{ background: '#fef2f2', color: '#b91c1c', padding: 8, borderRadius: 6, marginTop: 8, fontSize: 12 }}>
          {problems.map((p) => (
            <div key={p}>{p}</div>
          ))}
        </div>
      )}
      {block.unmet_requirements.length > 0 && (
        <div style={{ background: '#fffbeb', color: '#92400e', padding: 8, borderRadius: 6, marginTop: 8, fontSize: 12 }}>
          Needs {block.unmet_requirements.join(', ')} -- install it (pip) before placing this block.
        </div>
      )}

      <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap', margin: '12px 0' }}>
        {file ? (
          <button className="brand-primary" disabled={!placeable} onClick={() => onPlace(file.name, block.id).catch((e) => onError(e.message))}>
            Add to canvas
          </button>
        ) : (
          <button className="brand-primary" onClick={() => onPlaceBuiltin(block.id)}>
            Add to canvas
          </button>
        )}
        {file && (
          <>
            <button onClick={() => setView({ kind: 'edit', file: file.name })}>Edit</button>
            <button onClick={() => setView({ kind: 'ai', start: { file: file.name, block: block.id } })}>✦ Edit with AI</button>
            <button disabled={trying} onClick={tryIt} title={file.examples.includes(block.id) ? 'Run its _example_ self-check' : 'This block has no _example_ yet'}>
              {trying ? 'Trying…' : 'Try it'}
            </button>
          </>
        )}
        <button onClick={duplicate} title={file ? 'Copy it under a new id' : 'Copy this built-in into your library to change it'}>
          {file ? 'Duplicate' : 'Duplicate to library'}
        </button>
        {file && (
          <>
            <button onClick={() => download(api.libraryExportUrl({ file: file.name, blocks: [block.id] }))}>Export block</button>
            <button onClick={() => download(api.libraryExportUrl({ file: file.name }))}>Export file</button>
            {file.is_draft && (
              <button
                onClick={() => {
                  const to = prompt('Keep it in which library file?', userFiles[0]?.name ?? 'my_blocks.py')
                  if (to)
                    api
                      .libraryKeepDraft(file.name, block.id, to)
                      .then(() => {
                        onChanged()
                        setView({ kind: 'browse', selected: { file: to, id: block.id } })
                      })
                      .catch((e) => onError(e.message))
                }}
              >
                Keep
              </button>
            )}
            <button
              style={{ color: '#b91c1c', borderColor: '#fecaca', marginLeft: 'auto' }}
              onClick={() => {
                if (!confirm(`Delete ${block.id} from ${file.name}? Placed copies keep working.`)) return
                api
                  .libraryDeleteBlock(file.name, block.id)
                  .then(() => {
                    onChanged()
                    setView({ kind: 'browse' })
                  })
                  .catch((e) => onError(e.message))
              }}
            >
              Delete
            </button>
          </>
        )}
      </div>

      {tryResult && 'examples' in tryResult && <ExampleList results={tryResult.examples} only={block.id} />}

      {block.description && <div style={{ whiteSpace: 'pre-wrap', fontSize: 13, lineHeight: 1.5, marginBottom: 12 }}>{block.description}</div>}

      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(220px, 1fr))', gap: 10 }}>
        <div style={box}>
          <div style={{ fontWeight: 600, marginBottom: 4 }}>Ports</div>
          {block.inputs.map((p) => (
            <div key={`i${p.name}`} style={{ fontSize: 12 }}>
              ● {p.name} <span style={muted}>: {p.type}</span> {p.lazy && <Badge>lazy</Badge>}
            </div>
          ))}
          {block.outputs.map((p) => (
            <div key={`o${p.name}`} style={{ fontSize: 12 }}>
              {p.name} ● <span style={muted}>: {p.type}</span> {p.aggregate && <Badge>statistics table</Badge>} {p.lazy && <Badge>lazy</Badge>}
            </div>
          ))}
          {!block.inputs.length && !block.outputs.length && <div style={muted}>none</div>}
        </div>
        <div style={box}>
          <div style={{ fontWeight: 600, marginBottom: 4 }}>Params</div>
          {block.form?.fields.map((f) => (
            <div key={f.key} style={{ fontSize: 12 }}>
              {f.label} <span style={muted}>· {f.kind}</span>
              {f.default != null && <span style={muted}> = {JSON.stringify(f.default)}</span>}
            </div>
          ))}
          {block.form?.other_params.map((k) => (
            <div key={k} style={{ fontSize: 12 }}>
              {k} <span style={muted}>· JSON</span>
            </div>
          ))}
          {!block.form?.fields.length && !block.form?.other_params.length && <div style={muted}>none</div>}
        </div>
        <div style={box}>
          <div style={{ fontWeight: 600, marginBottom: 4 }}>What the app does around it</div>
          <ul style={{ margin: 0, paddingLeft: 16, fontSize: 12 }}>
            {arch.sampling && <li>Sample mode caps its output rows</li>}
            {arch.change_detection.map((c) => (
              <li key={c}>Change detection: {c}</li>
            ))}
            {block.block_type === 'input' && arch.change_detection.length === 0 && <li style={muted}>No change detection (no path or query field)</li>}
            <li>{arch.streaming ? `Streaming: ${arch.streaming}` : <span style={muted}>Runs as a checkpoint in a streaming run</span>}</li>
            {arch.hints ? <li>AI build hints: rule “{arch.hints}”</li> : <li style={muted}>No AI build hints</li>}
            {arch.statistics_tables.length > 0 && <li>Statistics tables: {arch.statistics_tables.join(', ')}</li>}
          </ul>
        </div>
        <div style={box}>
          <div style={{ fontWeight: 600, marginBottom: 4 }}>Details</div>
          <div style={{ fontSize: 12 }}>Type: {block.block_type}</div>
          {block.tags.length > 0 && <div style={{ fontSize: 12 }}>Tags: {block.tags.join(', ')}</div>}
          {block.requires.length > 0 && <div style={{ fontSize: 12 }}>Requires: {block.requires.join(', ')}</div>}
          {file && <div style={{ fontSize: 12 }}>Self-check: {file.examples.includes(block.id) ? 'yes' : <span style={{ color: '#92400e' }}>none</span>}</div>}
          <div style={{ fontSize: 12 }}>
            Used in this project:{' '}
            {usage.length ? usage.map((id) => graph?.blocks[id]?.name ?? id).join(', ') : <span style={muted}>not placed</span>}
          </div>
          <div style={{ ...mono, ...muted, fontSize: 10, marginTop: 4 }}>sha {block.sha}</div>
        </div>
      </div>
    </div>
  )
}

function ExampleList({ results, only }: { results: Record<string, ExampleResult>; only?: string }) {
  const entries = Object.entries(results).filter(([id]) => !only || id === only || id === '__file__')
  return (
    <div style={{ ...box, marginBottom: 12, fontSize: 12 }}>
      <div style={{ fontWeight: 600, marginBottom: 4 }}>Try it</div>
      {entries.map(([id, r]) => (
        <div key={id} style={{ color: r.ok ? '#3f6259' : r.ok === null ? '#6b7280' : '#b91c1c' }}>
          {r.ok ? '✓' : r.ok === null ? '–' : '✕'} {id === '__file__' ? 'file' : id}: {r.ok ? 'example passed' : (r.error ?? '')}
          {r.line ? ` (line ${r.line})` : ''}
        </div>
      ))}
    </div>
  )
}

// ---- editor ----------------------------------------------------------------------

function Checklist({ check }: { check: LibraryCheck | null }) {
  if (!check) return <div style={muted}>Checking…</div>
  return (
    <ul style={{ margin: 0, padding: 0, listStyle: 'none', fontSize: 12, display: 'flex', flexDirection: 'column', gap: 4 }}>
      {check.file_errors.map((e) => (
        <li key={e} style={{ color: '#b91c1c' }}>
          ✕ {e}
        </li>
      ))}
      {check.file_errors.length === 0 && <li style={{ color: '#3f6259' }}>✓ BLOCKS is a plain literal; imports are fine</li>}
      {check.blocks.map((b) => (
        <li key={b.id}>
          <div style={{ color: b.errors.length ? '#b91c1c' : '#3f6259' }}>
            {b.errors.length ? '✕' : '✓'} {b.id} <span style={muted}>v{b.version}</span>
            {!check.examples.includes(b.id) && <span style={{ color: '#92400e' }}> · no self-check</span>}
          </div>
          {b.errors.map((e) => (
            <div key={e} style={{ color: '#b91c1c', paddingLeft: 14 }}>
              {e}
            </div>
          ))}
        </li>
      ))}
      {check.blocks.length === 0 && check.file_errors.length === 0 && <li style={muted}>No blocks yet</li>}
    </ul>
  )
}

function FileEditor({
  file,
  library,
  onSaved,
  onError,
}: {
  file: string | null
  library: LibraryOut | null
  onSaved: (name: string) => void
  onError: (msg: string) => void
}) {
  const [name, setName] = useState(file ?? 'my_blocks.py')
  const [source, setSource] = useState('')
  const [sha, setSha] = useState<string | null>(null)
  const [loaded, setLoaded] = useState(false)
  const [check, setCheck] = useState<LibraryCheck | null>(null)
  const [tryResult, setTryResult] = useState<TryOut | null>(null)
  const [saving, setSaving] = useState(false)
  const [conflict, setConflict] = useState(false)
  const [original, setOriginal] = useState('')

  const loadFile = useCallback(() => {
    const p = file ? api.libraryFile(file).then((f) => ({ source: f.source ?? '', sha: f.sha })) : api.libraryTemplate().then((t) => ({ source: t.source, sha: null }))
    p.then(({ source, sha }) => {
      setSource(source)
      setOriginal(source)
      setSha(sha)
      setLoaded(true)
      setConflict(false)
    }).catch((e) => onError(e.message))
  }, [file, onError])
  useEffect(loadFile, [loadFile])

  const debounced = useDebounced(source, 500)
  useEffect(() => {
    if (!loaded) return
    api
      .libraryCheck(debounced, name)
      .then(setCheck)
      .catch(() => setCheck(null))
  }, [debounced, name, loaded])

  const save = () => {
    setSaving(true)
    api
      .librarySave(name, source, { expected_sha: sha, create: file === null })
      .then((f) => {
        setSha(f.sha)
        setOriginal(source)
        onSaved(f.name)
      })
      .catch((e: Error) => {
        if (e.message.includes('changed on disk')) setConflict(true)
        onError(e.message)
      })
      .finally(() => setSaving(false))
  }

  const dirty = source !== original
  const existingNames = new Set(library?.files.map((f) => f.name) ?? [])

  return (
    <>
      <div style={{ flex: 1, minWidth: 0, display: 'flex', flexDirection: 'column', padding: 12, gap: 8 }}>
        <div style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
          {file === null ? (
            <>
              <label style={{ fontSize: 12 }}>
                File name{' '}
                <input value={name} onChange={(e) => setName(e.target.value)} style={{ ...mono, fontSize: 12, width: 220 }} />
              </label>
              {existingNames.has(name) && <span style={{ color: '#b91c1c', fontSize: 12 }}>already exists</span>}
            </>
          ) : (
            <span style={{ ...mono, fontSize: 12 }}>{file}</span>
          )}
          {dirty && <Badge tone="warn">unsaved</Badge>}
          <span style={{ flex: 1 }} />
          {conflict && (
            <button onClick={loadFile} title="Someone saved this file since you opened it. Reload it (your edits are dropped) and re-apply them.">
              Reload from disk
            </button>
          )}
          <button className="brand-primary" disabled={saving || (!dirty && file !== null)} onClick={save}>
            {saving ? 'Saving…' : 'Save to library'}
          </button>
        </div>
        <div style={{ flex: 1, minHeight: 0 }}>
          <CodeEditor value={source} onChange={setSource} onSave={save} fill />
        </div>
      </div>
      <div style={{ width: 320, borderLeft: '1px solid #e5e7eb', padding: 12, overflowY: 'auto', background: '#fff' }}>
        <div style={sectionTitle}>Live check</div>
        <Checklist check={check} />
        <div style={sectionTitle}>Try it</div>
        <button
          onClick={() =>
            api
              .libraryTry(name, { source })
              .then(setTryResult)
              .catch((e) => onError(e.message))
          }
        >
          Run self-checks
        </button>
        {tryResult && 'examples' in tryResult && (
          <div style={{ marginTop: 8 }}>
            <ExampleList results={tryResult.examples} />
          </div>
        )}
        <div style={{ ...muted, fontSize: 11, marginTop: 12 }}>
          Ctrl/Cmd-S saves. Saving never changes placed blocks: they show <em>Update to vN</em> once you bump a block's version.
        </div>
      </div>
    </>
  )
}

// ---- Create with AI ------------------------------------------------------------------

function TryOutView({ result }: { result: TryOut }) {
  if ('examples' in result) return <ExampleList results={result.examples} />
  if (!result.ok) return <div style={{ ...box, color: '#b91c1c', fontSize: 12 }}>✕ {result.error}</div>
  return (
    <div style={{ ...box, fontSize: 12 }}>
      {Object.entries(result.outputs).map(([port, v]) => (
        <div key={port} style={{ marginBottom: 6 }}>
          <div style={{ fontWeight: 600 }}>
            {port}{' '}
            <span style={muted}>
              {v.type === 'dataframe' ? `${v.rows} rows × ${v.columns?.length} columns` : v.type}
            </span>
          </div>
          {v.type === 'dataframe' && v.head && v.columns && (
            <div style={{ overflowX: 'auto' }}>
              <table style={{ borderCollapse: 'collapse', fontSize: 11 }}>
                <thead>
                  <tr>
                    {v.columns.map((c) => (
                      <th key={c.name} style={{ textAlign: 'left', padding: '2px 6px', borderBottom: '1px solid #e5e7eb' }}>
                        {c.name}
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {v.head.slice(0, 8).map((r, i) => (
                    <tr key={i}>
                      {v.columns!.map((c) => (
                        <td key={c.name} style={{ padding: '2px 6px' }}>
                          {String(r[c.name])}
                        </td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
          {v.type === 'value' && <pre style={{ margin: 0, fontSize: 11 }}>{JSON.stringify(v.value, null, 2)}</pre>}
        </div>
      ))}
    </div>
  )
}

function CreateWithAI({
  start,
  library,
  graph,
  provider,
  onError,
  onDone,
}: {
  start: AIStart
  library: LibraryOut | null
  graph: GraphOut | null
  provider: string | null
  onError: (msg: string) => void
  onDone: (file: string, blockId: string, addToCanvas: boolean) => Promise<void>
}) {
  const [instruction, setInstruction] = useState('')
  const [contextBlock, setContextBlock] = useState(start.contextBlock ?? '')
  const [target, setTarget] = useState(start.file ?? '')
  const [addToCanvas, setAddToCanvas] = useState(start.addToCanvas ?? true)
  const [draft, setDraft] = useState<AIDraftOut | null>(null)
  const [source, setSource] = useState('')
  const [busy, setBusy] = useState(false)
  const [elapsed, setElapsed] = useState(0)
  const [refine, setRefine] = useState('')
  const [tryResult, setTryResult] = useState<TryOut | null>(null)
  const [check, setCheck] = useState<LibraryCheck | null>(null)
  const [showDiff, setShowDiff] = useState(false)

  useEffect(() => {
    if (!busy) return
    setElapsed(0)
    const id = setInterval(() => setElapsed((s) => s + 1), 1000)
    return () => clearInterval(id)
  }, [busy])

  const fixBlock = start.fixBlock ? graph?.blocks[start.fixBlock] : undefined
  const contextChoices = useMemo(
    () =>
      Object.values(graph?.blocks ?? {}).filter(
        (b) => (b.status === 'green' || b.status === 'orange') && b.outputs.some((p) => p.type === 'dataframe'),
      ),
    [graph],
  )
  const userFiles = library?.files.filter((f) => !f.is_draft) ?? []

  const debounced = useDebounced(source, 500)
  useEffect(() => {
    if (!draft) return
    api
      .libraryCheck(debounced, draft.file)
      .then(setCheck)
      .catch(() => setCheck(null))
  }, [debounced, draft])

  const ask = (refineText?: string) => {
    setBusy(true)
    setTryResult(null)
    api
      .libraryAIDraft({
        instruction: refineText ?? instruction,
        provider: provider ?? undefined,
        file: draft?.file ?? (target || undefined),
        block: start.block,
        context_block: contextBlock || undefined,
        fix_block: draft ? undefined : start.fixBlock,
        previous_code: refineText && draft ? draft.code : undefined,
        previous_block: refineText && draft ? draft.entry : undefined,
      })
      .then((d) => {
        setDraft(d)
        setSource(d.source)
        setRefine('')
      })
      .catch((e) => onError(e.message))
      .finally(() => setBusy(false))
  }

  const save = () => {
    if (!draft) return
    setBusy(true)
    api
      .librarySave(draft.file, source, { expected_sha: draft.base_sha, create: draft.base_sha === null })
      .then(() => onDone(draft.file, draft.block, addToCanvas))
      .catch((e) => onError(e.message))
      .finally(() => setBusy(false))
  }

  const tryExamples = () =>
    draft &&
    api
      .libraryTry(draft.file, { source, blocks: [draft.block] })
      .then(setTryResult)
      .catch((e) => onError(e.message))
  const tryOnContext = () =>
    draft &&
    api
      .libraryTry(draft.file, { source, block: draft.block, input_block: contextBlock })
      .then(setTryResult)
      .catch((e) => onError(e.message))

  return (
    <>
      <div style={{ width: 340, borderRight: '1px solid #e5e7eb', padding: 12, overflowY: 'auto', background: '#fff', display: 'flex', flexDirection: 'column', gap: 10 }}>
        {fixBlock ? (
          <div style={{ background: '#fef2f2', color: '#b91c1c', padding: 8, borderRadius: 6, fontSize: 12 }}>
            <strong>{fixBlock.name}</strong> failed: {fixBlock.last_error}
          </div>
        ) : start.block ? (
          <div style={{ fontSize: 12 }}>
            Changing <span style={mono}>{start.block}</span> in <span style={mono}>{start.file}</span>. The new version replaces it in the library; placed
            copies offer <em>Update</em>.
          </div>
        ) : null}
        <label style={{ fontSize: 12, fontWeight: 600 }}>
          {fixBlock ? 'Anything to add? (optional)' : start.block ? 'What should change?' : 'Describe the block'}
          <textarea
            value={instruction}
            onChange={(e) => setInstruction(e.target.value)}
            rows={5}
            placeholder={
              fixBlock ? 'e.g. the date column can be a string' : 'e.g. PSI per segment between a reference and a current sample; flag segments above 0.25'
            }
            style={{ width: '100%', fontSize: 12, fontWeight: 400, marginTop: 4 }}
          />
        </label>
        {!fixBlock && (
          <label style={{ fontSize: 12 }}>
            Example input (optional)
            <select value={contextBlock} onChange={(e) => setContextBlock(e.target.value)} style={{ width: '100%', fontSize: 12, marginTop: 2 }}>
              <option value="">(none)</option>
              {contextChoices.map((b) => (
                <option key={b.id} value={b.id}>
                  {b.name}
                </option>
              ))}
            </select>
            <span style={{ ...muted, fontSize: 11 }}>The AI sees its column names, types and roles -- never rows.</span>
          </label>
        )}
        {!fixBlock && !start.block && (
          <label style={{ fontSize: 12 }}>
            Save into
            <select value={target} onChange={(e) => setTarget(e.target.value)} disabled={!!draft} style={{ width: '100%', fontSize: 12, marginTop: 2 }}>
              <option value="">a new file</option>
              {userFiles.map((f) => (
                <option key={f.name} value={f.name}>
                  {f.name} (reuses its helpers)
                </option>
              ))}
            </select>
          </label>
        )}
        {!fixBlock && (
          <label style={{ fontSize: 12, display: 'flex', gap: 6, alignItems: 'center' }}>
            <input type="checkbox" checked={addToCanvas} onChange={(e) => setAddToCanvas(e.target.checked)} />
            Add to canvas after saving
          </label>
        )}
        <button className="brand-primary" disabled={busy || (!instruction.trim() && !fixBlock)} onClick={() => ask()}>
          {busy && !draft ? 'Drafting…' : draft ? 'Draft again' : '✦ Draft'}
        </button>
        {busy && <div style={{ ...muted, fontSize: 11 }}>Asking {provider ?? 'the AI'}… {elapsed > 4 ? `${elapsed}s` : ''}</div>}
        {draft && (
          <>
            <div style={{ fontSize: 12, color: 'var(--brand-dark)' }}>{draft.explanation}</div>
            <div style={sectionTitle}>Refine</div>
            <textarea value={refine} onChange={(e) => setRefine(e.target.value)} rows={3} placeholder="e.g. make alpha a param" style={{ width: '100%', fontSize: 12 }} />
            <button disabled={busy || !refine.trim()} onClick={() => ask(refine)}>
              Refine draft
            </button>
          </>
        )}
      </div>
      <div style={{ flex: 1, minWidth: 0, display: 'flex', flexDirection: 'column', padding: 12, gap: 8 }}>
        {!draft ? (
          <div style={{ ...muted, fontSize: 13, maxWidth: 560 }}>
            <p>The AI writes a library block: the function, its ports and params form, its column-role rules, and a small self-check.</p>
            <p>You review it here -- edit anything -- try it, and save it to your library. Only saved blocks can be placed on the canvas.</p>
          </div>
        ) : (
          <>
            <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
              <span style={{ ...mono, fontSize: 12 }}>
                {draft.file} · {draft.block}
              </span>
              {draft.base_sha && (
                <button style={{ fontSize: 11 }} onClick={() => setShowDiff((v) => !v)}>
                  {showDiff ? 'Show file' : 'Show diff'}
                </button>
              )}
              <span style={{ flex: 1 }} />
              <button onClick={tryExamples}>Run self-check</button>
              {contextBlock && <button onClick={tryOnContext}>Try on {graph?.blocks[contextBlock]?.name ?? 'input'}</button>}
              <button className="brand-primary" disabled={busy || (check !== null && !check.ok)} onClick={save} title={check && !check.ok ? 'Fix the errors on the right first' : undefined}>
                {fixBlock ? 'Save & update block' : 'Save to library'}
              </button>
            </div>
            <div style={{ flex: 1, minHeight: 0, display: 'flex', gap: 8 }}>
              <div style={{ flex: 1, minWidth: 0 }}>
                {showDiff ? (
                  <pre style={{ ...mono, fontSize: 11, margin: 0, height: '100%', overflow: 'auto', background: '#fff', border: '1px solid #e5e7eb', borderRadius: 6, padding: 8 }}>
                    {draft.diff.split('\n').map((l, i) => (
                      <div key={i} style={{ color: l.startsWith('+') ? '#166534' : l.startsWith('-') ? '#b91c1c' : undefined }}>
                        {l}
                      </div>
                    ))}
                  </pre>
                ) : (
                  <CodeEditor value={source} onChange={setSource} onSave={save} fill />
                )}
              </div>
              <div style={{ width: 300, overflowY: 'auto' }}>
                <div style={{ ...box, marginBottom: 8 }}>
                  <div style={{ fontWeight: 600, marginBottom: 4, fontSize: 12 }}>Check</div>
                  <Checklist check={check} />
                </div>
                {tryResult && <TryOutView result={tryResult} />}
              </div>
            </div>
          </>
        )}
      </div>
    </>
  )
}

// ---- import ----------------------------------------------------------------------------

function ImportView({ onDone, onError }: { onDone: () => void; onError: (msg: string) => void }) {
  const [upload, setUpload] = useState<{ name: string; b64: string } | null>(null)
  const [preview, setPreview] = useState<ImportPreview | null>(null)
  const [actions, setActions] = useState<Record<string, string>>({})
  const [openCode, setOpenCode] = useState<string | null>(null)

  const pick = async (file: File | undefined) => {
    if (!file) return
    try {
      const b64 = await fileToBase64(file)
      const p = await api.libraryImportPreview(file.name, b64)
      setUpload({ name: file.name, b64 })
      setPreview(p)
      const defaults: Record<string, string> = {}
      for (const f of p.files)
        for (const b of f.blocks) defaults[`${f.file}:${b.id}`] = b.errors.length ? 'skip' : b.clash ? (b.clash.kind === 'builtin' ? 'rename' : 'overwrite') : 'add'
      setActions(defaults)
    } catch (e) {
      onError((e as Error).message)
    }
  }

  const counts = Object.values(actions).reduce<Record<string, number>>((acc, a) => ({ ...acc, [a]: (acc[a] ?? 0) + 1 }), {})
  const importing = Object.values(actions).filter((a) => a !== 'skip').length

  return (
    <div style={{ flex: 1, overflowY: 'auto', padding: 16 }}>
      <div style={{ background: '#fffbeb', border: '1px solid #fde68a', borderRadius: 8, padding: 10, color: '#78350f', fontSize: 12, marginBottom: 12, maxWidth: 900 }}>
        Blocks are Python code that runs on your machine when you run a model. Only import files from people you trust, and open each row to read
        its code. This preview only parses the files -- nothing has run.
      </div>
      <input type="file" accept=".py,.zip" onChange={(e) => pick(e.target.files?.[0])} />
      {preview && (
        <table style={{ width: '100%', maxWidth: 1000, borderCollapse: 'collapse', marginTop: 12, fontSize: 12 }}>
          <thead>
            <tr>
              {['Block', 'Check', 'In your library', 'Action'].map((h) => (
                <th key={h} style={{ textAlign: 'left', padding: 6, borderBottom: '1px solid #e5e7eb', color: '#4b5563' }}>
                  {h}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {preview.files.map((f) => (
              <Fragment key={f.file}>
                {f.file_errors.length > 0 && (
                  <tr key={`${f.file}:err`}>
                    <td style={{ padding: 6, ...mono }}>{f.file}</td>
                    <td style={{ padding: 6, color: '#b91c1c' }} colSpan={3}>
                      ✕ {f.file_errors.join('; ')}
                    </td>
                  </tr>
                )}
                {f.blocks.map((b) => {
                  const key = `${f.file}:${b.id}`
                  return (
                    <Fragment key={key}>
                      <tr style={{ borderBottom: '1px solid #f3f4f6' }}>
                        <td style={{ padding: 6 }}>
                          <strong>{b.display_name}</strong>
                          <div style={{ ...mono, ...muted, fontSize: 11 }}>
                            {f.file} · {b.id} · v{b.version}{' '}
                            <button style={{ fontSize: 10, padding: '0 4px' }} onClick={() => setOpenCode(openCode === key ? null : key)}>
                              {openCode === key ? 'Hide code' : 'Show code'}
                            </button>
                          </div>
                        </td>
                        <td style={{ padding: 6 }}>
                          {b.errors.length ? (
                            <span style={{ color: '#b91c1c' }}>✕ {b.errors[0]}</span>
                          ) : (
                            <span style={{ color: '#3f6259', fontWeight: 600 }}>✓ Valid</span>
                          )}
                          {b.unmet_requirements.length > 0 && <div style={{ color: '#92400e' }}>needs {b.unmet_requirements.join(', ')}</div>}
                        </td>
                        <td style={{ padding: 6 }}>
                          {!b.clash ? (
                            <span style={muted}>new</span>
                          ) : b.clash.kind === 'builtin' ? (
                            <span style={{ color: '#92400e', fontWeight: 600 }}>same id as a built-in</span>
                          ) : (
                            <span style={{ color: '#92400e', fontWeight: 600 }}>
                              exists in {b.clash.file} · v{b.clash.version}
                            </span>
                          )}
                        </td>
                        <td style={{ padding: 6 }}>
                          <select value={actions[key]} onChange={(e) => setActions({ ...actions, [key]: e.target.value })} aria-label={`Action for ${b.id}`}>
                            {!b.clash && <option value="add">Add</option>}
                            {b.clash?.kind === 'library' && <option value="overwrite">Overwrite (v{b.clash.version} → v{b.version})</option>}
                            {b.clash && <option value="rename">Import as {b.id}_2</option>}
                            <option value="skip">Skip</option>
                          </select>
                        </td>
                      </tr>
                      {openCode === key && (
                        <tr>
                          <td colSpan={4} style={{ padding: 6 }}>
                            <CodeEditor value={b.code} readOnly maxHeight={260} />
                          </td>
                        </tr>
                      )}
                    </Fragment>
                  )
                })}
              </Fragment>
            ))}
          </tbody>
        </table>
      )}
      {preview && upload && (
        <div style={{ display: 'flex', gap: 8, alignItems: 'center', marginTop: 12, maxWidth: 1000 }}>
          <span style={{ ...muted, flex: 1 }}>
            {importing} to import{counts.overwrite ? ` · ${counts.overwrite} overwrite` : ''}
            {counts.skip ? ` · ${counts.skip} skipped` : ''}
          </span>
          <button
            className="brand-primary"
            disabled={!importing}
            onClick={() =>
              api
                .libraryImport(upload.name, upload.b64, actions)
                .then(onDone)
                .catch((e) => onError(e.message))
            }
          >
            Import {importing} block{importing === 1 ? '' : 's'}
          </button>
        </div>
      )}
    </div>
  )
}
