import { useEffect, useState } from 'react'
import { api } from './api'
import type { BlockTag, LibraryFile, RegistryEntry } from './types'

function InlineAdd({ placeholder, onSubmit }: { placeholder: string; onSubmit: (value: string) => void }) {
  const [open, setOpen] = useState(false)
  const [value, setValue] = useState('')

  if (!open) {
    return (
      <button onClick={() => setOpen(true)} style={{ display: 'block', width: '100%', textAlign: 'left', fontSize: 12 }}>
        + {placeholder}
      </button>
    )
  }

  const submit = () => {
    if (value.trim()) onSubmit(value.trim())
    setValue('')
    setOpen(false)
  }

  return (
    <div style={{ display: 'flex', gap: 4 }}>
      <input
        autoFocus
        value={value}
        onChange={(e) => setValue(e.target.value)}
        onKeyDown={(e) => {
          if (e.key === 'Enter') submit()
          if (e.key === 'Escape') setOpen(false)
        }}
        placeholder={placeholder}
        style={{ flex: 1, fontSize: 12, padding: '4px 6px', border: '1px solid #d1d5db', borderRadius: 6 }}
      />
      <button onClick={submit} style={{ fontSize: 12, padding: '2px 6px' }}>
        Add
      </button>
    </div>
  )
}

const GROUP_ORDER = ['input', 'standard', 'quality', 'modelling', 'stochastic', 'tests', 'output']
const GROUP_LABELS: Record<string, string> = {
  input: 'Input',
  standard: 'Standard',
  quality: 'Data quality',
  modelling: 'Modelling',
  stochastic: 'Stochastic',
  tests: 'Tests',
  output: 'Output',
}

// Payload dragged from a palette button, read back on drop in App.tsx
const DRAG_MIME = 'application/x-modelmaker-block'

export type DragPayload = { kind: 'registry'; category: string } | { kind: 'library'; file: string; id: string }

function dragStartFor(payload: DragPayload) {
  return (e: React.DragEvent) => {
    e.dataTransfer.setData(DRAG_MIME, JSON.stringify(payload))
    e.dataTransfer.effectAllowed = 'copy'
  }
}

// Why a library block can't be placed right now, or null when it can.
function unplaceableReason(b: LibraryFile['blocks'][number], file: LibraryFile): string | null {
  const problems = [...file.file_errors, ...b.errors]
  if (problems.length) return problems.join('; ')
  if (b.unmet_requirements.length) return `needs ${b.unmet_requirements.join(', ')}`
  return null
}

export function Palette({
  onAdd,
  onAddLibrary,
  onOpenLibrary,
  onCreateWithAI,
  onAddLane,
  libraryVersion,
}: {
  onAdd: (category: string) => void
  onAddLibrary: (file: string, id: string) => void
  onOpenLibrary: () => void
  onCreateWithAI: () => void
  onAddLane: (name: string) => void
  // Bumped by App whenever the library may have changed (a save, an import,
  // an AI build's drafts), so the palette refetches it.
  libraryVersion: number
}) {
  const [entries, setEntries] = useState<RegistryEntry[]>([])
  const [collapsedGroups, setCollapsedGroups] = useState<Set<string>>(new Set())
  const [tags, setTags] = useState<BlockTag[]>([])
  // '' shows every block; otherwise only blocks carrying this tag
  const [tagFilter, setTagFilter] = useState('')

  const [libraryFiles, setLibraryFiles] = useState<LibraryFile[]>([])

  useEffect(() => {
    api.registry().then(setEntries).catch(console.error)
    api.registryTags().then(setTags).catch(console.error)
  }, [])

  useEffect(() => {
    api
      .library()
      .then((lib) => setLibraryFiles(lib.files))
      .catch(console.error)
  }, [libraryVersion])

  const toggleGroup = (group: string) => {
    setCollapsedGroups((prev) => {
      const next = new Set(prev)
      if (next.has(group)) next.delete(group)
      else next.add(group)
      return next
    })
  }

  const groups: Record<string, RegistryEntry[]> = {}
  for (const e of entries) {
    if (tagFilter && !e.tags.includes(tagFilter)) continue
    ;(groups[e.group] ??= []).push(e)
  }
  const orderedGroupNames = [...GROUP_ORDER.filter((g) => groups[g]), ...Object.keys(groups).filter((g) => !GROUP_ORDER.includes(g))]

  const buttonStyle = {
    display: 'block' as const,
    width: '100%',
    textAlign: 'left' as const,
    padding: '6px 8px',
    marginBottom: 4,
    fontSize: 12,
    borderRadius: 6,
  }

  return (
    <div style={{ width: 200, borderRight: '1px solid #e5e7eb', padding: 12, overflowY: 'auto' }}>
      <h3 style={{ fontSize: 13, margin: '0 0 8px', color: 'var(--brand-ink)' }}>Blocks</h3>

      {tags.length > 0 && (
        <select
          value={tagFilter}
          onChange={(e) => setTagFilter(e.target.value)}
          title={tags.find((t) => t.tag === tagFilter)?.about ?? 'Show only blocks with this tag'}
          aria-label="Filter blocks by tag"
          style={{ width: '100%', marginBottom: 12, fontSize: 12, padding: '4px 6px', border: '1px solid #d1d5db', borderRadius: 6 }}
        >
          <option value="">All tags</option>
          {tags.map((t) => (
            <option key={t.tag} value={t.tag} title={t.about}>
              {t.tag.replace(/_/g, ' ')}
            </option>
          ))}
        </select>
      )}

      <div style={{ marginBottom: 14, paddingBottom: 10, borderBottom: '1px solid #f3f4f6' }}>
        <div style={{ display: 'flex', alignItems: 'center', marginBottom: 6 }}>
          <button
            onClick={onOpenLibrary}
            title="Browse, edit, import and export your block library"
            style={{ background: 'none', border: 'none', padding: 0, fontSize: 11, color: '#6b7280', textTransform: 'uppercase', cursor: 'pointer' }}
          >
            Library ↗
          </button>
        </div>
        <button
          onClick={onCreateWithAI}
          title="Describe a block; AI writes it into your library, you review and save it, then it's placed on the canvas"
          style={{
            ...buttonStyle,
            textAlign: 'center',
            fontWeight: 600,
            background: 'var(--brand-light)',
            border: '1px solid var(--brand-border)',
            color: 'var(--brand-dark)',
          }}
        >
          ✦ Create with AI
        </button>
        {libraryFiles.length === 0 && (
          <div style={{ fontSize: 11, color: '#9ca3af' }}>No library blocks yet -- create one with AI, or open the library.</div>
        )}
        {libraryFiles.map((file) => {
          const blocks = file.blocks.filter((b) => !tagFilter || b.tags.includes(tagFilter))
          if (!blocks.length && !file.file_errors.length) return null
          const key = `lib:${file.name}`
          const collapsed = !tagFilter && collapsedGroups.has(key)
          return (
            <div key={file.name} style={{ marginBottom: 8 }}>
              <button
                onClick={() => toggleGroup(key)}
                title={file.doc || file.name}
                style={{
                  display: 'flex',
                  alignItems: 'center',
                  gap: 4,
                  width: '100%',
                  textAlign: 'left',
                  background: 'none',
                  border: 'none',
                  padding: 0,
                  marginBottom: 4,
                  fontSize: 11,
                  color: file.is_draft ? '#6d28d9' : '#6b7280',
                  cursor: 'pointer',
                }}
              >
                <span style={{ fontSize: 9 }}>{collapsed ? '▸' : '▾'}</span>
                <span style={{ fontFamily: 'ui-monospace, monospace' }}>{file.is_draft ? `AI draft · ${file.name.slice(7)}` : file.name}</span>
              </button>
              {file.file_errors.length > 0 && <div style={{ fontSize: 11, color: '#b91c1c' }}>{file.file_errors[0]}</div>}
              {!collapsed &&
                blocks.map((b) => {
                  const reason = unplaceableReason(b, file)
                  return (
                    <button
                      key={b.id}
                      draggable={!reason}
                      disabled={!!reason}
                      onDragStart={dragStartFor({ kind: 'library', file: file.name, id: b.id })}
                      onClick={() => onAddLibrary(file.name, b.id)}
                      title={reason ?? `${b.summary || b.display_name}\n\nClick, or drag onto the canvas`}
                      style={{ ...buttonStyle, display: 'flex', alignItems: 'center', gap: 6 }}
                    >
                      <span style={{ flex: 1, overflow: 'hidden', textOverflow: 'ellipsis' }}>{b.display_name}</span>
                      <span style={{ fontSize: 9, fontWeight: 600, color: 'var(--brand)' }}>v{b.version}</span>
                    </button>
                  )
                })}
            </div>
          )
        })}
      </div>

      {orderedGroupNames.map((group) => {
        // a tag filter shows its matches even in groups the user collapsed
        const collapsed = !tagFilter && collapsedGroups.has(group)
        return (
          <div key={group} style={{ marginBottom: 12 }}>
            <button
              onClick={() => toggleGroup(group)}
              style={{
                display: 'flex',
                alignItems: 'center',
                gap: 4,
                width: '100%',
                textAlign: 'left',
                background: 'none',
                border: 'none',
                padding: 0,
                marginBottom: 4,
                fontSize: 11,
                color: '#6b7280',
                textTransform: 'uppercase',
                cursor: 'pointer',
              }}
            >
              <span style={{ fontSize: 9 }}>{collapsed ? '▸' : '▾'}</span>
              {GROUP_LABELS[group] ?? group}
            </button>
            {!collapsed &&
              groups[group].map((e) => (
                <button
                  key={e.category}
                  draggable
                  onDragStart={dragStartFor({ kind: 'registry', category: e.category })}
                  onClick={() => onAdd(e.category)}
                  title="Click, or drag onto the canvas"
                  style={buttonStyle}
                >
                  {e.display_name}
                </button>
              ))}
          </div>
        )
      })}

      <div>
        <div style={{ fontSize: 11, color: '#6b7280', textTransform: 'uppercase', marginBottom: 4 }}>Lanes</div>
        <InlineAdd placeholder="lane name" onSubmit={onAddLane} />
      </div>
    </div>
  )
}
