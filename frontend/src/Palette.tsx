import { useEffect, useState } from 'react'
import { api } from './api'
import type { RegistryEntry } from './types'

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

const GROUP_ORDER = ['input', 'standard', 'modelling', 'stochastic', 'tests', 'output']
const GROUP_LABELS: Record<string, string> = {
  input: 'Input',
  standard: 'Standard',
  modelling: 'Modelling',
  stochastic: 'Stochastic',
  tests: 'Tests',
  output: 'Output',
}

const AI_BLOCK_TYPES: { blockType: 'input' | 'standard' | 'output'; label: string; title: string }[] = [
  { blockType: 'input', label: '+ AI input block', title: 'A pipeline source with no upstream, like Read CSV -- Claude writes the code' },
  { blockType: 'standard', label: '+ AI standard block', title: 'A mid-pipeline transform (df in, df out) -- Claude writes the code' },
  { blockType: 'output', label: '+ AI output block', title: 'A terminal block (report, metric, export...) -- Claude writes the code' },
]

// Payload dragged from a palette button, read back on drop in App.tsx
const DRAG_MIME = 'application/x-modelmaker-block'

function dragStartFor(payload: { kind: 'registry'; category: string } | { kind: 'custom'; blockType: 'input' | 'standard' | 'output' }) {
  return (e: React.DragEvent) => {
    e.dataTransfer.setData(DRAG_MIME, JSON.stringify(payload))
    e.dataTransfer.effectAllowed = 'copy'
  }
}

export function Palette({
  onAdd,
  onAddCustom,
  onAddLane,
}: {
  onAdd: (category: string) => void
  onAddCustom: (blockType: 'input' | 'standard' | 'output') => void
  onAddLane: (name: string) => void
}) {
  const [entries, setEntries] = useState<RegistryEntry[]>([])
  const [collapsedGroups, setCollapsedGroups] = useState<Set<string>>(new Set())

  useEffect(() => {
    api.registry().then(setEntries).catch(console.error)
  }, [])

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

      <div style={{ marginBottom: 12 }}>
        {AI_BLOCK_TYPES.map(({ blockType, label, title }) => (
          <button
            key={blockType}
            draggable
            onDragStart={dragStartFor({ kind: 'custom', blockType })}
            onClick={() => onAddCustom(blockType)}
            title={`${title} (click, or drag onto the canvas)`}
            style={{
              ...buttonStyle,
              textAlign: 'center',
              fontWeight: 600,
              background: 'var(--brand-light)',
              border: '1px solid var(--brand-border)',
              color: 'var(--brand-dark)',
            }}
          >
            {label}
          </button>
        ))}
      </div>

      {orderedGroupNames.map((group) => {
        const collapsed = collapsedGroups.has(group)
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
