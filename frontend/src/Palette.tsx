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

export function Palette({
  onAdd,
  onAddCustom,
  onAddLane,
}: {
  onAdd: (category: string) => void
  onAddCustom: (name: string) => void
  onAddLane: (name: string) => void
}) {
  const [entries, setEntries] = useState<RegistryEntry[]>([])

  useEffect(() => {
    api.registry().then(setEntries).catch(console.error)
  }, [])

  const groups: Record<string, RegistryEntry[]> = {}
  for (const e of entries) {
    ;(groups[e.block_type] ??= []).push(e)
  }

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
      <h3 style={{ fontSize: 13, margin: '0 0 8px' }}>Blocks</h3>
      {Object.entries(groups).map(([type, items]) => (
        <div key={type} style={{ marginBottom: 12 }}>
          <div style={{ fontSize: 11, color: '#6b7280', textTransform: 'uppercase', marginBottom: 4 }}>{type}</div>
          {items.map((e) => (
            <button key={e.category} onClick={() => onAdd(e.category)} style={buttonStyle}>
              {e.display_name}
            </button>
          ))}
        </div>
      ))}

      <div style={{ marginBottom: 12 }}>
        <div style={{ fontSize: 11, color: '#6b7280', textTransform: 'uppercase', marginBottom: 4 }}>Custom</div>
        <InlineAdd placeholder="block name" onSubmit={onAddCustom} />
      </div>

      <div>
        <div style={{ fontSize: 11, color: '#6b7280', textTransform: 'uppercase', marginBottom: 4 }}>Lanes</div>
        <InlineAdd placeholder="lane name" onSubmit={onAddLane} />
      </div>
    </div>
  )
}
