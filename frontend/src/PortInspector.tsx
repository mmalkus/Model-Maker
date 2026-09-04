import { useEffect, useState } from 'react'
import { api } from './api'
import { PortDataView } from './PortDataView'
import type { GraphOut, PortType } from './types'

export function PortInspector({
  blockId,
  port,
  portType,
  graph,
  onClose,
  onChanged,
}: {
  blockId: string
  port: string
  portType: PortType
  graph: GraphOut
  onClose: () => void
  onChanged: () => void
}) {
  const block = graph.blocks[blockId]
  const dataName = block?.port_names[port]
  const [editing, setEditing] = useState(false)
  const [nameText, setNameText] = useState(dataName ?? '')

  useEffect(() => {
    setEditing(false)
    setNameText(dataName ?? '')
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [blockId, port])

  if (!block) {
    return (
      <div style={{ width: 340, borderLeft: '1px solid #e5e7eb', padding: 12, color: '#9ca3af', fontSize: 12 }}>
        This block no longer exists.
      </div>
    )
  }

  const title = dataName || `${block.name} :: ${port}`

  const saveName = () => {
    setEditing(false)
    if (nameText.trim() === (dataName ?? '')) return
    api.renamePort(blockId, port, nameText.trim() || null).then(onChanged).catch((err) => alert((err as Error).message))
  }

  return (
    <div style={{ width: 340, borderLeft: '1px solid #e5e7eb', padding: 12, overflowY: 'auto', fontSize: 12 }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'baseline', marginBottom: 4 }}>
        {editing ? (
          <input
            autoFocus
            value={nameText}
            placeholder="Name this data..."
            onChange={(e) => setNameText(e.target.value)}
            onBlur={saveName}
            onKeyDown={(e) => {
              if (e.key === 'Enter') saveName()
              if (e.key === 'Escape') {
                setNameText(dataName ?? '')
                setEditing(false)
              }
            }}
            style={{ fontSize: 14, fontWeight: 600, flex: 1, boxSizing: 'border-box', marginRight: 8 }}
          />
        ) : (
          <h3
            style={{ fontSize: 14, margin: 0, cursor: 'text' }}
            title="Click to name this data"
            onClick={() => setEditing(true)}
          >
            {title}
          </h3>
        )}
        <button onClick={onClose}>Close</button>
      </div>
      <div style={{ color: '#6b7280', marginBottom: 12 }}>
        {block.name} :: {port}
      </div>

      <PortDataView blockId={blockId} port={port} portType={portType} title={title} onChanged={onChanged} />
    </div>
  )
}
