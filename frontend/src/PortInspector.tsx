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

  if (!block) {
    return (
      <div style={{ width: 340, borderLeft: '1px solid #e5e7eb', padding: 12, color: '#9ca3af', fontSize: 12 }}>
        This block no longer exists.
      </div>
    )
  }

  const dataName = block.port_names[port]
  const title = dataName || `${block.name} :: ${port}`

  const rename = () => {
    const name = prompt('Name this data:', dataName ?? '')
    if (name === null) return
    api.renamePort(blockId, port, name.trim() || null).then(onChanged).catch((err) => alert((err as Error).message))
  }

  return (
    <div style={{ width: 340, borderLeft: '1px solid #e5e7eb', padding: 12, overflowY: 'auto', fontSize: 12 }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'baseline', marginBottom: 4 }}>
        <h3 style={{ fontSize: 14, margin: 0, cursor: 'text' }} title="Click to name this data" onClick={rename}>
          {title}
        </h3>
        <button onClick={onClose}>Close</button>
      </div>
      <div style={{ color: '#6b7280', marginBottom: 12 }}>
        {block.name} :: {port}
      </div>

      <PortDataView blockId={blockId} port={port} portType={portType} title={title} onChanged={onChanged} />
    </div>
  )
}
