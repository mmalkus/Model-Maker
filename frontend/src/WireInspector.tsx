import { api } from './api'
import { PortDataView } from './PortDataView'
import type { GraphOut, PortType, WireOut } from './types'

export function WireInspector({
  wireId,
  graph,
  onClose,
  onChanged,
}: {
  wireId: string
  graph: GraphOut
  onClose: () => void
  onChanged: () => void
}) {
  const wire: WireOut | undefined = graph.wires[wireId]
  const fromBlock = wire ? graph.blocks[wire.from_block] : undefined
  const portType: PortType | undefined = fromBlock?.outputs.find((p) => p.name === wire?.from_port)?.type

  if (!wire) {
    return (
      <div style={{ width: 340, borderLeft: '1px solid #e5e7eb', padding: 12, color: '#9ca3af', fontSize: 12 }}>
        This wire no longer exists.
      </div>
    )
  }

  const title = wire.name || fromBlock?.name || `${wire.from_block}:${wire.from_port}`

  const rename = () => {
    const name = prompt('Name this data:', wire.name ?? '')
    if (name === null) return
    api.renameWire(wireId, name.trim() || null).catch((err) => alert((err as Error).message))
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
        {wire.from_block}:{wire.from_port} &rarr; {wire.to_block}:{wire.to_port}
        {!wire.valid && <div style={{ color: '#ef4444', marginTop: 2 }}>This wire is no longer valid.</div>}
      </div>

      <PortDataView blockId={wire.from_block} port={wire.from_port} portType={portType} title={title} onChanged={onChanged} />
    </div>
  )
}
