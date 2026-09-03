import { Handle, Position, type Node, type NodeProps } from '@xyflow/react'
import type { BlockOut } from './types'

const STATUS_COLOR: Record<string, string> = {
  grey: '#9ca3af',
  green: '#22c55e',
  orange: '#f59e0b',
  red: '#ef4444',
}

export type BlockNodeData = { block: BlockOut }
export type BlockFlowNode = Node<BlockNodeData, 'modelBlock'>

export function BlockNode({ data, selected }: NodeProps<BlockFlowNode>) {
  const { block } = data
  const color = STATUS_COLOR[block.status] ?? STATUS_COLOR.grey

  return (
    <div
      style={{
        border: `2px solid ${color}`,
        borderRadius: 8,
        background: '#fff',
        minWidth: 160,
        boxShadow: selected ? '0 0 0 2px var(--brand)' : '0 1px 3px rgba(0,0,0,0.15)',
        fontSize: 12,
      }}
    >
      <div
        style={{
          display: 'flex',
          alignItems: 'center',
          gap: 6,
          padding: '6px 8px',
          borderBottom: '1px solid #eee',
          borderTopLeftRadius: 6,
          borderTopRightRadius: 6,
          background: '#fafafa',
        }}
      >
        <span style={{ width: 8, height: 8, borderRadius: 999, background: color, flexShrink: 0 }} />
        <strong style={{ overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{block.name}</strong>
      </div>
      <div style={{ padding: '4px 8px 8px', color: '#6b7280' }}>
        <div>{block.category}</div>
        {block.status === 'red' && block.last_error && (
          <div style={{ color: '#ef4444', marginTop: 4, wordBreak: 'break-word' }}>{block.last_error}</div>
        )}
      </div>

      {block.inputs.map((port, i) => (
        <Handle
          key={`in-${port.name}`}
          id={port.name}
          type="target"
          position={Position.Top}
          style={{ left: 20 + i * 16, background: '#555' }}
        />
      ))}
      {block.outputs.map((port, i) => (
        <Handle
          key={`out-${port.name}`}
          id={port.name}
          type="source"
          position={Position.Bottom}
          style={{ left: 20 + i * 16, background: '#555' }}
        />
      ))}
    </div>
  )
}
