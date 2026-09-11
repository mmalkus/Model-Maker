import { Handle, Position, type Node, type NodeProps } from '@xyflow/react'
import type { BlockOut, PortType } from './types'

const STATUS_COLOR: Record<string, string> = {
  grey: '#9ca3af',
  green: '#22c55e',
  orange: '#f59e0b',
  red: '#ef4444',
  running: '#2563eb',
}

// What kind of object each port type actually is, at a glance -- so a
// block's card shows the *object* it produces (a table, a fitted model, a
// single number, a chart) rather than just its plumbing.
const PORT_BADGE: Record<PortType, { icon: string; label: string; color: string }> = {
  dataframe: { icon: '▦', label: 'table', color: '#2563eb' },
  model: { icon: '◆', label: 'model', color: '#7c3aed' },
  scalar_metric: { icon: '#', label: 'metric', color: '#16a34a' },
  image: { icon: '▨', label: 'image', color: '#ea580c' },
  any: { icon: '?', label: 'value', color: '#6b7280' },
}

function formatParamValue(v: unknown): string {
  if (v === null || v === undefined) return String(v)
  if (Array.isArray(v)) return `[${v.length}]`
  if (typeof v === 'object') return '{…}'
  const s = String(v)
  return s.length > 18 ? `${s.slice(0, 16)}…` : s
}

export type BlockNodeData = { block: BlockOut; onViewPort: (blockId: string, port: string, portType: PortType) => void }
export type BlockFlowNode = Node<BlockNodeData, 'modelBlock'>

export function BlockNode({ data, selected }: NodeProps<BlockFlowNode>) {
  const { block, onViewPort } = data
  const color = STATUS_COLOR[block.status] ?? STATUS_COLOR.grey
  const paramEntries = Object.entries(block.params)

  return (
    <div
      style={{
        border: `2px solid ${color}`,
        borderRadius: 8,
        background: '#fff',
        minWidth: 160,
        // Without a ceiling, a long status line (a staleness reason naming
        // the upstream edit that caused it) stretches the card right across
        // the canvas instead of wrapping inside it.
        maxWidth: 280,
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
        <span
          style={{
            width: 8,
            height: 8,
            borderRadius: 999,
            background: color,
            flexShrink: 0,
            animation: block.status === 'running' ? 'mm-pulse 1s ease-in-out infinite' : undefined,
          }}
        />
        <strong style={{ overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{block.name}</strong>
        {block.group_by && (
          <span
            title={`Runs once per distinct value of '${block.group_by}'`}
            style={{ fontSize: 10, color: '#7c3aed', border: '1px solid #7c3aed55', borderRadius: 4, padding: '0 4px', flexShrink: 0 }}
          >
            ⟲ {block.group_by}
          </span>
        )}
      </div>
      <div style={{ padding: '4px 8px 8px', color: '#6b7280' }}>
        <div>{block.category}{block.status === 'running' && <span style={{ color: '#2563eb', fontWeight: 600 }}> · running…</span>}</div>

        {paramEntries.length > 0 && (
          <div style={{ marginTop: 4, display: 'flex', flexWrap: 'wrap', gap: 3 }}>
            {paramEntries.map(([k, v]) => (
              <span
                key={k}
                title={`${k} = ${JSON.stringify(v)}`}
                style={{
                  background: '#f3f4f6',
                  borderRadius: 4,
                  padding: '1px 4px',
                  fontSize: 10,
                  color: '#4b5563',
                  whiteSpace: 'nowrap',
                }}
              >
                {k}={formatParamValue(v)}
              </span>
            ))}
          </div>
        )}

        {block.outputs.length > 0 && (
          <div style={{ marginTop: 4, display: 'flex', flexWrap: 'wrap', gap: 4 }}>
            {block.outputs.map((p) => {
              const badge = PORT_BADGE[p.type] ?? PORT_BADGE.any
              const dataName = block.port_names[p.name]
              return (
                <button
                  key={p.name}
                  className="nodrag"
                  onClick={(e) => {
                    e.stopPropagation()
                    onViewPort(block.id, p.name, p.type)
                  }}
                  title={`${p.name}: produces a ${badge.label} -- click to view or name its data`}
                  style={{
                    display: 'inline-flex',
                    alignItems: 'center',
                    gap: 2,
                    fontSize: 10,
                    color: badge.color,
                    border: `1px solid ${badge.color}55`,
                    borderRadius: 4,
                    padding: '0 4px',
                    background: '#fff',
                    cursor: 'pointer',
                  }}
                >
                  <span>{badge.icon}</span>
                  {dataName ?? p.name}
                </button>
              )
            })}
          </div>
        )}

        {block.status === 'orange' && block.stale_reason && (
          <div
            title={`Stale: ${block.stale_reason}`}
            style={{
              color: '#b45309',
              marginTop: 4,
              display: '-webkit-box',
              WebkitLineClamp: 2,
              WebkitBoxOrient: 'vertical',
              overflow: 'hidden',
            }}
          >
            stale — {block.stale_reason}
          </div>
        )}

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
