import { Handle, Position, type Node, type NodeProps } from '@xyflow/react'
import type { PlanStep } from './types'

// A planned-but-not-built block from an AI build's plan (see BuildPanel and
// agent-builder-proposal.md §4.2). Frontend-only: never part of the graph,
// so it can't be run, saved or wired by hand. One generic input/output
// handle each -- enough to draw the plan's wiring without knowing every
// block type's real ports.
export type GhostNodeData = { step: PlanStep; highlighted: boolean }
export type GhostFlowNode = Node<GhostNodeData, 'ghostBlock'>

export const GHOST_IN = 'ghost_in'
export const GHOST_OUT = 'ghost_out'

export function GhostNode({ data }: NodeProps<GhostFlowNode>) {
  const { step, highlighted } = data
  return (
    <div
      title={`${step.why}${step.instruction ? `\n\n${step.instruction}` : ''}`}
      style={{
        border: `2px dashed ${highlighted ? 'var(--mm-ai)' : 'var(--mm-ai-border-3)'}`,
        borderRadius: 8,
        background: highlighted ? 'rgba(237,233,254,0.95)' : 'rgba(245,243,255,0.8)',
        minWidth: 160,
        maxWidth: 240,
        fontSize: 12,
        color: 'var(--mm-ai-3)',
        padding: '6px 8px',
      }}
    >
      <div style={{ display: 'flex', gap: 6, alignItems: 'baseline' }}>
        <span style={{ fontSize: 10, opacity: 0.7 }}>{step.ref}</span>
        <strong style={{ overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{step.name}</strong>
      </div>
      <div style={{ opacity: 0.8 }}>{step.category === 'custom' ? 'new block (AI writes it)' : step.category}</div>
      <Handle id={GHOST_IN} type="target" position={Position.Top} isConnectable={false} style={{ background: 'var(--mm-ai-border-3)' }} />
      <Handle id={GHOST_OUT} type="source" position={Position.Bottom} isConnectable={false} style={{ background: 'var(--mm-ai-border-3)' }} />
    </div>
  )
}

// A lane the plan would create, drawn as a dashed band under its ghosts.
export type GhostLaneData = { name: string; height: number }
export type GhostLaneNode = Node<GhostLaneData, 'ghostLane'>

export function GhostLane({ data }: NodeProps<GhostLaneNode>) {
  return (
    <div
      style={{
        width: 6000,
        height: data.height,
        border: '2px dashed var(--mm-ai-border)',
        background: 'rgba(245,243,255,0.35)',
        boxSizing: 'border-box',
        pointerEvents: 'none',
      }}
    >
      <div style={{ position: 'sticky', left: 0, padding: '4px 2008px', color: 'var(--mm-ai)', fontSize: 12, fontWeight: 600 }}>
        planned lane: {data.name}
      </div>
    </div>
  )
}
