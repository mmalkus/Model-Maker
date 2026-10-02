import { useEffect, useRef, useState } from 'react'
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
  master_scale: { icon: '▤', label: 'scale', color: '#0891b2' },
  binning: { icon: '▥', label: 'binning', color: '#0e7490' },
  any: { icon: '?', label: 'value', color: '#6b7280' },
  distribution: { icon: '~', label: 'distribution', color: '#db2777' },
  dependency: { icon: '⋈', label: 'dependency', color: '#4338ca' },
  proxy_function: { icon: '≈', label: 'proxy', color: '#b45309' },
  simulation_result: { icon: '◈', label: 'sim result', color: '#0d9488' },
}

function formatParamValue(v: unknown): string {
  if (v === null || v === undefined) return String(v)
  if (Array.isArray(v)) return `[${v.length}]`
  if (typeof v === 'object') return '{…}'
  const s = String(v)
  return s.length > 18 ? `${s.slice(0, 16)}…` : s
}

export type BlockNodeData = {
  block: BlockOut
  onViewPort: (blockId: string, port: string, portType: PortType) => void
  // AI build overlays (see BuildPanel): a change the plan proposes to this
  // (user) block, and whether the build's latest tool call touched it.
  aiPlannedChange?: string | null
  aiActive?: boolean
  // Opens an AI build's full report (see BuildReport.ReportModal).
  onOpenBuildReport?: (buildId: string) => void
}
export type BlockFlowNode = Node<BlockNodeData, 'modelBlock'>

// The AI / AI-changed badge's click-open popover: every AI build that
// created or changed this block, each with a link to its report -- so a
// report stays reachable from the canvas after later builds have taken
// over the build panel.
function ProvenancePopover({
  block,
  onOpenBuildReport,
  onClose,
}: {
  block: BlockOut
  onOpenBuildReport?: (buildId: string) => void
  onClose: () => void
}) {
  const p = block.provenance
  const ref = useRef<HTMLDivElement>(null)
  useEffect(() => {
    const onDown = (e: MouseEvent) => {
      if (ref.current && !ref.current.contains(e.target as globalThis.Node)) onClose()
    }
    document.addEventListener('mousedown', onDown)
    return () => document.removeEventListener('mousedown', onDown)
  }, [onClose])
  if (!p) return null
  const link = (buildId: string) =>
    onOpenBuildReport && (
      <button
        onClick={() => {
          onClose()
          onOpenBuildReport(buildId)
        }}
        style={{ background: 'none', border: 'none', padding: 0, color: '#2563eb', cursor: 'pointer', fontSize: 11 }}
      >
        Open build report
      </button>
    )
  const entries = [
    ...(p.source === 'agent' && p.build_id
      ? [
          {
            key: 'built',
            buildId: p.build_id,
            heading: `Built by AI build ${p.build_id}`,
            lines: [
              `${p.at ? new Date(p.at).toLocaleString() : '?'} · plan: ${p.plan_llm ?? '?'} · build: ${p.build_llm ?? '?'}`,
              p.goal ? `Goal: ${p.goal}` : '',
              p.modified_by_user ? 'Changed by a person since.' : 'Not edited since.',
            ],
          },
        ]
      : []),
    ...(p.changes ?? []).map((c, i) => ({
      key: `change-${i}`,
      buildId: c.build_id,
      heading: `Changed by AI build ${c.build_id}`,
      lines: [`${new Date(c.at).toLocaleString()} · ${c.change}`],
    })),
  ]
  return (
    <div
      ref={ref}
      className="nodrag nopan nowheel"
      onClick={(e) => e.stopPropagation()}
      style={{
        position: 'absolute',
        top: 30,
        right: -4,
        width: 240,
        background: '#fff',
        border: '1px solid #d1d5db',
        borderRadius: 6,
        boxShadow: '0 4px 12px rgba(0,0,0,0.15)',
        padding: '6px 8px',
        fontSize: 11,
        lineHeight: 1.45,
        color: '#374151',
        zIndex: 20,
        cursor: 'default',
      }}
    >
      {entries.map((e, i) => (
        <div key={e.key} style={i > 0 ? { marginTop: 6, paddingTop: 6, borderTop: '1px solid #f3f4f6' } : undefined}>
          <div style={{ fontWeight: 600 }}>{e.heading}</div>
          {e.lines.filter(Boolean).map((line, j) => (
            <div key={j} style={{ color: '#6b7280' }}>
              {line}
            </div>
          ))}
          {link(e.buildId)}
        </div>
      ))}
    </div>
  )
}

export function BlockNode({ data, selected }: NodeProps<BlockFlowNode>) {
  const { block, onViewPort, aiPlannedChange, aiActive, onOpenBuildReport } = data
  const [provenanceOpen, setProvenanceOpen] = useState(false)
  const color = STATUS_COLOR[block.status] ?? STATUS_COLOR.grey
  const paramEntries = Object.entries(block.params)
  const aiBuilt = block.provenance?.source === 'agent'
  const aiChanged = (block.provenance?.changes?.length ?? 0) > 0

  return (
    <div
      style={{
        position: 'relative',
        border: `2px solid ${color}`,
        borderRadius: 8,
        background: '#fff',
        minWidth: 160,
        // Without a ceiling, a long status line (a staleness reason naming
        // the upstream edit that caused it) stretches the card right across
        // the canvas instead of wrapping inside it.
        maxWidth: 280,
        boxShadow: aiActive
          ? '0 0 0 3px #a78bfa'
          : selected
            ? '0 0 0 2px var(--brand)'
            : '0 1px 3px rgba(0,0,0,0.15)',
        transition: 'box-shadow 0.2s',
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
        {(aiBuilt || aiChanged) && (
          <button
            className="nodrag"
            // Keeps the popover's outside-click close from firing first, which
            // would make a click meant to close it reopen it instead.
            onMouseDown={(e) => e.stopPropagation()}
            onClick={(e) => {
              e.stopPropagation()
              setProvenanceOpen((v) => !v)
            }}
            title="Which AI build made this block -- click for details and its report"
            style={{
              fontSize: 10,
              fontWeight: aiBuilt ? 600 : undefined,
              color: '#6d28d9',
              background: aiBuilt ? '#ede9fe' : '#fff',
              border: aiBuilt ? 'none' : '1px solid #c4b5fd',
              borderRadius: 4,
              padding: '0 4px',
              flexShrink: 0,
              marginLeft: 'auto',
              cursor: 'pointer',
            }}
          >
            {aiBuilt ? `AI${block.provenance?.modified_by_user ? ' · edited' : ''}` : 'AI-changed'}
          </button>
        )}
      </div>
      {provenanceOpen && (
        <ProvenancePopover block={block} onOpenBuildReport={onOpenBuildReport} onClose={() => setProvenanceOpen(false)} />
      )}
      {aiPlannedChange && (
        <div
          title="The AI build's plan proposes this change to your block -- approving the plan allows it"
          style={{ background: '#f5f3ff', color: '#6d28d9', fontSize: 10, padding: '3px 8px', borderBottom: '1px dashed #c4b5fd' }}
        >
          planned change: {aiPlannedChange}
        </div>
      )}
      <div style={{ padding: '4px 8px 8px', color: '#6b7280' }}>
        <div>
          {block.category}
          {block.status === 'running' && (
            <span style={{ color: '#2563eb', fontWeight: 600 }}>
              {' '}
              · running{block.running_seconds != null && ` (${Math.round(block.running_seconds)}s)`}…
            </span>
          )}
        </div>

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
