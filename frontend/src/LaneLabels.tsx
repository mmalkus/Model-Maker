import { useViewport } from '@xyflow/react'
import { BAND_COLLAPSED_HEIGHT, BAND_HEIGHT, laneY } from './LaneBand'
import type { LaneOut } from './types'

export interface LaneEntry {
  id: string
  lane: LaneOut
}

export function LaneLabels({
  lanes,
  collapsedLanes,
  onToggleCollapse,
  onRename,
  onDelete,
  onMove,
}: {
  lanes: LaneEntry[]
  collapsedLanes: Set<string>
  onToggleCollapse: (laneId: string) => void
  onRename: (laneId: string) => void
  onDelete: (laneId: string) => void
  onMove: (laneId: string, direction: -1 | 1) => void
}) {
  const viewport = useViewport()

  return (
    <>
      {lanes.map(({ id, lane }, i) => {
        const collapsed = collapsedLanes.has(id)
        const top = viewport.y + laneY(lane.order) * viewport.zoom
        return (
          <div
            key={id}
            style={{
              position: 'absolute',
              left: 8,
              top: top + 4,
              display: 'inline-flex',
              alignItems: 'center',
              gap: 4,
              background: '#fff',
              border: '1px solid #e5e7eb',
              borderRadius: 6,
              padding: '2px 6px',
              fontSize: 11,
              color: '#374151',
              boxShadow: '0 1px 2px rgba(0,0,0,0.06)',
              zIndex: 5,
              maxHeight: (collapsed ? BAND_COLLAPSED_HEIGHT : BAND_HEIGHT) * viewport.zoom - 8,
              overflow: 'hidden',
            }}
          >
            <button style={{ padding: '0 4px' }} onClick={() => onToggleCollapse(id)} title="Collapse/expand">
              {collapsed ? '▸' : '▾'}
            </button>
            <strong>{lane.name}</strong>
            <button style={{ padding: '0 4px' }} onClick={() => onMove(id, -1)} disabled={i === 0} title="Move up">
              {'↑'}
            </button>
            <button style={{ padding: '0 4px' }} onClick={() => onMove(id, 1)} disabled={i === lanes.length - 1} title="Move down">
              {'↓'}
            </button>
            <button style={{ padding: '0 4px' }} onClick={() => onRename(id)} title="Rename">
              {'✎'}
            </button>
            <button style={{ padding: '0 4px' }} onClick={() => onDelete(id)} title="Delete lane">
              {'✕'}
            </button>
          </div>
        )
      })}
    </>
  )
}
