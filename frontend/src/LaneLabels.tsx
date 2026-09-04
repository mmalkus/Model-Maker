import { useEffect, useState } from 'react'
import { useViewport } from '@xyflow/react'
import { MIN_LANE_HEIGHT, type LaneLayoutEntry } from './LaneBand'

export function LaneLabels({
  lanes,
  collapsedLanes,
  onToggleCollapse,
  onRename,
  onDelete,
  onMove,
}: {
  lanes: LaneLayoutEntry[]
  collapsedLanes: Set<string>
  onToggleCollapse: (laneId: string) => void
  onRename: (laneId: string, name: string) => void
  onDelete: (laneId: string) => void
  onMove: (laneId: string, direction: -1 | 1) => void
}) {
  const viewport = useViewport()
  const [renamingId, setRenamingId] = useState<string | null>(null)
  const [renameText, setRenameText] = useState('')

  const startRename = (laneId: string, currentName: string) => {
    setRenamingId(laneId)
    setRenameText(currentName)
  }

  const commitRename = (laneId: string) => {
    setRenamingId(null)
    if (renameText.trim()) onRename(laneId, renameText.trim())
  }

  return (
    <>
      {lanes.map(({ id, lane, top, height }, i) => {
        const collapsed = collapsedLanes.has(id)
        const screenTop = viewport.y + top * viewport.zoom
        return (
          <div
            key={id}
            style={{
              position: 'absolute',
              left: 8,
              top: screenTop + 4,
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
              maxHeight: height * viewport.zoom - 8,
              overflow: 'hidden',
            }}
          >
            <button style={{ padding: '0 4px' }} onClick={() => onToggleCollapse(id)} title="Collapse/expand">
              {collapsed ? '▸' : '▾'}
            </button>
            {renamingId === id ? (
              <input
                autoFocus
                value={renameText}
                onChange={(e) => setRenameText(e.target.value)}
                onBlur={() => commitRename(id)}
                onKeyDown={(e) => {
                  if (e.key === 'Enter') commitRename(id)
                  if (e.key === 'Escape') setRenamingId(null)
                }}
                style={{ fontSize: 11, width: 100, boxSizing: 'border-box' }}
              />
            ) : (
              <strong style={{ cursor: 'text' }} title="Click to rename" onDoubleClick={() => startRename(id, lane.name)}>
                {lane.name}
              </strong>
            )}
            <button style={{ padding: '0 4px' }} onClick={() => onMove(id, -1)} disabled={i === 0} title="Move up">
              {'↑'}
            </button>
            <button style={{ padding: '0 4px' }} onClick={() => onMove(id, 1)} disabled={i === lanes.length - 1} title="Move down">
              {'↓'}
            </button>
            <button style={{ padding: '0 4px' }} onClick={() => startRename(id, lane.name)} title="Rename">
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

// One draggable strip per boundary between two consecutive lanes -- since
// lanes stack with no gaps, the boundary below lane i *is* the boundary
// above lane i+1, so dragging it is "drag lane i's lower bound" and "drag
// lane i+1's upper bound" at once. Height changes are clamped to
// MIN_LANE_HEIGHT so lanes can never invert or overlap.
export function LaneResizeHandles({
  lanes,
  collapsedLanes,
  onResize,
}: {
  lanes: LaneLayoutEntry[]
  collapsedLanes: Set<string>
  onResize: (laneId: string, height: number) => void
}) {
  const viewport = useViewport()
  const [dragging, setDragging] = useState<{ laneId: string; startY: number; startHeight: number; liveY: number } | null>(null)

  useEffect(() => {
    if (!dragging) return
    const onMove = (e: MouseEvent) => setDragging((d) => (d ? { ...d, liveY: e.clientY } : d))
    const onUp = (e: MouseEvent) => {
      setDragging((d) => {
        if (d) {
          const deltaFlow = (e.clientY - d.startY) / viewport.zoom
          onResize(d.laneId, Math.max(MIN_LANE_HEIGHT, Math.round(d.startHeight + deltaFlow)))
        }
        return null
      })
    }
    window.addEventListener('mousemove', onMove)
    window.addEventListener('mouseup', onUp)
    return () => {
      window.removeEventListener('mousemove', onMove)
      window.removeEventListener('mouseup', onUp)
    }
  }, [dragging, viewport.zoom, onResize])

  return (
    <>
      {lanes.slice(0, -1).map((entry, i) => {
        const below = lanes[i + 1]
        if (collapsedLanes.has(entry.id) || collapsedLanes.has(below.id)) return null
        const baseScreenY = viewport.y + (entry.top + entry.height) * viewport.zoom
        const isDragging = dragging?.laneId === entry.id
        const screenY = isDragging ? dragging.liveY : baseScreenY
        return (
          <div
            key={entry.id}
            onMouseDown={(e) => {
              e.preventDefault()
              setDragging({ laneId: entry.id, startY: e.clientY, startHeight: entry.height, liveY: e.clientY })
            }}
            title={`Drag to resize "${entry.lane.name}"`}
            style={{
              position: 'absolute',
              left: 0,
              right: 0,
              top: screenY - 3,
              height: 6,
              cursor: 'row-resize',
              zIndex: 6,
              background: isDragging ? 'var(--brand)' : 'transparent',
            }}
          />
        )
      })}
    </>
  )
}
