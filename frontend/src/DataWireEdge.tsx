import { BaseEdge, getBezierPath, useViewport, type Edge, type EdgeProps } from '@xyflow/react'
import { createContext, useContext } from 'react'
import { createPortal } from 'react-dom'
import { api } from './api'
import type { WireOut } from './types'

export type DataWireData = {
  wire: WireOut
  onView: (wireId: string) => void
  isViewed: boolean
}
export type DataWireEdgeType = Edge<DataWireData, 'dataWire'>

// Where a wire's midpoint control (see below) portals its DOM content --
// supplied by App.tsx as a plain <div> rendered as a *direct* ReactFlow
// child, the same level as LaneLabels. Edge components render deep inside
// ReactFlow's internal node/edge layer, which the lane toolbar and its
// resize handles -- also direct ReactFlow children -- always paint over
// regardless of any z-index set from inside that layer (it's a lower
// stacking level, full stop); portaling out to a sibling at that same top
// level is what lets a wire's controls win instead of getting stuck behind
// a lane's label when the two overlap on screen.
export const WirePortalContext = createContext<HTMLDivElement | null>(null)

// A wire is data flowing from one block's output port to another's input --
// the small button at its midpoint opens a look at that data itself
// (whatever shape it is: a dataframe preview, an image, or a plain value) in
// the side panel, same as clicking a block does, and the wire can be given
// its own name, shown right there on the canvas.
export function DataWireEdge({
  id,
  sourceX,
  sourceY,
  targetX,
  targetY,
  sourcePosition,
  targetPosition,
  style,
  data,
  markerEnd,
  selected,
}: EdgeProps<DataWireEdgeType>) {
  const [edgePath, labelX, labelY] = getBezierPath({
    sourceX,
    sourceY,
    sourcePosition,
    targetX,
    targetY,
    targetPosition,
  })
  const viewport = useViewport()
  const portalEl = useContext(WirePortalContext)
  const wire = data?.wire

  const viewContent = (e: React.MouseEvent) => {
    e.stopPropagation()
    data?.onView(id)
  }

  const rename = (e: React.MouseEvent) => {
    e.stopPropagation()
    const name = prompt('Name this data:', wire?.name ?? '')
    if (name === null) return
    api.renameWire(id, name.trim() || null).catch((err) => alert((err as Error).message))
  }

  const screenX = viewport.x + labelX * viewport.zoom
  const screenY = viewport.y + labelY * viewport.zoom

  const overlay = (
    <div
      className="nodrag nopan"
      // React portals bubble events through the *React* tree, not the DOM
      // tree they're physically rendered into -- so a plain click here (this
      // is a React descendant of this edge, even though it's portaled miles
      // away in the DOM) would otherwise reach ReactFlow's onEdgeClick and
      // trigger the "delete this wire?" confirm. Stop it at the source.
      onClick={(e) => e.stopPropagation()}
      style={{
        position: 'absolute',
        left: screenX,
        top: screenY,
        transform: 'translate(-50%, -50%)',
        pointerEvents: 'all',
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        gap: 2,
      }}
    >
      <button
        onClick={viewContent}
        title="View the data on this wire"
        style={{
          width: 20,
          height: 20,
          padding: 0,
          borderRadius: '50%',
          border: selected || data?.isViewed ? '1px solid var(--brand)' : '1px solid #d1d5db',
          background: data?.isViewed ? 'var(--brand-light)' : '#fff',
          fontSize: 11,
          lineHeight: 1,
          cursor: 'pointer',
          boxShadow: '0 1px 2px rgba(0,0,0,0.15)',
        }}
      >
        👁
      </button>
      <span
        onDoubleClick={rename}
        title="Double-click to name this data"
        style={{
          fontSize: 10,
          background: wire?.name ? '#fff' : 'transparent',
          border: wire?.name ? '1px solid #e5e7eb' : 'none',
          borderRadius: 4,
          padding: wire?.name ? '0 4px' : 0,
          color: wire?.name ? '#374151' : '#9ca3af',
          whiteSpace: 'nowrap',
        }}
      >
        {wire?.name ?? '+ name'}
      </span>
    </div>
  )

  return (
    <>
      <BaseEdge id={id} path={edgePath} style={style} markerEnd={markerEnd} />
      {portalEl && createPortal(overlay, portalEl)}
    </>
  )
}
