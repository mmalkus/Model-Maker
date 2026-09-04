import { BaseEdge, getBezierPath, useViewport, type Edge, type EdgeProps } from '@xyflow/react'
import { createContext, useContext, useEffect, useState } from 'react'
import { createPortal } from 'react-dom'
import { api } from './api'
import { DataModal } from './DataModal'
import type { PortType, PreviewOut, WireOut } from './types'

export type DataWireData = {
  wire: WireOut
  portType: PortType | undefined
  fromLabel: string
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
// (whatever shape it is: a dataframe preview, an image, or a plain value),
// and the wire can be given its own name, shown right there on the canvas.
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
  const [open, setOpen] = useState(false)
  const wire = data?.wire

  const viewContent = (e: React.MouseEvent) => {
    e.stopPropagation()
    setOpen(true)
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
          border: selected ? '1px solid var(--brand)' : '1px solid #d1d5db',
          background: '#fff',
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
      {open && wire && (
        <WireDataModal
          title={wire.name || data?.fromLabel || `${wire.from_block}:${wire.from_port}`}
          blockId={wire.from_block}
          port={wire.from_port}
          portType={data?.portType}
          onClose={() => setOpen(false)}
        />
      )}
    </div>
  )

  return (
    <>
      <BaseEdge id={id} path={edgePath} style={style} markerEnd={markerEnd} />
      {portalEl && createPortal(overlay, portalEl)}
    </>
  )
}

function WireDataModal({
  title,
  blockId,
  port,
  portType,
  onClose,
}: {
  title: string
  blockId: string
  port: string
  portType: PortType | undefined
  onClose: () => void
}) {
  const [preview, setPreview] = useState<PreviewOut | null>(null)
  const [value, setValue] = useState<{ v: unknown } | null>(null)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    let cancelled = false
    if (portType === 'image') return
    const load =
      portType === 'dataframe'
        ? api.preview(blockId, { port, rows: 200, summary: true }).then((p) => {
            if (!cancelled) setPreview(p)
          })
        : api.value(blockId, port).then((v) => {
            if (!cancelled) setValue({ v })
          })
    load.catch((e) => {
      if (!cancelled) setError((e as Error).message)
    })
    return () => {
      cancelled = true
    }
  }, [blockId, port, portType])

  if (portType === 'dataframe') {
    return preview ? (
      <DataModal blockName={title} preview={preview} onClose={onClose} />
    ) : (
      <ModalShell title={title} onClose={onClose}>
        {error ?? 'Loading...'}
      </ModalShell>
    )
  }

  if (portType === 'image') {
    return (
      <ModalShell title={title} onClose={onClose}>
        <img src={api.imageUrl(blockId, port)} alt={title} style={{ maxWidth: '100%', borderRadius: 6 }} />
      </ModalShell>
    )
  }

  return (
    <ModalShell title={title} onClose={onClose}>
      {error ? (
        <span style={{ color: '#b91c1c' }}>{error}</span>
      ) : (
        <pre style={{ margin: 0, fontSize: 12, whiteSpace: 'pre-wrap' }}>
          {value ? JSON.stringify(value.v, null, 2) : 'Loading...'}
        </pre>
      )}
    </ModalShell>
  )
}

function ModalShell({ title, onClose, children }: { title: string; onClose: () => void; children: React.ReactNode }) {
  return (
    <div
      // The wire-controls portal container (see WirePortalContext) sets
      // pointer-events: none so it doesn't block clicks/drags elsewhere on
      // the canvas -- anything interactive inside it, this modal included,
      // has to opt back in explicitly or clicks fall through to the canvas.
      className="nodrag nopan"
      style={{
        position: 'fixed',
        inset: '20% 30%',
        minWidth: 320,
        maxHeight: '60vh',
        overflow: 'auto',
        background: '#fff',
        border: '1px solid #d1d5db',
        borderRadius: 8,
        padding: 16,
        zIndex: 50,
        pointerEvents: 'all',
        boxShadow: '0 10px 30px rgba(0,0,0,0.25)',
      }}
    >
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'baseline', marginBottom: 8 }}>
        <strong style={{ fontSize: 14 }}>{title}</strong>
        <button onClick={onClose}>Close</button>
      </div>
      {children}
    </div>
  )
}
