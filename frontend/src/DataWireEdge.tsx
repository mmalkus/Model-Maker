import { BaseEdge, getBezierPath, type Edge, type EdgeProps } from '@xyflow/react'

export type DataWireEdgeType = Edge<Record<string, never>, 'dataWire'>

// A wire is just a connector from one block's output port to another's
// input -- no interaction of its own. Viewing or naming the data that
// flows through it happens at its source, the output port badge on the
// producing block (see BlockNode's port badges and PortInspector), since
// that's what actually holds the data regardless of how many wires fan out
// from it.
export function DataWireEdge({
  sourceX,
  sourceY,
  targetX,
  targetY,
  sourcePosition,
  targetPosition,
  style,
  markerEnd,
}: EdgeProps<DataWireEdgeType>) {
  const [edgePath] = getBezierPath({
    sourceX,
    sourceY,
    sourcePosition,
    targetX,
    targetY,
    targetPosition,
  })

  return <BaseEdge path={edgePath} style={style} markerEnd={markerEnd} />
}
