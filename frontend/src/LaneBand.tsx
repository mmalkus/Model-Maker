import type { Node, NodeProps } from '@xyflow/react'
import type { LaneOut } from './types'

// Lanes are horizontal rows -- one per modeling phase -- stacked top to
// bottom by `order` with no gaps, so they can never overlap by
// construction: each lane's height is user-resizable (drag the border
// between two lanes -- see LaneResizeHandles), and resizing one only shifts
// the ones after it. Flow within/between lanes also runs top-to-bottom (see
// BlockNode's port positions).
export const DEFAULT_LANE_HEIGHT = 260
export const BAND_COLLAPSED_HEIGHT = 40
export const BAND_X = -2000
export const BAND_WIDTH = 6000
export const MIN_LANE_HEIGHT = 80

export interface LaneLayoutEntry {
  id: string
  lane: LaneOut
  top: number
  height: number
}

// Cumulative top/height for each lane in canvas (flow) coordinates, in
// order. A collapsed lane occupies BAND_COLLAPSED_HEIGHT regardless of its
// stored height, so expanding it later doesn't need to touch anyone else.
export function layoutLanes(
  entries: { id: string; lane: LaneOut }[],
  collapsedLanes: Set<string>,
): LaneLayoutEntry[] {
  let y = 0
  return entries.map(({ id, lane }) => {
    const height = collapsedLanes.has(id) ? BAND_COLLAPSED_HEIGHT : lane.height
    const top = y
    y += height
    return { id, lane, top, height }
  })
}

export type LaneBandData = {
  height: number
  collapsed: boolean
  order: number
}
export type LaneBandNode = Node<LaneBandData, 'laneBand'>

export function LaneBand({ data }: NodeProps<LaneBandNode>) {
  const height = data.collapsed ? BAND_COLLAPSED_HEIGHT : data.height
  return (
    <div
      style={{
        width: BAND_WIDTH,
        height,
        background: data.order % 2 === 0 ? 'rgba(127,166,154,0.08)' : 'rgba(107,114,128,0.04)',
        borderTop: '1px solid #e5e7eb',
        borderBottom: '1px solid #e5e7eb',
        boxSizing: 'border-box',
        pointerEvents: 'none',
      }}
    />
  )
}
