import type { Node, NodeProps } from '@xyflow/react'

// Lanes are horizontal rows -- one per modeling phase -- stacked top to
// bottom by `order`, matching how the phases themselves read (Data Prep ->
// Feature Engineering -> ... -> Reporting). Flow within/between lanes also
// runs top-to-bottom (see BlockNode's port positions), so a lane's row
// simply needs to be wide enough to hold whatever's dropped into it.
export const BAND_HEIGHT = 260
export const BAND_COLLAPSED_HEIGHT = 40
export const BAND_X = -2000
export const BAND_WIDTH = 6000

export type LaneBandData = {
  order: number
  collapsed: boolean
}
export type LaneBandNode = Node<LaneBandData, 'laneBand'>

export function laneY(order: number): number {
  return order * BAND_HEIGHT
}

export function LaneBand({ data }: NodeProps<LaneBandNode>) {
  const height = data.collapsed ? BAND_COLLAPSED_HEIGHT : BAND_HEIGHT
  return (
    <div
      style={{
        width: BAND_WIDTH,
        height,
        background: data.order % 2 === 0 ? 'rgba(59,130,246,0.04)' : 'rgba(107,114,128,0.04)',
        borderTop: '1px solid #e5e7eb',
        borderBottom: '1px solid #e5e7eb',
        boxSizing: 'border-box',
        pointerEvents: 'none',
      }}
    />
  )
}
