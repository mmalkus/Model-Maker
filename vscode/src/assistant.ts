// The parts of the Copilot Chat tools (tools.ts) that don't need VS Code:
// what the backend's /api/assistant/graph returns, and the decisions made
// from it. Unit-tested in test/unit/assistant.test.ts.

export type BlockBrief = {
  id: string
  name: string
  category: string
  lane: string | null
  status: 'grey' | 'green' | 'orange' | 'red' | 'running'
  params: Record<string, unknown>
  inputs: string[]
  outputs: string[]
  position: { x: number; y: number }
  error?: string
  stale_reason?: string | null
}

export type Graph = {
  project: string
  lanes: Array<{ id: string; name: string }>
  blocks: BlockBrief[]
  wires: Array<{ id: string; from: string; to: string; valid: boolean }>
  running: boolean
}

export class ToolInputError extends Error {}

/** A lane given by id or (case-insensitive) name, as models tend to use
 * the name they were shown. */
export function resolveLane(graph: Graph, lane: string | undefined): string | null {
  if (!lane) return null
  const byId = graph.lanes.find((l) => l.id === lane)
  if (byId) return byId.id
  const byName = graph.lanes.find((l) => l.name.toLowerCase() === lane.trim().toLowerCase())
  if (byName) return byName.id
  const known = graph.lanes.map((l) => `${l.name} (${l.id})`).join(', ') || 'none'
  throw new ToolInputError(`no lane "${lane}"; the lanes are: ${known}`)
}

// Roughly a block card's footprint on the canvas.
const COLUMN_WIDTH = 220
const ROW_HEIGHT = 160

/** Where a new block goes: in its lane, right of that lane's rightmost
 * block, or below everything when the lane is empty or not given -- never
 * on top of an existing card. */
export function placeNewBlock(graph: Graph, laneId: string | null): { x: number; y: number } {
  const inLane = laneId ? graph.blocks.filter((b) => b.lane === laneId) : []
  if (inLane.length) {
    const rightmost = inLane.reduce((a, b) => (b.position.x > a.position.x ? b : a))
    return { x: rightmost.position.x + COLUMN_WIDTH, y: rightmost.position.y }
  }
  const bottom = graph.blocks.reduce((y, b) => Math.max(y, b.position.y), -ROW_HEIGHT)
  return { x: 100, y: bottom + ROW_HEIGHT }
}

export function findBlock(graph: Graph, block: string): BlockBrief {
  const match = graph.blocks.find((b) => b.id === block) ?? graph.blocks.find((b) => b.name.toLowerCase() === block.trim().toLowerCase())
  if (!match) throw new ToolInputError(`no block "${block}" -- get the graph for block ids`)
  return match
}

/** What a finished run reports: the target's (or every block's) outcome,
 * with errors, so the model can fix a red block without another call. */
export function runOutcome(graph: Graph, target: string | null): object {
  const blocks = target ? graph.blocks.filter((b) => b.id === target) : graph.blocks
  const counts: Record<string, number> = {}
  for (const b of graph.blocks) counts[b.status] = (counts[b.status] ?? 0) + 1
  const out: Record<string, unknown> = {
    statuses: counts,
    blocks: blocks
      .filter((b) => (target ? true : b.status !== 'green'))
      .map((b) => ({ id: b.id, name: b.name, status: b.status, ...(b.error ? { error: b.error } : {}) })),
  }
  // A cascade doesn't (re)read data sources -- like the AI builder, that's
  // left to the user's Run all / Refresh sources.
  if (blocks.some((b) => /input block .* is not green/.test(b.error ?? ''))) {
    out.next = 'an upstream data source has not been read yet: run the whole pipeline (modelmaker_run without a block), which reads it'
  }
  return out
}
