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
  data_source?: boolean
  library_block?: boolean
  // A placed library block whose file changed since: it still runs runs_version.
  outdated?: { file: string; runs_version: number; library_version: number }
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

export function findBlock(graph: Graph, block: string): BlockBrief {
  const match = graph.blocks.find((b) => b.id === block) ?? graph.blocks.find((b) => b.name.toLowerCase() === block.trim().toLowerCase())
  if (!match) throw new ToolInputError(`no block "${block}" -- get the graph for block ids`)
  return match
}

/** The wire to remove: by id, or the one feeding an input port (an input
 * takes a single wire). */
export function findWire(graph: Graph, wire: { id?: string; to_block?: string; to_port?: string }): Graph['wires'][number] {
  if (wire.id) {
    const byId = graph.wires.find((w) => w.id === wire.id)
    if (!byId) throw new ToolInputError(`no wire "${wire.id}" -- get the graph for wire ids`)
    return byId
  }
  if (!wire.to_block || !wire.to_port) throw new ToolInputError('give the wire id, or to_block and to_port')
  const to = `${findBlock(graph, wire.to_block).id}.${wire.to_port}`
  const match = graph.wires.find((w) => w.to === to)
  if (!match) throw new ToolInputError(`nothing is wired into ${to}`)
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
  // A placed library block keeps the code it was placed with: say so, or a
  // long run silently produces numbers from an old version (#93).
  const outdated = graph.blocks.filter((b) => b.outdated).map((b) => ({ id: b.id, name: b.name, ...b.outdated }))
  if (outdated.length) {
    out.outdated_blocks = outdated
    out.outdated_note = OUTDATED_NOTE
  }
  // A cascade doesn't (re)read data sources -- like the AI builder, that's
  // left to the user's Run all / Refresh sources.
  // A block whose output didn't fit in memory (the engine's collect guard
  // stopped it, or its worker died): running it again fails the same way.
  if (blocks.some((b) => /too big to hold in memory|likely out of memory/.test(b.error ?? ''))) {
    out.next = OUT_OF_MEMORY_NEXT
  } else if (blocks.some((b) => /input block .* is not green/.test(b.error ?? ''))) {
    out.next = 'an upstream data source has not been read yet: run the whole pipeline (modelmaker_run without a block), which reads it'
  }
  return out
}

// Same wording as assistant_mcp.OUTDATED_NOTE.
const OUTDATED_NOTE =
  'these placed library blocks still have the code they were placed with, which is not what their library file ' +
  "has now (runs_version and library_version are the version numbers each says): to use the file's code, update " +
  'each with update_block (apply: true), then run again'

// Same wording as assistant_mcp.OUT_OF_MEMORY_NEXT.
const OUT_OF_MEMORY_NEXT =
  "a block's output is too big for this machine's memory -- running it again the same way fails the same way. " +
  'First, mark the dataframe input of the block it feeds "lazy": true (for a library block, in its file, then ' +
  'check_block_file) and keep that block polars expressions without a collect, then run the whole pipeline with ' +
  'streaming: true, so the two run as one streamed query that never holds the full table. Or reduce it inside the ' +
  "block that produces it: select only the needed columns, filter to the population, sample whole ids or " +
  "aggregate to the model's grain before returning (for big zipped or plain CSVs, read them through Stage CSV as " +
  'Parquet rather than a staging script of your own). Turn on sample mode while building (a streaming run needs ' +
  'it off).'
