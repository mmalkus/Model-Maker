import {
  Background,
  Controls,
  Panel,
  ReactFlow,
  ReactFlowProvider,
  applyNodeChanges,
  useReactFlow,
  type Connection,
  type NodeChange,
} from '@xyflow/react'
import '@xyflow/react/dist/style.css'
import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { api } from './api'
import { BlockNode, type BlockFlowNode } from './BlockNode'
import { BuildPanel } from './BuildPanel'
import { DataWireEdge, type DataWireEdgeType } from './DataWireEdge'
import { GHOST_IN, GHOST_OUT, GhostLane, GhostNode, type GhostFlowNode, type GhostLaneNode } from './GhostNode'
import { Inspector } from './Inspector'
import { BAND_X, DEFAULT_LANE_HEIGHT, LaneBand, layoutLanes, type LaneBandNode, type LaneLayoutEntry } from './LaneBand'
import { LaneLabels, LaneResizeHandles } from './LaneLabels'
import { Palette } from './Palette'
import { PortInspector } from './PortInspector'
import { Toolbar } from './Toolbar'
import type { BlockOut, BuildOut, GraphOut, LaneOut, LLMSettingsOut, PortType, RecoveryInfo } from './types'

const nodeTypes = { modelBlock: BlockNode, laneBand: LaneBand, ghostBlock: GhostNode, ghostLane: GhostLane }
const edgeTypes = { dataWire: DataWireEdge }

// Build phases during which the server refuses user edits (see api.py's
// agent canvas lock) -- mirrored here only to say so up front.
const BUILD_LOCKING = new Set(['building', 'awaiting_input', 'final_run'])
const BUILD_TERMINAL = new Set(['done', 'done_with_errors', 'stopped', 'failed', 'discarded'])
const BUILD_BLOCK_TOOLS = new Set(['add_block', 'add_custom_block', 'run_to', 'set_params', 'update_custom_block', 'connect'])

type AiEdge = { id: string; source: string; sourceHandle: string; target: string; targetHandle: string; style: React.CSSProperties; selectable: false }

// How long to sit idle after the last edit before writing the project file.
// Short enough that a save is never more than a few clicks stale, long
// enough that dragging a block across the canvas or stepping through a
// dozen param tweaks coalesces into one write instead of one per edit.
const AUTOSAVE_DEBOUNCE_MS = 2000

export type AutosaveStatus =
  | { state: 'idle' }
  | { state: 'saving' }
  | { state: 'saved'; at: number }
  | { state: 'error'; message: string }

type FlowNode = BlockFlowNode | LaneBandNode | GhostFlowNode | GhostLaneNode

function toBlockNodes(
  graph: GraphOut,
  collapsedLanes: Set<string>,
  onViewPort: (blockId: string, port: string, portType: PortType) => void,
): BlockFlowNode[] {
  return Object.values(graph.blocks)
    .filter((block) => !(block.lane && collapsedLanes.has(block.lane)))
    .map((block) => ({
      id: block.id,
      type: 'modelBlock' as const,
      position: block.position,
      data: { block, onViewPort },
    }))
}

function toEdges(graph: GraphOut): DataWireEdgeType[] {
  return Object.entries(graph.wires).map(([id, w]) => ({
    id,
    type: 'dataWire' as const,
    source: w.from_block,
    sourceHandle: w.from_port,
    target: w.to_block,
    targetHandle: w.to_port,
    style: w.valid ? undefined : { stroke: '#ef4444', strokeDasharray: '4 4' },
    animated: !w.valid,
  }))
}

export default function App() {
  return (
    <ReactFlowProvider>
      <AppInner />
    </ReactFlowProvider>
  )
}

function AppInner() {
  const [graph, setGraph] = useState<GraphOut | null>(null)
  const [blockNodes, setBlockNodes] = useState<BlockFlowNode[]>([])
  const [selectedId, setSelectedId] = useState<string | null>(null)
  const [selectedPort, setSelectedPort] = useState<{ blockId: string; port: string; portType: PortType } | null>(null)
  const [collapsedLanes, setCollapsedLanes] = useState<Set<string>>(new Set())
  const [llmSettings, setLlmSettings] = useState<LLMSettingsOut | null>(null)
  const [recovery, setRecovery] = useState<RecoveryInfo | null>(null)
  const [autosave, setAutosave] = useState<AutosaveStatus>({ state: 'idle' })
  const [buildOpen, setBuildOpen] = useState(false)
  const [build, setBuild] = useState<BuildOut | null>(null)
  const { fitView, screenToFlowPosition } = useReactFlow()
  const didInitialFit = useRef(false)

  const reloadLlmSettings = useCallback(() => {
    api.llmSettings().then(setLlmSettings).catch(() => setLlmSettings(null))
  }, [])

  useEffect(() => {
    reloadLlmSettings()
  }, [reloadLlmSettings])

  // Pick up an AI build that's still going on the server (e.g. after a page
  // reload) -- a live one keeps BuildPanel mounted, which polls it.
  useEffect(() => {
    api
      .agentCurrent()
      .then((s) => setBuild(s.build))
      .catch(() => {})
  }, [])

  const onViewPort = useCallback((blockId: string, port: string, portType: PortType) => {
    setSelectedPort({ blockId, port, portType })
  }, [])

  const reload = useCallback(() => {
    api.graph().then((g) => {
      setGraph(g)
      const nextBlockNodes = toBlockNodes(g, collapsedLanes, onViewPort)
      // Carry over React Flow's measured size and selection: the run poll
      // reloads every 200ms, and fresh nodes without `measured` are hidden
      // until re-measured, which blinks the whole canvas.
      setBlockNodes((prev) => {
        const prevById = new Map(prev.map((n) => [n.id, n]))
        return nextBlockNodes.map((n) => {
          const old = prevById.get(n.id)
          return old ? { ...n, measured: old.measured, selected: old.selected } : n
        })
      })
      // fit the view to the actual blocks (not the oversized lane bands) once,
      // on first load -- re-fitting on every later reload would yank the
      // viewport out from under someone mid-edit.
      if (!didInitialFit.current && nextBlockNodes.length > 0) {
        didInitialFit.current = true
        requestAnimationFrame(() => fitView({ nodes: nextBlockNodes.map((n) => ({ id: n.id })), padding: 0.2 }))
      }
      // A run started in the background (see api.py's _start_background_run)
      // can fail on a precondition (e.g. "Run" clicked on a block whose
      // upstream isn't green) before any block state changes -- surfaced
      // here once, the server clears it as soon as this read happens.
      if (g.run_error) alert(g.run_error)
    })
  }, [collapsedLanes, fitView, onViewPort])

  useEffect(() => {
    reload()
    // Offer to pick up work from a snapshot only when there's nothing open
    // to lose -- with a graph already on the canvas, a recovery prompt is
    // just a way to overwrite the work in front of you.
    api
      .graph()
      .then((g) => (Object.keys(g.blocks).length === 0 ? api.recoveryInfo() : null))
      .then((r) => setRecovery(r?.recovery ?? null))
      .catch(() => setRecovery(null))
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  const undo = useCallback(() => api.undo().then(reload).catch(() => {}), [reload])
  const redo = useCallback(() => api.redo().then(reload).catch(() => {}), [reload])

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (!e.metaKey && !e.ctrlKey) return
      const key = e.key.toLowerCase()
      const isUndo = key === 'z' && !e.shiftKey
      const isRedo = (key === 'z' && e.shiftKey) || key === 'y'
      if (!isUndo && !isRedo) return
      // Text fields and the code editor have their own undo stacks; taking
      // Ctrl-Z away from someone mid-sentence would be worse than not having
      // graph undo at all.
      const target = e.target as HTMLElement | null
      if (
        target &&
        (target.tagName === 'INPUT' ||
          target.tagName === 'TEXTAREA' ||
          target.isContentEditable ||
          target.closest('.cm-editor'))
      ) {
        return
      }
      e.preventDefault()
      if (isUndo) undo()
      else redo()
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [undo, redo])

  // Edits are always snapshotted server-side for crash recovery, and
  // autosave (below) catches up the project file itself within a couple of
  // seconds -- but that write is debounced and best-effort (e.g. it never
  // fires for a project with no save location yet), so leaving mid-debounce
  // or mid-failure is still worth one confirmation.
  useEffect(() => {
    if (!graph?.dirty) return
    const onBeforeUnload = (e: BeforeUnloadEvent) => e.preventDefault()
    window.addEventListener('beforeunload', onBeforeUnload)
    return () => window.removeEventListener('beforeunload', onBeforeUnload)
  }, [graph?.dirty])

  // Autosave: once a project has somewhere to write to, dirty edits write
  // themselves back after a short idle debounce, same target file as a
  // manual Save. This writes the working tree only -- it never stages or
  // commits -- so it doesn't interact with the project's own git history;
  // nothing here composes a commit, only `git commit` (via the Git panel)
  // does that. Unsaved-with-nowhere-to-write (no project_path yet) still
  // relies on crash recovery alone, same as before.
  const autosaveInFlight = useRef(false)
  useEffect(() => {
    if (!graph?.dirty || !graph.project_path) return
    const timer = setTimeout(() => {
      if (autosaveInFlight.current) return
      autosaveInFlight.current = true
      setAutosave({ state: 'saving' })
      api
        .save()
        .then(() => {
          setAutosave({ state: 'saved', at: Date.now() })
          reload()
        })
        .catch((e) => setAutosave({ state: 'error', message: (e as Error).message }))
        .finally(() => {
          autosaveInFlight.current = false
        })
    }, AUTOSAVE_DEBOUNCE_MS)
    return () => clearTimeout(timer)
  }, [graph?.dirty, graph?.project_path, reload])

  // Runs (a single block, a cascade, or a full sweep) execute on the server
  // in the background -- see api.py -- so a block that's still "running"
  // by the time reload() above returns means the run outlived its bounded
  // wait and is genuinely still in flight. Keep polling until nothing is,
  // so status badges (see BlockNode) and the Stop button (see Toolbar)
  // stay live without the user having to manually refresh. sweep_running is
  // included too: between one block's dispatch ending and the next one
  // starting there's a real gap where nothing is individually "running" yet
  // the sweep itself is still going -- without it, polling could stop mid-
  // sweep and the canvas would sit on a half-updated, stale-looking state.
  const anyRunning = useMemo(
    () => (graph ? Object.values(graph.blocks).some((b) => b.status === 'running') || graph.sweep_running : false),
    [graph],
  )
  useEffect(() => {
    if (!anyRunning) return
    // Faster than the steady-state UI needs on its own, so each block's
    // grey -> running -> green/red transition during a Run all/Force run
    // all sweep is actually visible rather than the canvas jumping straight
    // from all-grey to done between two 700ms-apart polls.
    const id = setInterval(reload, 200)
    return () => clearInterval(id)
  }, [anyRunning, reload])

  useEffect(() => {
    if (graph) setBlockNodes(toBlockNodes(graph, collapsedLanes, onViewPort))
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [collapsedLanes])

  const toggleCollapse = useCallback((laneId: string) => {
    setCollapsedLanes((prev) => {
      const next = new Set(prev)
      if (next.has(laneId)) next.delete(laneId)
      else next.add(laneId)
      return next
    })
  }, [])

  const renameLane = useCallback(
    (laneId: string, name: string) => {
      if (!graph) return
      const current = graph.lanes[laneId]
      api.upsertLane(laneId, name, current?.order ?? 0).then(reload)
    },
    [graph, reload],
  )

  const deleteLane = useCallback(
    (laneId: string) => {
      if (confirm('Delete this lane? Blocks in it become unassigned, not deleted.')) {
        api.deleteLane(laneId).then(reload)
      }
    },
    [reload],
  )

  const moveLane = useCallback(
    (laneId: string, direction: -1 | 1) => {
      if (!graph) return
      const entries = Object.entries(graph.lanes).sort((a, b) => a[1].order - b[1].order)
      const idx = entries.findIndex(([id]) => id === laneId)
      const swapIdx = idx + direction
      if (idx < 0 || swapIdx < 0 || swapIdx >= entries.length) return
      const [aId, aLane] = entries[idx]
      const [bId, bLane] = entries[swapIdx]
      Promise.all([api.upsertLane(aId, aLane.name, bLane.order), api.upsertLane(bId, bLane.name, aLane.order)]).then(
        reload,
      )
    },
    [graph, reload],
  )

  const addLane = useCallback(
    (name: string) => {
      if (!graph) return
      const nextOrder = Object.values(graph.lanes).reduce((max, l) => Math.max(max, l.order + 1), 0)
      api.upsertLane(`lane_${Date.now().toString(36)}`, name, nextOrder, DEFAULT_LANE_HEIGHT).then(reload)
    },
    [graph, reload],
  )

  const resizeLane = useCallback(
    (laneId: string, height: number) => {
      if (!graph) return
      const current = graph.lanes[laneId]
      if (!current) return
      api.upsertLane(laneId, current.name, current.order, height).then(reload)
    },
    [graph, reload],
  )

  const laneEntries: { id: string; lane: LaneOut }[] = useMemo(() => {
    if (!graph) return []
    return Object.entries(graph.lanes)
      .sort((a, b) => a[1].order - b[1].order)
      .map(([id, lane]) => ({ id, lane }))
  }, [graph])

  // Cumulative pixel layout: lanes stack end-to-end by `order` with no gaps,
  // so resizing one (see resizeLane) only ever shifts the ones after it --
  // they can never overlap.
  const laneLayout: LaneLayoutEntry[] = useMemo(
    () => layoutLanes(laneEntries, collapsedLanes),
    [laneEntries, collapsedLanes],
  )

  const laneNodes: LaneBandNode[] = useMemo(
    () =>
      laneLayout.map(({ id, lane, top, height }) => ({
        id: `laneband_${id}`,
        type: 'laneBand' as const,
        position: { x: BAND_X, y: top },
        draggable: false,
        selectable: false,
        zIndex: -10,
        data: { order: lane.order, height, collapsed: collapsedLanes.has(id) },
      })),
    [laneLayout, collapsedLanes],
  )

  // ---- AI build overlays (see BuildPanel / agent-builder-proposal.md §4.2) --
  // While a plan awaits review: its steps as ghost blocks (plus ghost bands
  // for lanes it would create) and its wiring as dashed edges. While
  // building: a highlight on whichever block the AI last touched.
  const reviewPlan = build?.phase === 'awaiting_approval' ? build.plan : null
  const ghostNodes: (GhostFlowNode | GhostLaneNode)[] = useMemo(() => {
    if (!reviewPlan) return []
    const lanes: GhostLaneNode[] = Object.entries(reviewPlan.lane_layout ?? {}).map(([key, l]) => ({
      id: `ghostlane_${key}`,
      type: 'ghostLane' as const,
      position: { x: BAND_X, y: l.top },
      draggable: false,
      selectable: false,
      zIndex: -9,
      data: { name: l.name, height: l.height },
    }))
    const steps: GhostFlowNode[] = reviewPlan.steps
      .filter((s) => reviewPlan.layout?.[s.ref])
      .map((s) => ({
        id: `ghost_${s.ref}`,
        type: 'ghostBlock' as const,
        position: { x: reviewPlan.layout[s.ref].x, y: reviewPlan.layout[s.ref].y },
        draggable: false,
        selectable: false,
        data: { step: s, highlighted: false },
      }))
    return [...lanes, ...steps]
  }, [reviewPlan])

  const ghostEdges: AiEdge[] = useMemo(() => {
    if (!reviewPlan) return []
    const refs = new Set(reviewPlan.steps.map((s) => s.ref))
    return reviewPlan.steps.flatMap((s) =>
      s.inputs.map((inp, i) => ({
        id: `ghostedge_${s.ref}_${i}`,
        source: refs.has(inp.from) ? `ghost_${inp.from}` : inp.from,
        sourceHandle: refs.has(inp.from) ? GHOST_OUT : inp.from_port,
        target: `ghost_${s.ref}`,
        targetHandle: GHOST_IN,
        style: { stroke: '#a78bfa', strokeDasharray: '6 4' },
        selectable: false as const,
      })),
    )
  }, [reviewPlan])

  const aiActiveBlock = useMemo(() => {
    if (!build || !BUILD_LOCKING.has(build.phase)) return null
    for (let i = build.events.length - 1; i >= 0; i--) {
      const e = build.events[i]
      if (e.kind === 'tool' && BUILD_BLOCK_TOOLS.has(String(e.tool))) {
        const args = (e.args ?? {}) as Record<string, unknown>
        const result = (e.result ?? {}) as Record<string, unknown>
        return String(args.block ?? args.to_block ?? result.block ?? '')
      }
    }
    return null
  }, [build])

  const plannedChanges = useMemo(() => {
    const m = new Map<string, string>()
    for (const c of reviewPlan?.changes_to_existing ?? []) m.set(c.block, m.has(c.block) ? `${m.get(c.block)}; ${c.change}` : c.change)
    return m
  }, [reviewPlan])

  const overlaidBlockNodes = useMemo(
    () =>
      blockNodes.map((n) =>
        n.id === aiActiveBlock || plannedChanges.has(n.id)
          ? { ...n, data: { ...n.data, aiActive: n.id === aiActiveBlock, aiPlannedChange: plannedChanges.get(n.id) ?? null } }
          : n,
      ),
    [blockNodes, aiActiveBlock, plannedChanges],
  )

  // Lane bands and ghosts are derived (memoized) rather than held in state,
  // so React Flow's measured sizes for them would otherwise be dropped --
  // and in controlled mode an unmeasured node stays `visibility: hidden`
  // and can't anchor edges. Keep their measurements here and hand them back.
  const [derivedMeasured, setDerivedMeasured] = useState<Record<string, { width: number; height: number }>>({})
  const nodes: FlowNode[] = useMemo(
    () => [
      ...laneNodes.map((n) => (derivedMeasured[n.id] ? { ...n, measured: derivedMeasured[n.id] } : n)),
      ...overlaidBlockNodes,
      ...ghostNodes.map((n) => (derivedMeasured[n.id] ? { ...n, measured: derivedMeasured[n.id] } : n)),
    ],
    [laneNodes, overlaidBlockNodes, ghostNodes, derivedMeasured],
  )

  const onNodesChange = useCallback((changes: NodeChange<FlowNode>[]) => {
    const isDerived = (id: string) => id.startsWith('laneband_') || id.startsWith('ghost')
    const derived = changes.filter((c) => c.type === 'dimensions' && isDerived(c.id) && c.dimensions)
    if (derived.length > 0) {
      setDerivedMeasured((prev) => {
        const next = { ...prev }
        for (const c of derived) if (c.type === 'dimensions' && c.dimensions) next[c.id] = c.dimensions
        return next
      })
    }
    const blockChanges = changes.filter((c) => !('id' in c) || !isDerived(c.id))
    if (blockChanges.length > 0) setBlockNodes((nds) => applyNodeChanges(blockChanges as NodeChange<BlockFlowNode>[], nds))
  }, [])

  const selectedIds = useMemo(() => {
    const ids = blockNodes.filter((n) => n.selected).map((n) => n.id)
    return ids.length > 0 ? ids : selectedId ? [selectedId] : []
  }, [blockNodes, selectedId])

  const buildLive = build != null && !BUILD_TERMINAL.has(build.phase)
  const canvasLocked = build != null && BUILD_LOCKING.has(build.phase)

  const laneForY = useCallback(
    (y: number): string | null => {
      const hit = laneLayout.find((l) => y >= l.top && y < l.top + l.height)
      return hit ? hit.id : null
    },
    [laneLayout],
  )

  const onNodeDragStop = useCallback(
    (_: unknown, node: FlowNode) => {
      if (node.type !== 'modelBlock' || !graph) return
      const newLane = laneForY(node.position.y)
      const currentLane = graph.blocks[node.id]?.lane ?? null
      const patch: Parameters<typeof api.updateBlock>[1] = { position: node.position }
      if (newLane !== currentLane) patch.lane = newLane
      api.updateBlock(node.id, patch).then(reload).catch(console.error)
    },
    [graph, reload, laneForY],
  )

  const onConnect = useCallback(
    (conn: Connection) => {
      if (!conn.source || !conn.target || !conn.sourceHandle || !conn.targetHandle) return
      api
        .createWire({
          from_block: conn.source,
          from_port: conn.sourceHandle,
          to_block: conn.target,
          to_port: conn.targetHandle,
        })
        .then(reload)
        .catch((e) => alert(e.message))
    },
    [reload],
  )

  const onEdgeClick = useCallback(
    (_: unknown, edge: DataWireEdgeType) => {
      if (edge.id.startsWith('ghostedge_')) return
      if (confirm('Delete this wire?')) {
        api.deleteWire(edge.id).then(reload)
      }
    },
    [reload],
  )

  const addBlock = useCallback(
    (category: string, at?: { x: number; y: number }) => {
      const pos = at ?? { x: 80 + Math.random() * 200, y: 80 + Math.random() * 200 }
      api
        .createBlock({ category, x: pos.x, y: pos.y })
        .then((b) => {
          reload()
          setSelectedId(b.id)
          setSelectedPort(null)
        })
        .catch((e) => alert(e.message))
    },
    [reload],
  )

  const addCustomBlock = useCallback(
    (blockType: 'input' | 'standard' | 'output', at?: { x: number; y: number }) => {
      // No name prompt -- create a blank block immediately and let the user
      // describe it via "Draft with AI" right away; rename later by clicking
      // the block's name in the inspector. An input block gets no `df`
      // parameter (it's a pipeline source, like read_csv); standard/output
      // both start as a df -> df passthrough the user redrafts from there.
      const fnName = `ai_block_${Date.now().toString(36)}`
      const isSource = blockType === 'input'
      const pos = at ?? { x: 80 + Math.random() * 200, y: 80 + Math.random() * 200 }
      api
        .createBlock({
          category: fnName,
          block_type: blockType,
          name: `New AI ${blockType} block`,
          x: pos.x,
          y: pos.y,
          inputs: isSource ? [] : [{ name: 'df', type: 'dataframe' }],
          outputs: [{ name: 'out', type: 'dataframe' }],
          code: isSource ? `def ${fnName}():\n    return pl.DataFrame()\n` : `def ${fnName}(df):\n    return df\n`,
          metadata_transform: { kind: isSource ? 'infer_dtypes' : 'passthrough' },
        })
        .then((b) => {
          reload()
          setSelectedId(b.id)
          setSelectedPort(null)
        })
        .catch((e) => alert(e.message))
    },
    [reload],
  )

  const onDragOver = useCallback((e: React.DragEvent) => {
    if (!e.dataTransfer.types.includes('application/x-modelmaker-block')) return
    e.preventDefault()
    e.dataTransfer.dropEffect = 'copy'
  }, [])

  const onDrop = useCallback(
    (e: React.DragEvent) => {
      const raw = e.dataTransfer.getData('application/x-modelmaker-block')
      if (!raw) return
      e.preventDefault()
      const at = screenToFlowPosition({ x: e.clientX, y: e.clientY })
      const dropped = JSON.parse(raw) as { kind: 'registry'; category: string } | { kind: 'custom'; blockType: 'input' | 'standard' | 'output' }
      if (dropped.kind === 'registry') addBlock(dropped.category, at)
      else addCustomBlock(dropped.blockType, at)
    },
    [screenToFlowPosition, addBlock, addCustomBlock],
  )

  const selectedBlock: BlockOut | null = graph && selectedId ? (graph.blocks[selectedId] ?? null) : null

  return (
    <div style={{ display: 'flex', flexDirection: 'column', height: '100vh' }}>
      <Toolbar
        onChanged={reload}
        projectPath={graph?.project_path ?? null}
        llmSettings={llmSettings}
        onLlmSettingsChange={setLlmSettings}
        running={anyRunning}
        dirty={graph?.dirty ?? false}
        canUndo={graph?.can_undo ?? false}
        canRedo={graph?.can_redo ?? false}
        onUndo={undo}
        onRedo={redo}
        sampleRows={graph?.sample_rows ?? null}
        autosave={autosave}
      />
      {graph?.sample_rows != null && (
        <div
          style={{
            display: 'flex',
            alignItems: 'center',
            gap: 8,
            padding: '5px 12px',
            background: '#fef3c7',
            borderBottom: '1px solid #fde68a',
            color: '#92400e',
            fontSize: 12,
          }}
        >
          <strong>Sample mode</strong>
          <span>
            every source is capped at the first {graph.sample_rows.toLocaleString()} rows — results are for
            iterating on, not for reporting.
          </span>
          <span style={{ flex: 1 }} />
          <button onClick={() => api.setSampleMode(null).then(reload)}>Use full data</button>
        </div>
      )}
      {recovery && (
        <div
          style={{
            display: 'flex',
            alignItems: 'center',
            gap: 8,
            padding: '5px 12px',
            background: 'var(--brand-light)',
            borderBottom: '1px solid var(--brand-border)',
            fontSize: 12,
          }}
        >
          <strong>Unsaved work found</strong>
          <span>
            {recovery.block_count} block{recovery.block_count === 1 ? '' : 's'}
            {recovery.project_name ? ` from "${recovery.project_name}"` : ''}
            {recovery.saved_at ? `, last changed ${new Date(recovery.saved_at).toLocaleString()}` : ''}.
          </span>
          <span style={{ flex: 1 }} />
          <button
            className="brand-primary"
            onClick={() => {
              api
                .recover()
                .then(reload)
                .then(() => setRecovery(null))
                .catch((e) => alert(e.message))
            }}
          >
            Restore
          </button>
          <button onClick={() => setRecovery(null)}>Dismiss</button>
        </div>
      )}
      <div style={{ display: 'flex', flex: 1, minHeight: 0 }}>
        <Palette onAdd={addBlock} onAddCustom={addCustomBlock} onAddLane={addLane} />
        <div style={{ flex: 1 }} onDragOver={onDragOver} onDrop={onDrop}>
          <ReactFlow
            nodes={nodes}
            edges={graph ? [...toEdges(graph), ...(ghostEdges as unknown as DataWireEdgeType[])] : []}
            nodeTypes={nodeTypes}
            edgeTypes={edgeTypes}
            onNodesChange={onNodesChange}
            onNodeDragStop={onNodeDragStop}
            onConnect={onConnect}
            onEdgeClick={onEdgeClick}
            onNodeClick={(_, node) => {
              if (node.type !== 'modelBlock') return
              setSelectedId(node.id)
              setSelectedPort(null)
            }}
            onPaneClick={() => {
              setSelectedId(null)
              setSelectedPort(null)
            }}
            proOptions={{ hideAttribution: true }}
          >
            <Background />
            <Controls />
            <LaneLabels
              lanes={laneLayout}
              collapsedLanes={collapsedLanes}
              onToggleCollapse={toggleCollapse}
              onRename={renameLane}
              onDelete={deleteLane}
              onMove={moveLane}
            />
            <LaneResizeHandles lanes={laneLayout} collapsedLanes={collapsedLanes} onResize={resizeLane} />
            <Panel position="top-right" style={{ display: 'flex', gap: 6, alignItems: 'center' }}>
              {canvasLocked && (
                <span
                  title="While the AI builds, the graph is read-only for you -- stop the build to edit"
                  style={{ background: '#ede9fe', color: '#5b21b6', borderRadius: 4, padding: '3px 8px', fontSize: 12 }}
                >
                  AI is editing -- canvas read-only
                </span>
              )}
              <button
                className={buildOpen ? undefined : 'brand-primary'}
                onClick={() => setBuildOpen((o) => !o)}
                title="Have an AI plan and build a model from the selected blocks"
              >
                {buildLive && !buildOpen ? 'AI build in progress…' : 'Build with AI'}
              </button>
            </Panel>
            <Panel
              position="bottom-right"
              style={{
                display: 'flex',
                gap: 10,
                fontSize: 11,
                color: 'var(--brand-charcoal, #6b7280)',
                background: 'rgba(255,255,255,0.85)',
                padding: '3px 8px',
                borderRadius: 4,
              }}
            >
              <a href="https://www.quantology.nl/" target="_blank" rel="noreferrer">
                Quantology
              </a>
              <a href="https://github.com/mmalkus/Model-Maker" target="_blank" rel="noreferrer">
                GitHub
              </a>
              <a href="https://www.apache.org/licenses/LICENSE-2.0" target="_blank" rel="noreferrer">
                Apache 2.0 License
              </a>
            </Panel>
          </ReactFlow>
        </div>
        {graph && (buildOpen || buildLive) && (
          // Kept mounted (just hidden) while a build is live, so it keeps
          // polling the build -- and the canvas -- with the panel closed.
          <div style={{ display: buildOpen ? 'flex' : 'none' }}>
            <BuildPanel
              graph={graph}
              selectedIds={selectedIds}
              llmSettings={llmSettings}
              onChanged={reload}
              onBuild={setBuild}
              onClose={() => setBuildOpen(false)}
            />
          </div>
        )}
        {buildOpen ? null : selectedPort && graph ? (
          <PortInspector
            blockId={selectedPort.blockId}
            port={selectedPort.port}
            portType={selectedPort.portType}
            graph={graph}
            onClose={() => setSelectedPort(null)}
            onChanged={reload}
          />
        ) : (
          <Inspector block={selectedBlock} onChanged={reload} provider={llmSettings?.active_provider ?? null} />
        )}
      </div>
    </div>
  )
}
