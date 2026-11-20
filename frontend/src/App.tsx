import {
  Background,
  Controls,
  Panel,
  ReactFlow,
  ReactFlowProvider,
  applyNodeChanges,
  useReactFlow,
  useStoreApi,
  type Connection,
  type NodeChange,
} from '@xyflow/react'
import '@xyflow/react/dist/style.css'
import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { api } from './api'
import { BlockNode, type BlockFlowNode } from './BlockNode'
import { BuildPanel } from './BuildPanel'
import { ReportModal } from './BuildReport'
import { DataWireEdge, type DataWireEdgeType } from './DataWireEdge'
import { GHOST_IN, GHOST_OUT, GhostLane, GhostNode, type GhostFlowNode, type GhostLaneNode } from './GhostNode'
import { Inspector } from './Inspector'
import { BAND_X, DEFAULT_LANE_HEIGHT, LaneBand, layoutLanes, type LaneBandNode, type LaneLayoutEntry } from './LaneBand'
import { LaneLabels, LaneResizeHandles } from './LaneLabels'
import { LibraryPanel, type LibraryView } from './LibraryPanel'
import { Palette, type DragPayload } from './Palette'
import { UpdateDialog } from './UpdateDialog'
import { PortInspector } from './PortInspector'
import { Toolbar } from './Toolbar'
import type { BlockOut, BuildOut, GraphOut, LaneOut, LLMSettingsOut, PortType, RecoveryInfo } from './types'
import { host } from './host'
import { has } from './features'

const STAMP_POLL_MS = 1500
// How long the view stops following new and running blocks after the user
// pans or zooms it themselves.
const FOLLOW_PAUSE_MS = 5000

const nodeTypes = { modelBlock: BlockNode, laneBand: LaneBand, ghostBlock: GhostNode, ghostLane: GhostLane }
const edgeTypes = { dataWire: DataWireEdge }

// Build phases during which the server refuses user edits (see api.py's
// agent canvas lock) -- mirrored here only to say so up front.
const BUILD_LOCKING = new Set(['building', 'awaiting_input', 'awaiting_stage_review', 'final_run'])
const BUILD_TERMINAL = new Set(['done', 'done_with_errors', 'stopped', 'failed', 'discarded'])
const BUILD_BLOCK_TOOLS = new Set(['add_block', 'create_library_block', 'run_to', 'set_params', 'update_library_block', 'connect'])

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
  onOpenBuildReport: (buildId: string) => void,
): BlockFlowNode[] {
  return Object.values(graph.blocks)
    .filter((block) => !(block.lane && collapsedLanes.has(block.lane)))
    .map((block) => ({
      id: block.id,
      type: 'modelBlock' as const,
      position: block.position,
      data: { block, onViewPort, onOpenBuildReport },
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
  // The block library overlay (null: closed), the counter the palette
  // refetches the library on, and the block whose update dialog is open.
  const [libraryView, setLibraryView] = useState<LibraryView | null>(null)
  // In VS Code the library is an editor tab of its own (vscode/src/library.ts);
  // in the browser, a dialog over the canvas.
  const openLibrary = useCallback((view: LibraryView) => {
    if (host.kind === 'vscode') host.post('openLibrary', view)
    else setLibraryView(view)
  }, [])
  const [libraryVersion, setLibraryVersion] = useState(0)
  const [updateBlockId, setUpdateBlockId] = useState<string | null>(null)
  const [selectedPort, setSelectedPort] = useState<{ blockId: string; port: string; portType: PortType } | null>(null)
  const [collapsedLanes, setCollapsedLanes] = useState<Set<string>>(new Set())
  const [llmSettings, setLlmSettings] = useState<LLMSettingsOut | null>(null)
  const [recovery, setRecovery] = useState<RecoveryInfo | null>(null)
  const [autosave, setAutosave] = useState<AutosaveStatus>({ state: 'idle' })
  const [buildOpen, setBuildOpen] = useState(false)
  // Inside VS Code the editor tab is narrow, so the palette can fold to a
  // rail (folded to start with; the choice is remembered), the inspector
  // shows only for a selection, and the footer links go.
  const lean = host.kind === 'vscode'
  const [paletteOpen, setPaletteOpen] = useState(() => !lean || host.getState('paletteOpen') === '1')
  const togglePalette = useCallback(() => {
    setPaletteOpen((open) => {
      host.setState('paletteOpen', open ? '0' : '1')
      return !open
    })
  }, [])
  const [build, setBuild] = useState<BuildOut | null>(null)
  // The AI build whose full report is open (see BuildReport.ReportModal).
  const [reportBuildId, setReportBuildId] = useState<string | null>(null)
  const { fitView, getNodesBounds, getZoom, screenToFlowPosition, setCenter } = useReactFlow()
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
    if (!has('ai_builder')) return
    api
      .agentCurrent()
      .then((s) => setBuild(s.build))
      .catch(() => {})
  }, [])

  const onViewPort = useCallback((blockId: string, port: string, portType: PortType) => {
    setSelectedPort({ blockId, port, portType })
  }, [])

  // Following changes made elsewhere (an assistant adding blocks, a run):
  // the ids on the canvas and the blocks running at the last load, and when
  // the user last moved the view themselves -- following never fights a
  // pan or zoom they just made.
  const knownIds = useRef<Set<string> | null>(null)
  const runningIds = useRef<Set<string>>(new Set())
  const userMovedAt = useRef(0)
  const flowStore = useStoreApi()

  // Shows `ids` (zooming in no further than now when `keepZoom`, else to
  // fit). When they can't all fit even fully zoomed out, zooms out as far
  // as the canvas allows and scrolls to `focus` instead.
  const followView = useCallback(
    (ids: string[], focus: string, keepZoom: boolean) => {
      const { width, height, minZoom } = flowStore.getState()
      const padding = 0.25
      const bounds = getNodesBounds(ids)
      if (!width || !height || !bounds.width || !bounds.height) return
      const fitZoom = Math.min(width / (bounds.width * (1 + 2 * padding)), height / (bounds.height * (1 + 2 * padding)))
      if (fitZoom >= minZoom) {
        fitView({ nodes: ids.map((id) => ({ id })), padding, maxZoom: keepZoom ? getZoom() : undefined, duration: 400 })
        return
      }
      const f = getNodesBounds([focus])
      setCenter(f.x + f.width / 2, f.y + f.height / 2, { zoom: minZoom, duration: 400 })
    },
    [flowStore, fitView, getNodesBounds, getZoom, setCenter],
  )

  const apply = useCallback(
    (g: GraphOut, follow: boolean) => {
        setGraph(g)
        const nextBlockNodes = toBlockNodes(g, collapsedLanes, onViewPort, setReportBuildId)
        const ids = new Set(Object.keys(g.blocks))
        const added = knownIds.current ? [...ids].filter((id) => !knownIds.current!.has(id)) : []
        knownIds.current = ids
        const running = new Set(Object.entries(g.blocks).filter(([, b]) => b.status === 'running').map(([id]) => id))
        const prevRunning = runningIds.current
        const startedRunning = [...running].some((id) => !prevRunning.has(id))
        runningIds.current = running
        const userBusy = Date.now() - userMovedAt.current < FOLLOW_PAUSE_MS
        if (follow && didInitialFit.current && !userBusy && (added.length > 0 || startedRunning)) {
          // New blocks: zoom out to the whole graph, focused on the newest.
          // A run: bring the running blocks into view, panning, and zooming
          // out only if they don't fit -- focused on the one that just started.
          const focus = added.length > 0 ? added[added.length - 1] : [...running].find((id) => !prevRunning.has(id))!
          // After the new nodes are measured, or the bounds miss them.
          setTimeout(() => followView(added.length > 0 ? [...ids] : [...running], focus, added.length === 0), 80)
        }
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
        if (g.run_error) host.notify(g.run_error)
    },
    [collapsedLanes, fitView, followView, onViewPort],
  )

  // One graph read at a time: a reload asked for while one is in flight
  // runs once it's back (so it sees the newer state), however many were
  // asked for meanwhile. Polls that pile up behind a slow or stuck backend
  // would otherwise take every connection the browser allows it and freeze
  // the canvas on its last state -- "running" forever. A read that fails
  // (timed out, backend gone) is retried by the next poll; after a few in
  // a row the canvas says it's reconnecting.
  const inFlight = useRef<Promise<void> | null>(null)
  const queued = useRef<{ follow: boolean; promise: Promise<void>; resolve: () => void } | null>(null)
  const failures = useRef(0)
  const [reconnecting, setReconnecting] = useState(false)
  const loadRef = useRef<(follow: boolean) => Promise<void>>(() => Promise.resolve())
  const load = useCallback(
    (follow: boolean): Promise<void> => {
      if (inFlight.current) {
        if (queued.current) {
          queued.current.follow ||= follow
          return queued.current.promise
        }
        let resolve!: () => void
        const promise = new Promise<void>((r) => (resolve = r))
        queued.current = { follow, promise, resolve }
        return promise
      }
      const p = api
        .graph()
        .then(
          (g) => {
            failures.current = 0
            setReconnecting(false)
            apply(g, follow)
          },
          () => {
            failures.current += 1
            if (failures.current >= 3) setReconnecting(true)
          },
        )
        .finally(() => {
          inFlight.current = null
          const next = queued.current
          queued.current = null
          if (next) loadRef.current(next.follow).then(next.resolve)
        })
      inFlight.current = p
      return p
    },
    [apply],
  )
  loadRef.current = load
  // After the canvas's own edits: the view stays where the user put it.
  const reload = useCallback(() => load(false), [load])
  // After changes from elsewhere (an assistant, a run, another tab): the
  // view follows new and running blocks.
  const reloadFollow = useCallback(() => load(true), [load])

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

  // VS Code (see host.ts): its Save and Revert commands, and a reload after
  // the project file changed on disk (git checkout, another editor).
  const projectPathRef = useRef<string | null>(null)
  projectPathRef.current = graph?.project_path ?? null
  useEffect(
    () =>
      host.onEvent(({ name }) => {
        const path = projectPathRef.current
        if (name === 'save' && path)
          api
            .save()
            .then(reload)
            .catch((e) => host.notify((e as Error).message))
        else if (name === 'reload' && path) api.load(path).then(reload).catch((e) => host.notify((e as Error).message))
        // The backend's graph changed underneath (a Copilot Chat tool edit).
        else if (name === 'refresh') reloadFollow()
      }),
    [reload, reloadFollow],
  )
  useEffect(() => {
    if (host.kind !== 'vscode') return
    const onKey = (e: KeyboardEvent) => {
      if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === 's' && projectPathRef.current) {
        e.preventDefault()
        api
          .save()
          .then(reload)
          .catch((err) => host.notify((err as Error).message))
      }
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [reload])

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
    const id = setInterval(reloadFollow, 200)
    return () => clearInterval(id)
  }, [anyRunning, reloadFollow])

  // Edits made elsewhere -- an assistant such as Claude Code over MCP,
  // another tab -- show up within a couple of seconds: poll a cheap stamp
  // and reload the graph only when it changes. (Runs poll faster, above.)
  const stampRef = useRef<string | null>(null)
  useEffect(() => {
    if (anyRunning) return
    let stopped = false
    const id = setInterval(() => {
      if (document.hidden) return
      api
        .graphStamp()
        .then(({ stamp }) => {
          if (stopped) return
          // Changed -- or the last graph read failed, so what's shown may
          // be stale even if the stamp looks the same.
          if ((stampRef.current !== null && stamp !== stampRef.current) || failures.current > 0) reloadFollow()
          stampRef.current = stamp
        })
        .catch(() => {})
    }, STAMP_POLL_MS)
    return () => {
      stopped = true
      clearInterval(id)
    }
  }, [anyRunning, reloadFollow])

  // Catch up as soon as the canvas is shown again: while hidden (another
  // tab in front, a minimized window) it doesn't poll.
  useEffect(() => {
    const onVisible = () => {
      if (!document.hidden) reloadFollow()
    }
    document.addEventListener('visibilitychange', onVisible)
    return () => document.removeEventListener('visibilitychange', onVisible)
  }, [reloadFollow])

  useEffect(() => {
    if (graph) setBlockNodes(toBlockNodes(graph, collapsedLanes, onViewPort, setReportBuildId))
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
    async (laneId: string) => {
      if (await host.confirm('Delete this lane? Blocks in it become unassigned, not deleted.')) {
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
  // While the outline awaits review: ghost bands for the lanes it would
  // create. While building: the current stage's planned-but-not-built
  // steps as ghost blocks, wired with dashed edges -- to each other, or to
  // the real blocks they build on -- and a highlight on whichever block
  // the AI last touched.
  const reviewPlan = build?.phase === 'awaiting_approval' ? build.plan : null
  const stagePlan = build && BUILD_LOCKING.has(build.phase) ? (build.stages?.[build.stage_index]?.plan ?? null) : null
  const buildId = build?.id
  const builtRefs = useMemo(() => {
    const m = new Map<string, string>()
    if (!graph || !buildId || !stagePlan) return m
    for (const [id, b] of Object.entries(graph.blocks)) {
      if (b.provenance?.build_id === buildId && b.provenance.plan_step) m.set(b.provenance.plan_step, id)
    }
    return m
  }, [graph, buildId, stagePlan])
  const ghostSteps = useMemo(
    () => (stagePlan ? stagePlan.steps.filter((s) => stagePlan.layout?.[s.ref] && !builtRefs.has(s.ref)) : []),
    [stagePlan, builtRefs],
  )

  const ghostNodes: (GhostFlowNode | GhostLaneNode)[] = useMemo(() => {
    const lanes: GhostLaneNode[] = Object.entries(reviewPlan?.lane_layout ?? {}).map(([key, l]) => ({
      id: `ghostlane_${key}`,
      type: 'ghostLane' as const,
      position: { x: BAND_X, y: l.top },
      draggable: false,
      selectable: false,
      zIndex: -9,
      data: { name: l.name, height: l.height },
    }))
    const steps: GhostFlowNode[] = ghostSteps.map((s) => ({
      id: `ghost_${s.ref}`,
      type: 'ghostBlock' as const,
      position: { x: stagePlan!.layout[s.ref].x, y: stagePlan!.layout[s.ref].y },
      draggable: false,
      selectable: false,
      data: { step: s, highlighted: false },
    }))
    return [...lanes, ...steps]
  }, [reviewPlan, stagePlan, ghostSteps])

  const ghostEdges: AiEdge[] = useMemo(() => {
    const refs = new Set(ghostSteps.map((s) => s.ref))
    return ghostSteps.flatMap((s) =>
      s.inputs.map((inp, i) => {
        const built = builtRefs.get(inp.from)
        return {
          id: `ghostedge_${s.ref}_${i}`,
          source: built ?? (refs.has(inp.from) ? `ghost_${inp.from}` : inp.from),
          sourceHandle: !built && refs.has(inp.from) ? GHOST_OUT : inp.from_port,
          target: `ghost_${s.ref}`,
          targetHandle: GHOST_IN,
          style: { stroke: 'var(--mm-ai-border-3)', strokeDasharray: '6 4' },
          selectable: false as const,
        }
      }),
    )
  }, [ghostSteps, builtRefs])

  const aiActiveBlock = useMemo(() => {
    if (!build || !BUILD_LOCKING.has(build.phase)) return null
    for (let i = build.events.length - 1; i >= 0; i--) {
      const e = build.events[i]
      if (e.kind === 'step' && e.block) return String(e.block) // a step plan_stage/build_stage built
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
        .catch((e) => host.notify(e.message))
    },
    [reload],
  )

  const onEdgeClick = useCallback(
    async (_: unknown, edge: DataWireEdgeType) => {
      if (edge.id.startsWith('ghostedge_')) return
      if (await host.confirm('Delete this wire?')) {
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
        .catch((e) => host.notify(e.message))
    },
    [reload],
  )

  const addLibraryBlock = useCallback(
    (file: string, id: string, at?: { x: number; y: number }) => {
      const pos = at ?? { x: 80 + Math.random() * 200, y: 80 + Math.random() * 200 }
      return api
        .createBlock({ library: { file, id }, x: pos.x, y: pos.y })
        .then((b) => {
          reload()
          setSelectedId(b.id)
          setSelectedPort(null)
        })
    },
    [reload],
  )

  // From the library's own tab (VS Code; see LibraryPage): place a block
  // here, so it's in this canvas's undo history; open the update dialog;
  // refresh the palette's library section.
  useEffect(
    () =>
      host.onEvent(({ name, data }) => {
        const d = (data ?? {}) as { file?: string; id?: string; category?: string; blockId?: string }
        if (name === 'placeLibraryBlock' && d.file && d.id) addLibraryBlock(d.file, d.id).catch((e) => host.notify((e as Error).message))
        else if (name === 'placeBuiltinBlock' && d.category) addBlock(d.category)
        else if (name === 'openUpdate' && d.blockId) {
          reload()
          setUpdateBlockId(d.blockId)
        } else if (name === 'libraryChanged') {
          setLibraryVersion((v) => v + 1)
          reload()
        }
      }),
    [addBlock, addLibraryBlock, reload],
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
      const dropped = JSON.parse(raw) as DragPayload
      if (dropped.kind === 'registry') addBlock(dropped.category, at)
      else addLibraryBlock(dropped.file, dropped.id, at).catch((err) => host.notify(err.message))
    },
    [screenToFlowPosition, addBlock, addLibraryBlock],
  )

  // An AI build saves the blocks it creates to the library (as drafts):
  // refetch the palette's library whenever the set of library blocks on the
  // canvas changes.
  const libraryKey = graph
    ? Object.values(graph.blocks)
        .map((b) => (b.definition_info?.source === 'library' ? `${b.category}@${b.definition_info.file}` : ''))
        .filter(Boolean)
        .sort()
        .join('|')
    : ''
  useEffect(() => {
    setLibraryVersion((v) => v + 1)
  }, [libraryKey])

  const selectedBlock: BlockOut | null = graph && selectedId ? (graph.blocks[selectedId] ?? null) : null
  // Create with AI from the palette uses the selected block's output as its
  // example input, when it has one to offer.
  const selectedBlockContext =
    selectedBlock && (selectedBlock.status === 'green' || selectedBlock.status === 'orange') && selectedBlock.outputs.some((p) => p.type === 'dataframe')
      ? selectedBlock.id
      : undefined

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
        onOpenLibrary={has('custom_blocks') ? () => openLibrary({ kind: 'browse' }) : undefined}
      />
      {graph?.sample_rows != null && (
        <div
          style={{
            display: 'flex',
            alignItems: 'center',
            gap: 8,
            padding: '5px 12px',
            background: 'var(--mm-warn-bg-2)',
            borderBottom: '1px solid var(--mm-warn-border)',
            color: 'var(--mm-warn)',
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
      {reconnecting && (
        <div
          role="status"
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
          <strong>Reconnecting to the backend…</strong>
          <span>what's shown may be out of date; it updates as soon as the backend answers again.</span>
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
                .catch((e) => host.notify(e.message))
            }}
          >
            Restore
          </button>
          <button onClick={() => setRecovery(null)}>Dismiss</button>
        </div>
      )}
      <div style={{ display: 'flex', flex: 1, minHeight: 0 }}>
        <Palette
          collapsed={lean && !paletteOpen}
          onToggle={lean ? togglePalette : undefined}
          onAdd={addBlock}
          onAddLibrary={(file, id) => addLibraryBlock(file, id).catch((e) => host.notify(e.message))}
          onOpenLibrary={() => openLibrary({ kind: 'browse' })}
          onCreateWithAI={() => openLibrary({ kind: 'ai', start: { addToCanvas: true, contextBlock: selectedBlockContext } })}
          onAddLane={addLane}
          libraryVersion={libraryVersion}
        />
        <div style={{ flex: 1 }} onDragOver={onDragOver} onDrop={onDrop}>
          <ReactFlow
            nodes={nodes}
            edges={graph ? [...toEdges(graph), ...(ghostEdges as unknown as DataWireEdgeType[])] : []}
            nodeTypes={nodeTypes}
            edgeTypes={edgeTypes}
            onNodesChange={onNodesChange}
            onMoveStart={(event) => {
              // Only the user's own pans and zooms (fitView's carry no event).
              if (event) userMovedAt.current = Date.now()
            }}
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
                  style={{ background: 'var(--mm-ai-bg-2)', color: 'var(--mm-ai-3)', borderRadius: 4, padding: '3px 8px', fontSize: 12 }}
                >
                  AI is editing -- canvas read-only
                </span>
              )}
              {/* In VS Code the assistant beside the canvas (Copilot, Claude Code) builds. */}
              {has('ai_builder') && !lean && (
                <button
                  className={buildOpen ? undefined : 'brand-primary'}
                  onClick={() => setBuildOpen((o) => !o)}
                  title="Have an AI plan and build a model from the selected blocks"
                >
                  {buildLive && !buildOpen ? 'AI build in progress…' : 'Build with AI'}
                </button>
              )}
            </Panel>
            {!lean && (
              <Panel
                position="bottom-right"
                style={{
                  display: 'flex',
                  gap: 10,
                  fontSize: 11,
                  color: 'var(--brand-charcoal, var(--mm-muted))',
                  background: 'var(--mm-overlay)',
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
            )}
          </ReactFlow>
        </div>
        {graph && has('ai_builder') && !lean && (buildOpen || buildLive) && (
          // Kept mounted (just hidden) while a build is live, so it keeps
          // polling the build -- and the canvas -- with the panel closed.
          <div style={{ display: buildOpen ? 'flex' : 'none' }}>
            <BuildPanel
              graph={graph}
              selectedIds={selectedIds}
              llmSettings={llmSettings}
              onChanged={reloadFollow}
              onBuild={setBuild}
              onClose={() => setBuildOpen(false)}
              onOpenReport={setReportBuildId}
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
        ) : lean && !selectedBlock ? null : (
          <Inspector
            block={selectedBlock}
            onChanged={reload}
            provider={llmSettings?.active_provider ?? null}
            onOpenUpdate={(b) => setUpdateBlockId(b.id)}
            onFixWithAI={(b) => openLibrary({ kind: 'ai', start: { fixBlock: b.id } })}
            onOpenInLibrary={(file, id) => openLibrary({ kind: 'browse', selected: { file, id } })}
          />
        )}
      </div>
      {libraryView && has('custom_blocks') && (
        <LibraryPanel
          initialView={libraryView}
          graph={graph}
          provider={llmSettings?.active_provider ?? null}
          onClose={() => {
            setLibraryView(null)
            reload()
          }}
          onLibraryChanged={() => setLibraryVersion((v) => v + 1)}
          onPlace={(file, id) => addLibraryBlock(file, id)}
          onPlaceBuiltin={(category) => addBlock(category)}
          onOpenUpdate={(blockId) => setUpdateBlockId(blockId)}
        />
      )}
      {updateBlockId && graph?.blocks[updateBlockId] && (
        <UpdateDialog block={graph.blocks[updateBlockId]} onClose={() => setUpdateBlockId(null)} onApplied={reload} />
      )}
      {reportBuildId && <ReportModal key={reportBuildId} buildId={reportBuildId} onClose={() => setReportBuildId(null)} />}
    </div>
  )
}
