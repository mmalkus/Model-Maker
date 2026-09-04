import {
  Background,
  Controls,
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
import { DataWireEdge, type DataWireEdgeType } from './DataWireEdge'
import { Inspector } from './Inspector'
import { BAND_X, DEFAULT_LANE_HEIGHT, LaneBand, layoutLanes, type LaneBandNode, type LaneLayoutEntry } from './LaneBand'
import { LaneLabels, LaneResizeHandles } from './LaneLabels'
import { Palette } from './Palette'
import { PortInspector } from './PortInspector'
import { Toolbar } from './Toolbar'
import type { BlockOut, GraphOut, LaneOut, LLMSettingsOut, PortType } from './types'

const nodeTypes = { modelBlock: BlockNode, laneBand: LaneBand }
const edgeTypes = { dataWire: DataWireEdge }

type FlowNode = BlockFlowNode | LaneBandNode

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
  const { fitView, screenToFlowPosition } = useReactFlow()
  const didInitialFit = useRef(false)

  const reloadLlmSettings = useCallback(() => {
    api.llmSettings().then(setLlmSettings).catch(() => setLlmSettings(null))
  }, [])

  useEffect(() => {
    reloadLlmSettings()
  }, [reloadLlmSettings])

  const onViewPort = useCallback((blockId: string, port: string, portType: PortType) => {
    setSelectedPort({ blockId, port, portType })
  }, [])

  const reload = useCallback(() => {
    api.graph().then((g) => {
      setGraph(g)
      const nextBlockNodes = toBlockNodes(g, collapsedLanes, onViewPort)
      setBlockNodes(nextBlockNodes)
      // fit the view to the actual blocks (not the oversized lane bands) once,
      // on first load -- re-fitting on every later reload would yank the
      // viewport out from under someone mid-edit.
      if (!didInitialFit.current && nextBlockNodes.length > 0) {
        didInitialFit.current = true
        requestAnimationFrame(() => fitView({ nodes: nextBlockNodes.map((n) => ({ id: n.id })), padding: 0.2 }))
      }
    })
  }, [collapsedLanes, fitView, onViewPort])

  useEffect(() => {
    reload()
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

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

  const nodes: FlowNode[] = useMemo(() => [...laneNodes, ...blockNodes], [laneNodes, blockNodes])

  const onNodesChange = useCallback((changes: NodeChange<FlowNode>[]) => {
    setBlockNodes((nds) => applyNodeChanges(changes, nds) as BlockFlowNode[])
  }, [])

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
      />
      <div style={{ display: 'flex', flex: 1, minHeight: 0 }}>
        <Palette onAdd={addBlock} onAddCustom={addCustomBlock} onAddLane={addLane} />
        <div style={{ flex: 1 }} onDragOver={onDragOver} onDrop={onDrop}>
          <ReactFlow
            nodes={nodes}
            edges={graph ? toEdges(graph) : []}
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
          </ReactFlow>
        </div>
        {selectedPort && graph ? (
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
