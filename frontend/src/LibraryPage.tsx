import { useCallback, useEffect, useState } from 'react'
import { api } from './api'
import { host } from './host'
import { LibraryPanel, type LibraryView } from './LibraryPanel'
import type { GraphOut } from './types'

// The block library in a VS Code editor tab of its own (Model-Maker: Open
// Block Library, or the canvas's Library button), on the same backend as
// its project's canvas. Whatever changes the canvas -- placing a block,
// opening the update dialog -- goes to the canvas through the extension
// (vscode/src/library.ts), so it lands in the canvas's undo history; then
// the canvas tab comes to the front.

const GRAPH_POLL_MS = 2000

export function LibraryPage() {
  const [view, setView] = useState<{ key: number; view: LibraryView }>({
    key: 0,
    view: (host.initialLibraryView as LibraryView | undefined) ?? { kind: 'browse' },
  })
  const [graph, setGraph] = useState<GraphOut | null>(null)
  const [provider, setProvider] = useState<string | null>(null)

  const loadGraph = useCallback(() => {
    api.graph().then(setGraph).catch(() => {})
  }, [])

  useEffect(() => {
    loadGraph()
    api
      .llmSettings()
      .then((s) => setProvider(s.active_provider ?? null))
      .catch(() => setProvider(null))
    // The canvas edits the same graph: keep what this page shows of it
    // (where a block is placed, Fix with AI's block) current.
    const id = setInterval(() => {
      if (!document.hidden) loadGraph()
    }, GRAPH_POLL_MS)
    return () => clearInterval(id)
  }, [loadGraph])

  // Opened again from the canvas on another view (e.g. Fix with AI).
  useEffect(
    () =>
      host.onEvent(({ name, data }) => {
        if (name === 'libraryView') setView((v) => ({ key: v.key + 1, view: data as LibraryView }))
      }),
    [],
  )

  return (
    <LibraryPanel
      key={view.key}
      page
      initialView={view.view}
      graph={graph}
      provider={provider}
      onClose={() => host.post('showCanvas')}
      onLibraryChanged={() => host.post('libraryChanged')}
      onPlace={async (file, id) => host.post('placeLibraryBlock', { file, id })}
      onPlaceBuiltin={(category) => host.post('placeBuiltinBlock', { category })}
      onOpenUpdate={(blockId) => host.post('openUpdate', { blockId })}
    />
  )
}
