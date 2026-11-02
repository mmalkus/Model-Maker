import * as vscode from 'vscode'
import { findBlock, type Graph, placeNewBlock, resolveLane, runOutcome, ToolInputError } from './assistant'
import { type Backend, type BackendManager, BackendError } from './backend'
import type { ModelEditorProvider } from './editor'

// Copilot Chat (agent mode) tools for the open Model-Maker canvas. They read
// through the backend's /api/assistant views -- schemas and summary
// statistics, never a dataframe's rows -- and edit through the canvas's own
// endpoints, so every change is in its undo history. Each edit asks first
// (VS Code's tool confirmation), and the canvas refreshes after it.

const RUN_TIMEOUT_MS = 10 * 60_000
const SCRIPT_CHARS = 30_000

type Ctx = { backend: Backend; folder: string }

type ToolSpec<T> = {
  // Shown while it runs, and (for edits) in the confirmation.
  describe(input: T): string
  edits: boolean
  run(ctx: Ctx, input: T, token: vscode.CancellationToken): Promise<unknown>
}

export function registerTools(context: vscode.ExtensionContext, backends: BackendManager, editor: ModelEditorProvider) {
  const register = <T>(name: string, spec: ToolSpec<T>) =>
    context.subscriptions.push(
      vscode.lm.registerTool<T>(name, {
        prepareInvocation: (options) => {
          const what = spec.describe(options.input)
          const project = editor.activeProject()
          return {
            invocationMessage: what,
            confirmationMessages: spec.edits
              ? { title: 'Edit the Model-Maker canvas?', message: `${what}${project ? ` in ${vscode.workspace.asRelativePath(project)}` : ''}.` }
              : undefined,
          }
        },
        invoke: async (options, token) => {
          const folder = editor.activeProject()
          if (!folder) throw new Error('No Model-Maker canvas is open -- open a project\'s model.json in the canvas first.')
          const backend = await backends.acquire(folder)
          try {
            const result = await spec.run({ backend, folder }, options.input, token)
            if (spec.edits) editor.refresh(folder)
            return new vscode.LanguageModelToolResult([new vscode.LanguageModelTextPart(JSON.stringify(result))])
          } catch (e) {
            // Input mistakes and API refusals go back to the model to correct.
            if (e instanceof ToolInputError || e instanceof BackendError) throw new Error(e.message)
            throw e
          } finally {
            backends.release(folder)
          }
        },
      }),
    )

  const graph = (ctx: Ctx) => ctx.backend.json<Graph>('/assistant/graph')

  register<Record<string, never>>('modelmaker_get_graph', {
    describe: () => 'Reading the Model-Maker graph',
    edits: false,
    run: (ctx) => graph(ctx),
  })

  register<{ tag?: string }>('modelmaker_list_block_types', {
    describe: (i) => (i.tag ? `Listing ${i.tag} blocks` : 'Listing block tags'),
    edits: false,
    run: (ctx, i) => ctx.backend.json(`/assistant/block_types${i.tag ? `?tag=${encodeURIComponent(i.tag)}` : ''}`),
  })

  register<{ category: string }>('modelmaker_describe_block_type', {
    describe: (i) => `Reading the docs for ${i.category}`,
    edits: false,
    run: (ctx, i) => ctx.backend.json(`/assistant/block_types/${encodeURIComponent(i.category)}`),
  })

  register<{ block: string; port?: string }>('modelmaker_get_output_summary', {
    describe: (i) => `Summarizing ${i.block}'s output`,
    edits: false,
    run: async (ctx, i) => {
      const block = findBlock(await graph(ctx), i.block)
      const q = new URLSearchParams({ block: block.id, ...(i.port ? { port: i.port } : {}) })
      return ctx.backend.json(`/assistant/output?${q}`)
    },
  })

  register<{ category: string; name?: string; lane?: string; params?: Record<string, unknown> }>('modelmaker_add_block', {
    describe: (i) => `Adding a ${i.category} block${i.name ? ` "${i.name}"` : ''}`,
    edits: true,
    run: async (ctx, i) => {
      const g = await graph(ctx)
      const lane = resolveLane(g, i.lane)
      const at = placeNewBlock(g, lane)
      const block = await ctx.backend.json<{ id: string }>('/blocks', {
        method: 'POST',
        body: { category: i.category, name: i.name, lane, params: i.params ?? {}, ...at },
      })
      return ctx.backend.json(`/assistant/blocks/${block.id}`)
    },
  })

  register<{ block: string; params: Record<string, unknown> }>('modelmaker_set_params', {
    describe: (i) => `Setting ${Object.keys(i.params ?? {}).join(', ') || 'params'} on ${i.block}`,
    edits: true,
    run: async (ctx, i) => {
      const block = findBlock(await graph(ctx), i.block)
      // Merged, so the model can send just what changes.
      await ctx.backend.json(`/blocks/${block.id}`, { method: 'PATCH', body: { params: { ...block.params, ...i.params } } })
      return ctx.backend.json(`/assistant/blocks/${block.id}`)
    },
  })

  register<{ from_block: string; from_port: string; to_block: string; to_port: string }>('modelmaker_connect', {
    describe: (i) => `Wiring ${i.from_block}.${i.from_port} → ${i.to_block}.${i.to_port}`,
    edits: true,
    run: async (ctx, i) => {
      const g = await graph(ctx)
      const from = findBlock(g, i.from_block)
      const to = findBlock(g, i.to_block)
      return ctx.backend.json('/wires', {
        method: 'POST',
        body: { from_block: from.id, from_port: i.from_port, to_block: to.id, to_port: i.to_port },
      })
    },
  })

  register<{ block?: string }>('modelmaker_run', {
    describe: (i) => (i.block ? `Running up to ${i.block}` : 'Running the whole pipeline'),
    edits: true,
    run: async (ctx, i, token) => {
      const target = i.block ? findBlock(await graph(ctx), i.block).id : null
      await ctx.backend.json(target ? `/blocks/${target}/run_to_here` : '/run_all', { method: 'POST' })
      // The canvas polls while it sees a run: let it see this one.
      setTimeout(() => editor.refresh(ctx.folder), 300)
      const deadline = Date.now() + RUN_TIMEOUT_MS
      for (;;) {
        const g = await graph(ctx)
        if (!g.running) return runOutcome(g, target)
        if (token.isCancellationRequested) {
          await ctx.backend.json('/run/cancel', { method: 'POST' }).catch(() => {})
          throw new Error('cancelled -- the run was stopped')
        }
        if (Date.now() > deadline) return { still_running: true, note: 'the run is still going; check the graph later', ...runOutcome(g, target) }
        await new Promise((r) => setTimeout(r, 1000))
      }
    },
  })

  register<{ streaming?: boolean }>('modelmaker_compile', {
    describe: () => 'Compiling the pipeline to a Python script',
    edits: false,
    run: async (ctx, i) => {
      const { source } = await ctx.backend.json<{ source: string }>('/compile', { method: 'POST', body: { stream: !!i.streaming } })
      return source.length > SCRIPT_CHARS ? { source: source.slice(0, SCRIPT_CHARS), truncated: true } : { source }
    },
  })
}
