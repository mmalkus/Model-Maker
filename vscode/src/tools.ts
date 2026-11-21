import * as vscode from 'vscode'
import { findBlock, findWire, type Graph, resolveLane, runOutcome, ToolInputError } from './assistant'
import { type Backend, type BackendManager, BackendError } from './backend'
import type { ModelEditorProvider } from './editor'

// Copilot Chat (agent mode) tools for the open Model-Maker canvas. They read
// through the backend's /api/assistant views -- schemas and summary
// statistics, never a dataframe's rows -- and edit through the canvas's own
// endpoints, so every change is in its undo history. Each edit asks first
// (VS Code's tool confirmation), and the canvas refreshes after it.
//
// Together they cover a whole build from an empty canvas: open or create a
// project, find the data, lay out lanes, add and wire blocks, tag column
// roles, run, check the result for leakage, fix, and compile.

const RUN_TIMEOUT_MS = 10 * 60_000
const SCRIPT_CHARS = 30_000

type Ctx = { backend: Backend; folder: string }

type ToolSpec<T> = {
  // Shown while it runs, and (for edits) in the confirmation.
  describe(input: T): string
  // Whether this call changes the canvas: asks first, refreshes after.
  edits: boolean | ((input: T) => boolean)
  run(ctx: Ctx, input: T, token: vscode.CancellationToken): Promise<unknown>
}

/** Opens (and with `create`, first creates) a project folder's canvas;
 * returns the folder's absolute path. Supplied by extension.ts. */
export type ProjectOpener = (folder: string, create: boolean) => Promise<string>

export function registerTools(context: vscode.ExtensionContext, backends: BackendManager, editor: ModelEditorProvider, openProject?: ProjectOpener) {
  // Only Copilot's agent mode uses these; an editor without vscode.lm's
  // tools API (Cursor, Windsurf: their agents use modelmaker-mcp) skips them.
  if (typeof vscode.lm?.registerTool !== 'function') return
  const register = <T>(name: string, spec: ToolSpec<T> & { folder?: (input: T) => Promise<string> }) => {
    const edits = (input: T) => (typeof spec.edits === 'function' ? spec.edits(input) : spec.edits)
    context.subscriptions.push(
      vscode.lm.registerTool<T>(name, {
        prepareInvocation: (options) => {
          const what = spec.describe(options.input)
          const project = spec.folder ? undefined : editor.activeProject()
          return {
            invocationMessage: what,
            confirmationMessages: edits(options.input)
              ? { title: 'Edit the Model-Maker canvas?', message: `${what}${project ? ` in ${vscode.workspace.asRelativePath(project)}` : ''}.` }
              : undefined,
          }
        },
        invoke: async (options, token) => {
          let folder: string | undefined
          try {
            folder = spec.folder ? await spec.folder(options.input) : editor.activeProject()
          } catch (e) {
            throw new Error(e instanceof Error ? e.message : String(e))
          }
          if (!folder) {
            throw new Error(
              'No Model-Maker canvas is open -- call modelmaker_open_project (with create: true for a new project), or open a project\'s model.json in the canvas.',
            )
          }
          const backend = await backends.acquire(folder)
          try {
            const result = await spec.run({ backend, folder }, options.input, token)
            if (edits(options.input)) editor.refresh(folder)
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
  }

  const graph = (ctx: Ctx) => ctx.backend.json<Graph>('/assistant/graph')
  const brief = (ctx: Ctx, id: string) => ctx.backend.json(`/assistant/blocks/${encodeURIComponent(id)}`)

  // ---- the project ----------------------------------------------------------

  if (openProject) {
    register<{ folder: string; create?: boolean }>('modelmaker_open_project', {
      describe: (i) => `${i.create ? 'Creating and opening' : 'Opening'} the Model-Maker project ${i.folder}`,
      edits: true,
      folder: (i) => openProject(i.folder, !!i.create),
      run: async (ctx) => ({ folder: ctx.folder, ...(await graph(ctx)) }),
    })
  }

  register<Record<string, never>>('modelmaker_get_graph', {
    describe: () => 'Reading the Model-Maker graph',
    edits: false,
    run: (ctx) => graph(ctx),
  })

  register<{ path?: string; ext?: string }>('modelmaker_list_files', {
    describe: (i) => `Listing data files${i.path ? ` in ${i.path}` : ''}`,
    edits: false,
    run: async (ctx, i) => {
      const q = new URLSearchParams({ ...(i.path ? { path: i.path } : {}), ...(i.ext ? { ext: i.ext } : {}) })
      const [files, datasets, sources] = await Promise.all([
        ctx.backend.json<{ path: string; parent: string | null; entries: Array<{ name: string; path: string; is_dir: boolean }> }>(`/browse?${q}`),
        ctx.backend.json<unknown[]>('/datasets').catch(() => []),
        ctx.backend.json<unknown[]>('/sources').catch(() => []),
      ])
      return {
        ...files,
        entries: files.entries.map(({ name, path, is_dir }) => ({ name, path, ...(is_dir ? { dir: true } : {}) })),
        // What the workspace offers instead of files, when it does (read_dataset / read_source blocks).
        ...(datasets.length ? { datasets } : {}),
        ...(sources.length ? { sources } : {}),
      }
    },
  })

  // ---- the catalogue --------------------------------------------------------

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

  // ---- reading results ------------------------------------------------------

  register<{ block: string; port?: string }>('modelmaker_get_output_summary', {
    describe: (i) => `Summarizing ${i.block}'s output`,
    edits: false,
    run: async (ctx, i) => {
      const block = findBlock(await graph(ctx), i.block)
      const q = new URLSearchParams({ block: block.id, ...(i.port ? { port: i.port } : {}) })
      return ctx.backend.json(`/assistant/output?${q}`)
    },
  })

  register<{ block: string }>('modelmaker_get_input_columns', {
    describe: (i) => `Reading the columns ${i.block} gets`,
    edits: false,
    run: async (ctx, i) => {
      const block = findBlock(await graph(ctx), i.block)
      const ports = await ctx.backend.json<Record<string, Array<{ name: string; dtype: string; role?: string | null }>>>(
        `/blocks/${block.id}/input_schema`,
      )
      const out: Record<string, unknown> = { block: block.id, ports }
      if (Object.values(ports).every((cols) => cols.length === 0)) {
        out.note = 'no columns yet: the blocks upstream have to run first (modelmaker_run)'
      }
      return out
    },
  })

  register<{ block?: string }>('modelmaker_check', {
    describe: (i) => (i.block ? `Checking ${i.block}` : 'Checking the pipeline for leakage'),
    edits: false,
    run: async (ctx, i) => {
      const target = i.block ? findBlock(await graph(ctx), i.block).id : null
      return ctx.backend.json(`/assistant/check${target ? `?block=${encodeURIComponent(target)}` : ''}`)
    },
  })

  // ---- writing library blocks -------------------------------------------------

  register<Record<string, never>>('modelmaker_get_block_guide', {
    describe: () => 'Reading how to write a library block',
    edits: false,
    run: (ctx) => ctx.backend.json('/assistant/block_guide'),
  })

  register<{ name: string }>('modelmaker_check_block_file', {
    describe: (i) => `Checking the block file ${i.name}`,
    edits: false,
    run: (ctx, i) => ctx.backend.json('/assistant/check_block_file', { method: 'POST', body: { name: i.name } }),
  })

  register<{ block: string; apply?: boolean }>('modelmaker_update_block', {
    describe: (i) => (i.apply ? `Updating ${i.block} to its library file's version` : `Previewing the update of ${i.block}`),
    // A preview only reads: no confirmation, no canvas refresh.
    edits: (i) => !!i.apply,
    run: async (ctx, i) => {
      const block = findBlock(await graph(ctx), i.block)
      if (!block.library_block) throw new ToolInputError(`${block.name} is a built-in block: only library blocks are updated this way`)
      const id = encodeURIComponent(block.id)
      if (!i.apply) return ctx.backend.json(`/blocks/${id}/update_preview`)
      const result = await ctx.backend.json<{ applied: unknown }>(`/blocks/${id}/update`, { method: 'POST' })
      return { applied: result.applied, block: await brief(ctx, block.id) }
    },
  })

  // ---- editing --------------------------------------------------------------

  register<{ name: string }>('modelmaker_add_lane', {
    describe: (i) => `Adding the lane "${i.name}"`,
    edits: true,
    run: (ctx, i) => ctx.backend.json('/assistant/lanes', { method: 'POST', body: { name: i.name } }),
  })

  register<{ category: string; name?: string; lane?: string; params?: Record<string, unknown> }>('modelmaker_add_block', {
    describe: (i) => `Adding a ${i.category} block${i.name ? ` "${i.name}"` : ''}`,
    edits: true,
    run: (ctx, i) =>
      ctx.backend.json('/assistant/blocks', { method: 'POST', body: { category: i.category, name: i.name, lane: i.lane, params: i.params ?? {} } }),
  })

  register<{ block: string; params?: Record<string, unknown>; name?: string; lane?: string }>('modelmaker_set_params', {
    describe: (i) => {
      const what = [...Object.keys(i.params ?? {}), ...(i.name ? ['name'] : []), ...(i.lane ? ['lane'] : [])]
      return `Setting ${what.join(', ') || 'params'} on ${i.block}`
    },
    edits: true,
    run: async (ctx, i) => {
      const g = await graph(ctx)
      const block = findBlock(g, i.block)
      const body: Record<string, unknown> = {}
      // Merged, so the model can send just what changes.
      if (i.params) body.params = { ...block.params, ...i.params }
      if (i.name) body.name = i.name
      if (i.lane) body.lane = resolveLane(g, i.lane)
      if (Object.keys(body).length === 0) throw new ToolInputError('nothing to change: give params, name or lane')
      await ctx.backend.json(`/blocks/${block.id}`, { method: 'PATCH', body })
      return brief(ctx, block.id)
    },
  })

  register<{ block: string; column: string; role: string }>('modelmaker_set_column_role', {
    describe: (i) => `Tagging ${i.column} as ${i.role} on ${i.block}`,
    edits: true,
    run: async (ctx, i) => {
      const block = findBlock(await graph(ctx), i.block)
      await ctx.backend.json(`/blocks/${block.id}/column_role`, { method: 'POST', body: { column: i.column, role: i.role } })
      return brief(ctx, block.id)
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

  register<{ wire?: string; to_block?: string; to_port?: string }>('modelmaker_disconnect', {
    describe: (i) => (i.wire ? `Removing wire ${i.wire}` : `Removing the wire into ${i.to_block}.${i.to_port}`),
    edits: true,
    run: async (ctx, i) => {
      const wire = findWire(await graph(ctx), { id: i.wire, to_block: i.to_block, to_port: i.to_port })
      await ctx.backend.json(`/wires/${wire.id}`, { method: 'DELETE' })
      return { removed: wire }
    },
  })

  register<{ block: string }>('modelmaker_delete_block', {
    describe: (i) => `Deleting ${i.block}`,
    edits: true,
    run: async (ctx, i) => {
      const block = findBlock(await graph(ctx), i.block)
      await ctx.backend.json(`/blocks/${block.id}`, { method: 'DELETE' })
      return { deleted: block.id, name: block.name }
    },
  })

  // ---- running ----------------------------------------------------------------

  register<{ block?: string; streaming?: boolean }>('modelmaker_run', {
    describe: (i) => (i.streaming ? 'Running the whole pipeline (streaming)' : i.block ? `Running up to ${i.block}` : 'Running the whole pipeline'),
    edits: true,
    run: async (ctx, i, token) => {
      if (i.streaming && i.block) throw new Error('a streaming run runs the whole pipeline: leave out block')
      const deadline = Date.now() + RUN_TIMEOUT_MS
      // A run already going (the canvas's, or an earlier call's that timed
      // out): queue behind it rather than fail with "already in progress".
      await waitIdle(ctx, deadline, token)
      // Cancelled while queued: nothing of this call's has started, and the
      // run already going isn't ours to stop.
      if (token.isCancellationRequested) throw new Error("cancelled -- this run hadn't started; the run already going was left running")
      const before = await graph(ctx)
      const target = i.block ? findBlock(before, i.block).id : null
      if (before.running) return stillRunning(before, target, "another run is still going, so this one hasn't started")
      const refreshed: string[] = []
      if (!target) {
        // A data source whose settings changed since it was read (a param, a
        // column role) is stale. Backends from 0.3.2 re-read it in Run all;
        // re-reading it here first keeps a 0.3.0 backend from failing on it.
        for (const b of before.blocks.filter((b) => b.data_source && b.status === 'orange')) {
          await ctx.backend.json(`/blocks/${b.id}/refresh`, { method: 'POST' })
          await waitIdle(ctx, deadline, token)
          refreshed.push(b.name)
        }
      }
      try {
        await ctx.backend.json(i.streaming ? '/run_all_streaming' : target ? `/blocks/${target}/run_to_here` : '/run_all', { method: 'POST' })
      } catch (e) {
        // A run started between the idle check and this one (the canvas's).
        if (/already in progress/i.test(String((e as Error)?.message ?? e))) {
          throw new Error(`${(e as Error).message}: modelmaker_wait_for_run waits for it and returns its outcome; then run again`)
        }
        throw e
      }
      // The canvas polls while it sees a run: let it see this one.
      setTimeout(() => editor.refresh(ctx.folder), 300)
      for (;;) {
        const g = await graph(ctx)
        if (!g.running) return withChecks(ctx, { ...runOutcome(g, target), ...(refreshed.length ? { refreshed_sources: refreshed } : {}) }, target)
        if (token.isCancellationRequested) {
          await ctx.backend.json('/run/cancel', { method: 'POST' }).catch(() => {})
          throw new Error('cancelled -- the run was stopped')
        }
        if (Date.now() > deadline) return stillRunning(g, target, 'the run is still going')
        await new Promise((r) => setTimeout(r, 1000))
      }
    },
  })

  register<{ seconds?: number }>('modelmaker_wait_for_run', {
    describe: () => 'Waiting for the run in progress',
    edits: false,
    run: async (ctx, i, token) => {
      const asked = i.seconds == null ? RUN_TIMEOUT_MS / 1000 : Number(i.seconds)
      const seconds = Math.min(Math.max(Number.isFinite(asked) ? asked : RUN_TIMEOUT_MS / 1000, 0), WAIT_MAX_SECONDS)
      // Cancelling stops the waiting, not the run.
      await waitIdle(ctx, Date.now() + seconds * 1000, token)
      const g = await graph(ctx)
      return g.running ? stillRunning(g, null, 'the run is still going') : withChecks(ctx, runOutcome(g, null), null)
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

const WAIT_MAX_SECONDS = 60 * 60

function stillRunning(g: Graph, target: string | null, note: string): object {
  return { still_running: true, note: `${note}; modelmaker_wait_for_run keeps waiting and returns the outcome`, ...runOutcome(g, target) }
}

async function waitIdle(ctx: Ctx, deadline: number, token: vscode.CancellationToken) {
  while (Date.now() < deadline && !token.isCancellationRequested) {
    if (!(await ctx.backend.json<Graph>('/assistant/graph')).running) return
    await new Promise((r) => setTimeout(r, 500))
  }
}

type Check = { leaks: unknown[]; suspicious_features: unknown[]; hints: Record<string, string[]>; ok: boolean }

/** A run's outcome plus what the AI builder would say about it: leakage
 * and suspicious features whenever there are any, and the target block's
 * decision hints. */
async function withChecks(ctx: Ctx, outcome: object, target: string | null): Promise<object> {
  const check = await ctx.backend.json<Check>(`/assistant/check${target ? `?block=${encodeURIComponent(target)}` : ''}`).catch(() => null)
  if (!check) return outcome
  const out: Record<string, unknown> = { ...outcome }
  if (check.leaks.length) out.leaks = check.leaks
  if (check.suspicious_features.length) out.suspicious_features = check.suspicious_features
  if (target && check.hints[target]) out.hints = check.hints[target]
  else if (!target && Object.keys(check.hints).length) out.next_check = 'modelmaker_check gives each block\'s decision hints'
  return out
}
