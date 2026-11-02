import assert from 'node:assert/strict'
import { spawn } from 'node:child_process'
import { cpSync, mkdtempSync } from 'node:fs'
import { tmpdir } from 'node:os'
import * as path from 'node:path'
import { after, before, test } from 'node:test'
import { Backend } from '../../src/backend'
import { registerTools } from '../../src/tools'
import { tools } from '../stub/vscode'

// The chat tools against a real backend on a copy of the demo project.
// Needs an interpreter with the package installed: MODELMAKER_TEST_PYTHON
// (CI's vscode job sets it); skipped without one.
const python = process.env.MODELMAKER_TEST_PYTHON
const opts = { skip: python ? false : 'set MODELMAKER_TEST_PYTHON to run' }

let backend: Backend
const token = { isCancellationRequested: false }

async function call(name: string, input: object): Promise<any> {
  const result = await tools.get(name)!.invoke({ input }, token)
  return JSON.parse(result.content[0].value)
}

before(async () => {
  if (!python) return
  const folder = path.join(mkdtempSync(path.join(tmpdir(), 'mm-tools-')), 'demo')
  cpSync(path.resolve(__dirname, '../../../projects/demo_pd_model'), folder, { recursive: true })
  const port = 18000 + Math.floor(Math.random() * 2000)
  const proc = spawn(python, ['-c', 'from modelmaker.api import main; main()'], {
    cwd: folder,
    env: { ...process.env, MODELMAKER_PORT: String(port), MODELMAKER_AUTH_TOKEN: 'test-token', MODELMAKER_CACHE_DIR: path.join(folder, '.cache') },
    stdio: 'ignore',
    detached: true,
  })
  backend = new Backend(folder, python, port, 'test-token', proc)
  for (let i = 0; i < 240; i++) {
    if (await fetch(`${backend.apiBase}/api/health`).then((r) => r.ok, () => false)) break
    await new Promise((r) => setTimeout(r, 250))
  }
  await backend.json('/project/load', { method: 'POST', body: { path: folder } })
  const refreshed: string[] = []
  const editor = { activeProject: () => folder, refresh: (f: string) => refreshed.push(f) }
  const manager = { acquire: async () => backend, release: () => {} }
  registerTools({ subscriptions: [] } as never, manager as never, editor as never)
})

after(() => backend?.stop())

test('reads the graph and the catalogue', opts, async () => {
  const graph = await call('modelmaker_get_graph', {})
  assert.equal(graph.blocks[0].id, 'b_load')
  assert.ok((await call('modelmaker_list_block_types', {})).tags.length > 0)
  assert.equal((await call('modelmaker_describe_block_type', { category: 'train_test_split' })).category, 'train_test_split')
})

test('edits, runs, summarizes and compiles', opts, async () => {
  const graph = await call('modelmaker_get_graph', {})
  const lane = graph.lanes[0]
  const added = await call('modelmaker_add_block', { category: 'train_test_split', name: 'Second split', lane: lane.name, params: { seed: 7 } })
  assert.equal(added.lane, lane.id)
  assert.equal(added.params.seed, 7)
  // Placed clear of the lane's existing cards.
  const others = graph.blocks.filter((b: any) => b.lane === lane.id)
  assert.ok(others.every((b: any) => b.position.x !== added.position.x || b.position.y !== added.position.y))

  const set = await call('modelmaker_set_params', { block: 'Second split', params: { test_size: 0.5 } })
  assert.deepEqual([set.params.seed, set.params.test_size], [7, 0.5])

  await call('modelmaker_connect', { from_block: 'b_load', from_port: graph.blocks[0].outputs[0].split(':')[0], to_block: added.id, to_port: 'df' })
  // The source hasn't been read: a single block's run says how to fix that.
  const blocked = await call('modelmaker_run', { block: added.id })
  assert.equal(blocked.blocks[0].status, 'red')
  assert.match(blocked.next, /run the whole pipeline/)
  await call('modelmaker_run', {})
  const run = await call('modelmaker_run', { block: added.id })
  assert.equal(run.blocks[0].status, 'green', JSON.stringify(run))

  const out = await call('modelmaker_get_output_summary', { block: added.id, port: 'test' })
  assert.equal(out.type, 'dataframe')
  assert.ok(out.row_count > 0 && !('rows' in out))

  const script = await call('modelmaker_compile', {})
  assert.match(script.source, /def /)
})

test('bad input comes back as a message the model can act on', opts, async () => {
  await assert.rejects(call('modelmaker_add_block', { category: 'filter', lane: 'Nowhere' }), /no lane "Nowhere"; the lanes are/)
  await assert.rejects(call('modelmaker_set_params', { block: 'nope', params: {} }), /no block "nope"/)
  await assert.rejects(call('modelmaker_add_block', { category: 'no_such_block' }))
})
