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
const opened: Array<[string, boolean]> = []
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
  const opener = async (f: string, create: boolean) => {
    opened.push([f, create])
    return folder
  }
  registerTools({ subscriptions: [] } as never, manager as never, editor as never, opener)
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

test('builds a model from an empty canvas', opts, async () => {
  const start = await call('modelmaker_open_project', { folder: 'demo', create: true })
  assert.deepEqual(opened.at(-1), ['demo', true])
  assert.ok(start.folder && Array.isArray(start.blocks))
  await backend.json('/project/new', { method: 'POST' })

  const files = await call('modelmaker_list_files', {})
  assert.ok(Array.isArray(files.entries) && files.path)
  const tags = (await call('modelmaker_list_block_types', {})).tags.map((t: any) => t.tag)
  assert.equal(tags[0], 'data_sources')

  const data = await call('modelmaker_add_lane', { name: 'Data' })
  await call('modelmaker_add_lane', { name: 'Estimation' })
  const load = await call('modelmaker_add_block', { category: 'demo_credit_data', name: 'Loans', lane: 'Data' })
  assert.equal(load.lane, data.id)
  const split = await call('modelmaker_add_block', { category: 'train_test_split', lane: 'Estimation' })
  const extra = await call('modelmaker_add_block', { category: 'filter', lane: 'Estimation' })
  await call('modelmaker_delete_block', { block: extra.id })
  await call('modelmaker_connect', { from_block: 'Loans', from_port: load.outputs[0].split(':')[0], to_block: split.id, to_port: 'df' })
  await call('modelmaker_disconnect', { to_block: split.id, to_port: 'df' })
  assert.equal((await call('modelmaker_get_graph', {})).wires.length, 0)
  await call('modelmaker_connect', { from_block: load.id, from_port: load.outputs[0].split(':')[0], to_block: split.id, to_port: 'df' })

  const run = await call('modelmaker_run', {})
  assert.deepEqual(run.statuses, { green: 2 }, JSON.stringify(run))
  const cols = await call('modelmaker_get_input_columns', { block: split.id })
  const names = cols.ports.df.map((c: any) => c.name)
  assert.ok(names.length > 3)
  const roled = await call('modelmaker_set_column_role', { block: load.id, column: names[0], role: 'excluded' })
  assert.deepEqual(roled.column_roles, { [names[0]]: 'excluded' })
  // The role made the source stale: a whole run re-reads it.
  const rerun = await call('modelmaker_run', {})
  assert.deepEqual([rerun.statuses, rerun.refreshed_sources], [{ green: 2 }, ['Loans']], JSON.stringify(rerun))
  const renamed = await call('modelmaker_set_params', { block: split.id, name: 'Holdout split', params: { seed: 11 } })
  assert.deepEqual([renamed.name, renamed.params.seed], ['Holdout split', 11])
  const check = await call('modelmaker_check', {})
  assert.equal(check.ok, true)
  await assert.rejects(call('modelmaker_set_params', { block: split.id }), /nothing to change/)
})
