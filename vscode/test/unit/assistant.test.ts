import assert from 'node:assert/strict'
import { test } from 'node:test'
import { findBlock, type Graph, placeNewBlock, resolveLane, runOutcome, ToolInputError } from '../../src/assistant'

const block = (id: string, lane: string | null, x: number, y: number, status: any = 'green', error?: string) => ({
  id, name: `Block ${id}`, category: 'filter', lane, status, params: {}, inputs: [], outputs: [], position: { x, y }, ...(error ? { error } : {}),
})

const graph: Graph = {
  project: 'p',
  lanes: [{ id: 'lane_prep', name: 'Data Prep' }, { id: 'lane_val', name: 'Validation' }],
  blocks: [block('a', 'lane_prep', 100, 50), block('b', 'lane_prep', 320, 50), block('c', null, 100, 600, 'red', 'boom')],
  wires: [],
  running: false,
}

test('lanes resolve by id or by name, any case', () => {
  assert.equal(resolveLane(graph, 'lane_val'), 'lane_val')
  assert.equal(resolveLane(graph, ' data prep '), 'lane_prep')
  assert.equal(resolveLane(graph, undefined), null)
  assert.throws(() => resolveLane(graph, 'Estimation'), (e: Error) => e instanceof ToolInputError && /Data Prep \(lane_prep\)/.test(e.message))
})

test('a new block goes right of its lane, or below everything', () => {
  assert.deepEqual(placeNewBlock(graph, 'lane_prep'), { x: 540, y: 50 })
  assert.deepEqual(placeNewBlock(graph, 'lane_val'), { x: 100, y: 760 })
  assert.deepEqual(placeNewBlock({ ...graph, blocks: [] }, null), { x: 100, y: 0 })
})

test('blocks are found by id or name', () => {
  assert.equal(findBlock(graph, 'b').id, 'b')
  assert.equal(findBlock(graph, 'block c').id, 'c')
  assert.throws(() => findBlock(graph, 'z'), ToolInputError)
})

test('a run reports the target, or every block that is not green', () => {
  assert.deepEqual(runOutcome(graph, 'a'), { statuses: { green: 2, red: 1 }, blocks: [{ id: 'a', name: 'Block a', status: 'green' }] })
  assert.deepEqual((runOutcome(graph, null) as any).blocks, [{ id: 'c', name: 'Block c', status: 'red', error: 'boom' }])
})

test('a run blocked on an unread source says how to unblock it', () => {
  const g = { ...graph, blocks: [block('d', null, 0, 0, 'red', "RuntimeError: input block 'b_load' is not green")] }
  assert.match((runOutcome(g, 'd') as any).next, /run the whole pipeline/)
  assert.equal((runOutcome(graph, 'a') as any).next, undefined)
})
