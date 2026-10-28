import assert from 'node:assert/strict'
import { test } from 'node:test'
import { parseRequest, RequestError, toChatMessages, toChatTools, toCompletion, type OpenAIMessage } from '../../src/openai'

// What OpenAILoop sends mid-build: a system prompt, the task, a turn that
// called two tools, both results, and the next instruction.
const conversation: OpenAIMessage[] = [
  { role: 'system', content: 'You build models.' },
  { role: 'user', content: 'Split the data.' },
  {
    role: 'assistant',
    content: 'Looking first.',
    tool_calls: [
      { id: 'c1', type: 'function', function: { name: 'get_graph', arguments: '{}' } },
      { id: 'c2', type: 'function', function: { name: 'describe_block_type', arguments: '{"category":"train_test_split"}' } },
    ],
  },
  { role: 'tool', tool_call_id: 'c1', content: '{"blocks":{}}' },
  { role: 'tool', tool_call_id: 'c2', content: '{"params":["ratio"]}' },
  { role: 'user', content: 'Go on.' },
]

test('system and user text merge into one user turn', () => {
  const [first] = toChatMessages(conversation)
  assert.equal(first.role, 'user')
  assert.deepEqual(first.parts, [
    { kind: 'text', text: 'You build models.' },
    { kind: 'text', text: 'Split the data.' },
  ])
})

test('tool calls keep their ids and parsed arguments', () => {
  const assistant = toChatMessages(conversation)[1]
  assert.equal(assistant.role, 'assistant')
  assert.deepEqual(assistant.parts, [
    { kind: 'text', text: 'Looking first.' },
    { kind: 'toolCall', callId: 'c1', name: 'get_graph', input: {} },
    { kind: 'toolCall', callId: 'c2', name: 'describe_block_type', input: { category: 'train_test_split' } },
  ])
})

test('tool results ride in one user turn with the text that follows', () => {
  const messages = toChatMessages(conversation)
  assert.equal(messages.length, 3)
  assert.deepEqual(messages[2], {
    role: 'user',
    parts: [
      { kind: 'toolResult', callId: 'c1', text: '{"blocks":{}}' },
      { kind: 'toolResult', callId: 'c2', text: '{"params":["ratio"]}' },
      { kind: 'text', text: 'Go on.' },
    ],
  })
})

test('roles always alternate', () => {
  const roles = toChatMessages(conversation).map((m) => m.role)
  for (let i = 1; i < roles.length; i++) assert.notEqual(roles[i], roles[i - 1])
})

test('malformed tool arguments become an empty object, not a crash', () => {
  const [, assistant] = toChatMessages([
    { role: 'user', content: 'x' },
    { role: 'assistant', content: null, tool_calls: [{ id: 'c', type: 'function', function: { name: 'f', arguments: '{oops' } }] },
  ])
  assert.deepEqual(assistant.parts, [{ kind: 'toolCall', callId: 'c', name: 'f', input: {} }])
})

test('content arrays are flattened to text', () => {
  const [m] = toChatMessages([{ role: 'user', content: [{ type: 'text', text: 'a' }, { type: 'text', text: 'b' }] }])
  assert.deepEqual(m.parts, [{ kind: 'text', text: 'ab' }])
})

test('function tools map to name, description and schema', () => {
  const schema = { type: 'object', properties: { category: { type: 'string' } } }
  assert.deepEqual(toChatTools([{ type: 'function', function: { name: 'f', description: 'does f', parameters: schema } }]), [
    { name: 'f', description: 'does f', inputSchema: schema },
  ])
  assert.deepEqual(toChatTools(undefined), [])
})

test('a request without messages is rejected', () => {
  assert.throws(() => parseRequest({ model: 'x' }), RequestError)
  assert.equal(parseRequest({ model: 'x', messages: [] }).model, 'x')
})

test('a text reply becomes a stop completion', () => {
  const c = toCompletion('gpt-4.1', [{ kind: 'text', text: 'Hel' }, { kind: 'text', text: 'lo' }], { prompt: 10, completion: 2 }) as {
    choices: Array<{ message: { content: string; tool_calls?: unknown }; finish_reason: string }>
    usage: { prompt_tokens: number; completion_tokens: number; total_tokens: number }
  }
  assert.equal(c.choices[0].message.content, 'Hello')
  assert.equal(c.choices[0].message.tool_calls, undefined)
  assert.equal(c.choices[0].finish_reason, 'stop')
  assert.deepEqual([c.usage.prompt_tokens, c.usage.completion_tokens, c.usage.total_tokens], [10, 2, 12])
})

test('tool calls become OpenAI tool_calls with JSON-string arguments', () => {
  const c = toCompletion('m', [{ kind: 'toolCall', callId: 'call_1', name: 'connect', input: { from: 'a' } }], { prompt: 0, completion: 0 }) as {
    choices: Array<{ message: { content: string | null; tool_calls: unknown }; finish_reason: string }>
  }
  assert.equal(c.choices[0].message.content, null)
  assert.equal(c.choices[0].finish_reason, 'tool_calls')
  assert.deepEqual(c.choices[0].message.tool_calls, [
    { id: 'call_1', type: 'function', function: { name: 'connect', arguments: '{"from":"a"}' } },
  ])
})
