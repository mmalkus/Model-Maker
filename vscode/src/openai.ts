// The OpenAI chat-completions wire format, translated to and from a neutral
// form that maps one-to-one onto vscode.lm's message classes (copilot.ts does
// that last step). Kept free of the vscode module so it's unit-testable.
//
// What the backend sends (modelmaker/agent/loop.py's OpenAILoop and the
// openai draft provider): system/user/assistant/tool messages, assistant
// tool_calls with JSON-string arguments, and function tool definitions.

export type OpenAIToolCall = { id: string; type: 'function'; function: { name: string; arguments: string } }

export type OpenAIMessage =
  | { role: 'system' | 'user'; content: string | Array<{ type: string; text?: string }> | null }
  | { role: 'assistant'; content: string | null; tool_calls?: OpenAIToolCall[] }
  | { role: 'tool'; tool_call_id: string; content: string | null }

export type OpenAITool = { type: 'function'; function: { name: string; description?: string; parameters?: object } }

export type OpenAIRequest = {
  model: string
  messages: OpenAIMessage[]
  tools?: OpenAITool[]
  tool_choice?: unknown
}

export type Part =
  | { kind: 'text'; text: string }
  | { kind: 'toolCall'; callId: string; name: string; input: object }
  | { kind: 'toolResult'; callId: string; text: string }

export type ChatMessage = { role: 'user' | 'assistant'; parts: Part[] }

export type ChatTool = { name: string; description: string; inputSchema?: object }

export class RequestError extends Error {
  constructor(
    message: string,
    readonly status = 400,
  ) {
    super(message)
  }
}

function textOf(content: OpenAIMessage['content']): string {
  if (content == null) return ''
  if (typeof content === 'string') return content
  return content.map((c) => c.text ?? '').join('')
}

function parseArgs(raw: string): object {
  try {
    const parsed = JSON.parse(raw || '{}')
    return parsed && typeof parsed === 'object' ? parsed : {}
  } catch {
    return {}
  }
}

/** OpenAI messages -> alternating user/assistant messages. vscode.lm has no
 * system role, so system prompts become user text; tool results travel in
 * a user message, as vscode.lm requires, merged with neighbouring user
 * content so roles still alternate. */
export function toChatMessages(messages: OpenAIMessage[]): ChatMessage[] {
  const out: ChatMessage[] = []
  const push = (role: ChatMessage['role'], parts: Part[]) => {
    if (parts.length === 0) return
    const last = out[out.length - 1]
    if (last && last.role === role) last.parts.push(...parts)
    else out.push({ role, parts })
  }
  for (const m of messages) {
    if (m.role === 'system' || m.role === 'user') {
      const text = textOf(m.content)
      if (text) push('user', [{ kind: 'text', text }])
    } else if (m.role === 'assistant') {
      const parts: Part[] = []
      const text = textOf(m.content)
      if (text) parts.push({ kind: 'text', text })
      for (const call of m.tool_calls ?? []) {
        parts.push({ kind: 'toolCall', callId: call.id, name: call.function.name, input: parseArgs(call.function.arguments) })
      }
      push('assistant', parts)
    } else if (m.role === 'tool') {
      push('user', [{ kind: 'toolResult', callId: m.tool_call_id, text: textOf(m.content) }])
    } else {
      throw new RequestError(`unsupported message role: ${(m as { role: string }).role}`)
    }
  }
  return out
}

export function toChatTools(tools: OpenAITool[] | undefined): ChatTool[] {
  return (tools ?? [])
    .filter((t) => t.type === 'function')
    .map((t) => ({ name: t.function.name, description: t.function.description ?? '', inputSchema: t.function.parameters }))
}

export function parseRequest(body: unknown): OpenAIRequest {
  const req = body as OpenAIRequest
  if (!req || typeof req !== 'object' || !Array.isArray(req.messages)) {
    throw new RequestError('expected a chat completions request with a messages array')
  }
  return req
}

/** What came back from the model, as an OpenAI non-streaming completion.
 * (The backend's loops don't stream, so the relay collects the stream.) */
export function toCompletion(
  model: string,
  parts: Array<{ kind: 'text'; text: string } | { kind: 'toolCall'; callId: string; name: string; input: object }>,
  usage: { prompt: number; completion: number },
): object {
  const text = parts
    .filter((p) => p.kind === 'text')
    .map((p) => (p as { text: string }).text)
    .join('')
  const calls = parts.filter((p) => p.kind === 'toolCall') as Array<{ callId: string; name: string; input: object }>
  const message: Record<string, unknown> = { role: 'assistant', content: text || null }
  if (calls.length) {
    message.tool_calls = calls.map((c) => ({
      id: c.callId,
      type: 'function',
      function: { name: c.name, arguments: JSON.stringify(c.input ?? {}) },
    }))
  }
  return {
    id: `chatcmpl-${Date.now().toString(36)}`,
    object: 'chat.completion',
    created: Math.floor(Date.now() / 1000),
    model,
    choices: [{ index: 0, message, finish_reason: calls.length ? 'tool_calls' : 'stop' }],
    // Counted with the model's own tokenizer, but over flattened text --
    // an estimate, not what Copilot bills.
    usage: { prompt_tokens: usage.prompt, completion_tokens: usage.completion, total_tokens: usage.prompt + usage.completion, estimated: true },
  }
}

export function errorBody(message: string): object {
  return { error: { message, type: 'relay_error' } }
}
