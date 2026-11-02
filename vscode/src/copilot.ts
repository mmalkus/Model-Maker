import { randomBytes } from 'node:crypto'
import * as http from 'node:http'
import type { AddressInfo } from 'node:net'
import * as vscode from 'vscode'
import {
  type ChatMessage,
  errorBody,
  parseRequest,
  RequestError,
  toChatMessages,
  toChatTools,
  toCompletion,
} from './openai'

const JUSTIFICATION = 'Model-Maker uses Copilot to draft pipeline blocks and run AI model builds.'

/** GitHub Copilot's models, served to the Python backend as an
 * OpenAI-compatible endpoint (only extension code can reach vscode.lm).
 * Listens on 127.0.0.1 with a random bearer token; the backend gets both
 * through MODELMAKER_COPILOT_BASE_URL / MODELMAKER_COPILOT_TOKEN. */
export class CopilotRelay implements vscode.Disposable {
  readonly token = randomBytes(24).toString('hex')
  private server: http.Server | undefined
  private listening: Promise<string> | undefined

  constructor(private readonly log: vscode.OutputChannel) {}

  /** Starts the server on first use; resolves to its base URL (…/v1). */
  url(): Promise<string> {
    this.listening ??= new Promise((resolve, reject) => {
      const server = http.createServer((req, res) => {
        this.handle(req, res).catch((e) => this.send(res, 500, errorBody(String(e))))
      })
      server.on('error', reject)
      server.listen(0, '127.0.0.1', () => {
        const { port } = server.address() as AddressInfo
        resolve(`http://127.0.0.1:${port}/v1`)
      })
      this.server = server
    })
    return this.listening
  }

  dispose() {
    this.server?.close()
  }

  private send(res: http.ServerResponse, status: number, body: object) {
    if (res.headersSent) return
    res.writeHead(status, { 'Content-Type': 'application/json' })
    res.end(JSON.stringify(body))
  }

  private async handle(req: http.IncomingMessage, res: http.ServerResponse) {
    if (req.headers.authorization !== `Bearer ${this.token}`) return this.send(res, 401, errorBody('bad relay token'))
    const path = (req.url ?? '').split('?')[0]
    if (req.method === 'GET' && path === '/v1/models') return this.send(res, 200, { object: 'list', data: await listModels() })
    if (req.method === 'POST' && path === '/v1/chat/completions') return this.complete(req, res)
    this.send(res, 404, errorBody(`no route ${req.method} ${path}`))
  }

  private async complete(req: http.IncomingMessage, res: http.ServerResponse) {
    const cts = new vscode.CancellationTokenSource()
    // The backend gave up (timeout, build stopped): stop generating.
    res.on('close', () => {
      if (!res.writableFinished) cts.cancel()
    })
    try {
      const request = parseRequest(JSON.parse(await readBody(req)))
      const model = await findModel(request.model)
      const messages = toChatMessages(request.messages).map(toVsCodeMessage)
      const tools: vscode.LanguageModelChatTool[] = toChatTools(request.tools)
      const response = await model.sendRequest(messages, { justification: JUSTIFICATION, tools }, cts.token)
      const parts: Parameters<typeof toCompletion>[1] = []
      for await (const part of response.stream) {
        if (part instanceof vscode.LanguageModelTextPart) parts.push({ kind: 'text', text: part.value })
        else if (part instanceof vscode.LanguageModelToolCallPart)
          parts.push({ kind: 'toolCall', callId: part.callId, name: part.name, input: part.input })
      }
      const usage = await countUsage(model, messages, parts, cts.token)
      this.send(res, 200, toCompletion(model.id, parts, usage))
    } catch (e) {
      const [status, message] = describeError(e)
      this.log.appendLine(`[copilot] ${status}: ${message}`)
      this.send(res, status, errorBody(message))
    } finally {
      cts.dispose()
    }
  }
}

function readBody(req: http.IncomingMessage): Promise<string> {
  return new Promise((resolve, reject) => {
    const chunks: Buffer[] = []
    req.on('data', (c: Buffer) => chunks.push(c))
    req.on('end', () => resolve(Buffer.concat(chunks).toString('utf8')))
    req.on('error', reject)
  })
}

async function copilotModels(): Promise<vscode.LanguageModelChat[]> {
  return vscode.lm.selectChatModels({ vendor: 'copilot' })
}

async function listModels() {
  return (await copilotModels()).map((m) => ({
    id: m.id,
    object: 'model',
    owned_by: m.vendor,
    name: m.name,
    family: m.family,
    // Read by the backend's CopilotLoop to keep a build inside the window.
    max_input_tokens: m.maxInputTokens,
  }))
}

async function findModel(id: string | undefined): Promise<vscode.LanguageModelChat> {
  const models = await copilotModels()
  if (models.length === 0) throw new RequestError('no GitHub Copilot models are available -- sign in to Copilot in VS Code', 404)
  if (!id) return models[0]
  const match = models.find((m) => m.id === id) ?? models.find((m) => m.family === id)
  if (!match) throw new RequestError(`Copilot has no model ${id}; available: ${models.map((m) => m.id).join(', ')}`, 404)
  return match
}

function toVsCodeMessage(m: ChatMessage): vscode.LanguageModelChatMessage {
  if (m.role === 'assistant') {
    return vscode.LanguageModelChatMessage.Assistant(
      m.parts.map((p) =>
        p.kind === 'toolCall'
          ? new vscode.LanguageModelToolCallPart(p.callId, p.name, p.input)
          : new vscode.LanguageModelTextPart(p.kind === 'text' ? p.text : ''),
      ),
    )
  }
  return vscode.LanguageModelChatMessage.User(
    m.parts.map((p) =>
      p.kind === 'toolResult'
        ? new vscode.LanguageModelToolResultPart(p.callId, [new vscode.LanguageModelTextPart(p.text)])
        : new vscode.LanguageModelTextPart(p.kind === 'text' ? p.text : ''),
    ),
  )
}

async function countUsage(
  model: vscode.LanguageModelChat,
  messages: vscode.LanguageModelChatMessage[],
  parts: Parameters<typeof toCompletion>[1],
  token: vscode.CancellationToken,
): Promise<{ prompt: number; completion: number }> {
  const flatten = (ps: readonly unknown[]): string =>
    ps
      .map((p): string =>
        p instanceof vscode.LanguageModelTextPart
          ? p.value
          : p instanceof vscode.LanguageModelToolCallPart
            ? JSON.stringify(p.input)
            : p instanceof vscode.LanguageModelToolResultPart
              ? flatten(p.content)
              : '',
      )
      .join('\n')
  const output = parts.map((p) => (p.kind === 'text' ? p.text : JSON.stringify(p.input))).join('\n')
  try {
    const [prompt, completion] = await Promise.all([
      model.countTokens(messages.map((m) => flatten(m.content)).join('\n'), token),
      output ? model.countTokens(output, token) : Promise.resolve(0),
    ])
    return { prompt, completion }
  } catch {
    return { prompt: 0, completion: 0 }
  }
}

function describeError(e: unknown): [number, string] {
  if (e instanceof RequestError) return [e.status, e.message]
  if (e instanceof SyntaxError) return [400, `request body isn't valid JSON: ${e.message}`]
  if (e instanceof vscode.LanguageModelError) {
    if (e.code === vscode.LanguageModelError.NoPermissions().code)
      return [403, 'Model-Maker may not use Copilot yet -- approve the access request in VS Code, then retry']
    if (e.code === vscode.LanguageModelError.NotFound().code) return [404, e.message]
    if (e.code === vscode.LanguageModelError.Blocked().code)
      return [429, `Copilot blocked the request (rate limit or quota): ${e.message}`]
    return [502, e.message]
  }
  if (e instanceof vscode.CancellationError) return [499, 'cancelled']
  return [500, e instanceof Error ? e.message : String(e)]
}
