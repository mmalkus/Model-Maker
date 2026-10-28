// Just enough of the vscode module for tests/unit to load src/tools.ts and
// src/backend.ts in plain Node: registered tools are captured in `tools`.
export const tools = new Map<string, { invoke: Function; prepareInvocation?: Function }>()

export class Disposable {
  constructor(private readonly fn: () => void = () => {}) {}
  dispose() {
    this.fn()
  }
}

export class EventEmitter<T> {
  private listeners: Array<(e: T) => void> = []
  event = (l: (e: T) => void) => {
    this.listeners.push(l)
    return new Disposable()
  }
  fire(e: T) {
    for (const l of this.listeners) l(e)
  }
  dispose() {
    this.listeners = []
  }
}

export class LanguageModelTextPart {
  constructor(readonly value: string) {}
}

export class LanguageModelToolResult {
  constructor(readonly content: unknown[]) {}
}

export const lm = {
  registerTool(name: string, tool: { invoke: Function }) {
    tools.set(name, tool)
    return new Disposable()
  },
}

export const workspace = { asRelativePath: (p: string) => p }
