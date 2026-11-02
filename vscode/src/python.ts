import * as vscode from 'vscode'

/** Which Python runs a project's backend: the modelmaker.pythonPath
 * setting; else, with modelmaker.environment "managed" (the default),
 * Model-Maker's own environment (managed.ts), which the caller sets up;
 * else the Python extension's selection for that folder, else
 * python3/python on PATH. */
export async function resolvePython(folder: vscode.Uri): Promise<{ managed: true } | { managed: false; python: string }> {
  const config = vscode.workspace.getConfiguration('modelmaker', folder)
  const configured = config.get<string>('pythonPath')?.trim()
  if (configured) return { managed: false, python: configured }
  if (config.get<string>('environment', 'managed') === 'managed') return { managed: true }
  const fromExtension = await fromPythonExtension(folder)
  if (fromExtension) return { managed: false, python: fromExtension }
  return { managed: false, python: process.platform === 'win32' ? 'python' : 'python3' }
}

/** Whether the Python extension's selection matters (and a change to it
 * should restart backends). */
export function usesSelectedInterpreter(): boolean {
  const config = vscode.workspace.getConfiguration('modelmaker')
  return !config.get<string>('pythonPath')?.trim() && config.get<string>('environment', 'managed') === 'selected'
}

/** The environment for processes that download (uv): VS Code's http.proxy
 * setting, unless the environment already names a proxy. */
export function downloadEnv(): NodeJS.ProcessEnv {
  const proxy = vscode.workspace.getConfiguration('http').get<string>('proxy')?.trim()
  const env = { ...process.env }
  if (proxy && !env.HTTPS_PROXY && !env.https_proxy) env.HTTPS_PROXY = env.HTTP_PROXY = proxy
  return env
}

async function fromPythonExtension(folder: vscode.Uri): Promise<string | undefined> {
  if (!vscode.extensions.getExtension('ms-python.python')) return undefined
  try {
    const { PythonExtension } = await import('@vscode/python-extension')
    const api = await PythonExtension.api()
    const path = api.environments.getActiveEnvironmentPath(folder)
    const env = await api.environments.resolveEnvironment(path)
    return env?.executable.uri?.fsPath
  } catch {
    return undefined
  }
}

/** Fires when the Python extension's interpreter selection changes. */
export async function onPythonChanged(listener: () => void): Promise<vscode.Disposable> {
  if (!vscode.extensions.getExtension('ms-python.python')) return new vscode.Disposable(() => {})
  try {
    const { PythonExtension } = await import('@vscode/python-extension')
    const api = await PythonExtension.api()
    return api.environments.onDidChangeActiveEnvironmentPath(listener)
  } catch {
    return new vscode.Disposable(() => {})
  }
}
