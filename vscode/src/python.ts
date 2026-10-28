import * as vscode from 'vscode'

/** The interpreter that runs a project's backend: the modelmaker.pythonPath
 * setting, else the Python extension's selection for that folder, else
 * python3/python on PATH. */
export async function resolvePython(folder: vscode.Uri): Promise<string> {
  const configured = vscode.workspace.getConfiguration('modelmaker', folder).get<string>('pythonPath')?.trim()
  if (configured) return configured
  const fromExtension = await fromPythonExtension(folder)
  if (fromExtension) return fromExtension
  return process.platform === 'win32' ? 'python' : 'python3'
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
