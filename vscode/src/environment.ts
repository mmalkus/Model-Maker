import * as path from 'node:path'
import * as vscode from 'vscode'
import {
  type Bundle,
  type Candidate,
  defaultCandidates,
  ensureManagedEnv,
  ensureSource,
  EnvironmentError,
  findPython,
  invalidateManagedEnv,
  minimumPython,
  missingRequirements,
  readBundle,
  run,
} from './pyenv'
import { resolvePython } from './python'

// Which Python runs a project's backend, and with which Model-Maker
// (modelmaker.python.environment):
//
// - managed (default): the extension's own environment (made from any
//   Python 3.11+, holding only Model-Maker's open-source requirements), with
//   the Model-Maker that ships in the extension;
// - interpreter: the interpreter you selected, with the shipped Model-Maker;
//   missing requirements are offered for install into it;
// - installedPackage: the interpreter you selected, with the
//   quantology-modelmaker installed in it (e.g. `pip install -e` of a
//   checkout, for developing Model-Maker itself).
//
// The first two never pip install a Model-Maker package. See pyenv.ts.

export type Mode = 'managed' | 'interpreter' | 'installedPackage'

/** How to run Model-Maker: `python` with these extra environment variables. */
export type Runtime = { python: string; env: Record<string, string>; mode: Mode; version?: string }

type Progress = vscode.Progress<{ message?: string }>

export class PythonEnvironments {
  private bundle: Promise<Bundle | undefined>
  // One setup at a time in this window; pyenv's lock covers other windows.
  private managed: Promise<string> | undefined

  constructor(
    private readonly context: vscode.ExtensionContext,
    private readonly log: vscode.OutputChannel,
  ) {
    this.bundle = readBundle(this.bundleDir)
  }

  private get bundleDir() {
    return path.join(this.context.extensionPath, 'python')
  }

  private get storage() {
    return this.context.globalStorageUri.fsPath
  }

  private get envDir() {
    return path.join(this.storage, 'env')
  }

  mode(folder?: vscode.Uri): Mode {
    return vscode.workspace.getConfiguration('modelmaker', folder).get<Mode>('python.environment', 'managed')
  }

  /** Gets the Python ready for `folder`'s backend (creating the managed
   * environment the first time) and says how to run Model-Maker in it. */
  async runtime(folder: vscode.Uri, progress?: Progress): Promise<Runtime> {
    const report = (message: string) => progress?.report({ message })
    const mode = this.mode(folder)
    if (mode === 'installedPackage') return { python: await resolvePython(folder), env: {}, mode }
    const bundle = await this.bundle
    if (!bundle) {
      this.log.appendLine('[environment] this build of the extension has no bundled Model-Maker (a development build?): using the installed package')
      return { python: await resolvePython(folder), env: {}, mode: 'installedPackage' }
    }
    try {
      const src = await ensureSource(this.bundleDir, path.join(this.storage, 'src'))
      const env = { PYTHONPATH: [src, process.env.PYTHONPATH].filter(Boolean).join(path.delimiter) }
      const python = mode === 'managed' ? await this.managedPython(folder, bundle, report) : await this.interpreter(folder, bundle, report)
      return { python, env, mode, version: bundle.version }
    } catch (e) {
      if (e instanceof EnvironmentError) throw e
      throw new EnvironmentError(`couldn't prepare Model-Maker's Python environment: ${e instanceof Error ? e.message : e}`)
    }
  }

  /** Model-Maker: Repair Python Environment. */
  async repair() {
    this.managed = undefined
    await invalidateManagedEnv(this.envDir)
  }

  private managedPython(folder: vscode.Uri, bundle: Bundle, report: (m: string) => void): Promise<string> {
    if (!this.managed) {
      this.managed = ensureManagedEnv({
        envDir: this.envDir,
        requirements: bundle.requirements,
        base: () => findPython(this.candidates(folder), minimumPython(bundle.requires_python)),
        log: (t) => this.log.append(`[environment] ${t}`),
        report,
      })
      this.managed.catch(() => (this.managed = undefined))
    }
    return this.managed
  }

  /** The Pythons the managed environment may be made from, in order. */
  private candidates(folder: vscode.Uri): Candidate[] {
    const base = vscode.workspace.getConfiguration('modelmaker', folder).get<string>('basePython')?.trim()
    if (base) return [{ command: base, label: `modelmaker.basePython (${base})` }]
    const configured = vscode.workspace.getConfiguration('modelmaker', folder).get<string>('pythonPath')?.trim()
    return [...(configured ? [{ command: configured }] : []), ...defaultCandidates()]
  }

  private async interpreter(folder: vscode.Uri, bundle: Bundle, report: (m: string) => void): Promise<string> {
    const python = await resolvePython(folder)
    const min = minimumPython(bundle.requires_python)
    await findPython([{ command: python }], min)
    report('checking the Python packages')
    const missing = await missingRequirements(python, bundle.requirements)
    if (missing.length === 0) return python
    const mode = vscode.workspace.getConfiguration('modelmaker').get<string>('autoInstall', 'ask')
    const why = `Model-Maker needs ${missing.length === 1 ? 'a package' : 'packages'} that ${python} doesn't have: ${missing.join(', ')}.`
    if (mode === 'never') throw new EnvironmentError(`${why} Install them yourself, or set modelmaker.autoInstall.`)
    if (mode === 'ask') {
      const choice = await vscode.window.showInformationMessage(`${why} Install them now?`, { modal: true }, 'Install', 'Use the Managed Environment')
      if (choice === 'Use the Managed Environment') {
        await vscode.workspace.getConfiguration('modelmaker').update('python.environment', 'managed', vscode.ConfigurationTarget.Global)
        return this.managedPython(folder, bundle, report)
      }
      if (choice !== 'Install') throw new EnvironmentError(`${why} Not installed.`)
    }
    report(`installing ${missing.join(', ')}`)
    this.log.appendLine(`[environment] $ ${python} -m pip install ${missing.join(' ')}`)
    const { code, out } = await run(python, ['-m', 'pip', 'install', '--disable-pip-version-check', ...missing], (t) => this.log.append(t))
    if (code !== 0) {
      const hint = /externally-managed-environment/.test(out)
        ? ' This Python is managed by the OS: use the managed environment (modelmaker.python.environment) or a virtual environment.'
        : ''
      throw new EnvironmentError(`pip install failed (exit ${code}).${hint} See Model-Maker: Show Backend Log.`)
    }
    return python
  }
}
