import { createHash } from 'node:crypto'
import { chmodSync, existsSync, mkdirSync, renameSync, writeFileSync } from 'node:fs'
import * as path from 'node:path'
import { extractEntry } from './zip'

// uv (https://docs.astral.sh/uv/), which sets up the managed environment
// (managed.ts). It comes from its PyPI wheel rather than a GitHub release,
// so a first run needs the same hosts installing the package does, and the
// download is checked against PyPI's published sha256. Pinned: a newer uv
// is taken on purpose, with an extension release.
export const UV_VERSION = '0.13.2'

export type WheelFile = { filename: string; url: string; digests: { sha256: string } }

const WINDOWS: Record<string, string> = { x64: 'win_amd64', arm64: 'win_arm64', ia32: 'win32' }
const MAC: Record<string, string> = { x64: 'x86_64', arm64: 'arm64' }
const LINUX: Record<string, string> = {
  x64: 'x86_64',
  arm64: 'aarch64',
  ia32: 'i686',
  arm: 'armv7l',
  ppc64: 'ppc64le',
  s390x: 's390x',
  riscv64: 'riscv64',
}

/** The wheel holding the uv binary for this machine. On Linux, the lowest
 * glibc baseline (most compatible), or a musllinux wheel on musl. */
export function pickUvWheel(files: WheelFile[], platform: string, arch: string, musl = false): WheelFile {
  const wheels = files.filter((f) => f.filename.endsWith('.whl'))
  let match: WheelFile | undefined
  if (platform === 'win32') {
    match = wheels.find((f) => f.filename.endsWith(`-${WINDOWS[arch]}.whl`))
  } else if (platform === 'darwin') {
    match = wheels.find((f) => f.filename.includes('macosx') && f.filename.includes(`_${MAC[arch]}`))
  } else if (platform === 'linux' && LINUX[arch]) {
    const candidates = wheels.filter((f) => f.filename.includes(`_${LINUX[arch]}`))
    const glibc = (f: WheelFile) => Number(/manylinux_2_(\d+)/.exec(f.filename)?.[1] ?? Infinity)
    match = musl
      ? candidates.find((f) => f.filename.includes('musllinux'))
      : candidates.filter((f) => f.filename.includes('manylinux')).sort((a, b) => glibc(a) - glibc(b))[0]
  }
  if (!match) throw new Error(`uv ${UV_VERSION} has no build for ${platform}/${arch}${musl ? ' (musl)' : ''}`)
  return match
}

/** Whether this Linux uses musl rather than glibc (Alpine, some containers). */
export function isMusl(): boolean {
  if (process.platform !== 'linux') return false
  const header = (process.report?.getReport() as { header?: { glibcVersionRuntime?: string } } | undefined)?.header
  return !header?.glibcVersionRuntime
}

/** The uv binary in `dir`, downloaded first if it isn't there yet. */
export async function ensureUv(dir: string, report: (message: string) => void, log: (line: string) => void): Promise<string> {
  const binary = path.join(dir, process.platform === 'win32' ? 'uv.exe' : 'uv')
  if (existsSync(binary)) return binary
  report(`downloading uv ${UV_VERSION}`)
  const meta = await fetch(`https://pypi.org/pypi/uv/${UV_VERSION}/json`, { signal: AbortSignal.timeout(60_000) })
  if (!meta.ok) throw new Error(`PyPI answered HTTP ${meta.status} for uv ${UV_VERSION}`)
  const wheel = pickUvWheel(((await meta.json()) as { urls: WheelFile[] }).urls, process.platform, process.arch, isMusl())
  log(`downloading ${wheel.url}\n`)
  const data = await download(wheel.url, (got, total) =>
    report(`downloading uv ${UV_VERSION} (${mb(got)}${total ? ` of ${mb(total)}` : ''} MB)`),
  )
  const sha256 = createHash('sha256').update(data).digest('hex')
  if (sha256 !== wheel.digests.sha256) throw new Error(`${wheel.filename} doesn't match PyPI's checksum -- not using it`)
  const exe = extractEntry(data, `uv-${UV_VERSION}.data/scripts/${path.basename(binary)}`)
  mkdirSync(dir, { recursive: true })
  // Written aside, then renamed: another window never sees half a binary.
  const partial = `${binary}.${process.pid}.partial`
  writeFileSync(partial, exe)
  chmodSync(partial, 0o755)
  renameSync(partial, binary)
  return binary
}

// A download with no data for this long has stalled: fail instead of hanging.
const STALL_MS = 60_000

const mb = (bytes: number) => (bytes / 1e6).toFixed(0)

/** Fetches `url` into memory, reporting progress, and failing if it stalls. */
export async function download(url: string, progress: (got: number, total?: number) => void): Promise<Buffer> {
  const abort = new AbortController()
  let stall = setTimeout(() => abort.abort(), STALL_MS)
  try {
    const res = await fetch(url, { signal: abort.signal })
    if (!res.ok || !res.body) throw new Error(`download failed: HTTP ${res.status} from ${url}`)
    const total = Number(res.headers.get('content-length')) || undefined
    const chunks: Uint8Array[] = []
    let got = 0
    let reported = 0
    const reader = res.body.getReader()
    for (;;) {
      const { done, value } = await reader.read()
      if (done) break
      clearTimeout(stall)
      stall = setTimeout(() => abort.abort(), STALL_MS)
      chunks.push(value)
      got += value.length
      if (got - reported > 1e6) {
        reported = got
        progress(got, total)
      }
    }
    return Buffer.concat(chunks)
  } catch (e) {
    if (abort.signal.aborted) throw new Error(`the download stalled (no data for ${STALL_MS / 1000}s): ${url}`)
    throw e
  } finally {
    clearTimeout(stall)
  }
}
