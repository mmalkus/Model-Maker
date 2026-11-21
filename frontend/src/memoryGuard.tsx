import { createRoot } from 'react-dom/client'
import { api } from './api'
import { type MemoryChoice, MemoryDialog } from './MemoryDialog'
import type { PreflightOut, RunMode } from './types'

/** Out-of-memory guarding for runs, in the UI (see modelmaker/preflight.py).
 *
 * guardedRun asks the backend, before a run starts, whether any input block
 * it would load whole is bigger than the memory the machine can spare. Only
 * then is the user asked -- with Run all (streaming) offered where it would
 * avoid loading the block, sample mode, or run anyway -- so a run that fits
 * never sees a prompt. MemoryAdvice (MemoryDialog.tsx) is the same offer on
 * a block that was stopped for running out of memory. */

/** Run `start` unless the preflight finds a block that won't fit -- then
 * whatever the user picks instead. A preflight that fails (an older backend,
 * a file that can't be read) never blocks the run: the run itself reports. */
export async function guardedRun(mode: RunMode, blockId: string | undefined, start: () => Promise<unknown>): Promise<unknown> {
  const pf = await api.runPreflight(mode, blockId).catch(() => null)
  if (!pf || pf.blocks.length === 0) return start()
  const choice = await askMemory(pf, mode)
  if (choice === 'stream') return api.runAllStreaming()
  if (choice === 'sample') {
    await api.setSampleMode(pf.sample_rows)
    return start()
  }
  if (choice === 'run') return start()
  return undefined
}

function askMemory(pf: PreflightOut, mode: RunMode): Promise<MemoryChoice> {
  return new Promise((resolve) => {
    const el = document.createElement('div')
    document.body.appendChild(el)
    const root = createRoot(el)
    const done = (c: MemoryChoice) => {
      resolve(c)
      // Not from inside the dialog's own click handler.
      queueMicrotask(() => {
        root.unmount()
        el.remove()
      })
    }
    root.render(<MemoryDialog pf={pf} mode={mode} onChoose={done} />)
  })
}
