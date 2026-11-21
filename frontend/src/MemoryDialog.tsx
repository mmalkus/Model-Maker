import { useEffect, useRef } from 'react'
import { api } from './api'
import type { BlockOut, PreflightOut, RunMode } from './types'

/** The pieces memoryGuard.tsx shows: the prompt before a run that won't fit,
 * and the same offer under a block stopped for running out of memory. */

export type MemoryChoice = 'stream' | 'sample' | 'run' | 'cancel'

const fmtRows = (n: number, exact: boolean) =>
  (exact ? '' : 'about ') + (n >= 1e6 ? `${(n / 1e6).toLocaleString(undefined, { maximumFractionDigits: 1 })}M` : n.toLocaleString())

export function MemoryDialog({ pf, mode, onChoose }: { pf: PreflightOut; mode: RunMode; onChoose: (c: MemoryChoice) => void }) {
  const streaming = mode === 'run_all_streaming'
  const canStream = !streaming && pf.blocks.every((b) => b.can_stream)
  const whole = [...new Set(pf.blocks.flatMap((b) => b.whole_consumers))]
  const first = useRef<HTMLButtonElement>(null)
  useEffect(() => {
    first.current?.focus()
    const onKey = (e: KeyboardEvent) => e.key === 'Escape' && onChoose('cancel')
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [onChoose])

  return (
    <div style={{ position: 'fixed', inset: 0, background: 'rgba(11,35,64,0.35)', zIndex: 60, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
      <div
        role="alertdialog"
        aria-label="This run may run out of memory"
        style={{ width: 560, maxWidth: 'calc(100vw - 32px)', background: 'var(--mm-bg)', borderRadius: 12, padding: '14px 18px', fontSize: 13 }}
      >
        <h2 style={{ margin: '0 0 6px', fontFamily: 'var(--font-heading)', fontSize: 18, color: 'var(--brand-ink)' }}>
          This run is likely to run out of memory
        </h2>
        <p style={{ margin: '0 0 8px', color: 'var(--mm-muted)' }}>
          {streaming
            ? 'Even streaming, these sources are loaded whole, because a block they feed takes its input whole (or they feed nothing):'
            : 'This run loads each block’s whole output into memory, and these sources won’t fit:'}
        </p>
        <ul style={{ margin: '0 0 8px', paddingLeft: 18 }}>
          {pf.blocks.map((b) => (
            <li key={b.id}>
              <strong>{b.name}</strong>: {fmtRows(b.rows, b.exact)} rows, about {b.size} in memory
            </li>
          ))}
        </ul>
        <p style={{ margin: '0 0 8px', color: 'var(--mm-muted)', fontSize: 12 }}>
          The limit for one block is {pf.limit} ({pf.limit_basis}).
        </p>
        {canStream && (
          <p style={{ margin: '0 0 8px' }}>
            <strong>Run all (streaming)</strong> reads them in batches through the blocks they feed, without loading them whole.
          </p>
        )}
        {!canStream && whole.length > 0 && (
          <p style={{ margin: '0 0 8px' }}>
            {whole.join(', ')} take{whole.length === 1 ? 's' : ''} the whole table: reduce it before (select, filter, aggregate or sample
            whole IDs) so streaming can help.
          </p>
        )}
        <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap', justifyContent: 'flex-end', marginTop: 12 }}>
          {canStream && (
            <button ref={first} className="brand-primary" onClick={() => onChoose('stream')}>
              Run all (streaming)
            </button>
          )}
          <button ref={canStream ? undefined : first} className={canStream ? undefined : 'brand-primary'} onClick={() => onChoose('sample')}>
            Sample mode ({pf.sample_rows.toLocaleString()} rows)
          </button>
          <button onClick={() => onChoose('run')} title="Start the run as asked; a block that doesn't fit is stopped with an error">
            Run anyway
          </button>
          <button onClick={() => onChoose('cancel')}>Cancel</button>
        </div>
      </div>
    </div>
  )
}

/** Under a red block's error, when it was stopped for running out of
 * memory: what to do instead, as buttons. */
export function MemoryAdvice({ block, run }: { block: BlockOut; run: (fn: () => Promise<unknown>) => unknown }) {
  const advice = block.memory_advice
  if (!advice) return null
  return (
    <div style={{ background: 'var(--mm-warn-bg)', color: 'var(--mm-warn)', padding: 8, borderRadius: 6, marginBottom: 8, fontSize: 12 }}>
      <div style={{ marginBottom: 6 }}>
        {advice.can_stream
          ? 'Every block this one feeds takes it lazily, so Run all (streaming) reads it in batches without loading it whole.'
          : advice.whole_consumers.length > 0
            ? `${advice.whole_consumers.join(', ')} take${advice.whole_consumers.length === 1 ? 's' : ''} this output whole, so even a streaming run loads it: reduce it first (select, filter, aggregate or sample whole IDs).`
            : 'Its output is loaded whole: reduce it inside the block, or feed it to a block that takes it lazily and use Run all (streaming).'}
      </div>
      <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap' }}>
        {advice.can_stream && <button onClick={() => run(() => api.runAllStreaming())}>Run all (streaming)</button>}
        <button onClick={() => run(() => api.setSampleMode(advice.sample_rows))}>Sample mode ({advice.sample_rows.toLocaleString()} rows)</button>
      </div>
    </div>
  )
}
