import { useState } from 'react'
import { createPortal } from 'react-dom'
import { api } from './api'
import { DataModal } from './DataModal'
import type { BlockDisplay, BlockOut, DisplayCell, PreviewOut } from './types'

// A display block's result, drawn right on its canvas node (see
// api._display_out): a small table, one headline number, a few key/value
// pairs or an image -- no click needed to see it. A stale (orange) block's
// result is dimmed rather than hidden, so the last value stays readable
// while it waits for a re-run.

function formatCell(v: DisplayCell): string {
  if (v === null) return '—'
  if (typeof v === 'number') {
    if (Number.isInteger(v)) return v.toLocaleString()
    return v.toLocaleString(undefined, { maximumFractionDigits: Math.abs(v) < 1 ? 4 : 2 })
  }
  return String(v)
}

function formatKpi(display: Extract<BlockDisplay, { kind: 'kpi' }>): string {
  const { value, number_format = 'number', decimals = 2, unit } = display
  if (value === null || value === undefined) return '—'
  const digits = Math.max(0, Math.min(10, decimals))
  const opts: Intl.NumberFormatOptions = { minimumFractionDigits: digits, maximumFractionDigits: digits }
  const text =
    number_format === 'percent'
      ? `${(value * 100).toLocaleString(undefined, opts)}%`
      : number_format === 'compact'
        ? value.toLocaleString(undefined, { notation: 'compact', maximumFractionDigits: digits })
        : value.toLocaleString(undefined, opts)
  return unit ? `${text} ${unit}` : text
}

const cellStyle = { padding: '2px 6px', borderBottom: '1px solid #f3f4f6', whiteSpace: 'nowrap' } as const

export function NodeDisplay({ block, display, onChanged }: { block: BlockOut; display: BlockDisplay; onChanged?: () => void }) {
  // The full table, opened straight from the node -- portalled out of the
  // canvas, whose transform would otherwise trap the fixed-position modal.
  const [fullTable, setFullTable] = useState<PreviewOut | null>(null)
  const openTable = () => {
    const port = block.outputs[0]?.name
    if (!port) return
    api
      .preview(block.id, { port, rows: 200, summary: true })
      .then(setFullTable)
      .catch((e) => alert((e as Error).message))
  }
  const stale = block.status === 'orange'
  const wrap = { marginTop: 6, opacity: stale ? 0.55 : 1, color: '#111827' }

  if (display.kind === 'kpi') {
    return (
      <div style={{ ...wrap, textAlign: 'center', padding: '4px 0' }} title={stale ? 'Stale: re-run to update' : undefined}>
        <div style={{ fontSize: 11, color: '#6b7280' }}>{display.label}</div>
        <div style={{ fontSize: 22, fontWeight: 700, fontVariantNumeric: 'tabular-nums', lineHeight: 1.2 }}>
          {formatKpi(display)}
        </div>
      </div>
    )
  }

  if (display.kind === 'image') {
    return (
      <img
        src={api.imageUrl(block.id, display.port, block.last_attempt_at)}
        alt={block.name}
        style={{ ...wrap, display: 'block', maxWidth: '100%', borderRadius: 4 }}
      />
    )
  }

  if (display.kind === 'value') {
    if (display.text !== undefined) {
      return (
        <pre style={{ ...wrap, fontSize: 10, margin: '6px 0 0', maxHeight: 160, overflow: 'auto', whiteSpace: 'pre-wrap' }} className="nowheel">
          {display.text}
        </pre>
      )
    }
    return (
      <table style={{ ...wrap, borderCollapse: 'collapse', fontSize: 11, width: '100%' }}>
        <tbody>
          {(display.items ?? []).map(([k, v]) => (
            <tr key={k}>
              <td style={{ ...cellStyle, color: '#6b7280' }}>{k}</td>
              <td style={{ ...cellStyle, textAlign: typeof v === 'number' ? 'right' : 'left', fontVariantNumeric: 'tabular-nums' }}>
                {formatCell(v)}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    )
  }

  const hiddenRows = display.row_count - display.rows.length
  const hiddenCols = display.column_count - display.columns.length
  return (
    <div style={wrap}>
      <div className="nowheel" style={{ overflowX: 'auto', border: '1px solid #e5e7eb', borderRadius: 4 }}>
        <table style={{ borderCollapse: 'collapse', fontSize: 10.5, width: '100%' }}>
          <thead>
            <tr style={{ background: '#f9fafb' }}>
              {display.columns.map((c) => (
                <th key={c.name} title={`${c.name} (${c.dtype})`} style={{ ...cellStyle, textAlign: 'left', fontWeight: 600 }}>
                  {c.name}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {display.rows.map((row, i) => (
              <tr key={i}>
                {row.map((v, j) => (
                  <td
                    key={j}
                    style={{ ...cellStyle, textAlign: typeof v === 'number' ? 'right' : 'left', fontVariantNumeric: 'tabular-nums' }}
                  >
                    {formatCell(v)}
                  </td>
                ))}
              </tr>
            ))}
            {display.rows.length === 0 && (
              <tr>
                <td colSpan={display.columns.length || 1} style={{ ...cellStyle, color: '#9ca3af' }}>
                  (no rows)
                </td>
              </tr>
            )}
          </tbody>
        </table>
      </div>
      <div style={{ display: 'flex', gap: 6, marginTop: 3, fontSize: 10, color: '#6b7280' }}>
        <span>
          {display.row_count.toLocaleString()} row{display.row_count === 1 ? '' : 's'}
          {hiddenRows > 0 && ` · first ${display.rows.length} shown`}
          {hiddenCols > 0 && ` · ${hiddenCols} more column${hiddenCols === 1 ? '' : 's'}`}
        </span>
        {(hiddenRows > 0 || hiddenCols > 0) && (
          <button
            className="nodrag"
            onClick={(e) => {
              e.stopPropagation()
              openTable()
            }}
            style={{ marginLeft: 'auto', border: 'none', background: 'none', padding: 0, color: '#2563eb', cursor: 'pointer', fontSize: 10 }}
          >
            all rows…
          </button>
        )}
      </div>
      {fullTable &&
        createPortal(
          <DataModal
            blockName={block.name}
            blockId={block.id}
            preview={fullTable}
            onClose={() => setFullTable(null)}
            onChanged={onChanged}
          />,
          document.body,
        )}
    </div>
  )
}
