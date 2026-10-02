import { useEffect, useState } from 'react'
import { api, buildReportId } from './api'
import type { Artifact, ArtifactSummary } from './types'

// An AI build's full report (see controller._attach_report) is a persistent
// Artifact of kind "build_report", one per build. These are the two ways
// to get at it: the modal that shows one, and the list of every report in
// the project -- reachable from the build panel and from the AI badge of
// any block a build created or changed (see BlockNode).

export function ReportModal({ buildId, onClose }: { buildId: string; onClose: () => void }) {
  const [artifact, setArtifact] = useState<Artifact | null>(null)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    // App keys this modal on buildId, so a different build gets a fresh
    // instance rather than a reset here.
    let cancelled = false
    api
      .getArtifact(buildReportId(buildId))
      .then((a) => {
        if (!cancelled) setArtifact(a)
      })
      .catch(() => {
        // A build that was stopped, failed or discarded never writes one,
        // and undoing a build removes its report along with its blocks.
        if (!cancelled) setError(`No report is saved for build ${buildId}. Builds that stop, fail or are undone don't keep one.`)
      })
    return () => {
      cancelled = true
    }
  }, [buildId])

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') onClose()
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [onClose])

  const download = () => {
    if (!artifact) return
    const url = URL.createObjectURL(new Blob([artifact.document], { type: 'text/markdown' }))
    const a = document.createElement('a')
    a.href = url
    a.download = `build-report-${buildId}.md`
    a.click()
    URL.revokeObjectURL(url)
  }

  return (
    <div
      className="nodrag nopan"
      onClick={onClose}
      style={{ position: 'fixed', inset: 0, background: 'rgba(0,0,0,0.3)', zIndex: 60, display: 'flex', alignItems: 'center', justifyContent: 'center' }}
    >
      <div
        onClick={(e) => e.stopPropagation()}
        style={{
          background: '#fff',
          border: '1px solid #d1d5db',
          borderRadius: 8,
          padding: 16,
          width: 'min(760px, calc(100vw - 48px))',
          maxHeight: 'calc(100vh - 48px)',
          display: 'flex',
          flexDirection: 'column',
          boxShadow: '0 10px 30px rgba(0,0,0,0.25)',
          fontSize: 13,
        }}
      >
        <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
          <strong style={{ flex: 1, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{artifact?.title ?? 'AI build report'}</strong>
          {artifact?.stale && (
            <span
              title="The block this report is attached to has changed since the report was written, so its numbers may be out of date"
              style={{ fontSize: 11, color: '#b45309', background: '#fef3c7', borderRadius: 4, padding: '1px 6px' }}
            >
              stale
            </span>
          )}
          <button disabled={!artifact} onClick={() => artifact && navigator.clipboard.writeText(artifact.document)} title="Copy the report as Markdown">
            Copy
          </button>
          <button disabled={!artifact} onClick={download} title="Download the report as a .md file">
            Download .md
          </button>
          <button onClick={onClose} title="Close (Esc)">
            ×
          </button>
        </div>
        {artifact && (
          <div style={{ fontSize: 11, color: '#6b7280', marginTop: 2 }}>
            Attached to {artifact.block_name ? `${artifact.block_name}.${artifact.port}` : 'a block that has since been deleted'} · written{' '}
            {new Date(artifact.updated_at).toLocaleString()}
          </div>
        )}
        <div style={{ borderTop: '1px solid #e5e7eb', marginTop: 8, paddingTop: 8, overflowY: 'auto' }}>
          {error && <div style={{ color: '#6b7280' }}>{error}</div>}
          {!artifact && !error && <div style={{ color: '#6b7280' }}>Loading…</div>}
          {artifact && <MarkdownLite text={artifact.document} />}
        </div>
      </div>
    </div>
  )
}

// Just the Markdown a build report is written in (see
// controller._attach_report, plus whatever the LLM puts in its own
// summary): headings, bullets, rules, **bold** and `code`. Anything else
// shows as plain text, which is still readable.
function MarkdownLite({ text }: { text: string }) {
  const inline = (line: string) =>
    line.split(/(\*\*[^*]+\*\*|`[^`]+`)/g).map((part, i) =>
      part.startsWith('**') && part.endsWith('**') && part.length > 4 ? (
        <strong key={i}>{part.slice(2, -2)}</strong>
      ) : part.startsWith('`') && part.endsWith('`') && part.length > 2 ? (
        <code key={i} style={{ background: '#f3f4f6', borderRadius: 3, padding: '0 3px', fontSize: 12, wordBreak: 'break-word' }}>
          {part.slice(1, -1)}
        </code>
      ) : (
        part
      ),
    )
  return (
    <div style={{ lineHeight: 1.5 }}>
      {text.split('\n').map((line, i) => {
        const heading = /^(#{1,3})\s+(.*)$/.exec(line)
        if (heading) {
          const size = [17, 15, 13][heading[1].length - 1]
          return (
            <div key={i} style={{ fontSize: size, fontWeight: 600, margin: i === 0 ? '0 0 4px' : '10px 0 4px' }}>
              {inline(heading[2])}
            </div>
          )
        }
        if (/^\s*([-*_])\1{2,}\s*$/.test(line)) return <hr key={i} style={{ border: 'none', borderTop: '1px solid #e5e7eb', margin: '10px 0' }} />
        const bullet = /^(\s*)[-*]\s+(.*)$/.exec(line)
        if (bullet) {
          return (
            <div key={i} style={{ display: 'flex', gap: 6, paddingLeft: 4 + bullet[1].length * 8 }}>
              <span style={{ color: '#9ca3af' }}>•</span>
              <span style={{ minWidth: 0 }}>{inline(bullet[2])}</span>
            </div>
          )
        }
        if (!line.trim()) return <div key={i} style={{ height: 6 }} />
        return <div key={i}>{inline(line)}</div>
      })}
    </div>
  )
}

const REPORT_PREFIX = buildReportId('')

// Every build report in the project, newest first. `refreshKey` changes
// whenever reports may have come or gone (a build finishing, an undo or
// redo); the list hides itself when there are none.
export function BuildReportList({ refreshKey, onOpen }: { refreshKey: string; onOpen: (buildId: string) => void }) {
  const [reports, setReports] = useState<ArtifactSummary[]>([])

  useEffect(() => {
    let cancelled = false
    api
      .listArtifacts()
      .then((all) => {
        if (!cancelled) setReports(all.filter((a) => a.kind === 'build_report' && a.id.startsWith(REPORT_PREFIX)))
      })
      .catch(() => {})
    return () => {
      cancelled = true
    }
  }, [refreshKey])

  if (reports.length === 0) return null
  return (
    <div style={{ border: '1px solid #e5e7eb', borderRadius: 6, padding: 8, marginBottom: 8 }}>
      <div style={{ fontSize: 11, color: '#6b7280', fontWeight: 600, marginBottom: 4 }}>Build reports</div>
      {reports.map((r) => (
        <button
          key={r.id}
          onClick={() => onOpen(r.id.slice(REPORT_PREFIX.length))}
          title={`Open this build's full report${r.block_name ? '' : ' (the block it was attached to has been deleted)'}`}
          style={{
            display: 'flex',
            alignItems: 'baseline',
            gap: 6,
            width: '100%',
            textAlign: 'left',
            background: 'none',
            border: 'none',
            borderTop: '1px solid #f3f4f6',
            padding: '4px 0',
            cursor: 'pointer',
            fontSize: 12,
          }}
        >
          <span style={{ flex: 1, minWidth: 0, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
            {r.title.replace(/^AI build report -- /, '')}
          </span>
          {r.stale && <span style={{ fontSize: 10, color: '#b45309' }}>stale</span>}
          {!r.block_name && <span style={{ fontSize: 10, color: '#9ca3af' }}>block deleted</span>}
          <span style={{ fontSize: 10, color: '#9ca3af', flexShrink: 0 }}>{new Date(r.created_at).toLocaleDateString()}</span>
        </button>
      ))}
    </div>
  )
}
