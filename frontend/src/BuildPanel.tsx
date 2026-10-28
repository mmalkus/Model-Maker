import { useCallback, useEffect, useRef, useState } from 'react'
import { api } from './api'
import { BuildReportList } from './BuildReport'
import type { BuildEvent, BuildOut, BuildPlan, BuildStage, GraphOut, LLMSettingsOut } from './types'
import { host } from './host'

// The AI model builder's side panel (see agent-builder-proposal.md §2):
// start a build from the selected blocks, review preflight, review and
// steer the outline plan (its new lanes shown as ghost bands on the canvas
// by App), watch each stage being planned (as ghost blocks) and built,
// review each stage, answer the build's questions, read its report.

const TERMINAL = new Set(['done', 'done_with_errors', 'stopped', 'failed', 'discarded'])
const WORKING = new Set(['planning', 'building', 'final_run'])

const PHASE_LABEL: Record<string, string> = {
  preflight: 'Checking the data',
  planning: 'Planning…',
  awaiting_approval: 'Plan ready for review',
  building: 'Building…',
  awaiting_input: 'Waiting for your answer',
  awaiting_stage_review: 'Stage ready for review',
  final_run: 'Running on the full data…',
  done: 'Done',
  done_with_errors: 'Done, with errors',
  stopped: 'Stopped',
  failed: 'Failed',
  discarded: 'Discarded',
}

const box: React.CSSProperties = { border: '1px solid var(--mm-border)', borderRadius: 6, padding: 8, marginBottom: 8 }
const label: React.CSSProperties = { fontSize: 11, color: 'var(--mm-muted)', fontWeight: 600, marginBottom: 4 }

type Props = {
  graph: GraphOut
  selectedIds: string[]
  llmSettings: LLMSettingsOut | null
  onChanged: () => void
  onBuild: (build: BuildOut | null) => void
  onClose: () => void
  onOpenReport: (buildId: string) => void
}

export function BuildPanel({ graph, selectedIds, llmSettings, onChanged, onBuild, onClose, onOpenReport }: Props) {
  const [build, setBuild] = useState<BuildOut | null>(null)
  const [events, setEvents] = useState<BuildEvent[]>([])
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const cursor = useRef(0)
  const buildId = useRef<string | null>(null)

  const apply = useCallback(
    (b: BuildOut | null, isBusy?: boolean) => {
      if (!b) {
        setBuild(null)
        onBuild(null)
        return
      }
      if (b.id !== buildId.current) {
        buildId.current = b.id
        cursor.current = 0
        setEvents(b.events)
      } else {
        setEvents((prev) => [...prev, ...b.events.filter((e) => e.seq >= prev.length)])
      }
      cursor.current = b.next_cursor
      setBuild(b)
      onBuild(b)
      if (isBusy !== undefined) setBusy(isBusy)
    },
    [onBuild],
  )

  const poll = useCallback(() => {
    api
      .agentCurrent(cursor.current)
      .then((s) => {
        apply(s.build, s.busy)
        onChanged()
      })
      .catch(() => {})
  }, [apply, onChanged])

  useEffect(() => {
    api
      .agentCurrent(0)
      .then((s) => apply(s.build, s.busy))
      .catch(() => {})
  }, [apply])

  const live = build && !TERMINAL.has(build.phase) && (busy || WORKING.has(build.phase))
  useEffect(() => {
    if (!live) return
    const id = setInterval(poll, 1000)
    return () => clearInterval(id)
  }, [live, poll])

  const act = useCallback(
    (fn: () => Promise<BuildOut>) => {
      setError(null)
      fn()
        .then((b) => {
          apply(b, true)
          onChanged()
          setTimeout(poll, 300)
        })
        .catch((e) => setError((e as Error).message))
    },
    [apply, onChanged, poll],
  )

  return (
    <div style={{ width: 380, borderLeft: '1px solid var(--mm-border)', padding: 12, overflowY: 'auto', fontSize: 12, display: 'flex', flexDirection: 'column' }}>
      <div style={{ display: 'flex', alignItems: 'center', marginBottom: 8 }}>
        <strong style={{ fontSize: 14 }}>Build with AI</strong>
        <span style={{ flex: 1 }} />
        <button onClick={onClose} title="Close (the build keeps going)">×</button>
      </div>
      {error && <div style={{ ...box, borderColor: 'var(--mm-error-border)', background: 'var(--mm-error-bg)', color: 'var(--mm-error)' }}>{error}</div>}

      {!build || TERMINAL.has(build.phase) ? (
        <>
          {build && <Finished build={build} graph={graph} onOpenReport={onOpenReport} />}
          <BuildReportList refreshKey={`${build?.id}:${build?.phase}:${graph.can_undo}:${graph.can_redo}`} onOpen={onOpenReport} />
          <StartForm graph={graph} selectedIds={selectedIds} llmSettings={llmSettings} onStart={(body) => act(() => api.agentStart(body))} />
        </>
      ) : (
        <>
          <Header build={build} busy={busy} />
          {build.phase === 'preflight' && <Preflight build={build} graph={graph} act={act} />}
          {build.phase === 'awaiting_approval' && <PlanReview build={build} graph={graph} act={act} busy={busy} />}
          {build.stages?.length > 0 && <Stages build={build} graph={graph} />}
          {build.phase === 'awaiting_stage_review' && <StageReview build={build} act={act} busy={busy} />}
          {build.phase === 'awaiting_input' && <Question build={build} act={act} busy={busy} />}
          <div style={{ display: 'flex', gap: 6, margin: '4px 0 8px' }}>
            {build.phase !== 'preflight' && build.phase !== 'awaiting_approval' && (
              <button onClick={() => act(() => api.agentAction('stop'))} title="Stop after the current step; keeps what's built (one Undo reverts it)">
                Stop
              </button>
            )}
            <button
              onClick={async () => {
                if (await host.confirm('Discard this build and put the graph back as it was?')) act(() => api.agentAction('discard'))
              }}
            >
              Discard
            </button>
          </div>
          <EventFeed events={events} graph={graph} />
        </>
      )}
    </div>
  )
}

// ---- start ---------------------------------------------------------------------

type StartBody = Parameters<typeof api.agentStart>[0]

function StartForm({
  graph,
  selectedIds,
  llmSettings,
  onStart,
}: {
  graph: GraphOut
  selectedIds: string[]
  llmSettings: LLMSettingsOut | null
  onStart: (body: StartBody) => void
}) {
  const [goal, setGoal] = useState('')
  const [planLlm, setPlanLlm] = useState<{ provider: string; model: string }>({ provider: '', model: '' })
  const [buildLlm, setBuildLlm] = useState<{ provider: string; model: string }>({ provider: '', model: '' })
  const [fullRun, setFullRun] = useState(true)
  const [autoBuild, setAutoBuild] = useState(false)
  const [allowCustom, setAllowCustom] = useState(true)
  const [useLibrary, setUseLibrary] = useState(true)
  const [sample, setSample] = useState<'auto' | 'off' | 'custom'>('auto')
  const [sampleRows, setSampleRows] = useState(50000)
  const anchors = selectedIds.filter((id) => graph.blocks[id])
  const capable = llmSettings?.agent.capable_providers ?? ['claude_cli', 'anthropic', 'openai', 'gemini', 'lmstudio']

  return (
    <div>
      <div style={label}>What should the AI build?</div>
      <textarea
        value={goal}
        onChange={(e) => setGoal(e.target.value)}
        rows={4}
        style={{ width: '100%', boxSizing: 'border-box' }}
        placeholder="e.g. PD model: logistic regression on WoE-transformed features, out-of-time validation on 2023, master scale with 10 grades"
      />
      <div style={{ ...label, marginTop: 8 }}>Build from (select blocks on the canvas; shift-click for several)</div>
      <div style={{ display: 'flex', flexWrap: 'wrap', gap: 4, minHeight: 22 }}>
        {anchors.length === 0 && <span style={{ color: 'var(--mm-warn-2)' }}>Nothing selected -- select your prepared input block(s).</span>}
        {anchors.map((id) => (
          <span key={id} style={{ background: 'var(--mm-bg-hover)', borderRadius: 4, padding: '2px 6px' }}>
            {graph.blocks[id].name}
          </span>
        ))}
      </div>

      <label
        style={{ display: 'flex', gap: 6, alignItems: 'center', marginTop: 8 }}
        title="Skip the reviews: the AI starts building as soon as its plan is ready, and goes straight on from one stage to the next (it still stops to ask if the plan has open questions)"
      >
        <input type="checkbox" checked={autoBuild} onChange={(e) => setAutoBuild(e.target.checked)} />
        Build automatically, without stopping for reviews
      </label>

      <details style={{ marginTop: 8 }}>
        <summary style={{ cursor: 'pointer', color: 'var(--mm-fg-4)' }}>Models and options</summary>
        <LlmPicker label="Plan LLM" value={planLlm} fallback={llmSettings?.agent.plan} capable={capable} onChange={setPlanLlm} />
        <LlmPicker label="Build LLM" value={buildLlm} fallback={llmSettings?.agent.build} capable={capable} onChange={setBuildLlm} />
        <label style={{ display: 'flex', gap: 6, alignItems: 'center', marginTop: 6 }}>
          <input type="checkbox" checked={fullRun} onChange={(e) => setFullRun(e.target.checked)} />
          Finish with a run on the full data
        </label>
        <label
          style={{ display: 'flex', gap: 6, alignItems: 'center', marginTop: 6 }}
          title="Let the AI create new blocks when no registry block does the job -- they're saved to your block library as AI drafts for you to keep or discard. Off: registry blocks only -- it asks you if the goal needs something they can't do."
        >
          <input type="checkbox" checked={allowCustom} onChange={(e) => setAllowCustom(e.target.checked)} />
          Allow new blocks (saved as library drafts)
        </label>
        <label
          style={{ display: 'flex', gap: 6, alignItems: 'center', marginTop: 6 }}
          title="Let the AI use blocks from your block library -- ones you wrote, or ones earlier AI builds created -- alongside the built-ins."
        >
          <input type="checkbox" checked={useLibrary} onChange={(e) => setUseLibrary(e.target.checked)} />
          Use my library blocks
        </label>
        <div style={{ display: 'flex', gap: 6, alignItems: 'center', marginTop: 6 }}>
          Build on a sample:
          <select value={sample} onChange={(e) => setSample(e.target.value as typeof sample)}>
            <option value="auto">automatic (large data only)</option>
            <option value="off">no, use full data</option>
            <option value="custom">yes, rows:</option>
          </select>
          {sample === 'custom' && (
            <input type="number" min={100} value={sampleRows} onChange={(e) => setSampleRows(Number(e.target.value))} style={{ width: 80 }} />
          )}
        </div>
      </details>

      <button
        className="brand-primary"
        style={{ marginTop: 10, width: '100%' }}
        disabled={!goal.trim() || anchors.length === 0}
        onClick={() =>
          onStart({
            goal,
            anchors,
            plan_llm: planLlm.provider || planLlm.model ? planLlm : undefined,
            build_llm: buildLlm.provider || buildLlm.model ? buildLlm : undefined,
            final_full_run: fullRun,
            sample_rows: sample === 'auto' ? null : sample === 'off' ? 0 : sampleRows,
            auto_build: autoBuild,
            allow_new_blocks: allowCustom,
            use_library_blocks: useLibrary,
          })
        }
      >
        {autoBuild ? 'Plan and build' : 'Plan the build'}
      </button>
      <div style={{ color: 'var(--mm-muted)', marginTop: 6 }}>
        {autoBuild
          ? 'The AI outlines the stages, then builds them all straight away -- Stop or Discard at any time, and one Undo reverts the build.'
          : 'The AI outlines the stages first and changes nothing until you approve. It then plans and builds one stage at a time, and stops for you to review each.'}{' '}
        It only sees column names, roles and summary statistics -- never rows.
      </div>
    </div>
  )
}

function LlmPicker({
  label: text,
  value,
  fallback,
  capable,
  onChange,
}: {
  label: string
  value: { provider: string; model: string }
  fallback?: { provider: string; model: string | null }
  capable: string[]
  onChange: (v: { provider: string; model: string }) => void
}) {
  return (
    <div style={{ display: 'flex', gap: 4, alignItems: 'center', marginTop: 6 }}>
      <span style={{ width: 64 }}>{text}</span>
      <select value={value.provider} onChange={(e) => onChange({ ...value, provider: e.target.value })}>
        <option value="">{fallback ? `default (${fallback.provider})` : 'default'}</option>
        {capable.map((p) => (
          <option key={p} value={p}>
            {p}
          </option>
        ))}
      </select>
      <input
        value={value.model}
        onChange={(e) => onChange({ ...value, model: e.target.value })}
        placeholder={fallback?.model ?? 'default model'}
        style={{ flex: 1, minWidth: 0 }}
      />
    </div>
  )
}

// ---- phases ----------------------------------------------------------------------

function Header({ build, busy }: { build: BuildOut; busy: boolean }) {
  return (
    <div style={{ ...box, background: 'var(--mm-ai-bg)', borderColor: 'var(--mm-ai-border-2)' }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
        {(busy || WORKING.has(build.phase)) && (
          <span style={{ width: 8, height: 8, borderRadius: 999, background: 'var(--mm-ai)', animation: 'mm-pulse 1s ease-in-out infinite' }} />
        )}
        <strong>{PHASE_LABEL[build.phase] ?? build.phase}</strong>
      </div>
      <div style={{ marginTop: 4, color: 'var(--mm-fg-4)' }}>{build.goal}</div>
      <div style={{ marginTop: 4, color: 'var(--mm-muted)', fontSize: 11 }}>
        plan: {build.plan_llm.provider}/{build.plan_llm.model ?? 'default'} · build: {build.build_llm.provider}/
        {build.build_llm.model ?? 'default'}
        {build.sample_rows_used ? ` · sample ${build.sample_rows_used.toLocaleString()} rows` : ''}
        {build.usage.cost_usd ? ` · $${build.usage.cost_usd.toFixed(2)}` : ''}
        {build.options.auto_build ? ' · builds automatically' : ''}
        {build.options.allow_new_blocks === false ? ' · registry blocks only' : ''}
        {build.options.small_context ? ' · small context' : ''}
        {build.options.decision_hints ? ' · decision hints' : ''}
      </div>
    </div>
  )
}

type Act = (fn: () => Promise<BuildOut>) => void

function Preflight({ build, graph, act }: { build: BuildOut; graph: GraphOut; act: Act }) {
  const { blocking, warnings } = build.preflight
  return (
    <div>
      {blocking.map((i) => (
        <div key={i.code} style={{ ...box, borderColor: 'var(--mm-error-border)', background: 'var(--mm-error-bg)', color: 'var(--mm-error-2)', whiteSpace: 'pre-wrap' }}>
          {i.message}
        </div>
      ))}
      {warnings.map((i) => (
        <div key={i.code} style={{ ...box, borderColor: 'var(--mm-warn-border)', background: 'var(--mm-warn-bg)', color: 'var(--mm-warn)', whiteSpace: 'pre-wrap' }}>
          {i.message}
        </div>
      ))}
      <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap', marginBottom: 8 }}>
        {(blocking.some((i) => i.code === 'anchor_not_run') || warnings.some((i) => i.code === 'upstream_not_current')) && (
          <button className="brand-primary" onClick={() => act(() => api.agentAction('run_upstream'))}>
            Run upstream
          </button>
        )}
        <button onClick={() => act(() => api.agentAction('recheck'))} title="Check again, e.g. after tagging roles">
          Re-check
        </button>
        <button disabled={blocking.length > 0} onClick={() => act(() => api.agentAction('proceed'))}>
          Proceed anyway
        </button>
      </div>
      {blocking.length === 0 && warnings.length > 0 && (
        <div style={{ color: 'var(--mm-muted)', marginBottom: 8 }}>
          Proceeding records these warnings in the build's report. Anchors: {build.anchors.map((a) => graph.blocks[a]?.name ?? a).join(', ')}
        </div>
      )}
    </div>
  )
}

// `busy`: the model is still finishing its turn (e.g. a slow local model
// writing a closing remark after submitting) -- the server refuses to act
// until it's done, so the buttons wait too.
function PlanReview({ build, graph, act, busy }: { build: BuildOut; graph: GraphOut; act: Act; busy: boolean }) {
  const [feedback, setFeedback] = useState('')
  const plan = build.plan
  return (
    <div>
      {plan ? (
        <PlanView plan={plan} graph={graph} />
      ) : build.pending_question ? (
        <div style={{ ...box, borderColor: 'var(--mm-warn-border)', background: 'var(--mm-warn-bg)', whiteSpace: 'pre-wrap' }}>{build.pending_question}</div>
      ) : (
        <div style={box}>The AI ended planning without a plan. Tell it what to do.</div>
      )}
      <div style={label}>{plan?.questions?.length ? 'Answer its questions' : 'Feedback (optional)'}</div>
      <textarea
        value={feedback}
        onChange={(e) => setFeedback(e.target.value)}
        rows={3}
        style={{ width: '100%', boxSizing: 'border-box' }}
        placeholder="e.g. use a 70/30 split instead of out-of-time; add a calibration stage"
      />
      <div style={{ display: 'flex', gap: 6, marginTop: 6, marginBottom: 8 }}>
        <button
          disabled={busy || !feedback.trim()}
          onClick={() => {
            act(() => api.agentFeedback(feedback))
            setFeedback('')
          }}
        >
          Re-plan with feedback
        </button>
        <button
          className="brand-primary"
          disabled={busy || !plan || (plan.questions?.length ?? 0) > 0}
          onClick={() => act(() => api.agentAction('approve'))}
          title="Let the AI build this outline, one stage at a time"
        >
          Approve and build
        </button>
      </div>
    </div>
  )
}

function PlanView({ plan, graph }: { plan: BuildPlan; graph: GraphOut }) {
  return (
    <div>
      <div style={box}>{plan.summary}</div>
      {!!plan.questions?.length && (
        <div style={{ ...box, borderColor: 'var(--mm-warn-border)', background: 'var(--mm-warn-bg)' }}>
          <div style={label}>Questions</div>
          <ul style={{ margin: 0, paddingLeft: 16 }}>{plan.questions.map((q, i) => <li key={i}>{q}</li>)}</ul>
        </div>
      )}
      {!!plan.assumptions?.length && (
        <div style={box}>
          <div style={label}>Assumptions</div>
          <ul style={{ margin: 0, paddingLeft: 16 }}>{plan.assumptions.map((a, i) => <li key={i}>{a}</li>)}</ul>
        </div>
      )}
      <div style={box}>
        <div style={label}>Stages</div>
        <ol style={{ margin: 0, paddingLeft: 18 }}>
          {plan.stages.map((st) => (
            <li key={st.key} style={{ marginBottom: 6 }}>
              <strong>{st.name}</strong>
              {st.lane && <span style={{ color: 'var(--mm-muted)' }}> (into lane {graph.lanes[st.lane]?.name ?? st.lane})</span>}
              <div style={{ color: 'var(--mm-fg-4)', whiteSpace: 'pre-wrap' }}>{st.goal}</div>
            </li>
          ))}
        </ol>
        <div style={{ color: 'var(--mm-muted)', marginTop: 4 }}>
          The AI picks each stage's blocks once the stages before it have run, so it can use what they found.
        </div>
      </div>
      {!!plan.changes_to_existing?.length && (
        <div style={{ ...box, borderColor: 'var(--mm-ai-border)' }}>
          <div style={label}>Changes to your existing blocks</div>
          {plan.changes_to_existing.map((c, i) => (
            <div key={i}>
              <strong>{graph.blocks[c.block]?.name ?? c.block}</strong>: {c.change} <span style={{ color: 'var(--mm-muted)' }}>-- {c.why}</span>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}

const STAGE_MARK: Record<BuildStage['status'], { mark: string; color: string }> = {
  done: { mark: '✓', color: 'var(--mm-ok)' },
  active: { mark: '▶', color: 'var(--mm-ai)' },
  pending: { mark: '○', color: 'var(--mm-faint)' },
  skipped: { mark: '–', color: 'var(--mm-faint)' },
}

// Progress through the outline: each stage's status, the current stage's
// planned steps (ticked off as they're built), and earlier stages' summaries.
function Stages({ build, graph }: { build: BuildOut; graph: GraphOut }) {
  const built = new Set(
    Object.values(graph.blocks)
      .filter((b) => b.provenance?.build_id === build.id && b.provenance.plan_step)
      .map((b) => b.provenance!.plan_step!),
  )
  return (
    <div style={box}>
      <div style={label}>
        Stage {Math.min(build.stage_index + 1, build.stages.length)} of {build.stages.length}
      </div>
      {build.stages.map((st, i) => {
        const { mark, color } = STAGE_MARK[st.status]
        const current = i === build.stage_index
        return (
          <div key={st.key} style={{ marginBottom: 4 }}>
            <div style={{ color: st.status === 'pending' || st.status === 'skipped' ? 'var(--mm-muted)' : 'var(--mm-fg-2)' }}>
              <span style={{ color, display: 'inline-block', width: 14 }}>{mark}</span>
              <strong>{st.name}</strong>
              {st.status === 'skipped' && <span style={{ color: 'var(--mm-muted)' }}> (dropped)</span>}
            </div>
            {current && st.status === 'active' && (
              <div style={{ marginLeft: 14 }}>
                <div style={{ color: 'var(--mm-fg-4)' }}>{st.goal}</div>
                {st.plan ? (
                  st.plan.steps.map((s) => (
                    <div key={s.ref} style={{ color: built.has(s.ref) ? 'var(--mm-ok)' : 'var(--mm-muted)' }}>
                      {built.has(s.ref) ? '✓' : '·'} <span style={{ color: 'var(--mm-ai)' }}>{s.ref}</span> {s.name}{' '}
                      <span style={{ color: 'var(--mm-faint)' }}>({s.category})</span>
                    </div>
                  ))
                ) : (
                  <div style={{ color: 'var(--mm-muted)', fontStyle: 'italic' }}>Planning this stage's blocks…</div>
                )}
              </div>
            )}
            {st.summary && !(current && build.phase === 'awaiting_stage_review') && (
              <details style={{ marginLeft: 14, color: 'var(--mm-fg-4)' }}>
                <summary style={{ cursor: 'pointer' }}>Summary</summary>
                <div style={{ whiteSpace: 'pre-wrap' }}>{st.summary}</div>
              </details>
            )}
          </div>
        )
      })}
    </div>
  )
}

// After each stage but the last: what it built and found, then go on to
// the next stage -- or send changes to this one.
function StageReview({ build, act, busy }: { build: BuildOut; act: Act; busy: boolean }) {
  const [feedback, setFeedback] = useState('')
  const stage = build.stages[build.stage_index]
  const next = build.stages[build.stage_index + 1]
  return (
    <div>
      <div style={{ ...box, borderColor: 'var(--mm-ok-border)', background: 'var(--mm-ok-bg)' }}>
        <div style={label}>{stage.name} is built</div>
        <div style={{ whiteSpace: 'pre-wrap' }}>{stage.summary}</div>
      </div>
      <Concerns concerns={(build.concerns ?? []).filter((c) => c.stage === stage.key)} />
      {next && (
        <div style={{ color: 'var(--mm-fg-4)', marginBottom: 6 }}>
          Next: <strong>{next.name}</strong> -- {next.goal}
        </div>
      )}
      <textarea
        value={feedback}
        onChange={(e) => setFeedback(e.target.value)}
        rows={3}
        style={{ width: '100%', boxSizing: 'border-box' }}
        placeholder={`Changes to ${stage.name} (optional), e.g. drop features with IV below 0.05`}
      />
      <div style={{ display: 'flex', gap: 6, marginTop: 6, marginBottom: 8 }}>
        <button
          disabled={busy || !feedback.trim()}
          onClick={() => {
            act(() => api.agentFeedback(feedback))
            setFeedback('')
          }}
        >
          Send changes
        </button>
        <button className="brand-primary" disabled={busy} onClick={() => act(() => api.agentAction('approve'))}>
          Continue{next ? ` to ${next.name}` : ''}
        </button>
      </div>
    </div>
  )
}

function Question({ build, act, busy }: { build: BuildOut; act: Act; busy: boolean }) {
  const [answer, setAnswer] = useState('')
  return (
    <div>
      <div style={{ ...box, borderColor: 'var(--mm-warn-border)', background: 'var(--mm-warn-bg)', whiteSpace: 'pre-wrap' }}>{build.pending_question}</div>
      <textarea value={answer} onChange={(e) => setAnswer(e.target.value)} rows={3} style={{ width: '100%', boxSizing: 'border-box' }} />
      <button
        className="brand-primary"
        style={{ marginTop: 6 }}
        disabled={busy || !answer.trim()}
        onClick={() => {
          act(() => api.agentFeedback(answer))
          setAnswer('')
        }}
      >
        Send
      </button>
    </div>
  )
}

// What the app itself flagged (see AgentBuild.concerns) -- not the model's
// account, so it shows whether or not the model mentions it.
function Concerns({ concerns }: { concerns: BuildOut['concerns'] }) {
  if (!concerns.length) return null
  return (
    <div style={{ ...box, borderColor: 'var(--mm-warn-border)', background: 'var(--mm-warn-bg)', color: 'var(--mm-warn)' }}>
      <div style={{ ...label, color: 'var(--mm-warn)' }}>Flagged by the app</div>
      {concerns.map((c, i) => (
        <div key={i}>⚠ {c.what}</div>
      ))}
    </div>
  )
}

// Only a build that ran to the end writes a report (see controller._attach_report).
const REPORTED = new Set(['done', 'done_with_errors'])

function Finished({ build, graph, onOpenReport }: { build: BuildOut; graph: GraphOut; onOpenReport: (buildId: string) => void }) {
  const good = build.phase === 'done'
  return (
    <div style={{ ...box, borderColor: good ? 'var(--mm-ok-border)' : 'var(--mm-border)', marginBottom: 12 }}>
      <div style={{ fontWeight: 600 }}>
        Last build: {PHASE_LABEL[build.phase]} <span style={{ color: 'var(--mm-muted)', fontWeight: 400 }}>-- {build.goal}</span>
      </div>
      {build.error && <div style={{ color: 'var(--mm-error)', marginTop: 4 }}>{build.error}</div>}
      {build.results.length > 0 && (
        <div style={{ marginTop: 6 }}>
          {build.results.map((r, i) => (
            <div key={i}>
              <strong>{r.label ?? `${graph.blocks[r.block]?.name ?? r.block}.${r.port}`}:</strong>{' '}
              <code style={{ wordBreak: 'break-word' }}>{JSON.stringify(r.value ?? r.error ?? r.row_count)}</code>
            </div>
          ))}
        </div>
      )}
      {(build.concerns ?? []).length > 0 && (
        <div style={{ marginTop: 6 }}>
          <Concerns concerns={build.concerns} />
        </div>
      )}
      {build.report && <div style={{ marginTop: 6, whiteSpace: 'pre-wrap', maxHeight: 260, overflowY: 'auto' }}>{build.report}</div>}
      {build.deviations.length > 0 && (
        <div style={{ marginTop: 6 }}>
          <div style={label}>Deviations from the plan</div>
          {build.deviations.map((d, i) => (
            <div key={i}>
              {d.plan_step && <span style={{ color: 'var(--mm-ai)' }}>{d.plan_step} </span>}
              {d.what} -- <span style={{ color: 'var(--mm-muted)' }}>{d.why}</span>
            </div>
          ))}
        </div>
      )}
      <div style={{ display: 'flex', alignItems: 'center', gap: 8, flexWrap: 'wrap', color: 'var(--mm-muted)', marginTop: 8 }}>
        {REPORTED.has(build.phase) && (
          <button onClick={() => onOpenReport(build.id)} title="Results, deviations from the plan, failures and warnings, in full">
            Open full report
          </button>
        )}
        <button
          onClick={() => downloadLog(build.id)}
          title={`Models, timings, every plan round and every step the AI took${build.log_path ? `\nSaved at ${build.log_path}` : ''}`}
        >
          Download log
        </button>
        {build.owned_blocks.length > 0 && build.phase !== 'discarded' && (
          <span>{build.owned_blocks.length} block(s) built · Undo reverts the whole build</span>
        )}
      </div>
    </div>
  )
}

function downloadLog(buildId: string) {
  api
    .agentBuildLog(buildId)
    .then((log) => host.saveFile(`${buildId}.json`, new Blob([JSON.stringify(log, null, 2)], { type: 'application/json' })))
    .catch((e) => host.notify((e as Error).message))
}

// ---- event feed ---------------------------------------------------------------------

function describeEvent(e: BuildEvent, graph: GraphOut): { text: string; tone: 'ok' | 'err' | 'info' | 'ai' } | null {
  const blockName = (id: unknown) => (typeof id === 'string' ? (graph.blocks[id]?.name ?? id) : '')
  if (e.kind === 'assistant') return { text: String(e.text), tone: 'ai' }
  if (e.kind === 'user') return { text: `You: ${String(e.text)}`, tone: 'info' }
  if (e.kind === 'phase') return { text: `— ${PHASE_LABEL[String(e.phase)] ?? e.phase}${e.note ? ` (${e.note})` : ''}`, tone: 'info' }
  if (e.kind === 'error') return { text: String(e.message), tone: 'err' }
  if (e.kind === 'sample_mode') return { text: e.rows ? `Sample mode: ${Number(e.rows).toLocaleString()} rows` : 'Back to full data', tone: 'info' }
  if (e.kind === 'approved_change') return { text: `Changed your block ${blockName(e.block)}: ${e.change}`, tone: 'info' }
  if (e.kind === 'stage') {
    const verb = { started: 'Started', planned: 'Planned', done: 'Finished', skipped: 'Dropped' }[String(e.status)] ?? String(e.status)
    const steps = e.status === 'planned' ? ` (${e.steps} step${e.steps === 1 ? '' : 's'})` : ''
    return { text: `— ${verb} stage ${String(e.name)}${steps}`, tone: 'info' }
  }
  if (e.kind === 'step')
    return {
      text: `${String(e.ref)} ${blockName(e.block)} → ${String(e.status)}${e.error ? `: ${String(e.error).slice(0, 300)}` : ''}`,
      tone: e.status === 'green' ? 'ok' : 'err',
    }
  if (e.kind === 'concern') return { text: `⚠ ${String(e.what)}`, tone: 'err' }
  if (e.kind === 'auto_continued') return { text: 'Going straight on to the next stage (building automatically)', tone: 'info' }
  if (e.kind !== 'tool') return null
  const args = (e.args ?? {}) as Record<string, unknown>
  const result = (e.result ?? {}) as Record<string, unknown>
  const tool = String(e.tool)
  const target =
    tool === 'plan_stage' || tool === 'complete_stage'
      ? ''
      : tool === 'add_block'
        ? String(args.category)
        : tool === 'create_library_block'
          ? `new block: ${args.name}`
          : tool === 'connect'
            ? `${blockName(args.from_block)}.${args.from_port} → ${blockName(args.to_block)}.${args.to_port}`
            : blockName(args.block ?? args.category ?? '')
  let text = `${tool}${target ? ` ${target}` : ''}`
  if (tool === 'run_to' && result.status) text += ` → ${result.status}`
  if (!e.ok) text += `: ${String(result.error ?? '').slice(0, 300)}`
  return { text, tone: e.ok && result.status !== 'red' ? 'ok' : 'err' }
}

function EventFeed({ events, graph }: { events: BuildEvent[]; graph: GraphOut }) {
  const end = useRef<HTMLDivElement>(null)
  useEffect(() => {
    end.current?.scrollIntoView({ block: 'nearest' })
  }, [events.length])
  const color = { ok: 'var(--mm-ok)', err: 'var(--mm-error)', info: 'var(--mm-muted)', ai: 'var(--mm-fg-2)' }
  return (
    <div style={{ ...box, flex: 1, minHeight: 120, overflowY: 'auto', fontSize: 11, background: 'var(--mm-bg-subtle-2)' }}>
      {events.map((e) => {
        const d = describeEvent(e, graph)
        if (!d) return null
        return (
          <div key={e.seq} style={{ color: color[d.tone], marginBottom: 3, whiteSpace: 'pre-wrap', wordBreak: 'break-word' }}>
            {d.tone === 'ok' ? '✓ ' : d.tone === 'err' ? '✗ ' : ''}
            {d.text}
          </div>
        )
      })}
      <div ref={end} />
    </div>
  )
}
