"""Drives one AI build through its lifecycle (see /agent-builder-proposal.md
§2-§4, §7): preflight -> planning an outline of stages -> approval ->
for each stage: plan its blocks, build, check (-> questions), stage review
-> final full-data run -> report. The model's turns run on a worker
thread; everything the API calls here returns promptly, and the frontend
polls the build's state."""

from __future__ import annotations

import re
import threading
from contextlib import ExitStack
from typing import Any, Callable

from ..packet import ColumnRole, DataFramePacket
from ..runslot import RunBusy
from . import prompts
from .build import (
    AWAITING_APPROVAL,
    AWAITING_INPUT,
    AWAITING_STAGE_REVIEW,
    BUILDING,
    DISCARDED,
    DONE,
    DONE_WITH_ERRORS,
    FAILED,
    FINAL_RUN,
    LOCKING_PHASES,
    PLANNING,
    PREFLIGHT,
    STAGE_ACTIVE,
    STAGE_DONE,
    STAGE_PENDING,
    STOPPED,
    TERMINAL_PHASES,
    AgentBuild,
    LLMChoice,
    ToolError,
)
from .loop import AgentLoop, LLMUnavailable
from .tools import resolve_lane, summarize_value, tools_for_phase

# Sample only when the data is big enough for it to matter.
AUTO_SAMPLE_THRESHOLD_ROWS = 100_000
AUTO_SAMPLE_ROWS = 50_000

LoopFactory = Callable[[LLMChoice, str], AgentLoop]  # (choice, "plan"|"build") -> loop


class BuildError(Exception):
    """A request that doesn't fit the build's current phase (HTTP 409)."""


def _issue(code: str, message: str, blocks: list[str] | None = None) -> dict[str, Any]:
    return {"code": code, "message": message, "blocks": blocks or []}


def run_preflight(build: AgentBuild) -> dict[str, Any]:
    """§4.1: blocking problems and warnings over the anchors and their whole
    upstream. No LLM involved."""
    session = build.session
    graph, runner = session.graph, session.runner
    blocking: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []
    missing = [a for a in build.anchors if a not in graph.blocks]
    if not build.anchors or missing:
        blocking.append(_issue("no_anchors", "Select the block(s) the build should start from." if not missing else f"Unknown blocks: {missing}"))
        return {"blocking": blocking, "warnings": warnings, "max_rows": None}

    no_output = []
    frames: list[DataFramePacket] = []
    for a in build.anchors:
        st = runner.state.get(a)
        entry = runner.cache.get(st.last_successful_key) if st and st.last_successful_key else None
        if entry is None:
            no_output.append(a)
            continue
        frames += [v for v in entry.outputs.values() if isinstance(v, DataFramePacket)]
    if no_output:
        names = ", ".join(graph.blocks[a].name for a in no_output)
        blocking.append(_issue("anchor_not_run", f"These haven't produced output yet, so there's nothing to plan from: {names}. Run upstream first.", no_output))

    stale = []
    scope = build.scope_block_ids()
    for bid in graph.topo_order():
        if bid not in scope:
            continue
        status = runner.status(bid)
        if status in ("green", "running"):
            continue
        if status == "orange":
            reason = f"out of date ({runner.stale_reason(bid) or 'changed since it ran'})"
        elif status == "red":
            st = runner.state.get(bid)
            lines = ((st.last_error if st else "") or "").strip().splitlines()
            reason = f"failed: {lines[-1] if lines else 'error'}"
        else:
            reason = "never run"
        stale.append((bid, f"{graph.blocks[bid].name}: {reason}"))
    if stale:
        warnings.append(
            _issue(
                "upstream_not_current",
                "Not up to date, so the build would plan against outputs that don't match the graph:\n"
                + "\n".join(f"  - {m}" for _, m in stale),
                [b for b, _ in stale],
            )
        )

    targets = sorted({n for f in frames for n, m in f.schema_meta.items() if m.role == ColumnRole.TARGET})
    if len(targets) > 1:
        blocking.append(_issue("multiple_targets", f"More than one column is tagged target across the anchors: {targets}. Tag exactly one."))
    elif not targets and frames:
        warnings.append(_issue("no_target", "No column is tagged target. If the goal is a model, tag the target column first."))
    if frames and not any(m.role == ColumnRole.EXCLUDED for f in frames for m in f.schema_meta.values()):
        warnings.append(_issue("no_excluded", "No columns are tagged excluded. Tag leaky or post-outcome columns so the build never uses them."))
    if re.search(r"out[- ]of[- ]time|\boot\b|vintage", build.goal, re.I) and frames and not any(
        m.role == ColumnRole.DATE for f in frames for m in f.schema_meta.values()
    ):
        warnings.append(_issue("no_date", "The goal mentions out-of-time/vintages but no column is tagged date."))
    max_rows = max((f.data.height for f in frames), default=None)
    return {"blocking": blocking, "warnings": warnings, "max_rows": max_rows}


def _record_resolved_model(choice: LLMChoice, loop: AgentLoop) -> None:
    """When no model was chosen, the loop picks one (a provider default, or
    whatever a local server has loaded) -- record which, so provenance and
    the report name the model that actually ran, not 'default'."""
    if not choice.model and getattr(loop, "model", None):
        choice.model = loop.model


class BuildController:
    """One per API server. At most one live build at a time."""

    def __init__(self, session_getter: Callable[[], Any], run_slot, loop_factory: LoopFactory) -> None:
        self._session_getter = session_getter
        self.run_slot = run_slot
        self.loop_factory = loop_factory
        self.build: AgentBuild | None = None
        self.token: str | None = None
        self._plan_loop: AgentLoop | None = None
        self._build_loop: AgentLoop | None = None
        self._worker: threading.Thread | None = None
        self._transaction: ExitStack | None = None
        self._snapshot_before = None
        self._lock = threading.RLock()

    # ---- state -----------------------------------------------------------

    @property
    def session(self):
        return self._session_getter()

    def canvas_locked(self) -> bool:
        return self.build is not None and self.build.phase in LOCKING_PHASES

    def _require(self, *phases: str) -> AgentBuild:
        b = self.build
        if b is None:
            raise BuildError("no AI build in progress")
        if phases and b.phase not in phases:
            raise BuildError(f"the build is {b.phase}; this needs {' or '.join(phases)}")
        return b

    def _busy(self) -> bool:
        return self._worker is not None and self._worker.is_alive()

    def _require_idle(self) -> None:
        """Checked before any state changes -- a request that has to wait
        for the model must not leave half its effects behind."""
        if self._busy():
            raise BuildError("the AI is still finishing its turn -- try again in a moment")

    def _spawn(self, fn: Callable[[], None]) -> None:
        if self._busy():
            raise BuildError("the AI is still working -- wait for it, or stop the build")

        if self.build is not None:
            self.build.turn_over = False

        def target() -> None:
            try:
                fn()
            except LLMUnavailable as e:
                self._pause_for_retry(str(e))
            except Exception as e:  # noqa: BLE001 -- nobody to raise to on a worker thread
                b = self.build
                if b is not None:
                    b.error = f"{type(e).__name__}: {e}"
                    b.log("error", message=b.error)
                    self._end(FAILED)

        self._worker = threading.Thread(target=target, daemon=True)
        self._worker.start()

    def _pause_for_retry(self, message: str) -> None:
        """The model's endpoint timed out or was unreachable mid-turn. Keep
        the build (and its conversation) and let the user retry by replying,
        rather than throwing away a long build over one slow response."""
        b = self.build
        if b is None:
            return
        b.log("error", message=message)
        if b.stop_requested:
            self._end(STOPPED)
            return
        retry = f"The AI model didn't answer: {message}\n\nReply (e.g. \"continue\") to retry, or stop the build."
        if b.phase == PLANNING:
            b.pending_question = retry
            b.set_phase(AWAITING_APPROVAL, note="the model didn't answer")
        elif b.phase == BUILDING:
            b.pending_question = retry
            b.set_phase(AWAITING_INPUT, note="the model didn't answer")
        else:
            b.error = message
            self._end(FAILED)

    def join(self, timeout: float | None = None) -> None:
        """Wait for the current worker turn (tests, discard)."""
        if self._worker is not None:
            self._worker.join(timeout)

    # ---- lifecycle -------------------------------------------------------

    def start(self, goal: str, anchors: list[str], plan_llm: LLMChoice, build_llm: LLMChoice, options, token: str) -> AgentBuild:
        with self._lock:
            if self.build is not None and self.build.phase not in TERMINAL_PHASES:
                raise BuildError("an AI build is already in progress -- finish, stop or discard it first")
            if not goal.strip():
                raise BuildError("describe what to build")
            b = AgentBuild(self.session, self.run_slot, goal.strip(), anchors, plan_llm, build_llm, options)
            self.build, self.token = b, token
            self._plan_loop = self._build_loop = None
            self._transaction = None
            self._snapshot_before = None
            b.log("phase", phase=PREFLIGHT)
            self._preflight_then_maybe_plan()
            return b

    def _preflight_then_maybe_plan(self) -> None:
        b = self.build
        b.preflight = run_preflight(b)
        if b.preflight["blocking"] or b.preflight["warnings"]:
            b.log("preflight", **{k: v for k, v in b.preflight.items() if k != "max_rows"})
            b.save_log()
            return
        self._start_planning()

    def recheck(self) -> AgentBuild:
        """Re-run preflight (e.g. after the user ran things by hand)."""
        with self._lock:
            b = self._require(PREFLIGHT)
            self._preflight_then_maybe_plan()
            return b

    def run_upstream(self) -> AgentBuild:
        """Bring the anchors' upstream up to date -- reading never-read or
        stale input blocks, then cascading to each anchor -- and re-check.
        A user action from the preflight panel, not the agent's."""
        with self._lock:
            b = self._require(PREFLIGHT)
            runner = self.session.runner

            def work() -> None:
                self._read_stale_inputs()
                for a in b.anchors:
                    runner.run_to_here(a)

            try:
                self.run_slot.run_sync(work)
            except RunBusy as e:
                raise BuildError(str(e)) from None
            b.log("upstream_run")
            self._preflight_then_maybe_plan()
            return b

    def proceed(self) -> AgentBuild:
        """Continue past preflight warnings (never past blocking problems)."""
        with self._lock:
            b = self._require(PREFLIGHT)
            if b.preflight["blocking"]:
                raise BuildError("fix the blocking problems first")
            b.log("proceeded_past_warnings", warnings=[w["code"] for w in b.preflight["warnings"]])
            self._start_planning()
            return b

    def _start_planning(self) -> None:
        b = self.build
        b.set_phase(PLANNING)

        def work() -> None:
            # Created on the worker, so a failure (missing key, no CLI)
            # ends the build as failed with its message, via _spawn.
            self._plan_loop = self.loop_factory(b.plan_llm, "plan")
            _record_resolved_model(b.plan_llm, self._plan_loop)
            outcome = self._plan_loop.start(b, prompts.plan_system(b, getattr(self._plan_loop, "compact", False)), prompts.plan_prompt(b), tools_for_phase(PLANNING))
            self._after_plan_turn(outcome)

        self._spawn(work)

    def _after_plan_turn(self, outcome) -> None:
        b = self.build
        if b.stop_requested:
            self._end(STOPPED)
            return
        if outcome.error:
            b.error = outcome.error
            b.log("error", message=outcome.error)
            self._end(FAILED)
            return
        if b.phase == PLANNING:
            # Ended its turn without a (valid) plan -- let the user nudge it.
            b.log("no_plan", text=outcome.final_text)
            b.set_phase(AWAITING_APPROVAL, note="the AI ended planning without submitting a plan")
            return
        if b.options.auto_build and b.phase == AWAITING_APPROVAL and b.plan is not None:
            if b.plan.get("questions"):
                # Can't approve over open questions -- wait for the answers
                # like a normal build; once re-planned, auto-build applies.
                b.log("auto_build_paused", reason="the plan has open questions")
                return
            # Still on this worker turn, so build right here rather than
            # spawning another one (approve() would find the worker busy).
            b.log("auto_approved")
            self._begin_build()()

    def feedback(self, text: str) -> AgentBuild:
        """Plan feedback (re-plan), changes to the stage under review, or the
        answer to an ask_user question."""
        with self._lock:
            b = self._require(AWAITING_APPROVAL, AWAITING_INPUT, AWAITING_STAGE_REVIEW)
            self._require_idle()
            if not text.strip():
                raise BuildError("write some feedback")
            if b.phase == AWAITING_APPROVAL:
                b.plan_history.append({"plan": b.plan, "feedback": text})
                b.plan = None
                b.pending_question = None
                b.set_phase(PLANNING)
                b.log("user", text=text)

                def replan() -> None:
                    outcome = self._plan_loop.resume(b, prompts.plan_feedback_prompt(text), tools_for_phase(PLANNING))
                    self._after_plan_turn(outcome)

                self._spawn(replan)
            elif b.phase == AWAITING_STAGE_REVIEW:
                # Back into the stage the user just reviewed.
                stage = b.current_stage
                stage["status"] = STAGE_ACTIVE
                b.set_phase(BUILDING)
                b.log("user", text=text)

                def revise() -> None:
                    outcome = self._build_loop.resume(b, prompts.stage_feedback_prompt(b, text), tools_for_phase(BUILDING))
                    self._after_build_turn(outcome)

                self._spawn(revise)
            else:
                b.pending_question = None
                b.set_phase(BUILDING)
                b.log("user", text=text)

                def answer() -> None:
                    outcome = self._build_loop.resume(b, prompts.answer_prompt(text), tools_for_phase(BUILDING))
                    self._after_build_turn(outcome)

                self._spawn(answer)
            return b

    def approve(self) -> AgentBuild:
        """Approve the outline and start building -- or, after a stage
        review, go on to the next stage."""
        with self._lock:
            b = self._require(AWAITING_APPROVAL, AWAITING_STAGE_REVIEW)
            self._require_idle()
            if b.phase == AWAITING_STAGE_REVIEW:
                b.set_phase(BUILDING)
                self._spawn(self._next_stage)
                return b
            if b.plan is None:
                raise BuildError("there's no plan to approve -- send feedback to have the AI plan again")
            if b.plan.get("questions"):
                raise BuildError("the plan has open questions -- answer them as feedback first")
            self._spawn(self._begin_build())
            return b

    def _begin_build(self) -> Callable[[], None]:
        """Open the build's undo transaction and move to building; returns
        the build turn to run (on a worker). Caller has checked the plan."""
        with self._lock:
            b = self.build
            session = self.session
            self._transaction = ExitStack()
            self._snapshot_before = self._transaction.enter_context(session.transaction())
            for change in b.plan.get("changes_to_existing") or []:
                b.approved_changes.setdefault(change["block"], []).append(change)
            b.stages = [
                {**{k: v for k, v in st.items() if k in ("key", "name", "goal", "lane")}, "status": STAGE_PENDING, "plan": None, "summary": None}
                for st in b.plan["stages"]
            ]
            b.stage_index = 0
            b.set_phase(BUILDING)
            self._maybe_sample()
            # Every stage's lane up front, so the outline shows on the
            # canvas; any left empty are removed when the build ends.
            for st in b.stages:
                if st.get("lane"):
                    b.lane_map[st["key"]] = st["lane"]
                else:
                    resolve_lane(b, st["key"])
            self._start_stage()

            def work() -> None:
                b.turn_over = False
                self._build_loop = self.loop_factory(b.build_llm, "build")
                _record_resolved_model(b.build_llm, self._build_loop)
                outcome = self._build_loop.start(b, prompts.build_system(b, getattr(self._build_loop, "compact", False)), prompts.build_prompt(b), tools_for_phase(BUILDING))
                self._after_build_turn(outcome)

            return work

    def _start_stage(self) -> None:
        b = self.build
        stage = b.current_stage
        stage["status"] = STAGE_ACTIVE
        b.log("stage", key=stage["key"], name=stage["name"], status="started", index=b.stage_index)

    def _next_stage(self) -> None:
        """Move on to the next stage, in the same build conversation (so the
        model keeps what it learned building the earlier ones). Runs on
        the worker."""
        b = self.build
        b.stage_index += 1
        self._start_stage()
        outcome = self._build_loop.resume(b, prompts.stage_prompt(b), tools_for_phase(BUILDING))
        self._after_build_turn(outcome)

    def _maybe_sample(self) -> None:
        """§7: build on a sample when the data is large. Switching sample
        mode makes input blocks stale, so they're re-read in sample mode
        here (run_block, not refresh -- refresh bumps the read counter and
        would make the full-data cache unreachable afterwards)."""
        b = self.build
        session = self.session
        requested = b.options.sample_rows
        max_rows = b.preflight.get("max_rows") or 0
        if requested is None:
            requested = AUTO_SAMPLE_ROWS if max_rows > AUTO_SAMPLE_THRESHOLD_ROWS else 0
        current = session.runner.sample_rows
        if not requested or current is not None:
            # Off, or the user already has sample mode on -- build on
            # whatever they're looking at and leave the setting alone.
            b.sample_rows_used = current
            return
        session.set_sample_rows(requested)
        b.sample_rows_used = requested
        b.changed_sample_mode = True
        b.log("sample_mode", rows=requested)
        self._reread_inputs()

    def _read_stale_inputs(self) -> None:
        """(Re)read the in-scope input blocks that aren't current. Caller
        holds the run slot."""
        graph, runner = self.session.graph, self.session.runner
        scope = self.build.scope_block_ids()
        for bid in graph.topo_order():
            if bid in scope and graph.blocks[bid].block_type == "input" and runner.status(bid) != "green":
                runner.run_block(bid)

    def _reread_inputs(self) -> None:
        self.run_slot.run_sync(self._read_stale_inputs)

    def _restore_sample_mode(self) -> None:
        """Undo _maybe_sample: back to full data. Re-reading the inputs
        (run_block again) finds their full-data read still cached."""
        b = self.build
        if b is None or not b.changed_sample_mode:
            return
        b.changed_sample_mode = False
        self.session.set_sample_rows(None)
        self._reread_inputs()
        b.log("sample_mode", rows=None)

    def _after_build_turn(self, outcome) -> None:
        b = self.build
        if b.stop_requested:
            self._end(STOPPED)
            return
        if b.finished:
            self._final_run()
            return
        if outcome.error:
            b.log("error", message=outcome.error)
            b.pending_question = f"The AI hit an error: {outcome.error}\n\nReply to have it continue, or stop the build."
            b.set_phase(AWAITING_INPUT)
            return
        if b.phase == AWAITING_INPUT:
            return  # ask_user: wait for feedback()
        stage = b.current_stage
        if stage is not None and stage["status"] == STAGE_DONE:
            # complete_stage: the user reviews it, unless building
            # automatically -- then straight on, on this worker turn.
            if b.options.auto_build:
                b.log("auto_continued", stage=stage["key"])
                self._next_stage()
            else:
                b.set_phase(AWAITING_STAGE_REVIEW)
            return
        b.pending_question = (
            "The AI ended its turn without finishing the build"
            + (f":\n\n{outcome.final_text}" if outcome.final_text else ".")
            + "\n\nReply to have it continue, or stop the build."
        )
        b.set_phase(AWAITING_INPUT)

    def _final_run(self) -> None:
        b = self.build
        b.set_phase(FINAL_RUN)
        session = self.session
        self._restore_sample_mode()
        if b.options.final_full_run:
            report = self.run_slot.run_sync(session.runner.run_all)
            b.log("final_run", statuses={k: v for k, v in (report or {}).items() if k in b.owned_blocks})
        b.results = []
        for k in b.key_outputs:
            try:
                port, value = b.current_output(k["block"], k.get("port"))
                summary = summarize_value(value, with_stats=False)
                if summary["type"] == "dataframe":
                    summary = {"type": "dataframe", "row_count": summary["row_count"]}
            except ToolError as e:
                port, summary = k.get("port"), {"error": str(e)}
            b.results.append({"block": k["block"], "port": port, "label": k.get("label"), **summary})
        failed = [bid for bid in b.owned_blocks if session.runner.status(bid) == "red"]
        self._attach_report(failed)
        self._end(DONE_WITH_ERRORS if failed else DONE)

    def _attach_report(self, failed: list[str]) -> None:
        b = self.build
        session = self.session
        target = next((k["block"] for k in b.key_outputs if k["block"] in session.graph.blocks), None)
        target = target or next((bid for bid in reversed(session.graph.topo_order()) if bid in b.owned_blocks), None)
        if target is None:
            return
        doc = [f"# AI build report\n\n**Goal:** {b.goal}\n", b.report or "(no report)"]
        if b.results:
            doc.append("\n## Results on the full data\n" if b.options.final_full_run else "\n## Results\n")
            for r in b.results:
                label = r.get("label") or f"{session.graph.blocks[r['block']].name}.{r.get('port')}"
                value = r.get("value", r.get("error", r.get("row_count")))
                doc.append(f"- **{label}:** `{value}`")
        stages = [st for st in b.stages if st.get("summary")]
        if stages:
            doc.append("\n## Stages\n")
            for st in stages:
                doc.append(f"### {st['name']}\n\n{st['summary']}\n")
        if b.deviations:
            doc.append("\n## Deviations from the plan\n")
            doc += [f"- {d.get('plan_step', '')} {d['what']} -- {d['why']}".strip() for d in b.deviations]
        if failed:
            doc.append("\n## Failed on the full data\n")
            doc += [f"- {session.graph.blocks[f].name}: {(session.runner.state[f].last_error or '').strip()[-300:]}" for f in failed]
        warnings = b.preflight.get("warnings") or []
        if warnings:
            doc.append("\n## Preflight warnings the build proceeded past\n")
            doc += [f"- {w['message']}" for w in warnings]
        doc.append(f"\n---\nBuild {b.id} · plan: {b.plan_llm.label()} · build: {b.build_llm.label()}")
        port = session.graph.blocks[target].outputs[0].name if session.graph.blocks[target].outputs else "out"
        with session.edit():
            session.upsert_artifact("build_report", target, port, f"AI build report -- {b.goal[:60]}", "\n".join(doc), key=b.id)

    def _cleanup_empty_lanes(self) -> None:
        b = self.build
        session = self.session
        used = {blk.lane for blk in session.graph.blocks.values()}
        for lane_id in list(b.owned_lanes):
            if lane_id in session.graph.lanes and lane_id not in used:
                with session.edit():
                    session.delete_lane(lane_id)

    def _end(self, phase: str) -> None:
        b = self.build
        if b is None:
            return
        if self._transaction is not None:
            if phase != DISCARDED:
                try:
                    self._cleanup_empty_lanes()
                except Exception:  # noqa: BLE001
                    pass
            try:
                self._restore_sample_mode()
            except Exception as e:  # noqa: BLE001
                b.log("error", message=f"couldn't restore sample mode: {e}")
            self._transaction.close()
            self._transaction = None
        b.set_phase(phase)

    def stop(self) -> AgentBuild:
        b = self._require()
        if b.phase in TERMINAL_PHASES:
            return b
        b.stop_requested = True
        b.log("stop_requested")
        for loop in (self._plan_loop, self._build_loop):
            if loop is not None:
                loop.cancel()
        if not self._busy():
            self._end(STOPPED)
        return b

    def discard(self) -> AgentBuild:
        """Stop, and put the graph back as it was before the build started
        changing it -- as its own undoable edit. Only while the build is
        live; once it has ended, plain Undo reverts it in one step."""
        b = self._require()
        if b.phase in TERMINAL_PHASES:
            raise BuildError("the build has already ended -- use Undo to revert it")
        b.stop_requested = True
        for loop in (self._plan_loop, self._build_loop):
            if loop is not None:
                loop.cancel()
        self.join(timeout=60)
        snapshot = self._snapshot_before
        self._end(DISCARDED)
        if snapshot is not None:
            self.session.restore_snapshot(snapshot)
            b.log("discarded")
            b.save_log()
        return b
