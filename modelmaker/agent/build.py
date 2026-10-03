"""One AI build's state: phase, plan, ownership, event log, counters -- and
the single choke point every tool call goes through (AgentBuild.call_tool),
which is where the guards, limits, logging and the stop flag live. See
/agent-builder-proposal.md §3."""

from __future__ import annotations

import json
import os
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ..packet import ColumnRole, DataFramePacket

if TYPE_CHECKING:
    from ..runslot import RunSlot
    from ..session import ProjectSession

# Phases. "building", "awaiting_input" and "awaiting_stage_review" are the
# ones during which the canvas is locked for the user (see api's agent lock
# middleware) -- "final_run" too, since it's still the build's run.
#
# Not to be confused with the plan's *stages* (data prep, estimation,
# validation, ...): the plan is an outline of stages, and the build plans
# and builds one stage at a time, pausing in awaiting_stage_review after
# each (see controller._after_build_turn).
PREFLIGHT = "preflight"
PLANNING = "planning"
AWAITING_APPROVAL = "awaiting_approval"
BUILDING = "building"
AWAITING_INPUT = "awaiting_input"
AWAITING_STAGE_REVIEW = "awaiting_stage_review"
FINAL_RUN = "final_run"
DONE = "done"
DONE_WITH_ERRORS = "done_with_errors"
STOPPED = "stopped"
FAILED = "failed"
DISCARDED = "discarded"

LOCKING_PHASES = frozenset({BUILDING, AWAITING_INPUT, AWAITING_STAGE_REVIEW, FINAL_RUN})
TERMINAL_PHASES = frozenset({DONE, DONE_WITH_ERRORS, STOPPED, FAILED, DISCARDED})
# Waiting on the user: doesn't count against the build's time limit.
WAITING_PHASES = frozenset({PREFLIGHT, AWAITING_APPROVAL, AWAITING_INPUT, AWAITING_STAGE_REVIEW})

# A stage's status (AgentBuild.stages).
STAGE_PENDING = "pending"
STAGE_ACTIVE = "active"
STAGE_DONE = "done"
STAGE_SKIPPED = "skipped"  # dropped, with the user's agreement (tools.finish)


class ToolError(Exception):
    """A tool call the model should see as a (recoverable) error result --
    a guard refusing, bad arguments, an unknown block -- rather than a
    crash of the build."""


class BuildStopped(Exception):
    pass


@dataclass
class LLMChoice:
    provider: str | None = None
    model: str | None = None

    def label(self) -> str:
        return f"{self.provider or 'default'}/{self.model or 'default'}"


@dataclass
class BuildLimits:
    plan_tool_calls: int = 30
    # The build also plans each stage's blocks now (plan_stage).
    build_tool_calls: int = 200
    custom_blocks: int = 5
    consecutive_failures_per_block: int = 3
    wall_seconds: float = 20 * 60


@dataclass
class BuildOptions:
    final_full_run: bool = True
    # None = decide from the data (see controller.approve): sample only
    # when the anchors' data is large.
    sample_rows: int | None = None
    # Approve the plan as soon as it's submitted (when it has no open
    # questions), and go straight on to each next stage, instead of
    # waiting for the user -- see controller._after_plan_turn and
    # _after_build_turn.
    auto_build: bool = False
    # Let the build write its own polars code (add_custom_block) when no
    # registry block does the job. Off: registry blocks only, and the
    # custom-code tools aren't offered at all (tools.tools_for_phase).
    allow_custom_blocks: bool = True
    # Small context: each stage starts a fresh conversation (the stage
    # prompt carries what it needs -- the outline, what's built, and the
    # decisions so far) instead of continuing one that grows stage by
    # stage. For models with a small context window; see
    # controller._next_stage and prompts.decisions_so_far.
    small_context: bool = False
    # Decision hints: run_to adds a short rule-based "what this result
    # means / what to do next" (`next`) for blocks the build has to judge
    # -- data checks, feature screens, fits, validation tests. For small
    # models that call tools well but read statistics poorly; see hints.py.
    decision_hints: bool = False
    limits: BuildLimits = field(default_factory=BuildLimits)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# Every build's full log (see AgentBuild.save_log) is written here, one
# JSON file per build: inside the project folder once it has been saved
# (so it's versioned with the model it built), else next to the cache.
BUILD_LOG_DIRNAME = "ai_builds"


def build_log_dir(session: "ProjectSession") -> Path:
    if session.project_path is not None:
        return session.project_path.parent / BUILD_LOG_DIRNAME
    from ..session import CACHE_DIR

    return CACHE_DIR / BUILD_LOG_DIRNAME


class AgentBuild:
    def __init__(
        self,
        session: "ProjectSession",
        run_slot: "RunSlot",
        goal: str,
        anchors: list[str],
        plan_llm: LLMChoice | None = None,
        build_llm: LLMChoice | None = None,
        options: BuildOptions | None = None,
    ) -> None:
        self.id = f"build_{uuid.uuid4().hex[:8]}"
        self.session = session
        self.run_slot = run_slot
        self.goal = goal
        self.anchors = list(anchors)
        self.plan_llm = plan_llm or LLMChoice()
        self.build_llm = build_llm or LLMChoice()
        self.options = options or BuildOptions()
        self.phase = PREFLIGHT
        self.created_at = _now()
        self.ended_at: str | None = None
        self.started_monotonic = time.monotonic()
        # Time spent waiting on the user (see active_seconds).
        self._waited_seconds = 0.0
        self._waiting_since: float | None = time.monotonic()
        self.log_path: Path | None = None  # where save_log last wrote
        self._log_failed = False
        self._log_write_lock = threading.Lock()

        self.preflight: dict[str, Any] = {"blocking": [], "warnings": []}
        # The approved outline: summary, assumptions, questions, stages,
        # changes_to_existing (see tools.submit_plan).
        self.plan: dict[str, Any] | None = None
        self.plan_history: list[dict[str, Any]] = []  # [{plan, feedback}]
        # The outline's stages as the build works through them, set on
        # approval: [{key, name, goal, lane?, status, plan, summary}] --
        # `plan` is that stage's steps (tools.plan_stage), `summary` what
        # the build reported when it completed the stage.
        self.stages: list[dict[str, Any]] = []
        self.stage_index = 0
        self.pending_question: str | None = None
        self.report: str | None = None
        self.key_outputs: list[dict[str, Any]] = []
        self.results: list[dict[str, Any]] = []
        self.deviations: list[dict[str, Any]] = []
        # Problems the app itself spotted (not the model's say-so), e.g. a
        # feature fit_binning banded 'suspicious' used in a model -- shown
        # at the stage review and in the report: [{stage, what}].
        self.concerns: list[dict[str, Any]] = []
        # The user's say during the build: answers to ask_user questions
        # and stage-review feedback ({stage, question?, text}), so a stage
        # started in a fresh conversation still knows them.
        self.user_inputs: list[dict[str, Any]] = []
        self.last_question: str | None = None  # the open ask_user question
        self.error: str | None = None

        self.owned_blocks: set[str] = set()
        self.owned_wires: set[str] = set()
        self.approved_changes: dict[str, list[dict[str, Any]]] = {}  # block id -> plan entries
        self.owned_lanes: set[str] = set()
        self.lane_map: dict[str, str] = {}  # plan lane key / name asked for -> lane id
        self.sample_rows_used: int | None = None
        self.changed_sample_mode = False  # the build turned sample mode on (and must turn it off)
        self.finished = False  # the model called finish
        # Set by a successful terminal tool (submit_plan, ask_user,
        # finish): the loop ends the model's turn right there instead of
        # asking it for a closing remark -- which on a slow local model
        # can take a minute, during which the user could already act.
        self.turn_over = False

        self.counters = {"plan_tool_calls": 0, "build_tool_calls": 0, "custom_blocks": 0}
        self.failures: dict[str, int] = {}
        self.usage = {"input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0}

        self.stop_requested = False
        self._events: list[dict[str, Any]] = []
        self._lock = threading.RLock()  # events + phase
        # Serializes tool calls (the MCP route can deliver them on any
        # request thread). Separate from _lock so polling the event log
        # never waits on a long run_to.
        self._tool_lock = threading.Lock()

    # ---- events ---------------------------------------------------------

    def log(self, kind: str, **data: Any) -> dict[str, Any]:
        with self._lock:
            event = {"seq": len(self._events), "at": _now(), "kind": kind, **data}
            self._events.append(event)
            return event

    def events_since(self, cursor: int = 0) -> list[dict[str, Any]]:
        with self._lock:
            return list(self._events[cursor:])

    def set_phase(self, phase: str, note: str | None = None) -> None:
        with self._lock:
            now = time.monotonic()
            if self._waiting_since is not None:
                self._waited_seconds += now - self._waiting_since
            self._waiting_since = now if phase in WAITING_PHASES else None
            self.phase = phase
            if phase in TERMINAL_PHASES and self.ended_at is None:
                self.ended_at = _now()
        self.log("phase", phase=phase, **({"note": note} if note else {}))
        # Every phase change rewrites the log, so even a build that dies
        # mid-way (or a server that's killed) leaves its trail on disk.
        self.save_log()

    def active_seconds(self) -> float:
        """Seconds since the start, less the time spent waiting on the user
        (reviewing the plan or a stage, answering a question)."""
        with self._lock:
            now = time.monotonic()
            waiting = now - self._waiting_since if self._waiting_since is not None else 0.0
            return now - self.started_monotonic - self._waited_seconds - waiting

    # ---- stages ---------------------------------------------------------

    @property
    def current_stage(self) -> dict[str, Any] | None:
        if 0 <= self.stage_index < len(self.stages):
            return self.stages[self.stage_index]
        return None

    def is_last_stage(self) -> bool:
        return self.stage_index >= len(self.stages) - 1

    def add_concern(self, what: str) -> None:
        if any(c["what"] == what for c in self.concerns):
            return
        stage = self.current_stage
        self.concerns.append({"stage": stage["key"] if stage else None, "what": what})
        self.log("concern", what=what)

    # ---- persisted log --------------------------------------------------

    def log_record(self) -> dict[str, Any]:
        """Everything about this build worth keeping after the server is
        gone: who/what/when, the models that ran each phase, the options,
        every plan round, the outcome, token usage and the full event log."""
        with self._lock:
            events = list(self._events)
            phase = self.phase
        return {
            "id": self.id,
            "goal": self.goal,
            "anchors": self.anchors,
            "anchor_names": {a: b.name for a in self.anchors if (b := self.session.graph.blocks.get(a)) is not None},
            "project": {
                "name": self.session.project_name,
                "path": str(self.session.project_path.parent) if self.session.project_path else None,
            },
            "phase": phase,
            "created_at": self.created_at,
            "ended_at": self.ended_at,
            "duration_seconds": round(time.monotonic() - self.started_monotonic, 1),
            "models": {
                "plan": {**asdict(self.plan_llm), "label": self.plan_llm.label()},
                "build": {**asdict(self.build_llm), "label": self.build_llm.label()},
            },
            "options": asdict(self.options),
            "preflight": self.preflight,
            "plan": self.plan,
            "plan_history": self.plan_history,
            "stages": self.stages,
            "pending_question": self.pending_question,
            "report": self.report,
            "key_outputs": self.key_outputs,
            "results": self.results,
            "deviations": self.deviations,
            "concerns": self.concerns,
            "owned_blocks": sorted(self.owned_blocks),
            "owned_lanes": sorted(self.owned_lanes),
            "sample_rows_used": self.sample_rows_used,
            "counters": self.counters,
            "usage": self.usage,
            "error": self.error,
            "events": events,
        }

    def save_log(self) -> Path | None:
        """Write log_record() to build_log_dir()/<id>.json (atomically).
        Never raises -- an unwritable folder must not fail the build; the
        failure is noted in the event log instead (once)."""
        with self._log_write_lock:
            try:
                directory = build_log_dir(self.session)
                directory.mkdir(parents=True, exist_ok=True)
                path = directory / f"{self.id}.json"
                tmp = path.with_suffix(".json.tmp")
                tmp.write_text(json.dumps(self.log_record(), indent=2, default=str) + "\n", encoding="utf-8")
                os.replace(tmp, path)
            except OSError as e:
                if not self._log_failed:
                    self._log_failed = True
                    self.log("error", message=f"couldn't save the build log: {e}")
                return None
            # A build started before the project was first saved moves
            # into the project folder on its next write; drop the old copy.
            if self.log_path is not None and self.log_path != path:
                try:
                    self.log_path.unlink(missing_ok=True)
                except OSError:
                    pass
            self.log_path = path
            return path

    # ---- ownership ------------------------------------------------------

    def is_owned(self, block_id: str) -> bool:
        return block_id in self.owned_blocks

    def require_block(self, block_id: str) -> Any:
        block = self.session.graph.blocks.get(block_id)
        if block is None:
            raise ToolError(f"no such block: {block_id!r} -- call get_graph to see current block ids")
        return block

    def require_modifiable(self, block_id: str, action: str) -> Any:
        """A block the build may change: one it created, or a pre-existing
        one the approved plan listed under changes_to_existing."""
        block = self.require_block(block_id)
        if self.is_owned(block_id):
            return block
        if block_id in self.approved_changes:
            return block
        raise ToolError(
            f"cannot {action} block {block.name!r} ({block_id}): it belongs to the user and the approved plan "
            "doesn't list a change to it. Build around it (wire from its outputs), or call ask_user if the "
            "goal really needs it changed."
        )

    def record_approved_change(self, block_id: str, change: str) -> None:
        """Log (and stamp into provenance) a change to a pre-existing block
        that the approved plan authorized."""
        if self.is_owned(block_id):
            return
        block = self.session.graph.blocks[block_id]
        entry = {"build_id": self.id, "at": _now(), "change": change}
        prov = dict(block.provenance or {})
        prov["changes"] = [*prov.get("changes", []), entry]
        block.provenance = prov
        self.log("approved_change", block=block_id, change=change)

    def provenance_for(self, plan_step: str | None) -> dict[str, Any]:
        return {
            "source": "agent",
            "build_id": self.id,
            "at": _now(),
            "goal": self.goal,
            "plan_llm": self.plan_llm.label(),
            "build_llm": self.build_llm.label(),
            "plan_step": plan_step,
            "stage": stage["key"] if (stage := self.current_stage) else None,
            "modified_by_user": False,
        }

    # ---- data guards ----------------------------------------------------

    def scope_block_ids(self) -> set[str]:
        """The user's side of the graph this build is working from: the
        anchors and everything upstream of them."""
        return self.session.graph.ancestors(self.anchors)

    def current_output(self, block_id: str, port: str | None = None) -> tuple[str, Any]:
        """(port, value) of a block's last successful output. Raises
        ToolError when there's none."""
        block = self.require_block(block_id)
        ports = [p.name for p in block.outputs]
        if not ports:
            raise ToolError(f"block {block.name!r} has no output ports")
        port = port or ports[0]
        if port not in ports:
            raise ToolError(f"block {block.name!r} has no output port {port!r}; ports: {ports}")
        runner = self.session.runner
        st = runner.state.get(block_id)
        if st is None or st.last_successful_key is None:
            raise ToolError(f"block {block.name!r} has no output yet (status {runner.status(block_id)}) -- run_to it first")
        entry = runner.cache.get(st.last_successful_key)
        if entry is None or port not in entry.outputs:
            raise ToolError(f"block {block.name!r}'s output isn't cached any more -- run_to it again")
        return port, entry.outputs[port]

    def excluded_columns(self) -> set[str]:
        """Columns the user tagged `excluded` anywhere in the build's scope
        -- never allowed as a feature."""
        out: set[str] = set()
        for bid in self.scope_block_ids():
            block = self.session.graph.blocks.get(bid)
            if block is None:
                continue
            out |= {c for c, r in block.column_role_overrides.items() if r == ColumnRole.EXCLUDED.value}
            for p in block.outputs:
                try:
                    _, value = self.current_output(bid, p.name)
                except ToolError:
                    continue
                if isinstance(value, DataFramePacket):
                    out |= {n for n, m in value.schema_meta.items() if m.role == ColumnRole.EXCLUDED}
        return out

    # ---- tool dispatch --------------------------------------------------

    def check_stop(self) -> None:
        if self.stop_requested:
            raise BuildStopped("The user stopped this build. End your turn now without calling more tools.")
        limit = self.options.limits.wall_seconds
        if self.active_seconds() > limit:
            self.stop_requested = True
            raise BuildStopped(f"The build hit its {limit / 60:.0f}-minute time limit. End your turn now.")

    def call_tool(self, name: str, args: dict[str, Any] | None) -> dict[str, Any]:
        """Run one tool call from the model. Always returns a JSON-able dict:
        the tool's result, or {"error": ...} for anything the model should
        see and react to. Never raises, except that the loop should end
        its conversation once `stop_requested` is set."""
        from .tools import CUSTOM_TOOLS, TOOLS  # local: tools imports this module

        args = args or {}
        tool = TOOLS.get(name)
        try:
            self.check_stop()
            if tool is None:
                raise ToolError(f"unknown tool {name!r}")
            if self.phase not in tool.phases:
                raise ToolError(f"{name} isn't available in the {self.phase} phase")
            if name in CUSTOM_TOOLS and not self.options.allow_custom_blocks:
                raise ToolError("custom blocks are turned off for this build -- use registry blocks, or ask_user")
            stage = self.current_stage
            if stage is not None and stage["status"] == STAGE_DONE and not self.finished:
                # complete_stage ended the turn; a model that keeps going
                # (the claude CLI loop can't be cut off mid-turn) mustn't
                # start on the next stage before the user has reviewed this one.
                raise ToolError(f"stage {stage['name']!r} is complete and waiting for the user's review -- end your turn now")
            counter = "plan_tool_calls" if self.phase == PLANNING else "build_tool_calls"
            limit = getattr(self.options.limits, counter)
            self.counters[counter] += 1
            if self.counters[counter] > limit:
                self.stop_requested = True
                raise BuildStopped(f"The build hit its limit of {limit} tool calls for this phase. End your turn now.")
            with self._tool_lock:
                result = tool.fn(self, **args)
            ok = True
        except BuildStopped as e:
            result, ok = {"error": str(e), "stopped": True}, False
        except ToolError as e:
            result, ok = {"error": str(e)}, False
        except TypeError as e:
            result, ok = {"error": f"bad arguments for {name}: {e}"}, False
        except Exception as e:  # noqa: BLE001 -- surfaced to the model, which can often recover
            result, ok = {"error": f"{type(e).__name__}: {e}"}, False
        self.log("tool", tool=name, args=_truncate(args), ok=ok, result=_truncate(result))
        return result

    # ---- serialization for the API --------------------------------------

    def to_dict(self, cursor: int = 0) -> dict[str, Any]:
        return {
            "id": self.id,
            "goal": self.goal,
            "anchors": self.anchors,
            "phase": self.phase,
            "created_at": self.created_at,
            "ended_at": self.ended_at,
            "plan_llm": asdict(self.plan_llm),
            "build_llm": asdict(self.build_llm),
            "options": {
                "final_full_run": self.options.final_full_run,
                "sample_rows": self.options.sample_rows,
                "auto_build": self.options.auto_build,
                "allow_custom_blocks": self.options.allow_custom_blocks,
                "small_context": self.options.small_context,
                "decision_hints": self.options.decision_hints,
            },
            "log_path": str(self.log_path) if self.log_path else None,
            "preflight": self.preflight,
            "plan": self.plan,
            "plan_rounds": len(self.plan_history),
            "stages": self.stages,
            "stage_index": self.stage_index,
            "pending_question": self.pending_question,
            "report": self.report,
            "results": self.results,
            "deviations": self.deviations,
            "concerns": self.concerns,
            "owned_blocks": sorted(self.owned_blocks),
            "sample_rows_used": self.sample_rows_used,
            "counters": self.counters,
            "usage": self.usage,
            "error": self.error,
            "events": self.events_since(cursor),
            "next_cursor": len(self._events),
        }


def _truncate(value: Any, limit: int = 4000) -> Any:
    text = json.dumps(value, default=str)
    if len(text) <= limit:
        return value
    return {"truncated": text[:limit] + "..."}
