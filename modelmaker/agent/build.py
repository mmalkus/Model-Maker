"""One AI build's state: phase, plan, ownership, event log, counters -- and
the single choke point every tool call goes through (AgentBuild.call_tool),
which is where the guards, limits, logging and the stop flag live. See
/agent-builder-proposal.md §3."""

from __future__ import annotations

import json
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

from ..packet import ColumnRole, DataFramePacket

if TYPE_CHECKING:
    from ..runslot import RunSlot
    from ..session import ProjectSession

# Phases. "building" and "awaiting_input" are the ones during which the
# canvas is locked for the user (see api's agent lock middleware) --
# "final_run" too, since it's still the build's run.
PREFLIGHT = "preflight"
PLANNING = "planning"
AWAITING_APPROVAL = "awaiting_approval"
BUILDING = "building"
AWAITING_INPUT = "awaiting_input"
FINAL_RUN = "final_run"
DONE = "done"
DONE_WITH_ERRORS = "done_with_errors"
STOPPED = "stopped"
FAILED = "failed"
DISCARDED = "discarded"

LOCKING_PHASES = frozenset({BUILDING, AWAITING_INPUT, FINAL_RUN})
TERMINAL_PHASES = frozenset({DONE, DONE_WITH_ERRORS, STOPPED, FAILED, DISCARDED})


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
    build_tool_calls: int = 150
    custom_blocks: int = 5
    consecutive_failures_per_block: int = 3
    wall_seconds: float = 20 * 60


@dataclass
class BuildOptions:
    final_full_run: bool = True
    # None = decide from the data (see controller.approve): sample only
    # when the anchors' data is large.
    sample_rows: int | None = None
    limits: BuildLimits = field(default_factory=BuildLimits)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


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
        self.started_monotonic = time.monotonic()

        self.preflight: dict[str, Any] = {"blocking": [], "warnings": []}
        self.plan: dict[str, Any] | None = None
        self.plan_history: list[dict[str, Any]] = []  # [{plan, feedback}]
        self.pending_question: str | None = None
        self.report: str | None = None
        self.key_outputs: list[dict[str, Any]] = []
        self.results: list[dict[str, Any]] = []
        self.deviations: list[dict[str, Any]] = []
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
            self.phase = phase
        self.log("phase", phase=phase, **({"note": note} if note else {}))

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
        if time.monotonic() - self.started_monotonic > limit:
            self.stop_requested = True
            raise BuildStopped(f"The build hit its {limit / 60:.0f}-minute time limit. End your turn now.")

    def call_tool(self, name: str, args: dict[str, Any] | None) -> dict[str, Any]:
        """Run one tool call from the model. Always returns a JSON-able dict:
        the tool's result, or {"error": ...} for anything the model should
        see and react to. Never raises, except that the loop should end
        its conversation once `stop_requested` is set."""
        from .tools import TOOLS  # local: tools imports this module

        args = args or {}
        tool = TOOLS.get(name)
        try:
            self.check_stop()
            if tool is None:
                raise ToolError(f"unknown tool {name!r}")
            if self.phase not in tool.phases:
                raise ToolError(f"{name} isn't available in the {self.phase} phase")
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
            "plan_llm": asdict(self.plan_llm),
            "build_llm": asdict(self.build_llm),
            "options": {"final_full_run": self.options.final_full_run, "sample_rows": self.options.sample_rows},
            "preflight": self.preflight,
            "plan": self.plan,
            "plan_rounds": len(self.plan_history),
            "pending_question": self.pending_question,
            "report": self.report,
            "results": self.results,
            "deviations": self.deviations,
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
