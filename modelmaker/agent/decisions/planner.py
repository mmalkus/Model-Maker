"""The plan loop: decide the model type, then per phase ask "which block
next?" over the options on offer (DONE always among them), run the pick,
and repeat until DONE. Blocks run for real on the workspace, so each next
decision sees the facts the last block produced."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .core import Decider, choice
from .phases import DONE, MODEL_TYPE_Q, PHASES, Phase, model_type_rule
from .workspace import Workspace, dataset_facts

MAX_STEPS_PER_PHASE = 10


@dataclass
class PlanResult:
    model_type: str
    steps: list[dict[str, Any]] = field(default_factory=list)
    stopped: str | None = None  # why planning ended early, if it did

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_type": self.model_type,
            "steps": self.steps,
            "stopped": self.stopped,
        }


def menu_question(phase: Phase, offered: list) -> dict[str, Any]:
    criteria = {o.key: o.describe + ("" if o.required else " (optional)") for o in offered}
    if not any(o.required for o in offered):
        criteria[DONE] = f"move to the next phase: {phase.done_describe}"
    return choice(
        f"Credit-risk PD build, phase '{phase.title}'. Which block comes next? Follow the usual credit-risk order "
        "and don't skip a check the facts call for.",
        criteria,
    )


def menu_state(ws: Workspace, phase: Phase) -> dict[str, Any]:
    facts = dataset_facts(ws)
    facts.pop("status_candidates", None)
    extra = {k: v for k, v in phase.extra_facts(ws).items() if v not in (None, [], {})}
    return {
        **facts,
        **extra,
        "done_in_phase": [k for k in ws.built if k in {o.key for o in phase.options}],
    }


def plan(ws: Workspace, decider: Decider) -> PlanResult:
    facts = dataset_facts(ws)
    model_type = decider.ask("model_type", facts, {"type": MODEL_TYPE_Q}, {"type": model_type_rule})["type"]
    ws.model_type = model_type
    result = PlanResult(model_type)
    phases = PHASES.get(model_type)
    if phases is None:
        result.stopped = f"phases for {model_type} aren't built yet"
        return result

    for phase in phases:
        for _ in range(MAX_STEPS_PER_PHASE):
            offered = [o for o in phase.options if o.offered(ws)]
            if not offered:
                break  # nothing left to do here: DONE without asking
            # DONE is on the menu only once nothing required is left (menu_question).
            q = menu_question(phase, offered)
            first = offered[0].key
            pick = decider.ask(
                f"{phase.key}.next",
                menu_state(ws, phase),
                {"next": q},
                {"next": lambda f, first=first: first},
                drop_order=["done_in_phase"],
            )["next"]
            if pick == DONE:
                result.steps.append(
                    {
                        "phase": phase.key,
                        "option": DONE,
                        "skipped": [o.key for o in offered],
                    }
                )
                break
            option = next(o for o in offered if o.key == pick)
            try:
                step = option.run(ws, decider)
            except Exception as e:  # a block that fails is reported, and its option closed
                step = {"block": None, "error": f"{type(e).__name__}: {e}"}
            ws.built.append(option.key)
            ws.resolved.add(option.key)
            result.steps.append({"phase": phase.key, "option": option.key, **step})
    return result
