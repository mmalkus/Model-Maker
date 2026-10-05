"""Guided builds, for small (local) models: instead of every tool and an
open-ended task, the app asks one narrow question at a time, each a fresh
short request answered with one tool call.

- Plan: the goal, a pure-code profile of the data (profile.py), the goal's
  catalogue built from the registry, and the rules -> submit_plan. A
  rejected plan comes back with its errors.
- Build, per stage and step: the stage goal, the data and results so far
  (each under a label -- D1, T1, M1, R1 -- the app maps back to block and
  port), the stage's blocks with their docs, and what the last step did ->
  place_block or stage_done.

GuidedLoop wraps the provider's loop behind the start/resume interface the
controller drives every loop through, so approval, stage reviews, undo, the
log and the report are unchanged."""

from __future__ import annotations

import inspect
import json
from typing import Any

from ..blocks.base import BLOCK_REGISTRY
from ..packet import DataFramePacket
from . import catalogue, prompts
from . import profile as prof
from .build import PLANNING, STAGE_DONE, AgentBuild, ToolError
from .loop import AgentLoop, Answer, LoopOutcome

PLAN_QUESTIONS = 4  # submit_plan attempts per planning turn
STAGE_QUESTIONS = 10  # questions per stage before asking the user
# Planning is one question that decides everything after it: it gets at
# least this thinking budget even when build questions get less.
PLAN_THINK_TOKENS = 4096
GUIDED_ONLY_PROVIDERS = frozenset({"local"})  # the built-in model runs guided builds only

PLAN_SYSTEM = (
    "You plan credit-risk models in Model-Maker. A model is a sequence of stages; each stage places blocks from "
    "the list you are given. Answer with one submit_plan call."
)
BUILD_SYSTEM = (
    "You build credit-risk models in Model-Maker, one block at a time. Each question gives a stage goal, the data "
    "and results you can use (each under a label such as D1 or M1), the blocks this stage may use and what the "
    "last step did. Answer with one tool call: place_block for the next block, or stage_done when the stage goal "
    "is met. You never see data rows. Fit anything learned from data on the development sample only."
)
PLAN_RULES = """Rules for the plan:
1. Stages in build order. Each stage lists the blocks it will use -- block names from the list above, never the group names.
2. Split the sample before any stage that learns from the target (binning, feature selection, fitting, calibration).
3. Fit everything on the development sample only; apply fitted results (binning, model) to the holdout samples; validation compares development with the holdout(s).
4. Name each sample in the stage goals and use those names consistently (e.g. "development" and "out-of-time").
5. Only the stages the goal needs -- but all of them: the plan must end in what the goal asks for (e.g. the fitted model and the validation it names), not stop at preparing or splitting the data."""

_S = {"type": "string"}
PLAN_TOOL = {"type": "function", "function": {
    "name": "submit_plan",
    "description": "Submit the plan: the stages in build order.",
    "parameters": {"type": "object", "required": ["summary", "stages"], "properties": {
        "summary": {"type": "string", "description": "One sentence: what will be built."},
        "stages": {"type": "array", "items": {"type": "object", "required": ["key", "name", "goal", "blocks"], "properties": {
            "key": {"type": "string", "description": "Short id, e.g. split, est, val."},
            "name": _S,
            "goal": {"type": "string", "description": "One or two sentences: what the stage does and the decisions that shape it."},
            "blocks": {"type": "array", "items": _S, "description": "Block names from the list."},
        }}},
    }},
}}
BUILD_TOOLS = [
    {"type": "function", "function": {
        "name": "place_block",
        "description": "Place the next block: the app adds it, wires its inputs and runs it.",
        "parameters": {"type": "object", "required": ["block", "inputs"], "properties": {
            "block": {"type": "string", "description": "A block name: one of the stage's blocks, or another if the stage needs it."},
            "inputs": {"type": "array", "items": _S, "description": "Labels of the data/results it takes, e.g. [\"D2\", \"M1\"]."},
            "settings": {"type": "object", "description": "Its params; leave out any you keep at the default."},
            "name": {"type": "string", "description": "A short name for it."},
            "why": _S,
        }},
    }},
    {"type": "function", "function": {
        "name": "stage_done",
        "description": "The stage goal is met: summarise what it built and decided.",
        "parameters": {"type": "object", "required": ["summary"], "properties": {"summary": _S}},
    }},
]


def is_guided_provider(provider: str | None) -> bool:
    """Guided by default: local models (the built-in one, LM Studio, llama.cpp, ...)."""
    return provider in ("lmstudio", "local")


def can_guide(loop: AgentLoop) -> bool:
    return type(loop).ask is not AgentLoop.ask


# ---- what the model sees ---------------------------------------------------------


def _outputs(build: AgentBuild, bid: str) -> dict[str, Any]:
    st = build.session.runner.state.get(bid)
    entry = build.session.runner.cache.get(st.last_successful_key) if st and st.last_successful_key else None
    return entry.outputs if entry is not None else {}


def _brief(value: Any) -> str:
    """A fitted artifact or result by kind (and a metric by its numbers) --
    never its JSON, which can carry values out of the data."""
    if not isinstance(value, dict):
        return type(value).__name__
    nums = {k: round(v, 4) if isinstance(v, float) else v for k, v in value.items() if isinstance(v, (int, float)) and not isinstance(v, bool)}
    return f"{value.get('kind') or 'result'}" + (f": {nums}" if nums and len(nums) <= 8 else "")


def available(build: AgentBuild, replaced: dict[str, str]) -> tuple[str, dict[str, tuple[str, str, str]]]:
    """What a stage can wire from -- the anchors (or what replaced them) and
    every block this build made, one output per labelled line: the first
    dataset as a column profile, later ones by what changed. Returns the
    text and label -> (block id, port, port type)."""
    graph = build.session.graph
    labels: dict[str, tuple[str, str, str]] = {}
    lines: list[str] = []
    base: set[str] | None = None
    for bid in graph.topo_order():
        if bid in replaced or (bid not in build.anchors and bid not in build.owned_blocks) or build.session.runner.status(bid) != "green":
            continue
        block = graph.blocks[bid]
        spec = BLOCK_REGISTRY.get(block.category)
        types = {p.name: p.type for p in block.outputs}
        for port, value in _outputs(build, bid).items():
            table = isinstance(value, DataFramePacket) and spec is not None and port in spec.aggregate_outputs
            prefix = "T" if table else "D" if isinstance(value, DataFramePacket) else "R" if types.get(port) in ("scalar_metric", "image") else "M"
            label = f"{prefix}{sum(k[0] == prefix for k in labels) + 1}"
            labels[label] = (bid, port, types.get(port, "any"))
            head = f'{label} = {port} of "{block.name}" ({block.category})'
            if prefix != "D":
                lines.append(f"{head}: " + (f"statistics table, {value.data.height} rows" if table else _brief(value)))
                continue
            cols = prof.profile(value)
            names = {c["name"] for c in cols}
            if base is None:
                base = names
                lines.append(f"{head}: " + prof.describe(cols, value.data.height).replace("\n", "\n    "))
                continue
            target = next((c for c in cols if c["role"] == "target" and "rate_of_1" in c), None)
            added, dropped = sorted(names - base), sorted(base - names)
            lines.append(f"{head}: {value.data.height:,} rows" + (f", target rate {target['rate_of_1']:.2%}" if target else "")
                         + (f"; added columns: {', '.join(added[:40])}" if added else "") + (f"; dropped columns: {', '.join(dropped)}" if dropped else ""))
        if build.options.decision_hints:
            lines += [f"    next: {hint}" for hint in prompts.block_hints(build, bid)]
    return "\n".join(lines) or "(nothing yet)", labels


def plan_prompt(build: AgentBuild, said: str | None, previous: dict | None, errors: list[str] | None) -> str:
    data = "\n\n".join(
        f'"{build.session.graph.blocks[bid].name}" ({port}): ' + prof.describe(prof.profile(v), v.data.height)
        for bid in build.anchors for port, v in _outputs(build, bid).items() if isinstance(v, DataFramePacket)
    )
    cat = "\n".join(
        f"{tag} ({catalogue.TAGS[tag]}):\n" + "\n".join(f"  - {b['category']}: {b['summary']}" for b in blocks)
        for tag, blocks in catalogue.goal_catalogue(build.goal)
    )
    parts = [f"Goal: {build.goal}", f"Data:\n{data or '(no data)'}", f"Blocks you can use, by group:\n{cat}", PLAN_RULES]
    if said:
        parts.append(f"The user said about the last plan: {said}")
    if errors:
        parts.append(f"Your last plan:\n{json.dumps(previous, indent=1)}\n\nIt was rejected:\n- " + "\n- ".join(errors)
                     + "\n\nFix those problems and submit the plan again.")
    else:
        parts.append("Write the plan: for each stage a key, a name, a goal of one or two sentences, and its blocks.")
    return "\n\n".join(parts)


def stage_prompt(build: AgentBuild, data: str, last: str) -> str:
    stage = build.current_stage
    earlier = [f"- {st['name']}: {st['summary'].strip()[:300]}" for st in build.stages[: build.stage_index] if st.get("summary")]
    return "\n\n".join([
        f"Stage {build.stage_index + 1} of {len(build.stages)}: {stage['name']}\nGoal: {stage['goal']}",
        *(["Earlier stages:\n" + "\n".join(earlier)] if earlier else []),
        f"Data and results you can use (refer to them by label):\n{data}",
        f"Blocks for this stage:\n{prompts.block_docs(_stage_blocks(stage))}\n\nIf the stage needs a block that isn't "
        "listed (e.g. predict, to score a sample with a fitted model), you may place it anyway -- say why.",
        f"Last step: {last}",
        "What block should we place next, with which settings and which inputs (labels)? Or is this stage done?",
    ])


# What a goal asks for -> a block of this tag the plan must include.
_COVERAGE = {
    "regression": (("model", "scorecard", "regression", "pd", "lgd", "ccf"), "a stage that fits the model (e.g. logistic_regression)"),
    "performance": (("validat", "gini", "ks", "auc", "discriminat"), "a stage that validates it (e.g. auc_gini, ks_test)"),
}


def _uncovered(goal: str, plan: dict[str, Any]) -> list[str]:
    """What the goal asks for that no stage of the plan does -- a small model
    can submit a plan that stops at preparing the data."""
    words = goal.casefold().replace(",", " ").replace("(", " ").replace(")", " ").split()
    listed = {b for st in plan.get("stages") or [] if isinstance(st, dict) for b in st.get("blocks") or []}
    return [
        f"the goal asks for it, so the plan needs {need}"
        for tag, (keys, need) in _COVERAGE.items()
        if any(w.startswith(k) for w in words for k in keys) and not listed & set(catalogue.tag_block_names(tag))
    ]


def _stage_blocks(stage: dict[str, Any]) -> list[str]:
    return [c for c in stage.get("blocks") or [] if c in BLOCK_REGISTRY and c not in catalogue.AGENT_DISALLOWED]


# ---- the loop ---------------------------------------------------------------------


class GuidedLoop(AgentLoop):
    """Asks the questions over the provider loop's ask(). start/resume act on
    the build's phase; a resume within the same stage carries the user's
    words into the next question."""

    compact = True

    def __init__(self, inner: AgentLoop) -> None:
        self.inner = inner
        self.replaced: dict[str, str] = {}  # anchor -> the block that parsed its dates
        self._stage_key: str | None = None
        self._warned: str | None = None  # the stage already told about its unplaced blocks
        self._last = ""

    def __getattr__(self, name: str) -> Any:  # model, context_tokens, ... of the provider's loop
        return getattr(self.inner, name)

    def cancel(self) -> None:
        self.inner.cancel()

    def start(self, build, system, prompt, tools) -> LoopOutcome:
        if build.phase == PLANNING:
            return self._plan(build, build.plan_history[-1]["feedback"] if build.plan_history else None)
        return self._build_stage(build)

    def resume(self, build, prompt, tools) -> LoopOutcome:
        return self.start(build, "", prompt, tools)

    def _ask(self, build: AgentBuild, system: str, prompt: str, tools: list[dict]) -> Answer:
        build.check_stop()
        answer = self.inner.ask(build, system, prompt, tools)
        if answer.text:
            build.log("assistant", text=answer.text)
        return answer

    def _plan(self, build: AgentBuild, said: str | None) -> LoopOutcome:
        budget = getattr(self.inner, "think_tokens", None)
        if budget is not None:
            self.inner.think_tokens = max(budget, PLAN_THINK_TOKENS)
        try:
            return self._plan_questions(build, said)
        finally:
            if budget is not None:
                self.inner.think_tokens = budget

    def _plan_questions(self, build: AgentBuild, said: str | None) -> LoopOutcome:
        previous, errors = None, None
        for _ in range(PLAN_QUESTIONS):
            answer = self._ask(build, PLAN_SYSTEM, plan_prompt(build, said, previous, errors), [PLAN_TOOL])
            if answer.tool != "submit_plan" or answer.args is None:
                errors = [answer.error or "you answered without calling submit_plan -- call it with the plan"]
                continue
            if missing := _uncovered(build.goal, answer.args):
                previous, errors = answer.args, missing
                continue
            result = build.call_tool("submit_plan", {"plan": answer.args})
            if result.get("ok"):
                return LoopOutcome(final_text=str(answer.args.get("summary") or "Plan submitted."))
            previous = answer.args
            errors = [e.strip("- ") for e in str(result.get("error", "")).split("\n")[1:] if e.strip()]
        return LoopOutcome(final_text=f"no valid plan after {PLAN_QUESTIONS} tries: {'; '.join(errors or [])}")

    def _build_stage(self, build: AgentBuild) -> LoopOutcome:
        stage = build.current_stage
        if stage["key"] != self._stage_key:
            self._stage_key, self._last = stage["key"], "none yet -- this is the stage's first step."
            if build.stage_index == 0:
                self._parse_dates(build)
        elif build.user_inputs:
            self._last += f"\nThe user then said: {build.user_inputs[-1]['text']}"
        for _ in range(STAGE_QUESTIONS):
            data, labels = available(build, self.replaced)
            answer = self._ask(build, BUILD_SYSTEM, stage_prompt(build, data, self._last), BUILD_TOOLS)
            args = answer.args or {}
            if answer.tool == "place_block":
                self._last = self._place(build, stage, labels, args)
            elif answer.tool == "stage_done" and (missing := self._unplaced(stage)):
                # Once per stage: blocks the plan listed aren't placed yet.
                self._warned = stage["key"]
                self._last = (f"not yet placed: {', '.join(missing)} -- place them, or call stage_done again to "
                              "finish the stage without them.")
            elif answer.tool == "stage_done":
                result = build.call_tool("stage_done", {"summary": str(args.get("summary") or "")})
                if result.get("ok"):
                    return LoopOutcome(final_text=str(args.get("summary") or ""))
                self._last = (f"stage_done refused: {result.get('error')}. This stage's blocks: "
                              f"{', '.join(_stage_blocks(stage))} -- place one with place_block.")
            else:
                self._last = answer.error or "you answered without a tool call -- answer with place_block or stage_done."
            if build.stop_requested or stage["status"] == STAGE_DONE or build.finished:
                break
        return LoopOutcome(final_text=f"Stage {stage['name']!r} isn't done after {STAGE_QUESTIONS} questions. Last step: {self._last}")

    def _unplaced(self, stage: dict[str, Any]) -> list[str]:
        """The stage's listed blocks not placed yet -- the app's own steps
        count -- or none once warned (the model may finish without them)."""
        placed = {s["category"] for s in (stage.get("plan") or {}).get("steps") or []}
        if self._warned == stage["key"]:
            return []
        return [c for c in _stage_blocks(stage) if c not in placed]

    def _parse_dates(self, build: AgentBuild) -> None:
        """Before the first question, in code: anchors with text dates get a
        parse_dates step, so the model works on real dates."""
        steps = [
            {"ref": f"{build.current_stage['key']}_dates{i + 1}", "category": "parse_dates", "by_app": True,
             "name": f"{build.session.graph.blocks[bid].name} (dates parsed)", "why": "the app converts text dates",
             "inputs": [{"port": "df", "from": bid, "from_port": port}]}
            for i, (bid, port) in enumerate(
                (bid, port) for bid in build.anchors for port, v in _outputs(build, bid).items()
                if isinstance(v, DataFramePacket) and prof.has_text_dates(prof.profile(v))
            )
        ]
        if not steps:
            return
        result = build.call_tool("plan_stage", {"steps": steps})
        for step, entry in zip(steps, result.get("steps") or []):
            if entry.get("status") == "green":
                self.replaced[step["inputs"][0]["from"]] = entry["block"]
        self._last = "the app parsed the text dates into real dates (shown above) -- none of the stage's own blocks placed yet."

    def _place(self, build: AgentBuild, stage: dict[str, Any], labels: dict, args: dict[str, Any]) -> str:
        category = str(args.get("block") or "")
        spec = BLOCK_REGISTRY.get(category)
        if spec is None or category in catalogue.AGENT_DISALLOWED:
            return f"place_block failed: {category!r} isn't a block you can use; this stage's blocks are {', '.join(_stage_blocks(stage))}."
        docs = f"\n\n{category}'s docs:\n{prompts.block_docs([category])}"
        try:
            inputs = _wire(spec, labels, args.get("inputs") or [])
        except ToolError as e:
            return f"place_block failed: {e}"
        params = dict(args.get("settings") or {})
        dropped = _fix_columns(build, params, inputs)
        filled, unusable = _fill_features(build, spec, params, inputs)
        dropped += [f"{f} (missing values or not numeric)" for f in unusable]
        existing = list((stage.get("plan") or {}).get("steps") or [])
        ref = f"{stage['key']}_{len(existing) + 1}"
        step = {"ref": ref, "category": category, "name": str(args.get("name") or category),
                "params": params, "inputs": inputs, "why": str(args.get("why") or "")}
        result = build.call_tool("plan_stage", {"steps": existing + [step]})
        if "error" in result:
            return f"place_block failed: {result['error']}{docs}"
        entry = next((e for e in result.get("steps") or [] if e.get("ref") == ref), {})
        if entry.get("status") != "green":
            # Don't leave a failed block behind: out of the graph and the plan.
            if entry.get("block"):
                build.call_tool("delete_block", {"block": entry["block"]})
            stage["plan"]["steps"] = existing
            return f"{category} failed to run and was removed: {entry.get('error') or str(result)[:400]}{docs}"
        if category not in _stage_blocks(stage):
            build.call_tool("note_deviation", {"what": f"used {category}, not in stage {stage['key']}'s plan",
                                               "why": step["why"] or "the stage needed it", "plan_step": ref})
        return (f"placed {category} on {', '.join(map(str, args.get('inputs') or [])) or 'no inputs'} -- it ran OK; its outputs are "
                "listed above under new labels." + (f" The app chose its features: {', '.join(filled)}." if filled else "")
                + (f" Not columns, so left out: {', '.join(dropped)}." if dropped else ""))


def _input_sample(build: AgentBuild, inputs: list[dict[str, str]]) -> DataFramePacket | None:
    return next((v for i in inputs if isinstance(v := _outputs(build, i["from"]).get(i["from_port"]), DataFramePacket)), None)


def _fix_columns(build: AgentBuild, params: dict[str, Any], inputs: list[dict[str, str]]) -> list[str]:
    """Column names in settings (cols, features, *_col, ...) that can only
    mean one column: a role word ("target", "id") as the one column with that
    role; names that aren't columns at all are dropped from lists. Returns
    what it dropped, to tell the model."""
    sample = _input_sample(build, inputs)
    if sample is None:
        return []
    roles = {name: getattr(m.role, "value", str(m.role)) for name, m in sample.schema_meta.items()}

    def fix(name: Any) -> Any:
        if name in roles:
            return name
        hits = [c for c, r in roles.items() if r == str(name).strip().lower()]
        return hits[0] if len(hits) == 1 else None

    dropped = []
    for key, value in list(params.items()):
        if not (key in ("cols", "columns", "features", "target") or key.endswith(("_col", "_cols"))):
            continue
        if isinstance(value, str):
            # Not a column at all: left out, so the param's default -- or its
            # role auto-fill (date_col from the date role) -- applies.
            if fix(value):
                params[key] = fix(value)
            else:
                del params[key]
                dropped.append(f"{key}={value}")
        elif isinstance(value, list):
            fixed = [fix(v) for v in value]
            dropped += [str(v) for v, f in zip(value, fixed) if f is None]
            params[key] = list(dict.fromkeys(f for f in fixed if f))
    return dropped


def _fill_features(build: AgentBuild, spec, params: dict[str, Any], inputs: list[dict[str, str]]) -> tuple[list[str], list[str]]:
    """A block that needs a `features` list the model didn't give (or gave
    as something other than a list): the app fills it from its input's
    profile -- the WoE columns when there are any, else every numeric
    column without missing values that isn't the target, an id, excluded
    or a prediction. Returns (what it filled in, what it left out)."""
    param = next((p for p in catalogue.block_params(spec) if p["name"] == "features"), None)
    given = params.get("features")
    sample = _input_sample(build, inputs)
    if param is None or sample is None or (given is None and not param.get("required")):
        return [], []
    cols = [c for c in prof.profile(sample) if c["role"] not in ("target", "id", "excluded", "predicted")]
    usable = [c["name"] for c in cols if "min" in c and "p75" not in c and not c.get("null_share")]
    if isinstance(given, list) and given:
        # A block that needs complete numeric features (its docs say
        # null-free): leave out the ones that aren't -- the run's error
        # wouldn't say which column broke it.
        if "null-free" not in (inspect.getdoc(spec.fn) or ""):
            return [], []
        params["features"] = [f for f in given if f in usable]
        return [], [f for f in given if f not in usable]
    chosen = [c["name"] for c in cols if c["name"].endswith("_woe")] or usable
    if chosen:
        params["features"] = chosen
    return chosen, []


def _wire(spec, labels: dict, given: list[Any]) -> list[dict[str, str]]:
    """Labels -> input ports: each to the first open input of its type, so
    order only matters between same-typed inputs (compare_samples)."""
    open_ports, wired = list(spec.inputs), []
    for raw in given:
        label = str(raw).strip().upper()
        if label not in labels:
            raise ToolError(f"{raw!r} isn't one of the labels above ({', '.join(labels) or 'none yet'})")
        bid, port, ptype = labels[label]
        fit = next((p for p in open_ports if p.type == ptype or "any" in (p.type, ptype)), None)
        if fit is None:
            ins = ", ".join(f"{p.name} ({p.type})" for p in spec.inputs) or "none"
            raise ToolError(f"{label} is a {ptype}, which no free input of {spec.category} takes (its inputs: {ins})")
        open_ports.remove(fit)
        wired.append({"port": fit.name, "from": bid, "from_port": port})
    if missing := [p for p in open_ports if p.required]:
        raise ToolError(f"{spec.category} also needs " + ", ".join(f"{p.name} ({p.type})" for p in missing) + " -- give a label for each")
    return wired
