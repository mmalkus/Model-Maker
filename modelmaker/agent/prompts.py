"""System prompts and opening messages for an AI build's two phases. The
plan phase produces an outline of stages, not blocks: which blocks a stage
needs depends on what the earlier stages produce, so the build plans each
stage's blocks just before building it. The build phase starts a fresh
conversation seeded with the approved outline -- not the planning
transcript -- so the outline is the contract and the plan and build LLMs
can be different providers (see /agent-builder-proposal.md §4.2, §4.3,
§9.1). One build conversation runs through every stage."""

from __future__ import annotations

import json
import re
from typing import Any

from ..blocks.base import BLOCK_REGISTRY
from ..packet import ColumnRole, DataFramePacket
from . import catalogue
from .build import AgentBuild, ToolError
from .tools import is_statistics_table, table_rows

COMMON = """You are building a model inside Model-Maker, a visual, Polars-based \
modelling tool used by credit-risk and actuarial model developers. A model \
is a graph of blocks connected by wires, arranged in horizontal lanes (one \
per modelling phase, e.g. Data prep, Feature engineering, Estimation, \
Validation). You work only through the tools you are given.

The user has prepared the data: the **anchor** blocks are where you build \
from. Ground rules, enforced by the tools:
- You see schemas and summary statistics, never data rows. Don't ask for rows. \
The exception is statistics tables -- outputs with one row per feature, bin, \
grade, period or sample, such as fit_binning's summary and bins or \
compare_samples' table (describe_block_type marks them statistics_table): \
get_output_summary returns their rows. Read the numbers there.
- Only quote numbers you have seen in a tool result. Never estimate, \
interpolate or fill in a value you haven't seen; say what you'd need to \
check instead.
- The user's blocks are theirs. You can read them and wire *from* them. You \
can only change blocks you created in this build, plus changes the user \
approved in the plan (changes_to_existing).
- Column roles matter: `target` is the modelling target (params named \
target/target_col auto-fill from it); `predicted` is set automatically on \
model outputs (score_col/predicted_col auto-fill from it); columns tagged \
`excluded` must never be used as features; `id` columns are identifiers, not \
features. Don't retag the user's columns.
{custom_rule}Registry block params are documented by describe_block_type -- read it \
before using a block you haven't used yet in this build.
- Keep the model proportionate to the goal. Don't add blocks the goal \
doesn't call for.

Modelling practice the registry is built around:
- Fit anything learned from data on the development/train sample only and \
apply the fitted artifact elsewhere: fit_binning on train, then \
apply_binning on train, test and out-of-time (never woe_transform on a test \
sample -- it re-derives WoE from that sample's own target); predict with the \
train model; calibrate_model on development data.
- fit_binning is also the univariate analysis (bin table, IV, \
direction-adjusted Gini/KS or rank correlation for LGD/CCF, monotonicity); \
pair it with characteristic_stability (drift per feature between samples) \
and target_trend (target over time) rather than one block per feature. A \
feature in iv_band "suspicious" (IV of 0.5 or more) usually means leakage -- \
the column is partly the outcome. Don't use it as a driver without asking \
the user (ask_user) first, and name it in your summary either way.
- Use time_split for an out-of-time sample, train_test_split with \
stratify_col for a low default rate, derive_columns for ratios/flags and \
one_hot_encode for categorical regressors rather than custom code.
- For LGD: discount_recoveries -> compute_lgd (tags lgd as the target); for \
CCF: compute_ccf -> lgd_regression -> compute_ead. Report compare_samples \
(train/test/OOT side by side) and, for a rating scale, grade_backtest."""

CUSTOM_CONTRACT = """Custom block contract (add_custom_block / update_custom_block):
- Exactly one top-level Python function. Its first parameters are its \
dataframe inputs (named as in `inputs`, default ['df']), typed pl.DataFrame; \
any further parameters are configurable values with defaults. It returns one \
pl.DataFrame.
- Use only polars (in scope as `pl`). No pandas, no I/O, no globals, no network.
- Never hardcode an existing input column's name in the body: take it as a \
`<descriptive>_col` str parameter defaulted to the real column name (or \
`<descriptive>_cols` for a list) and use pl.col(param). New columns you \
create get literal names via .alias(...).
- metadata_transform describes the column set change: {"kind": "passthrough"} \
(same columns), {"kind": "narrow"} (a subset), or {"kind": "declared", "base": \
"<input port>", "drops": [...], "adds": [{"name", "dtype" (e.g. Float64, \
Int64, Boolean, String, Date), "role"}]}."""

PLAN_INSTRUCTIONS = """## Your job now: plan

Explore with the read-only tools (get_graph, get_output_summary on the \
anchors, list_block_types, describe_block_type), then call submit_plan once \
with a complete plan. Nothing on the canvas changes while you plan; the user \
will see your plan as ghost blocks and approve it or send feedback.

A good plan:
- Restates the goal in `summary`, lists real `assumptions` (e.g. which column \
is the default flag and why), and puts anything the user must decide in \
`questions` (leave it empty if nothing is unclear -- don't ask for the sake of it).
- Is an outline of `stages` in build order (e.g. Data prep, Feature \
engineering, Estimation, Validation) -- each becomes one lane. Reuse an \
existing lane (its id in `lane`) where it fits. Keep to the stages the goal \
needs.
- Gives each stage a `goal`: what it does and produces, and the decisions \
that shape it -- the split design, the model family, the metrics and \
samples to report. These are what the user is approving. List the registry \
blocks the stage will most likely use in its `blocks`: the build is handed \
their docs when the stage starts.
- Splits the sample (train/test, out-of-time) BEFORE any stage that learns \
from the target -- univariate analysis and binning, feature selection, \
estimation, calibration -- so each is fitted on the development sample only \
and the holdout samples stay unseen. The app rejects applying a fit to a \
holdout sample it was fitted on.
- Does NOT list individual blocks, feature lists or bin settings: those \
depend on results nobody has seen yet (e.g. which features survive the \
univariate analysis), so the build plans each stage's blocks once the \
earlier stages have run. Check the registry has what the stages need.
- Lists in changes_to_existing any change to a block that already exists, \
including wiring into one of its inputs.

After submit_plan succeeds, end your turn with a one-line summary."""

BUILD_INSTRUCTIONS = """## Your job now: build the approved outline, one stage at a time

For the current stage:
1. Look at what you're building on (get_output_summary on the earlier \
stages' outputs; describe_block_type for a block's params) and call \
plan_stage with this stage's steps: category, inputs wired from this \
stage's step refs or existing block ids (blocks built in earlier stages \
included), key params (feature lists, bins, split settings) chosen from the \
results so far, and a short `why` each. Step refs are unique across the \
build: prefix them with the stage key (e.g. est1, est2).
2. plan_stage builds the plan for you: it adds each step, wires it and runs \
it, in order, and returns each step's status and outputs. Check them -- the \
columns and row counts, a metric in a sane range; get_output_summary for \
the statistics.
3. If a step failed, read its error, fix it (set_params{update_tool}, or \
plan_stage again with changed steps) and call build_stage to build the rest. \
After 3 failures of one block, ask_user.{custom_steps}
4. When the stage is built and green, call complete_stage with a short \
Markdown summary: what you built, the headline numbers, and the decisions \
you made from them. Then end your turn: the user reviews the stage, and you \
are resumed with the next one (or with their changes to this one).

Every tool call is a round trip that re-sends this whole conversation, so \
don't spend calls on things you already know: plan a stage in one \
plan_stage call, and skip get_graph unless you've lost track of block ids.

Small deviations from the stage plan are fine -- add_block (with \
plan_step), connect and run_to are there for them -- record each with \
note_deviation. Structural changes to the approved outline (dropping a stage, \
a different model family) and any change to a user block the outline didn't \
list need ask_user.

The build is running on {sample_note}. On the last stage, instead of \
complete_stage, call finish with a Markdown report (what you built across \
all stages, deviations, headline results, concerns -- e.g. weak features, \
instability, anything you'd check next) and key_outputs pointing at the \
headline metric blocks. The app then runs the whole graph on the full data \
and refreshes those metrics. End your turn after finish."""


CUSTOM_ON = """- Prefer registry blocks. Only write a custom block when no registry block \
does the job (e.g. a multi-column transformation, an out-of-time split by \
date). """

CUSTOM_STEPS = """
A custom step stops the build there: write it with add_custom_block \
(plan_step set to its ref, lane the stage key), connect its inputs, then \
call build_stage."""

CUSTOM_OFF = """- Custom blocks are turned off for this build: use registry blocks only. \
If the goal needs something no registry block does, say so -- in \
`questions` while planning, with ask_user while building. """


def _common(build: AgentBuild) -> str:
    return COMMON.replace("{custom_rule}", CUSTOM_ON if build.options.allow_custom_blocks else CUSTOM_OFF)


def _custom_contract(build: AgentBuild) -> list[str]:
    return [CUSTOM_CONTRACT] if build.options.allow_custom_blocks else []


def _anchor_context(build: AgentBuild) -> str:
    graph = build.session.graph
    lines = []
    for bid in build.anchors:
        block = graph.blocks.get(bid)
        if block is None:
            continue
        ports = ", ".join(f"{p.name}:{p.type}" for p in block.outputs)
        lines.append(f"- {block.name} (id {bid}, {block.category}) -> {ports}")
    return "\n".join(lines) or "- (none)"


def _catalogue_context(compact: bool = False) -> str:
    """The registry blocks for the system prompt. Full: one line per block
    with tags, ports and summary (~8k tokens). Compact, for small-context
    local models (see loop.LMStudioLoop): just the tags (~400 tokens) --
    the model pulls a tag's blocks with list_block_types(tag=...), and one
    block's ports and params with describe_block_type, only for what the
    goal needs."""
    if compact:
        # Tags with their blocks' names: names only, so a step's category
        # is never a guess (or a tag name) -- describe_block_type has the rest.
        lines = [
            f"- {t['tag']}: {t['about']}\n  blocks: {', '.join(catalogue.tag_block_names(t['tag']))}"
            for t in catalogue.list_tags()
        ]
        return (
            "## Registry blocks, by tag\n"
            "A step's category is one of these block names (never a tag). list_block_types(tag=...) gives each "
            "block's one-line summary, and describe_block_type its ports and params -- read it before using a "
            "block whose ports you haven't seen.\n" + "\n".join(lines)
        )
    lines = []
    for e in catalogue.list_block_types():
        ins = ", ".join(f"{p['name']}:{p['type']}" for p in e["inputs"]) or "-"
        outs = ", ".join(f"{p['name']}:{p['type']}" for p in e["outputs"]) or "-"
        lines.append(f"- {e['category']} [{', '.join(e['tags'])}] ({ins}) -> ({outs}): {e['summary']}")
    return "## Registry blocks\n" + "\n".join(lines)


def _preflight_context(build: AgentBuild) -> str:
    warnings = build.preflight.get("warnings") or []
    if not warnings:
        return ""
    body = "\n".join(f"- {w['message']}" for w in warnings)
    return f"\n\nPreflight warnings the user chose to proceed past:\n{body}"


def plan_system(build: AgentBuild, compact: bool = False) -> str:
    return "\n\n".join([_common(build), *_custom_contract(build), PLAN_INSTRUCTIONS, _catalogue_context(compact)])


def plan_prompt(build: AgentBuild) -> str:
    return (
        f"Goal: {build.goal}\n\nAnchor blocks to build from:\n{_anchor_context(build)}"
        f"{_preflight_context(build)}\n\nStart by looking at the graph and the anchors' outputs."
    )


def plan_feedback_prompt(feedback: str) -> str:
    return (
        "The user reviewed your plan and wants changes:\n\n"
        f"{feedback}\n\nRevise the plan accordingly and call submit_plan again."
    )


def build_system(build: AgentBuild, compact: bool = False) -> str:
    """The build always gets the compact catalogue (tags only): it's
    re-sent on every one of the build's many round trips, and the build
    looks blocks up as it plans each stage anyway. `compact` is kept for
    the loops' sake (see plan_system, where the full list helps the
    outline)."""
    if build.sample_rows_used:
        sample_note = f"a {build.sample_rows_used:,}-row sample of the data"
    else:
        sample_note = "the full data"
    return "\n\n".join(
        [
            _common(build),
            *_custom_contract(build),
            BUILD_INSTRUCTIONS.format(
                sample_note=sample_note,
                update_tool=" / update_custom_block" if build.options.allow_custom_blocks else "",
                custom_steps=CUSTOM_STEPS if build.options.allow_custom_blocks else "",
            ),
            _catalogue_context(compact=True),
        ]
    )


def _plan_for_prompt(plan: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in plan.items() if k not in ("layout", "lane_layout")}


def build_prompt(build: AgentBuild) -> str:
    lane_map = build.lane_map
    lanes = "\n".join(f"- stage {k!r} -> lane id {v}" for k, v in lane_map.items()) or "- (no new lanes)"
    feedback = [h["feedback"] for h in build.plan_history if h.get("feedback")]
    fb = ("\n\nFeedback the user gave while planning:\n" + "\n".join(f"- {f}" for f in feedback)) if feedback else ""
    return (
        f"Goal: {build.goal}\n\nAnchor blocks:\n{_anchor_context(build)}{_preflight_context(build)}{fb}\n\n"
        f"The user approved this outline:\n```json\n{json.dumps(_plan_for_prompt(build.plan or {}), indent=1)}\n```\n\n"
        f"Each stage's lane:\n{lanes}\n\n{stage_prompt(build)}"
    )


# What a stage's opening prompt carries about the graph so far (see
# stage_prompt) -- enough that the model doesn't spend its first calls
# fetching it, within a budget that keeps the prompt from bloating.
STAGE_TABLE_CHARS = 6_000  # a statistics table's rows, inline, at most (fits a ~20-feature binning summary)
STAGE_CONTEXT_CHARS = 16_000  # the whole "built so far" section
STAGE_BLOCK_DOCS = 8  # blocks documented up front


def _output_line(build: AgentBuild, block_id: str, port: str, seen: dict[tuple, str]) -> str:
    """One output port, briefly. `seen` maps a column list already shown
    to where, so a split's second sample (or a filter's output) reads
    "same columns as ..." instead of repeating thirty names."""
    try:
        _, value = build.current_output(block_id, port)
    except ToolError:
        return f"  {port}: no output yet"
    if isinstance(value, DataFramePacket):
        if is_statistics_table(build, block_id, port):
            table = table_rows(value)
            rows = json.dumps(table["rows"], default=str)
            if len(rows) <= STAGE_TABLE_CHARS and "rows_shown" not in table:
                return f"  {port}: statistics table, {table['row_count']} rows: {rows}"
            return f"  {port}: statistics table, {table['row_count']} rows -- get_output_summary for them"
        cols = []
        for name, meta in value.schema_meta.items():
            role = meta.role.value if hasattr(meta.role, "value") else str(meta.role)
            cols.append(name + (f" [{role}]" if role not in (ColumnRole.UNASSIGNED.value, ColumnRole.FEATURE.value) else ""))
        key = tuple(cols)
        if key in seen:
            return f"  {port}: {value.data.height:,} rows; same columns as {seen[key]}"
        seen[key] = f"{build.session.graph.blocks[block_id].name}.{port}"
        return f"  {port}: {value.data.height:,} rows; columns: {', '.join(cols)}"
    if isinstance(value, (bytes, bytearray)):
        return f"  {port}: image"
    # Metrics and model summaries are worth reading here; a fitted
    # artifact (binning, distribution, ...) is bulky and is used by wiring it.
    port_type = next((p.type for p in build.session.graph.blocks[block_id].outputs if p.name == port), None)
    limit = 600 if port_type in ("scalar_metric", "model") else 160
    text = json.dumps(value, default=str)
    return f"  {port} ({port_type}): {text if len(text) <= limit else text[:limit] + '... (get_output_summary for all)'}"


def built_so_far(build: AgentBuild) -> str:
    """The anchors and every block this build has made, in graph order,
    with their outputs: dataframes as row count + columns (and roles),
    small statistics tables in full, metrics and models by value."""
    graph = build.session.graph
    lines: list[str] = []
    seen: dict[tuple, str] = {}
    for bid in graph.topo_order():
        if bid not in build.anchors and bid not in build.owned_blocks:
            continue
        block = graph.blocks[bid]
        prov = block.provenance or {}
        tags = [block.category] + (["anchor"] if bid in build.anchors else [])
        if prov.get("plan_step"):
            tags.append(f"step {prov['plan_step']}")
        lines.append(f"- {block.name} (id {bid}; {', '.join(tags)}; {build.session.runner.status(bid)})")
        lines += [_output_line(build, bid, p.name, seen) for p in block.outputs]
    text = "\n".join(lines)
    if len(text) > STAGE_CONTEXT_CHARS:
        text = text[:STAGE_CONTEXT_CHARS] + "\n... (cut short -- get_graph / get_output_summary for the rest)"
    return text


def _stage_block_names(stage: dict[str, Any]) -> list[str]:
    """The blocks the outline listed for the stage, then any registry
    category its goal names."""
    catalogue.ensure_blocks_registered()
    names = [c for c in stage.get("blocks") or [] if c in BLOCK_REGISTRY]
    for category in BLOCK_REGISTRY:
        if category not in names and re.search(rf"\b{re.escape(category)}\b", stage.get("goal") or ""):
            names.append(category)
    return [c for c in names if c not in catalogue.AGENT_DISALLOWED][:STAGE_BLOCK_DOCS]


def block_docs(categories: list[str]) -> str:
    """describe_block_type, compacted: summary, ports, params -- what
    plan_stage needs to name ports and params right."""
    out = []
    for category in categories:
        d = catalogue.describe_block_type(category)
        ins = ", ".join(f"{p['name']}:{p['type']}" for p in d["inputs"]) or "-"
        outs = ", ".join(f"{p['name']}:{p['type']}" + (" (statistics table)" if p.get("statistics_table") else "") for p in d["outputs"])
        params = []
        for p in d.get("params") or []:
            item = f"{p['name']}: {p.get('type') or 'any'}"
            if p.get("required"):
                item += " (required)"
            elif "default" in p:
                item += f" = {json.dumps(p['default'])}"
            if p.get("auto_fills_from_role"):
                item += f" (auto-fills from the {p['auto_fills_from_role']} role)"
            params.append(item)
        out.append(f"### {category}\n{d['summary']}\nin: {ins} -> out: {outs}\nparams: {'; '.join(params) or '-'}")
    return "\n\n".join(out)


DECISION_SUMMARY_CHARS = 1_200  # each earlier stage's summary, at most


def decisions_so_far(build: AgentBuild) -> str:
    """What the earlier stages decided, for a stage that starts in a fresh
    conversation (BuildOptions.small_context): their summaries, the user's
    answers and feedback, deviations and concerns. Empty on the first
    stage."""
    lines: list[str] = []
    for st in build.stages[: build.stage_index]:
        if st.get("summary"):
            text = st["summary"].strip()
            if len(text) > DECISION_SUMMARY_CHARS:
                text = text[:DECISION_SUMMARY_CHARS].rstrip() + " ..."
            lines.append(f"### Stage {st['name']!r} ({st['status']})\n{text}")
    said = []
    for u in build.user_inputs:
        where = f" (stage {u['stage']})" if u.get("stage") else ""
        if u.get("question"):
            said.append(f"- You asked{where}: {u['question'].strip()}\n  The user answered: {u['text'].strip()}")
        else:
            said.append(f"- The user said{where}: {u['text'].strip()}")
    if said:
        lines.append("### The user's answers and feedback\n" + "\n".join(said))
    if build.deviations:
        lines.append(
            "### Deviations recorded\n"
            + "\n".join(f"- {d['what']}" + (f" -- {d['why']}" if d.get("why") else "") for d in build.deviations)
        )
    if build.concerns:
        lines.append("### Concerns flagged\n" + "\n".join(f"- {c['what']}" for c in build.concerns))
    return "\n\n".join(lines)


def stage_prompt(build: AgentBuild) -> str:
    stage = build.current_stage
    n = len(build.stages)
    then = (
        "When it's built and checked, call finish -- this is the last stage."
        if build.is_last_stage()
        else "When it's built and checked, call complete_stage and end your turn."
    )
    parts = [
        f"Stage {build.stage_index + 1} of {n}: {stage['name']} (key {stage['key']!r}).\nGoal: {stage['goal']}",
        f"Built so far (current outputs):\n{built_so_far(build)}",
    ]
    if build.options.small_context:
        decided = decisions_so_far(build)
        if decided:
            parts.append(
                "This stage starts a fresh conversation. What the earlier stages decided -- keep to it:\n\n" + decided
            )
    docs = block_docs(_stage_block_names(stage))
    if docs:
        parts.append(
            "Docs for the blocks this stage is likely to use (describe_block_type has the full text, and "
            f"list_block_types the rest of the registry):\n\n{docs}"
        )
    parts.append(
        "You don't need get_graph or get_output_summary for anything listed above. Plan the stage with "
        f"plan_stage, which builds it, and check the results. {then}"
    )
    return "\n\n".join(parts)


def stage_feedback_prompt(build: AgentBuild, feedback: str) -> str:
    stage = build.current_stage
    return (
        f"The user reviewed stage {stage['name']!r} and wants changes:\n\n{feedback}\n\n"
        "Make them (call plan_stage again if the steps change), check the results, then call complete_stage "
        "again with an updated summary and end your turn."
    )


def answer_prompt(answer: str) -> str:
    return f"The user answered:\n\n{answer}\n\nContinue the build."
