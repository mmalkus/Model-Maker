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
from typing import Any

from . import catalogue
from .build import AgentBuild

COMMON = """You are building a model inside Model-Maker, a visual, Polars-based \
modelling tool used by credit-risk and actuarial model developers. A model \
is a graph of blocks connected by wires, arranged in horizontal lanes (one \
per modelling phase, e.g. Data prep, Feature engineering, Estimation, \
Validation). You work only through the tools you are given.

The user has prepared the data: the **anchor** blocks are where you build \
from. Ground rules, enforced by the tools:
- You see schemas and summary statistics, never data rows. Don't ask for rows.
- The user's blocks are theirs. You can read them and wire *from* them. You \
can only change blocks you created in this build, plus changes the user \
approved in the plan (changes_to_existing).
- Column roles matter: `target` is the modelling target (params named \
target/target_col auto-fill from it); `predicted` is set automatically on \
model outputs (score_col/predicted_col auto-fill from it); columns tagged \
`excluded` must never be used as features; `id` columns are identifiers, not \
features. Don't retag the user's columns.
- Prefer registry blocks. Only write a custom block when no registry block \
does the job (e.g. a multi-column transformation, an out-of-time split by \
date). Registry block params are documented by describe_block_type -- read it \
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
and target_trend (target over time) rather than one block per feature.
- Use time_split for an out-of-time sample, train_test_split with \
stratify_col for a low default rate, derive_columns for ratios/flags and \
one_hot_encode for categorical regressors before writing custom code.
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
samples to report. These are what the user is approving.
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
stages' outputs) and call plan_stage with this stage's steps: category, \
inputs wired from this stage's step refs or existing block ids (blocks built \
in earlier stages included), key params (feature lists, bins, split \
settings) chosen from the results so far, and a short `why` each.
2. Then build the steps in order:
   a. add_block (or add_custom_block) with `plan_step` set to the step's ref \
and the stage key as `lane`.
   b. connect its inputs.
   c. run_to it and check the result: the status, the columns and row count, \
a metric in a sane range. Use get_output_summary when you need the statistics.
   d. If it fails, read the error, fix it (set_params / update_custom_block) \
and run again. After 3 failures of one block, ask_user.
3. When the stage is built and green, call complete_stage with a short \
Markdown summary: what you built, the headline numbers, and the decisions \
you made from them. Then end your turn: the user reviews the stage, and you \
are resumed with the next one (or with their changes to this one).

Small deviations from the stage plan are fine -- record each with \
note_deviation. Structural changes to the approved outline (dropping a stage, \
a different model family) and any change to a user block the outline didn't \
list need ask_user.

The build is running on {sample_note}. On the last stage, instead of \
complete_stage, call finish with a Markdown report (what you built across \
all stages, deviations, headline results, concerns -- e.g. weak features, \
instability, anything you'd check next) and key_outputs pointing at the \
headline metric blocks. The app then runs the whole graph on the full data \
and refreshes those metrics. End your turn after finish."""


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
        lines = [f"- {t['tag']} ({t['blocks']}): {t['about']}" for t in catalogue.list_tags()]
        return (
            "## Registry block tags\n"
            "Call list_block_types with tag=<tag> to see that tag's blocks, and "
            "describe_block_type for a block's ports and params, before using it.\n" + "\n".join(lines)
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
    return "\n\n".join([COMMON, CUSTOM_CONTRACT, PLAN_INSTRUCTIONS, _catalogue_context(compact)])


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
    if build.sample_rows_used:
        sample_note = f"a {build.sample_rows_used:,}-row sample of the data"
    else:
        sample_note = "the full data"
    return "\n\n".join(
        [
            COMMON,
            CUSTOM_CONTRACT,
            BUILD_INSTRUCTIONS.format(sample_note=sample_note),
            _catalogue_context(compact),
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


def stage_prompt(build: AgentBuild) -> str:
    stage = build.current_stage
    n = len(build.stages)
    then = (
        "When it's built and checked, call finish -- this is the last stage."
        if build.is_last_stage()
        else "When it's built and checked, call complete_stage and end your turn."
    )
    return (
        f"Stage {build.stage_index + 1} of {n}: {stage['name']} (key {stage['key']!r}).\n"
        f"Goal: {stage['goal']}\n\nPlan it with plan_stage, then build it. {then}"
    )


def stage_feedback_prompt(build: AgentBuild, feedback: str) -> str:
    stage = build.current_stage
    return (
        f"The user reviewed stage {stage['name']!r} and wants changes:\n\n{feedback}\n\n"
        "Make them (call plan_stage again if the steps change), check the results, then call complete_stage "
        "again with an updated summary and end your turn."
    )


def answer_prompt(answer: str) -> str:
    return f"The user answered:\n\n{answer}\n\nContinue the build."
