"""System prompts and opening messages for an AI build's two phases. The
build phase starts a fresh conversation seeded with the approved plan --
not the planning transcript -- so the plan is the contract and the plan
and build LLMs can be different providers (see /agent-builder-proposal.md
§4.3, §9.1)."""

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
- Lays steps out in lanes; reuse existing lanes by id where they fit, and \
create new ones (in `lanes`) only when needed.
- Wires every step's inputs from an earlier step ref or an existing block id, \
with real port names.
- Gives key params (feature lists, bins, split settings) with a short `why` per step.
- Lists in changes_to_existing any change to a block that already exists, \
including wiring into one of its inputs.

After submit_plan succeeds, end your turn with a one-line summary."""

BUILD_INSTRUCTIONS = """## Your job now: build the approved plan

Work step by step, in plan order:
1. add_block (or add_custom_block) with `plan_step` set to the step's ref and \
the step's lane key/id as `lane`.
2. connect its inputs.
3. run_to it and check the result: the status, the columns and row count, a \
metric in a sane range. Use get_output_summary when you need the statistics.
4. If it fails, read the error, fix it (set_params / update_custom_block) and \
run again. After 3 failures of one block, ask_user.

Small deviations from the plan are fine -- record each with note_deviation. \
Structural changes (dropping a lane, a different model family) and any change \
to a user block the plan didn't list need ask_user.

The build is running on {sample_note}. When everything is built and green, \
call finish with a Markdown report (what you built, deviations, headline \
results, concerns -- e.g. weak features, instability, anything you'd check \
next) and key_outputs pointing at the headline metric blocks. The app then \
runs the whole graph on the full data and refreshes those metrics. End your \
turn after finish."""


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
    with ports and summary (~5k tokens). Compact, for small-context local
    models (see loop.LMStudioLoop): just the block names per category
    (~200 tokens) -- the model pulls a category's ports and summaries with
    list_block_types(group=...), or one block's params with
    describe_block_type, only for what the goal needs."""
    entries = catalogue.list_block_types()
    if compact:
        groups: dict[str, list[str]] = {}
        for e in entries:
            groups.setdefault(e["group"], []).append(e["category"])
        lines = [f"- {g}: {', '.join(cats)}" for g, cats in groups.items()]
        return (
            "## Registry blocks by category\n"
            "Call list_block_types with group=<category> for ports and summaries, and "
            "describe_block_type for a block's params, before using it.\n" + "\n".join(lines)
        )
    lines = []
    for e in entries:
        ins = ", ".join(f"{p['name']}:{p['type']}" for p in e["inputs"]) or "-"
        outs = ", ".join(f"{p['name']}:{p['type']}" for p in e["outputs"]) or "-"
        lines.append(f"- {e['category']} [{e['group']}] ({ins}) -> ({outs}): {e['summary']}")
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
    lanes = "\n".join(f"- plan lane {k!r} -> lane id {v}" for k, v in lane_map.items()) or "- (no new lanes)"
    feedback = [h["feedback"] for h in build.plan_history if h.get("feedback")]
    fb = ("\n\nFeedback the user gave while planning:\n" + "\n".join(f"- {f}" for f in feedback)) if feedback else ""
    return (
        f"Goal: {build.goal}\n\nAnchor blocks:\n{_anchor_context(build)}{_preflight_context(build)}{fb}\n\n"
        f"The user approved this plan:\n```json\n{json.dumps(_plan_for_prompt(build.plan or {}), indent=1)}\n```\n\n"
        f"New lanes have been created:\n{lanes}\n\nBuild it now."
    )


def answer_prompt(answer: str) -> str:
    return f"The user answered:\n\n{answer}\n\nContinue the build."
