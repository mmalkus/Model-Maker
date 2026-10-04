from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Any, Callable, Literal

PortType = Literal[
    "dataframe",
    "model",
    "scalar_metric",
    "image",
    "master_scale",
    # A fitted binning (see blocks/binning.py fit_binning): bins + WoE per
    # feature, applied unchanged to other samples by apply_binning.
    "binning",
    "any",
    # Stochastic engine port types (see /stochastic-engine-proposal.md S2) --
    # each is a plain JSON-shaped dict, same "no special deserializer"
    # convention as "model" above, not a custom packet class.
    "distribution",
    "dependency",
    "proxy_function",
    "simulation_result",
]
# Pipeline role only -- governs runtime behavior (input blocks need no
# upstream and support refresh/probe; output blocks may take an injected
# output_dir). Orthogonal to where a block's code comes from: see
# BlockInstance.is_custom in graph.py for that axis. Any of the three can be
# either a registry (fixed-code) block or a custom (AI-authored) one.
BlockType = Literal["input", "standard", "output"]

# (input_metas_by_port, output_dataframes_by_port, params) -> output_metas_by_port
MetadataTransformFn = Callable[
    [dict[str, dict[str, Any]], dict[str, Any], dict[str, Any]],
    dict[str, dict[str, Any]],
]

# A block's core function is generic and plain-dataframe: it never touches
# DataFramePacket/ColumnMeta. The engine unwraps packets before calling it
# and rewraps the result afterward, applying the metadata_transform. This
# is what keeps the compiled --with-metadata=off script "clean" by
# construction (see plan section 7).
BlockFn = Callable[..., Any]


FieldKind = Literal["text", "number", "select", "column", "columns", "checkbox"]


@dataclass(frozen=True)
class FieldSpec:
    """One field of a block's params form in the web UI and the TUI (see
    forms.block_form). A block lists only the fields worth a better label,
    a placeholder or a fixed set of options; every other param gets a field
    inferred from the function's signature."""

    key: str
    label: str
    kind: FieldKind
    placeholder: str = ""
    step: float | None = None
    options: tuple[str, ...] = ()
    # 'target' picks up the role=target column; 'predicted' picks up
    # whichever column a modelling block upstream tagged role=predicted --
    # see packet.resolve_role_column / blocks/modelling.py. Only meaningful
    # for kind='column'; leaving the param out of `params` entirely puts it
    # back in this dynamically-resolved "Auto" state.
    auto_role: Literal["target", "predicted"] | None = None


# (outputs_by_port) -> short "what this result means" notes for an AI build
# (see agent/hints.decision_hints).
HintsFn = Callable[[dict[str, Any]], list[str]]


@dataclass
class PortSpec:
    name: str
    type: PortType = "dataframe"
    required: bool = True


@dataclass
class BlockSpec:
    category: str
    block_type: BlockType
    display_name: str
    inputs: list[PortSpec]
    outputs: list[PortSpec]
    fn: BlockFn
    metadata_transform: MetadataTransformFn
    probe: Callable[[dict[str, Any]], Any] | None = None
    # Palette display grouping only -- purely cosmetic, unrelated to
    # block_type's runtime role. Defaults to block_type so existing
    # input/standard/output blocks group as before; a block library can set
    # this to cluster related blocks (e.g. "modelling", "tests") regardless
    # of what pipeline role each one plays.
    group: str | None = None
    # What the block does and which risk model it serves (e.g. "regression",
    # "pd") -- an AI build browses the catalogue by tag (see
    # agent/catalogue.py's TAGS, which every tag here must appear in).
    tags: tuple[str, ...] = ()
    # Optional lazy-mode twin of `fn`: same params, but reads/returns
    # pl.LazyFrame instead of pl.DataFrame. None (the default, and every
    # custom/AI-authored block) means this block never participates in a
    # streaming run's fusion -- see Runner._fusable_spec/_build_fusion_groups
    # in runner.py. For most registry blocks that are already pure
    # expression-based code (filter/select/groupby_agg/join), `fn` itself
    # works unchanged over a LazyFrame and can be reused verbatim here; only
    # a block whose eager `fn` deliberately collects (read_csv) needs an
    # actual separate implementation.
    lazy_fn: BlockFn | None = None
    # Optional streaming-sink twin of `fn`: takes a still-uncollected
    # pl.LazyFrame plus the block's other params and writes it straight to
    # its destination (e.g. LazyFrame.sink_csv), returning None -- never
    # called through the normal eager path, only when this block is folded
    # onto the end of a streaming run's fusion group as a terminal write
    # (see Runner._build_fusion_groups' sink-attachment pass). None (the
    # default) means this block is never eligible for that: it always runs
    # as an ordinary checkpoint reading a materialized DataFrame from cache,
    # exactly as before.
    lazy_sink_fn: BlockFn | None = None
    # Output ports that are statistics tables: one row per feature, bin,
    # grade, period, sample or quantile -- never one per record. An AI
    # build may read these tables' rows (agent/tools.summarize_value);
    # every other dataframe reaches it as schema + summary statistics only.
    # Declared here, on the registry spec, so a block instance (or custom
    # code) can never claim it for itself.
    aggregate_outputs: tuple[str, ...] = ()
    # Params form fields to show ahead of the inferred ones (see FieldSpec).
    # Lives here, not in `fn`, so it never reaches a compiled script.
    form: tuple[FieldSpec, ...] = ()
    # Decision hints for an AI build once this block has run green (see
    # agent/hints.py for the thresholds they share). None: nothing to judge.
    hints: HintsFn | None = None


BLOCK_REGISTRY: dict[str, BlockSpec] = {}


def register_block(spec: BlockSpec) -> BlockSpec:
    frames = {p.name for p in spec.outputs if p.type == "dataframe"}
    unknown = set(spec.aggregate_outputs) - frames
    if unknown:
        raise ValueError(f"{spec.category}: aggregate_outputs {sorted(unknown)} aren't dataframe outputs")
    params = inspect.signature(spec.fn).parameters
    unknown = [f.key for f in spec.form if f.key not in params]
    if unknown:
        raise ValueError(f"{spec.category}: form fields {unknown} aren't params of {spec.fn.__name__}")
    BLOCK_REGISTRY[spec.category] = spec
    return spec
