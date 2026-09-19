from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Literal

PortType = Literal[
    "dataframe",
    "model",
    "scalar_metric",
    "image",
    "master_scale",
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


BLOCK_REGISTRY: dict[str, BlockSpec] = {}


def register_block(spec: BlockSpec) -> BlockSpec:
    BLOCK_REGISTRY[spec.category] = spec
    return spec
