"""The types every block resolves to, whatever file it came from.

A block is described by an entry in its file's `BLOCKS = {...}` literal plus
a plain function of the same name (see modelmaker/block_library.py, which
parses both, and docs/block-authoring.md for the format). This module holds
only the in-memory shape of that description -- `BlockDefinition` -- and the
registry of built-in blocks. Everything a block *doesn't* describe
(sampling, change detection, fusion, hints, metadata propagation) is
architecture applied to every definition alike: see runner.py, sources.py,
metadata_transforms.py and agent/hint_rules.py.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Literal

if TYPE_CHECKING:
    from ..signature import BlockSignature

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
PORT_TYPES: frozenset[str] = frozenset(PortType.__args__)  # type: ignore[attr-defined]

# Pipeline role only -- governs runtime behavior (input blocks need no
# upstream and support refresh/change detection; output blocks may take an
# injected output_dir).
BlockType = Literal["input", "standard", "output"]
BLOCK_TYPES: frozenset[str] = frozenset(BlockType.__args__)  # type: ignore[attr-defined]

# How a display block shows its result right on its canvas node, with no
# click (see api._display_out): `table` the first rows of its dataframe
# output plus a row count, `kpi` one headline number with a label (its
# scalar_metric output is {"label", "value", ...}), `value` whatever comes
# through -- a table, a dict of numbers, or an image.
DisplayKind = Literal["table", "kpi", "value"]
DISPLAY_KINDS: frozenset[str] = frozenset(DisplayKind.__args__)  # type: ignore[attr-defined]

# A block's core function is generic and plain-dataframe: it never touches
# DataFramePacket/ColumnMeta. The engine unwraps packets before calling it
# and rewraps the result afterward, applying the block's declared metadata
# rules (see metadata_transforms.schema_diff). This is what keeps the
# compiled script "clean" by construction.
BlockFn = Callable[..., Any]

# `path` (a file the block reads -- its mtime/size is the block's change
# fingerprint, see sources.py), `sql` (a query text) and `env` (the *name* of
# an environment variable) render like `text` with extra help in the UIs.
# `dataset` (the name of one of the workspace's datasets) and `source` (the
# name of an administrator-defined data source) render as pickers over what
# the workspace offers (GET /api/datasets, /api/sources); the engine resolves
# them to a read-only file before the block runs (see datasets.py). `path`
# and `env` need the workspace's `filesystem` capability, `dataset` its
# `datasets` and `source` its `sources` (see capabilities.py).
FieldKind = Literal[
    "text", "number", "select", "column", "columns", "checkbox", "path", "sql", "env", "dataset", "source"
]
FIELD_KINDS: frozenset[str] = frozenset(FieldKind.__args__)  # type: ignore[attr-defined]


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
    # A select's choices; for a `path` field, the accepted file extensions.
    options: tuple[str, ...] = ()
    # 'target' picks up the role=target column; 'predicted' picks up
    # whichever column a modelling block upstream tagged role=predicted --
    # see packet.resolve_role_column. Only meaningful for kind='column';
    # leaving the param out of `params` entirely puts it back in this
    # dynamically-resolved "Auto" state.
    auto_role: Literal["target", "predicted"] | None = None


@dataclass
class PortSpec:
    name: str
    type: PortType = "dataframe"
    required: bool = True
    # Input port: the function also works when handed a pl.LazyFrame here,
    # so a streaming run may fuse it (see runner._fusable_definition).
    # Output port: the function returns a pl.LazyFrame here; the runner
    # collects it (pushing sample mode's .head(n) down first).
    lazy: bool = False
    # Output port: a statistics table -- one row per feature, bin, grade,
    # period, sample or quantile, never one per record. An AI build may read
    # such a table's rows (agent/tools.summarize_value); every other
    # dataframe reaches it as schema + summary statistics only.
    aggregate: bool = False


def _port(raw: dict[str, Any]) -> PortSpec:
    return PortSpec(
        name=raw["name"],
        type=raw.get("type", "dataframe"),
        required=raw.get("required", True),
        lazy=bool(raw.get("lazy", False)),
        aggregate=bool(raw.get("aggregate", False)),
    )


def _field(raw: dict[str, Any]) -> FieldSpec:
    return FieldSpec(
        key=raw["key"],
        label=raw.get("label") or raw["key"],
        kind=raw.get("kind", "text"),
        placeholder=raw.get("placeholder", ""),
        step=raw.get("step"),
        options=tuple(raw.get("options", ())),
        auto_role=raw.get("auto_role"),
    )


@dataclass
class BlockDefinition:
    """One block, as every part of the engine sees it. Built-ins and library
    blocks are the same type; only `source` says where it came from.

    `meta` is the block's `BLOCKS` entry verbatim (it is what a project
    embeds and what a library file stores); the typed attributes below are
    derived from it once, at construction."""

    id: str
    meta: dict[str, Any]
    # The block function's source plus every top-level helper it references
    # (transitively) -- exactly what a compiled script emits for it.
    code: str
    source: Literal["builtin", "library"]
    origin: str | None
    sha: str
    description: str = ""
    # The real function: set at load for a built-in (trusted package code);
    # for a library definition, defined from `code` only inside a task
    # process (see the `fn` property).
    _fn: BlockFn | None = field(default=None, repr=False)
    _signature: BlockSignature | None = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        m = self.meta
        self.block_type: BlockType = m.get("block_type", "standard")
        self.display_name: str = m.get("display_name") or self.id
        self.version: int = int(m.get("version", 1))
        self.group: str | None = m.get("group")
        self.tags: tuple[str, ...] = tuple(m.get("tags", ()))
        self.inputs: list[PortSpec] = [_port(p) for p in m.get("inputs", [])]
        self.outputs: list[PortSpec] = [_port(p) for p in m.get("outputs", [])]
        self.form: tuple[FieldSpec, ...] = tuple(_field(f) for f in m.get("form", []))
        self.requires: tuple[str, ...] = tuple(m.get("requires", ()))
        # What the block needs the workspace to allow (see capabilities.py),
        # beyond what its form's field kinds already imply.
        self.capabilities: tuple[str, ...] = tuple(m.get("capabilities", ()))
        self.hints: str | None = m.get("hints")
        self.display: DisplayKind | None = m.get("display")
        self.metadata: dict[str, Any] = dict(m.get("meta") or {})
        self.renamed: dict[str, str] = dict(m.get("renamed") or {})
        self.renamed_ports: dict[str, str] = dict(m.get("renamed_ports") or {})

    @property
    def category(self) -> str:
        return self.id

    @property
    def summary(self) -> str:
        return " ".join(self.description.strip().split("\n\n", 1)[0].split())

    @property
    def aggregate_outputs(self) -> tuple[str, ...]:
        return tuple(p.name for p in self.outputs if p.aggregate and p.type == "dataframe")

    @property
    def fn(self) -> BlockFn:
        """The block's function. A built-in's is its module attribute; a
        library block's is defined from `code`, which executes the block's
        module-level code -- so this is only allowed in a task process
        (block_library.compile_fn raises anywhere else). Everything that
        only needs to know the parameters uses `signature` instead."""
        if self._fn is None:
            from ..block_library import compile_fn

            self._fn = compile_fn(self.id, self.code)
        return self._fn

    @property
    def signature(self) -> BlockSignature:
        """The function's parameters, read from `code` with the AST -- the
        same for a built-in and a library block, and never executing either
        (see signature.py). Raises signature.SignatureError when the code
        has no `def <id>(...)`."""
        if self._signature is None:
            from ..signature import BlockSignature

            self._signature = BlockSignature.from_code(self.code, self.id)
        return self._signature

    def param_defaults(self) -> dict[str, Any]:
        """Every parameter's signature default whose value is a literal
        (ports and required params left out) -- what the declarative
        metadata rules see for a param the user hasn't set. A non-literal
        default isn't known outside the task process, so it's left out."""
        return self.signature.literal_defaults()

    def embedded(self) -> dict[str, Any]:
        """The copy a placed library block keeps in the project file, so it
        runs without the library and never changes silently (see
        graph.BlockInstance.definition)."""
        return {"id": self.id, "meta": self.meta, "code": self.code, "sha": self.sha, "description": self.description}


# Built-in blocks by id, filled by block_library.load_builtin() at package
# import (see modelmaker/__init__.py).
BLOCK_REGISTRY: dict[str, BlockDefinition] = {}
