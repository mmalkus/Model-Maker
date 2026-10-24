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

import inspect
from dataclasses import dataclass, field
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
PORT_TYPES: frozenset[str] = frozenset(PortType.__args__)  # type: ignore[attr-defined]

# Pipeline role only -- governs runtime behavior (input blocks need no
# upstream and support refresh/change detection; output blocks may take an
# injected output_dir).
BlockType = Literal["input", "standard", "output"]
BLOCK_TYPES: frozenset[str] = frozenset(BlockType.__args__)  # type: ignore[attr-defined]

# A block's core function is generic and plain-dataframe: it never touches
# DataFramePacket/ColumnMeta. The engine unwraps packets before calling it
# and rewraps the result afterward, applying the block's declared metadata
# rules (see metadata_transforms.schema_diff). This is what keeps the
# compiled script "clean" by construction.
BlockFn = Callable[..., Any]

# `path` (a file the block reads -- its mtime/size is the block's change
# fingerprint, see sources.py), `sql` (a query text) and `env` (the *name* of
# an environment variable) render like `text` with extra help in the UIs.
FieldKind = Literal["text", "number", "select", "column", "columns", "checkbox", "path", "sql", "env"]
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
    # Resolved lazily for a library definition (see block_library.compile_fn).
    _fn: BlockFn | None = field(default=None, repr=False)

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
        self.hints: str | None = m.get("hints")
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
        if self._fn is None:
            from ..block_library import compile_fn

            self._fn = compile_fn(self.id, self.code)
        return self._fn

    def param_defaults(self) -> dict[str, Any]:
        """Every parameter's signature default (ports and required params
        left out) -- what the declarative metadata rules see for a param the
        user hasn't set."""
        return {
            name: p.default
            for name, p in inspect.signature(self.fn).parameters.items()
            if p.default is not inspect.Parameter.empty
        }

    def embedded(self) -> dict[str, Any]:
        """The copy a placed library block keeps in the project file, so it
        runs without the library and never changes silently (see
        graph.BlockInstance.definition)."""
        return {"id": self.id, "meta": self.meta, "code": self.code, "sha": self.sha, "description": self.description}


# Built-in blocks by id, filled by block_library.load_builtin() at package
# import (see modelmaker/__init__.py).
BLOCK_REGISTRY: dict[str, BlockDefinition] = {}
