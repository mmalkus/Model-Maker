"""What a workspace may do beyond pure computation: its capability set.

Most blocks only transform the data wired into them. A few reach outside
the project -- a file on disk, a database, an export -- and whether that's
allowed depends on where the app runs. The local app runs on the user's
own machine, so its blocks may read any file it can (`filesystem`). An
application hosting other people's workspaces may not want that, and
gives them data through datasets, admin-defined sources and exports
instead. A workspace says which of these it grants
(``Workspace(capabilities=...)``); a block says which it needs.

The capabilities this package knows:

- ``filesystem`` -- the block reads or writes files at paths the user
  types, or reaches a database through a connection variable: read_csv,
  read_parquet, read_json, read_excel, read_sql, write_csv. Any block with
  a form field of kind ``path`` or ``env`` needs it too. The local app
  grants it.
- ``datasets`` -- the block reads a dataset by name (a form field of kind
  ``dataset``): read_dataset. Granted when an embedding application gives
  the workspace a dataset provider (see datasets.py).
- ``sources`` -- the block runs a query against a source the application's
  administrator defined (a field of kind ``source``): read_source.
- ``exports`` -- the block's result becomes an export the user downloads:
  export.

A block declares what it needs in its ``BLOCKS`` entry
(``"capabilities": ["filesystem"]``), optionally with ``"use_instead"``
naming the block to suggest when it isn't allowed; field kinds add their
own (``required``). A workspace without one of a block's capabilities
doesn't offer the block (the palette, ``/api/registry`` and the AI
builder's catalogue leave it out) and refuses to place, load or run it,
with a message naming the alternative (``refusal``).

Capabilities are resolved per request, like the workspace itself:
``set_resolver`` is installed by the API (api.py) to read
``current_workspace().capabilities``; anything outside a request sees the
local app's set, ``LOCAL``.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .blocks.base import BlockDefinition

FILESYSTEM = "filesystem"
DATASETS = "datasets"
SOURCES = "sources"
EXPORTS = "exports"
KNOWN = frozenset({FILESYSTEM, DATASETS, SOURCES, EXPORTS})

# The local, single-user app: files at any path, as it always did. Datasets,
# sources and exports need a provider an embedding application supplies.
LOCAL: frozenset[str] = frozenset({FILESYSTEM})

# The capability a form field's kind implies.
FIELD_KIND_CAPABILITIES = {"path": FILESYSTEM, "env": FILESYSTEM, "dataset": DATASETS, "source": SOURCES}

_resolver: Callable[[], Iterable[str]] | None = None
_override: ContextVar[frozenset[str] | None] = ContextVar("modelmaker_capabilities", default=None)


class CapabilityError(ValueError):
    """A block needs a capability the workspace doesn't grant."""


def set_resolver(resolver: Callable[[], Iterable[str]] | None) -> None:
    """Install the function `granted()` asks (the API's: the request's
    workspace). None: the local app's set."""
    global _resolver
    _resolver = resolver


def granted() -> frozenset[str]:
    """The capabilities of whoever is asking right now."""
    forced = _override.get()
    if forced is not None:
        return forced
    return LOCAL if _resolver is None else frozenset(_resolver())


@contextmanager
def granting(capabilities: Iterable[str]) -> Iterator[frozenset[str]]:
    """Make granted() answer `capabilities` inside the block, whatever the
    resolver says (scripts, tests, tools listing a catalogue for another
    profile)."""
    caps = frozenset(capabilities)
    token = _override.set(caps)
    try:
        yield caps
    finally:
        _override.reset(token)


def required(definition: BlockDefinition) -> frozenset[str]:
    """Every capability the block needs: what it declares, plus what its
    form's field kinds imply."""
    needs = set(definition.capabilities)
    needs.update(FIELD_KIND_CAPABILITIES[f.kind] for f in definition.form if f.kind in FIELD_KIND_CAPABILITIES)
    return frozenset(needs)


def missing(definition: BlockDefinition, capabilities: Iterable[str] | None = None) -> list[str]:
    have = granted() if capabilities is None else frozenset(capabilities)
    return sorted(required(definition) - have)


def allowed(definition: BlockDefinition, capabilities: Iterable[str] | None = None) -> bool:
    return not missing(definition, capabilities)


def _explain(capability: str) -> str:
    return {
        FILESYSTEM: "reads or writes files on the server by path",
        DATASETS: "reads a dataset, and this workspace has no datasets",
        SOURCES: "queries a data source, and this workspace has no sources",
        EXPORTS: "makes an export, and this workspace has no exports",
    }.get(capability, f"needs '{capability}'")


def refusal(definition: BlockDefinition, needs: Iterable[str] | None = None) -> str:
    """Why `definition` can't be used here, and what to use instead."""
    needs = list(needs) if needs is not None else missing(definition)
    why = "; ".join(_explain(c) for c in needs) or "isn't available"
    text = f"'{definition.display_name}' ({definition.id}) {why}, which this workspace doesn't allow"
    instead = definition.meta.get("use_instead")
    if instead:
        from .blocks.base import BLOCK_REGISTRY

        alt = BLOCK_REGISTRY.get(instead)
        name = f"'{alt.display_name}' ({instead})" if alt is not None else f"'{instead}'"
        text += f" -- use {name} instead"
    return text


def check(definition: BlockDefinition, capabilities: Iterable[str] | None = None) -> None:
    """Raise CapabilityError when the block isn't allowed."""
    needs = missing(definition, capabilities)
    if needs:
        raise CapabilityError(refusal(definition, needs))
