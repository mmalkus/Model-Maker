"""Data that comes from the application, not from a path: datasets,
sources and exports.

The local app's input blocks read files at paths the user types
(read_csv and friends). An application hosting several users' workspaces
-- where block code runs in a sandbox that must not see the host's files
or network -- gives them data another way, through three small hooks it
implements and hands to its ``Workspace``:

- a **DatasetProvider**: the workspace's datasets, by name. read_dataset
  (a form field of kind ``dataset``) reads one. Before the block runs, the
  engine looks the name up and hands the block function the dataset's
  file, which the provider has made readable where block code runs (e.g. a
  read-only folder in the sandbox). The dataset's ``version`` -- something
  that changes whenever its content does, such as a content hash -- is
  part of the block's cache key, so replacing a dataset makes the block
  stale and never serves a result computed from the old content.
- a **SourceProvider**: named data sources (a database, typically) that
  the application's administrator defined. read_source (a field of kind
  ``source`` plus a ``sql`` query) asks the provider to run the query --
  in the application, with the administrator's credentials, never in the
  block's process -- and gets back a dataset holding the result, which the
  block function then reads like any other.
- an **ExportSink**: where an export block's result goes. When a block
  that needs the ``exports`` capability (the built-in ``export``) runs
  successfully, the engine passes its dataframe output to the sink, which
  stores it for the user to download.

None of this executes block code in the application or parses uploaded
files there: a dataset's raw bytes are parsed by the block, wherever block
code runs, and an export is written from the block's validated dataframe
output (see execution/wire.py). Which of the three a workspace offers is
part of its capability set (capabilities.py): ``datasets``, ``sources``
and ``exports``.

The local app supplies none of them, so read_dataset, read_source and
export aren't offered there; it keeps the path-based blocks.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from .capabilities import DATASETS, EXPORTS, LOCAL, SOURCES

if TYPE_CHECKING:
    import polars as pl

# File formats a dataset can be in; the block reading it parses it.
FORMATS = ("csv", "parquet", "json", "ndjson", "xlsx", "arrow")


class DatasetError(LookupError):
    """A dataset or source can't be used: unknown, refused, failed."""


@dataclass(frozen=True)
class DatasetInfo:
    """One dataset as a workspace sees it.

    `version` changes whenever the content does (a content hash is ideal);
    `path` is where block code reads it, read-only -- an absolute path as
    the block's process sees it."""

    name: str
    version: str
    format: str
    size: int = 0
    path: str = ""
    updated_at: str = ""
    origin: str = "upload"
    description: str = ""

    def public(self) -> dict[str, Any]:
        """What the API shows the user (never the path)."""
        return {
            "name": self.name,
            "version": self.version,
            "format": self.format,
            "size": self.size,
            "updated_at": self.updated_at,
            "origin": self.origin,
            "description": self.description,
        }


@dataclass(frozen=True)
class SourceInfo:
    name: str
    description: str = ""


@runtime_checkable
class DatasetProvider(Protocol):
    def list_datasets(self) -> list[DatasetInfo]: ...

    def get_dataset(self, name: str) -> DatasetInfo | None: ...


@runtime_checkable
class SourceProvider(Protocol):
    def list_sources(self) -> list[SourceInfo]: ...

    def query_source(self, source: str, query: str) -> DatasetInfo:
        """Run `query` against `source` and return the dataset holding the
        result (raise DatasetError with a message for the user on failure)."""
        ...


@dataclass(frozen=True)
class ExportRequest:
    block_id: str
    block_name: str
    name: str
    format: str
    data: pl.DataFrame
    # Sample mode was on: the export holds only the sampled rows.
    sampled: bool = False


@runtime_checkable
class ExportSink(Protocol):
    def export(self, request: ExportRequest) -> Any: ...


@dataclass(frozen=True)
class DataAccess:
    """What a workspace's runs may reach beyond the graph: its capability
    set and the providers behind it (see the module docstring). The local
    app's: the filesystem, nothing else."""

    capabilities: frozenset[str] = LOCAL
    datasets: DatasetProvider | None = None
    sources: SourceProvider | None = None
    exports: ExportSink | None = None

    @classmethod
    def make(
        cls,
        capabilities: frozenset[str] | set[str] | list[str] | tuple[str, ...] | None = None,
        datasets: DatasetProvider | None = None,
        sources: SourceProvider | None = None,
        exports: ExportSink | None = None,
    ) -> DataAccess:
        """`capabilities` None: the local app's, plus whatever the given
        providers make possible."""
        if capabilities is None:
            caps = set(LOCAL)
            if datasets is not None:
                caps.add(DATASETS)
            if sources is not None:
                caps.add(SOURCES)
            if exports is not None:
                caps.add(EXPORTS)
        else:
            caps = set(capabilities)
        return cls(frozenset(caps), datasets, sources, exports)
