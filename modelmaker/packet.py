from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Any

import polars as pl


class ColumnRole(str, Enum):
    ID = "id"
    TARGET = "target"
    PREDICTED = "predicted"
    WEIGHT = "weight"
    FEATURE = "feature"
    DATE = "date"
    SEGMENT = "segment"
    EXCLUDED = "excluded"
    UNASSIGNED = "unassigned"


# Roles that may sit on at most one column within any single schema (a
# block's own output, or wherever schemas get merged, e.g. join) --
# everything else (feature/segment/excluded/unassigned) is fine on any
# number of columns at once.
UNIQUE_ROLES = frozenset({ColumnRole.ID, ColumnRole.TARGET, ColumnRole.PREDICTED, ColumnRole.WEIGHT, ColumnRole.DATE})


@dataclass
class ColumnMeta:
    dtype: str
    role: ColumnRole = ColumnRole.UNASSIGNED
    description: str | None = None
    tags: list[str] = field(default_factory=list)


@dataclass
class ColumnStats:
    count: int
    null_count: int
    n_unique: int | None = None
    mean: float | None = None
    std: float | None = None
    min: Any = None
    max: Any = None


@dataclass
class DataFramePacket:
    data: pl.DataFrame
    schema_meta: dict[str, ColumnMeta]
    lineage: list[str] = field(default_factory=list)
    summary: dict[str, ColumnStats] | None = None

    def with_lineage(self, block_id: str) -> DataFramePacket:
        return replace(self, lineage=[*self.lineage, block_id], summary=None)

    def compute_summary(self, force: bool = False) -> DataFramePacket:
        """Populate per-column summary stats on demand (not eager on every wire)."""
        if self.summary is not None and not force:
            return self
        summary: dict[str, ColumnStats] = {}
        for name in self.data.columns:
            s = self.data[name]
            is_numeric = s.dtype.is_numeric()
            n = s.len()
            summary[name] = ColumnStats(
                count=n,
                null_count=s.null_count(),
                n_unique=s.n_unique(),
                mean=float(s.mean()) if is_numeric and n else None,
                std=float(s.std()) if is_numeric and n and n > 1 else None,
                min=s.min(),
                max=s.max(),
            )
        return replace(self, summary=summary)


def find_duplicate_unique_role(schema_meta: dict[str, ColumnMeta]) -> tuple[ColumnRole, list[str]] | None:
    """A unique role (see UNIQUE_ROLES) sitting on more than one column
    within one schema is a real conflict, not a preference -- e.g. a join
    silently combining two independently-tagged target columns. Returns the
    offending role and its (sorted) column names, or None if the schema is
    clean. Called both when a role is hand-tagged (immediate feedback) and
    after every block run (the backstop that catches a merge-time collision
    neither side could have seen on its own)."""
    by_role: dict[ColumnRole, list[str]] = {}
    for name, meta in schema_meta.items():
        if meta.role in UNIQUE_ROLES:
            by_role.setdefault(meta.role, []).append(name)
    for role, cols in by_role.items():
        if len(cols) > 1:
            return role, sorted(cols)
    return None


def resolve_target_column(schema_metas: list[dict[str, ColumnMeta]]) -> str | None:
    """The column tagged role=target among one or more input schemas, for
    auto-filling a block's `target`/`target_col` param when left unset (see
    util.find_target_param). None if no column is tagged, or if more than
    one distinctly-named column is (ambiguous -- caller leaves the param
    unset rather than guessing, which surfaces as a normal missing-argument
    error)."""
    candidates = {name for meta in schema_metas for name, m in meta.items() if m.role == ColumnRole.TARGET}
    return next(iter(candidates)) if len(candidates) == 1 else None
