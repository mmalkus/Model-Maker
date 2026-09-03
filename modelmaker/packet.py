from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Any

import polars as pl


class ColumnRole(str, Enum):
    ID = "id"
    TARGET = "target"
    WEIGHT = "weight"
    FEATURE = "feature"
    DATE = "date"
    SEGMENT = "segment"
    EXCLUDED = "excluded"
    UNASSIGNED = "unassigned"


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
