from __future__ import annotations

from typing import Any

from .base import ColumnInfo

# Polars dtype names (as ColumnMeta.dtype renders them) whose min/max are
# literal values out of the data rather than a numeric range.
_TEXT_DTYPES = ("String", "Utf8", "Categorical", "Enum", "Binary", "Object")


def _role_value(role: Any) -> str:
    return role.value if hasattr(role, "value") else str(role)


def reveals_values(dtype: str, role: Any) -> bool:
    """Whether a column's min/max would hand an LLM actual data values
    rather than a range: any `id`-role column (whatever its dtype -- the
    min/max of an id is some real customer's id), and any text column (the
    alphabetically-first/last value is a real value, e.g. a name)."""
    return _role_value(role) == "id" or str(dtype).startswith(_TEXT_DTYPES)


def column_info_for_llm(name: str, meta: Any, stats: Any = None) -> ColumnInfo:
    """The only way a ColumnInfo should be built for a prompt. Every LLM
    path sends schema plus summary statistics, never rows (see
    DraftContext) -- but a text or id column's min/max *are* rows' values,
    so they're dropped here (n_unique/count/nulls still go through). The
    UI's own previews (api._packet_preview) don't go through this: the
    user looking at their own data isn't what this guards against.

    `meta` is a packet.ColumnMeta; `stats` an optional packet.ColumnStats."""
    role = _role_value(meta.role)
    info = ColumnInfo(name=name, dtype=meta.dtype, role=role)
    if stats is None:
        return info
    info.count = stats.count
    info.null_count = stats.null_count
    info.n_unique = stats.n_unique
    info.mean = stats.mean
    info.std = stats.std
    if not reveals_values(meta.dtype, role):
        info.min = stats.min
        info.max = stats.max
    return info
