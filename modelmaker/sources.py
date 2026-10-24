"""Change detection for input blocks ("check for changes"), chosen from the
block's params form rather than written per block:

  - every field of kind `path` contributes the file's (mtime, size);
  - a block with a field of kind `sql` and one of kind `env` (the
    connection variable) contributes the first row of the block's optional
    `probe_query` -- a cheap query such as SELECT MAX(updated_at) FROM t --
    run against that connection. `probe_query` is an instance setting,
    never passed to the block's function.

A block with neither gets no fingerprint and never reports a change.
"""

from __future__ import annotations

import os
from typing import Any

from .blocks.base import BlockDefinition

# Instance params the architecture reads and the runner never passes on.
SOURCE_PARAMS = frozenset({"probe_query"})


def _file_fingerprint(path: Any) -> Any:
    try:
        st = os.stat(str(path))
    except OSError:
        return None
    return (st.st_mtime, st.st_size)


def _sql_fingerprint(definition: BlockDefinition, params: dict[str, Any]) -> Any:
    query = params.get("probe_query")
    env_field = next((f for f in definition.form if f.kind == "env"), None)
    if not query or env_field is None:
        return None
    try:
        import polars as pl

        uri = os.environ[params[env_field.key]]
        result = pl.read_database_uri(query, uri)
    except Exception:  # noqa: BLE001 -- an unreachable source is "unknown", not an error
        return None
    return str(result.row(0)) if result.height else ""


def fingerprint(definition: BlockDefinition, params: dict[str, Any]) -> Any:
    """A cheap, read-only value that changes when the block's source does,
    or None when the block's form gives nothing to watch."""
    parts: dict[str, Any] = {}
    for f in definition.form:
        if f.kind == "path" and params.get(f.key):
            parts[f.key] = _file_fingerprint(params[f.key])
    if any(f.kind == "sql" for f in definition.form):
        sql = _sql_fingerprint(definition, params)
        if sql is not None:
            parts["probe_query"] = sql
    return parts or None
