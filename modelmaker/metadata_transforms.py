"""Column metadata for a block's outputs, derived from the block's declared
`meta` rules -- never from per-block code. One function, `schema_diff`,
serves every block:

  - An output column that exists in an input keeps that input's metadata
    (role, description, tags), with its dtype refreshed. When several inputs
    have the column, the first-listed input port wins (a join's left side).
  - A column that is new gets its dtype inferred and `new_role` (default
    "unassigned").
  - `roles` rules then tag specific columns.

    "meta": {
        "inherit": "all" | "none" | {"port": "df"} | {"columns_param": "by"},
        "new_role": "feature",
        "roles": [
            {"column": "lgd", "role": "target", "if_taken": "feature"},
            {"column_param": "output_col", "role": "predicted", "displaced": "feature"},
            {"column_param": "flag_col", "fallback": "default_{horizon}m", "role": "target"},
            {"column_suffix": "_bin", "role": "segment"},
        ],
    }

`inherit`: "none" for a block whose outputs are new tables (statistics,
input readers); {"port": p} to inherit from one input only;
{"columns_param": p} to inherit only the columns a list param names
(group-by keys -- the aggregated columns are new values, not the inputs').

A role rule names its column directly, via a param (`fallback` is a
template over the params used when that param is unset), or by suffix (new
columns only). For a unique role (target, predicted, ...), `if_taken`
gives the column that role instead when another column already has it,
and `displaced` re-tags any other column that has it.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Callable

from .packet import UNIQUE_ROLES, ColumnMeta, ColumnRole

MetadataTransform = Callable[[dict[str, dict[str, ColumnMeta]], dict[str, Any], dict[str, Any]], dict[str, dict[str, ColumnMeta]]]


def _rule_column(rule: dict[str, Any], params: dict[str, Any]) -> str | None:
    if "column" in rule:
        return rule["column"]
    value = params.get(rule["column_param"])
    if not value and rule.get("fallback"):
        try:
            value = rule["fallback"].format_map(params)
        except (KeyError, IndexError, ValueError):
            return None
    return value if isinstance(value, str) and value else None


def schema_diff(spec: dict[str, Any] | None, defaults: dict[str, Any] | None = None) -> MetadataTransform:
    """The metadata transform for a block's `meta` rules (see module doc).
    `defaults` are the function's signature defaults, so a rule over an
    unset param sees what the function actually used."""
    spec = spec or {}
    inherit = spec.get("inherit", "all")
    new_role = ColumnRole(spec.get("new_role", "unassigned"))
    rules = spec.get("roles", [])

    def transform(input_metas, outputs, params):
        p = {**(defaults or {}), **(params or {})}
        if inherit == "none":
            sources: list[dict[str, ColumnMeta]] = []
        elif isinstance(inherit, dict) and "port" in inherit:
            sources = [input_metas.get(inherit["port"], {})]
        else:
            sources = list(input_metas.values())
        keep_only = set(p.get(inherit["columns_param"]) or []) if isinstance(inherit, dict) and "columns_param" in inherit else None

        result: dict[str, dict[str, ColumnMeta]] = {}
        for port, df in outputs.items():
            meta: dict[str, ColumnMeta] = {}
            new_cols: set[str] = set()
            for name in df.columns:
                dtype = str(df.schema[name])
                found = None
                if keep_only is None or name in keep_only:
                    found = next((s[name] for s in sources if name in s), None)
                if found is not None:
                    meta[name] = replace(found, dtype=dtype, tags=list(found.tags))
                else:
                    meta[name] = ColumnMeta(dtype=dtype, role=new_role)
                    new_cols.add(name)
            for rule in rules:
                role = ColumnRole(rule["role"])
                if "column_suffix" in rule:
                    targets = [c for c in new_cols if c.endswith(rule["column_suffix"])]
                else:
                    col = _rule_column(rule, p)
                    targets = [col] if col in meta else []
                for col in targets:
                    given = role
                    if role in UNIQUE_ROLES:
                        others = [n for n, m in meta.items() if n != col and m.role == role]
                        if others and "if_taken" in rule:
                            given = ColumnRole(rule["if_taken"])
                        elif others and "displaced" in rule:
                            for n in others:
                                meta[n] = replace(meta[n], role=ColumnRole(rule["displaced"]))
                    meta[col] = replace(meta[col], role=given)
            result[port] = meta
        return result

    return transform
