from __future__ import annotations

from typing import Any, Callable

import polars as pl

from .packet import ColumnMeta, ColumnRole


def infer_dtypes(
    input_metas: dict[str, dict[str, ColumnMeta]],
    outputs: dict[str, pl.DataFrame],
    params: dict[str, Any],
) -> dict[str, dict[str, ColumnMeta]]:
    """Used by input blocks: no upstream metadata to inherit, so build fresh
    ColumnMeta from dtype alone; role stays unassigned until tagged."""
    return {
        port: {name: ColumnMeta(dtype=str(df.schema[name])) for name in df.columns}
        for port, df in outputs.items()
    }


def passthrough(
    input_metas: dict[str, dict[str, ColumnMeta]],
    outputs: dict[str, pl.DataFrame],
    params: dict[str, Any],
) -> dict[str, dict[str, ColumnMeta]]:
    """Single input, single output, columns unchanged (e.g. filter)."""
    (in_meta,) = input_metas.values()
    result: dict[str, dict[str, ColumnMeta]] = {}
    for port, df in outputs.items():
        result[port] = {
            name: in_meta[name] if name in in_meta else ColumnMeta(dtype=str(df.schema[name]))
            for name in df.columns
        }
    return result


def narrow_from_single_input(
    input_metas: dict[str, dict[str, ColumnMeta]],
    outputs: dict[str, pl.DataFrame],
    params: dict[str, Any],
) -> dict[str, dict[str, ColumnMeta]]:
    """Single input, output columns are a subset of it (e.g. select)."""
    (in_meta,) = input_metas.values()
    result: dict[str, dict[str, ColumnMeta]] = {}
    for port, df in outputs.items():
        result[port] = {name: in_meta[name] for name in df.columns if name in in_meta}
    return result


def broadcast_to_all_outputs(
    input_metas: dict[str, dict[str, ColumnMeta]],
    outputs: dict[str, pl.DataFrame],
    params: dict[str, Any],
) -> dict[str, dict[str, ColumnMeta]]:
    """Single input, metadata unchanged, fanned out to every output port
    (e.g. train/test split: both sides keep the parent's column roles)."""
    (in_meta,) = input_metas.values()
    return {port: dict(in_meta) for port in outputs}


def declared(spec: dict[str, Any]) -> Callable[..., dict[str, dict[str, ColumnMeta]]]:
    """Build a metadata transform from a JSON-serializable declaration, the
    form used by custom/LLM-authored blocks in the project file:
        {"kind": "declared", "base": "in", "drops": [...], "adds": [...]}
    """

    def _fn(
        input_metas: dict[str, dict[str, ColumnMeta]],
        outputs: dict[str, pl.DataFrame],
        params: dict[str, Any],
    ) -> dict[str, dict[str, ColumnMeta]]:
        base_port = spec.get("base")
        base: dict[str, ColumnMeta] = dict(input_metas[base_port]) if base_port else {}
        for col in spec.get("drops", []):
            base.pop(col, None)
        for add in spec.get("adds", []):
            base[add["name"]] = ColumnMeta(
                dtype=add.get("dtype", "unknown"),
                role=ColumnRole(add.get("role", "unassigned")),
                description=add.get("description"),
            )
        result: dict[str, dict[str, ColumnMeta]] = {}
        for port, df in outputs.items():
            result[port] = {
                name: base[name] if name in base else ColumnMeta(dtype=str(df.schema[name]))
                for name in df.columns
            }
        return result

    return _fn


def resolve_metadata_transform(transform: Any) -> Callable[..., dict[str, dict[str, ColumnMeta]]]:
    if callable(transform):
        return transform
    kind = transform.get("kind")
    if kind == "infer_dtypes":
        return infer_dtypes
    if kind == "passthrough":
        return passthrough
    if kind == "narrow":
        return narrow_from_single_input
    if kind == "broadcast":
        return broadcast_to_all_outputs
    if kind == "declared":
        return declared(transform)
    raise ValueError(f"unknown metadata_transform kind: {kind!r}")
