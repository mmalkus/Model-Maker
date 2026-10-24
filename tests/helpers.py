from __future__ import annotations

import ast
from dataclasses import replace
from typing import Any

from modelmaker.block_library import _sha
from modelmaker.blocks.base import BLOCK_REGISTRY, PortSpec
from modelmaker.graph import BlockInstance, Position


def library_definition(
    code: str,
    block_type: str = "standard",
    inputs: list[PortSpec] | None = None,
    outputs: list[PortSpec] | None = None,
    meta: dict[str, Any] | None = None,
    display_name: str | None = None,
    version: int = 1,
) -> dict[str, Any]:
    """An embedded library definition (see BlockDefinition.embedded) for a
    block whose code is `code` -- its id is the code's public function."""
    tree = ast.parse(code)
    block_id = next(n.name for n in tree.body if isinstance(n, ast.FunctionDef) and not n.name.startswith("_"))

    def port(p: PortSpec) -> dict[str, Any]:
        return {"name": p.name, "type": p.type, "required": p.required, "lazy": p.lazy, "aggregate": p.aggregate}

    entry: dict[str, Any] = {
        "display_name": display_name or block_id,
        "version": version,
        "block_type": block_type,
        "inputs": [port(p) for p in inputs or []],
        "outputs": [port(p) for p in outputs or []],
    }
    if meta:
        entry["meta"] = meta
    return {"id": block_id, "meta": entry, "code": code, "sha": _sha(entry, code), "description": ""}


def make_block(
    bid: str,
    category: str,
    params: dict[str, Any] | None = None,
    block_type: str | None = None,
    lane: str | None = None,
    x: float = 0,
    y: float = 0,
    code: str | None = None,
    code_version: int = 1,
    inputs=None,
    outputs=None,
    meta: dict[str, Any] | None = None,
) -> BlockInstance:
    """A built-in block of `category`, or -- with `code` -- a library block
    embedding a definition built from that code (its category becomes the
    code's function name)."""
    definition = None
    if code is not None:
        definition = library_definition(code, block_type or "standard", inputs, outputs, meta)
        category = definition["id"]
        block_type = block_type or "standard"
    else:
        spec = BLOCK_REGISTRY[category]
        block_type = block_type or spec.block_type
        inputs = inputs if inputs is not None else [replace(p) for p in spec.inputs]
        outputs = outputs if outputs is not None else [replace(p) for p in spec.outputs]
    return BlockInstance(
        id=bid,
        block_type=block_type,
        category=category,
        name=bid,
        lane=lane,
        position=Position(x=x, y=y),
        code_version=code_version,
        params=params or {},
        inputs=list(inputs or []),
        outputs=list(outputs or []),
        definition=definition,
    )


def meta_of(category: str):
    """A built-in's declared metadata rules as a transform (see
    metadata_transforms.schema_diff) -- what the runner applies."""
    from modelmaker.runner import metadata_transform

    return metadata_transform(BLOCK_REGISTRY[category])


def set_code(block: BlockInstance, code: str) -> None:
    """Give a library block new code (a new embedded definition), the way
    an Update from the library does."""
    raw = dict(block.definition)
    raw["code"] = code
    raw["sha"] = _sha(raw["meta"], code)
    block.definition = raw
    block.code_version += 1


def write_library_file(name: str, blocks: dict[str, dict[str, Any]], code: str, doc: str = "Test blocks.") -> str:
    """Write a block file into the test's library folder (see conftest's
    block_library_dir) and return its name."""
    from modelmaker import library_store
    from modelmaker.block_library import blocks_literal

    source = f'"""{doc}"""\n\nimport polars as pl\n\n{blocks_literal(blocks)}\n\n{code}'
    path = library_store.library_dir() / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(source, encoding="utf-8")
    library_store.reload()
    return name
