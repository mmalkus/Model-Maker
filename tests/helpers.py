from __future__ import annotations

from typing import Any

from modelmaker.blocks.base import BLOCK_REGISTRY
from modelmaker.graph import BlockInstance, Position


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
    metadata_transform=None,
) -> BlockInstance:
    spec = BLOCK_REGISTRY.get(category)
    is_custom = spec is None
    if spec is not None:
        block_type = block_type or spec.block_type
        inputs = inputs if inputs is not None else list(spec.inputs)
        outputs = outputs if outputs is not None else list(spec.outputs)
    return BlockInstance(
        id=bid,
        block_type=block_type or "standard",
        category=category,
        name=bid,
        lane=lane,
        position=Position(x=x, y=y),
        code_version=code_version,
        params=params or {},
        inputs=inputs or [],
        outputs=outputs or [],
        code=code,
        metadata_transform=metadata_transform,
        is_custom=is_custom,
    )
