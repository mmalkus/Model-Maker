from __future__ import annotations

import inspect
from typing import Callable

# A block function opts into auto-filled target resolution purely by naming
# its parameter one of these -- same convention as output_dir/block_id
# (see runner.run_block, compiler.compile_graph): no per-block registry to
# keep in sync, and it works for custom AI-authored blocks for free as long
# as the drafted code names the parameter this way.
TARGET_PARAM_NAMES = ("target", "target_col")


def accepts_param(fn: Callable, name: str) -> bool:
    try:
        return name in inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return False


def find_target_param(fn: Callable) -> str | None:
    """Which of TARGET_PARAM_NAMES this function's signature accepts, if
    any -- the name to auto-fill (see packet.resolve_target_column) when the
    block's own params don't already set it."""
    return next((name for name in TARGET_PARAM_NAMES if accepts_param(fn, name)), None)
