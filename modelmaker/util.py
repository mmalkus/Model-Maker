from __future__ import annotations

import inspect
from typing import Callable

from .packet import ColumnRole

# A block function opts into auto-filled target/predicted resolution purely
# by naming its parameter one of these -- same convention as
# output_dir/block_id (see runner.run_block, compiler.compile_graph): no
# per-block registry to keep in sync, and it works for custom AI-authored
# blocks for free as long as the drafted code names the parameter this way.
TARGET_PARAM_NAMES = ("target", "target_col")
PREDICTED_PARAM_NAMES = ("score_col", "predicted_col")

# Which param names, for a given upstream column role, opt a block function
# into that role's dynamic default (see packet.resolve_role_column). Keyed
# by role so runner.run_block/compiler.compile_graph can loop over every
# auto-fillable role the same way, instead of hardcoding "target" alone.
ROLE_PARAM_NAMES: dict[ColumnRole, tuple[str, ...]] = {
    ColumnRole.TARGET: TARGET_PARAM_NAMES,
    ColumnRole.PREDICTED: PREDICTED_PARAM_NAMES,
}


def accepts_param(fn: Callable, name: str) -> bool:
    try:
        return name in inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return False


def find_target_param(fn: Callable) -> str | None:
    """Which of TARGET_PARAM_NAMES this function's signature accepts, if
    any -- the name to auto-fill (see packet.resolve_target_column) when the
    block's own params don't already set it."""
    return find_role_param(fn, ColumnRole.TARGET)


def find_role_param(fn: Callable, role: ColumnRole) -> str | None:
    """Which of ROLE_PARAM_NAMES[role] this function's signature accepts, if
    any -- the name to auto-fill (see packet.resolve_role_column) when the
    block's own params don't already set it."""
    return next((name for name in ROLE_PARAM_NAMES.get(role, ()) if accepts_param(fn, name)), None)
