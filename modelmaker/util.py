from __future__ import annotations

import inspect
from typing import Callable


def accepts_param(fn: Callable, name: str) -> bool:
    try:
        return name in inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return False
