"""Entry point wrapper for the modelmaker-tui console script.

The `tui` extra's dependencies (textual, httpx, plotext) are optional --
but `modelmaker-tui` itself is always installed, since setuptools entry
points can't be conditioned on an extra. Without this wrapper, running it
after a base install crashes with a raw ModuleNotFoundError traceback from
deep inside modelmaker.tui.app instead of saying what to actually do.
"""

from __future__ import annotations


def main() -> None:
    try:
        from .app import main as _main
    except ImportError as exc:
        raise SystemExit(
            f"modelmaker-tui is missing a dependency ({exc}). Install it with:\n"
            '  pip install "quantology-modelmaker[tui]"'
        ) from None
    _main()
