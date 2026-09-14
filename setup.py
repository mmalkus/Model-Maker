"""Packaging hook.

All metadata lives in pyproject.toml; this file only wires the frontend
build into `build_py` so that `python -m build` / `pip install .` produce
a self-contained wheel (UI bundled into modelmaker/static/) without a
separate manual `npm run build` step. Skipped for editable installs
(`pip install -e .`), where backend-only dev is the common case.
"""

import os
import shutil
import subprocess
from pathlib import Path

from setuptools import setup
from setuptools.command.build_py import build_py as _build_py

_ROOT = Path(__file__).parent
_FRONTEND = _ROOT / "frontend"


def _build_frontend() -> None:
    if os.environ.get("MODELMAKER_SKIP_FRONTEND_BUILD"):
        return
    if not (_FRONTEND / "package.json").is_file():
        return
    npm = shutil.which("npm")
    if npm is None:
        print(
            "modelmaker: npm not found on PATH, skipping frontend build -- "
            "the package will ship without the built-in UI. Set "
            "MODELMAKER_SKIP_FRONTEND_BUILD=1 to silence this."
        )
        return
    try:
        subprocess.run([npm, "install"], cwd=_FRONTEND, check=True)
        subprocess.run([npm, "run", "build"], cwd=_FRONTEND, check=True)
    except subprocess.CalledProcessError as exc:
        print(
            f"modelmaker: frontend build failed ({exc}) -- packaging "
            "without the built-in UI"
        )


class build_py(_build_py):
    def run(self):
        if not getattr(self, "editable_mode", False):
            _build_frontend()
        super().run()


setup(cmdclass={"build_py": build_py})
