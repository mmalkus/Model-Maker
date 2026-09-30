"""Packaging hook.

All metadata lives in pyproject.toml; this file only wires two build steps
into `build_py` so that `python -m build` / `pip install .` produce a
self-contained wheel without separate manual steps:

* the frontend build, bundled into modelmaker/static/;
* the demo project folders from projects/, copied into
  modelmaker/examples/. Their sample data is not copied -- the wheel ships
  modelmaker/demo_data.py, which regenerates it (fixed seed) when
  `modelmaker-demo` sets up a workspace.

Both are skipped for editable installs (`pip install -e .`), which keep
working from the repo's own frontend/ and projects/ directly.
"""

import os
import shutil
import subprocess
from pathlib import Path

from setuptools import setup
from setuptools.command.build_py import build_py as _build_py

_ROOT = Path(__file__).parent
_FRONTEND = _ROOT / "frontend"
_PROJECTS = _ROOT / "projects"
_EXAMPLES_OUT = _ROOT / "modelmaker" / "examples"
_DEMO_PROJECTS = ["demo_pd_model"]


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


def _copy_demo_projects() -> None:
    # Rebuilt from scratch so a renamed/removed demo never lingers in a wheel.
    if _EXAMPLES_OUT.exists():
        shutil.rmtree(_EXAMPLES_OUT)
    for name in _DEMO_PROJECTS:
        src = _PROJECTS / name
        if (src / "model.json").is_file():
            shutil.copytree(src, _EXAMPLES_OUT / name, ignore=shutil.ignore_patterns(".git", "__pycache__"))


class build_py(_build_py):
    def run(self):
        if not getattr(self, "editable_mode", False):
            _build_frontend()
            _copy_demo_projects()
        super().run()


setup(cmdclass={"build_py": build_py})
