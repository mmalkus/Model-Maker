"""Packaging hook.

All metadata lives in pyproject.toml; this file only wires the frontend
build and the bundled demo project into `build_py` so that `python -m
build` / `pip install .` produce a self-contained wheel (UI bundled into
modelmaker/static/, demo project bundled into modelmaker/examples/)
without separate manual steps. Skipped for editable installs (`pip
install -e .`), where working from the repo's own projects/ and
sample_data/ is the common case.
"""

import os
import shutil
import subprocess
from pathlib import Path

from setuptools import setup
from setuptools.command.build_py import build_py as _build_py

_ROOT = Path(__file__).parent
_FRONTEND = _ROOT / "frontend"
_DEMO_PROJECT = _ROOT / "projects" / "demo_pd_model.json"
_DEMO_DATA = _ROOT / "sample_data" / "pd_model_data.csv"
_EXAMPLES_OUT = _ROOT / "modelmaker" / "examples"


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


def _copy_examples() -> None:
    if not (_DEMO_PROJECT.is_file() and _DEMO_DATA.is_file()):
        return
    if _EXAMPLES_OUT.exists():
        shutil.rmtree(_EXAMPLES_OUT)
    _EXAMPLES_OUT.mkdir(parents=True)
    shutil.copy2(_DEMO_PROJECT, _EXAMPLES_OUT / _DEMO_PROJECT.name)
    data_dir = _EXAMPLES_OUT / "sample_data"
    data_dir.mkdir()
    shutil.copy2(_DEMO_DATA, data_dir / _DEMO_DATA.name)


class build_py(_build_py):
    def run(self):
        if not getattr(self, "editable_mode", False):
            _build_frontend()
            _copy_examples()
        super().run()


setup(cmdclass={"build_py": build_py})
