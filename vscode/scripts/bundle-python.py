"""Bundles the Model-Maker Python package into the extension (vscode/python/).

The extension runs Model-Maker from this copy -- in its own environment, or
in an interpreter the user picks -- so no Model-Maker package is ever pip
installed; only its standard open-source dependencies are (see
src/environment.ts). This builds the package's wheel exactly as for PyPI,
minus the browser UI (the extension has its own copy, media/webview), and
unpacks it whole: the dist-info keeps importlib.metadata's version (so
/api/version) right. bundle.json records the version and the core
requirements the environment needs.

    python scripts/bundle-python.py   (from vscode/; needs `pip install build`)
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import tomllib
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent  # vscode/
REPO = HERE.parent
OUT = HERE / "python"


def main() -> None:
    project = tomllib.loads((REPO / "pyproject.toml").read_text("utf-8"))["project"]
    with tempfile.TemporaryDirectory() as tmp:
        env = {**os.environ, "MODELMAKER_SKIP_FRONTEND_BUILD": "1"}
        subprocess.run([sys.executable, "-m", "build", "--wheel", "--outdir", tmp, str(REPO)], check=True, env=env)
        wheels = list(Path(tmp).glob("*.whl"))
        if len(wheels) != 1:
            sys.exit(f"expected one wheel, got {wheels}")
        if OUT.exists():
            shutil.rmtree(OUT)
        OUT.mkdir()
        with zipfile.ZipFile(wheels[0]) as z:
            z.extractall(OUT)
    static = OUT / "modelmaker" / "static"
    if static.exists():
        shutil.rmtree(static)
    bundle = {
        "version": project["version"],
        "requires_python": project.get("requires-python", ">=3.11"),
        "requirements": project["dependencies"],
    }
    (OUT / "bundle.json").write_text(json.dumps(bundle, indent=2) + "\n", "utf-8")
    size = sum(f.stat().st_size for f in OUT.rglob("*") if f.is_file())
    print(f"bundled modelmaker {bundle['version']} into {OUT} ({size // 1024} KB)")


if __name__ == "__main__":
    main()
