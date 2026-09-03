from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import modelmaker  # noqa: E402,F401 -- ensures the standard block registry is populated
