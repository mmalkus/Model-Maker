from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import modelmaker  # noqa: E402,F401

# Populate the full block registry the same way the app does (see api.py):
# importing these for their registration side effect is what makes
# BLOCK_REGISTRY complete, so a test module run on its own sees the same
# catalog as one run as part of the whole suite.
from modelmaker.blocks import library, modelling, stat_tests  # noqa: E402,F401
