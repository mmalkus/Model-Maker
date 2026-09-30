from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import modelmaker  # noqa: E402,F401

# Populate the full block registry the same way the app does (see api.py):
# importing these for their registration side effect is what makes
# BLOCK_REGISTRY complete, so a test module run on its own sees the same
# catalog as one run as part of the whole suite.
from modelmaker.blocks import binning, data_quality, feature_analysis, library, modelling, stat_tests, stochastic  # noqa: E402,F401

# The demo dataset isn't committed -- it's generated (fixed seed) by
# modelmaker/demo_data.py. Several tests and the demo project read it from
# sample_data/, so write it there before collection, and rewrite it if the
# generator has changed since it was last written.
import tempfile  # noqa: E402

from modelmaker import demo_data  # noqa: E402

_DEMO_CSV = Path(__file__).resolve().parents[1] / "sample_data" / demo_data.FILENAME
with tempfile.TemporaryDirectory() as _tmp:
    _fresh = demo_data.write_csv(Path(_tmp) / demo_data.FILENAME).read_bytes()
if not _DEMO_CSV.is_file() or _DEMO_CSV.read_bytes() != _fresh:
    _DEMO_CSV.parent.mkdir(parents=True, exist_ok=True)
    _DEMO_CSV.write_bytes(_fresh)
