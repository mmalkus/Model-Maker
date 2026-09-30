from __future__ import annotations

import os
import sys
from pathlib import Path

# The git tests commit into throwaway repos; don't depend on the machine
# having a git identity configured (a fresh CI runner doesn't).
for _var, _value in {
    "GIT_AUTHOR_NAME": "Model-Maker Tests",
    "GIT_AUTHOR_EMAIL": "tests@example.invalid",
    "GIT_COMMITTER_NAME": "Model-Maker Tests",
    "GIT_COMMITTER_EMAIL": "tests@example.invalid",
}.items():
    os.environ.setdefault(_var, _value)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import modelmaker  # noqa: E402,F401

# Populate the full block registry the same way the app does (see api.py):
# importing these for their registration side effect is what makes
# BLOCK_REGISTRY complete, so a test module run on its own sees the same
# catalog as one run as part of the whole suite.
from modelmaker.blocks import binning, data_quality, feature_analysis, library, modelling, stat_tests, stochastic  # noqa: E402,F401

# The demo dataset isn't committed -- it's generated (fixed seed) by
# modelmaker/demo_data.py. The agent tests read it as a file through Read
# CSV, so write it to sample_data/ before collection, and rewrite it if the
# generator has changed since it was last written.
import tempfile  # noqa: E402

from modelmaker import demo_data  # noqa: E402

_DEMO_CSV = Path(__file__).resolve().parents[1] / "sample_data" / demo_data.FILENAME
with tempfile.TemporaryDirectory() as _tmp:
    _fresh = demo_data.write_csv(Path(_tmp) / demo_data.FILENAME).read_bytes()
if not _DEMO_CSV.is_file() or _DEMO_CSV.read_bytes() != _fresh:
    _DEMO_CSV.parent.mkdir(parents=True, exist_ok=True)
    _DEMO_CSV.write_bytes(_fresh)
