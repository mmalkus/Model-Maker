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

# A private block cache + recovery snapshot per test process, set before
# modelmaker.session reads it at import. The default (.modelmaker-cache in
# the cwd, i.e. the repo root) is shared by every pytest-xdist worker and by
# the TUI tests' API servers, and a stray recovery.json there makes the next
# server's startup stop on a "restore snapshot?" prompt.
import atexit  # noqa: E402
import shutil  # noqa: E402
import tempfile  # noqa: E402

_CACHE_DIR = tempfile.mkdtemp(prefix="modelmaker-test-cache-")
os.environ["MODELMAKER_CACHE_DIR"] = _CACHE_DIR
atexit.register(shutil.rmtree, _CACHE_DIR, ignore_errors=True)

# Pre-started block workers (see modelmaker/worker_pool.py) pay off when
# runs come one at a time; across pytest-xdist workers they mostly add
# memory -- a few hundred MB each, times every test process -- and compete
# for the CPU the parallel tests need. Off under xdist, unless set.
if os.environ.get("PYTEST_XDIST_WORKER"):
    os.environ.setdefault("MODELMAKER_WARM_WORKERS", "0")

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
# The API's run endpoints answer "still running" after RUN_WAIT_SECONDS,
# which a cold block run can exceed on a loaded machine (parallel workers);
# the in-process API tests expect a run's result, so give it longer.
import modelmaker.api  # noqa: E402

modelmaker.api.RUN_WAIT_SECONDS = 120.0

from modelmaker import demo_data  # noqa: E402

_DEMO_CSV = Path(__file__).resolve().parents[1] / "sample_data" / demo_data.FILENAME
with tempfile.TemporaryDirectory() as _tmp:
    _fresh = demo_data.write_csv(Path(_tmp) / demo_data.FILENAME).read_bytes()
if not _DEMO_CSV.is_file() or _DEMO_CSV.read_bytes() != _fresh:
    # Write-then-rename: under pytest-xdist every worker runs this at once,
    # and a plain write would let one worker read another's half-written file.
    _DEMO_CSV.parent.mkdir(parents=True, exist_ok=True)
    _part = _DEMO_CSV.with_name(f"{_DEMO_CSV.name}.{os.getpid()}.part")
    _part.write_bytes(_fresh)
    try:
        os.replace(_part, _DEMO_CSV)
    except PermissionError:
        # Windows: another worker has the (identical) file open mid-replace.
        _part.unlink()
