"""Deterministic seed hierarchy (see stochastic-engine-proposal.md S3).

Every stochastic block resolves its RNG from a project/block-level `seed`
param plus a `path` of stable string identifiers -- never from ambient
`numpy.random` global state, wall clock, or worker PID -- so results are
identical regardless of execution order or parallelism. `path` typically
starts with the block's own id (auto-injected the same way output_dir/
block_id already are -- see runner.run_block) and grows with a chunk or
component index for anything that needs independent sub-streams.
"""

from __future__ import annotations

import hashlib

import numpy as np


def _path_to_spawn_key(path: list[str]) -> tuple[int, ...]:
    return tuple(int.from_bytes(hashlib.sha256(p.encode()).digest()[:4], "big") for p in path)


def spawn_rng(seed: int, path: list[str]) -> np.random.Generator:
    """A Generator keyed on (seed, path) alone -- two calls with the same
    arguments always produce the same stream, whether they happen in the
    same process, different worker processes, or in a different order."""
    seed_sequence = np.random.SeedSequence(entropy=seed, spawn_key=_path_to_spawn_key(path))
    return np.random.Generator(np.random.PCG64(seed_sequence))
