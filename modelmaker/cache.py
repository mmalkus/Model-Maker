from __future__ import annotations

import pickle
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass
class CacheEntry:
    outputs: dict[str, Any]
    timestamp: str


class CacheStore:
    """Keyed by lineage-based cache key (see Runner.compute_key) — never by
    a hash of the actual data, so computing a key is cheap regardless of
    DataFrame size. `cache_dir`, when given, is expected to be gitignored:
    it holds run outputs, not the versioned project definition."""

    def __init__(self, cache_dir: Path | None = None):
        self._mem: dict[str, CacheEntry] = {}
        self._dir = cache_dir
        if self._dir:
            self._dir.mkdir(parents=True, exist_ok=True)

    def get(self, key: str) -> CacheEntry | None:
        if key in self._mem:
            return self._mem[key]
        if self._dir:
            p = self._dir / f"{key}.pkl"
            if p.exists():
                entry = pickle.loads(p.read_bytes())
                self._mem[key] = entry
                return entry
        return None

    def set(self, key: str, outputs: dict[str, Any]) -> CacheEntry:
        entry = CacheEntry(outputs=outputs, timestamp=datetime.now(timezone.utc).isoformat())
        self._mem[key] = entry
        if self._dir:
            (self._dir / f"{key}.pkl").write_bytes(pickle.dumps(entry))
        return entry
