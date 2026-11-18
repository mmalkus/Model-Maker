"""Block outputs, kept by cache key: in memory, and on disk when a
directory is given (the app's `.modelmaker-cache`).

On disk each entry is one file, `<key>.mmw`, holding a single frame of the
data-only wire format task results already come back in (execution/
wire.py): dataframes as Arrow IPC, everything else as JSON-shaped data.
Reading a cache file therefore never runs code -- unlike the pickle files
this replaces, which handed anything able to write the cache folder code
execution in this process on the next read. A value the wire format only
admits through its allowlisting unpickler (a fitted scikit-learn estimator
on an `any` port, say) goes through that same allowlist here, never through
plain pickle.

Memory is bounded: entries that are also on disk are evicted from memory,
least recently used first, once the in-memory total passes a budget (see
memory_budget_bytes), and read back from disk on their next use. An entry
that only lives in memory (no cache folder, or a value the wire format
can't write) is never evicted -- evicting it would lose it -- and neither
is the most recently stored or read one, which is about to be used.

The cache is disposable: a file that is missing, from an older version
(`<key>.pkl`, which is never loaded -- it is deleted on sight), corrupt, or
tampered with is a miss, and a miss just re-runs the block. Likewise, a
value that can't be written as a frame stays in memory only.
"""

from __future__ import annotations

import logging
import os
import re
import tempfile
import threading
from collections import OrderedDict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .execution import wire
from .packet import ColumnMeta, ColumnRole, ColumnStats, DataFramePacket

log = logging.getLogger(__name__)

SUFFIX = ".mmw"
# What a frame's envelope says it is; anything else is ignored.
FRAME_KIND = "cache-entry"
FRAME_VERSION = 1
# Keys that may become file names (the runner's are 16 hex digits, see
# runner._hash); any other key is kept in memory only, so a key can never
# name a path outside the directory.
_KEY = re.compile(r"^[A-Za-z0-9_-]{1,128}$")
# Old pickle files swept when a store opens its directory: only the names
# the runner's keys produced -- the directory also holds other things
# (recovery snapshots, build logs). Any other `<key>.pkl` goes when its key
# is next read.
_LEGACY = re.compile(r"^[0-9a-f]{16}\.pkl$")
# Cache files are this process's own writes, so the cap is the wire
# format's generous default rather than the (possibly smaller) per-result
# cap a deployment sets for task results.
MAX_BYTES = wire.DEFAULT_MAX_BYTES
# The in-memory budget, in MB: MODELMAKER_CACHE_MEMORY_MB when set (0 = no
# limit), else this share of the machine's physical memory.
MEMORY_ENV = "MODELMAKER_CACHE_MEMORY_MB"
DEFAULT_MEMORY_SHARE = 0.25
MIN_MEMORY_BYTES = 256 * 1024**2


def memory_budget_bytes() -> int | None:
    """How much of the cache stays in memory, in bytes; None for no limit.
    A quarter of physical memory by default: block results are handed to
    worker processes and back, so the app process needs headroom well
    beyond what it caches."""
    raw = os.environ.get(MEMORY_ENV, "").strip()
    if raw:
        try:
            mb = int(raw)
        except ValueError:
            log.warning("ignoring %s=%r: not a whole number of MB", MEMORY_ENV, raw)
        else:
            return None if mb <= 0 else mb * 1024**2
    try:
        total = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    except (AttributeError, ValueError, OSError):
        return None  # not knowable here (e.g. Windows): no limit, as before
    return max(MIN_MEMORY_BYTES, int(total * DEFAULT_MEMORY_SHARE))


def entry_size(entry: "CacheEntry") -> int:
    """An entry's in-memory size, estimated from its dataframes (polars'
    own estimate); other values are small next to them and count as 0."""
    size = 0
    for value in entry.outputs.values():
        if isinstance(value, DataFramePacket):
            size += value.data.estimated_size()
    return size


@dataclass
class CacheEntry:
    outputs: dict[str, Any]
    timestamp: str


class CacheCorrupt(ValueError):
    """A cache file that isn't a valid entry."""


class CacheStore:
    """Keyed by lineage-based cache key (see Runner.compute_key) — never by
    a hash of the actual data, so computing a key is cheap regardless of
    DataFrame size. `cache_dir`, when given, is expected to be gitignored:
    it holds run outputs, not the versioned project definition."""

    def __init__(self, cache_dir: Path | None = None, memory_budget: int | None | object = ...):
        # Least recently used first. `_sizes` holds each entry's estimated
        # bytes and `_on_disk` the keys that can be read back after being
        # evicted (see _evict).
        self._mem: OrderedDict[str, CacheEntry] = OrderedDict()
        self._sizes: dict[str, int] = {}
        self._on_disk: set[str] = set()
        self._mem_total = 0
        self._lock = threading.Lock()
        # `...` (the default): from the environment / machine, see
        # memory_budget_bytes. None: no limit.
        self.memory_budget: int | None = memory_budget_bytes() if memory_budget is ... else memory_budget  # type: ignore[assignment]
        self._dir = cache_dir
        # The folder is made on the first write, not here: a store is
        # created whenever a session is (including the API module's default
        # workspace at import), and a process may well start somewhere it
        # can't write -- e.g. as an unprivileged user in a read-only
        # install directory -- without ever caching anything there.
        self._dir_ready = False
        if self._dir is not None and self._dir.is_dir():
            self._remove_legacy()

    def _ensure_dir(self) -> bool:
        """Whether the cache folder exists (creating it if need be). False
        when it can't be made: entries then stay in memory only."""
        if self._dir_ready:
            return True
        assert self._dir is not None
        try:
            self._dir.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            log.warning("can't create cache folder %s (%s); caching in memory only", self._dir, e)
            return False
        self._dir_ready = True
        return True

    def _remove_legacy(self) -> None:
        """Delete pickle files an older version wrote: never loaded, and
        left lying around they're only a trap."""
        assert self._dir is not None
        try:
            names = [p for p in self._dir.iterdir() if _LEGACY.match(p.name)]
        except OSError:
            return
        for p in names:
            _unlink(p)

    def _path(self, key: str) -> Path | None:
        if self._dir is None or not _KEY.match(key):
            return None
        return self._dir / f"{key}{SUFFIX}"

    @property
    def memory_bytes(self) -> int:
        """The estimated size of the entries held in memory right now."""
        return self._mem_total

    def _remember(self, key: str, entry: CacheEntry, on_disk: bool) -> None:
        """Hold `entry` in memory as the most recently used, then evict
        down to the budget."""
        with self._lock:
            self._mem_total -= self._sizes.pop(key, 0)
            self._mem[key] = entry
            self._mem.move_to_end(key)
            size = entry_size(entry)
            self._sizes[key] = size
            self._mem_total += size
            if on_disk:
                self._on_disk.add(key)
            else:
                self._on_disk.discard(key)
            self._evict()

    def _evict(self) -> None:
        """Drop the least recently used entries that are also on disk until
        the in-memory total fits the budget. Never the newest entry (it's
        about to be used), never one only memory holds. Caller holds the
        lock."""
        if self.memory_budget is None or self._mem_total <= self.memory_budget:
            return
        newest = next(reversed(self._mem), None)
        for key in list(self._mem):
            if self._mem_total <= self.memory_budget:
                break
            if key == newest or key not in self._on_disk:
                continue
            del self._mem[key]
            self._mem_total -= self._sizes.pop(key, 0)

    def get(self, key: str) -> CacheEntry | None:
        with self._lock:
            entry = self._mem.get(key)
            if entry is not None:
                self._mem.move_to_end(key)
                return entry
        p = self._path(key)
        if p is None:
            return None
        legacy = p.with_suffix(".pkl")
        if legacy.exists():
            _unlink(legacy)
        try:
            data = p.read_bytes()
        except OSError:
            return None
        try:
            entry = decode_entry(data)
        except (wire.WireError, CacheCorrupt) as e:
            log.warning("ignoring unreadable cache file %s: %s", p.name, e)
            _unlink(p)
            return None
        self._remember(key, entry, on_disk=True)
        return entry

    def set(self, key: str, outputs: dict[str, Any]) -> CacheEntry:
        entry = CacheEntry(outputs=outputs, timestamp=datetime.now(timezone.utc).isoformat())
        on_disk = False
        p = self._path(key)
        if p is not None:
            try:
                data = encode_entry(entry)
            except (wire.WireError, RecursionError) as e:
                log.info("not writing cache entry %s to disk: %s", key, e)
            else:
                if self._ensure_dir():
                    try:
                        _write_atomic(p, data)
                    except OSError as e:
                        log.warning("not writing cache entry %s to disk: %s", key, e)
                    else:
                        on_disk = True
                del data  # the encoded copy: not held while evicting
        self._remember(key, entry, on_disk=on_disk)
        return entry


# ---- the on-disk form ---------------------------------------------------------
#
# The frame's value is {"timestamp": str, "outputs": {port: node}}, a node
# being {"packet": {...}} for a DataFramePacket (the dataframe itself as
# Arrow, its column metadata as JSON) or {"value": v} for anything else.


def _meta_out(m: ColumnMeta) -> dict[str, Any]:
    return {"dtype": m.dtype, "role": m.role.value, "description": m.description, "tags": list(m.tags)}


def _stats_out(s: ColumnStats) -> dict[str, Any]:
    return {"count": s.count, "null_count": s.null_count, "n_unique": s.n_unique, "mean": s.mean, "std": s.std, "min": s.min, "max": s.max}


def _packet_out(p: DataFramePacket) -> dict[str, Any]:
    return {
        "data": p.data,
        "schema_meta": {name: _meta_out(m) for name, m in p.schema_meta.items()},
        "lineage": list(p.lineage),
        "summary": None if p.summary is None else {name: _stats_out(s) for name, s in p.summary.items()},
    }


def encode_entry(entry: CacheEntry) -> bytes:
    outputs = {
        port: {"packet": _packet_out(v)} if isinstance(v, DataFramePacket) else {"value": v}
        for port, v in entry.outputs.items()
    }
    value = {"timestamp": entry.timestamp, "outputs": outputs}
    try:
        return wire.encode_frame({"kind": FRAME_KIND, "version": FRAME_VERSION}, value, max_bytes=MAX_BYTES)
    except wire.WireError:
        if not any(isinstance(v, DataFramePacket) and v.summary is not None for v in entry.outputs.values()):
            raise
    # A summary is recomputed on demand; don't lose the entry over one
    # holding a statistic the format can't carry.
    for node in outputs.values():
        if "packet" in node:
            node["packet"]["summary"] = None
    return wire.encode_frame({"kind": FRAME_KIND, "version": FRAME_VERSION}, value, max_bytes=MAX_BYTES)


def _expect(cond: bool, what: str) -> None:
    if not cond:
        raise CacheCorrupt(what)


def _meta_in(raw: Any) -> ColumnMeta:
    _expect(isinstance(raw, dict) and isinstance(raw.get("dtype"), str), "bad column metadata")
    tags = raw.get("tags") or []
    description = raw.get("description")
    _expect(isinstance(tags, list) and all(isinstance(t, str) for t in tags), "bad column tags")
    _expect(description is None or isinstance(description, str), "bad column description")
    try:
        role = ColumnRole(raw.get("role", "unassigned"))
    except ValueError:
        raise CacheCorrupt("bad column role") from None
    return ColumnMeta(dtype=raw["dtype"], role=role, description=description, tags=list(tags))


def _stats_in(raw: Any) -> ColumnStats:
    _expect(isinstance(raw, dict) and type(raw.get("count")) is int and type(raw.get("null_count")) is int, "bad stats")
    return ColumnStats(
        count=raw["count"],
        null_count=raw["null_count"],
        n_unique=raw.get("n_unique"),
        mean=raw.get("mean"),
        std=raw.get("std"),
        min=raw.get("min"),
        max=raw.get("max"),
    )


def _packet_in(raw: Any) -> DataFramePacket:
    import polars as pl

    _expect(isinstance(raw, dict) and isinstance(raw.get("data"), pl.DataFrame), "bad packet")
    schema_meta, lineage, summary = raw.get("schema_meta"), raw.get("lineage"), raw.get("summary")
    _expect(isinstance(schema_meta, dict), "bad packet schema")
    _expect(isinstance(lineage, list) and all(isinstance(x, str) for x in lineage), "bad packet lineage")
    _expect(summary is None or isinstance(summary, dict), "bad packet summary")
    return DataFramePacket(
        data=raw["data"],
        schema_meta={str(k): _meta_in(v) for k, v in schema_meta.items()},
        lineage=list(lineage),
        summary=None if summary is None else {str(k): _stats_in(v) for k, v in summary.items()},
    )


def decode_entry(data: bytes) -> CacheEntry:
    """A CacheEntry from a cache file's bytes. Raises wire.WireError or
    CacheCorrupt for anything that isn't a well-formed entry."""
    fields, value = wire.decode_frame(data, max_bytes=MAX_BYTES)
    _expect(fields.get("kind") == FRAME_KIND and fields.get("version") == FRAME_VERSION, "not a cache entry")
    _expect(isinstance(value, dict) and isinstance(value.get("timestamp"), str), "bad entry")
    raw_outputs = value.get("outputs")
    _expect(isinstance(raw_outputs, dict), "bad entry outputs")
    outputs: dict[str, Any] = {}
    for port, node in raw_outputs.items():
        _expect(isinstance(port, str) and isinstance(node, dict) and len(node) == 1, "bad output")
        if "packet" in node:
            outputs[port] = _packet_in(node["packet"])
        else:
            _expect("value" in node, "bad output")
            outputs[port] = node["value"]
    return CacheEntry(outputs=outputs, timestamp=value["timestamp"])


def _write_atomic(path: Path, data: bytes) -> None:
    """Write via a temporary file and a rename, so a reader (or a crash
    mid-write) never sees half an entry."""
    try:
        fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.stem}.", suffix=".tmp")
    except OSError as e:
        log.info("can't write cache entry %s: %s", path.name, e)
        return
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
        os.replace(tmp, path)
    except OSError as e:
        log.info("can't write cache entry %s: %s", path.name, e)
        _unlink(Path(tmp))


def _unlink(path: Path) -> None:
    try:
        path.unlink()
    except OSError:
        pass
