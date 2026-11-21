"""Before a run: will an input block's output fit in memory?

A run that isn't streaming collects every block's whole output in its
worker (see run_worker._collect_lazy). When that is bigger than the memory
the machine can spare, the collect guard stops it -- but only after the
source has been read up to the limit, minutes into a big run. This module
answers the question up front, cheaply, so the app can offer Run all
(streaming) or sample mode before the run starts instead of after it fails.

Only built-in input blocks whose size can be told without reading the data
are estimated: Read Parquet (row counts from the file metadata), Stage CSV
as Parquet (the staged Parquet's metadata when it is current; otherwise the
text, below) and Read CSV. For text, a few MB from the start of the first
file are parsed with the block's own options to learn bytes in memory per
row and rows per byte of text, scaled by the total text size -- a zip's
members' sizes come from its directory, without decompressing. A gzip's
uncompressed size is not reliable (its trailer wraps at 4 GB), so a .gz
source is left unestimated. Every other block, and any estimate that fails,
is simply left out: the guard still stops a run that turns out too big."""

from __future__ import annotations

import inspect
import os
from dataclasses import dataclass
from typing import Any, Literal

from .graph import Graph
from .run_worker import _gb, collect_limit, frame_bytes

RunMode = Literal["run_all", "force_run_all", "run_all_streaming", "run_block", "run_to_here"]

# How much text, at most, is parsed to learn a source's row size.
SAMPLE_TEXT_BYTES = 4 * 1024 * 1024
# How many rows of a Parquet source are read to learn its row size.
SAMPLE_PARQUET_ROWS = 20_000
# Rows sample mode is offered with, from the warning and the advice.
SUGGESTED_SAMPLE_ROWS = 100_000
ESTIMATED_CATEGORIES = ("read_parquet", "stage_csv_parquet", "read_csv")


@dataclass
class SizeEstimate:
    rows: int
    bytes: int
    # True: rows counted (Parquet metadata); False: scaled from a sample.
    exact: bool


def lazy_consumers(graph: Graph, block_id: str) -> tuple[list[str], list[str]]:
    """(names of the blocks that take `block_id`'s output lazily, names of
    those that take it whole) -- a whole one makes even a streaming run
    collect the output."""
    lazy, whole = [], []
    for w in graph.wires.values():
        if w.from_block != block_id or w.to_block not in graph.blocks:
            continue
        to = graph.blocks[w.to_block]
        port = next((p for p in to.inputs if p.name == w.to_port), None)
        (lazy if port is not None and getattr(port, "lazy", False) else whole).append(to.name)
    return lazy, whole


def memory_advice(graph: Graph, block_id: str) -> dict[str, Any]:
    """What a block stopped for running out of memory could do instead:
    whether Run all (streaming) would take its output lazily (every block
    it feeds declares a lazy input), and the blocks that don't."""
    lazy, whole = lazy_consumers(graph, block_id)
    return {
        "can_stream": bool(lazy) and not whole,
        "whole_consumers": whole,
        "sample_rows": SUGGESTED_SAMPLE_ROWS,
    }


# The start of WorkerProcess.crash_message for a worker killed with SIGKILL.
KILLED_PHRASE = "killed by the system (signal 9)"


def is_out_of_memory_error(error: str | None) -> bool:
    """The errors the app shows memory advice for: the collect guard's, and
    a worker the system killed (see WorkerProcess.crash_message). Not any
    other crash: its message also says "likely out of memory", for the
    assistants, but a segfault isn't fixed by streaming or sampling."""
    from .run_worker import TOO_BIG_PHRASE

    return bool(error) and (TOO_BIG_PHRASE in error or KILLED_PHRASE in error)


def _in_scope(graph: Graph, runner: Any, mode: RunMode, block_id: str | None) -> list[str]:
    """The input blocks the run will read, asked of the runner itself so
    the two can't drift apart: Run all and Force run all read a source that
    was never read or whose settings changed (Runner.input_needs_read --
    Force run all forces only the blocks after the sources); a streaming run
    reads the ones Runner._streaming_to_run picks; Run block reads its block.
    Run to here never reads the sources upstream (an unread one leaves the
    block blocked), only the block itself when it is one."""
    inputs = [b for b, blk in graph.blocks.items() if blk.block_type == "input"]
    if mode in ("run_block", "run_to_here"):
        return [block_id] if block_id in inputs else []
    plan = runner.pin()
    if mode == "run_all_streaming":
        to_run = set(runner._streaming_to_run(plan)[0])
        return [b for b in inputs if b in to_run]
    return [b for b in inputs if runner.input_needs_read(b, plan)]


def preflight(graph: Graph, runner: Any, mode: RunMode, block_id: str | None = None) -> dict[str, Any]:
    """The input blocks this run would collect whole that are bigger than
    the collect limit, each with its estimated size and what to do instead.
    Nothing is flagged in sample mode (every source is capped) or with the
    guard turned off."""
    found = collect_limit()
    out: dict[str, Any] = {"limit_bytes": None, "limit_basis": None, "blocks": [], "sample_rows": SUGGESTED_SAMPLE_ROWS}
    if found is None or runner.sample_rows is not None:
        return out
    limit, basis = found
    out["limit_bytes"], out["limit_basis"] = limit, basis
    for bid in _in_scope(graph, runner, mode, block_id):
        block = graph.blocks[bid]
        lazy, whole = lazy_consumers(graph, bid)
        # A streaming run fuses a source into the lazy blocks it feeds; it
        # is collected only for a block that takes it whole, or when it
        # feeds nothing (its own output is then the result).
        if mode == "run_all_streaming" and lazy and not whole:
            continue
        est = estimate_block(block)
        if est is None or est.bytes <= limit:
            continue
        out["blocks"].append(
            {
                "id": bid,
                "name": block.name,
                "rows": est.rows,
                "bytes": est.bytes,
                "exact": est.exact,
                "size": _gb(est.bytes),
                "can_stream": bool(lazy) and not whole,
                "whole_consumers": whole,
            }
        )
    out["limit"] = _gb(limit)
    return out


def estimate_block(block: Any) -> SizeEstimate | None:
    """`block`'s output size, or None when it can't be told cheaply."""
    if block.category not in ESTIMATED_CATEGORIES or getattr(block, "definition", None) is not None:
        return None
    from .blocks.base import BLOCK_REGISTRY

    fn = BLOCK_REGISTRY[block.category].fn
    params = {
        name: p.default for name, p in inspect.signature(fn).parameters.items() if p.default is not inspect.Parameter.empty
    }
    params.update(block.params or {})
    try:
        if block.category == "read_parquet":
            return _estimate_parquet(str(params["path"]))
        if block.category == "read_csv":
            return _estimate_text([str(params["path"])], params, keep=None, positions=None)
        return _estimate_stage(params)
    except Exception:  # noqa: BLE001 -- an estimate is a hint: the run reports real errors
        return None


def _estimate_parquet(paths: str | list[str]) -> SizeEstimate | None:
    import polars as pl

    lf = pl.scan_parquet(paths)
    rows = lf.select(pl.len()).collect().item()  # metadata only
    if not rows:
        return SizeEstimate(0, 0, True)
    sample = lf.head(SAMPLE_PARQUET_ROWS).collect()
    return SizeEstimate(rows, int(frame_bytes(sample) / max(sample.height, 1) * rows), True)


def _estimate_stage(params: dict[str, Any]) -> SizeEstimate | None:
    from .blocks import library as lib

    path = str(params["path"])
    separator = _separator(params.get("separator", ","))
    has_header = bool(params.get("has_header", True))
    names = [n.strip() for n in str(params.get("column_names") or "").split(",") if n.strip()] or None
    keep = [c.strip() for c in str(params.get("columns") or "").split(",") if c.strip()] or None
    positions = None
    if keep and not has_header and any(lib._stage_is_position(c) for c in keep):
        positions, keep = lib._stage_positional(keep, names)
    files = lib._stage_files(path)
    staged = _current_stage(lib, params, path, files, separator, has_header, names, keep, positions)
    if staged:
        return _estimate_parquet(staged)
    return _estimate_text(files, params, keep=keep, positions=positions)


def _current_stage(lib, params, path, files, separator, has_header, names, keep, positions) -> list[str] | None:
    """The staged Parquet parts, when they exist and were made with the
    block's current options for the current files; else None (the run
    will stage, and the text is estimated instead)."""
    target = params.get("parquet_dir") or lib._stage_default_dir(path)
    manifest = lib._stage_read_manifest(os.path.join(target, lib._STAGE_MANIFEST))
    options: dict[str, Any] = {
        "separator": separator,
        "has_header": has_header,
        "column_names": names,
        "infer_schema_rows": int(params.get("infer_schema_rows", 10000)),
        "compression": params.get("compression", "zstd"),
    }
    if keep:
        options["columns"] = keep
    if positions:
        options["positions"] = positions
    if manifest.get("format") != lib._STAGE_FORMAT or manifest.get("options") != options:
        return None
    parts = manifest.get("parts", {})
    expected = _expected_parts(lib, files)
    if expected is None or set(expected) - set(parts):
        return None
    paths = [os.path.join(target, p) for p in expected]
    return paths if all(os.path.isfile(p) for p in paths) else None


def _expected_parts(lib, files: list[str]) -> list[str] | None:
    """The part names the files stage to, read from zip directories and
    file names only (no hashing: a changed file of the same name is still
    the same size of data)."""
    parts = []
    prefix = len(files) > 1
    for f in files:
        if f.lower().endswith(".zip"):
            # Names only: _stage_units with an empty manifest reads the zip
            # directory, never the data.
            parts += [u["part"] for u in lib._stage_units(f, {}, prefix=prefix)]
        else:
            parts.append(lib._stage_part_name(os.path.basename(f)))
    return parts or None


def _separator(sep: Any) -> str:
    sep = str(sep)
    return "\t" if sep in ("\\t", "tab", "TAB") else sep


def _text_size(f: str) -> tuple[int, str | None] | None:
    """(bytes of text the file holds, the zip member to sample) -- a zip's
    from its directory; None for a .gz, whose size can't be told. Only the
    first member is sampled: a zip whose members differ in layout (an
    acquisition file beside a performance file) gets a rougher estimate --
    a hint either way, never a reason a run can't start."""
    import zipfile

    low = f.lower()
    if low.endswith(".gz"):
        return None
    if low.endswith(".zip"):
        from .blocks.library import _STAGE_TEXT_EXTS

        with zipfile.ZipFile(f) as zf:
            members = [i for i in zf.infolist() if not i.is_dir() and not i.filename.startswith("__MACOSX/")]
            text = [i for i in members if i.filename.lower().endswith(_STAGE_TEXT_EXTS)] or members
            return sum(i.file_size for i in text), (text[0].filename if text else None)
    return os.path.getsize(f), None


def _head_text(f: str, member: str | None) -> bytes:
    """Up to SAMPLE_TEXT_BYTES from the start of the file (or zip member),
    cut back to the last whole line."""
    import zipfile

    if member is not None:
        with zipfile.ZipFile(f) as zf, zf.open(member) as fh:
            data = fh.read(SAMPLE_TEXT_BYTES)
    else:
        with open(f, "rb") as fh:
            data = fh.read(SAMPLE_TEXT_BYTES)
    if len(data) == SAMPLE_TEXT_BYTES and b"\n" in data:
        data = data[: data.rindex(b"\n") + 1]
    return data


def _estimate_text(files: list[str], params: dict[str, Any], keep, positions) -> SizeEstimate | None:
    import io

    import polars as pl

    sizes = [_text_size(f) for f in files]
    if not sizes or any(s is None for s in sizes):
        return None
    total = sum(s[0] for s in sizes)  # type: ignore[index]
    head = _head_text(files[0], sizes[0][1])  # type: ignore[index]
    if not head.strip():
        return SizeEstimate(0, 0, False)
    has_header = bool(params.get("has_header", True))
    names = [n.strip() for n in str(params.get("column_names") or "").split(",") if n.strip()] or None
    sample = pl.read_csv(
        io.BytesIO(head),
        separator=_separator(params.get("separator", ",")),
        has_header=has_header,
        new_columns=names,
        infer_schema_length=int(params.get("infer_schema_rows", 100)),
        truncate_ragged_lines=True,
    )
    if positions:
        sample = sample.select([pl.nth(p - 1).alias(n) for p, n in zip(positions, keep)])
    elif keep:
        sample = sample.select(keep)
    if sample.height == 0:
        return SizeEstimate(0, 0, False)
    data_bytes = len(head) - (head.index(b"\n") + 1 if has_header and b"\n" in head else 0)
    # Each file repeats its header: a few bytes in GBs, ignored.
    rows = int(sample.height * total / max(data_bytes, 1))
    return SizeEstimate(rows, int(frame_bytes(sample) / sample.height * rows), False)
