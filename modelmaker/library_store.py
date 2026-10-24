"""The user's block library: a folder of block files (see block_library.py
for the format) -- ~/.modelmaker/blocks by default, or wherever
MODELMAKER_BLOCK_LIBRARY points. It can be a git repo or a shared drive;
nothing here keeps state outside the files themselves apart from a cache
of the parsed listing, keyed by file mtimes.

An AI build writes the blocks it creates to drafts/<build id>.py, listed
under "AI drafts" until the user keeps (moves) or discards them.
"""

from __future__ import annotations

import ast
import hashlib
import io
import os
import re
import tempfile
import threading
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from . import block_library as bl
from .blocks.base import BLOCK_REGISTRY, BlockDefinition

LIBRARY_ENV = "MODELMAKER_BLOCK_LIBRARY"
DRAFTS_DIR = "drafts"

# Import limits (see import_preview).
MAX_FILE_BYTES = 1_000_000
MAX_UPLOAD_BYTES = 10_000_000
MAX_IMPORT_BLOCKS = 200

_NAME = re.compile(r"^(drafts/)?[A-Za-z_][A-Za-z0-9_]*\.py$")


class LibraryError(ValueError):
    pass


class ConflictError(LibraryError):
    """A save whose expected_sha no longer matches the file on disk --
    someone else saved it meanwhile."""


def library_dir() -> Path:
    raw = os.environ.get(LIBRARY_ENV)
    return Path(raw).expanduser() if raw else Path.home() / ".modelmaker" / "blocks"


def text_sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:12]


def check_name(name: str) -> str:
    """A library file name: `<identifier>.py`, or `drafts/<identifier>.py`.
    Nothing else -- no other folders, no `..`, no absolute paths."""
    name = name.replace("\\", "/")
    if not _NAME.match(name):
        raise LibraryError(f"{name!r} isn't a library file name -- use letters, digits and _, ending in .py")
    return name


@dataclass
class LibraryFile:
    name: str
    source: str
    sha: str
    parsed: bl.ParsedFile
    mtime: float

    @property
    def is_draft(self) -> bool:
        return self.name.startswith(f"{DRAFTS_DIR}/")

    def definition(self, block_id: str) -> BlockDefinition | None:
        return next((d for d in self.parsed.definitions if d.id == block_id), None)


_lock = threading.RLock()
_cache: dict[str, Any] = {"key": None, "files": [], "checked": 0.0, "root": None}
# How long a listing is trusted before the folder is stat'ed again -- the
# graph poll asks for update badges of every placed library block.
RESCAN_SECONDS = 1.0
# Bumped on every change seen in the folder (a save here, or files changing
# on disk underneath a running server).
generation = 0


def _scan_key(root: Path) -> tuple:
    if not root.is_dir():
        return ()
    paths = sorted(root.glob("*.py")) + sorted((root / DRAFTS_DIR).glob("*.py"))
    out = []
    for p in paths:
        try:
            st = p.stat()
        except OSError:
            continue
        out.append((str(p), st.st_mtime_ns, st.st_size))
    return tuple(out)


def list_files(*, force: bool = False) -> list[LibraryFile]:
    """Every library file, parsed and validated. Ids must be unique across
    the built-ins and every library file: a clash is an error on the later
    file (in name order); a clash with a built-in says it's shadowed."""
    global generation
    root = library_dir()
    with _lock:
        now = time.monotonic()
        if not force and _cache["root"] == root and now - _cache["checked"] < RESCAN_SECONDS:
            return _cache["files"]
        key = _scan_key(root)
        _cache["checked"], _cache["root"] = now, root
        if not force and key == _cache["key"]:
            return _cache["files"]
        files: list[LibraryFile] = []
        taken = set(BLOCK_REGISTRY)
        for path_str, mtime_ns, _ in key:
            path = Path(path_str)
            name = path.relative_to(root).as_posix()
            try:
                source = path.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError) as e:
                files.append(LibraryFile(name, "", "", bl.ParsedFile(name, "", [], file_errors=[str(e)]), mtime_ns / 1e9))
                continue
            parsed = bl.parse_source(source, name)
            bl.validate(parsed, set())
            for d in parsed.definitions:
                errs = parsed.errors.setdefault(d.id, [])
                if d.id in BLOCK_REGISTRY:
                    errs.insert(0, f"shadowed by built-in {d.id!r} -- rename this block")
                elif d.id in taken:
                    errs.insert(0, f"block id {d.id!r} is already used by another library file")
                taken.add(d.id)
            files.append(LibraryFile(name, source, text_sha(source), parsed, mtime_ns / 1e9))
        if _cache["key"] is not None:
            generation += 1
        _cache.update(key=key, files=files)
        return files


def reload() -> int:
    list_files(force=True)
    return generation


def get_file(name: str) -> LibraryFile:
    name = check_name(name)
    for f in list_files():
        if f.name == name:
            return f
    raise LibraryError(f"no library file {name!r}")


def find(file: str, block_id: str) -> BlockDefinition:
    d = get_file(file).definition(block_id)
    if d is None:
        raise LibraryError(f"{file} has no block {block_id!r}")
    return d


def find_by_id(block_id: str) -> tuple[LibraryFile, BlockDefinition] | None:
    for f in list_files():
        d = f.definition(block_id)
        if d is not None:
            return f, d
    return None


def library_ref(f: LibraryFile, d: BlockDefinition) -> dict[str, Any]:
    return {"file": f.name, "id": d.id, "version": d.version, "sha": d.sha}


def placeable(file: str, block_id: str) -> tuple[dict[str, Any], dict[str, Any]]:
    """(embedded definition, library_ref) for placing a library block --
    refused while the block has errors or unmet requirements."""
    f = get_file(file)
    d = f.definition(block_id)
    if d is None:
        raise LibraryError(f"{file} has no block {block_id!r}")
    problems = list(f.parsed.file_errors) + list(f.parsed.errors.get(block_id, []))
    if problems:
        raise LibraryError(f"{block_id} can't be placed: {'; '.join(problems)}")
    unmet = bl.check_requirements(d)
    if unmet:
        raise LibraryError(f"{block_id} needs {', '.join(unmet)} -- install it first")
    return d.embedded(), library_ref(f, d)


# ---- writing -------------------------------------------------------------


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(text)
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def save_file(name: str, source: str, *, expected_sha: str | None = None, create: bool = False) -> LibraryFile:
    """Write a whole file. An existing file needs the `expected_sha` the
    editor opened it with: a mismatch (someone saved meanwhile) raises
    ConflictError instead of the last writer winning. `create` refuses to
    overwrite an existing file."""
    name = check_name(name)
    path = library_dir() / name
    with _lock:
        if path.exists():
            if create:
                raise LibraryError(f"{name} already exists")
            current = text_sha(path.read_text(encoding="utf-8"))
            if expected_sha is None or expected_sha != current:
                raise ConflictError(f"{name} changed on disk since it was opened -- reload it and re-apply your edit")
        _atomic_write(path, source)
        list_files(force=True)
        return get_file(name)


def delete_file(name: str, *, expected_sha: str | None = None) -> None:
    name = check_name(name)
    path = library_dir() / name
    with _lock:
        if not path.exists():
            raise LibraryError(f"no library file {name!r}")
        if expected_sha is not None and expected_sha != text_sha(path.read_text(encoding="utf-8")):
            raise ConflictError(f"{name} changed on disk since it was opened")
        path.unlink()
        list_files(force=True)


def _node_span(source: str, node: ast.AST) -> tuple[int, int]:
    """Character offsets [start, end) of a top-level statement, decorators
    included, whole lines."""
    lines = source.splitlines(keepends=True)
    first = node.decorator_list[0].lineno if getattr(node, "decorator_list", None) else node.lineno
    start = sum(len(line) for line in lines[: first - 1])
    end = sum(len(line) for line in lines[: node.end_lineno])
    return start, end


def _offset(source: str, lineno: int, col: int) -> int:
    lines = source.splitlines(keepends=True)
    # ast column offsets count UTF-8 bytes, not characters.
    return sum(len(line) for line in lines[: lineno - 1]) + len(lines[lineno - 1].encode()[:col].decode())


def _top_level(tree: ast.Module) -> dict[str, ast.stmt]:
    out = {}
    for node in tree.body:
        for name in bl._top_level_names(node):
            out[name] = node
    return out


NEW_FILE_TEMPLATE = '''"""{doc}"""

import polars as pl

BLOCKS = {{}}
'''


def upsert_block(source: str, block_id: str, meta: dict[str, Any], code: str) -> str:
    """`source` with one block added or replaced: its BLOCKS entry and the
    top-level definitions in `code` (the function, plus any helpers or
    `_example_` it brings). Only those spans are rewritten -- every other
    entry, function and comment in the file stays as it was."""
    if not source.strip():
        source = NEW_FILE_TEMPLATE.format(doc="Blocks.")
    tree = ast.parse(source)
    blocks_node = next((n for n in tree.body if bl._top_level_names(n) == ["BLOCKS"]), None)
    if blocks_node is None or not isinstance(blocks_node.value, ast.Dict):  # type: ignore[attr-defined]
        raise LibraryError("the file has no BLOCKS = {...} dict to add the block to")
    d: ast.Dict = blocks_node.value  # type: ignore[attr-defined]
    entry_text = bl.format_literal(meta, 4)
    edits: list[tuple[int, int, str]] = []
    key_idx = next((i for i, k in enumerate(d.keys) if isinstance(k, ast.Constant) and k.value == block_id), None)
    if key_idx is not None:
        v = d.values[key_idx]
        edits.append((_offset(source, v.lineno, v.col_offset), _offset(source, v.end_lineno, v.end_col_offset), entry_text))
    else:
        close = _offset(source, d.end_lineno, d.end_col_offset) - 1  # the closing brace
        before = source[:close].rstrip()
        sep = "" if before.endswith(("{", ",")) else ","
        insert_at = len(before)
        edits.append((insert_at, close, f"{sep}\n    \"{block_id}\": {entry_text},\n"))

    top = _top_level(tree)
    new_tree = ast.parse(code)
    appended: list[str] = []
    for node in new_tree.body:
        names = bl._top_level_names(node)
        if not names or names == ["BLOCKS"]:
            continue
        start, end = _node_span(code, node)
        text = code[start:end].rstrip("\n") + "\n"
        existing = top.get(names[0])
        if existing is not None:
            s, e = _node_span(source, existing)
            edits.append((s, e, text))
        else:
            appended.append(text)
    edits.sort(key=lambda t: t[0], reverse=True)
    out = source
    for s, e, text in edits:
        out = out[:s] + text + out[e:]
    if appended:
        out = out.rstrip("\n") + "\n\n\n" + "\n\n".join(appended)
    return out


def remove_block(source: str, block_id: str) -> str:
    """`source` without one block: its BLOCKS entry, its function and its
    `_example_`. Helpers stay (another block may use them)."""
    tree = ast.parse(source)
    blocks_node = next((n for n in tree.body if bl._top_level_names(n) == ["BLOCKS"]), None)
    d: ast.Dict = blocks_node.value  # type: ignore[union-attr]
    idx = next((i for i, k in enumerate(d.keys) if isinstance(k, ast.Constant) and k.value == block_id), None)
    if idx is None:
        raise LibraryError(f"no block {block_id!r} in this file")
    edits = []
    k, v = d.keys[idx], d.values[idx]
    start = _offset(source, k.lineno, k.col_offset)
    end = _offset(source, v.end_lineno, v.end_col_offset)
    rest = source[end:]
    m = re.match(r"\s*,", rest)
    if m:
        end += m.end()
    line_start = source.rfind("\n", 0, start) + 1
    if not source[line_start:start].strip():
        start = line_start
        nl = source.find("\n", end)
        if nl != -1 and not source[end:nl].strip():
            end = nl + 1
    edits.append((start, end))
    top = _top_level(tree)
    for name in (block_id, f"_example_{block_id}"):
        if name in top:
            edits.append(_node_span(source, top[name]))
    for s, e in sorted(edits, reverse=True):
        source = source[:s] + source[e:]
    return re.sub(r"\n{4,}", "\n\n\n", source)


def add_block(file: str, meta: dict[str, Any], block_id: str, code: str, *, doc: str = "Blocks.") -> LibraryFile:
    """Add (or replace) one block in a library file, creating the file if
    needed -- the write behind Create with AI's Save and an agent build's
    draft blocks."""
    name = check_name(file)
    path = library_dir() / name
    with _lock:
        source = path.read_text(encoding="utf-8") if path.exists() else NEW_FILE_TEMPLATE.format(doc=doc)
        expected = text_sha(source) if path.exists() else None
        return save_file(name, upsert_block(source, block_id, meta, code), expected_sha=expected)


def delete_block(file: str, block_id: str) -> LibraryFile:
    with _lock:
        f = get_file(file)
        return save_file(f.name, remove_block(f.source, block_id), expected_sha=f.sha)


def move_block(src_file: str, block_id: str, dst_file: str) -> LibraryFile:
    """Move one block (and the helpers it uses) to another file -- the AI
    drafts' Keep action. The source file is deleted once it's empty."""
    with _lock:
        src = get_file(src_file)
        d = src.definition(block_id)
        if d is None:
            raise LibraryError(f"{src_file} has no block {block_id!r}")
        example = _example_source(src.source, block_id)
        code = d.code + ("\n\n" + example if example else "")
        out = add_block(dst_file, d.meta, block_id, code)
        remaining = remove_block(src.source, block_id)
        if not bl.parse_source(remaining, src.name).definitions:
            delete_file(src.name)
        else:
            save_file(src.name, remaining, expected_sha=src.sha)
        return out


def duplicate(source_def: BlockDefinition, new_id: str, dst_file: str, example: str | None = None) -> LibraryFile:
    """Copy a block (a built-in, or a library block) into a library file
    under a new id, as version 1 -- "Duplicate to library", the way to
    change a built-in."""
    if not new_id.isidentifier() or new_id.startswith("_"):
        raise LibraryError(f"{new_id!r} isn't a valid block id")
    taken = set(BLOCK_REGISTRY) | {d.id for f in list_files() for d in f.parsed.definitions}
    if new_id in taken:
        raise LibraryError(f"block id {new_id!r} is already taken")
    code = re.sub(rf"\bdef {re.escape(source_def.id)}\(", f"def {new_id}(", source_def.code, count=1)
    meta = {**source_def.meta, "version": 1}
    meta["display_name"] = f"{source_def.display_name} (copy)"
    if example:
        code += "\n\n" + example.replace(f"_example_{source_def.id}", f"_example_{new_id}").replace(f"{source_def.id}(", f"{new_id}(")
    return add_block(dst_file, meta, new_id, code)


def _example_source(source: str, block_id: str) -> str | None:
    tree = ast.parse(source)
    node = _top_level(tree).get(f"_example_{block_id}")
    if node is None:
        return None
    s, e = _node_span(source, node)
    return source[s:e]


def extract(file: str, ids: list[str] | None = None) -> str:
    """A standalone file holding just these blocks (all, if None), the
    helpers they use and their examples -- for exporting single blocks."""
    f = get_file(file)
    defs = [d for d in f.parsed.definitions if ids is None or d.id in ids]
    if not defs:
        raise LibraryError("nothing to export")
    out = NEW_FILE_TEMPLATE.format(doc=f.parsed.docstring or "Blocks.")
    for d in defs:
        example = _example_source(f.source, d.id)
        out = upsert_block(out, d.id, d.meta, d.code + ("\n\n" + example if example else ""))
    return out


def export_zip(names: list[str]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for name in names:
            f = get_file(name)
            zf.writestr(f.name.replace("/", "_") if f.is_draft else f.name, f.source)
    return buf.getvalue()


# ---- import --------------------------------------------------------------


def read_upload(filename: str, data: bytes) -> list[tuple[str, str]]:
    """The (file name, text) pairs in an upload: one .py, or a .zip of .py
    files with plain names. Enforces the size and name limits; never
    executes anything."""
    if len(data) > MAX_UPLOAD_BYTES:
        raise LibraryError(f"upload is over {MAX_UPLOAD_BYTES // 1_000_000} MB")
    lower = filename.lower()
    if lower.endswith(".py"):
        pairs = [(Path(filename).name, data)]
    elif lower.endswith(".zip"):
        pairs = []
        try:
            zf = zipfile.ZipFile(io.BytesIO(data))
        except zipfile.BadZipFile:
            raise LibraryError("not a valid .zip") from None
        for info in zf.infolist():
            if info.is_dir():
                continue
            n = info.filename
            if "/" in n or "\\" in n or n.startswith(".") or ".." in n or os.path.isabs(n):
                raise LibraryError(f"zip entry {n!r} must be a plain file name (no folders)")
            if not n.endswith(".py"):
                continue
            if info.file_size > MAX_FILE_BYTES:
                raise LibraryError(f"{n} is over {MAX_FILE_BYTES // 1_000_000} MB")
            pairs.append((n, zf.read(info)))
    else:
        raise LibraryError("upload a .py file or a .zip of .py files")
    out = []
    for name, raw in pairs:
        if len(raw) > MAX_FILE_BYTES:
            raise LibraryError(f"{name} is over {MAX_FILE_BYTES // 1_000_000} MB")
        try:
            out.append((name, raw.decode("utf-8")))
        except UnicodeDecodeError:
            raise LibraryError(f"{name} isn't UTF-8 text") from None
    return out


def import_preview(files: list[tuple[str, str]]) -> list[dict[str, Any]]:
    """Per file and block: validation (literal checks only -- nothing is
    defined or run), requirements, and whether the id already exists."""
    existing = {d.id: (f.name, d) for f in list_files() for d in f.parsed.definitions}
    out = []
    total = 0
    for name, text in files:
        parsed = bl.parse_source(text, name)
        bl.validate(parsed, set(), define=False)
        blocks = []
        for d in parsed.definitions:
            total += 1
            clash = None
            if d.id in BLOCK_REGISTRY:
                clash = {"kind": "builtin"}
            elif d.id in existing:
                f, cur = existing[d.id]
                clash = {"kind": "library", "file": f, "version": cur.version}
            blocks.append(
                {
                    "id": d.id,
                    "display_name": d.display_name,
                    "version": d.version,
                    "summary": d.summary,
                    "errors": parsed.errors.get(d.id, []),
                    "unmet_requirements": bl.check_requirements(d),
                    "clash": clash,
                    "code": d.code,
                }
            )
        out.append({"file": name, "file_errors": parsed.file_errors, "blocks": blocks})
    if total > MAX_IMPORT_BLOCKS:
        raise LibraryError(f"at most {MAX_IMPORT_BLOCKS} blocks per import (this has {total})")
    return out


def import_commit(files: list[tuple[str, str]], actions: dict[str, str]) -> list[str]:
    """Import blocks. `actions` maps "<file>:<id>" to "add" (the default for
    a new id), "overwrite", "rename" (as <id>_2, _3, ...) or "skip". A block
    that clashes and has no action is skipped. Each imported block goes to
    a library file of the uploaded file's name (merged into it when it
    exists). Returns the ids written."""
    preview = {(p["file"], b["id"]): b for p in import_preview(files) for b in p["blocks"]}
    written = []
    for name, text in files:
        parsed = bl.parse_source(text, name)
        target = check_name(Path(name).name)
        for d in parsed.definitions:
            info = preview[(name, d.id)]
            action = actions.get(f"{name}:{d.id}") or ("add" if info["clash"] is None else "skip")
            if action == "skip" or info["clash"] and info["clash"]["kind"] == "builtin" and action != "rename":
                continue
            block_id = d.id
            code = d.code
            example = _example_source(text, d.id)
            if action == "rename":
                taken = set(BLOCK_REGISTRY) | {x.id for f in list_files() for x in f.parsed.definitions}
                n = 2
                while f"{d.id}_{n}" in taken:
                    n += 1
                block_id = f"{d.id}_{n}"
                code = re.sub(rf"\bdef {d.id}\(", f"def {block_id}(", code, count=1)
                if example:
                    example = example.replace(f"_example_{d.id}", f"_example_{block_id}").replace(f"{d.id}(", f"{block_id}(")
            elif action == "overwrite" and info["clash"]:
                if info["clash"]["file"] != target:
                    delete_block(info["clash"]["file"], d.id)
            add_block(target, d.meta, block_id, code + ("\n\n" + example if example else ""), doc=parsed.docstring or "Blocks.")
            written.append(block_id)
    return written
