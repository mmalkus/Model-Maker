"""Create with AI: an LLM drafts a block for the block library -- its code
(function, helpers, `_example_` self-check) and its BLOCKS entry -- and the
result goes through the same parser and validator as a hand-written block.
The draft is returned as a proposed file; nothing is written until the user
saves it (library_api PUT), and only a saved block can be placed.

Three flavors share one pipeline: create (from an instruction), edit (an
existing library block plus an instruction) and fix (a placed block's code
plus its error). The model sees column names, dtypes, roles and summary
statistics of the input -- never rows (see llm/redact.py).
"""

from __future__ import annotations

import ast
import difflib
from dataclasses import dataclass, field
from typing import Any

from . import block_library as bl
from . import library_store as store
from .blocks.base import BLOCK_REGISTRY
from .llm import ColumnInfo, DraftContext, LLMProvider


class DraftError(ValueError):
    pass


@dataclass
class AIDraft:
    file: str
    block_id: str
    source: str
    explanation: str
    check: dict[str, Any]
    diff: str
    base_sha: str | None
    code: str = ""
    block: dict[str, Any] = field(default_factory=dict)


def _vocabulary() -> str:
    from .agent.catalogue import TAGS

    rules: dict[str, list[str]] = {}
    for d in BLOCK_REGISTRY.values():
        if d.hints:
            ports = ", ".join(f"{p.name}:{p.type}" for p in d.outputs)
            rules.setdefault(d.hints, []).append(f"{d.id} ({ports})")
    lines = ["Tags you may use: " + ", ".join(TAGS), "", "Decision-hint rules (and the built-in blocks whose output shape each reads):"]
    lines += [f"  - {rule}: as used by {'; '.join(blocks)}" for rule, blocks in sorted(rules.items())]
    return "\n".join(lines)


def _file_context(f: store.LibraryFile | None, exclude: str | None) -> str:
    if f is None:
        return ""
    others = [d for d in f.parsed.definitions if d.id != exclude]
    if not others and not f.source.strip():
        return ""
    return (
        f"The block goes into the library file {f.name}, which already holds these blocks and helpers -- "
        "reuse its helpers rather than duplicating them, and match its style:\n```python\n" + f.source + "\n```"
    )


def _block_id_of(code: str) -> str:
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        raise DraftError(f"the drafted code doesn't parse: line {e.lineno}: {e.msg}") from None
    public = [n.name for n in tree.body if isinstance(n, ast.FunctionDef) and not n.name.startswith("_")]
    if len(public) != 1:
        raise DraftError(f"the draft must define exactly one public function, got {public or 'none'}")
    return public[0]


def draft(
    provider: LLMProvider,
    instruction: str,
    *,
    file: str | None = None,
    block: str | None = None,
    input_ports: dict[str, list[ColumnInfo]] | None = None,
    existing: tuple[str, dict[str, Any]] | None = None,
    error: str | None = None,
    previous: tuple[str, dict[str, Any]] | None = None,
    include_reference: bool = False,
) -> AIDraft:
    """Draft a block and return the file it would produce.

    `file`: the library file the block goes to (new or existing). `block`:
    an existing block in that file to change. `existing` (code, BLOCKS
    entry): a placed block's definition to fix (with `error`) when it isn't
    in the library. `previous` (code, entry): the draft being refined."""
    target = None
    if file:
        try:
            target = store.get_file(file)
        except store.LibraryError:
            target = None  # a new file
    current_code = current_meta = None
    if block:
        if target is None or target.definition(block) is None:
            raise DraftError(f"{file} has no block {block!r} to edit")
        d = target.definition(block)
        current_code, current_meta = d.code, d.meta
    if existing is not None:
        current_code, current_meta = existing
    if previous is not None:
        current_code, current_meta = previous

    context = "\n\n".join(x for x in (_file_context(target, block), _vocabulary()) if x)
    ctx = DraftContext(
        instruction=instruction,
        function_name=block or (existing and _block_id_of(existing[0])) or "",
        input_ports=input_ports or {},
        existing_code=current_code,
        existing_block=current_meta,
        error=error,
        mode="library_block",
        library_context=context,
        include_reference=include_reference,
    )
    result = provider.draft(ctx)
    code = (result.code or "").strip("\n") + "\n"
    block_id = _block_id_of(code)
    entry = dict(result.block or {})
    if current_meta is not None:
        # A change is always a new version, whatever the model wrote.
        entry["version"] = max(int(entry.get("version", 1)), int(current_meta.get("version", 1)) + (previous is None))
    entry.setdefault("version", 1)

    name = file or f"{block_id}.py"
    base_source = target.source if target is not None else ""
    try:
        source = store.upsert_block(base_source, block_id, entry, code)
    except (store.LibraryError, SyntaxError) as e:
        raise DraftError(f"couldn't merge the draft into {name}: {e}") from None
    if target is None:
        doc = (entry.get("display_name") or block_id).strip()
        source = source.replace('"""Blocks."""', f'"""{doc}."""', 1)

    taken = set(BLOCK_REGISTRY) | {d.id for f in store.list_files() if f.name != name for d in f.parsed.definitions}
    parsed = bl.validate(bl.parse_source(source, name), taken)
    check = {
        "ok": parsed.ok,
        "file_errors": parsed.file_errors,
        "errors": parsed.errors.get(block_id, []),
        "has_example": block_id in parsed.examples,
    }
    diff = "".join(
        difflib.unified_diff(base_source.splitlines(keepends=True), source.splitlines(keepends=True), fromfile=name, tofile=name)
    )
    return AIDraft(
        file=name,
        block_id=block_id,
        source=source,
        explanation=result.explanation,
        check=check,
        diff=diff,
        base_sha=target.sha if target is not None else None,
        code=code,
        block=entry,
    )
