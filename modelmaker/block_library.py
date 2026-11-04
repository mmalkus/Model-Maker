"""Block files: parse, validate and load them.

A block file is a plain Python module:

    \"\"\"What this file's blocks are about (the group heading in the library).\"\"\"

    import polars as pl

    BLOCKS = {
        "my_block": {"display_name": "My block", "inputs": [...], "outputs": [...], ...},
    }

    def my_block(df: pl.DataFrame, threshold: float = 0.5) -> pl.DataFrame:
        \"\"\"First line is the summary; the rest describes the block.\"\"\"
        ...

    def _helper(...): ...          # private: shared code, not a block
    def _example_my_block(): ...   # optional self-check, run on request

`BLOCKS` is read with ast.literal_eval and never executed, so listing a
library or previewing an import runs no code. See docs/block-authoring.md
for every key; `validate` below checks them all.
"""

from __future__ import annotations

import __future__
import ast
import hashlib
import importlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .blocks.base import (
    BLOCK_REGISTRY,
    BLOCK_TYPES,
    DISPLAY_KINDS,
    FIELD_KINDS,
    PORT_TYPES,
    BlockDefinition,
)

BUILTIN_DIR = Path(__file__).resolve().parent / "blocks"
_NOT_BLOCK_MODULES = {"base"}  # plus every _-prefixed module (__init__, _template)
TEMPLATE_PATH = BUILTIN_DIR / "_template.py"

# Top-level imports a block file may have; everything else is imported
# inside the function that needs it, so each emitted function stays a
# self-contained unit in a compiled script.
ALLOWED_MODULE_IMPORTS = {"polars", "__future__"}

BLOCK_KEYS = {
    "display_name", "version", "block_type", "group", "tags", "requires", "inputs", "outputs",
    "form", "meta", "hints", "display", "renamed", "renamed_ports",
}
PORT_KEYS = {"name", "type", "required", "lazy", "aggregate"}
FIELD_KEYS = {"key", "label", "kind", "placeholder", "step", "options", "auto_role"}
META_KEYS = {"inherit", "new_role", "roles"}
ROLE_RULE_KEYS = {"column", "column_param", "column_suffix", "fallback", "role", "if_taken", "displaced", "new_only"}

# Params the engine fills in itself (see runner.run_block) -- never part of a
# params form, and allowed in a signature without a form field.
INJECTED_PARAMS = frozenset({"output_dir", "block_id", "_iteration_index"})


@dataclass
class ParsedFile:
    """One block file, parsed (never executed). `errors` is per block id;
    `file_errors` are problems with the file as a whole."""

    origin: str
    docstring: str
    definitions: list[BlockDefinition]
    errors: dict[str, list[str]] = field(default_factory=dict)
    file_errors: list[str] = field(default_factory=list)
    examples: dict[str, str] = field(default_factory=dict)  # block id -> example function name

    @property
    def ok(self) -> bool:
        return not self.file_errors and not any(self.errors.values())


def _sha(meta: dict[str, Any], code: str) -> str:
    return hashlib.sha256((json.dumps(meta, sort_keys=True) + "\n" + code).encode()).hexdigest()[:12]


def _top_level_names(node: ast.stmt) -> list[str]:
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        return [node.name]
    if isinstance(node, ast.Assign):
        return [t.id for t in node.targets if isinstance(t, ast.Name)]
    if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
        return [node.target.id]
    return []


def _referenced(node: ast.AST) -> set[str]:
    return {n.id for n in ast.walk(node) if isinstance(n, ast.Name)}


def parse_source(source: str, origin: str, kind: str = "library") -> ParsedFile:
    """Parse a block file's text into definitions. Runs no code: `BLOCKS` is
    literal_eval'd and every function is only located, not defined."""
    try:
        tree = ast.parse(source)
    except SyntaxError as e:
        return ParsedFile(origin, "", [], file_errors=[f"line {e.lineno}: {e.msg}"])
    parsed = ParsedFile(origin, ast.get_docstring(tree) or "", [])

    blocks_node = None
    top: dict[str, ast.stmt] = {}
    order: list[ast.stmt] = []
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            mods = [a.name for a in node.names] if isinstance(node, ast.Import) else [node.module or ""]
            for mod in mods:
                if mod.split(".")[0] not in ALLOWED_MODULE_IMPORTS:
                    parsed.file_errors.append(
                        f"line {node.lineno}: import {mod!r} at module level -- import it inside the function that uses it"
                    )
            continue
        names = _top_level_names(node)
        if names == ["BLOCKS"]:
            blocks_node = node
            continue
        for name in names:
            top[name] = node
        if names:
            order.append(node)

    if blocks_node is None:
        parsed.file_errors.append("no BLOCKS = {...} found")
        return parsed
    try:
        blocks = ast.literal_eval(blocks_node.value)  # type: ignore[attr-defined]
    except (ValueError, SyntaxError, TypeError):
        parsed.file_errors.append(f"line {blocks_node.lineno}: BLOCKS must be a plain literal (no names, calls or expressions)")
        return parsed
    if not isinstance(blocks, dict) or not all(isinstance(k, str) and isinstance(v, dict) for k, v in blocks.items()):
        parsed.file_errors.append(f"line {blocks_node.lineno}: BLOCKS must map block ids to dicts")
        return parsed

    public_fns = [
        n.name for n in order if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and not n.name.startswith("_")
    ]
    for name in public_fns:
        if name not in blocks:
            parsed.file_errors.append(f"function {name}() has no BLOCKS entry -- add one, or prefix it with _ to make it a helper")

    for block_id, meta in blocks.items():
        fn_node = top.get(block_id)
        if not isinstance(fn_node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            parsed.errors[block_id] = [f"BLOCKS entry {block_id!r} has no function def {block_id}(...)"]
            continue
        # The function plus every helper it reaches, in file order.
        needed: set[str] = set()
        stack = [block_id]
        while stack:
            name = stack.pop()
            if name in needed:
                continue
            needed.add(name)
            stack.extend(r for r in _referenced(top[name]) if r in top and r not in needed and not r.startswith("_example_"))
        segments = []
        for node in order:
            if any(n in needed for n in _top_level_names(node)):
                start = node.decorator_list[0].lineno if getattr(node, "decorator_list", None) else node.lineno
                lines = source.splitlines()[start - 1 : node.end_lineno]
                segments.append("\n".join(lines))
        code = "\n\n\n".join(segments) + "\n"
        parsed.definitions.append(
            BlockDefinition(
                id=block_id,
                meta=meta,
                code=code,
                source="builtin" if kind == "builtin" else "library",
                origin=origin,
                sha=_sha(meta, code),
                description=ast.get_docstring(fn_node) or "",
            )
        )
        if f"_example_{block_id}" in top:
            parsed.examples[block_id] = f"_example_{block_id}"
    return parsed


# ---- where block code may run ---------------------------------------------
#
# Defining a library block's function executes its module-level code: the
# helpers and constants it uses, its decorators, its default expressions.
# That only ever happens in a task process (see run_worker.py), which
# worker_pool.worker_main marks with `enter_task_process()` before it runs
# anything; everywhere else -- the API process, the CLI -- the block's
# signature comes from the AST (BlockDefinition.signature) and running it
# means submitting a task to an executor (see try_call / run_examples).
# This flag is what turns a slip back into executing code here into a loud
# error instead of a silent one.

_IN_TASK_PROCESS = False


class BlockCodeNotAllowed(RuntimeError):
    """Library block code was about to be executed outside a task process."""


def enter_task_process() -> None:
    """Mark this process as a task process: one that exists to run block
    code (called by worker_pool.worker_main, never by the API process)."""
    global _IN_TASK_PROCESS
    _IN_TASK_PROCESS = True


def in_task_process() -> bool:
    return _IN_TASK_PROCESS


def exec_block_source(source: str, filename: str, module_name: str) -> dict[str, Any]:
    """Execute block code (a definition's code, or a whole block file) in a
    fresh namespace that has `pl`, and return the namespace. The one place
    block code is executed; refused outside a task process."""
    if not _IN_TASK_PROCESS:
        raise BlockCodeNotAllowed(
            f"refusing to execute block code ({filename}) outside a task process -- "
            "read its signature from the AST (BlockDefinition.signature) or run it through an executor"
        )
    import polars as pl

    ns: dict[str, Any] = {"pl": pl, "__name__": module_name}
    flags = __future__.annotations.compiler_flag
    exec(compile(source, filename, "exec", flags=flags, dont_inherit=True), ns)  # block code, in a task process
    return ns


_COMPILED: dict[tuple[str, str], Any] = {}


def compile_fn(block_id: str, code: str):
    """Define (never call) a definition's function from its code -- in a
    task process only (see exec_block_source). Cached by (id, code)."""
    key = (block_id, code)
    if key not in _COMPILED:
        ns = exec_block_source(code, f"<block {block_id}>", f"modelmaker_block_{block_id}")
        fn = ns.get(block_id)
        if not callable(fn):
            raise ValueError(f"block code doesn't define {block_id}()")
        _COMPILED[key] = fn
    return _COMPILED[key]


def _known_tags() -> set[str]:
    from .agent.catalogue import TAGS

    return set(TAGS)


def _known_hint_rules() -> set[str]:
    from .agent.hint_rules import RULES

    return set(RULES)


def validate_definition(d: BlockDefinition, *, signature: bool = True) -> list[str]:
    """Every problem with one definition, as short messages. `signature`
    also checks the form, ports and renames against the function's
    signature -- read from the AST, so validating never runs the block's
    code (see signature.py); off, only the literal is checked."""
    from .packet import ColumnRole

    errors: list[str] = []
    m = d.meta
    if d.id.startswith("_") or not d.id.isidentifier():
        errors.append(f"block id {d.id!r} must be a public Python identifier")
    unknown = set(m) - BLOCK_KEYS
    if unknown:
        errors.append(f"unknown key(s) {sorted(unknown)}")
    if m.get("block_type", "standard") not in BLOCK_TYPES:
        errors.append(f"block_type must be one of {sorted(BLOCK_TYPES)}")
    if not isinstance(m.get("version", 1), int):
        errors.append("version must be an integer")
    for side in ("inputs", "outputs"):
        ports = m.get(side, [])
        if not isinstance(ports, list) or not all(isinstance(p, dict) and "name" in p for p in ports):
            errors.append(f"{side} must be a list of {{'name': ..., 'type': ...}}")
            continue
        names = [p["name"] for p in ports]
        if len(set(names)) != len(names):
            errors.append(f"duplicate {side[:-1]} port names")
        for p in ports:
            if set(p) - PORT_KEYS:
                errors.append(f"port {p['name']!r}: unknown key(s) {sorted(set(p) - PORT_KEYS)}")
            if p.get("type", "dataframe") not in PORT_TYPES:
                errors.append(f"port {p['name']!r}: unknown type {p.get('type')!r}")
            if p.get("aggregate") and (side == "inputs" or p.get("type", "dataframe") != "dataframe"):
                errors.append(f"port {p['name']!r}: only a dataframe output can be aggregate")
    for f in m.get("form", []):
        if not isinstance(f, dict) or "key" not in f:
            errors.append("form entries need a 'key'")
            continue
        if set(f) - FIELD_KEYS:
            errors.append(f"form field {f['key']!r}: unknown key(s) {sorted(set(f) - FIELD_KEYS)}")
        if f.get("kind", "text") not in FIELD_KINDS:
            errors.append(f"form field {f['key']!r}: unknown kind {f.get('kind')!r}")
    tags = set(m.get("tags", ())) - _known_tags()
    if tags:
        errors.append(f"unknown tag(s) {sorted(tags)}")
    if m.get("hints") is not None and m["hints"] not in _known_hint_rules():
        errors.append(f"unknown hints rule {m['hints']!r}")
    if m.get("display") is not None:
        if m["display"] not in DISPLAY_KINDS:
            errors.append(f"display must be one of {sorted(DISPLAY_KINDS)}")
        elif not m.get("outputs"):
            errors.append("a display block needs an output port to show")
    meta = m.get("meta") or {}
    if set(meta) - META_KEYS:
        errors.append(f"meta: unknown key(s) {sorted(set(meta) - META_KEYS)}")
    roles = {r.value for r in ColumnRole}
    if meta.get("new_role", "unassigned") not in roles:
        errors.append(f"meta.new_role must be one of {sorted(roles)}")
    for rule in meta.get("roles", []):
        if set(rule) - ROLE_RULE_KEYS:
            errors.append(f"meta.roles: unknown key(s) {sorted(set(rule) - ROLE_RULE_KEYS)}")
        for k in ("role", "if_taken", "displaced"):
            if k in rule and rule[k] not in roles:
                errors.append(f"meta.roles: {k} {rule[k]!r} isn't a role")
        if sum(k in rule for k in ("column", "column_param", "column_suffix")) != 1:
            errors.append("meta.roles: each rule needs exactly one of column / column_param / column_suffix")
    for req in m.get("requires", ()):
        if not isinstance(req, str) or not re.match(r"^[A-Za-z0-9_.\-]+", req):
            errors.append(f"requires: {req!r} isn't a pip requirement")
    if not signature:
        return errors
    from .signature import SignatureError, undefined_definition_names

    try:
        sig = d.signature
    except SignatureError as e:
        return errors + [str(e)]
    errors += undefined_definition_names(d.code, d.id)
    params = sig.names
    port_names = {p.name for p in d.inputs}
    for f in d.form:
        if f.key not in params:
            errors.append(f"form field {f.key!r} isn't a parameter of {d.id}()")
        elif f.key in port_names:
            errors.append(f"form field {f.key!r} is an input port, not a param")
    missing_ports = [p for p in port_names if p not in params]
    if missing_ports:
        errors.append(f"input port(s) {missing_ports} aren't parameters of {d.id}()")
    for old, new in d.renamed.items():
        if new not in params:
            errors.append(f"renamed: {old!r} -> {new!r}, but {new!r} isn't a parameter")
    unknown_defaults = {p.name for p in sig.params if p.has_default and not p.literal_default}
    for name in sorted(unknown_defaults & _meta_param_refs(d.metadata)):
        errors.append(
            f"param {name!r} is used by a meta rule, so its default must be a literal "
            f"(a number, string, bool, None, or a list/tuple/dict of those), not {sig.get(name).default_source}"
        )
    return errors


def _meta_param_refs(meta: dict[str, Any]) -> set[str]:
    """The params a block's declarative metadata rules read (see
    metadata_transforms.schema_diff), which therefore need a known default."""
    refs: set[str] = set()
    inherit = meta.get("inherit")
    if isinstance(inherit, dict) and isinstance(inherit.get("columns_param"), str):
        refs.add(inherit["columns_param"])
    for rule in meta.get("roles", []) or []:
        if not isinstance(rule, dict):
            continue
        if isinstance(rule.get("column_param"), str):
            refs.add(rule["column_param"])
        if isinstance(rule.get("fallback"), str):  # a str.format_map template over the params
            refs |= set(re.findall(r"{([A-Za-z_][A-Za-z0-9_]*)", rule["fallback"]))
    return refs


def validate(parsed: ParsedFile, taken_ids: set[str] = frozenset(), *, signature: bool = True) -> ParsedFile:
    """Fill in `parsed.errors` for every definition (in place; returns it).
    `taken_ids` are ids already claimed elsewhere (built-ins, other files)."""
    seen: set[str] = set()
    for d in parsed.definitions:
        errs = parsed.errors.setdefault(d.id, [])
        if d.id in taken_ids or d.id in seen:
            errs.append(f"block id {d.id!r} is already taken")
        seen.add(d.id)
        errs.extend(validate_definition(d, signature=signature))
    return parsed


def check_requirements(d: BlockDefinition) -> list[str]:
    """The block's `requires` entries that aren't satisfied by what's
    installed. Uses package metadata only -- nothing is imported."""
    from importlib import metadata

    unmet = []
    for req in d.requires:
        try:
            from packaging.requirements import Requirement

            r = Requirement(req)
            name, spec = r.name, r.specifier
        except Exception:  # noqa: BLE001 -- packaging missing or an odd string: presence check only
            name, spec = re.split(r"[<>=!~ ;\[]", req, maxsplit=1)[0], None
        try:
            version = metadata.version(name)
        except metadata.PackageNotFoundError:
            unmet.append(req)
            continue
        if spec is not None and version not in spec:
            unmet.append(req)
    return unmet


def load_builtin() -> dict[str, BlockDefinition]:
    """Parse every module in modelmaker/blocks/ and register its blocks in
    BLOCK_REGISTRY. The modules are also imported normally, so a built-in's
    function is the real module attribute (picklable, testable, with real
    tracebacks). A broken built-in is a bug in this package: it raises."""
    if BLOCK_REGISTRY:
        return BLOCK_REGISTRY
    loaded: dict[str, BlockDefinition] = {}
    for path in sorted(BUILTIN_DIR.glob("*.py")):
        if path.stem in _NOT_BLOCK_MODULES or path.stem.startswith("_"):
            continue
        module = importlib.import_module(f"modelmaker.blocks.{path.stem}")
        parsed = parse_source(path.read_text(encoding="utf-8"), f"modelmaker/blocks/{path.name}", kind="builtin")
        if parsed.file_errors:
            raise RuntimeError(f"{path.name}: {'; '.join(parsed.file_errors)}")
        for d in parsed.definitions:
            d._fn = getattr(module, d.id)
            if d.id in loaded:
                raise RuntimeError(f"{path.name}: block id {d.id!r} is defined twice")
            loaded[d.id] = d
    # Validation needs the hint table and tag vocabulary (agent package),
    # which import this registry -- so register first, then check.
    BLOCK_REGISTRY.update(loaded)
    return BLOCK_REGISTRY


def check_builtin() -> dict[str, list[str]]:
    """Validation errors for every built-in (empty lists when all is well);
    run by the test suite and `modelmaker check-blocks`."""
    load_builtin()
    return {bid: validate_definition(d) for bid, d in BLOCK_REGISTRY.items()}


def definition_from_embedded(raw: dict[str, Any], origin: str | None = None) -> BlockDefinition:
    """A definition from the copy a project embeds (see BlockDefinition.embedded)."""
    return BlockDefinition(
        id=raw["id"],
        meta=raw["meta"],
        code=raw["code"],
        source="library",
        origin=origin,
        sha=raw.get("sha") or _sha(raw["meta"], raw["code"]),
        description=raw.get("description", ""),
    )


_EMBEDDED: dict[str, BlockDefinition] = {}


def definition_for(block) -> BlockDefinition:
    """The definition a placed block runs: its embedded copy (a library
    block), else the built-in of its category."""
    raw = getattr(block, "definition", None)
    if raw:
        key = raw.get("sha") or _sha(raw["meta"], raw["code"])
        d = _EMBEDDED.get(key)
        if d is None or d.id != raw["id"]:
            d = _EMBEDDED[key] = definition_from_embedded(raw)
        return d
    try:
        return BLOCK_REGISTRY[block.category]
    except KeyError:
        raise KeyError(f"unknown block {block.category!r}") from None


# ---- running block code on request -----------------------------------------
#
# Examples ("Try it", `modelmaker check-blocks`) and trial runs on a canvas
# block's output execute block code, so they are tasks like any block run:
# submitted to an executor (by default the one the Runner uses, see
# runner.EXECUTOR), which runs them in a single-use task process started
# without this process's secrets, and hands back only data (execution/
# wire.py). A timeout stops a hanging one; a crashing one can't take this
# process down.

EXAMPLE_TIMEOUT = 30.0
CRASHED = "the process running the block crashed"


def _default_executor():
    from . import runner

    return runner.EXECUTOR


def _run_task(entry: str, args: tuple[Any, ...], timeout: float, executor=None) -> tuple[bool, Any]:
    """Run one run_worker entry point through `executor` with a timeout:
    (True, its value) or (False, a one-line error)."""
    import threading

    from .execution import Task, TaskCancelled

    executor = executor or _default_executor()
    stop = threading.Event()
    timer = threading.Timer(timeout, stop.set)
    timer.daemon = True
    timer.start()
    try:
        (result,) = executor.wait([executor.submit(Task(entry, args))], stop, max_concurrent=1)
    except TaskCancelled:
        return False, f"timed out after {timeout:.0f}s"
    finally:
        timer.cancel()
    if result.ok:
        return True, result.value
    if result.crashed:
        return False, f"{CRASHED} ({result.error})" if result.error else CRASHED
    return False, result.error or "the block failed without an error message"


def _example_report(raw: Any) -> dict[str, dict[str, Any]]:
    """The examples task's report, reshaped to exactly what callers read --
    it comes from a process that ran the file's code, so its shape isn't
    taken on trust (the wire format already guarantees it's only data)."""
    if not isinstance(raw, dict):
        return {"__file__": {"ok": False, "error": "the examples run returned no report"}}
    out: dict[str, dict[str, Any]] = {}
    for key, item in raw.items():
        if not isinstance(key, str) or not isinstance(item, dict):
            continue
        ok = item.get("ok")
        entry: dict[str, Any] = {
            "ok": ok if ok in (True, False, None) else False,
            "error": item["error"][:20_000] if isinstance(item.get("error"), str) else None,
        }
        if "line" in item:
            line = item["line"]
            entry["line"] = line if type(line) is int else None
        out[key] = entry
    return out


def run_examples(
    source: str, origin: str, ids: list[str] | None = None, timeout: float = EXAMPLE_TIMEOUT, executor=None
) -> dict[str, dict[str, Any]]:
    """Run a file's `_example_<id>` self-checks as a task: {block id: {"ok":
    True|False|None (no example), "error", "line"}}; a file that fails to
    run at all reports under "__file__" (see run_worker.run_examples_entry)."""
    ok, result = _run_task("run_examples_entry", (source, origin, ids), timeout, executor)
    return _example_report(result) if ok else {"__file__": {"ok": False, "error": result}}


def try_call(
    definition: BlockDefinition, kwargs: dict[str, Any], timeout: float = EXAMPLE_TIMEOUT, executor=None
) -> tuple[bool, Any]:
    """Call a definition's function once on real inputs (a canvas block's
    output) as a task: (ok, result or error message)."""
    code = definition.code if definition.source == "library" else None
    return _run_task("run_worker_entry", (definition.id, code, kwargs, None), timeout, executor)


# ---- writing literals ----------------------------------------------------


def format_literal(value: Any, indent: int = 0) -> str:
    """A BLOCKS-style literal: double-quoted strings, one key per line for
    dicts that don't fit on one line -- what a saved file looks like."""
    pad = " " * indent
    one_line = _format_flat(value)
    if len(one_line) + indent <= 100 or not isinstance(value, (dict, list)) or not value:
        return one_line
    inner = " " * (indent + 4)
    if isinstance(value, dict):
        items = [f"{inner}{json.dumps(k)}: {format_literal(v, indent + 4)}," for k, v in value.items()]
        return "{\n" + "\n".join(items) + f"\n{pad}}}"
    items = [f"{inner}{format_literal(v, indent + 4)}," for v in value]
    return "[\n" + "\n".join(items) + f"\n{pad}]"


def _format_flat(value: Any) -> str:
    if isinstance(value, dict):
        return "{" + ", ".join(f"{json.dumps(k)}: {_format_flat(v)}" for k, v in value.items()) + "}"
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(_format_flat(v) for v in value) + "]"
    if isinstance(value, str):
        return json.dumps(value, ensure_ascii=False)
    return repr(value)


def blocks_literal(entries: dict[str, dict[str, Any]]) -> str:
    return "BLOCKS = " + format_literal(entries) + "\n"
