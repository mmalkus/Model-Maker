"""A block function's signature, read from its source with the AST.

Everything outside a task process that needs to know about a block's
parameters -- the params form, the AI catalogue, validation, the runner's
"does this block take `block_id` / `output_dir` / a role param?" checks,
the compiler, the metadata rules' defaults -- reads it from here, never
from the live function. Getting the live function means executing the
block's module code (its top-level statements, decorators and default
expressions), and the API process never executes block code: that only
happens in the task processes that run a block (see run_worker.py).

What the AST can't give is the *value* of a default that isn't a literal
(`x=SOME_CONST`, `x=float("inf")`): that needs the expression evaluated.
Such a param is still known to have a default -- calls may leave it out, a
form shows it as unset -- but its value is `UNKNOWN_DEFAULT` here, and
`literal_defaults()` leaves it out (see docs/block-authoring.md).
"""

from __future__ import annotations

import ast
import builtins
import inspect
from dataclasses import dataclass
from typing import Any, Literal

ParamKind = Literal["positional_only", "positional_or_keyword", "var_positional", "keyword_only", "var_keyword"]

_INSPECT_KINDS: dict[Any, ParamKind] = {
    inspect.Parameter.POSITIONAL_ONLY: "positional_only",
    inspect.Parameter.POSITIONAL_OR_KEYWORD: "positional_or_keyword",
    inspect.Parameter.VAR_POSITIONAL: "var_positional",
    inspect.Parameter.KEYWORD_ONLY: "keyword_only",
    inspect.Parameter.VAR_KEYWORD: "var_keyword",
}


class _UnknownDefault:
    """The value of a default the AST can't evaluate (not a literal)."""

    _instance: _UnknownDefault | None = None

    def __new__(cls) -> _UnknownDefault:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __repr__(self) -> str:
        return "<non-literal default>"

    def __reduce__(self) -> str:
        return "UNKNOWN_DEFAULT"


UNKNOWN_DEFAULT: Any = _UnknownDefault()


class SignatureError(ValueError):
    """The code has no usable function definition for the block."""


@dataclass(frozen=True, eq=False)
class ParamSpec:
    """One parameter as written in the `def`.

    `annotation` and `default_source` are source text (annotations are
    never evaluated: block files use `from __future__ import annotations`
    semantics). `default` is the literal value when `has_default` and the
    expression is a literal, else UNKNOWN_DEFAULT."""

    name: str
    kind: ParamKind = "positional_or_keyword"
    annotation: str | None = None
    has_default: bool = False
    default: Any = None
    default_source: str | None = None

    @property
    def variadic(self) -> bool:
        return self.kind in ("var_positional", "var_keyword")

    @property
    def required(self) -> bool:
        return not self.has_default and not self.variadic

    @property
    def literal_default(self) -> bool:
        return self.has_default and self.default is not UNKNOWN_DEFAULT


@dataclass(frozen=True, eq=False)
class BlockSignature:
    """A function's parameters in order. Compared and hashed by their
    source text, so it can key a cache even when a default is a list."""

    params: tuple[ParamSpec, ...]

    def _key(self) -> tuple[tuple[Any, ...], ...]:
        return tuple((p.name, p.kind, p.annotation, p.has_default, p.default_source) for p in self.params)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, BlockSignature) and self._key() == other._key()

    def __hash__(self) -> int:
        return hash(self._key())

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(p.name for p in self.params)

    def get(self, name: str) -> ParamSpec | None:
        return next((p for p in self.params if p.name == name), None)

    def __contains__(self, name: object) -> bool:
        return any(p.name == name for p in self.params)

    def accepts(self, name: str) -> bool:
        """Whether the function names a parameter `name` -- the same test as
        `name in inspect.signature(fn).parameters` (a `**kwargs` doesn't
        count: the engine's injected params are opted into by name)."""
        return name in self

    def literal_defaults(self) -> dict[str, Any]:
        """Every parameter's default whose value is known (a literal)."""
        return {p.name: p.default for p in self.params if p.literal_default}

    # ---- construction --------------------------------------------------------

    @classmethod
    def from_node(cls, node: ast.FunctionDef | ast.AsyncFunctionDef) -> BlockSignature:
        a = node.args
        positional = [*a.posonlyargs, *a.args]
        # Defaults line up with the *last* positional parameters.
        pos_defaults: list[ast.expr | None] = [None] * (len(positional) - len(a.defaults)) + list(a.defaults)
        params: list[ParamSpec] = []
        for i, arg in enumerate(positional):
            kind: ParamKind = "positional_only" if i < len(a.posonlyargs) else "positional_or_keyword"
            params.append(_param(arg, kind, pos_defaults[i]))
        if a.vararg is not None:
            params.append(_param(a.vararg, "var_positional", None))
        for arg, default in zip(a.kwonlyargs, a.kw_defaults):
            params.append(_param(arg, "keyword_only", default))
        if a.kwarg is not None:
            params.append(_param(a.kwarg, "var_keyword", None))
        return cls(tuple(params))

    @classmethod
    def from_code(cls, code: str, name: str) -> BlockSignature:
        """The signature of the top-level function `name` defined in `code`.
        Raises SignatureError if the code doesn't parse or has no such def."""
        return cls.from_node(function_node(code, name))

    @classmethod
    def from_callable(cls, fn: Any) -> BlockSignature:
        """From a live function (for callers that already hold one -- never
        used to get a library block's signature)."""
        params = []
        for p in inspect.signature(fn).parameters.values():
            ann = p.annotation
            ann_text = None if ann is inspect.Parameter.empty else (ann if isinstance(ann, str) else getattr(ann, "__name__", str(ann)))
            has_default = p.default is not inspect.Parameter.empty
            params.append(
                ParamSpec(
                    p.name,
                    _INSPECT_KINDS[p.kind],
                    ann_text,
                    has_default,
                    p.default if has_default else None,
                    repr(p.default) if has_default else None,
                )
            )
        return cls(tuple(params))


def _param(arg: ast.arg, kind: ParamKind, default: ast.expr | None) -> ParamSpec:
    annotation = ast.unparse(arg.annotation) if arg.annotation is not None else None
    if default is None:
        return ParamSpec(arg.arg, kind, annotation)
    try:
        value = ast.literal_eval(default)
    except (ValueError, TypeError, SyntaxError, MemoryError, RecursionError):
        value = UNKNOWN_DEFAULT
    return ParamSpec(arg.arg, kind, annotation, True, value, ast.unparse(default))


def function_node(code: str, name: str) -> ast.FunctionDef | ast.AsyncFunctionDef:
    """The last top-level `def name(...)` in `code` (the one a module's
    namespace would end up holding)."""
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        raise SignatureError(f"code doesn't parse: line {e.lineno}: {e.msg}") from None
    found = None
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            found = node
        elif found is not None and name in _assigned_names(node):
            found = None  # rebound after the def
    if found is None:
        raise SignatureError(f"block code doesn't define {name}()")
    return found


def _assigned_names(node: ast.stmt) -> set[str]:
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        return {node.name}
    if isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        return {n.id for t in targets for n in ast.walk(t) if isinstance(n, ast.Name)}
    return set()


# Names a block's code may use at definition time without defining them:
# Python's builtins, plus `pl`, which the namespace a block is defined in
# always provides (see block_library.compile_fn).
_PROVIDED = frozenset(dir(builtins)) | {"pl", "__name__"}


def undefined_definition_names(code: str, name: str) -> list[str]:
    """Problems that would make defining `name` fail with a NameError --
    a name used in a decorator or a default that the code never binds at
    top level. Found statically, since the definition can't be run here;
    a best-effort check (it doesn't follow conditional or star imports)."""
    try:
        tree = ast.parse(code)
        node = function_node(code, name)
    except (SignatureError, SyntaxError):
        return []
    bound: set[str] = set(_PROVIDED)
    for stmt in tree.body:
        bound |= _assigned_names(stmt)
        if isinstance(stmt, (ast.Import, ast.ImportFrom)):
            bound |= {(a.asname or a.name).split(".")[0] for a in stmt.names}
    a = node.args
    exprs: list[tuple[str, ast.expr]] = [("decorator", d) for d in node.decorator_list]
    positional = [*a.posonlyargs, *a.args]
    for arg, default in zip(positional[len(positional) - len(a.defaults) :], a.defaults):
        exprs.append((f"default of {arg.arg!r}", default))
    exprs += [(f"default of {arg.arg!r}", d) for arg, d in zip(a.kwonlyargs, a.kw_defaults) if d is not None]
    problems = []
    for where, expr in exprs:
        local = {n.arg for lam in ast.walk(expr) if isinstance(lam, ast.Lambda) for n in ast.walk(lam.args) if isinstance(n, ast.arg)}
        local |= {
            t.id
            for comp in ast.walk(expr)
            if isinstance(comp, (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp))
            for gen in comp.generators
            for t in ast.walk(gen.target)
            if isinstance(t, ast.Name)
        }
        for n in ast.walk(expr):
            if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load) and n.id not in bound and n.id not in local:
                problems.append(f"{where} uses {n.id!r}, which the block's code doesn't define")
    return list(dict.fromkeys(problems))
