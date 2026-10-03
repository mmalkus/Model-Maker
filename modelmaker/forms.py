"""Params forms for registry blocks, served to the web UI and the TUI with
each block (api._block_out). A block's own BlockSpec.form fields come
first; every other param gets a field inferred from the function's
signature, by its name and type annotation:

  - `<name>_col` (or target/col/column/feature), a str  -> column picker
  - `<name>_cols` (or features/cols/columns/by/on), a list[str] -> multi-picker
  - bool -> checkbox;  int/float -> number;  any other str -> text

Anything else (dicts, rule lists, list[float]) can't be a flat field and
is listed in `other_params`, for the UI's JSON box. Custom AI-authored
blocks have no signature to read until they run, so the UIs derive their
`_col` pickers from the block's params instead."""

from __future__ import annotations

import functools
import inspect
import re
from dataclasses import asdict
from typing import Any

from .agent.catalogue import INJECTED_PARAMS
from .blocks.base import BlockSpec, FieldSpec
from .packet import ColumnRole
from .util import ROLE_PARAM_NAMES, find_role_param

COLUMN_NAMES = frozenset({"target", "col", "column", "feature"})
COLUMN_LIST_NAMES = frozenset({"features", "cols", "columns", "by", "on", "risk_factors"})
_AUTO_ROLES = {ColumnRole.TARGET: "target", ColumnRole.PREDICTED: "predicted"}


def _base_type(annotation: Any) -> str:
    """The annotation as a string, without an `| None` (every annotation is
    a string here: the block modules use `from __future__ import annotations`)."""
    text = annotation if isinstance(annotation, str) else getattr(annotation, "__name__", str(annotation))
    parts = [p.strip() for p in text.split("|") if p.strip() != "None"]
    return " | ".join(parts)


def prettify_label(key: str) -> str:
    if key.endswith("_col"):
        base = key[:-4].replace("_", " ")
        return f"{base[:1].upper()}{base[1:]} column"
    base = key.replace("_", " ")
    return f"{base[:1].upper()}{base[1:]}"


def infer_field(key: str, annotation: Any, auto_role: str | None = None) -> FieldSpec | None:
    """A field for one param from its name and annotation, or None when it
    can't be a flat field."""
    t = _base_type(annotation)
    label = prettify_label(key)
    if t == "str" and (key.endswith("_col") or key in COLUMN_NAMES):
        return FieldSpec(key, label, "column", auto_role=auto_role)
    if t == "list[str]" and (key.endswith("_cols") or key in COLUMN_LIST_NAMES):
        return FieldSpec(key, label, "columns")
    if t == "bool":
        return FieldSpec(key, label, "checkbox")
    if re.fullmatch(r"(int|float)( \| (int|float))?", t):
        return FieldSpec(key, label, "number")
    if t == "str":
        return FieldSpec(key, label, "text")
    return None


def _jsonable(value: Any) -> Any:
    if value is inspect.Parameter.empty:
        return None
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return repr(value)


def block_form(spec: BlockSpec) -> dict[str, Any]:
    """{"fields": [...], "other_params": [...]} for a registry block. Each
    field carries the param's signature default (null if required), so a
    checkbox or number box can show what an unset param means."""
    return _block_form(spec.category, spec.fn, spec.form, tuple(p.name for p in spec.inputs))


# Keyed on the spec's fixed parts (BlockSpec itself isn't hashable): a
# registry block's form never changes after import, and it's asked for on
# every /api/graph poll.
@functools.lru_cache(maxsize=None)
def _block_form(category: str, fn, form: tuple[FieldSpec, ...], port_names: tuple[str, ...]) -> dict[str, Any]:
    auto_roles = {
        name: _AUTO_ROLES[role] for role in ROLE_PARAM_NAMES if (name := find_role_param(fn, role)) is not None
    }
    params = {
        name: p
        for name, p in inspect.signature(fn).parameters.items()
        if name not in port_names
        and name not in INJECTED_PARAMS
        and p.kind not in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
    }
    fields = list(form)
    declared = {f.key for f in fields}
    other = []
    for name, p in params.items():
        if name in declared:
            continue
        field = infer_field(name, p.annotation, auto_roles.get(name))
        if field is None:
            other.append(name)
        else:
            fields.append(field)
    return {
        "fields": [
            {**asdict(f), "options": list(f.options), "default": _jsonable(params[f.key].default)} for f in fields
        ],
        "other_params": other,
    }
