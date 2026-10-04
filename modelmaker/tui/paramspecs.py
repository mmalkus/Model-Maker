from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

# The TUI's view of a block's params form. A registry block's fields come
# from the API with the block (block["form"], built from its BlockSpec and
# function signature -- see modelmaker/forms.py), so the TUI and the web UI
# render the same form with no per-block list here. Custom AI-authored
# blocks get a column picker per `_col` param, same as the web UI's
# ParamsForm.tsx; anything with no field is edited as raw JSON.

FieldKind = Literal["text", "number", "select", "column", "columns", "checkbox"]


@dataclass(frozen=True)
class FieldSpec:
    key: str
    label: str
    kind: FieldKind
    placeholder: str = ""
    step: float | None = None
    options: tuple[str, ...] = ()
    # 'target' picks up the role=target column; 'predicted' picks up
    # whichever column a modelling block upstream tagged role=predicted --
    # see packet.resolve_role_column / blocks/modelling.py. Only meaningful
    # for kind='column'; leaving the param out of `params` entirely puts it
    # back in this dynamically-resolved "Auto" state.
    auto_role: Literal["target", "predicted"] | None = None
    # The function's own default -- what leaving the param unset means.
    default: Any = None


def _prettify_column_param_label(key: str) -> str:
    base = key[:-4] if key.endswith("_col") else key
    base = base.replace("_", " ")
    return f"{base[:1].upper()}{base[1:]} column"


def derive_column_field_specs(params: dict[str, object]) -> list[FieldSpec]:
    """AI-drafted (custom) blocks name every existing input column they read
    as a `<name>_col` string parameter (see llm/prompts.CONTRACT) instead of
    hardcoding it into the function body. Turn each such param into a
    'column' field (a picker over the block's real input columns) instead
    of leaving it in raw JSON -- a stale value (e.g. after an upstream
    rename) then shows up as an extra, clearly-off option rather than
    silently failing at run time."""
    keys = sorted(k for k, v in params.items() if k.endswith("_col") and (v is None or isinstance(v, str)))
    return [FieldSpec(key, _prettify_column_param_label(key), "column") for key in keys]


def field_specs_for(block: dict[str, Any]) -> list[FieldSpec]:
    """The fields to render for a block: its served form if it's a registry
    block, else (custom blocks) the _col-derived column pickers."""
    form = block.get("form")
    if form:
        return [
            FieldSpec(
                key=f["key"],
                label=f["label"],
                kind=f["kind"],
                placeholder=f.get("placeholder") or "",
                step=f.get("step"),
                options=tuple(f.get("options") or ()),
                auto_role=f.get("auto_role"),
                default=f.get("default"),
            )
            for f in form["fields"]
        ]
    if block.get("is_custom"):
        return derive_column_field_specs(block.get("params") or {})
    return []


def json_param_keys(block: dict[str, Any]) -> list[str] | None:
    """The params edited as raw JSON alongside the fields: a registry block's
    `other_params`, or None meaning "all of them" (custom blocks, and any
    block with no fields at all)."""
    form = block.get("form")
    if form and form["fields"]:
        return list(form["other_params"])
    return None
