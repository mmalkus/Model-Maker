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


def field_specs_for(block: dict[str, Any]) -> list[FieldSpec]:
    """The fields to render for a block: its served form."""
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
    return []


def json_param_keys(block: dict[str, Any]) -> list[str] | None:
    """The params edited as raw JSON alongside the fields: the block's
    `other_params`, or None meaning "all of them" (a block with no fields at
    all)."""
    form = block.get("form")
    if form and form["fields"]:
        return list(form["other_params"])
    return None
