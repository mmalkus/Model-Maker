from __future__ import annotations

import json
from typing import Any

from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.widgets import Input, Label, Select, Static, TextArea

from .paramspecs import FieldSpec, field_specs_for

_NO_BLANK = object()


class InfoPanel(Static):
    """Name / category / status / stale reason / last error / group_by --
    the facts about a block that aren't editable."""

    DEFAULT_CSS = "InfoPanel { height: auto; padding: 0 1; border-bottom: solid $panel-lighten-2; }"

    def show(self, block: dict[str, Any] | None) -> None:
        if block is None:
            self.update("(no block selected)")
            return
        lines = [
            f"[b]{block.get('name')}[/b]  [dim]{block.get('category')}[/dim]  ({block.get('block_type')})",
            f"status: {block.get('status')}"
            + (f"  -- {block['stale_reason']}" if block.get("stale_reason") else ""),
        ]
        if block.get("last_error"):
            lines.append(f"[red]error: {block['last_error']}[/red]")
        if block.get("group_by"):
            lines.append(f"grouped by: {block['group_by']}")
        if block.get("is_custom"):
            lines.append("[magenta]AI-authored block[/magenta]")
        self.update("\n".join(lines))


class ParamField(Horizontal):
    """One labeled param field -- an Input or a Select, per FieldSpec.kind.
    'columns' (multi-value) renders as a comma-separated Input; the web
    UI's checkbox grid doesn't have a clean terminal analog at this size."""

    DEFAULT_CSS = """
    ParamField { height: 3; }
    ParamField > Label { width: 22; padding: 1 1 0 0; }
    ParamField > Input, ParamField > Select { width: 1fr; }
    """

    AUTO = "__auto__"

    def __init__(self, spec: FieldSpec, value: Any, columns: list[str]):
        super().__init__()
        self.spec = spec
        self._columns = columns
        self._widget: Input | Select
        if spec.kind in ("text", "number"):
            self._widget = Input(
                value="" if value is None else str(value),
                type="number" if spec.kind == "number" else "text",
                placeholder=spec.placeholder,
            )
        elif spec.kind == "select":
            opts = [(o, o) for o in spec.options]
            self._widget = Select(opts, value=value if value in spec.options else Select.NULL, allow_blank=True)
        elif spec.kind == "columns":
            text = ",".join(value) if isinstance(value, list) else ""
            self._widget = Input(value=text, placeholder="col_a, col_b")
        else:  # column
            opts = [(c, c) for c in columns]
            if isinstance(value, str) and value and value not in columns:
                opts.append((value, value))
            if spec.auto_role:
                opts.insert(0, (f"Auto ({spec.auto_role})", self.AUTO))
                current = self.AUTO if value is None else value
            else:
                opts.insert(0, ("(none)", ""))
                current = value if isinstance(value, str) else ""
            self._widget = Select(opts, value=current, allow_blank=False)

    def compose(self):
        yield Label(self.spec.label)
        yield self._widget

    def get_value(self) -> Any:
        """Returns _NO_BLANK sentinel to mean 'omit this key' (auto_role
        cleared back to dynamic resolution)."""
        if self.spec.kind == "number":
            text = self._widget.value.strip()  # type: ignore[union-attr]
            if not text:
                return None
            try:
                return int(text) if "." not in text else float(text)
            except ValueError:
                try:
                    return float(text)
                except ValueError:
                    return None
        if self.spec.kind == "columns":
            text = self._widget.value.strip()  # type: ignore[union-attr]
            return [c.strip() for c in text.split(",") if c.strip()]
        if self.spec.kind == "select":
            v = self._widget.value  # type: ignore[union-attr]
            return None if v is Select.NULL else v
        if self.spec.kind == "column":
            v = self._widget.value  # type: ignore[union-attr]
            if self.spec.auto_role and v == self.AUTO:
                return _NO_BLANK
            return None if v in (Select.NULL, "") else v
        text = self._widget.value.strip()  # type: ignore[union-attr]
        return text or None


class ParamsPanel(VerticalScroll):
    """Either a declarative field form (known categories, or _col-derived
    fields for custom blocks) or a raw JSON TextArea fallback -- matches
    ParamsForm.tsx's behavior in the web UI."""

    DEFAULT_CSS = "ParamsPanel { height: auto; max-height: 16; border-bottom: solid $panel-lighten-2; }"

    def __init__(self) -> None:
        super().__init__()
        self._fields: list[ParamField] = []
        self._json_area: TextArea | None = None
        self._block: dict[str, Any] | None = None
        self._shown_signature: tuple[Any, ...] | None = None

    async def show(self, block: dict[str, Any] | None, columns: list[str]) -> None:
        # show() runs on every refresh (including each 2s poll tick, even
        # when nothing changed) -- rebuilding unconditionally destroyed and
        # recreated the JSON TextArea every time, which both wiped any
        # in-progress edit and could race Textual's CSS component-style
        # resolution on the freshly-mounted widget (KeyError on
        # 'text-area--gutter' mid-render). Only rebuild when what would be
        # displayed actually changed.
        block_id = block["id"] if block is not None else None
        signature = (block_id, block.get("params") if block else None, block.get("category") if block else None, tuple(columns))
        self._block = block
        if signature == self._shown_signature:
            return
        self._shown_signature = signature
        await self.remove_children()
        self._fields = []
        self._json_area = None
        if block is None:
            return
        specs = field_specs_for(block["category"], block.get("params") or {}, block.get("is_custom", False))
        if specs:
            self._fields = [ParamField(s, (block.get("params") or {}).get(s.key), columns) for s in specs]
            await self.mount_all(self._fields)
        else:
            self._json_area = TextArea(json.dumps(block.get("params") or {}, indent=2), language="json")
            await self.mount(self._json_area)

    def collect(self) -> dict[str, Any] | None:
        """The params dict to PATCH, or None if unparseable JSON (caller
        should refuse to save rather than wipe params)."""
        if self._block is None:
            return None
        if self._json_area is not None:
            try:
                parsed = json.loads(self._json_area.text)
                return parsed if isinstance(parsed, dict) else None
            except (json.JSONDecodeError, ValueError):
                return None
        params = dict(self._block.get("params") or {})
        for f in self._fields:
            v = f.get_value()
            if v is _NO_BLANK:
                params.pop(f.spec.key, None)
            elif v is None:
                params.pop(f.spec.key, None)
            else:
                params[f.spec.key] = v
        return params


class CodePanel(Vertical):
    """Editable code for a custom (AI-authored) block; read-only source
    view for a registry block whose code is fixed."""

    DEFAULT_CSS = "CodePanel { height: 1fr; }"

    def __init__(self) -> None:
        super().__init__()
        self._area = TextArea(language="python")
        self._is_custom = False
        self._shown_signature: tuple[Any, ...] | None = None

    def compose(self):
        yield self._area

    def show(self, block: dict[str, Any] | None) -> None:
        # show() runs on every refresh, including each unchanged 2s poll
        # tick -- reloading unconditionally would wipe an in-progress edit
        # (and reset the cursor) every couple of seconds while typing.
        block_id = block["id"] if block is not None else None
        is_custom = bool(block.get("is_custom")) if block is not None else False
        code_value = (block.get("code") if is_custom else block.get("source")) if block is not None else None
        signature = (block_id, is_custom, code_value)
        if signature == self._shown_signature:
            return
        self._shown_signature = signature
        if block is None:
            self._area.load_text("")
            self._area.read_only = True
            return
        self._is_custom = is_custom
        self._area.load_text(code_value or "")
        self._area.read_only = not self._is_custom

    def collect_code(self) -> str | None:
        return self._area.text if self._is_custom else None


class AIPanel(Vertical):
    """Instruction input for 'Draft with AI' / 'Suggest fix', plus a
    read-only view of the last proposal. Orchestration (calling the API,
    applying the result) lives in the app -- this widget is pure display."""

    DEFAULT_CSS = """
    AIPanel { height: auto; max-height: 14; border-top: solid $panel-lighten-2; padding: 0 1; }
    AIPanel > .ai-hint { color: $text-muted; height: 1; }
    """

    def __init__(self) -> None:
        super().__init__()
        self.instruction = Input(placeholder="Instruction for Draft with AI (Enter to draft)...")
        self._hint = Static("", classes="ai-hint")
        self._proposal = TextArea("", read_only=True, language="python")
        self._proposal.display = False
        self._shown_block_id: str | None = "__unset__"

    def compose(self):
        yield self.instruction
        yield self._hint
        yield self._proposal

    def show(self, block: dict[str, Any] | None) -> None:
        # Only reset the in-progress instruction and any pending AI
        # proposal when the selected block actually changes -- show() runs
        # on every refresh (including each unchanged 2s poll tick), and
        # resetting unconditionally cleared whatever the user was typing,
        # or hid a just-received draft proposal, within a couple of seconds.
        block_id = block["id"] if block is not None else None
        if block_id != self._shown_block_id:
            self._shown_block_id = block_id
            self.instruction.value = ""
            self._proposal.display = False
        self._hint.update("Ctrl+D draft with AI" + ("  |  Ctrl+F suggest fix" if block and block.get("last_error") else ""))

    def show_proposal(self, code: str | None, params: dict[str, Any] | None, explanation: str | None) -> None:
        text = (explanation or "").strip()
        if code:
            text += ("\n\n" if text else "") + code
        if params:
            text += ("\n\n" if text else "") + f"params: {json.dumps(params, indent=2)}"
        self._proposal.load_text(text or "(no proposal)")
        self._proposal.display = True
        self._hint.update("Enter=apply   Esc=discard")

    def clear_proposal(self) -> None:
        self._proposal.display = False


class Inspector(Vertical):
    DEFAULT_CSS = "Inspector { width: 1fr; height: 60%; }"

    def __init__(self) -> None:
        super().__init__()
        self.info = InfoPanel()
        self.params = ParamsPanel()
        self.code = CodePanel()
        self.ai = AIPanel()

    def compose(self):
        yield self.info
        yield self.params
        yield self.code
        yield self.ai

    async def show(self, block: dict[str, Any] | None, columns: list[str]) -> None:
        self.info.show(block)
        await self.params.show(block, columns)
        self.code.show(block)
        self.ai.show(block)
