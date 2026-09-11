from __future__ import annotations

from typing import Any

from textual import on
from textual.app import ComposeResult
from textual.containers import Vertical
from textual.screen import ModalScreen
from textual.widgets import Input, Label, ListItem, ListView, Select, Static, TextArea

# Modal screens: each is dismissed with the value the caller asked for
# (None on cancel). Kept deliberately small/keyboard-first -- Escape always
# cancels, Enter always confirms the highlighted/typed choice.


class FilterListScreen(ModalScreen[str | None]):
    """A search box over a fixed list of (label, value) options -- used for
    the add-block palette and port pickers alike."""

    DEFAULT_CSS = """
    FilterListScreen { align: center middle; }
    FilterListScreen > Vertical { width: 60; height: 20; border: round $accent; background: $panel; }
    FilterListScreen .title { height: 1; padding: 0 1; text-style: bold; }
    """

    BINDINGS = [("escape", "dismiss(None)", "Cancel")]

    def __init__(self, title: str, options: list[tuple[str, str]]):
        super().__init__()
        self._title = title
        self._options = options
        self._search = Input(placeholder="type to filter...")
        self._list = ListView()

    def compose(self) -> ComposeResult:
        with Vertical():
            yield Static(self._title, classes="title")
            yield self._search
            yield self._list

    def on_mount(self) -> None:
        self._refresh(self._options)
        self._search.focus()

    def _refresh(self, options: list[tuple[str, str]]) -> None:
        self._list.clear()
        for label, value in options:
            item = ListItem(Label(label))
            item.value = value  # type: ignore[attr-defined]
            self._list.append(item)

    @on(Input.Changed)
    def _on_search(self, event: Input.Changed) -> None:
        q = event.value.lower()
        self._refresh([(label, value) for label, value in self._options if q in label.lower()])

    @on(Input.Submitted)
    def _on_submit(self, event: Input.Submitted) -> None:
        if self._list.children:
            item = self._list.children[self._list.index or 0]
            self.dismiss(item.value)  # type: ignore[attr-defined]
        else:
            self.dismiss(None)

    @on(ListView.Selected)
    def _on_select(self, event: ListView.Selected) -> None:
        self.dismiss(event.item.value)  # type: ignore[attr-defined]


class TextInputScreen(ModalScreen[str | None]):
    """A single-line text prompt (project path, save-as path, ...)."""

    DEFAULT_CSS = """
    TextInputScreen { align: center middle; }
    TextInputScreen > Vertical { width: 70; height: auto; border: round $accent; background: $panel; padding: 1; }
    TextInputScreen .title { height: 1; text-style: bold; margin: 0 0 1 0; }
    """

    BINDINGS = [("escape", "dismiss(None)", "Cancel")]

    def __init__(self, title: str, initial: str = "", placeholder: str = ""):
        super().__init__()
        self._title = title
        self._input = Input(value=initial, placeholder=placeholder)

    def compose(self) -> ComposeResult:
        with Vertical():
            yield Static(self._title, classes="title")
            yield self._input

    def on_mount(self) -> None:
        self._input.focus()

    @on(Input.Submitted)
    def _submit(self, event: Input.Submitted) -> None:
        self.dismiss(event.value or None)


class ConfirmScreen(ModalScreen[bool]):
    DEFAULT_CSS = """
    ConfirmScreen { align: center middle; }
    ConfirmScreen > Vertical { width: 60; height: auto; border: round $accent; background: $panel; padding: 1; }
    ConfirmScreen .hint { color: $text-muted; margin-top: 1; }
    """

    BINDINGS = [("y", "dismiss(True)", "Yes"), ("n", "dismiss(False)", "No"), ("escape", "dismiss(False)", "Cancel")]

    def __init__(self, message: str):
        super().__init__()
        self._message = message

    def compose(self) -> ComposeResult:
        with Vertical():
            yield Static(self._message)
            yield Static("y = yes    n / esc = no", classes="hint")


class TextViewScreen(ModalScreen[None]):
    """Read-only scrollable text (the compiled script, an AI explanation,
    ...), with 's' to save it to a path via a follow-up TextInputScreen."""

    DEFAULT_CSS = """
    TextViewScreen { align: center middle; }
    TextViewScreen > Vertical { width: 90%; height: 90%; border: round $accent; background: $panel; }
    TextViewScreen .title { height: 1; padding: 0 1; text-style: bold; }
    """

    BINDINGS = [("escape", "dismiss(None)", "Close"), ("ctrl+s", "save", "Save to file")]

    def __init__(self, title: str, text: str, language: str | None = None, on_save=None):
        super().__init__()
        self._title = title
        self._area = TextArea(text, read_only=True, language=language)
        self._on_save = on_save

    def compose(self) -> ComposeResult:
        with Vertical():
            yield Static(f"{self._title}   (Ctrl+S to save, Esc to close)", classes="title")
            yield self._area

    async def action_save(self) -> None:
        if self._on_save is not None:
            await self._on_save(self._area.text)


class ColumnRolePickScreen(ModalScreen[tuple[str, str] | None]):
    """Pick a column then a role to hand-tag it with (POST .../column_role)."""

    DEFAULT_CSS = """
    ColumnRolePickScreen { align: center middle; }
    ColumnRolePickScreen > Vertical { width: 50; height: auto; border: round $accent; background: $panel; padding: 1; }
    ColumnRolePickScreen > Vertical > * { margin-bottom: 1; }
    """

    ROLES = ["unassigned", "id", "target", "weight", "feature", "date", "segment", "excluded"]

    BINDINGS = [("escape", "dismiss(None)", "Cancel")]

    def __init__(self, columns: list[str]):
        super().__init__()
        self._column = Select([(c, c) for c in columns], allow_blank=False)
        self._role = Select([(r, r) for r in self.ROLES], value="feature", allow_blank=False)

    def compose(self) -> ComposeResult:
        with Vertical():
            yield Label("Column")
            yield self._column
            yield Label("Role")
            yield self._role
            yield Static("Enter = apply    Esc = cancel")

    @on(Select.Changed)
    def _noop(self, event: Select.Changed) -> None:
        event.stop()

    def key_enter(self) -> None:
        col = self._column.value
        role = self._role.value
        if col and col is not Select.NULL and role and role is not Select.NULL:
            self.dismiss((str(col), str(role)))
        else:
            self.dismiss(None)
