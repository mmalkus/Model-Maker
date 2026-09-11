from __future__ import annotations

import json
from typing import Any

from rich.text import Text
from textual.containers import Vertical
from textual.widgets import DataTable, Static

# One panel, four mutually-exclusive views (only one .display=True at a
# time): a dataframe table, a plotext-rendered chart, a plain value/JSON
# dump, or a status message (not run yet, error, unsupported port type).
# The app decides which to call based on the selected block's output port
# type; this widget just renders whatever it's given.


class PreviewPane(Vertical):
    DEFAULT_CSS = """
    PreviewPane { height: 1fr; padding: 0 1; }
    PreviewPane > Static { height: auto; }
    """

    def __init__(self) -> None:
        super().__init__()
        self._table = DataTable(zebra_stripes=True)
        self._chart = Static()
        self._value = Static()
        self._message = Static("(select a block)")

    def compose(self):
        yield self._table
        yield self._chart
        yield self._value
        yield self._message

    def _show_only(self, widget) -> None:
        for w in (self._table, self._chart, self._value, self._message):
            w.display = w is widget

    def show_dataframe(self, columns: list[dict[str, Any]], rows: list[dict[str, Any]], row_count: int) -> None:
        self._table.clear(columns=True)
        col_labels = [f"{c['name']} [{c['dtype']}]" for c in columns]
        self._table.add_columns(*col_labels)
        for row in rows:
            self._table.add_row(*[str(row.get(c["name"], "")) for c in columns])
        self._show_only(self._table)

    def show_chart(self, ansi_text: str) -> None:
        self._chart.update(Text.from_ansi(ansi_text))
        self._show_only(self._chart)

    def show_value(self, value: Any) -> None:
        try:
            text = json.dumps(value, indent=2)
        except TypeError:
            text = str(value)
        self._value.update(text)
        self._show_only(self._value)

    def show_message(self, text: str) -> None:
        self._message.update(text)
        self._show_only(self._message)
