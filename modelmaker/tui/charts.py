from __future__ import annotations

from pathlib import Path

import polars as pl

# Terminal-native charting for the TUI: renders the same kind/x/y/bins/title
# shape as blocks.library.generate_image, but with plotext instead of
# matplotlib, so a chart can be drawn straight into a Textual widget (no
# image protocol needed) and saved as a plain-text or HTML artifact instead
# of a PNG. Pinned to plotext's classic (<6) API in pyproject.toml -- 6.x
# restructured the package and dropped the top-level plot()/build()/
# save_fig() functions this module relies on.

CHART_KINDS = ("hist", "bar", "scatter", "line")


def _draw(df: pl.DataFrame, kind: str, x: str, y: str, bins: int, title: str, width: int | None, height: int | None) -> None:
    import plotext as plt

    plt.clear_figure()
    plt.theme("clear")  # no forced background -- matches whatever terminal theme is active
    plt.plotsize(width, height)
    if kind == "hist":
        plt.hist(df[x].to_list(), bins=bins)
    elif kind == "bar":
        plt.bar([str(v) for v in df[x].to_list()], df[y].to_list())
    elif kind == "scatter":
        plt.scatter(df[x].to_list(), df[y].to_list())
    elif kind == "line":
        plt.plot(df[x].to_list(), df[y].to_list())
    else:
        raise ValueError(f"unknown chart kind: {kind!r} (expected one of {CHART_KINDS})")
    plt.xlabel(x)
    if y:
        plt.ylabel(y)
    if title:
        plt.title(title)


def render_chart(
    df: pl.DataFrame,
    kind: str = "hist",
    x: str = "",
    y: str = "",
    bins: int = 30,
    title: str = "",
    width: int | None = None,
    height: int | None = None,
) -> str:
    """Render df as an ANSI-colored plotext chart and return it as a string,
    for embedding directly in a Textual widget. width/height are in
    character cells; None sizes to the terminal (or, inside Textual, the
    caller should pass the widget's own content size)."""
    import plotext as plt

    _draw(df, kind=kind, x=x, y=y, bins=bins, title=title, width=width, height=height)
    return plt.build()


def save_chart(
    df: pl.DataFrame,
    output_path: str | Path,
    kind: str = "hist",
    x: str = "",
    y: str = "",
    bins: int = 30,
    title: str = "",
    width: int | None = None,
    height: int | None = None,
    keep_colors: bool = False,
) -> Path:
    """Render the chart and save it to output_path. The extension picks the
    format: '.html' for a colored standalone page, anything else (typically
    '.txt') for plain text -- pass keep_colors=True to keep ANSI color codes
    in a .txt save too. Returns the resolved path."""
    import plotext as plt

    _draw(df, kind=kind, x=x, y=y, bins=bins, title=title, width=width, height=height)
    plt.build()  # populates the canvas save_fig writes out
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.save_fig(str(path), keep_colors=keep_colors)
    return path
