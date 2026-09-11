from __future__ import annotations

import pytest

pytest.importorskip("plotext")

import polars as pl

from modelmaker.tui import charts


@pytest.fixture
def df() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "score": [10, 20, 30, 40, 50, 60, 70],
            "count": [1, 2, 3, 4, 5, 6, 7],
            "region": ["A", "B", "A", "B", "A", "B", "A"],
        }
    )


@pytest.mark.parametrize(
    "kind,x,y",
    [
        ("hist", "score", ""),
        ("bar", "region", "count"),
        ("scatter", "score", "count"),
        ("line", "score", "count"),
    ],
)
def test_render_chart_produces_nonempty_ansi_text(df, kind, x, y):
    out = charts.render_chart(df, kind=kind, x=x, y=y, title="t", width=40, height=15)
    assert isinstance(out, str)
    assert out.strip()


def test_render_chart_rejects_unknown_kind(df):
    with pytest.raises(ValueError, match="unknown chart kind"):
        charts.render_chart(df, kind="pie", x="score")


def test_save_chart_writes_plain_text_by_default(df, tmp_path):
    path = charts.save_chart(df, tmp_path / "chart.txt", kind="hist", x="score", width=40, height=15)
    assert path == tmp_path / "chart.txt"
    text = path.read_text()
    assert text.strip()
    assert "\x1b[" not in text  # no ANSI codes without keep_colors=True


def test_save_chart_keep_colors_preserves_ansi(df, tmp_path):
    path = charts.save_chart(
        df, tmp_path / "chart.txt", kind="hist", x="score", width=40, height=15, keep_colors=True
    )
    assert "\x1b[" in path.read_text()


def test_save_chart_creates_parent_dirs(df, tmp_path):
    path = charts.save_chart(df, tmp_path / "nested" / "dir" / "chart.txt", kind="hist", x="score")
    assert path.exists()
