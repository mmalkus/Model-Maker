import polars as pl

from modelmaker.blocks import library


def test_read_csv_returns_a_lazy_scan_of_the_file(tmp_path):
    csv_path = tmp_path / "data.csv"
    csv_path.write_text("a,b\n1,10\n2,20\n3,30\n")

    out = library.read_csv(str(csv_path))
    assert isinstance(out, pl.LazyFrame)
    assert out.collect().to_dicts() == [{"a": 1, "b": 10}, {"a": 2, "b": 20}, {"a": 3, "b": 30}]


def test_sample_mode_pushes_head_into_the_scan(tmp_path):
    from modelmaker.run_worker import _collect

    csv_path = tmp_path / "data.csv"
    csv_path.write_text("a,b\n" + "".join(f"{i},{i * 10}\n" for i in range(1, 51)))

    out = _collect(library.read_csv(str(csv_path)), 5)
    assert out.height == 5
    assert out.to_dicts() == [{"a": i, "b": i * 10} for i in range(1, 6)]


def test_sample_rows_larger_than_the_file_returns_everything(tmp_path):
    from modelmaker.run_worker import _collect

    csv_path = tmp_path / "data.csv"
    csv_path.write_text("a\n1\n2\n3\n")

    assert _collect(library.read_csv(str(csv_path)), 1000).height == 3
