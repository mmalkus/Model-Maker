from modelmaker.blocks import library


def test_read_csv_without_sample_rows_uses_the_eager_reader(tmp_path):
    csv_path = tmp_path / "data.csv"
    csv_path.write_text("a,b\n1,10\n2,20\n3,30\n")

    out = library.read_csv(str(csv_path))
    assert out.to_dicts() == [{"a": 1, "b": 10}, {"a": 2, "b": 20}, {"a": 3, "b": 30}]


def test_read_csv_with_sample_rows_scans_instead_of_reading_the_whole_file(tmp_path, monkeypatch):
    csv_path = tmp_path / "data.csv"
    csv_path.write_text("a,b\n" + "".join(f"{i},{i * 10}\n" for i in range(1, 51)))

    def _boom(*args, **kwargs):
        raise AssertionError("sample mode must not fall back to the eager, whole-file reader")

    monkeypatch.setattr(library.pl, "read_csv", _boom)

    out = library.read_csv(str(csv_path), sample_rows=5)
    assert out.height == 5
    assert out.to_dicts() == [{"a": i, "b": i * 10} for i in range(1, 6)]


def test_read_csv_sample_rows_larger_than_the_file_returns_everything(tmp_path):
    csv_path = tmp_path / "data.csv"
    csv_path.write_text("a\n1\n2\n3\n")

    out = library.read_csv(str(csv_path), sample_rows=1000)
    assert out.height == 3
