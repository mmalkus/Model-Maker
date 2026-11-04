"""The wire format results come back in (modelmaker/execution/wire.py):
every value a port actually carries survives the round trip, and what
isn't data, is oversized or malformed is refused on decode."""

from __future__ import annotations

import datetime as dt
import decimal
import json
import math
import os
import pickle
import struct

import numpy as np
import polars as pl
import pytest

from modelmaker.blocks.binning import fit_binning
from modelmaker.blocks.data_quality import data_profile
from modelmaker.blocks.library import display_kpi, display_value, generate_image
from modelmaker.blocks.modelling import fit_master_scale, logistic_regression
from modelmaker.blocks.stochastic import build_dependency, fit_distribution, fit_proxy, risk_measures
from modelmaker.execution import wire
from modelmaker.execution.safe_pickle import restricted_loads


def same(a, b) -> bool:
    """Deep equality that also checks types, compares frames by value and
    treats NaN as equal to NaN."""
    if type(a) is not type(b):
        return False
    if isinstance(a, pl.DataFrame):
        return a.schema == b.schema and a.equals(b, null_equal=True)
    if isinstance(a, pl.Series):
        return a.name == b.name and a.dtype == b.dtype and a.equals(b)
    if isinstance(a, np.ndarray):
        return a.dtype == b.dtype and a.shape == b.shape and (a.dtype == object or np.array_equal(a, b, equal_nan=a.dtype.kind in "fc"))
    if isinstance(a, np.generic):
        return a.dtype == b.dtype and (bool(a == b) or (a.dtype.kind in "fc" and bool(np.isnan(a) and np.isnan(b))))
    if isinstance(a, float):
        return (math.isnan(a) and math.isnan(b)) or a == b
    if isinstance(a, dict):
        return list(a) == list(b) and all(same(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return len(a) == len(b) and all(same(x, y) for x, y in zip(a, b))
    return a == b


def roundtrip(value):
    return wire.decode(wire.encode(value))


@pytest.fixture(scope="module")
def credit() -> pl.DataFrame:
    rng = np.random.default_rng(7)
    n = 400
    x = rng.normal(size=n)
    y = (rng.random(n) < 1 / (1 + np.exp(-1.5 * x))).astype(int)
    return pl.DataFrame({"x": x, "x2": rng.normal(size=n), "loss": np.abs(rng.lognormal(size=n)), "y": y})


# Each entry: a real built-in block producing a value on that port type.
PRODUCERS = {
    "dataframe": lambda df, tmp: df,
    "model": lambda df, tmp: logistic_regression(df, "y", ["x", "x2"]),
    "scalar_metric": lambda df, tmp: data_profile(df),
    "image": lambda df, tmp: generate_image(df, kind="hist", x="x", output_dir=str(tmp)),
    "binning": lambda df, tmp: fit_binning(df, "y", ["x"], max_bins=4),
    "master_scale": lambda df, tmp: fit_master_scale(df, "x", "y", n_grades=3),
    "distribution": lambda df, tmp: fit_distribution(df, "loss", families=["lognorm"]),
    "dependency": lambda df, tmp: build_dependency(df, ["x", "x2"]),
    "proxy_function": lambda df, tmp: fit_proxy(df, ["x", "x2"], value_col="loss"),
    "simulation_result": lambda df, tmp: risk_measures(df, "loss", n_bootstrap=10),
    "any": lambda df, tmp: display_value({"kpi": display_kpi({"gini": 0.4}, column="gini")}, save_csv=False),
}


def test_every_port_type_has_a_producer():
    from modelmaker.blocks.base import PORT_TYPES

    assert set(PRODUCERS) == set(PORT_TYPES)


@pytest.mark.parametrize("port_type", sorted(PRODUCERS))
def test_real_block_outputs_round_trip_for_every_port_type(port_type, credit, tmp_path):
    value = PRODUCERS[port_type](credit, tmp_path)
    back = roundtrip(value)
    assert same(value, back), f"{port_type}: {value!r:.200} != {back!r:.200}"


def test_image_bytes_stay_png():
    png = b"\x89PNG\r\n\x1a\n" + bytes(range(256))
    assert roundtrip(png) == png


def test_dataframe_dtypes_round_trip():
    df = pl.DataFrame(
        {
            "i8": pl.Series([1, None, 3], dtype=pl.Int8),
            "u64": pl.Series([1, 2, 3], dtype=pl.UInt64),
            "f": [1.5, float("nan"), None],
            "s": ["a", None, "c"],
            "b": [True, False, None],
            "cat": pl.Series(["x", "y", "x"], dtype=pl.Categorical),
            "enum": pl.Series(["lo", "hi", "lo"], dtype=pl.Enum(["lo", "hi"])),
            "dec": pl.Series([decimal.Decimal("1.25"), None, decimal.Decimal("3.50")], dtype=pl.Decimal(10, 2)),
            "date": [dt.date(2020, 1, 1), None, dt.date(2021, 6, 30)],
            "ts": pl.Series([dt.datetime(2020, 1, 1, 12)] * 3).dt.replace_time_zone("Europe/Amsterdam"),
            "dur": [dt.timedelta(seconds=5)] * 3,
            "lst": [[1, 2], [], None],
            "st": [{"a": 1, "b": "x"}, {"a": 2, "b": None}, None],
            "bin": [b"\x00\x01", None, b""],
            "nul": [None, None, None],
        }
    )
    back = roundtrip(df)
    assert back.schema == df.schema
    assert back.equals(df, null_equal=True)
    assert same(roundtrip(pl.DataFrame()), pl.DataFrame())
    assert same(roundtrip(pl.DataFrame({"a": []})), pl.DataFrame({"a": []}))


@pytest.mark.parametrize(
    "value",
    [
        None,
        True,
        0,
        -(2**70),
        1.5,
        float("nan"),
        float("inf"),
        float("-inf"),
        "text",
        "",
        [1, "a", None, [2.5]],
        {"a": 1, "nested": {"b": [1, 2]}},
        {"$tuple": "a dict whose key looks like a tag"},
        {1: "int key", (1, 2): "tuple key", None: "none key", 2.5: "float key"},
        {False: "bool key", "s": 1},
        (pl.DataFrame({"a": [1]}), {"kind": "glm"}),  # a multi-output block's tuple
        ((1, 2), ()),
        {1, 2, 3},
        frozenset({"a"}),
        dt.datetime(2024, 5, 6, 7, 8, 9, 10),
        dt.datetime(2024, 5, 6, tzinfo=dt.timezone(dt.timedelta(hours=2))),
        dt.date(2024, 1, 31),
        dt.time(23, 59, 1),
        dt.timedelta(days=-1, seconds=3, microseconds=4),
        decimal.Decimal("-1.000000000000000000000001"),
        b"\x00bytes",
        pl.Series("s", [1, 2, None]),
        np.arange(12, dtype=np.int32).reshape(3, 4),
        np.array([1.5, np.nan]),
        np.array(["a", "bc"]),
        np.array([dt.datetime(2020, 1, 1)], dtype="datetime64[us]"),
        np.float32(2.5),
        np.int64(7),
        np.bool_(True),
        np.array([{"a": 1}, [1, 2], None], dtype=object),
    ],
    ids=repr,
)
def test_value_round_trips(value):
    assert same(value, roundtrip(value))


def test_numpy_scalars_keep_their_type_inline():
    value = {"f64": np.float64(0.25), "nan": np.float32("nan"), "u8": np.uint8(255), "b": np.bool_(False)}
    back = roundtrip(value)
    assert same(value, back)
    # inline in the JSON, not one blob each
    assert b'"$np"' in wire.encode(value) and wire.encode(value).count(b"$nd") == 0


def test_lazyframe_and_arbitrary_objects_are_refused_at_encode():
    with pytest.raises(wire.WireError, match="LazyFrame"):
        wire.encode(pl.LazyFrame({"a": [1]}))

    class Custom:
        pass

    with pytest.raises(wire.WireError, match="can't return"):
        wire.encode({"x": Custom()})


def test_fitted_sklearn_estimator_round_trips_through_the_interim_pickle_path(credit):
    from sklearn.linear_model import LogisticRegression

    model = LogisticRegression(random_state=np.random.RandomState(3)).fit(credit.select("x", "x2").to_numpy(), credit["y"].to_numpy())
    back = roundtrip({"estimator": model})["estimator"]
    assert type(back) is LogisticRegression
    x = credit.select("x", "x2").to_numpy()
    assert np.allclose(back.predict_proba(x), model.predict_proba(x))


def test_size_cap_is_enforced_on_encode_and_on_decode():
    df = pl.DataFrame({"a": np.zeros(500_000)})  # ~4 MB of Arrow
    with pytest.raises(wire.ValueTooLarge, match="limit"):
        wire.encode(df, max_bytes=1024**2)
    frame = wire.encode(df)
    with pytest.raises(wire.ValueTooLarge):
        wire.decode(frame, max_bytes=1024**2)


def test_size_cap_reads_the_environment(monkeypatch):
    monkeypatch.setenv("MODELMAKER_MAX_RESULT_MB", "1")
    with pytest.raises(wire.ValueTooLarge, match="MODELMAKER_MAX_RESULT_MB"):
        wire.encode(b"\x00" * (2 * 1024**2))


def test_json_part_has_its_own_cap(monkeypatch):
    monkeypatch.setattr(wire, "DEFAULT_MAX_HEADER_BYTES", 10_000)
    with pytest.raises(wire.ValueTooLarge, match="dataframe rather than"):
        wire.encode(list(range(10_000)))


def test_depth_cap():
    deep: list = []
    node = deep
    for _ in range(wire.MAX_DEPTH + 5):
        node.append([])
        node = node[0]
    with pytest.raises(wire.WireError, match="nested"):
        wire.encode(deep)
    cyclic: list = []
    cyclic.append(cyclic)
    with pytest.raises(wire.WireError, match="nested"):
        wire.encode(cyclic)
    # ... and a hand-made frame nested too deep is refused on decode.
    header = json.dumps({"fields": {}, "value": json.loads("[" * 500 + "]" * 500), "blobs": []}).encode()
    with pytest.raises(wire.WireError):
        wire.decode(wire.MAGIC + struct.pack(">I", len(header)) + header)


def _frame(value_node, blobs: list[bytes] = ()) -> bytes:
    header = json.dumps({"fields": {"ok": True}, "value": value_node, "blobs": [len(b) for b in blobs]}).encode()
    return wire.MAGIC + struct.pack(">I", len(header)) + header + b"".join(blobs)


@pytest.mark.parametrize(
    "frame",
    [
        b"",
        b"not a frame at all",
        pickle.dumps({"ok": True}),  # what the old protocol sent
        wire.MAGIC + struct.pack(">I", 10**9),
        _frame({"$bytes": 3}, [b"x"]),  # blob index out of range
        _frame({"$bytes": 0}, [b"x"])[:-1],  # truncated
        _frame({"$nd": 0, "dtype": "O", "shape": [1]}, [b"\x00" * 8]),  # object dtype
        _frame({"$nd": 0, "dtype": "<f8", "shape": [3]}, [b"\x00" * 8]),  # shape mismatch
        _frame({"$df": 0}, [b"garbage"]),
        _frame({"$unknown": 1}),
        _frame({"$f": "1e999"}),
        _frame({"$datetime": "x" * 1000}),
    ],
)
def test_malformed_frames_are_refused(frame):
    with pytest.raises(wire.WireError):
        wire.decode_frame(frame)


class _Exec:
    """Unpickling this would run code: the classic malicious payload."""

    def __init__(self, marker: str) -> None:
        self.marker = marker

    def __reduce__(self):
        return (exec, (f"open({self.marker!r}, 'w').write('pwned')",))


class _System:
    def __reduce__(self):
        return (os.system, ("echo pwned",))


def test_restricted_unpickler_refuses_code_execution(tmp_path):
    marker = tmp_path / "marker"
    for payload in (_Exec(str(marker)), _System(), [1, {"x": _Exec(str(marker))}]):
        data = pickle.dumps(payload)
        with pytest.raises(pickle.UnpicklingError, match="not allowed"):
            restricted_loads(data)
        with pytest.raises(wire.WireError, match="not allowed"):
            wire.decode(_frame({"$pickle": 0}, [data]))
    assert not marker.exists()


def test_restricted_unpickler_refuses_functions_under_allowed_prefixes_and_dotted_names():
    import sklearn.base
    import sklearn.utils

    # A function in an allowed module, a dotted attribute path out of an
    # allowed class, and numpy.load (which would itself unpickle a file).
    for obj in (sklearn.utils.check_random_state, sklearn.base.BaseEstimator.get_params, np.load):
        with pytest.raises(pickle.UnpicklingError, match="not allowed"):
            restricted_loads(pickle.dumps(obj))
