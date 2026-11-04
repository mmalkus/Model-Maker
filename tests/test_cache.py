"""The result cache on disk (cache.py): wire frames, never pickle."""

from __future__ import annotations

import datetime as dt
import decimal
import math
import pickle

import numpy as np
import polars as pl
import pytest

from modelmaker import cache as cache_module
from modelmaker.cache import CacheStore, decode_entry
from modelmaker.execution import wire
from modelmaker.graph import Graph, Wire
from modelmaker.packet import ColumnMeta, ColumnRole, DataFramePacket
from modelmaker.runner import Runner

from .helpers import make_block

KEY = "0123456789abcdef"


class _Planted:
    """What a malicious cache file would carry: code run on unpickling."""

    def __init__(self, marker):
        self.marker = str(marker)

    def __reduce__(self):
        return (open, (self.marker, "w"))


def _every_kind_of_output() -> dict:
    from sklearn.linear_model import LogisticRegression

    model = LogisticRegression().fit(np.array([[0.0], [1.0], [2.0], [3.0]]), np.array([0, 0, 1, 1]))
    packet = DataFramePacket(
        data=pl.DataFrame({"id": [1, 2], "x": [0.5, 1.5], "d": [dt.date(2024, 1, 1), dt.date(2024, 2, 1)]}),
        schema_meta={
            "id": ColumnMeta("Int64", ColumnRole.ID, "row id", ["key"]),
            "x": ColumnMeta("Float64", ColumnRole.FEATURE),
            "d": ColumnMeta("Date", ColumnRole.DATE),
        },
        lineage=["b_read", "b_x"],
    ).compute_summary()
    return {
        "out": packet,
        "metric": {"label": "Gini", "value": 0.42, "nan": math.nan, "per_grade": {1: 0.1, 2: 0.2}},
        "table": pl.DataFrame({"a": [1, 2, 3]}),
        "image": b"\x89PNG\r\n\x1a\nfake",
        "pair": (1, "two", [3.0]),
        "array": np.arange(6, dtype=np.float32).reshape(2, 3),
        "scalar": np.float64(1.5),
        "when": dt.datetime(2024, 5, 1, 12, 30),
        "money": decimal.Decimal("1.10"),
        "levels": {"a", "b"},
        "none": None,
        "model": model,
    }


def test_every_kind_of_output_round_trips_through_disk(tmp_path):
    outputs = _every_kind_of_output()
    CacheStore(tmp_path).set(KEY, outputs)
    assert (tmp_path / f"{KEY}.mmw").read_bytes().startswith(wire.MAGIC)
    got = CacheStore(tmp_path).get(KEY)  # a fresh store: read from the file
    assert got is not None
    packet, original = got.outputs["out"], outputs["out"]
    assert isinstance(packet, DataFramePacket)
    assert packet.data.equals(original.data) and packet.data.schema == original.data.schema
    assert packet.schema_meta == original.schema_meta and packet.lineage == original.lineage
    assert packet.summary == original.summary
    metric = got.outputs["metric"]
    assert metric["label"] == "Gini" and math.isnan(metric["nan"]) and metric["per_grade"] == {1: 0.1, 2: 0.2}
    assert got.outputs["table"].equals(outputs["table"])
    for name in ("image", "pair", "when", "money", "levels", "none"):
        assert got.outputs[name] == outputs[name], name
    assert got.outputs["array"].dtype == np.float32 and (got.outputs["array"] == outputs["array"]).all()
    assert type(got.outputs["scalar"]) is np.float64
    assert (got.outputs["model"].coef_ == outputs["model"].coef_).all()


def test_what_the_runner_stores_round_trips(tmp_path):
    csv = tmp_path / "a.csv"
    csv.write_text("a,b\n1,x\n2,y\n")
    graph = Graph(
        blocks={
            "b_read": make_block("b_read", "read_csv", params={"path": str(csv)}),
            "b_desc": make_block("b_desc", "data_profile", x=1),
        },
        wires={"w": Wire("w", "b_read", "out", "b_desc", "df")},
    )
    runner = Runner(graph, CacheStore(tmp_path / "cache"))
    assert runner.refresh("b_read") == "green"
    assert runner.run_block("b_desc") == "green", runner.state["b_desc"].last_error
    for bid in ("b_read", "b_desc"):
        key = runner.compute_key(bid)
        live = runner.cache.get(key)
        disk = CacheStore(tmp_path / "cache").get(key)
        assert disk is not None and disk.outputs.keys() == live.outputs.keys()
        for port, value in live.outputs.items():
            if isinstance(value, DataFramePacket):
                assert disk.outputs[port].data.equals(value.data)
                assert disk.outputs[port].schema_meta == value.schema_meta
            else:
                assert disk.outputs[port] == value


def test_old_pickle_files_are_never_loaded_and_are_deleted(tmp_path):
    swept, touched = tmp_path / "swept", tmp_path / "touched"
    (tmp_path / f"{KEY}.pkl").write_bytes(pickle.dumps(_Planted(swept)))
    (tmp_path / "recovery.json").write_text("{}")  # not the cache's: left alone
    store = CacheStore(tmp_path)
    assert not (tmp_path / f"{KEY}.pkl").exists() and (tmp_path / "recovery.json").exists()

    # One planted after the store opened the directory (and under a key
    # name the opening sweep doesn't cover) goes when its key is read.
    for key in (KEY, "custom_key"):
        (tmp_path / f"{key}.pkl").write_bytes(pickle.dumps(_Planted(touched)))
        assert store.get(key) is None
        assert not (tmp_path / f"{key}.pkl").exists()
    assert not swept.exists() and not touched.exists()


def test_a_tampered_frame_is_a_miss_and_runs_nothing(tmp_path):
    marker = tmp_path / "pwned"
    # A well-formed frame whose value smuggles a pickle of a global that
    # isn't allowed: refused by safe_pickle, never called.
    header = (
        b'{"fields":{"kind":"cache-entry","version":1},"value":{"timestamp":"t","outputs":'
        b'{"out":{"value":{"$pickle":0}}}},"blobs":[%d]}'
    )
    payload = pickle.dumps(_Planted(marker))
    head = header % len(payload)
    frame = wire.MAGIC + len(head).to_bytes(4, "big") + head + payload
    (tmp_path / f"{KEY}.mmw").write_bytes(frame)
    assert CacheStore(tmp_path).get(KEY) is None
    assert not marker.exists() and not (tmp_path / f"{KEY}.mmw").exists()

    CacheStore(tmp_path).set(KEY, {"out": {"a": 1}})
    data = bytearray((tmp_path / f"{KEY}.mmw").read_bytes())
    data[-3] ^= 0xFF  # corrupt the JSON header's tail
    (tmp_path / f"{KEY}.mmw").write_bytes(bytes(data))
    assert CacheStore(tmp_path).get(KEY) is None


@pytest.mark.parametrize(
    "value",
    [
        {"kind": "something-else", "version": 1},
        {"kind": "cache-entry", "version": 99},
    ],
)
def test_a_frame_that_isnt_a_cache_entry_is_refused(value):
    frame = wire.encode_frame(value, {"timestamp": "t", "outputs": {}})
    with pytest.raises(cache_module.CacheCorrupt):
        decode_entry(frame)


def test_a_value_that_isnt_data_stays_in_memory_only(tmp_path):
    store = CacheStore(tmp_path)
    def obj():  # a function: neither data nor allowlisted
        return None

    assert store.get(KEY) is None
    store.set(KEY, {"out": obj})
    assert store.get(KEY).outputs["out"] is obj
    assert not (tmp_path / f"{KEY}.mmw").exists()
    assert CacheStore(tmp_path).get(KEY) is None


def test_a_key_never_names_a_path_outside_the_directory(tmp_path):
    store = CacheStore(tmp_path / "cache")
    store.set("../escape", {"out": 1})
    assert store.get("../escape").outputs["out"] == 1
    assert not (tmp_path / "escape.mmw").exists()
