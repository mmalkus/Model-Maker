"""The wire format for values coming *back* from a task process.

A block's result used to come back pickled, and was unpickled here -- so a
block could return an object whose `__reduce__` ran whatever it liked in
this process. This module
is the replacement: a result is encoded into fixed, data-only formats in
the task process and decoded here without ever calling anything the payload
names (S6.3 of that doc):

- `pl.DataFrame` / `pl.Series` -> Arrow IPC stream (polars' own reader and
  writer; pyarrow isn't needed)
- `bytes` (what the `image` port carries: PNG bytes, see bin_chart and
  generate_image) -> raw bytes
- numpy arrays of plain (non-object) dtypes -> raw buffer plus dtype and
  shape; numpy bool/int/float scalars inline in the JSON, type kept
- JSON-shaped values (dicts, lists, str, numbers, bools, None -- what the
  `model`, `binning`, `scalar_metric`, stochastic ... ports actually carry)
  -> JSON, with tagged nodes for what JSON can't say on its own: tuples
  (a multi-output block returns one), non-finite floats, non-string or
  `$`-prefixed dict keys, sets, dates/times, Decimal
- anything else -> INTERIM: a pickle, decoded only by safe_pickle's
  allowlisting unpickler, and only for types it allows (fitted sklearn
  estimators, scipy sparse matrices). Opaque handles kept in a sandbox
  are meant to replace this; see safe_pickle.py.

The encoding is structural (by Python type), not by declared port type:
the same value shapes flow through `any` ports, library blocks whose
declared types are only advisory, and the runner's internal envelopes (a
fused group's `(schemas, results)`).

Frame layout: `MAGIC`, a 4-byte big-endian header length, the UTF-8 JSON
header, then the binary blobs back to back. The header is
`{"fields": {...}, "value": <node>, "blobs": [len, ...]}`; a blob node
refers to its blob by index. Every limit is checked on decode, which is the
security boundary; encode checks the same limits so a block that produces
too much fails with a clear message in its own process.
"""

from __future__ import annotations

import datetime as _dt
import decimal
import io
import json
import math
import os
import pickle
import struct
from typing import Any

MAGIC = b"MMW1"
_HEADER_LEN = struct.Struct(">I")

# Total frame size, default 8 GiB: generous because the local app
# legitimately moves big frames between processes (and had no cap at all
# before); a hosted profile sets its own `output_mb` (S6.5).
DEFAULT_MAX_BYTES = 8 * 1024**3
# The JSON header alone (every JSON-shaped value lives here, so a block
# that returns a huge Python list instead of a dataframe meets this one).
DEFAULT_MAX_HEADER_BYTES = 256 * 1024**2
# Nesting depth of the decoded value. Real results nest 3-5 deep.
MAX_DEPTH = 100
MAX_BLOBS = 100_000

# numpy dtype kinds that are plain bytes: bool, ints, floats, complex,
# datetimes/timedeltas, fixed-width bytes/str. Never 'O' (object) or 'V'
# (void/structured, which can embed object fields).
_NUMPY_KINDS = frozenset("biufcmMSU")


# numpy scalar kinds sent inline in the JSON (bool, ints, floats) rather
# than as a blob each: a dict of a few thousand np.float64 statistics is
# common, and shouldn't cost a few thousand blobs.
_INLINE_NUMPY_KINDS = frozenset("biuf")


def _is_numpy_scalar(v: Any) -> bool:
    import sys

    np = sys.modules.get("numpy")
    return np is not None and isinstance(v, np.generic)


class WireError(ValueError):
    """A value that can't be encoded, or a frame that fails validation."""


class ValueTooLarge(WireError):
    pass


def max_bytes_from_env() -> int:
    """MODELMAKER_MAX_RESULT_MB: the cap on one task's encoded result."""
    try:
        mb = int(os.environ.get("MODELMAKER_MAX_RESULT_MB", "0"))
    except ValueError:
        mb = 0
    return mb * 1024**2 if mb > 0 else DEFAULT_MAX_BYTES


def _fmt_mb(n: int) -> str:
    return f"{n / 1024**2:,.1f} MB"


# ---- encode (task process) ------------------------------------------------


class _Encoder:
    def __init__(self, max_bytes: int) -> None:
        self.blobs: list[bytes] = []
        self.total = 0
        self.max_bytes = max_bytes

    def blob(self, data: bytes) -> int:
        self.total += len(data)
        if self.total > self.max_bytes:
            raise ValueTooLarge(
                f"block result is over the {_fmt_mb(self.max_bytes)} limit (MODELMAKER_MAX_RESULT_MB)"
            )
        if len(self.blobs) >= MAX_BLOBS:
            raise ValueTooLarge(f"block result has more than {MAX_BLOBS} binary parts")
        self.blobs.append(bytes(data))
        return len(self.blobs) - 1

    def node(self, v: Any, depth: int = 0) -> Any:
        if depth > MAX_DEPTH:
            raise WireError(f"block result is nested more than {MAX_DEPTH} levels deep")
        if v is None or isinstance(v, (bool, str)):
            return v
        if _is_numpy_scalar(v) and v.dtype.kind in _INLINE_NUMPY_KINDS:
            # Before the float check: np.float64 *is* a float, but a block
            # that returned one should get one back, not a plain float.
            item = v.item()
            if isinstance(item, float) and not math.isfinite(item):
                item = repr(item)  # 'nan', 'inf', '-inf'
            return {"$np": v.dtype.str, "v": item}
        if isinstance(v, int):
            return int(v)
        if isinstance(v, float):
            if math.isfinite(v):
                return float(v)
            return {"$f": "nan" if math.isnan(v) else ("inf" if v > 0 else "-inf")}
        if isinstance(v, (bytes, bytearray, memoryview)):
            return {"$bytes": self.blob(bytes(v))}
        if isinstance(v, tuple):
            return {"$tuple": [self.node(x, depth + 1) for x in v]}
        if isinstance(v, list):
            return [self.node(x, depth + 1) for x in v]
        if isinstance(v, dict):
            if all(isinstance(k, str) and not k.startswith("$") for k in v):
                return {k: self.node(x, depth + 1) for k, x in v.items()}
            return {"$map": [[self.node(k, depth + 1), self.node(x, depth + 1)] for k, x in v.items()]}
        if isinstance(v, (set, frozenset)):
            return {"$set": [self.node(x, depth + 1) for x in v], "frozen": isinstance(v, frozenset)}
        # datetime before date: a datetime is a date.
        if isinstance(v, _dt.datetime):
            return {"$datetime": v.isoformat()}
        if isinstance(v, _dt.date):
            return {"$date": v.isoformat()}
        if isinstance(v, _dt.time):
            return {"$time": v.isoformat()}
        if isinstance(v, _dt.timedelta):
            return {"$timedelta": [v.days, v.seconds, v.microseconds]}
        if isinstance(v, decimal.Decimal):
            return {"$decimal": str(v)}
        node = self._polars(v)
        if node is None:
            node = self._numpy(v)
        if node is None:
            node = self._pickled(v)
        return node

    def _polars(self, v: Any) -> Any:
        import polars as pl

        if isinstance(v, pl.LazyFrame):
            raise WireError("a LazyFrame can't be sent back from a block (it should have been collected)")
        if isinstance(v, (pl.DataFrame, pl.Series)):
            frame = v if isinstance(v, pl.DataFrame) else v.to_frame()
            buf = io.BytesIO()
            try:
                frame.write_ipc_stream(buf)
            except Exception as e:  # noqa: BLE001 -- e.g. an Object column, which Arrow can't hold
                raise WireError(f"dataframe can't be sent back as Arrow: {type(e).__name__}: {e}") from None
            return {("$df" if isinstance(v, pl.DataFrame) else "$series"): self.blob(buf.getbuffer())}
        return None

    def _numpy(self, v: Any) -> Any:
        import sys

        np = sys.modules.get("numpy")
        if np is None or not isinstance(v, (np.ndarray, np.generic)):
            return None
        arr = np.asarray(v)
        if arr.dtype.kind not in _NUMPY_KINDS or arr.dtype.fields is not None:
            if isinstance(v, np.generic):
                return self._pickled(v)
            # An object array: its elements go through this same encoding
            # (so they must be data themselves), flattened, plus the shape.
            return {"$ndobj": [self.node(x, 1) for x in arr.ravel()], "shape": list(arr.shape)}
        return {
            "$nd": self.blob(np.ascontiguousarray(arr).tobytes()),
            "dtype": arr.dtype.str,
            "shape": list(arr.shape),
            "scalar": isinstance(v, np.generic),
        }

    def _pickled(self, v: Any) -> Any:
        from .safe_pickle import is_allowed_type

        if not is_allowed_type(type(v)):
            t = type(v)
            raise WireError(
                f"a block can't return a {t.__module__}.{t.__qualname__}: results must be dataframes, "
                "JSON-shaped values (dicts, lists, numbers, strings), bytes, numpy arrays, "
                "or scikit-learn / scipy.sparse objects"
            )
        try:
            data = pickle.dumps(v, protocol=pickle.HIGHEST_PROTOCOL)
        except Exception as e:  # noqa: BLE001 -- reported as the block's error
            raise WireError(f"can't send back a {type(v).__name__}: {type(e).__name__}: {e}") from None
        return {"$pickle": self.blob(data)}


def encode_frame(fields: dict[str, Any], value: Any = None, *, max_bytes: int | None = None) -> bytes:
    """`fields` (plain JSON: the envelope -- ok/error/...) plus one `value`
    (anything encodable, see the module docstring) as a single frame."""
    limit = max_bytes if max_bytes is not None else max_bytes_from_env()
    enc = _Encoder(limit)
    node = enc.node(value)
    try:
        header = json.dumps(
            {"fields": fields, "value": node, "blobs": [len(b) for b in enc.blobs]},
            allow_nan=False,
            separators=(",", ":"),
        ).encode()
    except (TypeError, ValueError) as e:
        raise WireError(f"envelope isn't plain JSON: {e}") from None
    header_cap = min(limit, DEFAULT_MAX_HEADER_BYTES)
    if len(header) > header_cap:
        raise ValueTooLarge(
            f"block result's JSON part is {_fmt_mb(len(header))}, over the {_fmt_mb(header_cap)} limit -- "
            "return large tables as a dataframe rather than as Python lists"
        )
    if len(header) + enc.total + len(MAGIC) + _HEADER_LEN.size > limit:
        raise ValueTooLarge(f"block result is over the {_fmt_mb(limit)} limit (MODELMAKER_MAX_RESULT_MB)")
    return b"".join([MAGIC, _HEADER_LEN.pack(len(header)), header, *enc.blobs])


def encode(value: Any, *, max_bytes: int | None = None) -> bytes:
    return encode_frame({}, value, max_bytes=max_bytes)


# ---- decode (this process: the security boundary) ------------------------


class _Decoder:
    def __init__(self, blobs: list[memoryview]) -> None:
        self.blobs = blobs

    def blob(self, ref: Any) -> memoryview:
        if type(ref) is not int or not 0 <= ref < len(self.blobs):
            raise WireError(f"bad blob reference {ref!r}")
        return self.blobs[ref]

    def node(self, n: Any, depth: int = 0) -> Any:
        if depth > MAX_DEPTH:
            raise WireError(f"value nested more than {MAX_DEPTH} levels deep")
        if n is None or isinstance(n, (bool, str, int, float)):
            return n
        if isinstance(n, list):
            return [self.node(x, depth + 1) for x in n]
        if not isinstance(n, dict):
            raise WireError(f"unexpected JSON node {type(n).__name__}")
        tag = next((k for k in n if k.startswith("$")), None)
        if tag is None:
            return {k: self.node(x, depth + 1) for k, x in n.items()}
        body = n[tag]
        if tag == "$tuple":
            return tuple(self.node(x, depth + 1) for x in _list(body))
        if tag == "$map":
            out: dict[Any, Any] = {}
            for pair in _list(body):
                if not isinstance(pair, list) or len(pair) != 2:
                    raise WireError("bad $map entry")
                key = self.node(pair[0], depth + 1)
                try:
                    out[key] = self.node(pair[1], depth + 1)
                except TypeError:
                    raise WireError("unhashable $map key") from None
            return out
        if tag == "$set":
            try:
                items = [self.node(x, depth + 1) for x in _list(body)]
                return frozenset(items) if n.get("frozen") else set(items)
            except TypeError:
                raise WireError("unhashable $set item") from None
        if tag == "$f":
            if body not in ("nan", "inf", "-inf"):
                raise WireError(f"bad $f value {body!r}")
            return float(body)
        if tag == "$np":
            return _read_numpy_scalar(body, n.get("v"))
        if tag == "$bytes":
            return bytes(self.blob(body))
        if tag in ("$df", "$series"):
            return _read_frame(self.blob(body), series=tag == "$series")
        if tag == "$nd":
            return _read_ndarray(self.blob(body), n)
        if tag == "$ndobj":
            import numpy as np

            items = [self.node(x, depth + 1) for x in _list(body)]
            shape = n.get("shape")
            if not isinstance(shape, list) or not all(type(d) is int and d >= 0 for d in shape):
                raise WireError("bad ndarray shape")
            if math.prod(shape) != len(items):
                raise WireError("ndarray shape doesn't match its items")
            arr = np.empty(len(items), dtype=object)
            for i, item in enumerate(items):  # item by item: numpy would broadcast a list
                arr[i] = item
            return arr.reshape(shape)
        if tag == "$pickle":
            from .safe_pickle import restricted_loads

            try:
                return restricted_loads(bytes(self.blob(body)))
            except pickle.UnpicklingError as e:
                raise WireError(str(e)) from None
            except Exception as e:  # noqa: BLE001 -- malformed payload
                raise WireError(f"malformed pickle ({type(e).__name__}: {e})") from None
        try:
            if tag == "$datetime":
                return _dt.datetime.fromisoformat(_str(body))
            if tag == "$date":
                return _dt.date.fromisoformat(_str(body))
            if tag == "$time":
                return _dt.time.fromisoformat(_str(body))
            if tag == "$timedelta":
                days, seconds, micros = _list(body)
                return _dt.timedelta(days=_int(days), seconds=_int(seconds), microseconds=_int(micros))
            if tag == "$decimal":
                return decimal.Decimal(_str(body))
        except (ValueError, TypeError, OverflowError, decimal.InvalidOperation) as e:
            raise WireError(f"bad {tag} value: {e}") from None
        raise WireError(f"unknown tag {tag!r}")


def _list(x: Any) -> list[Any]:
    if not isinstance(x, list):
        raise WireError("expected a list")
    return x


def _str(x: Any) -> str:
    if not isinstance(x, str) or len(x) > 64:
        raise WireError("expected a short string")
    return x


def _int(x: Any) -> int:
    if type(x) is not int:
        raise WireError("expected an int")
    return x


def _read_frame(data: memoryview, series: bool) -> Any:
    import polars as pl

    try:
        df = pl.read_ipc_stream(bytes(data))
    except (KeyboardInterrupt, SystemExit):
        raise
    except BaseException as e:  # noqa: BLE001 -- incl. a Rust panic (pyo3's PanicException is a BaseException)
        raise WireError(f"bad Arrow IPC stream: {type(e).__name__}: {e}") from None
    if series:
        if df.width != 1:
            raise WireError("a $series frame must have exactly one column")
        return df.to_series(0)
    return df


def _read_numpy_scalar(dtype_str: Any, v: Any) -> Any:
    import numpy as np

    if not isinstance(dtype_str, str) or len(dtype_str) > 16:
        raise WireError("bad numpy scalar dtype")
    try:
        dtype = np.dtype(dtype_str)
    except (TypeError, ValueError):
        raise WireError(f"bad numpy scalar dtype {dtype_str!r}") from None
    if dtype.kind not in _INLINE_NUMPY_KINDS:
        raise WireError(f"numpy scalar dtype {dtype_str!r} is not allowed inline")
    if isinstance(v, str) and dtype.kind == "f" and v in ("nan", "inf", "-inf"):
        v = float(v)
    numeric = isinstance(v, (int, float)) and not isinstance(v, bool)
    if not (isinstance(v, bool) if dtype.kind == "b" else numeric):
        raise WireError("bad numpy scalar value")
    try:
        return dtype.type(v)
    except (OverflowError, ValueError, TypeError):
        raise WireError(f"numpy scalar value out of range for {dtype_str!r}") from None


def _read_ndarray(data: memoryview, n: dict[str, Any]) -> Any:
    import numpy as np

    dtype_str, shape = n.get("dtype"), n.get("shape")
    if not isinstance(dtype_str, str) or len(dtype_str) > 16:
        raise WireError("bad ndarray dtype")
    try:
        dtype = np.dtype(dtype_str)
    except (TypeError, ValueError):
        raise WireError(f"bad ndarray dtype {dtype_str!r}") from None
    if dtype.kind not in _NUMPY_KINDS or dtype.fields is not None or dtype.hasobject:
        raise WireError(f"ndarray dtype {dtype_str!r} is not allowed")
    if not isinstance(shape, list) or len(shape) > 32 or not all(type(d) is int and d >= 0 for d in shape):
        raise WireError("bad ndarray shape")
    count = math.prod(shape)
    if count * dtype.itemsize != len(data):
        raise WireError("ndarray shape doesn't match its data")
    arr = np.frombuffer(bytes(data), dtype=dtype).reshape(shape).copy()
    return arr[()] if n.get("scalar") else arr


def decode_frame(data: bytes | bytearray | memoryview, *, max_bytes: int | None = None) -> tuple[dict[str, Any], Any]:
    """(fields, value) from a frame made by `encode_frame`. Raises WireError
    for anything malformed, oversized or not allowed; never calls a global
    named by the payload except through safe_pickle's allowlist."""
    limit = max_bytes if max_bytes is not None else max_bytes_from_env()
    view = memoryview(data)
    if len(view) > limit:
        raise ValueTooLarge(f"result is {_fmt_mb(len(view))}, over the {_fmt_mb(limit)} limit")
    head = len(MAGIC) + _HEADER_LEN.size
    if len(view) < head or bytes(view[: len(MAGIC)]) != MAGIC:
        raise WireError("not a result frame")
    (header_len,) = _HEADER_LEN.unpack(view[len(MAGIC) : head])
    if header_len > min(limit, DEFAULT_MAX_HEADER_BYTES) or head + header_len > len(view):
        raise WireError("bad header length")
    try:
        header = json.loads(bytes(view[head : head + header_len]))
    except (ValueError, RecursionError) as e:
        raise WireError(f"bad header: {type(e).__name__}") from None
    if not isinstance(header, dict) or not isinstance(header.get("fields"), dict):
        raise WireError("bad header")
    lengths = header.get("blobs")
    if not isinstance(lengths, list) or len(lengths) > MAX_BLOBS or not all(type(x) is int and x >= 0 for x in lengths):
        raise WireError("bad blob table")
    offset = head + header_len
    if offset + sum(lengths) != len(view):
        raise WireError("blob table doesn't match the frame size")
    blobs = []
    for length in lengths:
        blobs.append(view[offset : offset + length])
        offset += length
    value = _Decoder(blobs).node(header.get("value"))
    return header["fields"], value


def decode(data: bytes | bytearray | memoryview, *, max_bytes: int | None = None) -> Any:
    return decode_frame(data, max_bytes=max_bytes)[1]
