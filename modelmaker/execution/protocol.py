"""The runner protocol: how a client (RemoteExecutor, remote.py) and a
runner agent (agent.py) talk over one stream socket.

Framing. Every message is an 8-byte big-endian length followed by that
many bytes, which are one wire frame (wire.py: `MMW1`, a JSON header,
binary blobs). The receiver checks the length against its cap *before*
reading the frame, so an oversized message costs nothing but the
connection.

Every frame's `fields` carry an `op`:

client -> agent
  hello   {"protocol": PROTOCOL, "version": VERSION, "features": [...]}
  task    {"task": <int id>, "entry": <run_worker entry point name>,
           "env": {name: value} | null, "cwd": str | null,
           "limits": {"wall_seconds": float | null, "result_bytes": int | null}}
          value: the entry point's args, as a tuple -- wire-encoded, never
          pickled as a whole (see below)
  cancel  {"task": <id>}
  ping    {"seq": <int>}
  bye     {}

agent -> client
  hello   {"protocol", "version", "features", "pid"}
  result  {"task", "ok", "error", "crashed", "cancelled"}
          value: the task's result, exactly as the task process encoded it
          (the agent relays it without decoding it)
  pong    {"seq"}
  error   {"message"}   -- fatal; the agent closes the connection after it

The client sends `hello` first; the agent answers with its own `hello`, or
with `error` and a close if the protocol or version differs. Either side
treats anything unexpected -- a bad frame, an unknown op, a result for a
task it doesn't know -- as the end of the connection.

What crosses in which direction. A task's args are encoded by wire.py's
encoder like a result is: dataframes as Arrow IPC, JSON-shaped values as
tagged JSON, numpy arrays as raw buffers. The one interim exception is the
allowlisted pickle wire.py still admits for fitted scikit-learn / scipy
objects (a `$pickle` node), which can travel to a task when an upstream
block returned one; the agent never decodes args at all (it hands the
frame to the task process, which does), so it never unpickles anything.
Results come back through the client's own validating decoder, the same
one LocalExecutor uses: nothing the agent or a task sends is unpickled
except through safe_pickle's allowlist.

Extension points. `features` in the hellos is how optional capabilities
are negotiated without a version bump: a feature is used only when both
sides list it. Content-addressed values (a `{"$ref": "sha256:..."}` value
node, fetched over this same socket) are meant to arrive that way; the tag
is reserved and today's decoder rejects it as unknown.
"""

from __future__ import annotations

import socket
import struct
from typing import Any

from . import wire

PROTOCOL = "modelmaker-runner"
VERSION = 1
FEATURES: tuple[str, ...] = ()

_LEN = struct.Struct(">Q")

# Room for the envelope fields a relay adds around a result of the
# maximum size (task id, error text, flags).
ENVELOPE_SLACK = 1024 * 1024
# Cap on a control message (hello, cancel, ping, ...): they're tiny.
MAX_CONTROL_BYTES = 1024 * 1024
# Error and env limits on what one message may carry.
MAX_ERROR_CHARS = 20_000
MAX_ENV_VARS = 10_000
MAX_ENV_CHARS = 1024 * 1024


class ProtocolError(ConnectionError):
    """The peer broke the protocol (or isn't speaking it): the connection
    is over."""


def send_msg(sock: socket.socket, frame: bytes) -> None:
    """One message. Not thread-safe: callers sharing a socket hold a lock."""
    sock.sendall(_LEN.pack(len(frame)))
    sock.sendall(frame)


def _recv_exactly(sock: socket.socket, n: int) -> bytearray | None:
    buf = bytearray(n)
    view = memoryview(buf)
    got = 0
    while got < n:
        k = sock.recv_into(view[got:], min(n - got, 4 * 1024 * 1024))
        if k == 0:
            return None
        got += k
    return buf


def recv_msg(sock: socket.socket, max_bytes: int) -> bytearray | None:
    """The next message's frame, or None at a clean end of stream (the peer
    closed between messages). Raises ProtocolError for a message announced
    larger than `max_bytes` (before reading it) or a stream cut mid-message."""
    head = _recv_exactly(sock, _LEN.size)
    if head is None:
        return None
    (length,) = _LEN.unpack(head)
    if length > max_bytes:
        raise ProtocolError(f"peer sent a {length:,}-byte message, over the {max_bytes:,}-byte limit")
    body = _recv_exactly(sock, length)
    if body is None:
        raise ProtocolError("connection closed mid-message")
    return body


def message(op: str, value: Any = None, **fields: Any) -> bytes:
    """A message frame: `op` plus `fields`, and an optional value."""
    return wire.encode_frame({"op": op, **fields}, value, max_bytes=wire.DEFAULT_MAX_BYTES)


def read_fields(frame: bytes | bytearray, max_bytes: int) -> dict[str, Any]:
    """A message's fields (its value not decoded). ProtocolError if the
    frame is malformed or has no string `op`."""
    try:
        header, _ = wire.read_header(frame, max_bytes=max_bytes)
    except wire.WireError as e:
        raise ProtocolError(f"bad frame: {e}") from None
    fields = header["fields"]
    if not isinstance(fields.get("op"), str):
        raise ProtocolError("message without an op")
    return fields


def hello_fields(**extra: Any) -> dict[str, Any]:
    return {"protocol": PROTOCOL, "version": VERSION, "features": list(FEATURES), **extra}


def check_hello(fields: dict[str, Any]) -> None:
    """ProtocolError unless `fields` is a hello for this protocol and version."""
    if fields.get("op") != "hello":
        raise ProtocolError(f"expected a hello, got {fields.get('op')!r}")
    if fields.get("protocol") != PROTOCOL:
        raise ProtocolError(f"peer speaks {fields.get('protocol')!r}, not {PROTOCOL!r}")
    if fields.get("version") != VERSION:
        raise ProtocolError(f"peer speaks protocol version {fields.get('version')!r}, this side {VERSION}")
