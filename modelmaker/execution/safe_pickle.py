"""A restricted unpickler for the few result values the wire format can't
express as data (see wire.py's `$pickle` node).

INTERIM -- to be replaced by opaque handles once block code runs in
persistent sandboxes: there, a value that isn't data (a fitted
estimator, say) stays inside the sandbox that made it and the control plane
only ever sees `{"kind": "handle", "ref": ...}`. Until session sandboxes
exist there is nowhere to keep such a value but this process, so it still
has to cross the boundary -- and the only way to make that crossing not
mean "the block runs code here" is to refuse every global the payload
names unless it is on an explicit allowlist.

Why this shape of allowlist, and not module prefixes for everything:
`pickle` can call any global it can look up (REDUCE), with arguments of
the payload's choosing. A prefix like `numpy.` would admit `numpy.load`
(which unpickles a file of the attacker's choosing) and `numpy.testing`
(which shells out), so *functions* are only ever allowed by exact name.
Classes are allowed by module prefix, because constructing a scikit-learn
estimator or a scipy sparse matrix, or restoring one's __dict__, has no
side effects worth having -- and even then only a class *defined* under
that prefix (`cls.__module__`), so a re-export such as `from subprocess
import Popen` inside an allowed module doesn't slip through.

Nothing in the core test suite needs this path at all: every built-in
block returns dataframes, JSON-shaped dicts or bytes (see wire.py). It
exists so a library block that hands, say, a fitted sklearn estimator to
another block over an `any` port keeps working.
"""

from __future__ import annotations

import importlib
import io
import pickle
import re
from typing import Any

# Exact (module, name) pairs: the reconstructors that pickles of numpy
# arrays/scalars/dtypes/random states and of plain objects go through.
# Both numpy 1.x (`numpy.core`) and 2.x (`numpy._core`) spellings.
ALLOWED_GLOBALS: frozenset[tuple[str, str]] = frozenset(
    {
        ("builtins", "set"),
        ("builtins", "frozenset"),
        ("builtins", "complex"),
        ("builtins", "slice"),
        ("builtins", "range"),
        ("builtins", "bytearray"),
        ("builtins", "object"),
        ("copyreg", "_reconstructor"),
        ("collections", "OrderedDict"),
        ("numpy", "dtype"),
        ("numpy", "ndarray"),
        ("numpy.core.multiarray", "_reconstruct"),
        ("numpy.core.multiarray", "scalar"),
        ("numpy._core.multiarray", "_reconstruct"),
        ("numpy._core.multiarray", "scalar"),
        ("numpy.core.numeric", "_frombuffer"),
        ("numpy._core.numeric", "_frombuffer"),
        ("numpy.random._pickle", "__randomstate_ctor"),
        ("numpy.random._pickle", "__generator_ctor"),
        ("numpy.random._pickle", "__bit_generator_ctor"),
        ("numpy.random.mtrand", "RandomState"),
        ("numpy.random._mt19937", "MT19937"),
        ("numpy.random._pcg64", "PCG64"),
        ("numpy.random._generator", "Generator"),
    }
)

# Modules under which any *class* (never a function) may be named.
ALLOWED_CLASS_PREFIXES: tuple[str, ...] = ("sklearn.", "scipy.sparse.")

# numpy's dtype classes (numpy.dtypes.Float64DType, ...) appear in numpy 2
# pickles of dtypes; they're types with no constructor side effects.
_NUMPY_DTYPE_MODULES = frozenset({"numpy.dtypes"})

_IDENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _in_prefix(module: str, prefixes: tuple[str, ...]) -> bool:
    return any(module == p.rstrip(".") or module.startswith(p) for p in prefixes)


def is_allowed_type(cls: type) -> bool:
    """Whether instances of `cls` may travel as a `$pickle` node at all --
    checked in the task process too, so a block returning something that
    could never be accepted gets a clear error there instead of a rejected
    payload here. Not a security check: the unpickler below is."""
    module = getattr(cls, "__module__", "") or ""
    return _in_prefix(module, ALLOWED_CLASS_PREFIXES) or (module, cls.__qualname__) in ALLOWED_GLOBALS


class _RestrictedUnpickler(pickle.Unpickler):
    def find_class(self, module: str, name: str) -> Any:
        # No dotted names: protocol 4 lets `name` be an attribute path
        # (`Cls.method.__globals__`), which would walk out of any allowlist.
        if not _IDENT.match(name):
            raise pickle.UnpicklingError(f"global '{module}.{name}' is not allowed in a block result")
        if (module, name) in ALLOWED_GLOBALS:
            return super().find_class(module, name)
        if _in_prefix(module, ALLOWED_CLASS_PREFIXES) or module in _NUMPY_DTYPE_MODULES:
            try:
                obj = getattr(importlib.import_module(module), name)
            except (ImportError, AttributeError):
                raise pickle.UnpicklingError(f"global '{module}.{name}' not found") from None
            defined_in = getattr(obj, "__module__", "") or ""
            if isinstance(obj, type) and (
                _in_prefix(defined_in, ALLOWED_CLASS_PREFIXES) or defined_in in _NUMPY_DTYPE_MODULES
            ):
                return obj
        raise pickle.UnpicklingError(f"global '{module}.{name}' is not allowed in a block result")

    def persistent_load(self, pid: Any) -> Any:
        raise pickle.UnpicklingError("persistent ids are not allowed in a block result")


def restricted_loads(data: bytes) -> Any:
    """`pickle.loads`, except that every global the payload names must be on
    the allowlist above; anything else raises pickle.UnpicklingError before
    it is ever called."""
    return _RestrictedUnpickler(io.BytesIO(data)).load()
