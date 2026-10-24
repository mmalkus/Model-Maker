from .block_library import load_builtin as _load_builtin

# Parse every built-in block file into BLOCK_REGISTRY (see block_library.py)
# -- in the app, the tests and every worker process alike.
_load_builtin()
