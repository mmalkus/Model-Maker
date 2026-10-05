"""The built-in local model: a small open model the app runs itself, on CPU,
through llama.cpp (llama-cpp-python, `pip install modelmaker[local]`). It
isn't bundled: Settings > AI builder downloads it with one click into
MODELMAKER_MODELS_DIR (default ~/.modelmaker/models) -- in the background,
resuming a partial download, checked against its SHA-256 before use.
MODELMAKER_LOCAL_MODEL_PATH points at a .gguf you already have instead."""

from __future__ import annotations

import hashlib
import os
import threading
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class LocalModelSpec:
    key: str
    name: str
    url: str
    filename: str
    size: int
    sha256: str
    license_url: str
    context_tokens: int


# MiniCPM5 2B, 4-bit (1.6 GB): small enough for a laptop CPU; what guided
# builds are tuned for (agent/guided.py, agent/loop.LocalLoop).
DEFAULT_MODEL = LocalModelSpec(
    key="minicpm5-2b",
    name="MiniCPM5 2B (Q4_K_M)",
    url="https://huggingface.co/openbmb/MiniCPM5-2B-GGUF/resolve/main/MiniCPM5-2B-Q4_K_M.gguf",
    filename="MiniCPM5-2B-Q4_K_M.gguf",
    size=1_561_318_368,
    sha256="ec2d5801640099e97d8d7e8003ad4d81f336e757811f03a26173dddf386602fd",
    license_url="https://huggingface.co/openbmb/MiniCPM5-2B-GGUF",
    context_tokens=16_384,
)


def model_path() -> Path:
    override = os.environ.get("MODELMAKER_LOCAL_MODEL_PATH")
    if override:
        return Path(override).expanduser()
    return Path(os.environ.get("MODELMAKER_MODELS_DIR") or Path.home() / ".modelmaker" / "models").expanduser() / DEFAULT_MODEL.filename


def is_downloaded() -> bool:
    path = model_path()
    return path.is_file() and (bool(os.environ.get("MODELMAKER_LOCAL_MODEL_PATH")) or path.stat().st_size == DEFAULT_MODEL.size)


def runtime_available() -> bool:
    try:
        import llama_cpp  # noqa: F401
    except ImportError:
        return False
    return True


class Download:
    """A background download into `<file>.part` (resumed with an HTTP Range),
    verified by size and SHA-256, then moved into place."""

    def __init__(self, spec: LocalModelSpec, dest: Path) -> None:
        self.spec, self.dest, self.part = spec, dest, dest.with_name(dest.name + ".part")
        self.done_bytes = self.part.stat().st_size if self.part.exists() else 0
        self.state, self.error = "downloading", None  # downloading | verifying | done | failed | cancelled
        self._cancel = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def status(self) -> dict[str, Any]:
        return {"state": self.state, "bytes": self.done_bytes, "total": self.spec.size, "error": self.error}

    def _run(self) -> None:
        try:
            self.dest.parent.mkdir(parents=True, exist_ok=True)
            headers = {"User-Agent": "modelmaker", **({"Range": f"bytes={self.done_bytes}-"} if self.done_bytes else {})}
            with urllib.request.urlopen(urllib.request.Request(self.spec.url, headers=headers), timeout=60) as resp:
                if resp.status != 206:  # the server sent the whole file: start over
                    self.done_bytes = 0
                with open(self.part, "ab" if self.done_bytes else "wb") as out:
                    while not self._cancel.is_set() and (chunk := resp.read(1 << 20)):
                        out.write(chunk)
                        self.done_bytes += len(chunk)
            if self._cancel.is_set():
                self.state = "cancelled"
                return
            self.state = "verifying"
            h = hashlib.sha256()
            with open(self.part, "rb") as f:
                for block in iter(lambda: f.read(1 << 22), b""):
                    h.update(block)
            if self.done_bytes != self.spec.size or h.hexdigest() != self.spec.sha256:
                self.part.unlink(missing_ok=True)
                raise RuntimeError("the downloaded file doesn't match its checksum -- download it again")
            os.replace(self.part, self.dest)
            self.state = "done"
        except Exception as e:  # noqa: BLE001 -- shown in the Settings panel
            self.state, self.error = "failed", str(e)


_download: Download | None = None


def start_download() -> None:
    global _download
    if _download is None or _download.state not in ("downloading", "verifying"):
        _download = Download(DEFAULT_MODEL, model_path())


def cancel_download() -> None:
    if _download is not None:
        _download._cancel.set()


def status() -> dict[str, Any]:
    """Everything the Settings panel shows about the local model."""
    spec = DEFAULT_MODEL
    return {
        "name": spec.name, "size": spec.size, "license_url": spec.license_url, "path": str(model_path()),
        "downloaded": is_downloaded(), "runtime_available": runtime_available(),
        "download": _download.status() if _download is not None else None,
    }


_loaded: Any = None
_load_lock = threading.Lock()


def load():
    """The llama.cpp model, loaded once and shared (~2 GB of memory). CPU by
    default; MODELMAKER_LOCAL_GPU_LAYERS offloads to a GPU build of llama.cpp."""
    global _loaded
    with _load_lock:
        if _loaded is None:
            if not runtime_available():
                raise RuntimeError("the local model needs llama-cpp-python -- pip install modelmaker[local]")
            if not model_path().is_file():
                raise RuntimeError(f"the local model isn't downloaded -- download {DEFAULT_MODEL.name} in Settings > AI builder")
            from llama_cpp import Llama

            _loaded = Llama(model_path=str(model_path()), n_ctx=DEFAULT_MODEL.context_tokens,
                            n_gpu_layers=int(os.environ.get("MODELMAKER_LOCAL_GPU_LAYERS", "0")), verbose=False)
        return _loaded
