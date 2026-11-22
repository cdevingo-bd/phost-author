"""Import generated or submitted Python source as a module.

Warp reads a kernel's source from the file it was defined in, so source that was exec'd from a string cannot define
kernels. This writes the source to a file in a private temporary directory (removed at exit) and imports it from there.
The module is named by the hash of its source: equal source is one module, different source never collides, and loading
the same source again returns the module that is already built (its kernels are not compiled twice).
"""

from __future__ import annotations

import atexit
import hashlib
import importlib.util
import os
import shutil
import sys
import tempfile
import threading
from types import ModuleType

_LOCK = threading.Lock()
_DIRECTORY: list[str] = []


def _directory() -> str:
    if not _DIRECTORY:
        path = tempfile.mkdtemp(prefix="phost_modules_")
        _DIRECTORY.append(path)
        atexit.register(shutil.rmtree, path, ignore_errors=True)
    return _DIRECTORY[0]


def load_source(source: str, prefix: str = "phost_generated") -> ModuleType:
    """The module defined by ``source``, built on first use. Raises whatever executing the source raises, and leaves no
    half-built module registered when it does."""
    name = f"{prefix}_{hashlib.sha256(source.encode()).hexdigest()[:16]}"
    with _LOCK:
        if name in sys.modules:
            return sys.modules[name]
        path = os.path.join(_directory(), f"{name}.py")
        with open(path, "w") as stream:
            stream.write(source)
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        try:
            spec.loader.exec_module(module)
        except BaseException:
            sys.modules.pop(name, None)
            raise
        return module
