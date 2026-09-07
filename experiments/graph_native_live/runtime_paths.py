"""Platform-neutral environment handling for optional native experiments."""

from __future__ import annotations

import os
import sys
from typing import Mapping


def native_environment(
    base: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Return an environment honoring an explicitly configured native library.

    No installation directory is guessed. If ``OLLAMA_LIB_DIR`` is unset, the
    operating system's normal dynamic-library search path is left unchanged.
    """

    environment = dict(os.environ if base is None else base)
    library_directory = environment.get("OLLAMA_LIB_DIR", "").strip()
    if not library_directory:
        return environment

    search_variable = (
        "DYLD_LIBRARY_PATH" if sys.platform == "darwin" else "LD_LIBRARY_PATH"
    )
    existing = environment.get(search_variable, "").strip()
    entries = [library_directory]
    if existing:
        entries.append(existing)
    environment[search_variable] = os.pathsep.join(entries)
    return environment


__all__ = ["native_environment"]
