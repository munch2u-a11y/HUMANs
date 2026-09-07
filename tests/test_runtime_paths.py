from __future__ import annotations

import os
import sys

from experiments.graph_native_live.runtime_paths import native_environment


def test_native_environment_does_not_guess_an_installation_path() -> None:
    assert native_environment({"EXAMPLE": "1"}) == {"EXAMPLE": "1"}


def test_native_environment_prepends_only_an_explicit_library_directory() -> None:
    variable = "DYLD_LIBRARY_PATH" if sys.platform == "darwin" else "LD_LIBRARY_PATH"
    result = native_environment(
        {
            "OLLAMA_LIB_DIR": "relative-or-user-selected-lib",
            variable: "existing-search-path",
        }
    )
    assert result[variable] == os.pathsep.join(
        ("relative-or-user-selected-lib", "existing-search-path")
    )
