#!/usr/bin/env python3
"""Reject machine-specific paths from tracked Habitus source and docs."""

from __future__ import annotations

from pathlib import Path
import re
import subprocess


PROJECT_ROOT = Path(__file__).resolve().parents[1]
THIS_FILE = Path(__file__).resolve()
TEXT_SUFFIXES = {
    ".cff",
    ".cfg",
    ".cpp",
    ".ini",
    ".json",
    ".md",
    ".py",
    ".sh",
    ".toml",
    ".yaml",
    ".yml",
}
CODE_SUFFIXES = {
    ".cff",
    ".cfg",
    ".cpp",
    ".ini",
    ".json",
    ".py",
    ".sh",
    ".toml",
    ".yaml",
    ".yml",
}
CODE_FILENAMES = {"Makefile"}

# Fragments are joined so this checker does not report its own rule strings.
USER_PATH_PATTERNS = (
    ("Linux user home", re.compile("/" + r"home/[^/\s`'\"]+")),
    ("macOS user home", re.compile("/" + r"Users/[^/\s`'\"]+")),
    ("Windows user home", re.compile(r"[A-Za-z]:\\Users\\[^\\\s`'\"]+")),
)
ABSOLUTE_CODE_PATTERNS = (
    (
        "absolute local path literal",
        re.compile(
            r"[\"'](?:/" + r"(?:home|Users|tmp|usr/local)/|[A-Za-z]:\\Users\\)"
        ),
    ),
    ("absolute pathlib literal", re.compile(r"\bPath\([\"']/")),
    (
        "absolute local assignment",
        re.compile(
            r"(?m)^[A-Za-z_][A-Za-z0-9_]*(?:\s*[?:+]?=)\s*"
            r"/(?:home|Users|tmp|usr/local)/"
        ),
    ),
)


def candidate_files() -> tuple[Path, ...]:
    completed = subprocess.run(
        [
            "git",
            "-C",
            str(PROJECT_ROOT),
            "ls-files",
            "--cached",
            "--others",
            "--exclude-standard",
            "-z",
        ],
        check=True,
        capture_output=True,
    )
    paths = []
    for raw in completed.stdout.split(b"\0"):
        if not raw:
            continue
        path = PROJECT_ROOT / raw.decode("utf-8")
        if path.is_file() and path.resolve() != THIS_FILE:
            paths.append(path)
    return tuple(paths)


def main() -> int:
    violations: list[str] = []
    for path in candidate_files():
        if path.suffix not in TEXT_SUFFIXES and path.name not in CODE_FILENAMES:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        relative = path.relative_to(PROJECT_ROOT)
        for label, pattern in USER_PATH_PATTERNS:
            for match in pattern.finditer(text):
                line = text.count("\n", 0, match.start()) + 1
                violations.append(f"{relative}:{line}: {label}")
        if path.suffix in CODE_SUFFIXES or path.name in CODE_FILENAMES:
            for label, pattern in ABSOLUTE_CODE_PATTERNS:
                for match in pattern.finditer(text):
                    line = text.count("\n", 0, match.start()) + 1
                    violations.append(f"{relative}:{line}: {label}")

    if violations:
        print("Machine-specific paths found:")
        for violation in violations:
            print(f"- {violation}")
        return 1
    print("Portability check passed: no machine-specific tracked paths found.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
