#!/usr/bin/env python3
"""Check local Markdown links in tracked and newly added documentation."""

from __future__ import annotations

from pathlib import Path
import re
import subprocess


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LINK = re.compile(r"!?\[[^\]]*\]\(([^)]+)\)")


def markdown_files() -> tuple[Path, ...]:
    completed = subprocess.run(
        [
            "git",
            "-C",
            str(PROJECT_ROOT),
            "ls-files",
            "--cached",
            "--others",
            "--exclude-standard",
            "*.md",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return tuple(
        PROJECT_ROOT / line
        for line in completed.stdout.splitlines()
        if line and (PROJECT_ROOT / line).is_file()
    )


def main() -> int:
    missing: list[str] = []
    for document in markdown_files():
        text = document.read_text(encoding="utf-8")
        for match in LINK.finditer(text):
            raw = match.group(1).strip().strip("<>")
            target_text = raw.split("#", 1)[0]
            if not target_text or "://" in target_text or target_text.startswith("mailto:"):
                continue
            target = (document.parent / target_text).resolve()
            if not target.exists():
                line = text.count("\n", 0, match.start()) + 1
                relative = document.relative_to(PROJECT_ROOT)
                missing.append(f"{relative}:{line}: {raw}")
    if missing:
        print("Broken local documentation links:")
        for item in missing:
            print(f"- {item}")
        return 1
    print("Documentation link check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
