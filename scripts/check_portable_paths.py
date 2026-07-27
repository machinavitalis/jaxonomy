#!/usr/bin/env python3
"""Portability check: reject machine- or session-specific paths in tracked files.

Examples should be reproducible: their code and their *saved outputs* must not
depend on where any one contributor happened to run them. The most common way a
non-portable path sneaks in is through executed-notebook outputs — Python
warnings and tracebacks print absolute source paths, and `nbconvert` bakes that
stderr into the saved `.ipynb`. This check greps every tracked text file
(notebooks included, since their outputs are JSON strings) for such paths and
fails if any are found.

Flagged: home directories (``/Users/<name>``, ``/home/<name>``,
``C:\\Users\\<name>``), per-user OS temp dirs (``/var/folders/...``), Jupyter
kernel temp files (``ipykernel_<pid>``), git worktree paths (``.claude/
worktrees/<branch>``), and random temp filenames. The CI runner's workspace
(``/home/runner``) is allow-listed.

Usage:  python scripts/check_portable_paths.py
"""
from __future__ import annotations

import os
import re
import subprocess
import sys

SELF = "scripts/check_portable_paths.py"

PATTERNS = [
    re.compile(r"/Users/[^/\s\"']+"),
    re.compile(r"/home/(?!runner/)[^/\s\"']+"),
    re.compile(r"[A-Za-z]:\\Users\\[^\\\s\"']+"),
    re.compile(r"/var/folders/[A-Za-z0-9_+]"),
    re.compile(r"ipykernel_\d"),
    re.compile(r"\.claude/worktrees/"),
    re.compile(r"/tmp/[A-Za-z0-9+/=]{16,}"),
]
MAX_REPORT = 50


def tracked_files() -> list[str]:
    out = subprocess.run(["git", "ls-files"], capture_output=True, text=True, check=True)
    return out.stdout.splitlines()


def main() -> int:
    offenders: list[tuple[str, int, str]] = []
    for path in tracked_files():
        if path == SELF:  # this file necessarily contains the patterns it searches for
            continue
        try:
            raw = open(path, "rb").read()
        except OSError:
            continue
        if b"\x00" in raw[:8192]:
            continue
        for lineno, line in enumerate(raw.decode("utf-8", "replace").splitlines(), 1):
            if "/home/runner/" in line:
                continue
            if any(p.search(line) for p in PATTERNS):
                offenders.append((path, lineno, line.strip()[:160]))

    if offenders:
        print(f"ERROR: {len(offenders)} non-portable path(s) in tracked files:\n")
        for path, lineno, snippet in offenders[:MAX_REPORT]:
            print(f"  {path}:{lineno}: {snippet}")
        if len(offenders) > MAX_REPORT:
            print(f"  ... and {len(offenders) - MAX_REPORT} more")
        print(
            "\nRelativise these before committing. For notebooks, the path usually\n"
            "lives in a saved warning/traceback output: re-run with warnings/tracebacks\n"
            "suppressed (PYTHONWARNINGS=ignore / warnings.filterwarnings), or edit the\n"
            "saved cell output to a repo-relative form."
        )
        return 1

    print("OK: no machine- or session-specific paths in tracked files.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
