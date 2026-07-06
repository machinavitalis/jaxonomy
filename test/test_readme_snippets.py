# SPDX-License-Identifier: MIT
"""Executable-documentation gate: every ```python block in README.md must run.

The README's code samples are the first thing a new user copies. This test
extracts each fenced ``python`` block from README.md and executes it in an
isolated namespace, so a snippet can never silently drift from the real API
(wrong kwarg, renamed method, dead callback) without turning CI red.

Each block is expected to be self-contained (its own imports). If you add a
block that is an intentional fragment, mark its fence as ```py-skip`` instead
of ```python`` and it will be ignored here.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

pytest.importorskip("optax")  # README's parameter-ID snippet uses optax

README = Path(__file__).resolve().parents[1] / "README.md"

_BLOCK_RE = re.compile(r"```python\n(.*?)```", re.DOTALL)


def _blocks() -> list[tuple[int, str]]:
    text = README.read_text(encoding="utf-8")
    return list(enumerate(_BLOCK_RE.findall(text), start=1))


def _label(case: tuple[int, str]) -> str:
    idx, code = case
    first = next(
        (ln.strip() for ln in code.splitlines()
         if ln.strip() and not ln.strip().startswith("#")),
        "",
    )
    return f"block{idx:02d}-{first[:40]}"


def test_readme_has_expected_block_count():
    # Guard against a refactor silently dropping every snippet (which would
    # make the parametrized test vacuously pass).
    assert len(_blocks()) >= 6, "expected the README to carry its worked examples"


@pytest.mark.parametrize("case", _blocks(), ids=_label)
def test_readme_python_block_executes(case):
    _idx, code = case
    namespace: dict = {"__name__": "__readme_snippet__"}
    exec(compile(code, f"{README.name}:block", "exec"), namespace)
