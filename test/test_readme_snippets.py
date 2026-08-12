# SPDX-License-Identifier: MIT
"""Executable-documentation gate: every ```python block in README.md must run
*and* must actually demonstrate something.

The README's code samples are the first thing a new user copies. This test
extracts each fenced ``python`` block from README.md and executes it in an
isolated namespace, so a snippet can never silently drift from the real API
(wrong kwarg, renamed method, dead callback) without turning CI red.

Executing is necessary but not sufficient. A snippet can run clean and still
teach nothing: `simulate` without ``recorded_signals=`` returns a
``SimulationResults`` whose ``time`` and ``outputs`` are both ``None``, and a
plant started at its setpoint produces a trajectory that is identically zero.
Both shipped in this README for months precisely because "it did not raise" was
the whole bar. So any ``SimulationResults`` a block leaves bound at top level
must carry a recorded time series, and at least one recorded signal must vary —
if the reader can reach the object, there has to be something in it.

Each block is expected to be self-contained (its own imports). If you add a
block that is an intentional fragment, mark its fence as ```py-skip`` instead
of ```python`` and it will be ignored here. A block that deliberately reads only
``results.context`` (a gradient or final-state example) should keep the results
object function-local rather than binding it at top level.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("optax")  # README's parameter-ID snippet uses optax

from jaxonomy.simulation import SimulationResults  # noqa: E402

README = Path(__file__).resolve().parents[1] / "README.md"

_BLOCK_RE = re.compile(r"```python\n(.*?)```", re.DOTALL)


def _blocks() -> list[tuple[int, str]]:
    text = README.read_text(encoding="utf-8")
    return list(enumerate(_BLOCK_RE.findall(text), start=1))


def _label(case: tuple[int, str]) -> str:
    idx, code = case
    first = next(
        (
            ln.strip()
            for ln in code.splitlines()
            if ln.strip() and not ln.strip().startswith("#")
        ),
        "",
    )
    return f"block{idx:02d}-{first[:40]}"


def test_readme_has_expected_block_count():
    # Guard against a refactor silently dropping every snippet (which would
    # make the parametrized test vacuously pass).
    assert len(_blocks()) >= 6, "expected the README to carry its worked examples"


def _varies(array) -> bool:
    """True if a recorded signal is not a single repeated value."""
    values = np.asarray(array)
    if values.size == 0:
        return False
    if not np.issubdtype(values.dtype, np.number):
        return False
    return bool(np.ptp(values) > 1e-9)


def _assert_demonstrates_something(name: str, results: SimulationResults) -> None:
    assert results.time is not None, (
        f"`{name}` has no time series: `simulate` was called without "
        "`recorded_signals=`, so `results.time` and `results.outputs` are both "
        "None. Name the signals to record, or keep the results object "
        "function-local if the snippet only reads `results.context`."
    )
    assert results.outputs, f"`{name}.outputs` is empty — nothing was recorded."
    assert any(_varies(signal) for signal in results.outputs.values()), (
        f"every signal recorded in `{name}` is constant over the run "
        f"({sorted(results.outputs)}). The snippet executes but demonstrates "
        "nothing — check the initial condition and the excitation."
    )


@pytest.mark.parametrize("case", _blocks(), ids=_label)
def test_readme_python_block_executes(case):
    _idx, code = case
    namespace: dict = {"__name__": "__readme_snippet__"}
    exec(compile(code, f"{README.name}:block", "exec"), namespace)

    for name, value in namespace.items():
        if isinstance(value, SimulationResults):
            _assert_demonstrates_something(name, value)
