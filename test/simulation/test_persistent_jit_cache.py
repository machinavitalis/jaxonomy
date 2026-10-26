# SPDX-License-Identifier: MIT

"""Evidence for ``docs/jit_cache.md``: what the persistent cache stores, that
a later process reuses it for other parameter and initial values, and what
repeated runs and repeated gradients compile within one process.

Every case runs ``benchmarks/jit_cache.py``'s child in a fresh process. The
cache is process-global JAX configuration and must not leak into the test
session, and "a later process" can only be shown with one.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

_BENCH = Path(__file__).resolve().parents[2] / "benchmarks" / "jit_cache.py"


def _child(model: str, mode: str, cache_dir="") -> dict:
    # These cases are about the cache, so a session-wide switch for it (a CI
    # job or a test gate that turns JAX's disk cache off) must not reach them.
    env = {
        k: v
        for k, v in os.environ.items()
        if k not in ("JAX_ENABLE_COMPILATION_CACHE", "JAX_COMPILATION_CACHE_DIR")
    }
    p = subprocess.run(
        [sys.executable, str(_BENCH), "--child", model, mode, str(cache_dir)],
        capture_output=True,
        text=True,
        timeout=170,
        env=env,
    )
    assert p.returncode == 0, p.stderr[-2000:]
    return json.loads(p.stdout.strip().splitlines()[-1])


def test_small_model_is_stored_and_reused_by_a_later_process(tmp_path):
    cache = tmp_path / "jit"
    first = _child("pid_loop", "first", cache)
    # A 6-block model compiles in a fraction of a second and is stored anyway.
    assert first["new_entries"] > 0
    # Other parameter values and other initial conditions: nothing new to compile.
    later = _child("pid_loop", "other", cache)
    assert later["new_entries"] == 0


def test_sweeps_reuse_the_kernel_through_the_context_and_through_copies(tmp_path):
    out = _child("pid_loop", "sweep", tmp_path / "jit")
    assert out["first"][1] >= 1
    assert out["again"][1] == 0
    assert out["other_state"][1] == 0
    assert all(compiles == 0 for _, compiles in out["context"])
    # Diagram.with_parameters returns a new diagram, which runs on the kernel
    # of the one it was copied from: nothing is compiled and nothing stored.
    assert all(compiles == 0 for _, compiles in out["diagram"])
    assert out["diagram_new_entries"] == 0


def test_repeated_gradients_compile_on_the_first_call_only():
    for mode in ("grad_bare", "grad_jit"):
        calls = _child("pid_loop", mode)["calls"]
        assert calls[0][1] >= 1
        assert all(compiles == 0 for _, compiles in calls[1:]), mode
