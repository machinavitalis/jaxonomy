# SPDX-License-Identifier: MIT
"""
Helpers for T-002 determinism tests.

``assert_bitwise_reproducible`` runs a zero-arg callable twice, flattens
every returned pytree, and asserts that the raw bytes of each leaf match
between the two runs. On mismatch it reports which leaf diverged, the
first differing element index, and both hex values — so diagnostics don't
stop at "values differ".
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import jax


class NonDeterministic(AssertionError):
    """Raised when a simulation that should be bit-exact-reproducible isn't."""


@dataclass
class _LeafDiff:
    path: str
    dtype: str
    shape: tuple
    first_bad_index: tuple
    hex1: str
    hex2: str

    def __str__(self) -> str:
        return (
            f"leaf {self.path} (dtype={self.dtype}, shape={self.shape}) "
            f"differs at index {self.first_bad_index}: "
            f"run1=0x{self.hex1} run2=0x{self.hex2}"
        )


def _format_path(path) -> str:
    parts = []
    for key in path:
        if hasattr(key, "idx"):
            parts.append(f"[{key.idx}]")
        elif hasattr(key, "key"):
            parts.append(f".{key.key}")
        elif hasattr(key, "name"):
            parts.append(f".{key.name}")
        else:
            parts.append(f"[{key!r}]")
    return "".join(parts) or "<root>"


def assert_bitwise_reproducible(
    run: Callable[[], Any],
    *,
    label: str = "simulation",
    runs: int = 2,
) -> None:
    """Run ``run()`` ``runs`` times and assert every run is byte-identical.

    Args:
        run: Zero-argument callable returning a pytree of arrays.
        label: Descriptive name of the scenario for error messages.
        runs: Number of back-to-back runs to compare (default 2).  More
            runs catch intermittent non-determinism (e.g. GPU races) but
            extend test runtime linearly.
    """
    if runs < 2:
        raise ValueError("assert_bitwise_reproducible needs at least 2 runs")

    outputs = [run() for _ in range(runs)]
    leaves_runs = [jax.tree_util.tree_leaves_with_path(out) for out in outputs]

    # All trees must have matching structure — compare leaf paths.
    paths_0 = [_format_path(p) for p, _ in leaves_runs[0]]
    for i, lr in enumerate(leaves_runs[1:], start=2):
        paths_i = [_format_path(p) for p, _ in lr]
        if paths_i != paths_0:
            raise NonDeterministic(
                f"{label}: pytree structure differs between run 1 and run {i}: "
                f"{paths_0} vs {paths_i}"
            )

    # For each leaf, compare raw bytes of each run's value against run 1.
    for li, (path, leaf0) in enumerate(leaves_runs[0]):
        arr0 = np.asarray(leaf0)
        bytes0 = arr0.tobytes()
        for ri, lr in enumerate(leaves_runs[1:], start=2):
            arr_i = np.asarray(lr[li][1])
            if arr_i.tobytes() == bytes0:
                continue
            # Surface the first differing element.
            if arr0.shape != arr_i.shape or arr0.dtype != arr_i.dtype:
                raise NonDeterministic(
                    f"{label}: leaf {_format_path(path)} shape/dtype differs "
                    f"between run 1 and run {ri}: "
                    f"({arr0.shape}, {arr0.dtype}) vs ({arr_i.shape}, {arr_i.dtype})"
                )
            flat0 = arr0.reshape(-1)
            flati = arr_i.reshape(-1)
            diff_idx = np.flatnonzero(flat0.view(np.uint8) != flati.view(np.uint8))
            # Recover element (not byte) index.
            itemsize = arr0.dtype.itemsize
            elem_idx = int(diff_idx[0] // itemsize)
            hex1 = flat0[elem_idx].tobytes().hex()
            hex2 = flati[elem_idx].tobytes().hex()
            raise NonDeterministic(
                str(
                    _LeafDiff(
                        path=f"{label}:{_format_path(path)}",
                        dtype=str(arr0.dtype),
                        shape=tuple(arr0.shape),
                        first_bad_index=tuple(
                            np.unravel_index(elem_idx, arr0.shape)
                        ) if arr0.shape else (),
                        hex1=hex1,
                        hex2=hex2,
                    )
                )
                + f" (run 1 vs run {ri})"
            )


def assert_not_bitwise_equal(
    run1: Callable[[], Any],
    run2: Callable[[], Any],
    *,
    label: str = "negative-control",
) -> None:
    """Assert two configurations produce *different* outputs.

    Used to guard against a test that would trivially pass because its
    output is constant (e.g. the simulation is crashing silently or
    returning the initial condition). If two deliberately-different
    configurations produce byte-identical outputs, something is wrong
    with the test setup.
    """
    out1 = run1()
    out2 = run2()
    leaves1 = jax.tree_util.tree_leaves(out1)
    leaves2 = jax.tree_util.tree_leaves(out2)
    if len(leaves1) != len(leaves2):
        return  # structurally different → definitely not equal
    any_different = any(
        np.asarray(a).tobytes() != np.asarray(b).tobytes()
        for a, b in zip(leaves1, leaves2)
    )
    if not any_different:
        raise AssertionError(
            f"{label}: two configurations produced byte-identical outputs. "
            "The test is not exercising the axis it claims to exercise."
        )
