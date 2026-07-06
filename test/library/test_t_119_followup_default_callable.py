# SPDX-License-Identifier: MIT
"""T-119-followup-default-callable: TruthTable callable ``default_output``.

Phase 1 + the numeric-output followup allowed CALLABLE row outputs but
required ``default_output`` to remain a constant scalar/array. This
followup extends the same callable semantics to ``default_output`` —
the fallback (used when no row pattern matches) may now be a callable
``f(*inputs) -> scalar_or_vector`` invoked with the RAW inputs.

Use case: ``default_output=lambda *inputs: max(inputs)`` — fallback
to a computed value rather than a fixed constant.

Default-off / non-touched-API contract: existing tests with constant
``default_output`` continue to pass byte-equivalent (see
``test_t_119_truth_table_phase1.py`` and the other followup tests).
"""

from __future__ import annotations

import numpy as np
import pytest

import jaxonomy
from jaxonomy import library
from jaxonomy.library import TruthTable
from jaxonomy.testing.markers import skip_if_not_jax

skip_if_not_jax()


def _run_truth_table(tt, input_values):
    """Wire ``len(input_values)`` Constants into ``tt`` and simulate.

    Returns the recorded output at the final timestep as a ``np.ndarray``.
    """
    sources = [library.Constant(v) for v in input_values]
    builder = jaxonomy.DiagramBuilder()
    builder.add(*sources, tt)
    for i, src in enumerate(sources):
        builder.connect(src.output_ports[0], tt.input_ports[i])
    diagram = builder.build()
    ctx = diagram.create_context()
    results = jaxonomy.simulate(
        diagram, ctx, (0.0, 0.1),
        recorded_signals={"out": tt.output_ports[0]},
    )
    return np.asarray(results.outputs["out"])[-1]


# ---------------------------------------------------------------------------
# Default-off byte-equivalence — constant default_output behaves unchanged.
# ---------------------------------------------------------------------------


def test_constant_default_output_unchanged_behaviour():
    """A pure-constant table (incl. constant default) still behaves identically.

    Mirrors the phase 1 AND-gate; no callables anywhere.
    """
    rows = [
        ((True, True), 1.0),
        ((True, False), 0.0),
        ((False, True), 0.0),
        ((False, False), 0.0),
    ]
    tt = TruthTable(rows=rows, n_inputs=2, default_output=0.0)
    for a, b, expected in [
        (True, True, 1.0),
        (True, False, 0.0),
        (False, True, 0.0),
        (False, False, 0.0),
    ]:
        out = _run_truth_table(tt, [a, b])
        np.testing.assert_allclose(float(out), expected)


def test_constant_default_with_no_matching_row():
    """Constant default returned when no row matches (phase 1 semantics)."""
    tt = TruthTable(
        rows=[((True, True), 99.0)],
        n_inputs=2,
        default_output=-7.0,
    )
    # (False, True) misses the only row.
    out = _run_truth_table(tt, [False, True])
    np.testing.assert_allclose(float(out), -7.0)


# ---------------------------------------------------------------------------
# Callable default_output — invoked when no row matches.
# ---------------------------------------------------------------------------


def test_callable_default_invoked_when_no_row_matches():
    """``default_output=lambda *inputs: max(inputs)`` returns the max input.

    Uses ``jnp.maximum`` so the callable traces under JAX (Python's
    ``max`` would short-circuit on tracers); semantics identical for
    scalar inputs.
    """
    import jax.numpy as jnp

    tt = TruthTable(
        # Only one row: requires (True, True). Anything else → default.
        rows=[((True, True), -1.0)],
        n_inputs=2,
        default_output=lambda a, b: jnp.maximum(a, b),
    )
    # (5.0, 0.0) → bool=(T, F), no row matches → default = max(5.0, 0.0) = 5.0
    out = _run_truth_table(tt, [5.0, 0.0])
    np.testing.assert_allclose(float(out), 5.0)
    # (0.0, 7.0) → bool=(F, T), no match → default = max(0.0, 7.0) = 7.0
    out = _run_truth_table(tt, [0.0, 7.0])
    np.testing.assert_allclose(float(out), 7.0)


def test_callable_default_uses_raw_input_values():
    """Two distinct non-matching inputs produce DIFFERENT defaults via callable.

    A constant default would emit the same value for both. A callable default
    can distinguish them because it sees the raw numerical inputs.
    """
    tt = TruthTable(
        # No row matches (T, F) — both default-fall-through cases.
        rows=[((True, True), 0.0)],
        n_inputs=2,
        default_output=lambda a, b: a * 10.0 + b,
    )
    # Both (T, F): 1.0,0.0 vs 4.0,0.0 — bool pattern identical, raw values differ.
    out_1 = _run_truth_table(tt, [1.0, 0.0])
    out_2 = _run_truth_table(tt, [4.0, 0.0])
    np.testing.assert_allclose(float(out_1), 1.0 * 10.0 + 0.0)
    np.testing.assert_allclose(float(out_2), 4.0 * 10.0 + 0.0)
    # Sanity: outputs differ — proves the callable saw the raw values.
    assert float(out_1) != float(out_2)


def test_callable_default_not_used_when_row_matches():
    """A matching row's output should win over the callable default.

    Same earlier-row precedence semantics as the constant-default case.
    """
    sentinel_calls = []

    def default_fn(a, b):
        sentinel_calls.append((a, b))
        return a + b + 999.0

    tt = TruthTable(
        rows=[((True, True), 42.0)],
        n_inputs=2,
        default_output=default_fn,
    )
    # (True, True) → row matches → output 42.0, NOT the default expression.
    out = _run_truth_table(tt, [True, True])
    np.testing.assert_allclose(float(out), 42.0)
    # The callable IS evaluated unconditionally (branchless ``where``)
    # but its result is masked out by the matching row. We don't assert
    # on call count here because JAX may trace the callable any number
    # of times — the contract is on the OUTPUT VALUE, not the call count.


# ---------------------------------------------------------------------------
# Callable default + callable rows — full mix.
# ---------------------------------------------------------------------------


def test_callable_default_with_callable_rows():
    """Both rows and default may be callables simultaneously."""
    tt = TruthTable(
        rows=[
            ((True, True), lambda a, b: a + b),
            ((True, False), lambda a, b: a - b),
        ],
        n_inputs=2,
        default_output=lambda a, b: a * b,
    )
    # (T, T) → a + b
    out = _run_truth_table(tt, [3.0, 4.0])
    np.testing.assert_allclose(float(out), 7.0)
    # (T, F) → a - b. Need numerically-False b (== 0).
    out = _run_truth_table(tt, [5.0, 0.0])
    np.testing.assert_allclose(float(out), 5.0)
    # (F, T) → no row matches → default a*b = 0 * 6 = 0
    out = _run_truth_table(tt, [0.0, 6.0])
    np.testing.assert_allclose(float(out), 0.0 * 6.0)


# ---------------------------------------------------------------------------
# Vector callable default — callable may return a vector.
# ---------------------------------------------------------------------------


def test_callable_default_vector_output():
    """Callable default may return a vector value matching row-output shape.

    Use a row pattern that won't match (False,) so the default fires;
    the callable then sees the raw input and returns a vector.
    """
    tt = TruthTable(
        rows=[((True,), np.array([1.0, 2.0]))],
        n_inputs=1,
        # Default returns [3 + x, 5 + x] when no row matches.
        default_output=lambda x: np.array([3.0, 5.0]) + x,
    )
    # bool(0.0) is False → row doesn't match → default fires with x=0.0.
    out = _run_truth_table(tt, [0.0])
    np.testing.assert_allclose(np.asarray(out), [3.0, 5.0])


# ---------------------------------------------------------------------------
# Serialization rejects callable default_output.
# ---------------------------------------------------------------------------


def test_to_dict_rejects_callable_default_output():
    """``to_dict`` must raise a clear error on callable default_output."""
    tt = TruthTable(
        rows=[((True,), 1.0)],
        n_inputs=1,
        default_output=lambda x: x + 1.0,
    )
    with pytest.raises(ValueError, match="default_output"):
        tt.to_dict()


def test_to_dict_still_works_for_constant_default():
    """A table with a constant default continues to serialize cleanly."""
    tt = TruthTable(
        rows=[((True,), 1.0), ((False,), 0.0)],
        n_inputs=1,
        default_output=-1.0,
    )
    d = tt.to_dict()
    assert d["n_inputs"] == 1
    assert "default_output" in d
    # Round-trip works.
    tt2 = TruthTable.from_dict(d)
    out = _run_truth_table(tt2, [True])
    np.testing.assert_allclose(float(out), 1.0)


def test_to_csv_rejects_callable_default_output(tmp_path):
    """``to_csv`` must raise a clear error on callable default_output."""
    tt = TruthTable(
        rows=[((True,), 1.0)],
        n_inputs=1,
        default_output=lambda x: x + 1.0,
    )
    with pytest.raises(ValueError, match="default_output"):
        tt.to_csv(tmp_path / "t.csv")


# ---------------------------------------------------------------------------
# T-005 default-float64 preservation — callable returning float stays float64.
# ---------------------------------------------------------------------------


def test_callable_default_preserves_float64_dtype():
    """Callable returning a float yields a float64 output (T-005 policy)."""
    tt = TruthTable(
        rows=[((True,), -1.0)],
        n_inputs=1,
        default_output=lambda x: x + 0.5,
    )
    out = _run_truth_table(tt, [0.0])  # (F,) → default
    assert np.asarray(out).dtype == np.float64
