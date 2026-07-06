# SPDX-License-Identifier: MIT
"""T-119 phase 1: TruthTable block.

A ``TruthTable`` evaluates a fixed list of ``(input_pattern, output)``
rows over boolean-castable inputs and emits the matching row's output
(or ``default_output`` if none match). Patterns may use the literal
string ``"X"`` as a wildcard for an input slot.

Earlier rows take precedence over later rows when patterns overlap.

Static-completeness/ambiguity checking is deferred —
see ``T-119-followup-completeness-checker``. JSON serialization of the
rows table is deferred — see ``T-119-followup-serialization``.
"""

from __future__ import annotations

import numpy as np
import pytest

import jaxonomy
from jaxonomy import library
from jaxonomy.framework.error import BlockParameterError
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
# 2-input AND truth table — verify all 4 combinations.
# ---------------------------------------------------------------------------


AND_ROWS = [
    ((True, True), 1.0),
    ((True, False), 0.0),
    ((False, True), 0.0),
    ((False, False), 0.0),
]


@pytest.mark.parametrize(
    "a, b, expected",
    [
        (True, True, 1.0),
        (True, False, 0.0),
        (False, True, 0.0),
        (False, False, 0.0),
    ],
)
def test_two_input_and_truth_table(a, b, expected):
    tt = TruthTable(rows=AND_ROWS, n_inputs=2, default_output=0.0)
    out = _run_truth_table(tt, [a, b])
    np.testing.assert_allclose(float(out), expected)


# ---------------------------------------------------------------------------
# Wildcard — first input is ignored via "X".
# ---------------------------------------------------------------------------


WILDCARD_ROWS = [
    (("X", True), 1.0),
    (("X", False), 0.0),
]


@pytest.mark.parametrize(
    "a, b, expected",
    [
        (True, True, 1.0),
        (False, True, 1.0),
        (True, False, 0.0),
        (False, False, 0.0),
    ],
)
def test_wildcard_pattern_ignores_first_input(a, b, expected):
    tt = TruthTable(rows=WILDCARD_ROWS, n_inputs=2, default_output=-1.0)
    out = _run_truth_table(tt, [a, b])
    np.testing.assert_allclose(float(out), expected)


# ---------------------------------------------------------------------------
# Default fallback — partial table uses default_output for unmatched rows.
# ---------------------------------------------------------------------------


def test_default_fallback_when_no_row_matches():
    """Partial table covers only (T, T); other inputs hit default_output."""
    rows = [((True, True), 99.0)]
    tt = TruthTable(rows=rows, n_inputs=2, default_output=-7.0)

    # Matching row — should produce 99.0.
    out_match = _run_truth_table(tt, [True, True])
    np.testing.assert_allclose(float(out_match), 99.0)

    # Each of the three remaining combinations falls through to the default.
    for a, b in [(True, False), (False, True), (False, False)]:
        out = _run_truth_table(tt, [a, b])
        np.testing.assert_allclose(float(out), -7.0)


# ---------------------------------------------------------------------------
# Vector outputs — outputs can be 1-D arrays, not just scalars.
# ---------------------------------------------------------------------------


def test_vector_outputs():
    """Output values may be 1-D arrays; the matching row's vector flows out."""
    rows = [
        ((True,), np.array([1.0, 2.0, 3.0])),
        ((False,), np.array([4.0, 5.0, 6.0])),
    ]
    tt_t = TruthTable(
        rows=rows, n_inputs=1, default_output=np.array([0.0, 0.0, 0.0])
    )
    out_t = _run_truth_table(tt_t, [True])
    np.testing.assert_allclose(np.asarray(out_t), [1.0, 2.0, 3.0])

    tt_f = TruthTable(
        rows=rows, n_inputs=1, default_output=np.array([0.0, 0.0, 0.0])
    )
    out_f = _run_truth_table(tt_f, [False])
    np.testing.assert_allclose(np.asarray(out_f), [4.0, 5.0, 6.0])


def test_vector_default_output_shape_used_when_no_match():
    """A partial vector-output table falls through to the vector default."""
    rows = [((True,), np.array([10.0, 20.0]))]
    tt = TruthTable(
        rows=rows, n_inputs=1, default_output=np.array([-1.0, -2.0])
    )
    out = _run_truth_table(tt, [False])
    np.testing.assert_allclose(np.asarray(out), [-1.0, -2.0])


# ---------------------------------------------------------------------------
# Earlier rows take precedence on overlapping patterns.
# ---------------------------------------------------------------------------


def test_earlier_rows_take_precedence_over_overlapping_wildcard():
    """A specific row before a wildcard row wins on the overlap."""
    rows = [
        ((True, True), 100.0),       # specific
        (("X", True), 1.0),          # wildcard, also matches (T, T)
    ]
    tt = TruthTable(rows=rows, n_inputs=2, default_output=0.0)

    out_specific = _run_truth_table(tt, [True, True])
    np.testing.assert_allclose(float(out_specific), 100.0)

    out_wildcard = _run_truth_table(tt, [False, True])
    np.testing.assert_allclose(float(out_wildcard), 1.0)


# ---------------------------------------------------------------------------
# __init__ validation — misconfigurations fail fast (not inside JAX trace).
# ---------------------------------------------------------------------------


def test_n_inputs_must_be_at_least_one():
    with pytest.raises(BlockParameterError):
        TruthTable(rows=[], n_inputs=0, default_output=0.0)


def test_pattern_length_must_match_n_inputs():
    with pytest.raises(BlockParameterError):
        TruthTable(
            rows=[((True,), 1.0)],  # length-1 pattern
            n_inputs=2,             # but 2 input ports
            default_output=0.0,
        )


def test_pattern_entries_must_be_bool_or_wildcard():
    with pytest.raises(BlockParameterError):
        TruthTable(
            rows=[((1, True), 1.0)],  # 1 (int) is not bool / "X"
            n_inputs=2,
            default_output=0.0,
        )
