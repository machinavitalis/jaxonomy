# SPDX-License-Identifier: MIT
"""T-119-followup-builder-api: fluent ``TruthTable.builder()`` API.

Adds a ``TruthTableBuilder`` helper that lets callers assemble rows by
named-input keyword (``in1``, ``in2`` by default, or any custom names)
instead of positional pattern tuples. Omitted inputs default to the
wildcard ``"X"``. ``.build()`` calls back into the existing
``TruthTable(rows=...)`` constructor with no behavioural change, so the
non-builder path remains byte-equivalent.
"""

from __future__ import annotations

import numpy as np
import pytest

import jaxonomy
from jaxonomy import library
from jaxonomy.library import TruthTable, TruthTableBuilder
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
# Equivalence — builder produces same outputs as positional constructor.
# ---------------------------------------------------------------------------


# 3-input AND, exhaustively enumerated.
AND3_ROWS_POSITIONAL = [
    ((True,  True,  True),  1.0),
    ((True,  True,  False), 0.0),
    ((True,  False, True),  0.0),
    ((True,  False, False), 0.0),
    ((False, True,  True),  0.0),
    ((False, True,  False), 0.0),
    ((False, False, True),  0.0),
    ((False, False, False), 0.0),
]


def _build_and3_via_builder():
    return (
        TruthTable.builder(n_inputs=3, default_output=0.0)
        .row(in1=True,  in2=True,  in3=True,  output=1.0)
        .row(in1=True,  in2=True,  in3=False, output=0.0)
        .row(in1=True,  in2=False, in3=True,  output=0.0)
        .row(in1=True,  in2=False, in3=False, output=0.0)
        .row(in1=False, in2=True,  in3=True,  output=0.0)
        .row(in1=False, in2=True,  in3=False, output=0.0)
        .row(in1=False, in2=False, in3=True,  output=0.0)
        .row(in1=False, in2=False, in3=False, output=0.0)
        .build()
    )


@pytest.mark.parametrize(
    "a, b, c, expected",
    [
        (True,  True,  True,  1.0),
        (True,  True,  False, 0.0),
        (True,  False, True,  0.0),
        (True,  False, False, 0.0),
        (False, True,  True,  0.0),
        (False, True,  False, 0.0),
        (False, False, True,  0.0),
        (False, False, False, 0.0),
    ],
)
def test_builder_3input_and_matches_positional_constructor(a, b, c, expected):
    """Each of 8 input combinations agrees between builder and constructor."""
    tt_builder = _build_and3_via_builder()
    tt_positional = TruthTable(
        rows=AND3_ROWS_POSITIONAL, n_inputs=3, default_output=0.0
    )

    out_builder = _run_truth_table(tt_builder, [a, b, c])
    out_positional = _run_truth_table(tt_positional, [a, b, c])

    np.testing.assert_allclose(float(out_builder), expected)
    np.testing.assert_allclose(float(out_builder), float(out_positional))


# ---------------------------------------------------------------------------
# Wildcard via omitted keyword — missing inputs default to "X".
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "a, b, expected",
    [
        (True,  True,  1.0),
        (False, True,  1.0),  # in1 was omitted in the row → wildcard match
        (True,  False, 0.0),
        (False, False, 0.0),
    ],
)
def test_omitted_keyword_acts_as_wildcard(a, b, expected):
    """A missing input keyword in ``.row(...)`` defaults to the ``"X"``
    wildcard, matching either True or False at that input slot — same as
    the positional ``("X", True)`` form.
    """
    tt = (
        TruthTable.builder(n_inputs=2, default_output=-1.0)
        .row(in2=True,  output=1.0)   # in1 omitted → wildcard
        .row(in2=False, output=0.0)   # in1 omitted → wildcard
        .build()
    )
    out = _run_truth_table(tt, [a, b])
    np.testing.assert_allclose(float(out), expected)


# ---------------------------------------------------------------------------
# Custom input names via ``input_names=("a", "b", "c")``.
# ---------------------------------------------------------------------------


def test_custom_input_names_used_for_row_kwargs():
    """When ``input_names=("a","b","c")`` is given, ``.row(...)`` keys must
    use those names (not ``in1``/``in2``/``in3``).
    """
    tt = (
        TruthTable.builder(
            n_inputs=3,
            default_output=0.0,
            input_names=("a", "b", "c"),
        )
        .row(a=True, b=True, c=True, output=7.0)
        .row(a=True, c=False, output=3.0)   # b omitted → wildcard
        .build()
    )

    # First row matches → 7.0
    np.testing.assert_allclose(
        float(_run_truth_table(tt, [True, True, True])), 7.0
    )
    # Second row matches (b is wildcard, a=T, c=F) for both b values.
    np.testing.assert_allclose(
        float(_run_truth_table(tt, [True, True, False])), 3.0
    )
    np.testing.assert_allclose(
        float(_run_truth_table(tt, [True, False, False])), 3.0
    )
    # No row matches → default 0.0
    np.testing.assert_allclose(
        float(_run_truth_table(tt, [False, False, True])), 0.0
    )


# ---------------------------------------------------------------------------
# Validation — passing an unknown input name raises a clear error.
# ---------------------------------------------------------------------------


def test_unknown_input_name_in_row_raises():
    """``.row(typo=...)`` for a name that wasn't declared raises ValueError
    naming both the bad key and the expected names.
    """
    tt_builder = TruthTable.builder(n_inputs=2, default_output=0.0)
    with pytest.raises(ValueError) as exc:
        tt_builder.row(in1=True, in_two=True, output=1.0)
    msg = str(exc.value)
    assert "in_two" in msg
    assert "in1" in msg and "in2" in msg


def test_unknown_input_name_with_custom_names_raises():
    """The error message reports the configured custom names."""
    tt_builder = TruthTable.builder(
        n_inputs=2, default_output=0.0, input_names=("alpha", "beta")
    )
    with pytest.raises(ValueError) as exc:
        tt_builder.row(alpha=True, gamma=True, output=1.0)
    msg = str(exc.value)
    assert "gamma" in msg
    assert "alpha" in msg and "beta" in msg


def test_input_names_length_must_match_n_inputs():
    """Mismatched ``input_names`` length is rejected at builder construction."""
    with pytest.raises(ValueError):
        TruthTable.builder(
            n_inputs=3, default_output=0.0, input_names=("a", "b")
        )


def test_input_names_must_be_unique():
    """Duplicate names would silently merge rows — reject up front."""
    with pytest.raises(ValueError):
        TruthTable.builder(
            n_inputs=3, default_output=0.0, input_names=("a", "b", "a")
        )


def test_n_inputs_must_be_at_least_one():
    """Builder rejects ``n_inputs < 1`` before ever calling TruthTable."""
    with pytest.raises(ValueError):
        TruthTable.builder(n_inputs=0, default_output=0.0)


# ---------------------------------------------------------------------------
# ``.row(...)`` returns the builder for fluent chaining.
# ---------------------------------------------------------------------------


def test_row_returns_self_for_chaining():
    """``.row(...)`` returns the builder itself so calls can be chained."""
    tt_builder = TruthTable.builder(n_inputs=1, default_output=0.0)
    returned = tt_builder.row(in1=True, output=1.0)
    assert returned is tt_builder


# ---------------------------------------------------------------------------
# ``TruthTableBuilder`` is also importable directly (not only via classmethod).
# ---------------------------------------------------------------------------


def test_truth_table_builder_class_is_directly_constructible():
    """Direct ``TruthTableBuilder(...)`` works identically to the classmethod."""
    tt = (
        TruthTableBuilder(n_inputs=1, default_output=0.0)
        .row(in1=True, output=42.0)
        .build()
    )
    np.testing.assert_allclose(float(_run_truth_table(tt, [True])), 42.0)
    np.testing.assert_allclose(float(_run_truth_table(tt, [False])), 0.0)
