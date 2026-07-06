# SPDX-License-Identifier: MIT
"""T-119-followup-import-from-csv: load a TruthTable from a CSV file.

Adds ``TruthTable.from_csv(path, **block_kwargs)`` — a pure-stdlib loader
that reads a CSV with a header row of input column names followed by one
or more ``output*`` columns. Input cells accept ``T``/``True``/``1`` for
True, ``F``/``False``/``0`` for False, and ``X``/``-``/``*`` (or empty)
for the wildcard. The loader produces a working :class:`TruthTable`
whose simulated behaviour matches the equivalent in-memory construction.

The default-off / non-touched-API path is byte-equivalent: ``TruthTable``'s
existing constructor is unchanged. T-005 default-float64 is preserved
(``float(...)`` parse + the constructor's ``npa.asarray``).
"""

from __future__ import annotations

import os
import tempfile

import numpy as np
import pytest

import jaxonomy
from jaxonomy import library
from jaxonomy.library import TruthTable
from jaxonomy.testing.markers import skip_if_not_jax

skip_if_not_jax()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


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


def _write_csv(tmpdir, lines, name="table.csv"):
    """Write a CSV file under ``tmpdir`` and return the path."""
    path = os.path.join(tmpdir, name)
    with open(path, "w", newline="") as fh:
        fh.write("\n".join(lines) + "\n")
    return path


# ---------------------------------------------------------------------------
# Basic loader — example from the task spec.
# ---------------------------------------------------------------------------


def test_from_csv_loads_3input_table_per_task_spec():
    """The exact CSV layout shown in the T-119 followup spec must load.

    in1,in2,in3,output
    T,T,T,1.0
    T,T,F,0.5
    T,F,X,0.25
    F,X,X,0.0
    """
    with tempfile.TemporaryDirectory() as td:
        path = _write_csv(td, [
            "in1,in2,in3,output",
            "T,T,T,1.0",
            "T,T,F,0.5",
            "T,F,X,0.25",
            "F,X,X,0.0",
        ])
        tt = TruthTable.from_csv(path)

    # Row 1: (T,T,T) -> 1.0
    np.testing.assert_allclose(float(_run_truth_table(tt, [True, True, True])), 1.0)
    # Row 2: (T,T,F) -> 0.5
    np.testing.assert_allclose(float(_run_truth_table(tt, [True, True, False])), 0.5)
    # Row 3 wildcard: (T,F,*) -> 0.25 (both T,F,T and T,F,F hit the X)
    np.testing.assert_allclose(float(_run_truth_table(tt, [True, False, True])), 0.25)
    np.testing.assert_allclose(float(_run_truth_table(tt, [True, False, False])), 0.25)
    # Row 4 wildcard: (F,*,*) -> 0.0
    np.testing.assert_allclose(float(_run_truth_table(tt, [False, True, True])), 0.0)
    np.testing.assert_allclose(float(_run_truth_table(tt, [False, False, False])), 0.0)


# ---------------------------------------------------------------------------
# All accepted token forms parse correctly (case-insensitive).
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "true_token,false_token",
    [
        ("T", "F"),
        ("t", "f"),
        ("True", "False"),
        ("TRUE", "FALSE"),
        ("true", "false"),
        ("1", "0"),
    ],
)
def test_from_csv_accepts_all_true_false_token_forms(true_token, false_token):
    with tempfile.TemporaryDirectory() as td:
        path = _write_csv(td, [
            "in1,in2,output",
            f"{true_token},{true_token},1.0",
            f"{true_token},{false_token},0.5",
            f"{false_token},{true_token},0.25",
            f"{false_token},{false_token},0.0",
        ])
        tt = TruthTable.from_csv(path)

    np.testing.assert_allclose(float(_run_truth_table(tt, [True, True])), 1.0)
    np.testing.assert_allclose(float(_run_truth_table(tt, [True, False])), 0.5)
    np.testing.assert_allclose(float(_run_truth_table(tt, [False, True])), 0.25)
    np.testing.assert_allclose(float(_run_truth_table(tt, [False, False])), 0.0)


@pytest.mark.parametrize("wildcard", ["X", "x", "-", "*", ""])
def test_from_csv_accepts_all_wildcard_token_forms(wildcard):
    """All accepted wildcard tokens select the wildcard row identically."""
    with tempfile.TemporaryDirectory() as td:
        path = _write_csv(td, [
            "in1,in2,output",
            f"T,{wildcard},1.0",
            f"F,{wildcard},0.0",
        ])
        tt = TruthTable.from_csv(path)

    # Both T,T and T,F should hit the wildcard row -> 1.0
    np.testing.assert_allclose(float(_run_truth_table(tt, [True, True])), 1.0)
    np.testing.assert_allclose(float(_run_truth_table(tt, [True, False])), 1.0)
    # Both F,* should hit row 2 -> 0.0
    np.testing.assert_allclose(float(_run_truth_table(tt, [False, True])), 0.0)
    np.testing.assert_allclose(float(_run_truth_table(tt, [False, False])), 0.0)


# ---------------------------------------------------------------------------
# Whitespace in cells is tolerated.
# ---------------------------------------------------------------------------


def test_from_csv_strips_whitespace_around_cells():
    with tempfile.TemporaryDirectory() as td:
        path = _write_csv(td, [
            " in1 , in2 , output ",
            "  T , F , 1.5 ",
            " F ,  X ,  2.5 ",
        ])
        tt = TruthTable.from_csv(path)

    np.testing.assert_allclose(float(_run_truth_table(tt, [True, False])), 1.5)
    np.testing.assert_allclose(float(_run_truth_table(tt, [False, True])), 2.5)
    np.testing.assert_allclose(float(_run_truth_table(tt, [False, False])), 2.5)


# ---------------------------------------------------------------------------
# Multi-output (vector) CSV support.
# ---------------------------------------------------------------------------


def test_from_csv_supports_vector_outputs_via_output_prefix():
    """Multiple ``output*`` columns are stacked into a 1-D vector output."""
    with tempfile.TemporaryDirectory() as td:
        path = _write_csv(td, [
            "in1,output_x,output_y",
            "T,1.0,2.0",
            "F,3.0,4.0",
        ])
        tt = TruthTable.from_csv(path)

    out_t = _run_truth_table(tt, [True])
    np.testing.assert_allclose(np.asarray(out_t), [1.0, 2.0])
    out_f = _run_truth_table(tt, [False])
    np.testing.assert_allclose(np.asarray(out_f), [3.0, 4.0])


# ---------------------------------------------------------------------------
# default_output override via block_kwargs.
# ---------------------------------------------------------------------------


def test_from_csv_forwards_default_output_block_kwarg():
    """Caller-supplied ``default_output`` overrides the zero-default."""
    with tempfile.TemporaryDirectory() as td:
        path = _write_csv(td, [
            "in1,output",
            "T,1.0",
            # No row for False -> hits default_output.
        ])
        tt = TruthTable.from_csv(path, default_output=-99.0)

    np.testing.assert_allclose(float(_run_truth_table(tt, [True])), 1.0)
    np.testing.assert_allclose(float(_run_truth_table(tt, [False])), -99.0)


def test_from_csv_forwards_name_block_kwarg():
    with tempfile.TemporaryDirectory() as td:
        path = _write_csv(td, [
            "in1,output",
            "T,1.0",
            "F,0.0",
        ])
        tt = TruthTable.from_csv(path, name="from_csv_block")
    assert tt.name == "from_csv_block"


# ---------------------------------------------------------------------------
# T-005 default-float64 preservation.
# ---------------------------------------------------------------------------


def test_from_csv_preserves_float64_default():
    """Scalar outputs parsed from CSV trace through as float64 by default."""
    with tempfile.TemporaryDirectory() as td:
        path = _write_csv(td, [
            "in1,output",
            "T,1.0",
            "F,0.0",
        ])
        tt = TruthTable.from_csv(path)

    out = _run_truth_table(tt, [True])
    assert np.asarray(out).dtype == np.float64


# ---------------------------------------------------------------------------
# Validation — malformed CSV raises clear errors.
# ---------------------------------------------------------------------------


def test_from_csv_empty_file_raises():
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "empty.csv")
        open(path, "w").close()
        with pytest.raises(ValueError, match="empty"):
            TruthTable.from_csv(path)


def test_from_csv_missing_output_column_raises():
    with tempfile.TemporaryDirectory() as td:
        path = _write_csv(td, [
            "in1,in2,value",
            "T,T,1.0",
        ])
        with pytest.raises(ValueError, match="no 'output' column"):
            TruthTable.from_csv(path)


def test_from_csv_no_input_columns_raises():
    """An ``output``-only header has no inputs to match against."""
    with tempfile.TemporaryDirectory() as td:
        path = _write_csv(td, [
            "output",
            "1.0",
        ])
        with pytest.raises(ValueError, match="no input columns"):
            TruthTable.from_csv(path)


def test_from_csv_unrecognised_input_token_raises():
    with tempfile.TemporaryDirectory() as td:
        path = _write_csv(td, [
            "in1,in2,output",
            "T,MAYBE,1.0",
        ])
        with pytest.raises(ValueError, match="unrecognised input token"):
            TruthTable.from_csv(path)


def test_from_csv_unparseable_output_raises():
    with tempfile.TemporaryDirectory() as td:
        path = _write_csv(td, [
            "in1,output",
            "T,not_a_float",
        ])
        with pytest.raises(ValueError, match="did not parse as float"):
            TruthTable.from_csv(path)


def test_from_csv_wrong_cell_count_raises():
    with tempfile.TemporaryDirectory() as td:
        path = _write_csv(td, [
            "in1,in2,output",
            "T,F",  # missing output cell
        ])
        with pytest.raises(ValueError, match="expected 3"):
            TruthTable.from_csv(path)


def test_from_csv_header_only_no_data_raises():
    with tempfile.TemporaryDirectory() as td:
        path = _write_csv(td, [
            "in1,output",
        ])
        with pytest.raises(ValueError, match="no data rows"):
            TruthTable.from_csv(path)


def test_from_csv_non_contiguous_output_columns_raises():
    """Output columns must trail the inputs (no interleaving)."""
    with tempfile.TemporaryDirectory() as td:
        path = _write_csv(td, [
            "in1,output_x,in2,output_y",
            "T,1.0,F,2.0",
        ])
        with pytest.raises(ValueError, match="non-contiguous"):
            TruthTable.from_csv(path)


# ---------------------------------------------------------------------------
# Blank lines in the body are skipped.
# ---------------------------------------------------------------------------


def test_from_csv_skips_blank_lines_in_body():
    with tempfile.TemporaryDirectory() as td:
        path = _write_csv(td, [
            "in1,output",
            "T,1.0",
            "",
            "F,0.0",
            "",
        ])
        tt = TruthTable.from_csv(path)

    np.testing.assert_allclose(float(_run_truth_table(tt, [True])), 1.0)
    np.testing.assert_allclose(float(_run_truth_table(tt, [False])), 0.0)


# ---------------------------------------------------------------------------
# Default-off / non-touched-API: existing TruthTable constructor unchanged.
# ---------------------------------------------------------------------------


def test_existing_constructor_unchanged_byte_equivalent():
    """Phase-1 constructor still produces a working block with no change."""
    AND_ROWS = [
        ((True, True), 1.0),
        ((True, False), 0.0),
        ((False, True), 0.0),
        ((False, False), 0.0),
    ]
    tt = TruthTable(rows=AND_ROWS, n_inputs=2, default_output=0.0)
    np.testing.assert_allclose(float(_run_truth_table(tt, [True, True])), 1.0)
    np.testing.assert_allclose(float(_run_truth_table(tt, [True, False])), 0.0)
    np.testing.assert_allclose(float(_run_truth_table(tt, [False, True])), 0.0)
    np.testing.assert_allclose(float(_run_truth_table(tt, [False, False])), 0.0)


def test_from_csv_matches_equivalent_in_memory_construction():
    """End-to-end: CSV-loaded block matches the hand-built equivalent."""
    AND_ROWS = [
        ((True, True), 1.0),
        ((True, False), 0.0),
        ((False, True), 0.0),
        ((False, False), 0.0),
    ]
    in_memory = TruthTable(rows=AND_ROWS, n_inputs=2, default_output=0.0)

    with tempfile.TemporaryDirectory() as td:
        path = _write_csv(td, [
            "in1,in2,output",
            "T,T,1.0",
            "T,F,0.0",
            "F,T,0.0",
            "F,F,0.0",
        ])
        from_csv = TruthTable.from_csv(path)

    for combo in [(True, True), (True, False), (False, True), (False, False)]:
        a = _run_truth_table(in_memory, list(combo))
        b = _run_truth_table(from_csv, list(combo))
        np.testing.assert_allclose(np.asarray(a), np.asarray(b))
