# SPDX-License-Identifier: MIT
"""T-119-followup-export-to-csv: write a TruthTable to a CSV file.

Adds ``TruthTable.to_csv(path, **csv_kwargs)`` — a pure-stdlib writer
that is the inverse of :meth:`TruthTable.from_csv`. Emits the same
header layout (``in1,in2,...,output`` or ``output_i`` columns for
vector outputs), with input cells written as ``T`` / ``F`` / ``X``
and output cells written as ``float(...)``.

The default-off / non-touched-API path is byte-equivalent: the existing
constructor and other API are unchanged. T-005 default-float64 is
preserved (``float(...)`` write + ``np.asarray`` on the row outputs).
"""

from __future__ import annotations

import csv
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
    """Wire ``len(input_values)`` Constants into ``tt`` and simulate."""
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


def _read_csv_lines(path):
    with open(path, "r", newline="") as fh:
        return [row for row in csv.reader(fh)]


def _patterns_equal(a, b):
    """Compare two TruthTable patterns (mix of bool / 'X')."""
    if len(a) != len(b):
        return False
    for pa, pb in zip(a, b):
        if pa == "X" or pb == "X":
            if pa != pb:
                return False
        else:
            if bool(pa) != bool(pb):
                return False
    return True


def _rows_equal(rows_a, rows_b):
    """Compare two ``_rows`` lists: patterns + output arrays + callability."""
    if len(rows_a) != len(rows_b):
        return False
    for (pa, oa, ca), (pb, ob, cb) in zip(rows_a, rows_b):
        if ca != cb:
            return False
        if not _patterns_equal(pa, pb):
            return False
        if not np.allclose(np.asarray(oa), np.asarray(ob)):
            return False
    return True


def _tables_equivalent(t1, t2):
    """Round-trip equivalence: rows + n_inputs + default_output match."""
    if t1._n_inputs != t2._n_inputs:
        return False
    if not np.allclose(
        np.asarray(t1._default_output), np.asarray(t2._default_output)
    ):
        return False
    if np.asarray(t1._default_output).shape != np.asarray(t2._default_output).shape:
        return False
    return _rows_equal(t1._rows, t2._rows)


# ---------------------------------------------------------------------------
# Layout — to_csv emits exactly the expected header + cell strings.
# ---------------------------------------------------------------------------


def test_to_csv_layout_matches_task_spec_example():
    """The task spec example must serialise byte-for-byte as documented.

    Expected:
        in1,in2,in3,output
        T,T,T,1.0
        T,T,F,0.5
        T,F,X,0.25
        F,X,X,0.0
    """
    rows = [
        ((True, True, True), 1.0),
        ((True, True, False), 0.5),
        ((True, False, "X"), 0.25),
        ((False, "X", "X"), 0.0),
    ]
    tt = TruthTable(rows=rows, n_inputs=3, default_output=0.0)

    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "out.csv")
        ret = tt.to_csv(path)
        assert ret == path  # to_csv returns the path it wrote.
        lines = _read_csv_lines(path)

    assert lines[0] == ["in1", "in2", "in3", "output"]
    assert lines[1] == ["T", "T", "T", "1.0"]
    assert lines[2] == ["T", "T", "F", "0.5"]
    assert lines[3] == ["T", "F", "X", "0.25"]
    assert lines[4] == ["F", "X", "X", "0.0"]
    assert len(lines) == 5


# ---------------------------------------------------------------------------
# Round-trip: from_csv(to_csv(t)) == t (rows + n_inputs + default_output).
# ---------------------------------------------------------------------------


def test_to_csv_round_trip_preserves_rows_and_defaults():
    rows = [
        ((True, True), 1.0),
        ((True, False), 0.5),
        ((False, "X"), 0.25),
    ]
    tt = TruthTable(rows=rows, n_inputs=2, default_output=0.0)

    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "rt.csv")
        tt.to_csv(path)
        tt2 = TruthTable.from_csv(path)

    assert _tables_equivalent(tt, tt2)


def test_to_csv_round_trip_matches_runtime_behaviour():
    """Simulating the round-tripped table reproduces every row output."""
    rows = [
        ((True, True, True), 1.0),
        ((True, True, False), 0.5),
        ((True, False, "X"), 0.25),
        ((False, "X", "X"), 0.0),
    ]
    tt = TruthTable(rows=rows, n_inputs=3, default_output=-1.0)

    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "rt.csv")
        tt.to_csv(path)
        tt2 = TruthTable.from_csv(path, default_output=-1.0)

    for combo in [
        (True, True, True),
        (True, True, False),
        (True, False, True),
        (True, False, False),
        (False, True, True),
        (False, False, False),
    ]:
        a = _run_truth_table(tt, list(combo))
        b = _run_truth_table(tt2, list(combo))
        np.testing.assert_allclose(np.asarray(a), np.asarray(b))


# ---------------------------------------------------------------------------
# Callable outputs (T-119-followup-numeric-output) — to_csv refuses.
# ---------------------------------------------------------------------------


def test_to_csv_rejects_callable_row_outputs():
    """Callable row outputs have no portable CSV representation."""
    rows = [
        ((True, True), lambda a, b: a + b),
        ((True, False), 0.0),
        ((False, "X"), 0.0),
    ]
    tt = TruthTable(rows=rows, n_inputs=2, default_output=0.0)

    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "callable.csv")
        with pytest.raises(ValueError, match="callable"):
            tt.to_csv(path)
        # File must not exist on disk after the failed write.
        assert not os.path.exists(path)


# ---------------------------------------------------------------------------
# Vector outputs — produce output_i columns and round-trip correctly.
# ---------------------------------------------------------------------------


def test_to_csv_vector_outputs_emit_output_i_columns():
    rows = [
        ((True,), np.asarray([1.0, 2.0])),
        ((False,), np.asarray([3.0, 4.0])),
    ]
    tt = TruthTable(
        rows=rows, n_inputs=1, default_output=np.asarray([0.0, 0.0])
    )

    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "vec.csv")
        tt.to_csv(path)
        lines = _read_csv_lines(path)

    assert lines[0] == ["in1", "output_0", "output_1"]
    assert lines[1] == ["T", "1.0", "2.0"]
    assert lines[2] == ["F", "3.0", "4.0"]


def test_to_csv_vector_outputs_round_trip():
    rows = [
        ((True, True), np.asarray([1.0, 2.0, 3.0])),
        ((True, False), np.asarray([4.0, 5.0, 6.0])),
        ((False, "X"), np.asarray([7.0, 8.0, 9.0])),
    ]
    tt = TruthTable(
        rows=rows, n_inputs=2, default_output=np.asarray([0.0, 0.0, 0.0])
    )

    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "vec_rt.csv")
        tt.to_csv(path)
        tt2 = TruthTable.from_csv(path)

    assert _tables_equivalent(tt, tt2)
    # And simulated behaviour matches.
    out_a = _run_truth_table(tt, [True, True])
    out_b = _run_truth_table(tt2, [True, True])
    np.testing.assert_allclose(np.asarray(out_a), np.asarray(out_b))


# ---------------------------------------------------------------------------
# T-005 default-float64 preservation through the write path.
# ---------------------------------------------------------------------------


def test_to_csv_preserves_float64_on_round_trip():
    rows = [
        ((True,), 1.0),
        ((False,), 0.0),
    ]
    tt = TruthTable(rows=rows, n_inputs=1, default_output=0.0)

    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "f64.csv")
        tt.to_csv(path)
        tt2 = TruthTable.from_csv(path)

    out = _run_truth_table(tt2, [True])
    assert np.asarray(out).dtype == np.float64


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
