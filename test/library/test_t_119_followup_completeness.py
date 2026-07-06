# SPDX-License-Identifier: MIT
"""T-119-followup-completeness-checker: static analysis on the truth table.

The conventional TruthTable static analysis performs two construction-time checks:

1. **Completeness** — every one of the 2^N input combinations is matched
   by at least one row's pattern (treating ``"X"`` as a wildcard). If
   not, those combinations would silently fall through to
   ``default_output`` at runtime.
2. **Disjointness** — no two rows match the same input combination.
   Jaxonomy resolves overlap by earlier-row-wins, but flagging it
   surfaces unintended row shadowing.

The new ``TruthTable.validate(strict_completeness=False,
strict_disjointness=False)`` method returns a report dict and (with
either flag set) raises ``BlockParameterError`` on findings. Default-off
so existing call sites remain unaffected.
"""

from __future__ import annotations

import warnings

import pytest

from jaxonomy.framework.error import BlockParameterError
from jaxonomy.library import TruthTable
from jaxonomy.testing.markers import skip_if_not_jax

skip_if_not_jax()


# ---------------------------------------------------------------------------
# Complete tables — covered == total, no missing, no overlap.
# ---------------------------------------------------------------------------


def test_complete_2_input_table_reports_full_coverage():
    """A full 2-input AND covers all 4 combinations with no overlap."""
    tt = TruthTable(
        rows=[
            ((True, True), 1.0),
            ((True, False), 0.0),
            ((False, True), 0.0),
            ((False, False), 0.0),
        ],
        n_inputs=2,
        default_output=0.0,
    )
    report = tt.validate()
    assert report["covered_combinations"] == 4
    assert report["total_combinations"] == 4
    assert report["missing_patterns"] == []
    assert report["overlapping_pairs"] == []


def test_complete_3_input_table_reports_full_coverage():
    """8 rows enumerating every 3-bool combination → fully covered."""
    rows = []
    for a in (False, True):
        for b in (False, True):
            for c in (False, True):
                rows.append(((a, b, c), float(a and b and c)))
    tt = TruthTable(rows=rows, n_inputs=3, default_output=0.0)
    report = tt.validate()
    assert report["covered_combinations"] == 8
    assert report["total_combinations"] == 8
    assert report["missing_patterns"] == []
    assert report["overlapping_pairs"] == []


# ---------------------------------------------------------------------------
# Incomplete tables — missing combinations reported.
# ---------------------------------------------------------------------------


def test_incomplete_table_reports_missing_patterns():
    """Drop the (False, False) row — it should appear in missing_patterns."""
    tt = TruthTable(
        rows=[
            ((True, True), 1.0),
            ((True, False), 0.0),
            ((False, True), 0.0),
        ],
        n_inputs=2,
        default_output=0.0,
    )
    report = tt.validate()
    assert report["covered_combinations"] == 3
    assert report["total_combinations"] == 4
    assert report["missing_patterns"] == [(False, False)]
    assert report["overlapping_pairs"] == []


def test_incomplete_table_strict_completeness_raises():
    """``strict_completeness=True`` escalates missing patterns to an error."""
    tt = TruthTable(
        rows=[((True, True), 1.0)],  # only 1 of 4 covered
        n_inputs=2,
        default_output=0.0,
    )
    with pytest.raises(BlockParameterError, match="incomplete"):
        tt.validate(strict_completeness=True)


def test_incomplete_table_strict_completeness_default_off_does_not_raise():
    """Default ``validate()`` never raises, even with missing patterns."""
    tt = TruthTable(
        rows=[((True, True), 1.0)],
        n_inputs=2,
        default_output=0.0,
    )
    # Must not raise.
    report = tt.validate()
    assert len(report["missing_patterns"]) == 3


# ---------------------------------------------------------------------------
# Overlap detection.
# ---------------------------------------------------------------------------


def test_overlapping_rows_reported_as_pair_indices():
    """Two rows both matching (True, True) → reported as pair (0, 1)."""
    tt = TruthTable(
        rows=[
            ((True, True), 1.0),     # row 0
            (("X", True), 0.5),      # row 1 — overlaps row 0 at (True, True)
            ((True, False), 0.0),
            ((False, False), 0.0),
        ],
        n_inputs=2,
        default_output=0.0,
    )
    report = tt.validate()
    # Rows 0 and 1 both match (True, True). Row 1 also matches
    # (False, True) which row 0 does not — so the only pair is (0, 1).
    assert (0, 1) in report["overlapping_pairs"]
    # Coverage: rows together cover {(T,T), (F,T), (T,F), (F,F)} → 4.
    assert report["covered_combinations"] == 4


def test_overlapping_rows_strict_disjointness_raises():
    """``strict_disjointness=True`` escalates overlaps to an error."""
    tt = TruthTable(
        rows=[
            (("X", "X"), 1.0),       # matches everything
            ((True, True), 2.0),     # shadowed by row 0
            ((True, False), 0.0),
            ((False, True), 0.0),
            ((False, False), 0.0),
        ],
        n_inputs=2,
        default_output=0.0,
    )
    with pytest.raises(BlockParameterError, match="overlapping"):
        tt.validate(strict_disjointness=True)


def test_overlapping_rows_default_off_does_not_raise():
    """Default ``validate()`` reports overlap without raising."""
    tt = TruthTable(
        rows=[
            (("X", "X"), 1.0),
            ((True, True), 2.0),
        ],
        n_inputs=2,
        default_output=0.0,
    )
    report = tt.validate()
    assert report["overlapping_pairs"]  # non-empty


# ---------------------------------------------------------------------------
# Wildcard handling — "X" pattern entry covers both bool values.
# ---------------------------------------------------------------------------


def test_wildcard_covers_both_truth_values_for_that_input():
    """``("X", True)`` covers both ``(False, True)`` AND ``(True, True)``."""
    tt = TruthTable(
        rows=[
            (("X", True), 1.0),       # covers (F,T) and (T,T)
            (("X", False), 0.0),      # covers (F,F) and (T,F)
        ],
        n_inputs=2,
        default_output=0.0,
    )
    report = tt.validate()
    # Both rows together cover all 4 combinations.
    assert report["covered_combinations"] == 4
    assert report["total_combinations"] == 4
    assert report["missing_patterns"] == []
    # No two rows share an input vector → no overlap.
    assert report["overlapping_pairs"] == []


def test_full_wildcard_row_alone_covers_everything():
    """A single ``("X", "X")`` row alone covers all 2**2 combinations."""
    tt = TruthTable(
        rows=[(("X", "X"), 7.0)],
        n_inputs=2,
        default_output=0.0,
    )
    report = tt.validate()
    assert report["covered_combinations"] == 4
    assert report["missing_patterns"] == []


# ---------------------------------------------------------------------------
# Combined strict mode — both flags raise on their respective conditions.
# ---------------------------------------------------------------------------


def test_strict_completeness_does_not_raise_on_overlap_only():
    """Overlap without missing should not trigger ``strict_completeness``."""
    tt = TruthTable(
        rows=[
            (("X", "X"), 1.0),
            ((True, True), 2.0),
        ],
        n_inputs=2,
        default_output=0.0,
    )
    # Fully covered (the wildcard row covers everything), so this must
    # NOT raise on completeness even though overlaps exist.
    report = tt.validate(strict_completeness=True)
    assert report["missing_patterns"] == []
    assert report["overlapping_pairs"]


def test_strict_disjointness_does_not_raise_on_missing_only():
    """Missing without overlap should not trigger ``strict_disjointness``."""
    tt = TruthTable(
        rows=[((True, True), 1.0)],
        n_inputs=2,
        default_output=0.0,
    )
    # Incomplete but no overlap → strict_disjointness alone does not raise.
    report = tt.validate(strict_disjointness=True)
    assert report["overlapping_pairs"] == []
    assert report["missing_patterns"]


# ---------------------------------------------------------------------------
# Large-n warning.
# ---------------------------------------------------------------------------


def test_large_n_inputs_emits_warning():
    """For n_inputs > 10, ``validate()`` should warn about cost."""
    n = 11
    # Single wildcard row covers everything — fast even at n=11.
    rows = [(tuple("X" for _ in range(n)), 0.0)]
    tt = TruthTable(rows=rows, n_inputs=n, default_output=0.0)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        report = tt.validate()
        warning_messages = [str(w.message) for w in caught]
    assert any("2**11" in msg or "2 ** 11" in msg or "may be slow" in msg
               for msg in warning_messages), warning_messages
    assert report["total_combinations"] == 2 ** n
    assert report["covered_combinations"] == 2 ** n


def test_n_inputs_at_threshold_does_not_warn():
    """At n_inputs == 10 (1024 combos), no warning yet."""
    n = 10
    rows = [(tuple("X" for _ in range(n)), 0.0)]
    tt = TruthTable(rows=rows, n_inputs=n, default_output=0.0)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        tt.validate()
    # Filter to UserWarning only — other unrelated warnings are fine.
    cost_warnings = [
        w for w in caught
        if issubclass(w.category, UserWarning) and "may be slow" in str(w.message)
    ]
    assert cost_warnings == []


# ---------------------------------------------------------------------------
# Report shape — keys and types contract.
# ---------------------------------------------------------------------------


def test_report_dict_has_expected_keys_and_types():
    """The report dict's contract: keys present, ints + lists."""
    tt = TruthTable(
        rows=[((True, True), 1.0)],
        n_inputs=2,
        default_output=0.0,
    )
    report = tt.validate()
    assert set(report.keys()) == {
        "covered_combinations",
        "total_combinations",
        "missing_patterns",
        "overlapping_pairs",
    }
    assert isinstance(report["covered_combinations"], int)
    assert isinstance(report["total_combinations"], int)
    assert isinstance(report["missing_patterns"], list)
    assert isinstance(report["overlapping_pairs"], list)
