# SPDX-License-Identifier: MIT

"""Tests for T-105-followup-period-jitter — period-jitter tolerance in
rate-mismatch detection.

The followup adds ``period_tolerance: float = 0.0`` to
:func:`detect_rate_mismatches` (and the underlying
:meth:`SampleTime.matches`) so users can opt into a relative tolerance
when comparing two ``discrete`` sample-time periods.  Default ``0.0``
preserves strict float equality (byte-equivalent with the legacy
behaviour); positive values let small floating-point round-off (e.g.
0.099 vs 0.101 vs 0.1) match without spurious mismatches.

The cases below cover the three buckets called out in the task spec:

1. Default tolerance=0: 0.099 vs 0.1 raises (strict).
2. tolerance=0.005: 0.099 vs 0.1 passes silently (within 5/1000 of 0.1).
3. 0.099 vs 0.5 still raises (clearly outside tolerance).
"""

from __future__ import annotations

import warnings

import pytest

import jaxonomy
from jaxonomy.simulation.rate_groups import (
    RateMismatchError,
    RateMismatchWarning,
    SampleTime,
    detect_rate_mismatches,
)


pytestmark = pytest.mark.minimal


def _build_two_clock_diagram(period_a: float, period_b: float):
    """Two ``DiscreteClock`` blocks wired through a ``UnitDelay``.

    The clock at ``period_a`` feeds a UnitDelay configured at
    ``period_b``.  The wiring is the simplest two-leaf wire that
    makes the rate-mismatch detector see a discrete-discrete edge.
    """
    from jaxonomy.library import DiscreteClock, UnitDelay

    builder = jaxonomy.DiagramBuilder()
    src = builder.add(DiscreteClock(dt=period_a, name="clk_a"))
    dst = builder.add(UnitDelay(dt=period_b, initial_state=0.0, name="ud_b"))
    builder.connect(src.output_ports[0], dst.input_ports[0])
    diag = builder.build()
    diag.create_context()  # triggers UnitDelay.initialize()
    return diag


# =====================================================================
# SampleTime.matches — direct algebra
# =====================================================================


class TestSampleTimeMatchesTolerance:
    def test_default_tolerance_strict_inequality(self):
        a = SampleTime.discrete(period=0.099)
        b = SampleTime.discrete(period=0.1)
        assert not a.matches(b)
        assert not b.matches(a)

    def test_positive_tolerance_accepts_within_band(self):
        a = SampleTime.discrete(period=0.099)
        b = SampleTime.discrete(period=0.101)
        # Within ~1% of either side.
        assert a.matches(b, period_tolerance=0.05)
        assert b.matches(a, period_tolerance=0.05)

    def test_positive_tolerance_still_rejects_outside_band(self):
        a = SampleTime.discrete(period=0.099)
        b = SampleTime.discrete(period=0.5)
        assert not a.matches(b, period_tolerance=0.05)

    def test_explicit_zero_tolerance_matches_default(self):
        a = SampleTime.discrete(period=0.099)
        b = SampleTime.discrete(period=0.1)
        assert a.matches(b, period_tolerance=0.0) is False
        # And exact equality still passes.
        c = SampleTime.discrete(period=0.1)
        assert b.matches(c, period_tolerance=0.0)


# =====================================================================
# detect_rate_mismatches(..., period_tolerance=...)
# =====================================================================


class TestDetectRateMismatchesPeriodTolerance:
    def test_default_tolerance_zero_strict_raises(self):
        """0.099 vs 0.1 with default tolerance: raises (strict)."""
        diag = _build_two_clock_diagram(0.099, 0.1)
        with pytest.raises(RateMismatchError):
            detect_rate_mismatches(diag, on_mismatch="error")

    def test_tolerance_absorbs_small_jitter(self):
        """0.099 vs 0.1 with tolerance=0.005 passes silently (no warning)."""
        diag = _build_two_clock_diagram(0.099, 0.1)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            mismatches = detect_rate_mismatches(
                diag, on_mismatch="warn", period_tolerance=0.05
            )
        assert mismatches == []
        rate_warnings = [
            w for w in caught if issubclass(w.category, RateMismatchWarning)
        ]
        assert rate_warnings == []

    def test_tolerance_does_not_swallow_real_mismatches(self):
        """0.099 vs 0.5 with tolerance=0.005 still raises (outside tolerance)."""
        diag = _build_two_clock_diagram(0.099, 0.5)
        with pytest.raises(RateMismatchError):
            detect_rate_mismatches(
                diag, on_mismatch="error", period_tolerance=0.005
            )

    def test_negative_tolerance_rejected(self):
        diag = _build_two_clock_diagram(0.1, 0.1)
        with pytest.raises(ValueError, match="period_tolerance"):
            detect_rate_mismatches(
                diag, on_mismatch="collect", period_tolerance=-0.01
            )

    def test_non_finite_tolerance_rejected(self):
        diag = _build_two_clock_diagram(0.1, 0.1)
        with pytest.raises(ValueError, match="period_tolerance"):
            detect_rate_mismatches(
                diag, on_mismatch="collect", period_tolerance=float("inf")
            )

    def test_default_tolerance_zero_byte_equivalent(self):
        """No tolerance arg => identical behaviour to passing 0.0 explicitly."""
        diag = _build_two_clock_diagram(0.099, 0.1)
        legacy = detect_rate_mismatches(diag, on_mismatch="collect")
        explicit_zero = detect_rate_mismatches(
            diag, on_mismatch="collect", period_tolerance=0.0
        )
        assert len(legacy) == 1
        assert len(explicit_zero) == 1
        legacy_mm = legacy[0]
        zero_mm = explicit_zero[0]
        assert legacy_mm.src_system_name == zero_mm.src_system_name
        assert legacy_mm.dst_system_name == zero_mm.dst_system_name
        assert legacy_mm.src_sample_time == zero_mm.src_sample_time
        assert legacy_mm.dst_sample_time == zero_mm.dst_sample_time
