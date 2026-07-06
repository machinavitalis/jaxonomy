# SPDX-License-Identifier: MIT

"""Regression tests for ``jaxonomy.profiling.Profiler``."""

from __future__ import annotations

import pytest

from jaxonomy.profiling import Profiler


def test_profiler_stop_without_start_does_not_raise():
    """``Profiler.stop`` must be a no-op when ``start`` was never called.

    Previously ``stop`` raised ``KeyError`` inside ``ScopedProfiler.__exit__``,
    which masked the original exception coming out of the wrapped code.
    """
    Profiler.clear()
    # Should silently return rather than raise KeyError.
    Profiler.stop("never_started")


def test_scoped_profiler_does_not_mask_inner_exception():
    """The real exception must propagate through ``ScopedProfiler.__exit__``.

    Construct a scenario where ``start`` succeeds, the body raises, and
    ``stop`` runs cleanly — i.e. the inner exception is what the user sees.
    """
    Profiler.clear()
    with pytest.raises(RuntimeError, match="boom"):
        with Profiler.ScopedProfiler("scope_a"):
            raise RuntimeError("boom")
    # Internal bookkeeping should not retain the started entry.
    assert "scope_a" not in Profiler._profiles


def test_scoped_profiler_records_count_on_clean_exit():
    """Basic happy path: timings accumulate after a normal scope exit."""
    Profiler.clear()
    with Profiler.ScopedProfiler("scope_b"):
        pass
    assert Profiler._counts.get("scope_b") == 1
    assert "scope_b" not in Profiler._profiles  # popped after stop
