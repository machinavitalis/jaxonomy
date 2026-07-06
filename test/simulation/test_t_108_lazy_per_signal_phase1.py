# SPDX-License-Identifier: MIT
"""
T-108 phase 1 — Lazy per-signal native sample-time accessors.

Covers the public surface added on top of the per-signal-timestamps
machinery shipped by T-013 / T-013a:

  - ``LazyResults.signal(name)`` returns ``(time, value)`` at the
    signal's native cadence — for a periodic signal the returned
    ``time`` is shorter than the global time vector.
  - ``LazyResults.cadence_of(name)`` classifies the recording cadence
    as ``"continuous"``, ``"periodic"``, ``"event-driven"``, or
    ``"default"`` based on the recorded-array shapes (no live
    OutputPort introspection required).
  - ``LazyResults.align_to(name)`` resamples every signal to
    ``name``'s native cadence — convenience wrapper over
    ``resample`` that targets a per-signal time vector.
  - Default-off: when ``per_signal_timestamps`` is not set on the
    simulator, ``cadence_of`` returns ``"default"`` for every signal
    and ``signal()`` returns the global time vector — fully
    backwards-compatible with the legacy LazyResults surface.

Each test directly constructs a :class:`SimulationResults` to avoid
coupling phase-1 verification to the live recording pipeline (the
underlying machinery is already covered by ``test_per_signal_timestamps.py``).
"""

from __future__ import annotations

import numpy as np
import pytest

from jaxonomy.simulation import LazyResults, SimulationResults


# ── fixtures ──────────────────────────────────────────────────────────────


def _make_legacy_results():
    """A SimulationResults with no per-signal cadence info (default off)."""
    time = np.linspace(0.0, 1.0, 21)
    outputs = {
        "x": np.exp(-time),
        "y": np.sin(time),
    }
    return SimulationResults(
        context=None,
        time=time,
        outputs=outputs,
        per_signal_times=None,
    )


def _make_mode_a_results():
    """Mode A-style result: continuous signal + periodic 1 Hz signal.

    ``continuous_y`` runs at every major step.  ``periodic_y`` is
    trimmed to its 1 Hz schedule on BOTH the time vector AND the
    output vector (Mode A storage saving).
    """
    time = np.linspace(0.0, 10.0, 101)
    cont_y = np.exp(-time)
    # 1 Hz schedule: 11 ticks at t = 0, 1, 2, ..., 10.
    periodic_t = np.arange(0.0, 10.0 + 1e-12, 1.0)
    periodic_y = periodic_t * 2.0  # arbitrary content
    return SimulationResults(
        context=None,
        time=time,
        outputs={
            "continuous_y": cont_y,
            "periodic_y": periodic_y,
        },
        per_signal_times={
            "continuous_y": time,
            "periodic_y": periodic_t,
        },
    )


def _make_mode_b_results():
    """Mode B-style result: a value-diff-deduped signal.

    ``event_y`` jumps at three points; outputs stay at full length
    while ``per_signal_times["event_y"]`` is the deduplicated subset.
    """
    time = np.linspace(0.0, 1.0, 21)
    event_y = np.zeros_like(time)
    event_y[time >= 0.3] = 1.0
    event_y[time >= 0.6] = 2.0
    event_y[time >= 0.9] = 3.0
    # Mode B keeps the indices where the value changed (plus t=0).
    change_idx = [0]
    for i in range(1, len(time)):
        if event_y[i] != event_y[i - 1]:
            change_idx.append(i)
    deduped_t = time[change_idx]
    return SimulationResults(
        context=None,
        time=time,
        outputs={"event_y": event_y},
        per_signal_times={"event_y": deduped_t},
    )


# ── signal(name) ──────────────────────────────────────────────────────────


def test_signal_returns_native_time_and_value_for_periodic():
    """For a periodic 1 Hz signal in a 10 s sim, signal() returns ~11
    samples — much shorter than the global vector (101 samples)."""
    res = _make_mode_a_results()
    lazy = res.lazy()
    t, v = lazy.signal("periodic_y")
    assert t.shape == v.shape
    assert t.shape[0] < np.asarray(res.time).shape[0]
    # Exact 1 Hz schedule.
    assert t.shape[0] == 11
    np.testing.assert_allclose(t, np.arange(0.0, 11.0))
    np.testing.assert_allclose(v, t * 2.0)


def test_signal_continuous_matches_global_time():
    res = _make_mode_a_results()
    t, v = res.lazy().signal("continuous_y")
    np.testing.assert_array_equal(t, np.asarray(res.time))
    np.testing.assert_array_equal(v, np.asarray(res.outputs["continuous_y"]))


def test_signal_legacy_falls_back_to_global_time():
    """Default-off path: signal() still works, returns the global vector."""
    res = _make_legacy_results()
    t, v = res.lazy().signal("x")
    np.testing.assert_array_equal(t, np.asarray(res.time))
    np.testing.assert_array_equal(v, np.asarray(res.outputs["x"]))


def test_signal_mode_b_back_projects_value():
    """Mode B "event-driven": the deduped time vector is shorter than
    the full output vector; signal() back-projects via searchsorted so
    the returned (t, v) pair still has consistent shape."""
    res = _make_mode_b_results()
    t, v = res.lazy().signal("event_y")
    assert t.shape == v.shape
    # Four edges: t=0 (value 0), then 0.3 (1), 0.6 (2), 0.9 (3).
    assert t.shape[0] == 4
    np.testing.assert_array_equal(v, np.array([0.0, 1.0, 2.0, 3.0]))


def test_signal_unknown_raises():
    res = _make_mode_a_results()
    with pytest.raises(KeyError, match="unknown signal"):
        res.lazy().signal("nope")


# ── cadence_of(name) ──────────────────────────────────────────────────────


def test_cadence_of_classifies_continuous_periodic_default():
    res = _make_mode_a_results()
    lazy = res.lazy()
    assert lazy.cadence_of("continuous_y") == "continuous"
    assert lazy.cadence_of("periodic_y") == "periodic"


def test_cadence_of_classifies_event_driven():
    res = _make_mode_b_results()
    assert res.lazy().cadence_of("event_y") == "event-driven"


def test_cadence_of_default_when_no_per_signal_times():
    res = _make_legacy_results()
    lazy = res.lazy()
    assert lazy.cadence_of("x") == "default"
    assert lazy.cadence_of("y") == "default"


def test_cadence_of_unknown_raises():
    res = _make_mode_a_results()
    with pytest.raises(KeyError, match="unknown signal"):
        res.lazy().cadence_of("nope")


# ── align_to(name) ────────────────────────────────────────────────────────


def test_align_to_resamples_continuous_onto_periodic_grid():
    """Align continuous_y onto the 1 Hz periodic_y grid."""
    res = _make_mode_a_results()
    # The Mode A periodic outputs is shorter than the global vector,
    # so collect()-ing the full chain would be inconsistent.  Project
    # onto a sub-range: keep just continuous_y and resample to the
    # periodic grid.
    out = res.lazy().select("continuous_y").align_to("periodic_y").collect()
    expected_t = np.arange(0.0, 11.0)
    np.testing.assert_allclose(out["time"], expected_t)
    # exp(-t) at t = 0, 1, ..., 10 — close to the analytic answer
    # because the source grid is 100-point dense.
    np.testing.assert_allclose(
        out["continuous_y"], np.exp(-expected_t), atol=1e-3,
    )


def test_align_to_self_is_identity_on_continuous():
    """Aligning a continuous signal to its own (== global) grid is a no-op."""
    res = _make_mode_a_results()
    out = res.lazy().select("continuous_y").align_to("continuous_y").collect()
    np.testing.assert_array_equal(out["time"], np.asarray(res.time))
    np.testing.assert_allclose(
        out["continuous_y"], np.asarray(res.outputs["continuous_y"]),
    )


def test_align_to_unknown_raises():
    res = _make_mode_a_results()
    with pytest.raises(KeyError, match="unknown signal"):
        res.lazy().align_to("nope")


# ── backwards compatibility / smoke ───────────────────────────────────────


def test_legacy_lazy_results_unchanged():
    """Existing LazyResults surface (collect / select / where / resample)
    is byte-equivalent on a default-off result — no per-signal-cadence
    plumbing leaks into the legacy path."""
    res = _make_legacy_results()
    out = res.lazy().select("x").collect()
    assert set(out) == {"time", "x"}
    np.testing.assert_array_equal(out["time"], np.asarray(res.time))
    np.testing.assert_array_equal(out["x"], np.asarray(res.outputs["x"]))


def test_per_signal_times_threaded_through_chain_ops():
    """Carrying _per_signal_times across _chain() means cadence_of()
    still classifies after other ops are appended (the new field is
    propagated through the dataclass copy)."""
    res = _make_mode_a_results()
    chain = res.lazy().select("continuous_y", "periodic_y")
    # cadence_of reads from the underlying _per_signal_times; it
    # should keep returning the original classifications regardless of
    # how the deferred chain has grown.
    assert chain.cadence_of("continuous_y") == "continuous"
    assert chain.cadence_of("periodic_y") == "periodic"
