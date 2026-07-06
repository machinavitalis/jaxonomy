# SPDX-License-Identifier: MIT

"""Tests for `jaxonomy.diagnostics`.

Each test corresponds to a real bug the diagnostic would have caught,
plus the happy-path no-warning case.
"""

import warnings

import numpy as np
import pytest

from jaxonomy.diagnostics import (
    analyze_saturation,
    analyze_phase_activity,
    analyze_control_oscillation,
    SaturationReport,
    PhaseActivityReport,
    OscillationReport,
)


# ──────────────────────────────────────────────────────────────────────────
# analyze_saturation
# ──────────────────────────────────────────────────────────────────────────


def test_saturation_normal_signal_no_warning():
    """A signal that uses its range moderately should not warn."""
    rng = np.random.default_rng(0)
    # Centered, well within ±1.0 limits.
    sig = 0.3 * rng.standard_normal(1000)
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # any warning -> test fails
        r = analyze_saturation(sig, lower=-1.0, upper=1.0)
    assert isinstance(r, SaturationReport)
    assert r.fraction_saturated < 0.05
    assert not r.warning_triggered


def test_saturation_bang_bang_triggers_warning():
    """A bang-bang signal at ±limit triggers the warning. This is the v3-v7
    Part-3 BBB gimbal: 95% of the time at -10° saturation."""
    # 95% at -1, 5% in between
    sig = np.concatenate([
        -np.ones(950),
        np.linspace(-0.5, 0.5, 50),
    ])
    with pytest.warns(UserWarning, match="saturation|saturated"):
        r = analyze_saturation(sig, lower=-1.0, upper=1.0)
    assert r.warning_triggered
    assert r.fraction_at_lower > 0.9
    assert r.fraction_at_upper < 0.05


def test_saturation_empty_input_handled():
    """No samples -> empty report, no crash."""
    r = analyze_saturation(np.array([]), lower=-1.0, upper=1.0, warn=False)
    assert r.n_samples == 0
    assert not r.warning_triggered


def test_saturation_one_sided_limit():
    """Only an upper limit specified: only upper-saturation reported."""
    sig = np.array([1.0, 1.0, 1.0, 1.0, 0.5, 0.5])
    r = analyze_saturation(sig, upper=1.0, warn=False)
    assert r.fraction_at_upper > 0.6
    assert r.fraction_at_lower == 0.0


# ──────────────────────────────────────────────────────────────────────────
# analyze_phase_activity
# ──────────────────────────────────────────────────────────────────────────


def test_phase_activity_all_phases_seen_no_warning():
    """All expected phases active at some point — no warning."""
    phases = np.array([0, 0, 1, 1, 1, 2, 2])
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        r = analyze_phase_activity(phases, expected_phases=[0, 1, 2])
    assert not r.warning_triggered
    assert r.phases_seen == [0, 1, 2]
    assert r.never_fired == []


def test_phase_activity_silent_phase_triggers_warning():
    """A declared phase that never fires triggers a warning. This is the
    v7-v17 Part-3 dispatcher bug: BBB silently disabled."""
    # Phase 0 (BBB) never fires: simulation only sees 1 (glide) and 2 (landing)
    phases = np.concatenate([
        np.ones(50, dtype=int),
        2 * np.ones(50, dtype=int),
    ])
    with pytest.warns(UserWarning, match="NEVER FIRED|never fires"):
        r = analyze_phase_activity(phases, expected_phases=[0, 1, 2])
    assert r.warning_triggered
    assert r.never_fired == [0]


def test_phase_activity_float_input_rounded():
    """Phase signals may come from float-valued state-machine outputs."""
    phases = np.array([0.0, 0.0, 1.0, 1.0, 2.0, 2.0])
    r = analyze_phase_activity(phases, expected_phases=[0, 1, 2], warn=False)
    assert r.phases_seen == [0, 1, 2]
    assert not r.warning_triggered


def test_phase_activity_no_expected_phases_no_warning():
    """Without `expected_phases`, the diagnostic just reports; no warning."""
    phases = np.array([0, 0, 1, 1])
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        r = analyze_phase_activity(phases)
    assert r.phases_seen == [0, 1]
    assert not r.warning_triggered


# ──────────────────────────────────────────────────────────────────────────
# analyze_control_oscillation
# ──────────────────────────────────────────────────────────────────────────


def test_oscillation_smooth_signal_no_warning():
    """A smooth control signal does not warn."""
    t = np.linspace(0, 10, 1000)
    u = 0.5 + 0.3 * np.sin(2 * np.pi * 0.1 * t)  # 0.1 Hz, gentle
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        r = analyze_control_oscillation(u, t)
    assert not r.warning_triggered
    assert r.crossings_per_second < 1.0


def test_oscillation_bang_bang_triggers_warning():
    """A high-frequency oscillating signal triggers a warning."""
    t = np.linspace(0, 10, 10000)
    u = np.sign(np.sin(2 * np.pi * 10 * t))  # 10 Hz square wave
    with pytest.warns(UserWarning, match="bang-bang|crossings"):
        r = analyze_control_oscillation(u, t, warn_threshold_per_second=5.0)
    assert r.warning_triggered
    assert r.crossings_per_second > 5.0


def test_oscillation_short_input_handled():
    """Fewer than 3 samples returns an empty report without crashing."""
    r = analyze_control_oscillation(np.array([0.0, 1.0]), np.array([0.0, 1.0]), warn=False)
    assert r.zero_crossings == 0
    assert not r.warning_triggered


if __name__ == "__main__":
    # Run all directly for a quick local smoke test
    test_saturation_normal_signal_no_warning()
    test_saturation_bang_bang_triggers_warning()
    test_saturation_empty_input_handled()
    test_saturation_one_sided_limit()
    test_phase_activity_all_phases_seen_no_warning()
    test_phase_activity_silent_phase_triggers_warning()
    test_phase_activity_float_input_rounded()
    test_phase_activity_no_expected_phases_no_warning()
    test_oscillation_smooth_signal_no_warning()
    test_oscillation_bang_bang_triggers_warning()
    test_oscillation_short_input_handled()
    print("All diagnostics tests passed.")
