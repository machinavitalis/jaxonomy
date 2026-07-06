# SPDX-License-Identifier: MIT

"""Regression tests for the diagnostics-family follow-ups:

- ``analyze_phase_activity`` gains a ``name=`` kwarg (API symmetry with
  ``analyze_saturation`` / ``analyze_control_oscillation``).
- ``analyze_saturation`` gains a ``mode="upper_only" | "lower_only" |
  "symmetric"`` kwarg + an auto-promote rule for one-sided actuators
  (throttle / brake / PWM duty cycle).
- ``analyze_horizon_completion`` (new) catches the
  ``LapTimeAccumulator``-style zero-gradient gotcha — a smoothed
  indicator integrator that never crosses its event has a meaningless
  gradient.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from jaxonomy.diagnostics import (
    analyze_horizon_completion,
    analyze_phase_activity,
    analyze_saturation,
)


# -----------------------------------------------------------------------------
# analyze_phase_activity name= kwarg
# -----------------------------------------------------------------------------


def test_analyze_phase_activity_accepts_name_kwarg():
    r = analyze_phase_activity(
        [0, 1, 2], expected_phases=[0, 1, 2, 3], name="gear", warn=False,
    )
    assert "gear:" in r.message


def test_analyze_phase_activity_emits_named_warning():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        analyze_phase_activity(
            [0, 1, 2], expected_phases=[0, 1, 2, 3], name="gear",
        )
    msgs = [str(w.message) for w in caught]
    assert any("gear:" in m for m in msgs), msgs


def test_analyze_phase_activity_without_name_is_unprefixed():
    r = analyze_phase_activity(
        [0, 1, 2], expected_phases=[0, 1, 2, 3], warn=False,
    )
    # No leading "name:" prefix when name= is omitted.
    assert not r.message.startswith(":") and not r.message.startswith(" ")


# -----------------------------------------------------------------------------
# analyze_saturation one-sided mode
# -----------------------------------------------------------------------------


def _throttle_trace():
    """40% at WOT, 25% at idle, 35% in between — the F1 throttle example."""
    return np.concatenate([np.ones(400), np.zeros(250), np.linspace(0, 1, 350)])


def test_analyze_saturation_symmetric_mode_fires_on_one_sided_actuator():
    """Without the followup, a throttle pedal at idle 25% of the time gets
    counted as 'saturated', triggering the warning even though idle is the
    natural rest state."""
    u = _throttle_trace()
    r = analyze_saturation(
        u, lower=0.0, upper=1.0, mode="symmetric", name="throttle", warn=False,
    )
    assert r.warning_triggered  # 0.40 + 0.25 = 0.65 >= 0.5


def test_analyze_saturation_upper_only_mode_ignores_lower_rail():
    """``mode='upper_only'`` only counts the upper rail toward the warning
    threshold — the canonical fix for throttle / brake / PWM signals."""
    u = _throttle_trace()
    r = analyze_saturation(
        u, lower=0.0, upper=1.0, mode="upper_only", name="throttle", warn=False,
    )
    # Both fractions still reported.
    assert r.fraction_at_upper == pytest.approx(0.4, abs=0.01)
    assert r.fraction_at_lower == pytest.approx(0.25, abs=0.01)
    # But only the upper rail counts: 0.40 < 0.50 threshold.
    assert not r.warning_triggered
    assert "upper_only mode" in r.message


def test_analyze_saturation_auto_promotes_on_single_rail():
    """Passing only ``upper=`` (the documented workaround) auto-promotes to
    ``upper_only`` mode so single-rail callers get the right semantics."""
    u = _throttle_trace()
    r = analyze_saturation(
        u, upper=1.0, name="throttle", warn=False,  # lower=None
    )
    assert "upper_only mode" in r.message


def test_analyze_saturation_lower_only_mode():
    u = _throttle_trace()
    r = analyze_saturation(
        u, lower=0.0, upper=1.0, mode="lower_only", warn=False,
    )
    # 25% at lower < 0.50 threshold.
    assert not r.warning_triggered
    assert "lower_only mode" in r.message


def test_analyze_saturation_rejects_unknown_mode():
    with pytest.raises(ValueError, match="mode"):
        analyze_saturation([0.0, 1.0], lower=0.0, upper=1.0, mode="bogus")


# -----------------------------------------------------------------------------
# analyze_horizon_completion (LapTimeAccumulator zero-gradient gotcha)
# -----------------------------------------------------------------------------


def test_horizon_completion_warns_when_pegged_at_horizon():
    """When the accumulator pegs at ~t_end, the underlying event didn't
    fire — jax.grad of this readout is meaningless."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        r = analyze_horizon_completion(
            final_value=120.0, t_end=120.0, name="laptime",
        )
    msgs = [str(w.message) for w in caught]
    assert r.warning_triggered
    assert any("laptime" in m and "completion_ratio" in m for m in msgs), msgs


def test_horizon_completion_silent_on_real_lap_finish():
    """When the lap actually finished comfortably before t_end, no warning
    fires."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        r = analyze_horizon_completion(
            final_value=90.0, t_end=120.0, name="laptime",
        )
    assert not r.warning_triggered
    assert not any("analyze_horizon_completion" in str(w.message) for w in caught)


def test_horizon_completion_accepts_jax_array_input():
    """The final-value input is allowed to be a 0-D JAX array (the natural
    shape of ``ctx[...].continuous_state[0]``)."""
    import jax.numpy as jnp

    r = analyze_horizon_completion(
        final_value=jnp.asarray(110.0), t_end=120.0, name="laptime", warn=False,
    )
    assert r.completion_ratio == pytest.approx(110.0 / 120.0, abs=1e-6)


def test_horizon_completion_validates_inputs():
    with pytest.raises(ValueError, match="t_end"):
        analyze_horizon_completion(final_value=1.0, t_end=0.0)
    with pytest.raises(ValueError, match="atol_fraction"):
        analyze_horizon_completion(
            final_value=1.0, t_end=1.0, atol_fraction=1.5,
        )
