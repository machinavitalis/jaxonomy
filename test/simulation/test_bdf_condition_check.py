# SPDX-License-Identifier: MIT
"""Tests for the BDF Newton-iteration condition-number diagnostic
(T-038a-followup-bdf-condition-check).

Companion to ``test_dae_drift_threshold.py`` (DAE drift threshold) and
``test_dae_projection.py`` (Newton projection).  Verifies opt-in
semantics, the aggregated warning surface (one warning at simulate()
exit, not per major step), threshold above/below condition-number
behaviour, and the default-off byte-equivalent path.

The check is wired host-side via ``jax.debug.callback`` from inside
the BDF ``newton_iteration`` — the callback updates a Python-side
``_BDFConditionMonitor`` that tracks the running max plus
time-of-occurrence, then ``simulate`` emits ONE ``UserWarning`` on
exit if the max exceeded the threshold.
"""

from __future__ import annotations

import re
import warnings

import numpy as np
import pytest
import jax.numpy as jnp

import jaxonomy
from jaxonomy import LeafSystem, SimulatorOptions, simulate
from jaxonomy.testing.markers import skip_if_not_jax, requires_jax

skip_if_not_jax()


# ---------------------------------------------------------------------------
# Test systems
# ---------------------------------------------------------------------------


class DampedHarmonic(LeafSystem):
    """Well-conditioned 2-D damped harmonic oscillator.

    ``x' = v``, ``v' = -2v - 4x``.  Newton matrix ``M - c*J`` has
    moderate condition number (typically O(10) at any reasonable step
    size).  Used as the "below threshold → no warning" case.
    """

    def __init__(self, name=None):
        super().__init__(name=name)
        self.declare_continuous_state(
            default_value=jnp.array([1.0, 0.0]),
            ode=self._ode,
        )
        self.declare_continuous_state_output(name="x")

    def _ode(self, time, state, **params):
        x = state.continuous_state
        return jnp.array([x[1], -2.0 * x[1] - 4.0 * x[0]])


class TwoTimescale(LeafSystem):
    """Stiff 2-D linear ODE with widely-separated eigenvalues.

    ``x' = -1e6 x``, ``y' = -1 y``.  Spectrum spans six decades, so the
    BDF Newton matrix ``M - c*J`` has condition number >> 1 at any
    step size that doesn't drive ``c`` to zero.  Used as the "above
    threshold → warning fires" case.
    """

    def __init__(self, name=None):
        super().__init__(name=name)
        self.declare_continuous_state(
            default_value=jnp.array([1.0, 1.0]),
            ode=self._ode,
        )
        self.declare_continuous_state_output(name="x")

    def _ode(self, time, state, **params):
        x = state.continuous_state
        return jnp.array([-1e6 * x[0], -1.0 * x[1]])


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


_WARN_RE = re.compile(
    r"BDF Newton-iteration condition number ([\d.e+-]+) exceeds threshold "
    r"([\d.e+-]+) \(max observed at t=([\d.+-]+)\)"
)


def _condition_warnings(records):
    return [
        r for r in records
        if issubclass(r.category, UserWarning)
        and "BDF Newton-iteration condition number" in str(r.message)
    ]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_default_off_no_warning():
    """Default ``SimulatorOptions()`` emits no condition-number warning."""
    assert SimulatorOptions().bdf_condition_warning_threshold is None
    sys = TwoTimescale()
    ctx = sys.create_context()
    opts = SimulatorOptions(
        math_backend="jax", ode_solver_method="bdf",
    )
    with warnings.catch_warnings(record=True) as records:
        warnings.simplefilter("always")
        simulate(sys, ctx, (0.0, 1e-3), options=opts)
    assert _condition_warnings(records) == []


@requires_jax()
def test_below_threshold_no_warning():
    """Well-conditioned damped oscillator + high threshold = no warning."""
    sys = DampedHarmonic()
    ctx = sys.create_context()
    opts = SimulatorOptions(
        math_backend="jax", ode_solver_method="bdf",
        bdf_condition_warning_threshold=1e8,
    )
    with warnings.catch_warnings(record=True) as records:
        warnings.simplefilter("always")
        simulate(sys, ctx, (0.0, 5.0), options=opts)
    assert _condition_warnings(records) == []


@requires_jax()
def test_above_threshold_warns():
    """Stiff multi-timescale ODE + low threshold = ``UserWarning`` fires.

    The two-timescale system has a Newton matrix with condition number
    well above 10 at any non-trivial step size, so a threshold of 10
    triggers the warning.  The warning message must name the
    threshold, the max value, and the time of occurrence.
    """
    sys = TwoTimescale()
    ctx = sys.create_context()
    opts = SimulatorOptions(
        math_backend="jax", ode_solver_method="bdf",
        bdf_condition_warning_threshold=10.0,
    )
    with warnings.catch_warnings(record=True) as records:
        warnings.simplefilter("always")
        simulate(sys, ctx, (0.0, 1e-3), options=opts)
    cond_warns = _condition_warnings(records)
    assert len(cond_warns) >= 1, (
        "Expected at least one condition-number warning; got none."
    )
    msg = str(cond_warns[0].message)
    m = _WARN_RE.search(msg)
    assert m is not None, f"Warning message did not match expected format: {msg!r}"
    max_val = float(m.group(1))
    threshold_val = float(m.group(2))
    t_at_max = float(m.group(3))
    assert max_val > threshold_val, (
        f"Reported max ({max_val}) should exceed threshold ({threshold_val})."
    )
    assert threshold_val == pytest.approx(10.0)
    # Time of occurrence must lie inside the simulation interval.
    assert 0.0 <= t_at_max <= 1e-3 + 1e-9


@requires_jax()
def test_warning_fires_once_at_exit():
    """The aggregated diagnostic emits ONE warning at simulate() exit,
    not one per major step (the entire point of the aggregation
    surface).  Many BDF major steps are expected on the stiff system,
    so a per-step emission would produce many duplicates.
    """
    sys = TwoTimescale()
    ctx = sys.create_context()
    opts = SimulatorOptions(
        math_backend="jax", ode_solver_method="bdf",
        bdf_condition_warning_threshold=10.0,
    )
    with warnings.catch_warnings(record=True) as records:
        warnings.simplefilter("always")
        simulate(sys, ctx, (0.0, 1e-3), options=opts)
    cond_warns = _condition_warnings(records)
    assert len(cond_warns) == 1, (
        f"Expected exactly one aggregated warning, got {len(cond_warns)}: "
        f"{[str(w.message) for w in cond_warns]}"
    )


@requires_jax()
def test_threshold_above_max_no_warning():
    """Threshold set well above any condition number observed: no warning.

    Even on the stiff TwoTimescale system, a threshold of 1e20 is
    above any plausible 2x2 condition number, so no warning fires.
    Demonstrates that the threshold is honoured in both directions.
    """
    sys = TwoTimescale()
    ctx = sys.create_context()
    opts = SimulatorOptions(
        math_backend="jax", ode_solver_method="bdf",
        bdf_condition_warning_threshold=1e20,
    )
    with warnings.catch_warnings(record=True) as records:
        warnings.simplefilter("always")
        simulate(sys, ctx, (0.0, 1e-3), options=opts)
    assert _condition_warnings(records) == []


@requires_jax()
def test_non_bdf_solver_silently_ignores():
    """Non-BDF solver + threshold set: no warning, no error.

    The diagnostic is BDF-specific (it monitors the Newton-system
    matrix that only BDF builds).  Setting the option on a Dopri5 run
    must not raise — the simulator silently skips attaching the
    monitor.
    """
    sys = DampedHarmonic()
    ctx = sys.create_context()
    opts = SimulatorOptions(
        math_backend="jax", ode_solver_method="dopri5",
        bdf_condition_warning_threshold=1.0,
    )
    with warnings.catch_warnings(record=True) as records:
        warnings.simplefilter("always")
        simulate(sys, ctx, (0.0, 1.0), options=opts)
    assert _condition_warnings(records) == []
