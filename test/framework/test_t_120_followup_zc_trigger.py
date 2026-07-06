# SPDX-License-Identifier: MIT

"""T-120-followup-zc-trigger — ``ZeroCrossingTriggeredSubsystem``.

Covers the zero-crossing-driven triggered-subsystem variant deferred
from T-120 phase 1. Phase 1 ships a periodic-grid edge detector
(:class:`TriggeredSubsystem`); this follow-up adds a block that hooks
into the framework's continuous zero-crossing detector so the latched
event time has sub-sample-period precision.

Tests assert:

- The submodel fires *at* the analytical zero-crossing instant
  (``t = 0.5`` for a guard ``time - 0.5``), regardless of the
  integrator's step size.
- Compared to the phase-1 periodic-grid block sampled at
  ``sample_period = 0.1``, the ZC version's latched output reflects
  the crossing instant to within integrator tolerance, while the
  periodic block only resolves it to the nearest sample.
- ``"rising"``, ``"falling"`` and ``"either"`` edge selection routes
  through the framework's matching direction string and fires only on
  the requested transition.
- Multiple consecutive crossings are handled correctly: the latched
  output advances on every qualifying crossing, not just the first.
- Bad inputs raise at construction time (parity with the phase-1
  block).
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

import jaxonomy
from jaxonomy.framework import (
    TriggerEdge,
    TriggeredSubsystem,
    ZeroCrossingTriggeredSubsystem,
)
from jaxonomy.library import Clock, Constant
from jaxonomy.testing.markers import skip_if_not_jax


skip_if_not_jax()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _identity(u):
    """Submodel that returns its input — handy for latching the trigger
    instant when the submodel is fed a Clock."""
    return u


def _scale_by_two(u):
    return 2.0 * u


class _Ramp(jaxonomy.LeafSystem):
    """Source emitting ``time - shift`` via a continuous-state integrator.

    Integrating ``dx/dt = 1`` from ``x(0) = -shift`` gives ``x(t) = t - shift``,
    which crosses zero exactly once at ``t = shift``. Driving the trigger
    via continuous state (rather than a pure ``time`` source) is what
    lets the framework's zero-crossing detector localize the crossing
    with bisection precision; a pure ``time`` source has no continuous
    trajectory for the integrator to bisect on.
    """

    def __init__(self, shift: float, name: str | None = None):
        super().__init__(name=name)
        self._shift = float(shift)
        self.declare_continuous_state(
            default_value=jnp.array(-float(shift)),
            ode=lambda t, s, **p: jnp.array(1.0),
        )
        self.declare_output_port(lambda t, s, *u, **p: s.continuous_state)


class _NegRamp(jaxonomy.LeafSystem):
    """Source emitting ``shift - time`` via a continuous-state integrator
    (``dx/dt = -1``, ``x(0) = shift``). Crosses zero going DOWN at
    ``t = shift`` — useful for testing the falling-edge path."""

    def __init__(self, shift: float, name: str | None = None):
        super().__init__(name=name)
        self._shift = float(shift)
        self.declare_continuous_state(
            default_value=jnp.array(float(shift)),
            ode=lambda t, s, **p: jnp.array(-1.0),
        )
        self.declare_output_port(lambda t, s, *u, **p: s.continuous_state)


class _Sin(jaxonomy.LeafSystem):
    """Source emitting ``sin(omega * t)`` via a 2-state continuous oscillator.

    Implementation: ``dx/dt = omega * y``, ``dy/dt = -omega * x`` with
    ``x(0) = 0``, ``y(0) = 1`` gives ``x(t) = sin(omega t)``. The pair
    of continuous states gives the framework a real ODE trajectory to
    bisect on for precise zero-crossing localization.
    """

    def __init__(self, omega: float, name: str | None = None):
        super().__init__(name=name)
        self._omega = float(omega)

        def _ode(t, s, **p):
            x, y = s.continuous_state[0], s.continuous_state[1]
            w = self._omega
            return jnp.array([w * y, -w * x])

        self.declare_continuous_state(
            default_value=jnp.array([0.0, 1.0]),
            ode=_ode,
        )
        self.declare_output_port(
            lambda t, s, *u, **p: s.continuous_state[0]
        )


# ---------------------------------------------------------------------------
# Construction / validation
# ---------------------------------------------------------------------------


def test_zc_triggered_subsystem_invalid_edge_raises():
    with pytest.raises(ValueError, match="edge"):
        ZeroCrossingTriggeredSubsystem(_scale_by_two, n_inputs=1, edge="bogus")


def test_zc_triggered_subsystem_invalid_n_inputs_raises():
    with pytest.raises(ValueError, match="n_inputs"):
        ZeroCrossingTriggeredSubsystem(_scale_by_two, n_inputs=-1)


def test_zc_triggered_subsystem_initial_output_is_initial_value():
    """Before any crossing fires, output equals the initial latch."""
    blk = ZeroCrossingTriggeredSubsystem(
        _scale_by_two,
        n_inputs=1,
        edge=TriggerEdge.RISING,
        initial_value=jnp.array(-7.0),
    )
    bld = jaxonomy.DiagramBuilder()
    # Use a constant trigger of -1 (negative, no crossing event).
    trig = bld.add(Constant(jnp.array(-1.0), name="trig"))
    u = bld.add(Constant(jnp.array(3.0), name="u"))
    c = bld.add(blk)
    bld.connect(trig.output_ports[0], c.input_ports[0])
    bld.connect(u.output_ports[0], c.input_ports[1])
    diagram = bld.build()
    ctx = diagram.create_context()
    y = c.output_ports[0].eval(ctx)
    assert float(y) == -7.0


# ---------------------------------------------------------------------------
# Sub-sample precision: trigger fires AT the analytical zero crossing
# ---------------------------------------------------------------------------


def _build_zc_with_clock_latch(edge: str, shift: float):
    """ZC-triggered block whose submodel returns the current Clock value,
    fed by a continuous ``_Ramp`` trigger that crosses zero at
    ``t = shift``. After simulation past ``shift``, the discrete state
    holds the simulator time AT the crossing."""
    blk = ZeroCrossingTriggeredSubsystem(
        _identity,
        n_inputs=1,
        edge=edge,
        initial_value=jnp.array(0.0),
    )
    bld = jaxonomy.DiagramBuilder()
    trig = bld.add(_Ramp(shift=shift, name="trig"))
    clk = bld.add(Clock(name="clk"))
    c = bld.add(blk)
    bld.connect(trig.output_ports[0], c.input_ports[0])
    bld.connect(clk.output_ports[0], c.input_ports[1])
    diagram = bld.build()
    ctx = diagram.create_context()
    return diagram, ctx, c


def test_zc_triggered_fires_at_exact_crossing_time():
    """Trigger crosses zero exactly at t=0.5; the latched output (Clock
    at the crossing) must equal 0.5 to integrator tolerance, not to
    the major-step grid."""
    diagram, ctx, c = _build_zc_with_clock_latch(TriggerEdge.RISING, shift=0.5)
    opts = jaxonomy.SimulatorOptions(
        math_backend="jax", rtol=1e-9, atol=1e-11, max_major_step_length=0.1,
    )
    res = jaxonomy.simulate(diagram, ctx, (0.0, 1.0), options=opts)
    latched = float(np.asarray(res.context[c.system_id].discrete_state))
    # Sub-sample-period precision: must match 0.5 to integrator tol,
    # not to the 0.1-second major-step grid.
    assert latched == pytest.approx(0.5, abs=1e-6), (
        f"latched time {latched} not within integrator tol of 0.5"
    )


def test_zc_triggered_beats_periodic_grid_resolution():
    """Compare ZC against phase-1 periodic-grid TriggeredSubsystem at
    ``sample_period=0.1``. The ZC version must localize the crossing
    more precisely than the grid version.

    Use a crossing time of 0.55 so the periodic grid (which can only
    sample at t ∈ {0.0, 0.1, …}) cannot align exactly with the true
    crossing."""

    # ZC version: should land at exactly 0.55 (to integrator tol).
    diagram_zc, ctx_zc, c_zc = _build_zc_with_clock_latch(
        TriggerEdge.RISING, shift=0.55,
    )
    opts = jaxonomy.SimulatorOptions(
        math_backend="jax", rtol=1e-9, atol=1e-11, max_major_step_length=0.1,
    )
    res_zc = jaxonomy.simulate(diagram_zc, ctx_zc, (0.0, 1.0), options=opts)
    latched_zc = float(np.asarray(res_zc.context[c_zc.system_id].discrete_state))

    # Periodic-grid version: trigger as a 0/1 step at t=0.55, sampled
    # every 0.1s. On the sample grid the signal first reads 1 at t=0.6,
    # so the latched Clock value is at best 0.6 (50ms after the true
    # crossing).
    blk_grid = TriggeredSubsystem(
        _identity,
        n_inputs=1,
        edge=TriggerEdge.RISING,
        sample_period=0.1,
        initial_value=jnp.array(0.0),
    )
    bld = jaxonomy.DiagramBuilder()

    class _Step(jaxonomy.LeafSystem):
        def __init__(self, **kw):
            super().__init__(**kw)
            self.declare_output_port(
                lambda t, s, *u, **p: jnp.where(t < 0.55, 0.0, 1.0),
                prerequisites_of_calc=[],
                requires_inputs=False,
            )

    trig_g = bld.add(_Step(name="trig_g"))
    clk_g = bld.add(Clock(name="clk_g"))
    c_g = bld.add(blk_grid)
    bld.connect(trig_g.output_ports[0], c_g.input_ports[0])
    bld.connect(clk_g.output_ports[0], c_g.input_ports[1])
    diagram_g = bld.build()
    ctx_g = diagram_g.create_context()
    res_g = jaxonomy.simulate(diagram_g, ctx_g, (0.0, 1.0), options=opts)
    ds_g = np.asarray(res_g.context[c_g.system_id].discrete_state)
    latched_grid = float(ds_g[1])  # discrete_state layout: [prev_trig, latch]

    # Both should have fired and latched something positive.
    assert latched_zc > 0.0, f"ZC variant did not latch (got {latched_zc})"
    assert latched_grid > 0.0, f"grid variant did not latch (got {latched_grid})"

    # ZC is strictly closer to the true crossing time (0.55) than the
    # grid. Concretely: ZC within integrator tol (≤ 1e-6); grid is at
    # least 0.04s off because the next sample after 0.55 is at 0.6.
    err_zc = abs(latched_zc - 0.55)
    err_grid = abs(latched_grid - 0.55)
    assert err_zc < err_grid, (
        f"ZC error {err_zc} should be smaller than grid error {err_grid}"
    )
    assert err_zc < 1e-6, f"ZC error {err_zc} not within integrator tol"
    assert err_grid >= 0.04, (
        f"grid error {err_grid} unexpectedly small (sample_period=0.1)"
    )


# ---------------------------------------------------------------------------
# Edge selection: rising / falling / either
# ---------------------------------------------------------------------------


def test_zc_triggered_falling_edge_does_not_fire_on_rising():
    """Trigger goes negative→positive at t=0.3 (rising). With
    ``edge="falling"`` the block must NOT fire, and the latch must
    stay at the initial value."""
    blk = ZeroCrossingTriggeredSubsystem(
        _identity,
        n_inputs=1,
        edge=TriggerEdge.FALLING,
        initial_value=jnp.array(-1.0),
    )
    bld = jaxonomy.DiagramBuilder()
    trig = bld.add(_Ramp(shift=0.3, name="trig"))  # crosses 0 going up
    clk = bld.add(Clock(name="clk"))
    c = bld.add(blk)
    bld.connect(trig.output_ports[0], c.input_ports[0])
    bld.connect(clk.output_ports[0], c.input_ports[1])
    diagram = bld.build()
    ctx = diagram.create_context()
    opts = jaxonomy.SimulatorOptions(math_backend="jax", rtol=1e-9, atol=1e-11)
    res = jaxonomy.simulate(diagram, ctx, (0.0, 1.0), options=opts)
    latched = float(np.asarray(res.context[c.system_id].discrete_state))
    assert latched == pytest.approx(-1.0), (
        f"falling-only block fired on rising edge: latch={latched}"
    )


def test_zc_triggered_falling_edge_fires_on_falling():
    """Trigger ``shift - t`` is positive at t=0, crosses zero going down
    at t=shift. With ``edge="falling"`` the latch should capture t≈shift."""
    blk = ZeroCrossingTriggeredSubsystem(
        _identity,
        n_inputs=1,
        edge=TriggerEdge.FALLING,
        initial_value=jnp.array(0.0),
    )
    bld = jaxonomy.DiagramBuilder()
    trig = bld.add(_NegRamp(shift=0.4, name="trig"))
    clk = bld.add(Clock(name="clk"))
    c = bld.add(blk)
    bld.connect(trig.output_ports[0], c.input_ports[0])
    bld.connect(clk.output_ports[0], c.input_ports[1])
    diagram = bld.build()
    ctx = diagram.create_context()
    opts = jaxonomy.SimulatorOptions(math_backend="jax", rtol=1e-9, atol=1e-11)
    res = jaxonomy.simulate(diagram, ctx, (0.0, 1.0), options=opts)
    latched = float(np.asarray(res.context[c.system_id].discrete_state))
    assert latched == pytest.approx(0.4, abs=1e-6)


def test_zc_triggered_either_fires_on_rising_and_falling():
    """``edge="either"`` fires on every zero crossing. ``sin(omega t)``
    with ``omega = π/0.4`` (so ``sin(omega t)`` crosses zero at
    t ∈ {0, 0.4, 0.8, …}). Over [0, 0.95] the most recent crossing is
    at t=0.8."""
    omega = float(np.pi / 0.4)
    blk = ZeroCrossingTriggeredSubsystem(
        _identity,
        n_inputs=1,
        edge=TriggerEdge.EITHER,
        initial_value=jnp.array(-1.0),
    )
    bld = jaxonomy.DiagramBuilder()
    trig = bld.add(_Sin(omega=omega, name="trig"))
    clk = bld.add(Clock(name="clk"))
    c = bld.add(blk)
    bld.connect(trig.output_ports[0], c.input_ports[0])
    bld.connect(clk.output_ports[0], c.input_ports[1])
    diagram = bld.build()
    ctx = diagram.create_context()
    opts = jaxonomy.SimulatorOptions(math_backend="jax", rtol=1e-9, atol=1e-11)
    res = jaxonomy.simulate(diagram, ctx, (0.0, 0.95), options=opts)
    latched = float(np.asarray(res.context[c.system_id].discrete_state))

    # Latch must have advanced off the initial value: at least one
    # crossing fired.
    assert latched != -1.0, "either-edge block never fired"
    # The most recent crossing in (0, 0.95] is t = 0.8.
    assert latched == pytest.approx(0.8, abs=1e-4), (
        f"expected latch near 0.8 (last crossing), got {latched}"
    )


# ---------------------------------------------------------------------------
# Multiple crossings: latch advances on every qualifying crossing
# ---------------------------------------------------------------------------


def test_zc_triggered_handles_multiple_rising_crossings():
    """``sin(omega t)`` with ``omega = π/0.4`` and ``edge="rising"``
    has rising-edge zero crossings at t = 0.8, 1.6, ... (the first
    rising-after-zero crossing is at t=0.8 since sin starts at 0 going
    UP — but the framework treats the t=0 sample as the initial value
    so it does NOT count as a crossing). After (0, 0.95], one rising
    crossing at t=0.8 should fire."""
    omega = float(np.pi / 0.4)
    blk = ZeroCrossingTriggeredSubsystem(
        _identity,
        n_inputs=1,
        edge=TriggerEdge.RISING,
        initial_value=jnp.array(-1.0),
    )
    bld = jaxonomy.DiagramBuilder()
    trig = bld.add(_Sin(omega=omega, name="trig"))
    clk = bld.add(Clock(name="clk"))
    c = bld.add(blk)
    bld.connect(trig.output_ports[0], c.input_ports[0])
    bld.connect(clk.output_ports[0], c.input_ports[1])
    diagram = bld.build()
    ctx = diagram.create_context()
    opts = jaxonomy.SimulatorOptions(math_backend="jax", rtol=1e-9, atol=1e-11)
    res = jaxonomy.simulate(diagram, ctx, (0.0, 1.7), options=opts)
    latched = float(np.asarray(res.context[c.system_id].discrete_state))
    # Most recent rising-edge crossing in (0, 1.7] is at t = 1.6.
    # (sin(omega t) crosses zero going UP at t = 0, 0.8, 1.6, ...; we
    # exclude t=0 since the framework's "negative_then_non_negative"
    # detector requires a strictly negative previous sample.)
    assert latched == pytest.approx(1.6, abs=1e-4), (
        f"expected last-rising-crossing latch ≈ 1.6, got {latched}"
    )


# ---------------------------------------------------------------------------
# Submodel takes a non-trigger input — value flows through latch
# ---------------------------------------------------------------------------


def test_zc_triggered_latch_reflects_submodel_output():
    """Submodel = ``2 * u`` with a constant ``u = 4.0``; trigger crosses
    zero at t=0.3. After simulation past 0.3, the latched output is
    ``2 * 4.0 = 8.0``."""
    blk = ZeroCrossingTriggeredSubsystem(
        _scale_by_two,
        n_inputs=1,
        edge=TriggerEdge.RISING,
        initial_value=jnp.array(0.0),
    )
    bld = jaxonomy.DiagramBuilder()
    trig = bld.add(_Ramp(shift=0.3, name="trig"))
    u = bld.add(Constant(jnp.array(4.0), name="u"))
    c = bld.add(blk)
    bld.connect(trig.output_ports[0], c.input_ports[0])
    bld.connect(u.output_ports[0], c.input_ports[1])
    diagram = bld.build()
    ctx = diagram.create_context()
    opts = jaxonomy.SimulatorOptions(math_backend="jax", rtol=1e-9, atol=1e-11)
    res = jaxonomy.simulate(diagram, ctx, (0.0, 1.0), options=opts)
    latched = float(np.asarray(res.context[c.system_id].discrete_state))
    assert latched == pytest.approx(8.0, abs=1e-9), (
        f"expected 2*u = 8.0, got {latched}"
    )
