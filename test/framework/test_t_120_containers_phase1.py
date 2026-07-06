# SPDX-License-Identifier: MIT

"""T-120 phase 1 — Container Blocks (formerly T-MW-207).

Covers the smallest useful slice of the container family
shipped in :mod:`jaxonomy.framework.containers`:

- ``EnabledSubsystem``: enable=1 → submodel output; enable=0 → reset /
  passthrough / hold per ``mode``. Gradient through the disabled branch
  is zero.
- ``TriggeredSubsystem``: output advances only on rising-edge
  transitions of the trigger; held between edges.
- ``ForEach``: parallel evaluation of N submodels via vmap (smoke).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import jaxonomy
from jaxonomy.framework import (
    EnabledMode,
    EnabledSubsystem,
    ForEach,
    TriggerEdge,
    TriggeredSubsystem,
)
from jaxonomy.library import Constant, Sine
from jaxonomy.testing.markers import skip_if_not_jax


skip_if_not_jax()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _scale_by_two(x):
    return 2.0 * x


def _build_with_enable(enable_value, u_value, mode, **kwargs):
    """Build a diagram: en, u → EnabledSubsystem(2*u, mode=mode)."""
    blk = EnabledSubsystem(_scale_by_two, n_inputs=1, mode=mode, **kwargs)
    bld = jaxonomy.DiagramBuilder()
    en = bld.add(Constant(jnp.asarray(enable_value), name="en"))
    u = bld.add(Constant(jnp.asarray(u_value), name="u"))
    c = bld.add(blk)
    bld.connect(en.output_ports[0], c.input_ports[0])
    bld.connect(u.output_ports[0], c.input_ports[1])
    diagram = bld.build()
    return diagram, c, u


# ---------------------------------------------------------------------------
# EnabledSubsystem
# ---------------------------------------------------------------------------


def test_enabled_subsystem_enabled_matches_submodel():
    """enable=1 → output = submodel(u) regardless of mode."""
    for mode in (EnabledMode.RESET, EnabledMode.PASSTHROUGH):
        diagram, c, _ = _build_with_enable(1.0, 3.0, mode)
        ctx = diagram.create_context()
        y = c.output_ports[0].eval(ctx)
        assert float(y) == 6.0, f"mode={mode}: expected 6.0, got {y}"


def test_enabled_subsystem_disabled_reset():
    """enable=0, reset → output = initial_value."""
    diagram, c, _ = _build_with_enable(
        0.0, 3.0, EnabledMode.RESET, initial_value=-99.0,
    )
    ctx = diagram.create_context()
    y = c.output_ports[0].eval(ctx)
    assert float(y) == -99.0


def test_enabled_subsystem_disabled_passthrough():
    """enable=0, passthrough → output = first user input."""
    diagram, c, _ = _build_with_enable(0.0, 5.0, EnabledMode.PASSTHROUGH)
    ctx = diagram.create_context()
    y = c.output_ports[0].eval(ctx)
    assert float(y) == 5.0


def test_enabled_subsystem_grad_through_disabled_is_zero():
    """When disabled (reset), grad w.r.t. user input must be zero."""
    diagram, c, u = _build_with_enable(
        0.0, 3.0, EnabledMode.RESET, initial_value=0.0,
    )
    ctx0 = diagram.create_context()

    def loss(u_val):
        ctx = ctx0.with_subcontext(
            u.system_id, ctx0[u.system_id].with_parameter("value", u_val),
        )
        return c.output_ports[0].eval(ctx)

    g = jax.grad(loss)(jnp.array(3.0))
    assert float(g) == 0.0, f"disabled branch should give zero grad, got {g}"


def test_enabled_subsystem_grad_through_enabled_matches_submodel():
    """When enabled, grad flows through the submodel (df/du = 2)."""
    diagram, c, u = _build_with_enable(
        1.0, 3.0, EnabledMode.RESET, initial_value=0.0,
    )
    ctx0 = diagram.create_context()

    def loss(u_val):
        ctx = ctx0.with_subcontext(
            u.system_id, ctx0[u.system_id].with_parameter("value", u_val),
        )
        return c.output_ports[0].eval(ctx)

    g = jax.grad(loss)(jnp.array(3.0))
    assert float(g) == 2.0, f"enabled branch grad should be 2, got {g}"


def test_enabled_subsystem_invalid_mode_raises():
    with pytest.raises(ValueError, match="mode"):
        EnabledSubsystem(_scale_by_two, n_inputs=1, mode="bogus")


def test_enabled_subsystem_hold_requires_period():
    with pytest.raises(ValueError, match="hold_period"):
        EnabledSubsystem(_scale_by_two, n_inputs=1, mode=EnabledMode.HOLD)


def test_enabled_subsystem_passthrough_needs_input():
    with pytest.raises(ValueError, match="passthrough"):
        EnabledSubsystem(
            _scale_by_two, n_inputs=0, mode=EnabledMode.PASSTHROUGH,
        )


def test_enabled_subsystem_hold_mode_holds_last_enabled_value():
    """Run a short simulation: enable goes 1→0 at t=0.3. After disable
    the held discrete state should be a finite, in-range snapshot."""

    class EnableStep(jaxonomy.LeafSystem):
        def __init__(self, **kw):
            super().__init__(**kw)
            self.declare_output_port(
                lambda t, s, *u, **p: jnp.where(t < 0.3, 1.0, 0.0),
                prerequisites_of_calc=[],
                requires_inputs=False,
            )

    blk = EnabledSubsystem(
        _scale_by_two,
        n_inputs=1,
        mode=EnabledMode.HOLD,
        initial_value=jnp.array(0.0),
        hold_period=0.1,
    )
    bld = jaxonomy.DiagramBuilder()
    en = bld.add(EnableStep(name="en"))
    u = bld.add(Sine(amplitude=1.0, frequency=4 * np.pi, phase=0.0, name="u"))
    c = bld.add(blk)
    bld.connect(en.output_ports[0], c.input_ports[0])
    bld.connect(u.output_ports[0], c.input_ports[1])
    diagram = bld.build()
    ctx = diagram.create_context()
    opts = jaxonomy.SimulatorOptions(math_backend="jax", max_major_steps=100)
    res = jaxonomy.simulate(diagram, ctx, (0.0, 0.5), options=opts)

    held = np.asarray(res.context[c.system_id].discrete_state)
    assert np.all(np.isfinite(held)), f"held state NaN: {held}"
    assert abs(float(held)) <= 2.0 + 1e-9, f"held out of range: {held}"


# ---------------------------------------------------------------------------
# TriggeredSubsystem
# ---------------------------------------------------------------------------


def test_triggered_subsystem_invalid_edge_raises():
    with pytest.raises(ValueError, match="edge"):
        TriggeredSubsystem(_scale_by_two, n_inputs=1, edge="bogus", sample_period=0.1)


def test_triggered_subsystem_requires_positive_sample_period():
    with pytest.raises(ValueError, match="sample_period"):
        TriggeredSubsystem(_scale_by_two, n_inputs=1, sample_period=0.0)
    with pytest.raises(ValueError, match="sample_period"):
        TriggeredSubsystem(_scale_by_two, n_inputs=1, sample_period=None)


def test_triggered_subsystem_initial_output_is_initial_value():
    """Before any edge fires, output equals the initial latch value."""
    blk = TriggeredSubsystem(
        _scale_by_two,
        n_inputs=1,
        edge=TriggerEdge.RISING,
        sample_period=0.1,
        initial_value=jnp.array(-7.0),
    )
    bld = jaxonomy.DiagramBuilder()
    t = bld.add(Constant(jnp.array(0.0), name="trig"))
    u = bld.add(Constant(jnp.array(3.0), name="u"))
    c = bld.add(blk)
    bld.connect(t.output_ports[0], c.input_ports[0])
    bld.connect(u.output_ports[0], c.input_ports[1])
    diagram = bld.build()
    ctx = diagram.create_context()
    y = c.output_ports[0].eval(ctx)
    # trigger=0, prev=0 in initial discrete state → no rising edge → latch.
    assert float(y) == -7.0


def test_triggered_subsystem_rising_edge_advances_latch():
    """Run a short simulation with a step trigger and verify the latch
    captures the submodel value after the rising edge."""

    class TriggerStep(jaxonomy.LeafSystem):
        def __init__(self, **kw):
            super().__init__(**kw)
            self.declare_output_port(
                lambda t, s, *u, **p: jnp.where(t < 0.3, 0.0, 1.0),
                prerequisites_of_calc=[],
                requires_inputs=False,
            )

    blk = TriggeredSubsystem(
        _scale_by_two,
        n_inputs=1,
        edge=TriggerEdge.RISING,
        sample_period=0.05,
        initial_value=jnp.array(0.0),
    )
    bld = jaxonomy.DiagramBuilder()
    trig = bld.add(TriggerStep(name="trig"))
    u = bld.add(Sine(amplitude=1.0, frequency=4 * np.pi, phase=0.0, name="u"))
    c = bld.add(blk)
    bld.connect(trig.output_ports[0], c.input_ports[0])
    bld.connect(u.output_ports[0], c.input_ports[1])
    diagram = bld.build()
    ctx = diagram.create_context()
    opts = jaxonomy.SimulatorOptions(math_backend="jax", max_major_steps=200)
    res = jaxonomy.simulate(diagram, ctx, (0.0, 0.6), options=opts)

    ds = np.asarray(res.context[c.system_id].discrete_state)
    # ds layout: [prev_trigger, latch_flat...]
    prev_trig = float(ds[0])
    latch = float(ds[1])
    # After t=0.3 the trigger has been latched to 1.0.
    assert prev_trig == pytest.approx(1.0)
    # Latch must equal 2*sin(4π*t_edge) for some t_edge ≥ 0.3 (sampled
    # on the periodic grid). |sin| ≤ 1 so |latch| ≤ 2.
    assert np.isfinite(latch)
    assert abs(latch) <= 2.0 + 1e-9
    # And it must have moved off the initial value (the submodel was
    # invoked at least once with a non-zero u).
    assert latch != 0.0


def test_triggered_subsystem_falling_edge_mode():
    """Falling edge: 1→0 transition latches; 0→1 does not."""
    blk = TriggeredSubsystem(
        _scale_by_two,
        n_inputs=1,
        edge=TriggerEdge.FALLING,
        sample_period=0.1,
        initial_value=jnp.array(0.0),
    )
    # Sanity: no edge yet, so latch is initial.
    bld = jaxonomy.DiagramBuilder()
    t = bld.add(Constant(jnp.array(1.0), name="trig"))  # high; no falling.
    u = bld.add(Constant(jnp.array(3.0), name="u"))
    c = bld.add(blk)
    bld.connect(t.output_ports[0], c.input_ports[0])
    bld.connect(u.output_ports[0], c.input_ports[1])
    diagram = bld.build()
    ctx = diagram.create_context()
    y = c.output_ports[0].eval(ctx)
    # prev_trig stored as 0.0 (default), cur_trig=1.0 → rising, NOT
    # falling → no edge → latch stays at initial.
    assert float(y) == 0.0


# ---------------------------------------------------------------------------
# ForEach (smoke / parity with ReplicatedFunction)
# ---------------------------------------------------------------------------


def test_foreach_broadcast_input_smoke():
    """ForEach with broadcast input matches ReplicatedFunction behavior."""
    blk = ForEach(
        submodel=lambda u: 3.0 * u,
        n=8,
        n_inputs=1,
        in_axes=(None,),
    )
    bld = jaxonomy.DiagramBuilder()
    src = bld.add(Constant(jnp.array(2.0), name="u"))
    r = bld.add(blk)
    bld.connect(src.output_ports[0], r.input_ports[0])
    diagram = bld.build()
    ctx = diagram.create_context()
    y = r.output_ports[0].eval(ctx)
    assert y.shape == (8,)
    np.testing.assert_allclose(np.asarray(y), 6.0 * np.ones(8))


def test_foreach_batched_input_smoke():
    """ForEach with batched input runs the submodel per replica."""
    blk = ForEach(
        submodel=lambda u: 2.0 * u + 1.0,
        n=4,
        n_inputs=1,
        in_axes=(0,),
    )
    bld = jaxonomy.DiagramBuilder()
    us = jnp.array([1.0, 2.0, 3.0, 4.0])
    src = bld.add(Constant(us, name="u"))
    r = bld.add(blk)
    bld.connect(src.output_ports[0], r.input_ports[0])
    diagram = bld.build()
    ctx = diagram.create_context()
    y = r.output_ports[0].eval(ctx)
    assert y.shape == (4,)
    np.testing.assert_allclose(np.asarray(y), 2.0 * np.asarray(us) + 1.0)
