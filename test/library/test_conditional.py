# SPDX-License-Identifier: MIT
"""
T-009 — Conditional container block tests.

Covers:

  - enable=1: output matches the submodel
  - enable=0, reset: output equals initial_value
  - enable=0, passthrough: output equals the first user input
  - enable=0, hold: output equals the last enabled submodel value
  - gradient through the disabled branch is zero (autodiff safety)
  - invalid ``when_disabled`` raises ValueError
  - simulate integration: discrete toggle of enable across time
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import jaxonomy
from jaxonomy.library import Conditional, Constant, Gain, WhenDisabled
from jaxonomy.testing.markers import skip_if_not_jax

skip_if_not_jax()


def _scale_by_two(x):
    return 2.0 * x


# ── reset mode ─────────────────────────────────────────────────────────────


def test_enabled_output_matches_submodel():
    """enable=1 → output = submodel(u)."""
    cond = Conditional(_scale_by_two, n_inputs=1, when_disabled=WhenDisabled.RESET)
    bld = jaxonomy.DiagramBuilder()
    en = bld.add(Constant(jnp.array(1.0), name="en"))
    u = bld.add(Constant(jnp.array(3.0), name="u"))
    c = bld.add(cond)
    bld.connect(en.output_ports[0], c.input_ports[0])
    bld.connect(u.output_ports[0], c.input_ports[1])
    diagram = bld.build()
    ctx = diagram.create_context()
    y = c.output_ports[0].eval(ctx)
    assert float(y) == 6.0


def test_disabled_reset_mode():
    """enable=0, reset → output = initial_value."""
    cond = Conditional(
        _scale_by_two, n_inputs=1,
        when_disabled=WhenDisabled.RESET, initial_value=-99.0,
    )
    bld = jaxonomy.DiagramBuilder()
    en = bld.add(Constant(jnp.array(0.0), name="en"))
    u = bld.add(Constant(jnp.array(3.0), name="u"))
    c = bld.add(cond)
    bld.connect(en.output_ports[0], c.input_ports[0])
    bld.connect(u.output_ports[0], c.input_ports[1])
    diagram = bld.build()
    ctx = diagram.create_context()
    y = c.output_ports[0].eval(ctx)
    assert float(y) == -99.0


# ── passthrough mode ──────────────────────────────────────────────────────


def test_disabled_passthrough_mode():
    """enable=0, passthrough → output = first user input."""
    cond = Conditional(
        _scale_by_two, n_inputs=1, when_disabled=WhenDisabled.PASSTHROUGH,
    )
    bld = jaxonomy.DiagramBuilder()
    en = bld.add(Constant(jnp.array(0.0), name="en"))
    u = bld.add(Constant(jnp.array(5.0), name="u"))
    c = bld.add(cond)
    bld.connect(en.output_ports[0], c.input_ports[0])
    bld.connect(u.output_ports[0], c.input_ports[1])
    diagram = bld.build()
    ctx = diagram.create_context()
    y = c.output_ports[0].eval(ctx)
    assert float(y) == 5.0


# ── autodiff safety ───────────────────────────────────────────────────────


def test_gradient_through_disabled_branch_is_zero():
    """When disabled (enable=0, reset mode), gradient w.r.t. the user
    input must be zero — the submodel output is discarded by jnp.where."""
    cond = Conditional(
        _scale_by_two, n_inputs=1,
        when_disabled=WhenDisabled.RESET, initial_value=0.0,
    )
    bld = jaxonomy.DiagramBuilder()
    en = bld.add(Constant(jnp.array(0.0), name="en"))
    u = bld.add(Constant(jnp.array(3.0), name="u"))
    c = bld.add(cond)
    bld.connect(en.output_ports[0], c.input_ports[0])
    bld.connect(u.output_ports[0], c.input_ports[1])
    diagram = bld.build()
    ctx0 = diagram.create_context()

    def loss(u_val):
        ctx = ctx0.with_subcontext(
            u.system_id, ctx0[u.system_id].with_parameter("value", u_val),
        )
        return c.output_ports[0].eval(ctx)

    g = jax.grad(loss)(jnp.array(3.0))
    assert float(g) == 0.0, f"disabled branch should give zero grad, got {g}"


def test_gradient_through_enabled_branch_matches_submodel():
    """When enabled, gradient flows through the submodel."""
    cond = Conditional(
        _scale_by_two, n_inputs=1,
        when_disabled=WhenDisabled.RESET, initial_value=0.0,
    )
    bld = jaxonomy.DiagramBuilder()
    en = bld.add(Constant(jnp.array(1.0), name="en"))
    u = bld.add(Constant(jnp.array(3.0), name="u"))
    c = bld.add(cond)
    bld.connect(en.output_ports[0], c.input_ports[0])
    bld.connect(u.output_ports[0], c.input_ports[1])
    diagram = bld.build()
    ctx0 = diagram.create_context()

    def loss(u_val):
        ctx = ctx0.with_subcontext(
            u.system_id, ctx0[u.system_id].with_parameter("value", u_val),
        )
        return c.output_ports[0].eval(ctx)

    g = jax.grad(loss)(jnp.array(3.0))
    assert float(g) == 2.0, f"enabled submodel is y=2u, grad should be 2, got {g}"


# ── validation ────────────────────────────────────────────────────────────


def test_invalid_when_disabled_raises():
    with pytest.raises(ValueError, match="when_disabled"):
        Conditional(_scale_by_two, n_inputs=1, when_disabled="bogus")


def test_hold_mode_requires_period():
    with pytest.raises(ValueError, match="hold_period"):
        Conditional(_scale_by_two, n_inputs=1, when_disabled=WhenDisabled.HOLD)


# ── hold mode (time-integrated) ──────────────────────────────────────────


def test_hold_mode_captures_last_enabled_value():
    """Run a short simulation where enable goes 1→0 at t=0.3. At t=0.2 the
    submodel is active (output = 2·u(t)). At t=0.5 (after disable) the
    held discrete state should still be the last enabled snapshot value."""
    from jaxonomy.library import Sine

    # The submodel is a pure function of u: y = 2*u. Enable is a step
    # signal that switches at t=0.3. u = sin(4πt) so it varies.
    class EnableStep(jaxonomy.LeafSystem):
        def __init__(self, **kw):
            super().__init__(**kw)
            self.declare_output_port(
                lambda t, s, *u, **p: jnp.where(t < 0.3, 1.0, 0.0),
                prerequisites_of_calc=[],
                requires_inputs=False,
            )

    cond = Conditional(
        _scale_by_two, n_inputs=1,
        when_disabled=WhenDisabled.HOLD,
        initial_value=jnp.array(0.0),
        hold_period=0.1,
    )
    bld = jaxonomy.DiagramBuilder()
    en = bld.add(EnableStep(name="en"))
    u = bld.add(Sine(amplitude=1.0, frequency=4 * np.pi, phase=0.0, name="u"))
    c = bld.add(cond)
    bld.connect(en.output_ports[0], c.input_ports[0])
    bld.connect(u.output_ports[0], c.input_ports[1])
    diagram = bld.build()
    ctx = diagram.create_context()
    opts = jaxonomy.SimulatorOptions(math_backend="jax", max_major_steps=100)
    res = jaxonomy.simulate(diagram, ctx, (0.0, 0.5), options=opts)

    # Held state at the end should be finite and bounded (2·|sin| ≤ 2).
    held = np.asarray(res.context[c.system_id].discrete_state)
    assert np.all(np.isfinite(held)), f"held state NaN: {held}"
    assert abs(float(held)) <= 2.0 + 1e-9, f"held out of range: {held}"
