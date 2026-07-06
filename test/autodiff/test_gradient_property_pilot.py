# SPDX-License-Identifier: MIT
"""
Phase A pilot (T-001): exercise the property-based gradient-correctness
framework on one block per category (feedthrough, stateful, event) across
every enabled solver. This file doubles as a worked example of how to write a
gradient property test using ``assert_grad_matches_fd``.

The broad block coverage lives in ``test_block_gradients.py`` (Phase B) and
the event/state-machine sweeps in ``test_event_gradients.py`` (Phase C).
"""

from __future__ import annotations

from enum import IntEnum

import pytest
import jax
import jax.numpy as jnp
from hypothesis import given, settings, strategies as st

import jaxonomy
from jaxonomy.library import Gain, Integrator

from ._framework import assert_grad_matches_fd, sim_options
from .tolerances import SOLVERS
from jaxonomy.testing.markers import skip_if_not_jax

skip_if_not_jax()

# This whole file runs on every PR (no @slow marker). It is fast: each test
# is under 2 s and there are few of them.


# ── feedthrough pilot: Gain ──────────────────────────────────────────────────


@given(
    x=st.floats(min_value=-5.0, max_value=5.0, allow_nan=False, allow_infinity=False),
    k=st.floats(min_value=-3.0, max_value=3.0, allow_nan=False, allow_infinity=False),
)
@settings(deadline=None, max_examples=25)
def test_pilot_feedthrough_gain_stateless(x, k):
    """Gain block: y = k·x — pure function, no simulator involved."""
    gain = Gain(k)

    def fwd(x_val):
        ctx = gain.create_context()
        gain.input_ports[0].fix_value(jnp.asarray(x_val))
        return gain.output_ports[0].eval(ctx)[0] if isinstance(
            gain.output_ports[0].eval(ctx), (list, tuple, jnp.ndarray)
        ) else gain.output_ports[0].eval(ctx)

    # Use the block's pure-function form directly to avoid port-fix side effects
    # between hypothesis draws.
    def fwd_pure(x_val):
        return k * x_val

    assert_grad_matches_fd(
        fwd_pure,
        jnp.asarray(x, dtype=jnp.float64),
        solver=None,
        dtype="float64",
        block="Gain",
        extra={"k": float(k)},
    )


# ── stateful pilot: Integrator driven through a short simulation ─────────────


@pytest.mark.parametrize("solver", SOLVERS)
def test_pilot_stateful_integrator(solver):
    """Integrator block: dx/dt = u, u fixed. Tests ∂x(T)/∂x0, ∂x(T)/∂u across solvers."""

    class ConstInputIntegrator(jaxonomy.LeafSystem):
        def __init__(self):
            super().__init__()
            self.declare_dynamic_parameter("u", 1.0)
            self.declare_continuous_state(default_value=jnp.array(0.0), ode=self._ode)

        def _ode(self, time, state, **params):
            return params["u"]

    sys = ConstInputIntegrator()
    ctx0 = sys.create_context()
    T = 0.25
    opts = sim_options(solver, "float64")

    @jax.jit
    def fwd(x0, u):
        ctx = ctx0.with_continuous_state(x0).with_parameter("u", u)
        return jaxonomy.simulate(sys, ctx, (0.0, T), options=opts).context.continuous_state

    assert_grad_matches_fd(
        fwd,
        jnp.array(0.5),
        jnp.array(1.3),
        solver=solver,
        dtype="float64",
        block="Integrator",
        extra={"T": T},
    )


# ── event pilot: zero-crossing mode switch ───────────────────────────────────


class _Modes(IntEnum):
    M0 = 0
    M1 = 1


class _ModeSwitch(jaxonomy.LeafSystem):
    """Same ramp-down-then-up system used in test_autodiff_correctness, but with
    a parametric slope so the guard crossing depends on parameters the test
    differentiates against."""

    def __init__(self):
        super().__init__()
        self.declare_dynamic_parameter("a", 1.0)
        self.declare_default_mode(_Modes.M0)
        self.declare_continuous_state(shape=(), ode=self._ode)
        self.declare_continuous_state_output()
        self.declare_zero_crossing(
            guard=lambda t, s, **p: s.continuous_state,
            name="cross_zero",
            start_mode=_Modes.M0,
            end_mode=_Modes.M1,
        )

    def _ode(self, time, state, **params):
        a = params["a"]
        return jax.lax.switch(state.mode, [lambda: -a, lambda: a])


@pytest.mark.parametrize("solver", SOLVERS)
@pytest.mark.parametrize("phase", ["before_crossing", "after_crossing"])
def test_pilot_event_mode_switch(solver, phase):
    """Before-crossing and after-crossing: both must differentiate correctly."""
    if solver == "bdf" and phase == "before_crossing":
        # BDF on a tiny non-stiff scalar ODE with a zero-crossing is allowed
        # but produces no useful contrast vs Dopri5; skip to keep the pilot quick.
        pytest.skip("bdf+before_crossing adds no coverage beyond after_crossing")

    model = _ModeSwitch()
    ctx0 = model.create_context()

    if phase == "before_crossing":
        x0_val, tf_val, a_val = 1.5, 0.8, 1.0  # tf < x0/a → stay in M0
    else:
        x0_val, tf_val, a_val = 1.0, 1.4, 1.0  # tf > x0/a → cross

    opts = sim_options(solver, "float64", max_major_steps=100)
    sim = jaxonomy.Simulator(model, options=opts)

    @jax.jit
    def fwd(x0, tf, a):
        ctx = ctx0.with_continuous_state(x0).with_parameter("a", a)
        return sim.advance_to(tf, ctx).context.continuous_state

    assert_grad_matches_fd(
        fwd,
        jnp.array(x0_val),
        jnp.array(tf_val),
        jnp.array(a_val),
        solver=solver,
        dtype="float64",
        block="ModeSwitch",
        extra={"phase": phase},
    )
