# SPDX-License-Identifier: MIT
"""T-100: forward-mode autodiff public API (`simulate_jacfwd`).

Pins the contract that ``jaxonomy.simulate_jacfwd`` returns a Jacobian
matching analytic and reverse-mode results. Forward-mode is essentially
free to expose because it's already used internally by ``linearize``,
the BDF Jacobian solve, and the Kalman/EKF blocks; the public function
just wires the existing plumbing through ``jax.jacfwd`` after
disabling the custom-VJP that would otherwise block forward-trace.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import jaxonomy
from jaxonomy import LeafSystem, SimulatorOptions, simulate_jacfwd
from jaxonomy.testing.markers import skip_if_not_jax

skip_if_not_jax()


class _ScalarDecay(LeafSystem):
    """dx/dt = -a*x; output = x. A 1-param, 1-state, 1-output baseline."""

    def __init__(self, a: float = 1.5):
        super().__init__()
        self.declare_dynamic_parameter("a", a)
        self.declare_continuous_state(default_value=jnp.array(1.0), ode=self._ode)
        self.declare_output_port(
            lambda t, s, **p: s.continuous_state, default_value=jnp.zeros(())
        )

    def _ode(self, time, state, **params):
        return -params["a"] * state.continuous_state


def _opts() -> SimulatorOptions:
    return SimulatorOptions(
        math_backend="jax",
        ode_solver_method="dopri5",
        rtol=1e-9,
        atol=1e-12,
        max_major_steps=200,
    )


def test_jacfwd_scalar_decay_matches_analytic():
    """∂x(T)/∂a = -T·x0·exp(-a·T) for dx/dt = -a*x, x(0) = x0."""
    sys = _ScalarDecay(a=1.5)
    T = 2.0
    x0 = 4.0

    def make_ctx(a):
        ctx = sys.create_context()
        ctx = ctx.with_continuous_state(jnp.array(x0))
        ctx.parameters["a"] = a
        return ctx

    a = jnp.array(1.5)
    J = simulate_jacfwd(sys, make_ctx, (0.0, T), a, options=_opts())
    expected = -T * x0 * jnp.exp(-a * T)
    np.testing.assert_allclose(J, expected, rtol=1e-5)


def test_jacfwd_matches_reverse_mode():
    """Forward and reverse mode should agree on the same problem."""
    sys = _ScalarDecay(a=1.5)
    T = 2.0
    x0 = 4.0

    def make_ctx(a):
        ctx = sys.create_context()
        ctx = ctx.with_continuous_state(jnp.array(x0))
        ctx.parameters["a"] = a
        return ctx

    def fwd(a):
        ctx = make_ctx(a)
        opts_grad = SimulatorOptions(
            math_backend="jax",
            enable_autodiff=True,
            ode_solver_method="dopri5",
            rtol=1e-9,
            atol=1e-12,
            max_major_steps=200,
        )
        return jaxonomy.simulate(sys, ctx, t_span=(0.0, T), options=opts_grad).context.continuous_state.sum()

    a = jnp.array(1.5)
    J_fwd = simulate_jacfwd(sys, make_ctx, (0.0, T), a, options=_opts())
    J_rev = jax.grad(fwd)(a)
    np.testing.assert_allclose(J_fwd, J_rev, rtol=1e-6)


class _MultiOutDecay(LeafSystem):
    """3-state independent decay with shared time horizon — exposes a
    diagonal Jacobian when each rate parameter affects only its slot."""

    def __init__(self):
        super().__init__()
        self.declare_dynamic_parameter("a", jnp.array([1.0, 0.5, 2.0]))
        self.declare_continuous_state(
            default_value=jnp.array([1.0, 1.0, 1.0]), ode=self._ode
        )
        self.declare_output_port(
            lambda t, s, **p: s.continuous_state, default_value=jnp.zeros(3)
        )

    def _ode(self, time, state, **params):
        return -params["a"] * state.continuous_state


def test_jacfwd_multi_output_diagonal_jacobian():
    """3 params × 3 outputs: ∂x_i(T)/∂a_j = -T·x0_i·exp(-a_i·T) δ_ij."""
    sys = _MultiOutDecay()
    T = 1.0
    x0 = jnp.array([1.0, 1.0, 1.0])

    def make_ctx(a):
        ctx = sys.create_context()
        ctx = ctx.with_continuous_state(x0)
        ctx.parameters["a"] = a
        return ctx

    a = jnp.array([1.0, 0.5, 2.0])
    J = simulate_jacfwd(sys, make_ctx, (0.0, T), a, options=_opts())
    # Expected: diagonal matrix with entries -T·x0_i·exp(-a_i·T).
    expected = jnp.diag(-T * x0 * jnp.exp(-a * T))
    np.testing.assert_allclose(J, expected, rtol=1e-5, atol=1e-7)


def test_jacfwd_with_jit():
    """Wrapping simulate_jacfwd in jax.jit should not retrace per call."""
    sys = _ScalarDecay()
    T = 1.0

    def make_ctx(a):
        ctx = sys.create_context()
        ctx = ctx.with_continuous_state(jnp.array(2.0))
        ctx.parameters["a"] = a
        return ctx

    @jax.jit
    def jac(a):
        return simulate_jacfwd(sys, make_ctx, (0.0, T), a, options=_opts())

    J1 = jac(jnp.array(0.5))
    J2 = jac(jnp.array(0.7))
    # Both should be finite and approximately equal to -T·x0·exp(-a·T).
    assert jnp.isfinite(J1) and jnp.isfinite(J2)
    np.testing.assert_allclose(J1, -T * 2.0 * jnp.exp(-0.5 * T), rtol=1e-5)
    np.testing.assert_allclose(J2, -T * 2.0 * jnp.exp(-0.7 * T), rtol=1e-5)


def test_jacfwd_default_output_fn_returns_continuous_state():
    """When output_fn is None, simulate_jacfwd returns ∂(continuous_state)/∂params."""
    sys = _ScalarDecay()
    T = 1.5

    def make_ctx(a):
        ctx = sys.create_context()
        ctx = ctx.with_continuous_state(jnp.array(3.0))
        ctx.parameters["a"] = a
        return ctx

    a = jnp.array(0.8)
    J_default = simulate_jacfwd(sys, make_ctx, (0.0, T), a, options=_opts())
    J_explicit = simulate_jacfwd(
        sys, make_ctx, (0.0, T), a,
        output_fn=lambda ctx: ctx.continuous_state,
        options=_opts(),
    )
    np.testing.assert_allclose(J_default, J_explicit, rtol=1e-12)
