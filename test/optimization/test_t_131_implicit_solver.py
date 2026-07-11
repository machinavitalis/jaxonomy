# SPDX-License-Identifier: MIT

"""T-131: reverse-mode differentiation through iterative solvers in
dynamics via :func:`jaxonomy.optimization.implicit_solver`.

``jax.grad`` cannot reverse-differentiate ``lax.while_loop``; an RHS that
runs a Newton/fixed-point/constraint solve therefore forecloses
reverse-mode through ``simulate``. The IFT wrapper cuts the loop from the
tape and solves the adjoint system from the residual instead.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import jaxonomy
from jaxonomy.optimization import implicit_solver
from jaxonomy.simulation import SimulatorOptions
from jaxonomy.testing.markers import skip_if_not_jax

skip_if_not_jax()

pytestmark = pytest.mark.minimal


# ── unit level: the wrapper itself ───────────────────────────────────


def _newton_sqrt(a):
    """x* = sqrt(a) via a while_loop Newton iteration (not reverse-AD-able)."""

    def cond(carry):
        x, i = carry
        return (jnp.abs(x * x - a) > 1e-14) & (i < 100)

    def body(carry):
        x, i = carry
        return 0.5 * (x + a / x), i + 1

    x, _ = jax.lax.while_loop(cond, body, (jnp.maximum(a, 1.0), 0))
    return x


def _sqrt_residual(x, a):
    return x * x - a


def test_raw_while_loop_solver_is_not_reverse_differentiable():
    """Pin the failure mode the wrapper exists for."""
    with pytest.raises(ValueError, match="while_loop|Reverse-mode"):
        jax.grad(_newton_sqrt)(jnp.float64(2.0))


def test_wrapped_scalar_solver_matches_analytic_gradient():
    diff_sqrt = implicit_solver(_newton_sqrt, _sqrt_residual)
    a = jnp.float64(2.0)
    assert float(diff_sqrt(a)) == pytest.approx(np.sqrt(2.0), rel=1e-12)
    grad = jax.grad(diff_sqrt)(a)
    assert float(grad) == pytest.approx(0.5 / np.sqrt(2.0), rel=1e-8)


def test_wrapped_solver_jits_and_vmaps():
    diff_sqrt = implicit_solver(_newton_sqrt, _sqrt_residual)
    grads = jax.jit(jax.vmap(jax.grad(diff_sqrt)))(jnp.array([1.0, 4.0, 9.0]))
    np.testing.assert_allclose(
        np.asarray(grads), 0.5 / np.sqrt([1.0, 4.0, 9.0]), rtol=1e-8
    )


def test_wrapped_vector_solver_pytree_theta():
    """2-D root with a dict-valued theta: solve x = A(theta) @ tanh(x) + b."""

    def residual(x, theta):
        return x - theta["gain"] * jnp.tanh(x) - theta["bias"]

    def solver(theta):
        def cond(carry):
            x, i = carry
            return (jnp.max(jnp.abs(residual(x, theta))) > 1e-13) & (i < 200)

        def body(carry):
            x, i = carry
            return theta["gain"] * jnp.tanh(x) + theta["bias"], i + 1

        x, _ = jax.lax.while_loop(
            cond, body, (jnp.zeros_like(theta["bias"]), 0)
        )
        return x

    diff_solve = implicit_solver(solver, residual)
    theta = {"gain": jnp.float64(0.3), "bias": jnp.array([0.5, -0.2])}

    def loss(theta):
        return jnp.sum(diff_solve(theta) ** 2)

    grads = jax.grad(loss)(theta)

    # FD check on both pytree leaves.
    eps = 1e-6

    def loss_at(gain, bias):
        return float(loss({"gain": jnp.float64(gain), "bias": bias}))

    fd_gain = (
        loss_at(0.3 + eps, theta["bias"]) - loss_at(0.3 - eps, theta["bias"])
    ) / (2 * eps)
    assert float(grads["gain"]) == pytest.approx(fd_gain, rel=1e-5)
    e0 = jnp.array([eps, 0.0])
    fd_b0 = (
        loss_at(0.3, theta["bias"] + e0) - loss_at(0.3, theta["bias"] - e0)
    ) / (2 * eps)
    assert float(grads["bias"][0]) == pytest.approx(fd_b0, rel=1e-5)


# ── end to end: the solver inside a simulated block's RHS ────────────


class ImplicitDamper(jaxonomy.LeafSystem):
    """dx/dt = -v(x, c) where v solves v + c*tanh(v) = x (Newton loop).

    The implicit velocity law stands in for an iterative constraint /
    friction solve inside a dynamics callback — the T-131 pattern.
    """

    def __init__(self, wrap: bool, name=None):
        super().__init__(name=name)
        self.declare_dynamic_parameter("c", jnp.float64(0.5))
        self.declare_continuous_state(default_value=jnp.float64(1.0), ode=self.ode)
        self.declare_continuous_state_output(name="x")
        self._wrap = wrap

    @staticmethod
    def _residual(v, theta):
        x, c = theta
        return v + c * jnp.tanh(v) - x

    @staticmethod
    def _solve_v(theta):
        x, c = theta

        def cond(carry):
            v, i = carry
            return (jnp.abs(v + c * jnp.tanh(v) - x) > 1e-13) & (i < 100)

        def body(carry):
            v, i = carry
            g = v + c * jnp.tanh(v) - x
            dg = 1.0 + c / jnp.cosh(v) ** 2
            return v - g / dg, i + 1

        v, _ = jax.lax.while_loop(cond, body, (x, 0))
        return v

    def ode(self, t, state, **params):
        x = state.continuous_state
        theta = (x, params["c"])
        if self._wrap:
            v = implicit_solver(self._solve_v, self._residual)(theta)
        else:
            v = self._solve_v(theta)
        return -v


def _terminal_x(wrap: bool):
    model = ImplicitDamper(wrap=wrap)
    opts = SimulatorOptions(
        math_backend="jax",
        ode_solver_method="rk4",
        max_minor_step_size=0.01,
        enable_autodiff=True,
    )
    base_ctx = model.create_context()

    def fwd(c, context):
        context = context.with_parameter("c", c)
        res = jaxonomy.simulate(model, context, (0.0, 1.0), options=opts)
        return res.context.continuous_state

    return fwd, base_ctx


def test_grad_through_simulate_with_wrapped_solver_matches_fd():
    """The headline T-131 result: reverse-mode through simulate works
    when the RHS's iterative solve is wrapped, and matches FD."""
    fwd, ctx = _terminal_x(wrap=True)
    value, grad = jax.jit(jax.value_and_grad(fwd))(jnp.float64(0.5), ctx)
    assert np.isfinite(float(value))

    f = jax.jit(fwd)
    eps = 1e-5
    fd = (
        float(f(jnp.float64(0.5 + eps), ctx))
        - float(f(jnp.float64(0.5 - eps), ctx))
    ) / (2 * eps)
    assert float(grad) == pytest.approx(fd, rel=1e-4), (
        f"adjoint {float(grad):.8f} vs FD {fd:.8f}"
    )


def test_grad_through_simulate_with_raw_solver_fails():
    """Without the wrapper the same model forecloses reverse-mode —
    pins that the wrapper is load-bearing, not incidental."""
    fwd, ctx = _terminal_x(wrap=False)
    with pytest.raises(ValueError, match="while_loop|Reverse-mode"):
        jax.value_and_grad(fwd)(jnp.float64(0.5), ctx)
