# SPDX-License-Identifier: MIT
"""T-125 phase 1 — public event-time gradient API.

Verify that :func:`jaxonomy.event_time_gradient` correctly computes
``∂t_event/∂params`` via the implicit-function theorem applied at the
recorded event boundary.

Worked example: 1-D bouncing ball.

    State:    (y, v)
    Dynamics: dy/dt = v, dv/dt = -g
    Initial:  y(0) = h0, v(0) = 0
    Guard:    g(t, x, p) = y      (fires when ball hits floor)

Closed-form trajectory:  y(t) = h0 - (1/2) g t²
First crossing:           t_e = sqrt(2 h0 / g)
Analytic sensitivities:
    ∂t_e/∂h0 =  1 / sqrt(2 g h0)
    ∂t_e/∂g  = -sqrt(h0 / (2 g)) / g
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import numpy as np

from jaxonomy import event_time_gradient, event_time_jacobian
from jaxonomy.testing.markers import skip_if_not_jax

skip_if_not_jax()


# ── Bouncing-ball helpers (closed-form trajectory) ───────────────────────


def _t_bounce_analytic(h0: float, g: float) -> float:
    return math.sqrt(2.0 * h0 / g)


# IMPORTANT: ``state_fn`` must return the *trajectory* state evaluated at
# a fixed numerical time ``t_e`` (NOT the simplified ``y = 0`` form).
# Otherwise the JAX derivative through ``state_fn`` is identically zero
# and the implicit-function chain rule fails.
#
# Concretely: y(t; h0, g) = h0 - g t² / 2 evaluates to zero numerically
# at t = sqrt(2 h0 / g), but ∂y(t; h0)/∂h0 |_{t fixed} = 1 ≠ 0.


def _make_state_fn_dh0(t_e_fixed: float, g_const: float):
    """Return state_fn(h0) -> trajectory state at fixed time t_e_fixed."""

    def _state_fn(h0):
        y = h0 - 0.5 * g_const * (t_e_fixed ** 2)
        v = -g_const * t_e_fixed
        return jnp.stack([y, jnp.asarray(v)])

    return _state_fn


def _make_state_fn_dg(t_e_fixed: float, h0_const: float):
    """Return state_fn(g) -> trajectory state at fixed time t_e_fixed."""

    def _state_fn(g):
        y = h0_const - 0.5 * g * (t_e_fixed ** 2)
        v = -g * t_e_fixed
        return jnp.stack([y, v])

    return _state_fn


# ── Tests: bouncing-ball event-time gradient ─────────────────────────────


def test_event_time_gradient_bouncing_ball_dh0():
    """``∂t_bounce/∂h0`` matches the analytic ``1 / sqrt(2 g h0)``."""
    h0 = 1.0
    g = 9.81
    t_e = jnp.asarray(_t_bounce_analytic(h0, g))

    def guard_fn(t, x, p):
        # Guard does not depend on the differentiation parameter.
        return x[0]

    def rhs_fn(t, x, p):
        # rhs uses the constant g — independent of the differentiation
        # parameter h0.
        return jnp.stack([x[1], jnp.asarray(-g)])

    grad_h0 = event_time_gradient(
        guard_fn,
        rhs_fn,
        t_e,
        _make_state_fn_dh0(float(t_e), g),
        jnp.asarray(h0),
    )

    expected = 1.0 / math.sqrt(2.0 * g * h0)
    np.testing.assert_allclose(np.asarray(grad_h0), expected, rtol=1e-6)


def test_event_time_gradient_bouncing_ball_dg():
    """``∂t_bounce/∂g`` matches the analytic ``-sqrt(h0 / (2 g)) / g``."""
    h0 = 1.0
    g = 9.81
    t_e = jnp.asarray(_t_bounce_analytic(h0, g))

    def guard_fn(t, x, p):
        return x[0]

    def rhs_fn(t, x, p):
        # rhs uses the differentiation parameter g.
        return jnp.stack([x[1], -p])

    grad_g = event_time_gradient(
        guard_fn,
        rhs_fn,
        t_e,
        _make_state_fn_dg(float(t_e), h0),
        jnp.asarray(g),
    )

    expected = -math.sqrt(h0 / (2.0 * g)) / g
    np.testing.assert_allclose(np.asarray(grad_g), expected, rtol=1e-6)


# ── Test: gradient flows through downstream cost ─────────────────────────


def test_event_time_gradient_under_jax_grad():
    """Compose with ``jax.grad``: gradient flows through the helper.

    Build a cost ``c(h0) = (dt_e/dh0)² + h0²``.  The first term has a
    closed-form derivative through the helper (the recorded ``t_e`` is
    fixed and the rhs has no h0 dependence, so ``dt_e/dh0`` is locally
    constant w.r.t. ``h0`` and contributes zero); the second term has
    derivative ``2 h0``.  Verifies that JAX traces through the helper
    without erroring and produces finite output.
    """
    g_val = 9.81
    h0_nominal = 1.0
    t_e_nominal = math.sqrt(2.0 * h0_nominal / g_val)

    def cost(h0):
        h0 = jnp.asarray(h0)
        t_e = jnp.asarray(t_e_nominal)

        def state_fn(h):
            y = h - 0.5 * g_val * (t_e_nominal ** 2)
            v = -g_val * t_e_nominal
            return jnp.stack([y, jnp.asarray(v)])

        def guard_fn(t, x, p):
            return x[0]

        def rhs_fn(t, x, p):
            return jnp.stack([x[1], jnp.asarray(-g_val)])

        dt_dh0 = event_time_gradient(guard_fn, rhs_fn, t_e, state_fn, h0)
        return dt_dh0 * dt_dh0 + h0 * h0

    h0 = 1.0
    g_ad = float(jax.grad(cost)(h0))
    eps = 1e-5
    fd = (float(cost(h0 + eps)) - float(cost(h0 - eps))) / (2 * eps)
    assert math.isfinite(g_ad)
    np.testing.assert_allclose(g_ad, fd, rtol=1e-3, atol=1e-5)


# ── Test: grazing crossing — numerical safety ────────────────────────────


def test_event_time_gradient_grazing_returns_finite():
    """Tangential crossing (``∂g/∂x · ẋ → 0``): helper must stay finite."""
    t_e = jnp.asarray(1.0)

    def guard_fn(t, x, p):
        return x[0]

    def rhs_fn(t, x, p):
        return jnp.stack([x[1], -p])

    # Both y and v zero → ẏ = 0 → denom = 0.
    state_const = jnp.array([0.0, 0.0])

    grad_p = event_time_gradient(
        guard_fn, rhs_fn, t_e, state_const, jnp.asarray(9.81),
    )
    val = float(grad_p)
    assert math.isfinite(val), f"Expected finite value at grazing, got {val}"


# ── Test: vector wrapper ─────────────────────────────────────────────────


def test_event_time_jacobian_array_output():
    """``event_time_jacobian`` returns an ndarray for a vector ``params``."""
    g_val = 9.81
    h0_val = 1.0
    t_e = jnp.asarray(_t_bounce_analytic(h0_val, g_val))

    t_e_static = float(t_e)

    def state_fn(p):
        # p is shape (1,) — only h0.
        h0 = p[0]
        y = h0 - 0.5 * g_val * (t_e_static ** 2)
        v = -g_val * t_e_static
        return jnp.stack([y, jnp.asarray(v)])

    def guard_fn(t, x, p):
        return x[0]

    def rhs_fn(t, x, p):
        return jnp.stack([x[1], jnp.asarray(-g_val)])

    params = jnp.array([h0_val])
    jac = event_time_jacobian(guard_fn, rhs_fn, t_e, state_fn, params)
    expected = 1.0 / math.sqrt(2.0 * g_val * h0_val)
    np.testing.assert_allclose(np.asarray(jac).ravel(), [expected], rtol=1e-6)


# ── Test: dict params (PyTree) ───────────────────────────────────────────


def test_event_time_gradient_dict_params():
    """``params`` as a dict — PyTree handling preserved on output."""
    g_val = 9.81
    h0_val = 1.0
    t_e = jnp.asarray(_t_bounce_analytic(h0_val, g_val))

    t_e_static = float(t_e)

    def state_fn(p):
        h0 = p["h0"]
        y = h0 - 0.5 * g_val * (t_e_static ** 2)
        v = -g_val * t_e_static
        return jnp.stack([y, jnp.asarray(v)])

    def guard_fn(t, x, p):
        return x[0]

    def rhs_fn(t, x, p):
        return jnp.stack([x[1], jnp.asarray(-g_val)])

    params = {"h0": jnp.asarray(h0_val)}
    grads = event_time_gradient(guard_fn, rhs_fn, t_e, state_fn, params)
    assert isinstance(grads, dict) and "h0" in grads
    expected = 1.0 / math.sqrt(2.0 * g_val * h0_val)
    np.testing.assert_allclose(np.asarray(grads["h0"]), expected, rtol=1e-6)


# ── Test: public-API surface ─────────────────────────────────────────────


def test_public_api_exported():
    """``event_time_gradient`` is exposed at top level and via .simulation."""
    import jaxonomy
    import jaxonomy.simulation

    assert hasattr(jaxonomy, "event_time_gradient")
    assert hasattr(jaxonomy, "event_time_jacobian")
    assert hasattr(jaxonomy.simulation, "event_time_gradient")
    assert hasattr(jaxonomy.simulation, "event_time_jacobian")
