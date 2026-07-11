# SPDX-License-Identifier: MIT

"""T-129: differentiating through a BDF-DAE simulate must be jit-hoistable.

The reported ~30 s per ``value_and_grad`` call on an acausal pack was
re-trace cost, not solver runtime: every bare ``jaxonomy.simulate`` call
builds a fresh traced closure, so JAX's jit cache misses per call. The
supported fix (documented in ``docs/jit_cache.md``) is to wrap the outer
``value_and_grad`` in ``jax.jit`` with the context passed as an argument.
These tests pin that the pattern (a) works on the implicit BDF/DAE path,
(b) produces gradients matching finite differences, and (c) actually
caches — repeat calls are much faster than the first.
"""

from __future__ import annotations

import time

import numpy as np
import jax
import jax.numpy as jnp
import pytest

import jaxonomy
from jaxonomy.testing.markers import requires_jax, skip_if_not_jax

skip_if_not_jax()


class PlanarPendulum(jaxonomy.LeafSystem):
    """Index-2 pendulum DAE (same fixture as the T-113 tests) with the
    gravity constant declared as a differentiable dynamic parameter."""

    def __init__(self, L=1.0, g0=9.8, name=None):
        super().__init__(name=name)
        x0 = np.array(
            [0.0, 0.8660254037844386, 0.0, -4.9, -0.5,
             -4.243524478543744, -7.35, -7.35, 0.0]
        )
        self.declare_dynamic_parameter("L", L)
        self.declare_dynamic_parameter("g0", g0)
        M = np.concatenate([np.ones(2), np.zeros(7)])
        self.declare_continuous_state(
            default_value=x0, mass_matrix=M, ode=self.ode
        )
        self.declare_continuous_state_output(name="x")

    def ode(self, time, state, **parameters):
        L, g0 = parameters["L"], parameters["g0"]
        x = state.continuous_state[:2]
        z = state.continuous_state[2:]
        f = jnp.array([z[3], x[0]])
        g = jnp.array([
            -(L**2) + x[1] ** 2 + z[2] ** 2,
            2 * z[0] * z[2] + 2 * x[1] * x[0],
            z[0] - z[6],
            2 * z[3] * x[1] + 2 * z[4] * z[2] + 2 * z[0] ** 2 + 2 * x[0] ** 2,
            z[4] - z[5],
            z[5] + g0 - z[1] * z[2],
            -z[1] * x[1] + z[3],
        ])
        return jnp.concatenate([f, g])


T_END = 2.0

_OPTS = jaxonomy.SimulatorOptions(
    math_backend="jax",
    ode_solver_method="bdf",
    rtol=1e-6,
    atol=1e-8,
    enable_autodiff=True,
)


def _fwd(model):
    def fwd(g0, context):
        context = context.with_parameter("g0", g0)
        res = jaxonomy.simulate(model, context, (0.0, T_END), options=_OPTS)
        return res.context.continuous_state[1]

    return fwd


@requires_jax()
def test_jitted_value_and_grad_matches_fd():
    model = PlanarPendulum()
    ctx = model.create_context()
    fwd = _fwd(model)

    vg = jax.jit(jax.value_and_grad(fwd))
    fwd_jit = jax.jit(fwd)

    g0 = jnp.float64(9.9)
    value, grad = vg(g0, ctx)

    h = 1e-4
    fd = (
        float(fwd_jit(jnp.float64(9.9 + h), ctx))
        - float(fwd_jit(jnp.float64(9.9 - h), ctx))
    ) / (2 * h)
    assert np.isfinite(float(value))
    assert float(grad) == pytest.approx(fd, rel=1e-3), (
        f"BDF-DAE adjoint {float(grad):+.6f} vs FD {fd:+.6f}"
    )


@requires_jax()
def test_jitted_value_and_grad_caches_across_calls():
    """The whole point of T-129: after the first (compiling) call, repeat
    calls with new parameter values must be dramatically cheaper. The
    measured ratio is ~180×; the 5× threshold leaves generous headroom
    for CI noise while still catching a per-call retrace regression."""
    model = PlanarPendulum()
    ctx = model.create_context()
    vg = jax.jit(jax.value_and_grad(_fwd(model)))

    def timed(g):
        t0 = time.perf_counter()
        value, grad = vg(jnp.float64(g), ctx)
        jax.block_until_ready((value, grad))
        return time.perf_counter() - t0, float(value), float(grad)

    t_first, _, _ = timed(9.8)
    t_second, v2, g2 = timed(9.9)
    t_third, v3, g3 = timed(10.0)

    assert np.isfinite(v2) and np.isfinite(g2)
    assert np.isfinite(v3) and np.isfinite(g3)
    t_repeat = min(t_second, t_third)
    assert t_repeat * 5 < t_first, (
        f"repeat value_and_grad call not cached: first={t_first:.3f}s, "
        f"repeat={t_repeat:.3f}s — the jit-hoist pattern has regressed "
        "to per-call retracing"
    )
