# SPDX-License-Identifier: MIT
"""Gradient-correctness and convergence-reporting tests for
``project_constraints`` (consumer-reported: projection gradients,
non-convergence signal, iteration budget).

System under test: a two-state semi-explicit DAE with one algebraic
unknown, small enough that finite differences are exact to ~1e-8:

    dx/dt = -x + y          (differential row)
        0 = y**3 + y - x    (algebraic row; unique real root for any x)
"""

import warnings

import numpy as np
import pytest

import jax
import jax.numpy as jnp

import jaxonomy as jx
from jaxonomy.framework import LeafSystem
from jaxonomy.simulation.dae_projection import project_constraints


class CubicDAE(LeafSystem):
    """M diag = [1, 0]: x' = -x + y; 0 = y^3 + y - x."""

    def __init__(self, x0=1.0, y0=0.0, **kwargs):
        super().__init__(**kwargs)
        self.declare_continuous_state(
            default_value=jnp.array([float(x0), float(y0)]),
            ode=self._ode,
            mass_matrix=jnp.array([1.0, 0.0]),
        )

    def _ode(self, time, state, *inputs, **params):
        x, y = state.continuous_state
        return jnp.array([-x + y, y**3 + y - x])


def _build(x0=1.0, y0=0.0):
    builder = jx.DiagramBuilder()
    plant = builder.add(CubicDAE(x0=x0, y0=y0))
    diagram = builder.build()
    return diagram, plant


def _consistent_y(x):
    # real root of y^3 + y = x (Cardano; unique because y^3+y is monotone)
    roots = np.roots([1.0, 0.0, 1.0, -float(x)])
    real = roots[np.isreal(roots)].real
    return float(real[0])


def test_projection_reaches_manifold_from_bad_guess():
    diagram, plant = _build(x0=2.0, y0=37.0)  # wildly inconsistent y0
    ctx = diagram.create_context()
    proj = project_constraints(diagram, ctx, tol=1e-12)
    y_star = float(proj[plant.system_id].continuous_state[1])
    assert y_star == pytest.approx(_consistent_y(2.0), abs=1e-10)
    # differential entry untouched
    assert float(proj[plant.system_id].continuous_state[0]) == 2.0


@pytest.mark.parametrize("mode", ["implicit", "stop"])
def test_projected_value_gradient_vs_fd(mode):
    """AD of the projected algebraic value w.r.t. the differential state.

    'implicit' must match FD (IFT: dy*/dx = 1/(3 y*^2 + 1)); 'stop' must
    return exactly zero for the algebraic path.
    """
    diagram, plant = _build()
    ctx0 = diagram.create_context()
    sid = plant.system_id

    def y_star(x):
        state = jnp.array([x, 0.0])
        c = ctx0.with_subcontext(sid, ctx0[sid].with_continuous_state(state))
        p = project_constraints(diagram, c, tol=1e-13, gradient=mode)
        return p[sid].continuous_state[1]

    x = 2.0
    g_ad = float(jax.grad(y_star)(jnp.asarray(x)))
    if mode == "stop":
        assert g_ad == 0.0
        return
    eps = 1e-6
    g_fd = (float(y_star(jnp.asarray(x + eps))) - float(y_star(jnp.asarray(x - eps)))) / (2 * eps)
    ys = _consistent_y(x)
    g_ift = 1.0 / (3.0 * ys**2 + 1.0)
    assert g_ad == pytest.approx(g_fd, rel=1e-6)
    assert g_ad == pytest.approx(g_ift, rel=1e-8)


def test_unconverged_projection_warns():
    diagram, _ = _build(x0=2.0, y0=1e6)  # far-off guess, starved budget
    ctx = diagram.create_context()
    with pytest.warns(UserWarning, match="did not converge"):
        # cubic term makes undamped Newton from 1e6 need ~30+ halvings
        project_constraints(diagram, ctx, tol=1e-12, max_iter=2)


def test_converged_projection_is_silent():
    diagram, _ = _build(x0=2.0, y0=1.0)
    ctx = diagram.create_context()
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # any warning -> test failure
        project_constraints(diagram, ctx, tol=1e-10)


def test_stop_mode_correct_through_integration():
    """Reset-then-integrate: AD w.r.t. the differential IC must match FD.

    This is the pattern the default mode exists for — the projected
    algebraic value is re-enforced by BDF, so its IC carries no true
    sensitivity.
    """
    diagram, plant = _build()
    ctx0 = diagram.create_context()
    sid = plant.system_id
    opts = jx.SimulatorOptions(
        math_backend="jax", ode_solver_method="bdf",
        rtol=1e-9, atol=1e-11, enable_autodiff=True, max_major_steps=64,
    )

    def x_final(x0):
        state = jnp.array([x0, 0.0])
        c = ctx0.with_subcontext(sid, ctx0[sid].with_continuous_state(state))
        c = project_constraints(diagram, c, tol=1e-13)
        r = jx.simulate(diagram, c, (0.0, 0.5), options=opts)
        return r.context[sid].continuous_state[0]

    x = 2.0
    g_ad = float(jax.grad(x_final)(jnp.asarray(x)))
    eps = 1e-5
    g_fd = (float(x_final(jnp.asarray(x + eps))) - float(x_final(jnp.asarray(x - eps)))) / (2 * eps)
    assert g_ad == pytest.approx(g_fd, rel=1e-4)
