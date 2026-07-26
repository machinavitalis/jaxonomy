# SPDX-License-Identifier: MIT

"""Discrete-time path of ``step_response`` / ``impulse_response``.

Previously both raised ``ValueError`` on a discrete :class:`LinearizedSystem`
(after an earlier sweep fixed them silently returning continuous-formula
nonsense). This tests the real discrete recurrence:

* ZOH exactness: a ``discretize(..., method="zoh")`` system's step samples
  equal the continuous step response at ``t = k*dt`` to machine precision,
* analytic first-order geometric-series step response,
* impulse convention ``y[0]=D``, ``y[k]=C A^{k-1} B`` vs direct matrix powers
  and vs ``scipy.signal.dstep``/``dimpulse``,
* the off-grid-time policy (raise, no silent interpolation), negative-time
  causality, scalar/MIMO shapes, and differentiability through (A, B).
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaxonomy.library import LinearizedSystem, discretize, impulse_response, step_response
from jaxonomy.testing.markers import skip_if_not_jax

skip_if_not_jax()

DT = 0.05


def _cont_2x2():
    # Stable, well-damped two-state SISO plant.
    A = np.array([[0.0, 1.0], [-4.0, -1.2]])
    B = np.array([[0.0], [1.0]])
    C = np.array([[1.0, 0.0]])
    D = np.array([[0.0]])
    return LinearizedSystem(
        jnp.asarray(A), jnp.asarray(B), jnp.asarray(C), jnp.asarray(D), {}
    )


def _disc(A, B, C, D, dt=DT):
    return LinearizedSystem(
        jnp.asarray(np.atleast_2d(A)),
        jnp.asarray(np.atleast_2d(B)),
        jnp.asarray(np.atleast_2d(C)),
        jnp.asarray(np.atleast_2d(D)),
        {},
        dt=dt,
    )


def test_zoh_step_matches_continuous_at_samples():
    # ZOH is exact for piecewise-constant inputs, so the discrete step
    # response must equal the continuous one on the sampling grid.
    cont = _cont_2x2()
    disc = discretize(cont, DT, method="zoh")
    t = np.arange(40) * DT
    y_d = np.asarray(step_response(disc, t))
    y_c = np.asarray(step_response(cont, jnp.asarray(t)))
    np.testing.assert_allclose(y_d, y_c, rtol=1e-8, atol=1e-10)


def test_first_order_step_geometric_series():
    # x[k+1] = a x[k] + b, y = x: y[k] = b*(1-a^k)/(1-a).
    a, b = 0.9, 0.5
    sys_ = _disc([[a]], [[b]], [[1.0]], [[0.0]])
    k = np.arange(25)
    y = np.asarray(step_response(sys_, k * DT)).ravel()
    y_ref = b * (1.0 - a**k) / (1.0 - a)
    np.testing.assert_allclose(y, y_ref, rtol=1e-10, atol=1e-12)


def test_impulse_matches_matrix_powers_and_D_at_zero():
    rng = np.random.default_rng(0)
    A = 0.5 * rng.standard_normal((3, 3))
    B = rng.standard_normal((3, 2))
    C = rng.standard_normal((2, 3))
    D = rng.standard_normal((2, 2))
    sys_ = _disc(A, B, C, D)
    t = np.arange(12) * DT
    y = np.asarray(impulse_response(sys_, t))  # (12, 2, 2)

    np.testing.assert_allclose(y[0], D, rtol=1e-12)
    Ak = np.eye(3)
    for k in range(1, 12):
        np.testing.assert_allclose(y[k], C @ Ak @ B, rtol=1e-9, atol=1e-11)
        Ak = A @ Ak


def test_matches_scipy_dstep_dimpulse():
    signal = pytest.importorskip("scipy.signal")
    A = np.array([[0.8, 0.1], [0.0, 0.7]])
    B = np.array([[0.0], [1.0]])
    C = np.array([[1.0, 0.5]])
    D = np.array([[0.2]])
    sys_ = _disc(A, B, C, D)
    n_steps = 30
    t = np.arange(n_steps) * DT

    y_step = np.asarray(step_response(sys_, t))[:, 0, 0]
    _, (y_step_ref,) = signal.dstep((A, B, C, D, DT), n=n_steps)
    np.testing.assert_allclose(y_step, y_step_ref.ravel(), rtol=1e-9, atol=1e-11)

    y_imp = np.asarray(impulse_response(sys_, t))[:, 0, 0]
    _, (y_imp_ref,) = signal.dimpulse((A, B, C, D, DT), n=n_steps)
    np.testing.assert_allclose(y_imp, y_imp_ref.ravel(), rtol=1e-9, atol=1e-11)


def test_off_grid_time_raises():
    sys_ = _disc([[0.9]], [[1.0]], [[1.0]], [[0.0]])
    with pytest.raises(ValueError, match="off-grid times"):
        step_response(sys_, np.array([0.0, DT, 1.7 * DT]))
    with pytest.raises(ValueError, match="off-grid times"):
        impulse_response(sys_, np.array([0.5 * DT]))


def test_negative_times_are_causally_zero():
    sys_ = _disc([[0.9]], [[1.0]], [[1.0]], [[0.3]])
    t = np.array([-2, -1, 0, 1]) * DT
    y_step = np.asarray(step_response(sys_, t)).ravel()
    y_imp = np.asarray(impulse_response(sys_, t)).ravel()
    np.testing.assert_allclose(y_step[:2], 0.0)
    np.testing.assert_allclose(y_imp[:2], 0.0)
    assert y_step[2] == pytest.approx(0.3)  # D at k=0
    assert y_imp[2] == pytest.approx(0.3)


def test_scalar_time_and_mimo_shapes():
    rng = np.random.default_rng(1)
    sys_ = _disc(
        0.5 * rng.standard_normal((3, 3)),
        rng.standard_normal((3, 2)),
        rng.standard_normal((4, 3)),
        np.zeros((4, 2)),
    )
    y_scalar = np.asarray(step_response(sys_, 5 * DT))
    assert y_scalar.shape == (4, 2)
    y_vec = np.asarray(step_response(sys_, np.arange(7) * DT))
    assert y_vec.shape == (7, 4, 2)


def test_discrete_step_is_differentiable():
    k_grid = np.arange(10) * DT

    def loss(a):
        sys_ = LinearizedSystem(
            jnp.array([[a]]), jnp.array([[1.0]]), jnp.array([[1.0]]),
            jnp.array([[0.0]]), {}, dt=DT,
        )
        return jnp.sum(step_response(sys_, k_grid) ** 2)

    g = jax.grad(loss)(0.9)
    # d/da sum_k ((1-a^k)/(1-a))^2 at a=0.9: finite-difference check.
    eps = 1e-6
    g_fd = (loss(0.9 + eps) - loss(0.9 - eps)) / (2 * eps)
    assert np.isfinite(float(g))
    np.testing.assert_allclose(float(g), float(g_fd), rtol=1e-4)
