# SPDX-License-Identifier: MIT

"""Tests for Koopman / eDMD operator approximation (T-147)."""

import numpy as np
import jax.numpy as jnp
import pytest

import jaxonomy
from jaxonomy.library.rom.koopman import (
    edmd,
    identity_dictionary,
    polynomial_dictionary,
    rbf_dictionary,
    KoopmanPredictor,
    EDMDResult,
    _lift_columns,
)

pytestmark = pytest.mark.minimal


# A mildly nonlinear map with a quadratic Koopman invariant subspace:
#   x1' = a x1
#   x2' = b x2 + c x1^2
_A, _B, _C = 0.9, 0.7, 0.3


def _step(x):
    return np.array([_A * x[0], _B * x[1] + _C * x[0] ** 2])


def _trajectory(x0, steps, seed=None):
    xs = [np.asarray(x0, dtype=float)]
    for _ in range(steps):
        xs.append(_step(xs[-1]))
    return np.array(xs).T


class TestDictionaries:
    def test_identity_first(self):
        x = np.array([2.0, -1.0])
        poly = polynomial_dictionary(2)
        z = np.asarray(poly(jnp.asarray(x)))
        assert np.allclose(z[:2], x)  # identity observables lead the vector

    def test_rbf_includes_state(self):
        centers = np.array([[0.0, 0.0], [1.0, 1.0]])
        g = rbf_dictionary(centers, epsilon=0.5)
        x = np.array([0.5, 0.5])
        z = np.asarray(g(jnp.asarray(x)))
        assert np.allclose(z[:2], x)
        assert z.shape[0] == 4  # 2 states + 2 RBF features


class TestEDMD:
    def test_beats_plain_dmd_on_nonlinear_map(self):
        X = _trajectory([0.6, -0.4], 80)
        X1, X2 = X[:, :-1], X[:, 1:]

        poly = edmd(X1, X2, polynomial_dictionary(2))
        plain = edmd(X1, X2, identity_dictionary())

        Zp = _lift_columns(X1, poly.dictionary)
        Zi = _lift_columns(X1, plain.dictionary)
        err_poly = np.abs(poly.C @ poly.K @ Zp - X2).max()
        err_plain = np.abs(plain.C @ plain.K @ Zi - X2).max()

        assert err_poly < err_plain
        assert err_poly < 1e-8          # polynomial lift captures the map exactly
        assert err_plain > 1e-3         # linear model cannot

    def test_recovers_exact_koopman_operator(self):
        # Dictionary spanning a genuine Koopman-invariant subspace {x1, x2, x1^2}.
        def invariant_dict(x):
            x = jnp.atleast_1d(x)
            return jnp.array([x[0], x[1], x[0] ** 2])

        X = _trajectory([0.5, 0.2], 80)
        X1, X2 = X[:, :-1], X[:, 1:]
        result = edmd(X1, X2, invariant_dict)
        assert isinstance(result, EDMDResult)

        Z1 = _lift_columns(X1, invariant_dict)
        Z2 = _lift_columns(X2, invariant_dict)
        # K advances the observables exactly.
        assert np.abs(result.K @ Z1 - Z2).max() < 1e-9

        # And it matches the analytic operator on {x1, x2, x1^2}.
        K_true = np.array([[_A, 0.0, 0.0], [0.0, _B, _C], [0.0, 0.0, _A ** 2]])
        assert np.allclose(result.K, K_true, atol=1e-8)

    def test_delift_matrix_recovers_state(self):
        X = _trajectory([0.7, 0.1], 60)
        X1 = X[:, :-1]
        result = edmd(X1, X[:, 1:], polynomial_dictionary(2))
        Z1 = _lift_columns(X1, result.dictionary)
        assert np.allclose(result.C @ Z1, X1, atol=1e-8)

    def test_edmdc_with_input(self):
        rng = np.random.default_rng(0)
        # Linear-in-observables controlled map on {x1, x2, x1^2}.
        xs = [np.array([0.4, -0.2])]
        us = []
        for _ in range(70):
            u = rng.standard_normal(1)
            us.append(u)
            x = xs[-1]
            xs.append(np.array([_A * x[0] + 0.1 * u[0], _B * x[1] + _C * x[0] ** 2]))
        X = np.array(xs).T
        U = np.array(us).T

        def invariant_dict(x):
            x = jnp.atleast_1d(x)
            return jnp.array([x[0], x[1], x[0] ** 2])

        result = edmd(X[:, :-1], X[:, 1:], invariant_dict, U=U)
        assert result.B is not None
        assert result.B.shape == (3, 1)


class TestKoopmanPredictor:
    def test_simulates_and_delifts(self):
        poly = polynomial_dictionary(2)
        X = _trajectory([0.5, -0.3], 80)
        result = edmd(X[:, :-1], X[:, 1:], poly)

        x0 = np.array([0.5, -0.3])
        dt = 1.0
        steps = 8
        block = KoopmanPredictor(
            K=np.asarray(result.K),
            C=np.asarray(result.C),
            dictionary=poly,
            dt=dt,
            initial_state=x0,
            name="koopman",
        )
        ctx = block.create_context()
        sim = jaxonomy.simulate(
            block, ctx, (0.0, steps * dt),
            recorded_signals={"x": block.output_ports[0]},
        )
        x_sim = np.array(sim.outputs["x"])[-1]

        xk = x0.copy()
        for _ in range(steps):
            xk = _step(xk)
        assert np.allclose(x_sim, xk, atol=1e-6)

    def test_scalar_state_predictor(self):
        # Scalar logistic-like map lifted with polynomials.
        def step1(x):
            return np.array([0.8 * x[0] + 0.1 * x[0] ** 2])

        xs = [np.array([0.5])]
        for _ in range(60):
            xs.append(step1(xs[-1]))
        X = np.array(xs).T
        poly = polynomial_dictionary(2)
        result = edmd(X[:, :-1], X[:, 1:], poly)

        x0 = np.array([0.5])
        dt = 1.0
        steps = 6
        block = KoopmanPredictor(
            K=np.asarray(result.K), C=np.asarray(result.C),
            dictionary=poly, dt=dt, initial_state=x0, name="koopman_scalar",
        )
        ctx = block.create_context()
        sim = jaxonomy.simulate(
            block, ctx, (0.0, steps * dt),
            recorded_signals={"x": block.output_ports[0]},
        )
        x_sim = np.atleast_1d(np.array(sim.outputs["x"])[-1])

        xk = x0.copy()
        for _ in range(steps):
            xk = step1(xk)
        assert np.allclose(x_sim, xk, atol=1e-5)
