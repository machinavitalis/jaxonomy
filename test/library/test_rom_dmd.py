# SPDX-License-Identifier: MIT

"""Tests for data-driven operator identification: DMD / DMDc / ERA (T-146)."""

import numpy as np
import pytest

import jaxonomy
from jaxonomy.library.rom.dmd import (
    dmd,
    dmdc,
    era,
    DMDForecaster,
    DMDResult,
    DMDcResult,
    ERAResult,
)

pytestmark = pytest.mark.minimal


def _snapshots(A, x0, steps):
    x = np.asarray(x0, dtype=float)
    cols = [x]
    for _ in range(steps):
        x = A @ x
        cols.append(x)
    return np.array(cols).T


class TestDMD:
    def test_recovers_eigenvalues(self):
        # Known stable discrete linear system.
        A = np.array([[0.9, 0.1], [0.0, 0.8]])
        X = _snapshots(A, [1.0, -0.5], 25)

        result = dmd(X)
        assert isinstance(result, DMDResult)

        eig = np.sort(np.real(result.eigenvalues))
        true = np.sort(np.real(np.linalg.eigvals(A)))
        assert np.allclose(eig, true, atol=1e-8)

    def test_recovers_growth_from_complex_spectrum(self):
        # Rotation + decay -> a complex-conjugate eigenvalue pair.
        theta = 0.3
        r = 0.95
        A = r * np.array(
            [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]
        )
        X = _snapshots(A, [1.0, 0.0], 40)

        result = dmd(X, rank=2)
        # Magnitudes (growth/decay) recovered.
        assert np.allclose(np.sort(np.abs(result.eigenvalues)), [r, r], atol=1e-6)
        # A_tilde is r x r.
        assert result.A_tilde.shape == (2, 2)

    def test_explicit_snapshot_pair(self):
        A = np.array([[0.5, 0.0], [0.2, 0.7]])
        X = _snapshots(A, [2.0, 1.0], 20)
        r1 = dmd(X[:, :-1], X[:, 1:])
        r2 = dmd(X)
        assert np.allclose(
            np.sort(r1.eigenvalues.real), np.sort(r2.eigenvalues.real), atol=1e-8
        )


class TestDMDc:
    def test_recovers_A_and_B_unknown_B(self):
        rng = np.random.default_rng(0)
        A = np.array([[0.9, 0.1], [-0.05, 0.85]])
        B = np.array([[0.1], [0.2]])

        xs = [np.zeros(2)]
        us = []
        for _ in range(80):
            u = rng.standard_normal(1)
            us.append(u)
            xs.append(A @ xs[-1] + B @ u)
        X = np.array(xs).T
        U = np.array(us).T

        result = dmdc(X[:, :-1], X[:, 1:], U)
        assert isinstance(result, DMDcResult)
        assert np.allclose(result.A, A, atol=1e-8)
        assert np.allclose(result.B, B, atol=1e-8)

    def test_recovers_A_known_B(self):
        rng = np.random.default_rng(1)
        A = np.array([[0.8, 0.05], [0.0, 0.9]])
        B = np.array([[0.3], [0.1]])

        xs = [np.zeros(2)]
        us = []
        for _ in range(60):
            u = rng.standard_normal(1)
            us.append(u)
            xs.append(A @ xs[-1] + B @ u)
        X = np.array(xs).T
        U = np.array(us).T

        result = dmdc(X[:, :-1], X[:, 1:], U, B_known=B)
        assert np.allclose(result.A, A, atol=1e-8)
        assert np.allclose(result.B, B, atol=1e-12)

    def test_reduced_operators_shapes(self):
        rng = np.random.default_rng(2)
        A = np.array([[0.9, 0.1, 0.0], [0.0, 0.8, 0.05], [0.0, 0.0, 0.7]])
        B = np.array([[0.1], [0.0], [0.2]])
        xs = [np.zeros(3)]
        us = []
        for _ in range(90):
            u = rng.standard_normal(1)
            us.append(u)
            xs.append(A @ xs[-1] + B @ u)
        X = np.array(xs).T
        U = np.array(us).T

        result = dmdc(X[:, :-1], X[:, 1:], U, rank=2)
        assert result.A_tilde.shape == (2, 2)
        assert result.B_tilde.shape == (2, 1)
        assert result.basis.shape == (3, 2)


class TestERA:
    def _markov(self, A, B, C, D, n):
        seq = [D]
        Ak = np.eye(A.shape[0])
        for _ in range(n):
            seq.append(C @ Ak @ B)
            Ak = Ak @ A
        return np.array(seq)

    def test_realizes_known_impulse_response(self):
        A = np.array([[0.7, 0.2], [0.0, 0.5]])
        B = np.array([[1.0], [0.5]])
        C = np.array([[1.0, 0.5]])
        D = np.array([[0.3]])
        markov = self._markov(A, B, C, D, 40)

        result = era(markov, n_inputs=1, n_outputs=1)
        assert isinstance(result, ERAResult)

        # A minimal realization is unique only up to similarity, so compare the
        # reproduced impulse response rather than the matrices themselves.
        markov_hat = self._markov(result.A, result.B, result.C, result.D, 40)
        assert np.allclose(markov_hat, markov, atol=1e-8)
        assert result.A.shape[0] == 2  # minimal order recovered

    def test_mimo_realization(self):
        A = np.array([[0.6, 0.1], [-0.1, 0.4]])
        B = np.array([[1.0, 0.0], [0.0, 1.0]])
        C = np.array([[1.0, 0.0], [0.5, 1.0]])
        D = np.zeros((2, 2))
        markov = self._markov(A, B, C, D, 30)

        result = era(markov, n_inputs=2, n_outputs=2)
        markov_hat = self._markov(result.A, result.B, result.C, result.D, 30)
        assert np.allclose(markov_hat, markov, atol=1e-8)


class TestDMDForecaster:
    def test_autonomous_matches_direct_propagation(self):
        A = np.array([[0.9, 0.1], [0.0, 0.8]])
        x0 = np.array([1.0, -0.5])
        dt = 0.1
        steps = 12

        block = DMDForecaster(A=A, dt=dt, initial_state=x0, name="forecaster")
        ctx = block.create_context()
        results = jaxonomy.simulate(
            block,
            ctx,
            (0.0, steps * dt),
            recorded_signals={"y": block.output_ports[0]},
        )

        y = np.array(results.outputs["y"])
        xk = x0.copy()
        for _ in range(steps):
            xk = A @ xk
        assert np.allclose(y[-1], xk, atol=1e-6)

    def test_forecaster_from_fitted_operator(self):
        # Fit A_tilde with DMD, then forecast in the reduced coordinates.
        A = np.array([[0.95, 0.05], [0.0, 0.9]])
        x0 = np.array([1.0, 0.5])
        X = _snapshots(A, x0, 30)
        result = dmd(X)
        A_hat = np.real(result.A_tilde)

        dt = 0.1
        steps = 8
        block = DMDForecaster(A=A_hat, dt=dt, initial_state=x0, name="fc")
        ctx = block.create_context()
        sim = jaxonomy.simulate(
            block, ctx, (0.0, steps * dt),
            recorded_signals={"y": block.output_ports[0]},
        )
        y = np.array(sim.outputs["y"])
        xk = x0.copy()
        for _ in range(steps):
            xk = A_hat @ xk
        assert np.allclose(y[-1], xk, atol=1e-6)

    def test_controlled_forecaster(self):
        A = np.array([[0.9, 0.1], [0.0, 0.8]])
        B = np.array([[0.0], [1.0]])
        x0 = np.array([0.0, 0.0])
        dt = 0.1
        steps = 6

        block = DMDForecaster(A=A, B=B, dt=dt, initial_state=x0, name="fc_u")
        block.input_ports[0].fix_value(np.array([1.0]))
        ctx = block.create_context()
        sim = jaxonomy.simulate(
            block, ctx, (0.0, steps * dt),
            recorded_signals={"y": block.output_ports[0]},
        )
        y = np.array(sim.outputs["y"])
        xk = x0.copy()
        for _ in range(steps):
            xk = A @ xk + B @ np.array([1.0])
        assert np.allclose(y[-1], xk, atol=1e-6)
