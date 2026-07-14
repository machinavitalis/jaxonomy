# SPDX-License-Identifier: MIT

"""Tests for POD-Galerkin projection ROMs and DEIM hyper-reduction (T-145)."""

import numpy as np
import jax.numpy as jnp
import pytest

import jaxonomy
from jaxonomy.framework import LeafSystem
from jaxonomy.library.rom.snapshots import (
    collect_snapshots,
    relative_error,
    retained_energy,
    projection_error,
)
from jaxonomy.library.rom.pod import (
    pod_basis,
    galerkin_reduce,
    deim,
    deim_galerkin_reduce,
)

pytestmark = pytest.mark.minimal


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _laplacian_1d(n, dx):
    """Dirichlet tridiagonal second-difference operator, shape (n, n)."""
    D2 = (
        np.diag(-2.0 * np.ones(n))
        + np.diag(np.ones(n - 1), 1)
        + np.diag(np.ones(n - 1), -1)
    ) / dx**2
    return D2


class _FullODE(LeafSystem):
    """Minimal full-order integrator ``ẋ = rhs(t, x)`` for reference sims."""

    def __init__(self, rhs, n, x0, name=None, **kwargs):
        super().__init__(name=name, **kwargs)
        self._rhs = rhs
        self.declare_continuous_state(
            shape=(n,), ode=self._ode, default_value=jnp.asarray(x0)
        )
        self.declare_continuous_state_output(name="x")

    def _ode(self, time, state, *inputs, **params):
        return self._rhs(time, state.continuous_state)


def _resample(t, Y, tq):
    """Interpolate a time-major trajectory ``Y`` (T, n) onto grid ``tq``."""
    t = np.asarray(t)
    Y = np.asarray(Y)
    return np.stack([np.interp(tq, t, Y[:, i]) for i in range(Y.shape[1])], axis=1)


# ---------------------------------------------------------------------------
# (1) POD basis & metrics
# ---------------------------------------------------------------------------

class TestPODBasisAndMetrics:
    def test_basis_orthonormal(self):
        rng = np.random.default_rng(0)
        X = rng.standard_normal((40, 25))
        Phi, sigma, r = pod_basis(X, rank=6)
        assert Phi.shape == (40, 6)
        assert r == 6
        gram = np.asarray(Phi.T @ Phi)
        assert np.allclose(gram, np.eye(6), atol=1e-10)

    def test_retained_energy_monotone(self):
        rng = np.random.default_rng(1)
        X = rng.standard_normal((30, 20))
        _, sigma, _ = pod_basis(X)
        energies = [retained_energy(sigma, r) for r in range(0, len(sigma) + 1)]
        assert energies[0] == 0.0
        assert np.isclose(energies[-1], 1.0)
        diffs = np.diff(energies)
        assert np.all(diffs >= -1e-12)  # non-decreasing

    def test_energy_rank_selection(self):
        rng = np.random.default_rng(2)
        # Low-rank-ish data: decaying singular values.
        U, _ = np.linalg.qr(rng.standard_normal((50, 50)))
        V, _ = np.linalg.qr(rng.standard_normal((30, 30)))
        s = np.exp(-np.arange(30))
        X = U[:, :30] @ np.diag(s) @ V.T
        Phi, sigma, r = pod_basis(X, energy=0.99)
        assert retained_energy(sigma, r) >= 0.99
        assert retained_energy(sigma, r - 1) < 0.99

    def test_projection_error_decreases(self):
        rng = np.random.default_rng(3)
        s = np.exp(-0.4 * np.arange(20))
        U, _ = np.linalg.qr(rng.standard_normal((40, 40)))
        V, _ = np.linalg.qr(rng.standard_normal((20, 20)))
        X = U[:, :20] @ np.diag(s) @ V.T
        errs = []
        for r in (1, 3, 6, 10):
            Phi, _, _ = pod_basis(X, rank=r)
            errs.append(projection_error(X, np.asarray(Phi)))
        assert all(errs[i + 1] <= errs[i] + 1e-12 for i in range(len(errs) - 1))
        assert errs[-1] < errs[0]


# ---------------------------------------------------------------------------
# (2) Linear heat equation: POD-Galerkin ROM
# ---------------------------------------------------------------------------

class TestGalerkinHeatEquation:
    def _build(self):
        n = 50
        dx = 1.0 / (n + 1)
        alpha = 0.2
        D2 = _laplacian_1d(n, dx)
        A = alpha * D2
        A_j = jnp.asarray(A)

        def rhs(t, x):
            return A_j @ x

        grid = np.linspace(dx, 1.0 - dx, n)
        x0 = np.sin(np.pi * grid) + 0.4 * np.sin(3 * np.pi * grid)
        return n, rhs, x0

    def test_reduced_matches_full(self):
        n, rhs, x0 = self._build()
        tf = 0.5

        full = _FullODE(rhs, n, x0, name="full")
        ctx = full.create_context()
        res_full = jaxonomy.simulate(
            full, ctx, (0.0, tf),
            recorded_signals={"x": full.output_ports[0]},
        )

        snaps = collect_snapshots(res_full, signals=["x"])
        assert snaps.X.shape[0] == n
        Phi, sigma, r = pod_basis(snaps.X, rank=5)
        assert r == 5

        rom = galerkin_reduce(rhs, Phi, input_size=0, name="rom")
        xr0 = np.asarray(Phi).T @ x0
        ctx_r = rom.create_context().with_continuous_state(jnp.asarray(xr0))
        res_red = jaxonomy.simulate(
            rom, ctx_r, (0.0, tf),
            recorded_signals={"x": rom.output_ports[0]},
        )

        tq = np.linspace(0.0, tf, 50)
        Yf = _resample(res_full.time, res_full.outputs["x"], tq)
        Yr = _resample(res_red.time, res_red.outputs["x"], tq)
        assert relative_error(Yf, Yr) < 1e-2


# ---------------------------------------------------------------------------
# (3) Nonlinear reaction-diffusion: DEIM hyper-reduction
# ---------------------------------------------------------------------------

class TestDEIM:
    def _build(self):
        n = 60
        dx = 1.0 / (n + 1)
        alpha = 0.05
        rho = 3.0
        A_j = jnp.asarray(alpha * _laplacian_1d(n, dx))

        def linear_rhs(t, x):
            return A_j @ x

        def nonlinear(x):  # elementwise Fisher-KPP reaction term
            return rho * x * (1.0 - x)

        def full_rhs(t, x):
            return linear_rhs(t, x) + nonlinear(x)

        grid = np.linspace(dx, 1.0 - dx, n)
        x0 = 0.6 * np.exp(-30.0 * (grid - 0.5) ** 2)
        return n, linear_rhs, nonlinear, full_rhs, x0

    def test_deim_indices_distinct(self):
        n, _, nonlinear, full_rhs, x0 = self._build()
        full = _FullODE(full_rhs, n, x0, name="full")
        ctx = full.create_context()
        res = jaxonomy.simulate(
            full, ctx, (0.0, 0.6),
            recorded_signals={"x": full.output_ports[0]},
        )
        snaps = collect_snapshots(res, signals=["x"])
        N = np.asarray(nonlinear(jnp.asarray(snaps.X)))  # (n, n_samples)
        indices, projector = deim(N, rank=8)
        assert indices.shape == (8,)
        assert len(set(indices.tolist())) == 8      # distinct
        assert np.all((indices >= 0) & (indices < n))
        assert projector.shape == (n, 8)

    def test_deim_reduced_matches_full_and_is_cheap(self):
        n, linear_rhs, nonlinear, full_rhs, x0 = self._build()
        tf = 0.6

        full = _FullODE(full_rhs, n, x0, name="full")
        ctx = full.create_context()
        res_full = jaxonomy.simulate(
            full, ctx, (0.0, tf),
            recorded_signals={"x": full.output_ports[0]},
        )

        snaps = collect_snapshots(res_full, signals=["x"])
        Phi, _, r = pod_basis(snaps.X, rank=8)
        N = np.asarray(nonlinear(jnp.asarray(snaps.X)))
        deim_res = deim(N, rank=8)
        m = deim_res[0].shape[0]
        assert m < n  # hyper-reduction: far fewer than full nodes

        seen_sizes = []

        def nonlinear_tracked(xp):
            seen_sizes.append(xp.shape[-1])
            return nonlinear(xp)

        rom = deim_galerkin_reduce(
            linear_rhs, nonlinear_tracked, Phi, deim_res,
            input_size=0, name="deim_rom",
        )
        xr0 = np.asarray(Phi).T @ x0
        ctx_r = rom.create_context().with_continuous_state(jnp.asarray(xr0))
        res_red = jaxonomy.simulate(
            rom, ctx_r, (0.0, tf),
            recorded_signals={"x": rom.output_ports[0]},
        )

        # The nonlinearity was only ever evaluated at the m DEIM points.
        assert seen_sizes and all(s == m for s in seen_sizes)

        tq = np.linspace(0.0, tf, 50)
        Yf = _resample(res_full.time, res_full.outputs["x"], tq)
        Yr = _resample(res_red.time, res_red.outputs["x"], tq)
        assert relative_error(Yf, Yr) < 5e-2
