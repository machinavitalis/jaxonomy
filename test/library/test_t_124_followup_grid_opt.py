# SPDX-License-Identifier: MIT

"""T-124-followup-grid-optimization — joint xp + yp optimisation.

Phase 1 :func:`fit_table_1d` solves yp at a fixed user-supplied xp.
This follow-up adds :func:`fit_table_1d_with_grid` which ALSO optimises
the grid placement.

Acceptance:
- On a sharp-peaked function (narrow Gaussian), the optimised grid
  clusters more interior breakpoints near the peak than uniform.
- On a smooth function (low-frequency sine), the optimised grid stays
  roughly evenly spaced (no incentive to cluster).
- The joint fit beats the fixed-uniform fit on residual.
- Differentiability: ``jax.grad`` of the loss-at-the-fitted-table
  w.r.t. ``y_data`` is finite (the gradient-descent path supports
  end-to-end autodiff).
- Phase-1 :func:`fit_table_1d` is byte-for-byte unchanged
  (regression-pinned by the existing phase-1 test file, which we run
  alongside).
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaxonomy.library.lookup_table import fit_table_1d
from jaxonomy.library.lookup_table_fitting import (
    _xp_from_deltas,
    _deltas_from_xp,
    fit_table_1d_with_grid,
)


# ---------------------------------------------------------------------------
# Parametrisation primitives
# ---------------------------------------------------------------------------


class TestParametrisation:
    """The cumsum(softplus(deltas)) parametrisation must be monotone and
    invertible at a uniform grid."""

    def test_uniform_init_round_trips(self):
        xp = jnp.linspace(-2.0, 7.0, 8)
        deltas = _deltas_from_xp(xp, -2.0, 7.0)
        xp_back = _xp_from_deltas(deltas, -2.0, 7.0)
        assert jnp.allclose(xp_back, xp, atol=1e-12)

    def test_xp_strictly_increasing_for_arbitrary_deltas(self):
        rng = np.random.default_rng(0)
        for _ in range(5):
            d = jnp.asarray(rng.normal(size=10))
            xp = _xp_from_deltas(d, 0.0, 1.0)
            diffs = jnp.diff(xp)
            assert jnp.all(diffs > 0), diffs

    def test_endpoints_pinned(self):
        rng = np.random.default_rng(1)
        d = jnp.asarray(rng.normal(size=6))
        xp = _xp_from_deltas(d, -3.0, 4.0)
        assert float(xp[0]) == pytest.approx(-3.0, abs=1e-12)
        assert float(xp[-1]) == pytest.approx(4.0, abs=1e-12)


# ---------------------------------------------------------------------------
# Behavioural acceptance: peaked data attracts grid points to the peak
# ---------------------------------------------------------------------------


def _interp_residual(xp, yp, x_data, y_data):
    return float(jnp.linalg.norm(jnp.interp(x_data, xp, yp) - y_data))


class TestSharpPeakAttractsGrid:
    """A narrow-Gaussian truth function should pull breakpoints toward
    the peak — measured as: more interior breakpoints land in the
    central [-1, 1] window than under a uniform grid."""

    def test_narrow_gaussian_grid_clusters_near_peak(self):
        rng = np.random.default_rng(0)
        x_data = jnp.asarray(rng.uniform(-5.0, 5.0, size=2000))
        y_data = jnp.exp(-(x_data ** 2) / 0.1)
        n = 9

        xp_uniform = jnp.linspace(-5.0, 5.0, n)
        # Uniform grid: only the breakpoint at exactly 0 lies in (-1, 1).
        n_central_uniform = int(jnp.sum((xp_uniform > -1.0) & (xp_uniform < 1.0)))

        xp_opt, yp_opt = fit_table_1d_with_grid(
            n,
            x_data,
            y_data,
            x_lo=-5.0,
            x_hi=5.0,
            max_iter=500,
            learning_rate=1e-3,
        )
        n_central_opt = int(jnp.sum((xp_opt > -1.0) & (xp_opt < 1.0)))

        assert n_central_opt > n_central_uniform, (
            f"expected more interior grid points near peak after optimisation: "
            f"uniform had {n_central_uniform}, optimised has {n_central_opt}; "
            f"xp_opt = {xp_opt}"
        )

    def test_narrow_gaussian_residual_drops(self):
        rng = np.random.default_rng(0)
        x_data = jnp.asarray(rng.uniform(-5.0, 5.0, size=2000))
        y_data = jnp.exp(-(x_data ** 2) / 0.1)
        n = 9

        xp_uniform = jnp.linspace(-5.0, 5.0, n)
        yp_uniform = fit_table_1d(xp_uniform, x_data, y_data)
        r_uniform = _interp_residual(xp_uniform, yp_uniform, x_data, y_data)

        xp_opt, yp_opt = fit_table_1d_with_grid(
            n,
            x_data,
            y_data,
            x_lo=-5.0,
            x_hi=5.0,
            max_iter=500,
            learning_rate=1e-3,
        )
        r_opt = _interp_residual(xp_opt, yp_opt, x_data, y_data)
        # Sharp-peaked truth + 9-point grid: optimisation should give a
        # large multiplicative improvement.  5x is conservative — we
        # see ~8x in practice on this fixture (residual drops from
        # ~4.4 to ~0.5).
        assert r_opt < r_uniform / 5.0, (r_uniform, r_opt)


class TestSmoothFunctionStaysRoughlyUniform:
    """A low-frequency smooth function gives the optimiser little
    incentive to cluster — the grid should stay close to uniform."""

    def test_smooth_sine_grid_close_to_uniform(self):
        # f(x) = sin(x) on [0, 2π] with 5 grid points.  The truth has
        # no localised features → the optimiser should not pile points
        # into one region.
        rng = np.random.default_rng(2)
        x_data = jnp.asarray(rng.uniform(0.0, 2.0 * math.pi, size=1000))
        y_data = jnp.sin(x_data)
        n = 5

        xp_opt, _ = fit_table_1d_with_grid(
            n,
            x_data,
            y_data,
            x_lo=0.0,
            x_hi=2.0 * math.pi,
            max_iter=200,
            learning_rate=1e-3,
        )
        # Compare interior diffs to the uniform spacing (2π / 4).
        diffs = jnp.diff(xp_opt)
        uniform_dx = 2.0 * math.pi / (n - 1)
        # No diff should be smaller than 25% of uniform or larger than
        # 4x uniform — this is "roughly evenly spaced".  (In practice
        # the optimiser nudges them a little but not catastrophically.)
        assert float(jnp.min(diffs)) > 0.25 * uniform_dx, (diffs, uniform_dx)
        assert float(jnp.max(diffs)) < 4.0 * uniform_dx, (diffs, uniform_dx)


# ---------------------------------------------------------------------------
# Joint fit beats fixed-uniform fit
# ---------------------------------------------------------------------------


class TestOptimisedBeatsUniform:
    def test_double_exponential_optimised_smaller_residual(self):
        # Two narrow bumps off-centre — uniform 7-point grid has no
        # chance; optimised should do considerably better.
        rng = np.random.default_rng(3)
        x_data = jnp.asarray(rng.uniform(-4.0, 4.0, size=1500))
        y_data = jnp.exp(-((x_data - 2.0) ** 2) / 0.1) + jnp.exp(
            -((x_data + 2.0) ** 2) / 0.1
        )
        n = 7

        xp_uniform = jnp.linspace(-4.0, 4.0, n)
        yp_uniform = fit_table_1d(xp_uniform, x_data, y_data)
        r_uniform = _interp_residual(xp_uniform, yp_uniform, x_data, y_data)

        xp_opt, yp_opt = fit_table_1d_with_grid(
            n,
            x_data,
            y_data,
            x_lo=-4.0,
            x_hi=4.0,
            max_iter=600,
            learning_rate=1e-3,
        )
        r_opt = _interp_residual(xp_opt, yp_opt, x_data, y_data)
        assert r_opt < r_uniform


# ---------------------------------------------------------------------------
# Differentiability
# ---------------------------------------------------------------------------


class TestDifferentiability:
    """The gradient-descent optimiser supports end-to-end autodiff:
    ``jax.grad(loss(xp_opt, yp_opt))`` w.r.t. ``y_data`` flows through
    the unrolled scan."""

    def test_grad_through_y_data_finite(self):
        rng = np.random.default_rng(4)
        x_data = jnp.asarray(rng.uniform(-2.0, 2.0, size=200))

        def loss(y_data):
            xp_opt, yp_opt = fit_table_1d_with_grid(
                5,
                x_data,
                y_data,
                x_lo=-2.0,
                x_hi=2.0,
                # Keep this short — we just need a finite gradient,
                # not full convergence.
                max_iter=20,
                learning_rate=1e-3,
            )
            return jnp.sum(yp_opt ** 2)

        y_data0 = jnp.asarray(rng.normal(size=200))
        g = jax.grad(loss)(y_data0)
        assert g.shape == y_data0.shape
        assert jnp.all(jnp.isfinite(g)), g


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


class TestValidation:
    def test_too_few_grid_points_raises(self):
        with pytest.raises(ValueError):
            fit_table_1d_with_grid(1, jnp.zeros(10), jnp.zeros(10))

    def test_unknown_optimizer_raises(self):
        with pytest.raises(ValueError):
            fit_table_1d_with_grid(
                5, jnp.zeros(10), jnp.zeros(10), optimizer="adam"
            )

    def test_negative_smoothness_raises(self):
        with pytest.raises(ValueError):
            fit_table_1d_with_grid(
                5, jnp.zeros(10), jnp.zeros(10), smoothness=-1.0
            )

    def test_mismatched_xy_shapes_raise(self):
        with pytest.raises(ValueError):
            fit_table_1d_with_grid(5, jnp.zeros(10), jnp.zeros(11))

    def test_init_xp_wrong_size_raises(self):
        with pytest.raises(ValueError):
            fit_table_1d_with_grid(
                5,
                jnp.zeros(10),
                jnp.zeros(10),
                init_xp=jnp.linspace(0.0, 1.0, 4),
            )

    def test_endpoints_default_to_data_range(self):
        x_data = jnp.linspace(2.5, 17.5, 100)
        y_data = jnp.sin(x_data)
        xp_opt, _ = fit_table_1d_with_grid(
            5, x_data, y_data, max_iter=10, learning_rate=1e-3
        )
        assert float(xp_opt[0]) == pytest.approx(2.5, abs=1e-6)
        assert float(xp_opt[-1]) == pytest.approx(17.5, abs=1e-6)


# ---------------------------------------------------------------------------
# LBFGS path (forward-only)
# ---------------------------------------------------------------------------


class TestLBFGSOptionalPath:
    """The ``optimizer='lbfgs'`` branch is forward-only — verify it
    produces a reasonable fit without expecting it to support backprop
    through itself."""

    def test_lbfgs_runs_and_returns_finite(self):
        rng = np.random.default_rng(5)
        x_data = jnp.asarray(rng.uniform(-1.0, 1.0, size=300))
        y_data = jnp.sin(3.0 * x_data)
        xp_opt, yp_opt = fit_table_1d_with_grid(
            6,
            x_data,
            y_data,
            x_lo=-1.0,
            x_hi=1.0,
            optimizer="lbfgs",
            max_iter=50,
        )
        assert xp_opt.shape == (6,)
        assert yp_opt.shape == (6,)
        assert jnp.all(jnp.isfinite(xp_opt))
        assert jnp.all(jnp.isfinite(yp_opt))
        # Endpoints still pinned by the parametrisation.
        assert float(xp_opt[0]) == pytest.approx(-1.0, abs=1e-6)
        assert float(xp_opt[-1]) == pytest.approx(1.0, abs=1e-6)


# ---------------------------------------------------------------------------
# Phase-1 fit_table_1d untouched (sanity)
# ---------------------------------------------------------------------------


class TestPhase1Unchanged:
    """Quick spot-check that fit_table_1d at fixed uniform xp gives the
    same result it always did.  Full regression is in
    ``test_t_124_lookup_fitting_phase1.py``."""

    def test_linear_truth_recovered_exactly(self):
        xp = jnp.linspace(0.0, 10.0, 11)
        x_data = jnp.linspace(0.0, 10.0, 201)
        y_data = 2.0 * x_data + 3.0
        yp = fit_table_1d(xp, x_data, y_data)
        expected = 2.0 * xp + 3.0
        assert jnp.allclose(yp, expected, atol=1e-6)
