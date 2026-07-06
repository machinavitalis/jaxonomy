# SPDX-License-Identifier: MIT

"""T-124-followup-2d-and-nd — bilinear lookup-table fitting in 2-D.

Ships:
- ``jaxonomy.library.lookup_table_fitting.fit_table_2d`` — pure-functional
  bilinear least-squares solver.
- ``jaxonomy.library.fit_lookup_table_2d`` — block wrapper that returns
  a fully-built ``LookupTable2d``.
- ``LookupTable2d.fit_from_data`` — ergonomic classmethod wrapper.

Acceptance:
- A bilinear truth ``z = a*x + b*y + c`` is recovered exactly at the
  grid corners (within float noise).
- A noisy paraboloid is fit smoothly with the Laplacian penalty.
- ``jax.grad`` flows through ``z_data``.
- The resulting ``LookupTable2d`` block evaluates inside a diagram.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import jaxonomy
from jaxonomy import library
from jaxonomy.library import LookupTable2d, fit_lookup_table_2d
from jaxonomy.library.lookup_table_fitting import fit_table_2d


# ---------------------------------------------------------------------------
# Pure-functional fit (math layer)
# ---------------------------------------------------------------------------


class TestFitBilinearFunction:
    """A bilinear truth surface should be recovered to floating-point noise."""

    def test_recovers_z_eq_2x_plus_3y_plus_1(self):
        xp = jnp.linspace(0.0, 4.0, 5)
        yp = jnp.linspace(0.0, 3.0, 4)
        # Dense data covering every cell — over-determined and the
        # truth lies in the column space of A, so lstsq returns it
        # exactly (modulo float roundoff).
        rng = np.random.default_rng(0)
        x_data = jnp.asarray(rng.uniform(0.0, 4.0, size=400))
        y_data = jnp.asarray(rng.uniform(0.0, 3.0, size=400))
        z_data = 2.0 * x_data + 3.0 * y_data + 1.0
        zp = fit_table_2d(xp, yp, x_data, y_data, z_data)
        # Expected table: zp[i, j] = 2*xp[i] + 3*yp[j] + 1
        expected = 2.0 * xp[:, None] + 3.0 * yp[None, :] + 1.0
        assert zp.shape == expected.shape
        assert jnp.allclose(zp, expected, atol=1e-6, rtol=0.0), (zp, expected)

    def test_pure_constant_recovered_exactly(self):
        xp = jnp.linspace(0.0, 1.0, 4)
        yp = jnp.linspace(0.0, 1.0, 3)
        rng = np.random.default_rng(1)
        x_data = jnp.asarray(rng.uniform(0.0, 1.0, size=200))
        y_data = jnp.asarray(rng.uniform(0.0, 1.0, size=200))
        z_data = jnp.full_like(x_data, 7.5)
        zp = fit_table_2d(xp, yp, x_data, y_data, z_data)
        assert jnp.allclose(zp, 7.5, atol=1e-6)


class TestFitNoisyParaboloid:
    """A noisy paraboloid should fit the truth surface smoothly with the
    Laplacian penalty active."""

    def test_smoothness_reduces_high_freq_residual(self):
        rng = np.random.default_rng(2)
        xp = jnp.linspace(-1.0, 1.0, 9)
        yp = jnp.linspace(-1.0, 1.0, 9)
        x_data = jnp.asarray(rng.uniform(-1.0, 1.0, size=4000))
        y_data = jnp.asarray(rng.uniform(-1.0, 1.0, size=4000))
        truth = x_data ** 2 + y_data ** 2
        noise = jnp.asarray(rng.normal(scale=0.10, size=4000))
        z_data = truth + noise
        # Fit twice — once unsmoothed, once with smoothness.  The
        # smoothed fit should be no further from the analytic truth
        # than the unsmoothed and have smaller cell-to-cell variation.
        zp_raw = fit_table_2d(xp, yp, x_data, y_data, z_data)
        zp_smooth = fit_table_2d(
            xp, yp, x_data, y_data, z_data, smoothness=0.5
        )
        truth_grid = xp[:, None] ** 2 + yp[None, :] ** 2
        # Both should be close to truth — bilinear-interp's inherent
        # discretization error on a quadratic surface is roughly
        # ``(Δ/2)² ≈ (0.25/2)² = 0.0156`` per axis on a 9-point grid
        # over ``[-1, 1]``, so allow up to ~0.15 absolute deviation.
        assert jnp.max(jnp.abs(zp_raw - truth_grid)) < 0.15
        assert jnp.max(jnp.abs(zp_smooth - truth_grid)) < 0.15
        # Smoothness should reduce neighbour-to-neighbour table jitter
        # measured on the residual (truth_grid is smooth, so any
        # residual is fitting noise).
        roughness_raw = jnp.sum(jnp.diff(zp_raw - truth_grid, axis=0) ** 2) + jnp.sum(
            jnp.diff(zp_raw - truth_grid, axis=1) ** 2
        )
        roughness_smooth = jnp.sum(
            jnp.diff(zp_smooth - truth_grid, axis=0) ** 2
        ) + jnp.sum(jnp.diff(zp_smooth - truth_grid, axis=1) ** 2)
        assert roughness_smooth <= roughness_raw + 1e-12


class TestWeightedLeastSquares:
    """Setting per-sample weights to zero on a junk subset should make
    the fit ignore those samples."""

    def test_zero_weights_ignore_junk(self):
        xp = jnp.linspace(0.0, 1.0, 4)
        yp = jnp.linspace(0.0, 1.0, 4)
        rng = np.random.default_rng(3)
        x_truth = jnp.asarray(rng.uniform(0.05, 0.95, size=200))
        y_truth = jnp.asarray(rng.uniform(0.05, 0.95, size=200))
        z_truth = 5.0 * x_truth - 2.0 * y_truth
        x_junk = jnp.asarray(rng.uniform(0.05, 0.95, size=200))
        y_junk = jnp.asarray(rng.uniform(0.05, 0.95, size=200))
        z_junk = jnp.full_like(x_junk, -100.0)
        x_data = jnp.concatenate([x_truth, x_junk])
        y_data = jnp.concatenate([y_truth, y_junk])
        z_data = jnp.concatenate([z_truth, z_junk])
        weights = jnp.concatenate([jnp.ones(200), jnp.zeros(200)])
        zp = fit_table_2d(xp, yp, x_data, y_data, z_data, weights=weights)
        # Junk is ignored — fit recovers truth surface 5*x - 2*y.
        expected = 5.0 * xp[:, None] - 2.0 * yp[None, :]
        assert jnp.allclose(zp, expected, atol=1e-5)


# ---------------------------------------------------------------------------
# Differentiability
# ---------------------------------------------------------------------------


class TestDifferentiability:
    def test_grad_through_z_data_is_finite(self):
        xp = jnp.linspace(0.0, 1.0, 4)
        yp = jnp.linspace(0.0, 1.0, 4)
        rng = np.random.default_rng(4)
        x_data = jnp.asarray(rng.uniform(0.0, 1.0, size=80))
        y_data = jnp.asarray(rng.uniform(0.0, 1.0, size=80))

        def loss(z_data):
            zp = fit_table_2d(xp, yp, x_data, y_data, z_data)
            return jnp.sum(zp ** 2)

        z_data0 = jnp.asarray(rng.normal(size=80))
        g = jax.grad(loss)(z_data0)
        assert g.shape == z_data0.shape
        assert jnp.all(jnp.isfinite(g))

    def test_grad_with_smoothness_is_finite(self):
        xp = jnp.linspace(0.0, 1.0, 5)
        yp = jnp.linspace(0.0, 1.0, 4)
        rng = np.random.default_rng(5)
        x_data = jnp.asarray(rng.uniform(0.0, 1.0, size=120))
        y_data = jnp.asarray(rng.uniform(0.0, 1.0, size=120))

        def loss(z_data):
            zp = fit_table_2d(
                xp, yp, x_data, y_data, z_data, smoothness=0.1
            )
            return jnp.sum(zp ** 2)

        z_data0 = jnp.asarray(rng.normal(size=120))
        g = jax.grad(loss)(z_data0)
        assert g.shape == z_data0.shape
        assert jnp.all(jnp.isfinite(g))


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


class TestInputValidation:
    def test_negative_smoothness_rejected(self):
        xp = jnp.linspace(0.0, 1.0, 3)
        yp = jnp.linspace(0.0, 1.0, 3)
        x_data = jnp.array([0.5])
        y_data = jnp.array([0.5])
        z_data = jnp.array([1.0])
        with pytest.raises(ValueError, match="smoothness must be >= 0"):
            fit_table_2d(xp, yp, x_data, y_data, z_data, smoothness=-1.0)

    def test_mismatched_z_shape_rejected(self):
        xp = jnp.linspace(0.0, 1.0, 3)
        yp = jnp.linspace(0.0, 1.0, 3)
        x_data = jnp.array([0.5, 0.6])
        y_data = jnp.array([0.5, 0.6])
        z_data = jnp.array([1.0])
        with pytest.raises(ValueError, match="z_data shape"):
            fit_table_2d(xp, yp, x_data, y_data, z_data)

    def test_too_few_grid_points_rejected(self):
        xp = jnp.array([0.0])
        yp = jnp.linspace(0.0, 1.0, 3)
        x_data = jnp.array([0.5])
        y_data = jnp.array([0.5])
        z_data = jnp.array([1.0])
        with pytest.raises(ValueError, match="at least 2 grid points"):
            fit_table_2d(xp, yp, x_data, y_data, z_data)


# ---------------------------------------------------------------------------
# Block wrapper
# ---------------------------------------------------------------------------


class TestBlockWrapper:
    def test_returns_lookup_table_2d_instance(self):
        xp = jnp.linspace(0.0, 4.0, 5)
        yp = jnp.linspace(0.0, 3.0, 4)
        rng = np.random.default_rng(6)
        x_data = jnp.asarray(rng.uniform(0.0, 4.0, size=200))
        y_data = jnp.asarray(rng.uniform(0.0, 3.0, size=200))
        z_data = 2.0 * x_data + 3.0 * y_data + 1.0
        block = fit_lookup_table_2d(xp, yp, x_data, y_data, z_data)
        assert isinstance(block, LookupTable2d)

    def test_classmethod_returns_lookup_table_2d_instance(self):
        xp = jnp.linspace(0.0, 1.0, 4)
        yp = jnp.linspace(0.0, 1.0, 3)
        rng = np.random.default_rng(7)
        x_data = jnp.asarray(rng.uniform(0.0, 1.0, size=100))
        y_data = jnp.asarray(rng.uniform(0.0, 1.0, size=100))
        z_data = x_data + y_data
        block = LookupTable2d.fit_from_data(xp, yp, x_data, y_data, z_data)
        assert isinstance(block, LookupTable2d)

    def test_classmethod_parity_with_standalone_helper(self):
        # Pure-delegation parity: the classmethod returns a block that
        # evaluates identically to the one produced by the standalone
        # helper.  Compared at a query point (rather than via private
        # attributes) so we don't lean on the block's internal storage
        # layout.
        xp = jnp.linspace(0.0, 1.0, 4)
        yp = jnp.linspace(0.0, 1.0, 4)
        rng = np.random.default_rng(8)
        x_data = jnp.asarray(rng.uniform(0.0, 1.0, size=150))
        y_data = jnp.asarray(rng.uniform(0.0, 1.0, size=150))
        z_data = jnp.sin(jnp.pi * x_data) * jnp.cos(jnp.pi * y_data)
        block_cls = LookupTable2d.fit_from_data(
            xp, yp, x_data, y_data, z_data, smoothness=0.05
        )
        block_fn = fit_lookup_table_2d(
            xp, yp, x_data, y_data, z_data, smoothness=0.05
        )
        outs = []
        for block in (block_cls, block_fn):
            builder = jaxonomy.DiagramBuilder()
            b = builder.add(block)
            sx = builder.add(library.Constant(0.42))
            sy = builder.add(library.Constant(0.71))
            builder.connect(sx.output_ports[0], b.input_ports[0])
            builder.connect(sy.output_ports[0], b.input_ports[1])
            diagram = builder.build()
            ctx = diagram.create_context()
            outs.append(b.output_ports[0].eval(ctx))
        assert jnp.allclose(outs[0], outs[1], atol=0.0, rtol=0.0)

    def test_fitted_block_evaluates_in_diagram(self):
        # Drive the fitted block at a known query and verify the
        # bilinear evaluation matches the truth function.
        xp = jnp.linspace(0.0, 4.0, 5)
        yp = jnp.linspace(0.0, 3.0, 4)
        rng = np.random.default_rng(9)
        x_data = jnp.asarray(rng.uniform(0.0, 4.0, size=400))
        y_data = jnp.asarray(rng.uniform(0.0, 3.0, size=400))
        z_data = 2.0 * x_data + 3.0 * y_data + 1.0
        block = LookupTable2d.fit_from_data(xp, yp, x_data, y_data, z_data)

        builder = jaxonomy.DiagramBuilder()
        b = builder.add(block)
        sx = builder.add(library.Constant(2.5))
        sy = builder.add(library.Constant(1.5))
        builder.connect(sx.output_ports[0], b.input_ports[0])
        builder.connect(sy.output_ports[0], b.input_ports[1])
        diagram = builder.build()
        ctx = diagram.create_context()
        out = b.output_ports[0].eval(ctx)
        # Truth at (2.5, 1.5) = 2*2.5 + 3*1.5 + 1 = 10.5.
        assert float(out) == pytest.approx(10.5, abs=1e-4)

    def test_block_kwargs_forwarded(self):
        xp = jnp.linspace(0.0, 1.0, 3)
        yp = jnp.linspace(0.0, 1.0, 3)
        rng = np.random.default_rng(10)
        x_data = jnp.asarray(rng.uniform(0.0, 1.0, size=80))
        y_data = jnp.asarray(rng.uniform(0.0, 1.0, size=80))
        z_data = x_data + y_data
        block = LookupTable2d.fit_from_data(
            xp, yp, x_data, y_data, z_data, name="fit_2d_table"
        )
        assert block.name == "fit_2d_table"
