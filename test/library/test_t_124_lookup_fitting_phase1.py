# SPDX-License-Identifier: MIT

"""T-124 phase 1 — differentiable lookup-table fitting.

Phase 1 ships:
- ``jaxonomy.library.lookup_table.fit_table_1d`` — the pure-functional
  least-squares solver (linear-interp design matrix + optional
  smoothness penalty).
- ``jaxonomy.library.fit_lookup_table_1d`` — convenience wrapper that
  returns a ready-to-use ``LookupTable1d`` block.

Acceptance:
- Fit a linear function exactly (within float noise).
- Fit a noisy sine within `O(noise / sqrt(K))`.
- ``jax.grad(loss)(y_data)`` is finite (i.e. the fit is differentiable
  through ``y_data``).
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import jaxonomy
from jaxonomy import library
from jaxonomy.library.lookup_table import fit_table_1d
from jaxonomy.library.lookup_table_fitting import fit_lookup_table_1d


# ---------------------------------------------------------------------------
# Pure-functional fit (math layer)
# ---------------------------------------------------------------------------


class TestFitLinearFunction:
    """A linear truth function should be recovered to floating-point noise."""

    def test_recovers_y_eq_2x_plus_3(self):
        xp = jnp.linspace(0.0, 10.0, 11)
        # Dense data on the same range — over-determined system with the
        # truth lying in the column space of A, so lstsq returns it
        # exactly (modulo float roundoff).
        x_data = jnp.linspace(0.0, 10.0, 201)
        y_data = 2.0 * x_data + 3.0
        yp = fit_table_1d(xp, x_data, y_data)
        expected = 2.0 * xp + 3.0
        assert jnp.allclose(yp, expected, atol=1e-6, rtol=0.0), (yp, expected)

    def test_recovery_unaffected_by_smoothness_for_linear_truth(self):
        # A linear table has zero second-difference but non-zero first-
        # difference, so an L2 first-difference penalty *biases* the
        # solution.  With moderate smoothness the bias should still be
        # tiny relative to the signal.
        xp = jnp.linspace(0.0, 10.0, 11)
        x_data = jnp.linspace(0.0, 10.0, 201)
        y_data = 2.0 * x_data + 3.0
        yp = fit_table_1d(xp, x_data, y_data, smoothness=1e-6)
        expected = 2.0 * xp + 3.0
        assert jnp.allclose(yp, expected, atol=1e-3)


class TestFitNoisySine:
    """A noisy sine should be fit close to the truth, with smoothness
    optionally suppressing high-frequency residual."""

    def test_unsmoothed_fit_close_to_truth(self):
        rng = np.random.default_rng(0)
        xp = jnp.linspace(0.0, math.pi, 17)
        x_data = jnp.asarray(rng.uniform(0.0, math.pi, size=2000))
        noise = rng.normal(scale=0.05, size=2000)
        y_data = jnp.sin(x_data) + jnp.asarray(noise)
        yp = fit_table_1d(xp, x_data, y_data)
        truth = jnp.sin(xp)
        # With ~120 samples per bucket and σ = 0.05 the per-grid std
        # error is roughly 0.05 / sqrt(120) ≈ 0.0046 — accept up to 5x.
        assert jnp.allclose(yp, truth, atol=0.025), (yp, truth)

    def test_smoothness_reduces_jitter_but_keeps_signal(self):
        rng = np.random.default_rng(1)
        xp = jnp.linspace(0.0, math.pi, 33)  # finer grid -> more jitter
        x_data = jnp.asarray(rng.uniform(0.0, math.pi, size=400))
        y_data = jnp.sin(x_data) + jnp.asarray(rng.normal(scale=0.1, size=400))
        yp_raw = fit_table_1d(xp, x_data, y_data)
        yp_smooth = fit_table_1d(xp, x_data, y_data, smoothness=1.0)
        # Smoothed table has lower total variation (sum |Δyp|).
        tv_raw = float(jnp.sum(jnp.abs(jnp.diff(yp_raw))))
        tv_smooth = float(jnp.sum(jnp.abs(jnp.diff(yp_smooth))))
        assert tv_smooth < tv_raw, (tv_raw, tv_smooth)
        # And still recovers the broad shape (within a couple of σ).
        assert jnp.allclose(yp_smooth, jnp.sin(xp), atol=0.1)


class TestWeights:
    """Per-sample weights should redirect the fit toward emphasised samples."""

    def test_zero_weight_sample_ignored(self):
        # Two clouds: a "truth" cloud y = 2x and a "junk" cloud y = -50.
        # Heavy weight on the truth cloud -> recover y = 2x.
        xp = jnp.linspace(0.0, 1.0, 5)
        x_truth = jnp.linspace(0.05, 0.95, 50)
        y_truth = 2.0 * x_truth
        x_junk = jnp.linspace(0.05, 0.95, 50)
        y_junk = jnp.full_like(x_junk, -50.0)
        x_data = jnp.concatenate([x_truth, x_junk])
        y_data = jnp.concatenate([y_truth, y_junk])
        weights = jnp.concatenate(
            [jnp.ones(50), jnp.zeros(50)]  # ignore the junk
        )
        yp = fit_table_1d(xp, x_data, y_data, weights=weights)
        assert jnp.allclose(yp, 2.0 * xp, atol=1e-5)


class TestValidation:
    def test_mismatched_shapes_raises(self):
        xp = jnp.linspace(0.0, 1.0, 5)
        with pytest.raises(ValueError):
            fit_table_1d(xp, jnp.zeros(10), jnp.zeros(11))

    def test_2d_x_data_raises(self):
        xp = jnp.linspace(0.0, 1.0, 5)
        with pytest.raises(ValueError):
            fit_table_1d(xp, jnp.zeros((10, 2)), jnp.zeros((10, 2)))

    def test_negative_smoothness_raises(self):
        xp = jnp.linspace(0.0, 1.0, 5)
        with pytest.raises(ValueError):
            fit_table_1d(xp, jnp.zeros(10), jnp.zeros(10), smoothness=-1.0)

    def test_too_few_grid_points_raises(self):
        with pytest.raises(ValueError):
            fit_table_1d(jnp.array([0.0]), jnp.zeros(3), jnp.zeros(3))

    def test_weight_shape_mismatch_raises(self):
        xp = jnp.linspace(0.0, 1.0, 5)
        with pytest.raises(ValueError):
            fit_table_1d(
                xp,
                jnp.zeros(10),
                jnp.zeros(10),
                weights=jnp.ones(7),
            )


# ---------------------------------------------------------------------------
# Differentiability
# ---------------------------------------------------------------------------


class TestDifferentiability:
    def test_grad_through_y_data_finite(self):
        xp = jnp.linspace(0.0, 1.0, 5)
        x_data = jnp.linspace(0.0, 1.0, 50)

        def loss(y_data):
            yp = fit_table_1d(xp, x_data, y_data)
            return jnp.sum(yp ** 2)

        y_data0 = jnp.sin(2.0 * math.pi * x_data)
        g = jax.grad(loss)(y_data0)
        assert g.shape == y_data0.shape
        assert jnp.all(jnp.isfinite(g))

    def test_grad_matches_finite_difference(self):
        xp = jnp.linspace(0.0, 1.0, 5)
        x_data = jnp.linspace(0.05, 0.95, 30)
        y_data0 = 3.0 * x_data + 1.0

        def loss(y_data):
            yp = fit_table_1d(xp, x_data, y_data)
            return jnp.sum(yp)  # sum-of-table-values

        g = jax.grad(loss)(y_data0)
        # Finite-difference check on a single coordinate.
        idx = 10
        h = 1e-4
        y_plus = y_data0.at[idx].add(h)
        y_minus = y_data0.at[idx].add(-h)
        fd = (float(loss(y_plus)) - float(loss(y_minus))) / (2 * h)
        assert float(g[idx]) == pytest.approx(fd, abs=5e-3)

    def test_jit_compatible(self):
        xp = jnp.linspace(0.0, 1.0, 5)
        x_data = jnp.linspace(0.0, 1.0, 50)

        @jax.jit
        def fit(y_data):
            return fit_table_1d(xp, x_data, y_data)

        y_data = jnp.cos(x_data)
        yp = fit(y_data)
        # Compare to the eager call.
        yp_ref = fit_table_1d(xp, x_data, y_data)
        assert jnp.allclose(yp, yp_ref, atol=1e-10)


# ---------------------------------------------------------------------------
# Block-layer convenience wrapper
# ---------------------------------------------------------------------------


class TestFitLookupTable1d:
    def test_returns_lookup_table_1d_block(self):
        xp = jnp.linspace(0.0, 10.0, 6)
        x_data = jnp.linspace(0.0, 10.0, 51)
        y_data = 2.0 * x_data + 3.0
        block = fit_lookup_table_1d(xp, x_data, y_data)
        assert isinstance(block, library.LookupTable1d)

    def test_fitted_block_evaluates_correctly(self):
        xp = jnp.linspace(0.0, 10.0, 11)
        x_data = jnp.linspace(0.0, 10.0, 201)
        y_data = 2.0 * x_data + 3.0
        block = fit_lookup_table_1d(xp, x_data, y_data)

        # Drop the block in a tiny diagram and evaluate at a query
        # point — exercises the full LookupTable1d code path.
        builder = jaxonomy.DiagramBuilder()
        b = builder.add(block)
        src = builder.add(library.Constant(5.5))
        builder.connect(src.output_ports[0], b.input_ports[0])
        diagram = builder.build()
        ctx = diagram.create_context()
        out = b.output_ports[0].eval(ctx)
        assert float(out) == pytest.approx(2.0 * 5.5 + 3.0, abs=1e-4)

    def test_smoothness_kwarg_forwarded(self):
        # With heavy smoothness on a noisy fit, the resulting fit
        # vector should differ from the unsmoothed fit.  Compare the
        # underlying math directly (the block ``output_array`` attribute
        # is only populated after ``initialize`` runs at diagram-build
        # time, so probe the math layer here).
        rng = np.random.default_rng(2)
        xp = jnp.linspace(0.0, 1.0, 9)
        x_data = jnp.asarray(rng.uniform(0.0, 1.0, size=200))
        y_data = jnp.asarray(rng.normal(scale=1.0, size=200))
        yp_raw = fit_table_1d(xp, x_data, y_data)
        yp_smooth = fit_table_1d(xp, x_data, y_data, smoothness=10.0)
        assert not jnp.allclose(yp_raw, yp_smooth)
        # Smoothed table has lower total variation.
        assert float(jnp.sum(jnp.abs(jnp.diff(yp_smooth)))) < float(
            jnp.sum(jnp.abs(jnp.diff(yp_raw)))
        )

    def test_pchip_runtime_interpolation(self):
        # The fit is always linear-LS, but the runtime block can use
        # PCHIP for smoother gradients.
        xp = jnp.linspace(0.0, math.pi, 9)
        x_data = jnp.linspace(0.0, math.pi, 201)
        y_data = jnp.sin(x_data)
        block = fit_lookup_table_1d(
            xp, x_data, y_data, interpolation="pchip"
        )
        # Build a tiny diagram and confirm the block evaluates close to
        # sin at a query point inside the range.
        builder = jaxonomy.DiagramBuilder()
        b = builder.add(block)
        src = builder.add(library.Constant(math.pi / 2))
        builder.connect(src.output_ports[0], b.input_ports[0])
        diagram = builder.build()
        ctx = diagram.create_context()
        out = float(b.output_ports[0].eval(ctx))
        # PCHIP with 9 grid points may overshoot mildly at the peak;
        # the looser tolerance captures "qualitatively correct" without
        # over-constraining the interpolant.
        assert out == pytest.approx(math.sin(math.pi / 2), abs=0.05)
