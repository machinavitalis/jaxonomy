# SPDX-License-Identifier: MIT

"""T-114-followup-natural-cubic-spline — natural cubic spline interpolation.

Ships:
- ``jaxonomy.library.lookup_table.interp_1d(..., method="cubic")`` —
  natural cubic spline (C^2-continuous, second derivative zero at the
  boundaries).  Matches
  ``scipy.interpolate.CubicSpline(bc_type='natural')`` to high accuracy.
- ``LookupTable1d(interpolation="cubic")`` — block-level wiring.

Default-path constraint: when ``LookupTable1d`` is built with the
defaults (``interpolation="linear"``, ``extrapolation="clip"``), the
block routes through the legacy ``npa.interp`` fast path verbatim — the
test below confirms the new ``"cubic"`` mode does not alter the default
path.
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import jaxonomy
from jaxonomy import library
from jaxonomy.backend import numpy_api as npa
from jaxonomy.library.lookup_table import (
    interp_1d,
    natural_cubic_second_derivs,
)


# ---------------------------------------------------------------------------
# Pure-functional backend — interp_1d(method="cubic")
# ---------------------------------------------------------------------------


class TestInterp1dCubicReproducesSmoothFunctions:
    """Natural cubic spline interpolates ``sin`` to high accuracy off-grid."""

    def test_exact_at_breakpoints(self):
        xp = jnp.linspace(0.0, math.pi, 9)
        fp = jnp.sin(xp)
        ys = interp_1d(xp, xp, fp, method="cubic")
        # Natural cubic spline interpolates the data exactly at the knots.
        assert jnp.allclose(ys, fp, atol=1e-12)

    def test_dense_grid_high_accuracy_off_grid(self):
        # 17-point grid over [0, pi].  Natural cubic spline error on
        # smooth sinusoids should be O(h^4) ~ a few 1e-6 on this grid.
        xp = jnp.linspace(0.0, math.pi, 17)
        fp = jnp.sin(xp)
        # Query at points strictly between breakpoints.
        xs = jnp.linspace(0.05, math.pi - 0.05, 31)
        ys = interp_1d(xs, xp, fp, method="cubic")
        truth = jnp.sin(xs)
        err = float(jnp.max(jnp.abs(ys - truth)))
        # On a 17-point grid the cubic spline is comfortably below 1e-4.
        assert err < 1e-4, f"cubic spline error {err:.3e} too large"

    def test_more_accurate_than_linear_on_smooth(self):
        xp = jnp.linspace(0.0, math.pi, 17)
        fp = jnp.sin(xp)
        xs = jnp.linspace(0.05, math.pi - 0.05, 31)
        truth = jnp.sin(xs)
        err_lin = float(jnp.max(jnp.abs(interp_1d(xs, xp, fp, method="linear") - truth)))
        err_cub = float(jnp.max(jnp.abs(interp_1d(xs, xp, fp, method="cubic") - truth)))
        assert err_cub < err_lin
        # On smooth data the cubic should be at least an order of
        # magnitude better than linear at this resolution.
        assert err_cub * 10.0 < err_lin

    def test_matches_scipy_cubic_spline_natural(self):
        # Cross-check against scipy CubicSpline with bc_type='natural'.
        try:
            from scipy.interpolate import CubicSpline
        except ImportError:
            pytest.skip("scipy not available")

        xp_np = np.linspace(0.0, 5.0, 11)
        fp_np = np.sin(xp_np) + 0.3 * np.cos(2.0 * xp_np)
        cs = CubicSpline(xp_np, fp_np, bc_type="natural")

        xs_np = np.linspace(0.1, 4.9, 41)
        scipy_vals = cs(xs_np)

        xp = jnp.asarray(xp_np)
        fp = jnp.asarray(fp_np)
        xs = jnp.asarray(xs_np)
        ours = np.asarray(interp_1d(xs, xp, fp, method="cubic"))

        # The two implementations should agree to floating-point noise.
        assert np.allclose(ours, scipy_vals, atol=1e-9, rtol=1e-9), (
            f"max diff = {np.max(np.abs(ours - scipy_vals)):.3e}"
        )


class TestNaturalCubicSplineSmoothness:
    """C^2 continuity: second derivative is continuous at every interior knot."""

    def test_natural_bc_zero_second_deriv_at_boundaries(self):
        # Direct check on the second-derivative computation: the natural
        # boundary condition pins M[0] = M[N-1] = 0.
        xp = jnp.linspace(0.0, 2.0 * math.pi, 13)
        fp = jnp.sin(xp)
        M = natural_cubic_second_derivs(xp, fp)
        assert float(M[0]) == pytest.approx(0.0, abs=1e-12)
        assert float(M[-1]) == pytest.approx(0.0, abs=1e-12)

    def test_second_derivative_continuous_at_interior_knots(self):
        """Compare second-derivative limits from left and right at each knot."""
        xp = jnp.linspace(0.0, 2.0 * math.pi, 9)
        fp = jnp.sin(xp)

        def f(x):
            return interp_1d(x, xp, fp, method="cubic")

        # Second derivative via two nested ``jax.grad`` calls.
        d2 = jax.grad(jax.grad(f))

        # Probe symmetrically on either side of each interior knot at
        # tiny offsets — the two limits must agree.
        eps = 1e-4
        for i in range(1, xp.shape[0] - 1):
            xi = float(xp[i])
            left = float(d2(xi - eps))
            right = float(d2(xi + eps))
            # Within numerical tolerance of the finite-difference probe.
            assert left == pytest.approx(right, abs=1e-3), (
                f"second derivative discontinuous at knot {i} (x={xi}): "
                f"left={left}, right={right}"
            )

    def test_first_derivative_continuous_at_interior_knots(self):
        """C^1 continuity is implied by the construction; sanity-check it."""
        xp = jnp.linspace(0.0, 2.0 * math.pi, 9)
        fp = jnp.sin(xp)

        def f(x):
            return interp_1d(x, xp, fp, method="cubic")

        d1 = jax.grad(f)

        eps = 1e-5
        for i in range(1, xp.shape[0] - 1):
            xi = float(xp[i])
            left = float(d1(xi - eps))
            right = float(d1(xi + eps))
            assert left == pytest.approx(right, abs=1e-4), (
                f"first derivative discontinuous at knot {i}: "
                f"left={left}, right={right}"
            )


class TestNaturalCubicSplineDifferentiability:
    """jax.grad flows through both the query and the table values."""

    def test_grad_through_query(self):
        xp = jnp.linspace(0.0, 2.0 * math.pi, 13)
        fp = jnp.sin(xp)

        def f(x):
            return interp_1d(x, xp, fp, method="cubic")

        x_q = jnp.asarray(1.7)
        g = float(jax.grad(f)(x_q))
        # Analytic derivative cos(1.7).
        truth = float(jnp.cos(x_q))
        assert g == pytest.approx(truth, abs=1e-2), (
            f"d/dx interp_cubic at 1.7 = {g}, expected ~{truth}"
        )
        assert math.isfinite(g)

    def test_grad_through_table_values(self):
        xp = jnp.linspace(0.0, 2.0 * math.pi, 9)
        fp = jnp.sin(xp)

        x_q = jnp.asarray(1.7)

        def loss(fp_in):
            return interp_1d(x_q, xp, fp_in, method="cubic")

        g_fp = jax.grad(loss)(fp)
        # The output is a linear combination of fp (the spline coefficients
        # depend linearly on fp through the tridiagonal solve), so the
        # gradient sums to 1 (partition of unity for any interpolant that
        # reproduces constants).
        assert jnp.all(jnp.isfinite(g_fp))
        weight_sum = float(jnp.sum(g_fp))
        assert weight_sum == pytest.approx(1.0, abs=1e-9)

    def test_jit_compatible(self):
        xp = jnp.linspace(0.0, 2.0 * math.pi, 9)
        fp = jnp.sin(xp)

        @jax.jit
        def f(x):
            return interp_1d(x, xp, fp, method="cubic")

        x_q = jnp.asarray(1.7)
        out = float(f(x_q))
        assert math.isfinite(out)


class TestNaturalCubicSplineExtrapolation:
    """OOB policies (clip / linear / nan) all behave sensibly with cubic."""

    def test_clip_holds_endpoint_value(self):
        xp = jnp.linspace(0.0, 1.0, 5)
        fp = jnp.sin(xp)
        # Below xp[0] should clamp to fp[0] under the default 'clip' policy.
        below = float(interp_1d(jnp.asarray(-0.5), xp, fp, method="cubic"))
        assert below == pytest.approx(float(fp[0]), abs=1e-12)
        above = float(interp_1d(jnp.asarray(1.5), xp, fp, method="cubic"))
        assert above == pytest.approx(float(fp[-1]), abs=1e-12)

    def test_nan_outside(self):
        xp = jnp.linspace(0.0, 1.0, 5)
        fp = jnp.sin(xp)
        below = float(
            interp_1d(jnp.asarray(-0.5), xp, fp, method="cubic", extrapolation="nan")
        )
        above = float(
            interp_1d(jnp.asarray(1.5), xp, fp, method="cubic", extrapolation="nan")
        )
        assert math.isnan(below)
        assert math.isnan(above)


class TestNaturalCubicSplineValidation:
    def test_too_few_points_raises_in_interp_1d(self):
        # 3 points is below the minimum for natural cubic spline.
        xp = jnp.array([0.0, 1.0, 2.0])
        fp = jnp.array([0.0, 1.0, 0.0])
        with pytest.raises(ValueError, match="at least 4 breakpoints"):
            interp_1d(jnp.asarray(0.5), xp, fp, method="cubic")

    def test_too_few_points_raises_in_helper(self):
        xp = jnp.array([0.0, 1.0, 2.0])
        fp = jnp.array([0.0, 1.0, 0.0])
        with pytest.raises(ValueError, match="at least 4 breakpoints"):
            natural_cubic_second_derivs(xp, fp)

    def test_two_points_raises(self):
        xp = jnp.array([0.0, 1.0])
        fp = jnp.array([0.0, 1.0])
        with pytest.raises(ValueError, match="at least 4 breakpoints"):
            interp_1d(jnp.asarray(0.5), xp, fp, method="cubic")


# ---------------------------------------------------------------------------
# Block-level — LookupTable1d(interpolation="cubic")
# ---------------------------------------------------------------------------


class TestLookupTable1dCubicBlock:
    @pytest.fixture
    def grid_and_values(self):
        xp = jnp.linspace(0.0, 2.0 * math.pi, 13)
        fp = jnp.sin(xp)
        return xp, fp

    @staticmethod
    def _evaluate(xp, fp, x_q, interpolation):
        builder = jaxonomy.DiagramBuilder()
        lookup = builder.add(
            library.LookupTable1d(xp, fp, interpolation)
        )
        src = builder.add(library.Constant(x_q))
        builder.connect(src.output_ports[0], lookup.input_ports[0])
        diagram = builder.build()
        ctx = diagram.create_context()
        return float(lookup.output_ports[0].eval(ctx))

    def test_block_cubic_matches_backend(self, grid_and_values):
        xp, fp = grid_and_values
        x_q = 1.7
        got = self._evaluate(xp, fp, x_q, "cubic")
        want = float(interp_1d(jnp.asarray(x_q), xp, fp, method="cubic"))
        assert got == pytest.approx(want, abs=1e-12)

    def test_block_cubic_exact_at_breakpoint(self, grid_and_values):
        xp, fp = grid_and_values
        x_q = float(xp[5])
        got = self._evaluate(xp, fp, x_q, "cubic")
        assert got == pytest.approx(float(fp[5]), abs=1e-10)

    def test_block_default_linear_byte_equivalent(self, grid_and_values):
        # The default ``interpolation="linear"`` + ``extrapolation="clip"``
        # path must remain byte-equivalent to the legacy ``npa.interp``
        # call — adding the ``"cubic"`` method must not perturb the
        # default path.
        xp, fp = grid_and_values
        x_q = 1.7
        got = self._evaluate(xp, fp, x_q, "linear")
        want = float(npa.interp(jnp.asarray(x_q), xp, fp))
        assert got == pytest.approx(want, abs=0.0)

    def test_block_cubic_too_few_points_raises(self):
        # 3-point block must reject ``interpolation="cubic"`` loudly.  The
        # error fires when the context is created (initialize) — wrapped
        # in StaticError by the framework.
        from jaxonomy.framework.error import StaticError

        xp = jnp.array([0.0, 1.0, 2.0])
        fp = jnp.array([0.0, 1.0, 0.0])
        with pytest.raises((ValueError, StaticError)):
            block = library.LookupTable1d(xp, fp, "cubic")
            block.create_context()


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
