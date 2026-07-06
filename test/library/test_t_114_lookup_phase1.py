# SPDX-License-Identifier: MIT

"""T-114 phase 1 — lookup-table backend + extrapolation kwarg.

Phase 1 ships:
- ``jaxonomy.library.lookup_table.interp_1d`` (linear + PCHIP, with
  ``extrapolation in {"clip","linear","nan"}``).
- ``even_spacing`` auto-detector.
- ``LookupTable1d(extrapolation=...)`` kwarg + ``"pchip"`` interpolation
  routed through the new backend.

Default-path constraint: when ``LookupTable1d`` is built without
``extrapolation=`` (i.e. the kwarg defaults to ``"clip"``) and
``interpolation`` is one of the original three, the block uses the
historical ``jnp.interp`` / ``argmin`` / ``where`` path verbatim — the
existing ``test/library/test_lookup_tables.py`` suite enforces this
byte-equivalence.
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.interpolate import PchipInterpolator

import jaxonomy
from jaxonomy import library
from jaxonomy.library.lookup_table import (
    even_spacing,
    interp_1d,
    pchip_slopes,
)


# ---------------------------------------------------------------------------
# Pure-functional backend
# ---------------------------------------------------------------------------


class TestInterp1dLinear:
    """Linear interpolation through a known function (sin)."""

    def test_matches_analytic_sin_at_breakpoints(self):
        xp = jnp.linspace(0.0, math.pi, 65)
        fp = jnp.sin(xp)
        # Query at the breakpoints themselves — interp must be exact.
        ys = interp_1d(xp, xp, fp, method="linear")
        assert jnp.allclose(ys, fp, atol=1e-12)

    def test_dense_grid_close_to_sin(self):
        xp = jnp.linspace(0.0, math.pi, 257)
        fp = jnp.sin(xp)
        xs = jnp.linspace(0.05, math.pi - 0.05, 33)
        ys = interp_1d(xs, xp, fp, method="linear")
        assert jnp.allclose(ys, jnp.sin(xs), atol=5e-4)


class TestInterp1dPchip:
    """PCHIP must match SciPy's PchipInterpolator inside the breakpoint range."""

    def test_pchip_matches_scipy(self):
        xp = jnp.array([0.0, 1.0, 2.0, 3.0, 4.0])
        fp = jnp.array([0.0, 1.0, 4.0, 9.0, 16.0])
        scipy_p = PchipInterpolator(np.asarray(xp), np.asarray(fp), extrapolate=False)
        xs = jnp.array([0.1, 0.5, 1.5, 2.5, 3.5, 3.9])
        got = interp_1d(xs, xp, fp, method="pchip")
        want = scipy_p(np.asarray(xs))
        assert jnp.allclose(got, want, atol=1e-10), (got, want)

    def test_pchip_monotone_on_monotone_data(self):
        xp = jnp.linspace(0.0, 5.0, 11)
        fp = jnp.array([0.0, 0.1, 0.3, 0.7, 1.0, 1.05, 1.07, 1.08, 1.085, 1.087, 1.09])
        xs = jnp.linspace(0.0, 5.0, 401)
        ys = interp_1d(xs, xp, fp, method="pchip")
        # Strictly non-decreasing (within float noise).
        assert jnp.all(jnp.diff(ys) >= -1e-12)


class TestExtrapolation:
    """Out-of-bounds policy."""

    @pytest.fixture
    def fix(self):
        xp = jnp.array([0.0, 1.0, 2.0, 3.0, 4.0])
        fp = jnp.array([0.0, 1.0, 4.0, 9.0, 16.0])
        return xp, fp

    def test_clip_holds_boundary(self, fix):
        xp, fp = fix
        assert float(interp_1d(-1.5, xp, fp, extrapolation="clip")) == pytest.approx(0.0)
        assert float(interp_1d(99.0, xp, fp, extrapolation="clip")) == pytest.approx(16.0)

    def test_linear_extends_slope(self, fix):
        xp, fp = fix
        # Left slope = (1 - 0) / (1 - 0) = 1; at x = -2, expect 0 + 1*(-2) = -2.
        got_left = float(interp_1d(-2.0, xp, fp, extrapolation="linear"))
        assert got_left == pytest.approx(-2.0)
        # Right slope = (16 - 9) / (4 - 3) = 7; at x = 5, expect 16 + 7 = 23.
        got_right = float(interp_1d(5.0, xp, fp, extrapolation="linear"))
        assert got_right == pytest.approx(23.0)

    def test_nan_returns_nan(self, fix):
        xp, fp = fix
        assert math.isnan(float(interp_1d(-0.1, xp, fp, extrapolation="nan")))
        assert math.isnan(float(interp_1d(4.1, xp, fp, extrapolation="nan")))
        # In-range still finite.
        assert not math.isnan(float(interp_1d(2.5, xp, fp, extrapolation="nan")))

    def test_unknown_extrap_raises(self, fix):
        xp, fp = fix
        with pytest.raises(ValueError):
            interp_1d(0.5, xp, fp, extrapolation="bogus")


class TestEvenSpacing:
    def test_even(self):
        ok, dx = even_spacing(jnp.array([0.0, 0.5, 1.0, 1.5, 2.0]))
        assert ok and dx == pytest.approx(0.5)

    def test_uneven(self):
        ok, dx = even_spacing(jnp.array([0.0, 0.5, 1.4, 2.0]))
        assert not ok and dx is None

    def test_too_short(self):
        ok, dx = even_spacing(jnp.array([1.0]))
        assert not ok and dx is None


class TestPchipSlopesShape:
    def test_slope_count(self):
        xp = jnp.linspace(0.0, 1.0, 6)
        fp = jnp.sin(xp)
        slopes = pchip_slopes(xp, fp)
        assert slopes.shape == (6,)


# ---------------------------------------------------------------------------
# Differentiability
# ---------------------------------------------------------------------------


class TestDifferentiability:
    def test_grad_through_linear(self):
        xp = jnp.array([0.0, 1.0, 2.0, 3.0, 4.0])
        fp = jnp.array([0.0, 1.0, 4.0, 9.0, 16.0])

        def loss(x):
            return interp_1d(x, xp, fp, method="linear")

        # Inside interval [2, 3] — slope = (9 - 4) / 1 = 5.
        g = jax.grad(loss)(2.5)
        assert float(g) == pytest.approx(5.0)

    def test_grad_through_pchip(self):
        xp = jnp.linspace(0.0, math.pi, 33)
        fp = jnp.sin(xp)

        def loss(x):
            return interp_1d(x, xp, fp, method="pchip")

        # Compare to finite difference at x = pi/4.
        x0 = math.pi / 4
        g = float(jax.grad(loss)(x0))
        h = 1e-4
        fd = (float(loss(x0 + h)) - float(loss(x0 - h))) / (2 * h)
        assert abs(g - fd) < 1e-5, (g, fd)

    def test_grad_through_table_values(self):
        """Differentiate w.r.t. the table values themselves (the T-124 use
        case: gradient-based table fitting)."""
        xp = jnp.array([0.0, 1.0, 2.0, 3.0, 4.0])

        def loss(fp):
            return interp_1d(jnp.array([0.5, 1.5, 2.5, 3.5]), xp, fp, method="linear").sum()

        fp0 = jnp.array([0.0, 1.0, 4.0, 9.0, 16.0])
        g = jax.grad(loss)(fp0)
        # Each query at the midpoint contributes 0.5 to the two flanking
        # entries; sum-of-gradients per entry is therefore well-defined.
        assert g.shape == fp0.shape
        assert jnp.all(jnp.isfinite(g))


# ---------------------------------------------------------------------------
# Block-level wiring (LookupTable1d extrapolation kwarg + pchip)
# ---------------------------------------------------------------------------


def _eval_block(input_array, output_array, interpolation, x, **kwargs):
    builder = jaxonomy.DiagramBuilder()
    blk = builder.add(
        library.LookupTable1d(input_array, output_array, interpolation, **kwargs)
    )
    src = builder.add(library.Constant(x))
    builder.connect(src.output_ports[0], blk.input_ports[0])
    diagram = builder.build()
    ctx = diagram.create_context()
    return blk.output_ports[0].eval(ctx)


class TestLookupTable1dExtrapolation:
    @pytest.fixture
    def arrays(self):
        return jnp.array([0.0, 1.0, 2.0, 3.0, 4.0]), jnp.array([0.0, 1.0, 4.0, 9.0, 16.0])

    def test_default_clip_byte_equivalent(self, arrays):
        xp, fp = arrays
        # No extrapolation kwarg = "clip" = jnp.interp default.
        out = _eval_block(xp, fp, "linear", -2.0)
        assert float(out) == pytest.approx(0.0)
        out = _eval_block(xp, fp, "linear", 99.0)
        assert float(out) == pytest.approx(16.0)

    def test_linear_extrapolation(self, arrays):
        xp, fp = arrays
        out = _eval_block(xp, fp, "linear", -2.0, extrapolation="linear")
        assert float(out) == pytest.approx(-2.0)
        out = _eval_block(xp, fp, "linear", 5.0, extrapolation="linear")
        assert float(out) == pytest.approx(23.0)

    def test_pchip_block(self, arrays):
        xp, fp = arrays
        out = _eval_block(xp, fp, "pchip", 2.5)
        scipy_p = PchipInterpolator(np.asarray(xp), np.asarray(fp))
        assert float(out) == pytest.approx(float(scipy_p(2.5)), abs=1e-9)

    def test_invalid_extrapolation_rejected(self, arrays):
        xp, fp = arrays
        with pytest.raises(ValueError):
            library.LookupTable1d(xp, fp, "linear", extrapolation="bogus")
