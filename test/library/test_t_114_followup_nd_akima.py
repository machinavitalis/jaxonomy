# SPDX-License-Identifier: MIT

"""T-114 phase 2 — 2-D extrapolation kwarg + Akima 1-D interpolation.

Phase 2 ships:
- ``jaxonomy.library.lookup_table.interp_2d`` (bilinear, with
  ``extrapolation in {"clip","linear","nan"}``).
- ``LookupTable2d(extrapolation=...)`` kwarg routed through the new
  backend (default ``"clip"`` is byte-equivalent to the legacy
  ``npa.interp2d`` path).
- Akima 1970 interpolation in ``interp_1d(method="akima")`` —
  matches :class:`scipy.interpolate.Akima1DInterpolator`.

Default-path constraint: when ``LookupTable2d`` is built without
``extrapolation=`` (i.e. the kwarg defaults to ``"clip"``), the block
uses the historical ``npa.interp2d`` path verbatim — the existing
``test/library/test_lookup_tables.py`` suite enforces this byte-
equivalence.
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.interpolate import Akima1DInterpolator

import jaxonomy
from jaxonomy import library
from jaxonomy.library.lookup_table import (
    akima_slopes,
    interp_1d,
    interp_2d,
)


# ---------------------------------------------------------------------------
# 2-D pure-functional backend
# ---------------------------------------------------------------------------


class TestInterp2dLinear:
    """Bilinear interpolation through f(x, y) = x * y."""

    @pytest.fixture
    def grid(self):
        xp = jnp.array([0.0, 1.0, 2.0, 3.0])
        yp = jnp.array([0.0, 1.0, 2.0])
        zp = xp[:, None] * yp[None, :]  # f(x, y) = x * y
        return xp, yp, zp

    def test_at_breakpoints_exact(self, grid):
        xp, yp, zp = grid
        # Query at every breakpoint corner — bilinear must be exact.
        for i, x in enumerate(xp):
            for j, y in enumerate(yp):
                got = float(interp_2d(x, y, xp, yp, zp))
                assert got == pytest.approx(float(zp[i, j]), abs=1e-12)

    def test_interior_matches_xy(self, grid):
        xp, yp, zp = grid
        # f is exactly bilinear (it's x*y on a uniform grid), so any
        # interior query should be exact.
        xs = jnp.array([0.5, 1.25, 2.7])
        ys = jnp.array([0.3, 1.5, 0.8])
        got = jax.vmap(lambda x, y: interp_2d(x, y, xp, yp, zp))(xs, ys)
        assert jnp.allclose(got, xs * ys, atol=1e-12)


class TestInterp2dExtrapolation:
    @pytest.fixture
    def grid(self):
        xp = jnp.array([0.0, 1.0, 2.0])
        yp = jnp.array([0.0, 1.0, 2.0])
        zp = xp[:, None] * yp[None, :]  # f(x, y) = x * y
        return xp, yp, zp

    def test_clip_holds_boundary(self, grid):
        xp, yp, zp = grid
        # OOB-x left: clipped to (0, y), so f(-1, 0.5) -> f(0, 0.5) = 0.
        got = float(interp_2d(-1.0, 0.5, xp, yp, zp, extrapolation="clip"))
        assert got == pytest.approx(0.0, abs=1e-12)
        # OOB-x right + OOB-y right: f(5, 5) -> f(2, 2) = 4.
        got = float(interp_2d(5.0, 5.0, xp, yp, zp, extrapolation="clip"))
        assert got == pytest.approx(4.0, abs=1e-12)

    def test_linear_extends_bilinear(self, grid):
        xp, yp, zp = grid
        # f(x, y) = x*y is itself bilinear, so the bilinear extrapolation
        # past the grid edges must reproduce x*y exactly even for OOB.
        for x, y in [(-0.5, 0.7), (3.0, 1.5), (2.5, -0.4), (4.0, 4.0)]:
            got = float(
                interp_2d(x, y, xp, yp, zp, extrapolation="linear")
            )
            assert got == pytest.approx(x * y, abs=1e-12), (x, y, got)

    def test_nan_outside_grid(self, grid):
        xp, yp, zp = grid
        # Both inputs OOB -> NaN.
        got = float(interp_2d(-1.0, 5.0, xp, yp, zp, extrapolation="nan"))
        assert math.isnan(got)
        # One input OOB -> NaN.
        got = float(interp_2d(0.5, 5.0, xp, yp, zp, extrapolation="nan"))
        assert math.isnan(got)
        got = float(interp_2d(-1.0, 0.5, xp, yp, zp, extrapolation="nan"))
        assert math.isnan(got)
        # In-range -> finite.
        got = float(interp_2d(0.5, 0.5, xp, yp, zp, extrapolation="nan"))
        assert math.isfinite(got)

    def test_unknown_method_raises(self, grid):
        xp, yp, zp = grid
        with pytest.raises(ValueError):
            interp_2d(0.5, 0.5, xp, yp, zp, method="bogus")

    def test_unknown_extrap_raises(self, grid):
        xp, yp, zp = grid
        with pytest.raises(ValueError):
            interp_2d(0.5, 0.5, xp, yp, zp, extrapolation="bogus")


class TestInterp2dDifferentiability:
    def test_grad_through_query_point(self):
        xp = jnp.array([0.0, 1.0, 2.0])
        yp = jnp.array([0.0, 1.0, 2.0])
        zp = xp[:, None] * yp[None, :]

        def loss(x):
            return interp_2d(x, jnp.array(0.5), xp, yp, zp)

        # df/dx at (1.5, 0.5) for f(x, y) = x*y is y = 0.5.
        g = float(jax.grad(loss)(jnp.array(1.5)))
        assert g == pytest.approx(0.5, abs=1e-6)

    def test_grad_through_table_values(self):
        xp = jnp.array([0.0, 1.0, 2.0])
        yp = jnp.array([0.0, 1.0, 2.0])

        def loss(zp):
            return interp_2d(
                jnp.array(0.5), jnp.array(0.5), xp, yp, zp
            )

        zp0 = xp[:, None] * yp[None, :]
        g = jax.grad(loss)(zp0)
        # Bilinear at (0.5, 0.5) is the average of the four corner
        # values of cell [0,0]; gradient w.r.t. each corner is 0.25.
        assert g.shape == zp0.shape
        assert float(g[0, 0]) == pytest.approx(0.25, abs=1e-12)
        assert float(g[1, 1]) == pytest.approx(0.25, abs=1e-12)


# ---------------------------------------------------------------------------
# Block-level wiring (LookupTable2d extrapolation kwarg)
# ---------------------------------------------------------------------------


def _eval_2d_block(xp, yp, zp, x, y, **kwargs):
    builder = jaxonomy.DiagramBuilder()
    blk = builder.add(library.LookupTable2d(xp, yp, zp, "linear", **kwargs))
    src_x = builder.add(library.Constant(x))
    src_y = builder.add(library.Constant(y))
    builder.connect(src_x.output_ports[0], blk.input_ports[0])
    builder.connect(src_y.output_ports[0], blk.input_ports[1])
    diagram = builder.build()
    ctx = diagram.create_context()
    return blk.output_ports[0].eval(ctx)


class TestLookupTable2dExtrapolation:
    @pytest.fixture
    def arrays(self):
        xp = jnp.array([0.0, 1.0, 2.0, 3.0])
        yp = jnp.array([0.0, 1.0, 2.0])
        zp = xp[:, None] * yp[None, :]
        return xp, yp, zp

    def test_default_clip_byte_equivalent(self, arrays):
        xp, yp, zp = arrays
        # OOB-x left: clip to boundary, f(0, 0.5) = 0.
        got = float(_eval_2d_block(xp, yp, zp, -1.0, 0.5))
        assert got == pytest.approx(0.0, abs=1e-12)
        # OOB-x right + OOB-y right: clip to (3, 2), f = 6.
        got = float(_eval_2d_block(xp, yp, zp, 5.0, 5.0))
        assert got == pytest.approx(6.0, abs=1e-12)

    def test_linear_extrapolation(self, arrays):
        xp, yp, zp = arrays
        # f(x, y) = x*y is exactly bilinear, so linear extrap must
        # reproduce x*y for OOB queries.
        got = float(
            _eval_2d_block(xp, yp, zp, -0.5, 0.5, extrapolation="linear")
        )
        assert got == pytest.approx(-0.25, abs=1e-12)
        got = float(
            _eval_2d_block(xp, yp, zp, 4.0, 3.0, extrapolation="linear")
        )
        assert got == pytest.approx(12.0, abs=1e-12)

    def test_nan_extrapolation(self, arrays):
        xp, yp, zp = arrays
        got = float(
            _eval_2d_block(xp, yp, zp, -1.0, 0.5, extrapolation="nan")
        )
        assert math.isnan(got)
        got = float(
            _eval_2d_block(xp, yp, zp, 0.5, 0.5, extrapolation="nan")
        )
        assert math.isfinite(got)

    def test_invalid_extrapolation_rejected(self, arrays):
        xp, yp, zp = arrays
        with pytest.raises(ValueError):
            library.LookupTable2d(xp, yp, zp, "linear", extrapolation="bogus")


# ---------------------------------------------------------------------------
# Akima 1-D
# ---------------------------------------------------------------------------


class TestAkimaSlopes:
    def test_slope_count(self):
        xp = jnp.linspace(0.0, 1.0, 7)
        fp = jnp.sin(xp)
        slopes = akima_slopes(xp, fp)
        assert slopes.shape == (7,)


class TestInterp1dAkima:
    def test_akima_matches_scipy_smooth(self):
        # Smooth function: should agree with SciPy's Akima to <1e-6.
        xp = jnp.linspace(0.0, math.pi, 17)
        fp = jnp.sin(xp)
        scipy_a = Akima1DInterpolator(np.asarray(xp), np.asarray(fp))
        xs = jnp.linspace(0.05, math.pi - 0.05, 41)
        got = interp_1d(xs, xp, fp, method="akima")
        want = scipy_a(np.asarray(xs))
        assert jnp.allclose(got, want, atol=1e-6), (
            float(jnp.max(jnp.abs(got - want))),
        )

    def test_akima_matches_scipy_kinky(self):
        # Non-smooth (piecewise-linear) data: Akima's reflection-formula
        # boundary slopes are well-defined; SciPy is the reference.
        xp = jnp.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        fp = jnp.array([0.0, 1.0, 4.0, 9.0, 4.0, 1.0, 0.0])
        scipy_a = Akima1DInterpolator(np.asarray(xp), np.asarray(fp))
        xs = jnp.array([0.3, 1.5, 2.7, 3.5, 4.2, 5.5])
        got = interp_1d(xs, xp, fp, method="akima")
        want = scipy_a(np.asarray(xs))
        assert jnp.allclose(got, want, atol=1e-10), (got, want)

    def test_akima_passes_through_breakpoints(self):
        xp = jnp.linspace(0.0, 4.0, 9)
        fp = jnp.cos(xp)
        got = interp_1d(xp, xp, fp, method="akima")
        assert jnp.allclose(got, fp, atol=1e-10)

    def test_akima_extrapolation_clip(self):
        xp = jnp.array([0.0, 1.0, 2.0, 3.0, 4.0])
        fp = jnp.array([0.0, 1.0, 4.0, 9.0, 16.0])
        # Default "clip" — boundary value, NOT cubic continuation.
        assert float(interp_1d(-1.0, xp, fp, method="akima")) == pytest.approx(0.0)
        assert float(interp_1d(5.0, xp, fp, method="akima")) == pytest.approx(16.0)

    def test_akima_grad_through_query(self):
        xp = jnp.linspace(0.0, math.pi, 17)
        fp = jnp.sin(xp)

        def loss(x):
            return interp_1d(x, xp, fp, method="akima")

        x0 = math.pi / 4
        g = float(jax.grad(loss)(jnp.asarray(x0)))
        # Compare to finite difference at x0.
        h = 1e-4
        fd = (
            float(loss(jnp.asarray(x0 + h))) - float(loss(jnp.asarray(x0 - h)))
        ) / (2 * h)
        assert abs(g - fd) < 1e-4, (g, fd)


class TestLookupTable1dAkimaBlock:
    def test_block_akima(self):
        xp = jnp.linspace(0.0, math.pi, 17)
        fp = jnp.sin(xp)
        builder = jaxonomy.DiagramBuilder()
        blk = builder.add(library.LookupTable1d(xp, fp, "akima"))
        src = builder.add(library.Constant(jnp.asarray(1.234)))
        builder.connect(src.output_ports[0], blk.input_ports[0])
        diagram = builder.build()
        ctx = diagram.create_context()
        got = float(blk.output_ports[0].eval(ctx))
        scipy_a = Akima1DInterpolator(np.asarray(xp), np.asarray(fp))
        assert got == pytest.approx(float(scipy_a(1.234)), abs=1e-6)
