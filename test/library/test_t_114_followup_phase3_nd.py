# SPDX-License-Identifier: MIT

"""T-114-followup-phase3-2d-cubic — N-D lookup-table block + backend.

Phase 3 ships:
- ``jaxonomy.library.lookup_table.interp_nd`` (N-D multilinear, with
  ``extrapolation in {"clip","linear","nan"}``).  Implemented as N
  successive 1-D linear interpolations under the hood because JAX has
  no ``jnp.interpn`` today.
- ``jaxonomy.library.LookupTableND`` block — mirrors the
  ``LookupTable1d`` / ``LookupTable2d`` API for arbitrary N.

The default ``extrapolation="clip"`` matches the standard lookup-table convention.
The block is differentiable through both the query vector and the
table values (modulo discrete bucket-index ``searchsorted`` boundaries).
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import jaxonomy
from jaxonomy import library
from jaxonomy.library.lookup_table import interp_nd


# ---------------------------------------------------------------------------
# Pure-functional backend — interp_nd
# ---------------------------------------------------------------------------


class TestInterpNd3DAtGridPoints:
    """3-D linear interp on f(x, y, z) = x * y * z over [0, 1]^3."""

    @pytest.fixture
    def grid_and_values(self):
        xp = jnp.linspace(0.0, 1.0, 5)
        yp = jnp.linspace(0.0, 1.0, 4)
        zp = jnp.linspace(0.0, 1.0, 6)
        X, Y, Z = jnp.meshgrid(xp, yp, zp, indexing="ij")
        values = X * Y * Z
        return (xp, yp, zp), values

    def test_at_every_grid_corner(self, grid_and_values):
        (xp, yp, zp), values = grid_and_values
        # At every breakpoint corner the multilinear must be exact.
        for i, x in enumerate(xp):
            for j, y in enumerate(yp):
                for k, z in enumerate(zp):
                    q = jnp.array([x, y, z])
                    got = float(interp_nd((xp, yp, zp), values, q))
                    want = float(values[i, j, k])
                    assert got == pytest.approx(want, abs=1e-12), (
                        i, j, k, got, want
                    )

    def test_off_grid_trilinear_exact(self, grid_and_values):
        # f(x,y,z) = x*y*z is a tensor product of monomials, but on a
        # uniform grid each cell evaluates the multilinear interpolant
        # of x*y*z, which is NOT identically x*y*z (only at corners).
        # Use a finer grid + a function that IS multilinear (ax + by +
        # cz + d xy + ...) for the off-grid exact check.
        xp = jnp.array([0.0, 1.0])
        yp = jnp.array([0.0, 1.0])
        zp = jnp.array([0.0, 1.0])
        X, Y, Z = jnp.meshgrid(xp, yp, zp, indexing="ij")
        # f is multilinear (sum of 1, x, y, z, xy, xz, yz, xyz monomials
        # restricted to [0,1] — multilinear interp through the corners
        # IS the function itself).
        values = 1.0 + 2.0 * X + 3.0 * Y - 0.5 * Z + X * Y - 0.7 * Y * Z
        for q in [
            jnp.array([0.3, 0.4, 0.6]),
            jnp.array([0.0, 0.5, 1.0]),
            jnp.array([0.7, 0.7, 0.2]),
        ]:
            x, y, z = float(q[0]), float(q[1]), float(q[2])
            want = 1.0 + 2.0 * x + 3.0 * y - 0.5 * z + x * y - 0.7 * y * z
            got = float(interp_nd((xp, yp, zp), values, q))
            assert got == pytest.approx(want, abs=1e-12), (q, got, want)


class TestInterpNdAgainstScipy:
    """Cross-check interp_nd against scipy.RegularGridInterpolator."""

    def test_off_grid_matches_scipy(self):
        from scipy.interpolate import RegularGridInterpolator

        xp = jnp.linspace(0.0, 2.0, 7)
        yp = jnp.linspace(-1.0, 1.0, 5)
        zp = jnp.linspace(0.0, 1.0, 4)
        # Pick a function that is NOT itself multilinear; both
        # interp_nd and RegularGridInterpolator should give the same
        # multilinear approximation at every query point.
        X, Y, Z = jnp.meshgrid(xp, yp, zp, indexing="ij")
        values = jnp.sin(X) * jnp.cos(Y) + Z**2

        rgi = RegularGridInterpolator(
            (np.asarray(xp), np.asarray(yp), np.asarray(zp)),
            np.asarray(values),
            method="linear",
        )

        rng = np.random.default_rng(0)
        # Random off-grid queries inside the box.
        qs = rng.uniform(
            low=[0.05, -0.95, 0.05],
            high=[1.95, 0.95, 0.95],
            size=(50, 3),
        )
        for q in qs:
            got = float(interp_nd((xp, yp, zp), values, jnp.asarray(q)))
            # rgi expects a 2-D array for a batch of queries; pass a
            # single-row 2-D array and pull the scalar.
            want = float(rgi(np.atleast_2d(q))[0])
            assert got == pytest.approx(want, abs=1e-10), (q, got, want)


class TestInterpNdExtrapolation:
    @pytest.fixture
    def setup(self):
        xp = jnp.array([0.0, 1.0, 2.0])
        yp = jnp.array([0.0, 1.0, 2.0])
        zp = jnp.array([0.0, 1.0, 2.0])
        # f(x,y,z) = x*y*z restricted to corners — IS multilinear here
        # because each cell's corners are at integer coords and the
        # function evaluated there is exactly x*y*z (the multilinear
        # interpolant agrees with the function on the corners and is
        # linear in each variable holding others fixed — but x*y*z is
        # NOT multilinear in general).  So we use a function that IS:
        X, Y, Z = jnp.meshgrid(xp, yp, zp, indexing="ij")
        values = 1.0 + X + Y + Z + X * Y - 0.5 * Y * Z
        return (xp, yp, zp), values

    def test_clip_holds_boundary(self, setup):
        (xp, yp, zp), values = setup
        # OOB-x left only — should clip x to 0 and match in-grid value.
        q_oob = jnp.array([-1.0, 0.5, 0.5])
        q_clip = jnp.array([0.0, 0.5, 0.5])
        got = float(interp_nd((xp, yp, zp), values, q_oob, extrapolation="clip"))
        want = float(interp_nd((xp, yp, zp), values, q_clip, extrapolation="clip"))
        assert got == pytest.approx(want, abs=1e-12)

    def test_linear_extends_multilinear(self, setup):
        (xp, yp, zp), values = setup
        # f = 1 + x + y + z + x*y - 0.5*y*z is multilinear, so
        # multilinear extrapolation past the grid must reproduce f
        # exactly even for OOB queries.
        for q in [
            jnp.array([-0.5, 0.5, 0.5]),
            jnp.array([3.0, 1.5, 0.5]),
            jnp.array([0.5, 0.5, 4.0]),
        ]:
            x, y, z = float(q[0]), float(q[1]), float(q[2])
            want = 1.0 + x + y + z + x * y - 0.5 * y * z
            got = float(
                interp_nd((xp, yp, zp), values, q, extrapolation="linear")
            )
            assert got == pytest.approx(want, abs=1e-10), (q, got, want)

    def test_nan_outside_grid(self, setup):
        (xp, yp, zp), values = setup
        # Any axis OOB -> NaN.
        for q in [
            jnp.array([-1.0, 0.5, 0.5]),
            jnp.array([0.5, 5.0, 0.5]),
            jnp.array([0.5, 0.5, -2.0]),
            jnp.array([5.0, 5.0, 5.0]),
        ]:
            got = float(
                interp_nd((xp, yp, zp), values, q, extrapolation="nan")
            )
            assert math.isnan(got), (q, got)
        # Inside -> finite.
        got = float(
            interp_nd(
                (xp, yp, zp),
                values,
                jnp.array([0.5, 0.5, 0.5]),
                extrapolation="nan",
            )
        )
        assert math.isfinite(got)


class TestInterpNdValidation:
    def test_unknown_method_raises(self):
        xp = jnp.array([0.0, 1.0])
        yp = jnp.array([0.0, 1.0])
        values = jnp.zeros((2, 2))
        with pytest.raises(ValueError, match="unknown method"):
            interp_nd((xp, yp), values, jnp.array([0.5, 0.5]), method="bogus")

    def test_unknown_extrapolation_raises(self):
        xp = jnp.array([0.0, 1.0])
        yp = jnp.array([0.0, 1.0])
        values = jnp.zeros((2, 2))
        with pytest.raises(ValueError, match="unknown extrapolation"):
            interp_nd(
                (xp, yp), values, jnp.array([0.5, 0.5]), extrapolation="bogus"
            )

    def test_empty_grid_raises(self):
        with pytest.raises(ValueError, match="non-empty"):
            interp_nd((), jnp.zeros(()), jnp.array([0.5]))

    def test_grid_values_mismatch_raises(self):
        xp = jnp.array([0.0, 1.0])
        yp = jnp.array([0.0, 1.0])
        # values is 3-D but grid is 2 axes.
        with pytest.raises(ValueError, match="ndim"):
            interp_nd((xp, yp), jnp.zeros((2, 2, 2)), jnp.array([0.5, 0.5]))

    def test_grid_shape_mismatch_raises(self):
        xp = jnp.array([0.0, 1.0])
        yp = jnp.array([0.0, 1.0, 2.0])
        # Wrong shape for axis 1.
        with pytest.raises(ValueError, match="shape"):
            interp_nd((xp, yp), jnp.zeros((2, 2)), jnp.array([0.5, 0.5]))

    def test_query_dim_mismatch_raises(self):
        xp = jnp.array([0.0, 1.0])
        yp = jnp.array([0.0, 1.0])
        with pytest.raises(ValueError, match="must equal the"):
            interp_nd((xp, yp), jnp.zeros((2, 2)), jnp.array([0.5, 0.5, 0.5]))


class TestInterpNdBatched:
    """vmap-style batched query (leading dims on the query)."""

    def test_batched_query(self):
        xp = jnp.linspace(0.0, 1.0, 4)
        yp = jnp.linspace(0.0, 1.0, 4)
        zp = jnp.linspace(0.0, 1.0, 4)
        X, Y, Z = jnp.meshgrid(xp, yp, zp, indexing="ij")
        values = X + Y + Z

        # Single shape (3, 3) -> batch of 3 queries returning 3 values.
        qs = jnp.array(
            [
                [0.25, 0.5, 0.75],
                [0.0, 0.0, 0.0],
                [1.0, 1.0, 1.0],
            ]
        )
        got = interp_nd((xp, yp, zp), values, qs)
        assert got.shape == (3,)
        want = jnp.array([0.25 + 0.5 + 0.75, 0.0, 3.0])
        assert jnp.allclose(got, want, atol=1e-12)

    def test_2d_batched_query(self):
        xp = jnp.linspace(0.0, 1.0, 3)
        yp = jnp.linspace(0.0, 1.0, 3)
        values = xp[:, None] + yp[None, :]
        qs = jnp.array(
            [
                [[0.0, 0.0], [0.5, 0.5]],
                [[1.0, 1.0], [0.25, 0.75]],
            ]
        )
        got = interp_nd((xp, yp), values, qs)
        assert got.shape == (2, 2)
        want = jnp.array(
            [
                [0.0, 1.0],
                [2.0, 1.0],
            ]
        )
        assert jnp.allclose(got, want, atol=1e-12)


# ---------------------------------------------------------------------------
# Differentiability
# ---------------------------------------------------------------------------


class TestInterpNdDifferentiability:
    def test_grad_through_query_3d(self):
        xp = jnp.linspace(0.0, 1.0, 5)
        yp = jnp.linspace(0.0, 1.0, 5)
        zp = jnp.linspace(0.0, 1.0, 5)
        # Multilinear function: f(x,y,z) = x + 2y + 3z + xy
        X, Y, Z = jnp.meshgrid(xp, yp, zp, indexing="ij")
        values = X + 2.0 * Y + 3.0 * Z + X * Y

        def loss(q):
            return interp_nd((xp, yp, zp), values, q)

        q0 = jnp.array([0.3, 0.4, 0.6])
        g = jax.grad(loss)(q0)
        # df/dx = 1 + y, df/dy = 2 + x, df/dz = 3
        want = jnp.array([1.0 + 0.4, 2.0 + 0.3, 3.0])
        assert jnp.allclose(g, want, atol=1e-6), (g, want)

    def test_grad_through_table_values(self):
        xp = jnp.array([0.0, 1.0])
        yp = jnp.array([0.0, 1.0])
        zp = jnp.array([0.0, 1.0])

        def loss(values):
            return interp_nd(
                (xp, yp, zp), values, jnp.array([0.5, 0.5, 0.5])
            )

        v0 = jnp.zeros((2, 2, 2))
        g = jax.grad(loss)(v0)
        # At (0.5, 0.5, 0.5) the trilinear is the average of all 8
        # corners; gradient w.r.t. each corner is 1/8 = 0.125.
        assert g.shape == v0.shape
        for idx in np.ndindex(2, 2, 2):
            assert float(g[idx]) == pytest.approx(0.125, abs=1e-12), idx


# ---------------------------------------------------------------------------
# Higher-dimensional smoke test (4-D)
# ---------------------------------------------------------------------------


class TestInterpNdHighDim:
    def test_4d_smoke(self):
        # 4-D table on [0,1]^4.  Pick a multilinear function so the
        # multilinear interpolant is exact at off-grid queries too.
        grid = tuple(jnp.linspace(0.0, 1.0, 3) for _ in range(4))
        G = jnp.meshgrid(*grid, indexing="ij")
        # f(a,b,c,d) = a + b + c + d + a*b + c*d
        values = G[0] + G[1] + G[2] + G[3] + G[0] * G[1] + G[2] * G[3]
        for q in [
            jnp.array([0.0, 0.0, 0.0, 0.0]),
            jnp.array([1.0, 1.0, 1.0, 1.0]),
            jnp.array([0.25, 0.5, 0.75, 0.5]),
            jnp.array([0.3, 0.7, 0.1, 0.9]),
        ]:
            a, b, c, d = (float(q[i]) for i in range(4))
            want = a + b + c + d + a * b + c * d
            got = float(interp_nd(grid, values, q))
            assert got == pytest.approx(want, abs=1e-10), (q, got, want)

    def test_5d_smoke(self):
        # 5-D — slightly bigger but still multilinear so we can check
        # exact values.
        grid = tuple(jnp.linspace(0.0, 1.0, 3) for _ in range(5))
        G = jnp.meshgrid(*grid, indexing="ij")
        # f = sum of inputs (perfectly multilinear, gradient is 1 each)
        values = sum(G)
        q = jnp.array([0.3, 0.5, 0.7, 0.2, 0.9])
        got = float(interp_nd(grid, values, q))
        want = float(jnp.sum(q))
        assert got == pytest.approx(want, abs=1e-10)


# ---------------------------------------------------------------------------
# Block-level wiring (LookupTableND)
# ---------------------------------------------------------------------------


def _eval_nd_block(grid_axes, output_array, query, **kwargs):
    builder = jaxonomy.DiagramBuilder()
    blk = builder.add(library.LookupTableND(grid_axes, output_array, **kwargs))
    src = builder.add(library.Constant(jnp.asarray(query)))
    builder.connect(src.output_ports[0], blk.input_ports[0])
    diagram = builder.build()
    ctx = diagram.create_context()
    return blk.output_ports[0].eval(ctx)


class TestLookupTableNDBlock:
    @pytest.fixture
    def grid_and_values(self):
        xp = jnp.linspace(0.0, 1.0, 4)
        yp = jnp.linspace(0.0, 1.0, 4)
        zp = jnp.linspace(0.0, 1.0, 4)
        X, Y, Z = jnp.meshgrid(xp, yp, zp, indexing="ij")
        # Multilinear so off-grid queries are exact.
        values = 1.0 + X + Y + Z + 2.0 * X * Y
        return (xp, yp, zp), values

    def test_block_3d_at_grid_corner(self, grid_and_values):
        grid, values = grid_and_values
        got = float(_eval_nd_block(grid, values, jnp.array([0.0, 0.0, 0.0])))
        assert got == pytest.approx(1.0, abs=1e-12)

    def test_block_3d_off_grid(self, grid_and_values):
        grid, values = grid_and_values
        q = jnp.array([0.3, 0.4, 0.6])
        got = float(_eval_nd_block(grid, values, q))
        x, y, z = 0.3, 0.4, 0.6
        want = 1.0 + x + y + z + 2.0 * x * y
        assert got == pytest.approx(want, abs=1e-10)

    def test_block_default_clip(self, grid_and_values):
        grid, values = grid_and_values
        # OOB query, default clip -> matches interior boundary value.
        got = float(_eval_nd_block(grid, values, jnp.array([-1.0, 0.5, 0.5])))
        want = float(_eval_nd_block(grid, values, jnp.array([0.0, 0.5, 0.5])))
        assert got == pytest.approx(want, abs=1e-12)

    def test_block_linear_extrapolation(self, grid_and_values):
        grid, values = grid_and_values
        q = jnp.array([-0.5, 0.5, 1.5])  # x and z OOB
        got = float(
            _eval_nd_block(grid, values, q, extrapolation="linear")
        )
        x, y, z = -0.5, 0.5, 1.5
        want = 1.0 + x + y + z + 2.0 * x * y
        assert got == pytest.approx(want, abs=1e-10)

    def test_block_nan_extrapolation(self, grid_and_values):
        grid, values = grid_and_values
        got = float(
            _eval_nd_block(
                grid, values, jnp.array([-1.0, 0.5, 0.5]), extrapolation="nan"
            )
        )
        assert math.isnan(got)
        got = float(
            _eval_nd_block(
                grid, values, jnp.array([0.5, 0.5, 0.5]), extrapolation="nan"
            )
        )
        assert math.isfinite(got)

    def test_block_4d(self):
        grid = tuple(jnp.linspace(0.0, 1.0, 3) for _ in range(4))
        G = jnp.meshgrid(*grid, indexing="ij")
        values = G[0] + G[1] + G[2] + G[3]
        q = jnp.array([0.3, 0.5, 0.7, 0.2])
        got = float(_eval_nd_block(grid, values, q))
        assert got == pytest.approx(0.3 + 0.5 + 0.7 + 0.2, abs=1e-10)

    def test_invalid_extrapolation_rejected(self, grid_and_values):
        grid, values = grid_and_values
        with pytest.raises(ValueError, match="extrapolation"):
            library.LookupTableND(grid, values, extrapolation="bogus")

    def test_invalid_interpolation_rejected(self, grid_and_values):
        grid, values = grid_and_values
        with pytest.raises(NotImplementedError):
            library.LookupTableND(grid, values, interpolation="pchip")

    def test_empty_grid_axes_rejected(self):
        with pytest.raises(ValueError, match="non-empty"):
            library.LookupTableND((), jnp.zeros(()))

    def test_non_monotonic_grid_rejected(self):
        with pytest.raises(ValueError, match="monotonically"):
            library.LookupTableND(
                (jnp.array([0.0, 1.0, 0.5]),), jnp.zeros((3,))
            )

    def test_non_1d_axis_rejected(self):
        with pytest.raises(ValueError, match="1-D"):
            library.LookupTableND(
                (jnp.zeros((2, 2)),), jnp.zeros((2,))
            )

    def test_shape_mismatch_rejected(self):
        with pytest.raises(ValueError, match="ndim"):
            # 2-D axes but 1-D table.
            library.LookupTableND(
                (jnp.array([0.0, 1.0]), jnp.array([0.0, 1.0])),
                jnp.zeros((2,)),
            )

    def test_table_shape_mismatch_rejected(self):
        with pytest.raises(ValueError, match="shape"):
            library.LookupTableND(
                (jnp.array([0.0, 1.0]), jnp.array([0.0, 1.0, 2.0])),
                jnp.zeros((2, 2)),
            )


class TestLookupTableNDDtype:
    def test_dtype_kwarg(self):
        # Per-block dtype override should cast both the grid and table
        # to the requested dtype.
        xp = np.array([0.0, 1.0, 2.0], dtype=np.float64)
        yp = np.array([0.0, 1.0, 2.0], dtype=np.float64)
        values = np.zeros((3, 3), dtype=np.float64)
        blk = library.LookupTableND((xp, yp), values, dtype=jnp.float32)
        assert blk._output_array.dtype == jnp.float32
        for axis in blk._grid_axes:
            assert axis.dtype == jnp.float32
