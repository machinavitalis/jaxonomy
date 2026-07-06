# SPDX-License-Identifier: MIT

"""T-114-followup-prelookup-extrap -- Prelookup/InterpolationUsingPrelookup
extrapolation modes.

The shipped ``T-114-followup-prelookup`` pair clipped ``alpha`` to
``[0, 1]`` so out-of-range queries collapsed to the nearest endpoint
(matching :class:`LookupTable1d` ``extrapolation="clip"``).  This
follow-up adds an ``extrapolation`` kwarg to both blocks so users can
opt in to:

  * ``"linear"`` -- alpha left raw past the grid; the downstream blend
    ``(1 - alpha) * yp[i] + alpha * yp[i+1]`` extends the boundary
    slope, matching :class:`LookupTable1d` ``extrapolation="linear"``.
  * ``"nan"`` -- alpha set to NaN past the grid; the blend propagates
    NaN, matching :class:`LookupTable1d` ``extrapolation="nan"``.

The kwarg defaults to ``"clip"`` on BOTH blocks so the pre-existing
T-114-fu-prelookup behaviour is byte-equivalent (covered separately by
the original test file, plus a smoke test below).

The two blocks share the alpha signal, so they must agree on mode --
``Prelookup`` does the math, ``InterpolationUsingPrelookup`` records
the same intent for API symmetry / IDE discoverability.  The pair
agreement is a user convention; we don't bind the producer at
construction time, so a mismatch is a silent semantic bug -- documented
on the kwarg docstrings.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import jaxonomy
from jaxonomy import library
from jaxonomy.library import (
    InterpolationUsingPrelookup,
    LookupTable1d,
    Prelookup,
)


# ---------------------------------------------------------------------------
# Helper: build (Constant -> Prelookup -> InterpolationUsingPrelookup) for
# a chosen extrapolation mode and return the scalar output.
# ---------------------------------------------------------------------------


def _eval_pair(xp, yp, x_query, extrapolation):
    builder = jaxonomy.DiagramBuilder()
    src = builder.add(library.Constant(jnp.asarray(x_query)))
    pre = builder.add(Prelookup(xp, extrapolation=extrapolation))
    interp = builder.add(
        InterpolationUsingPrelookup(yp, extrapolation=extrapolation)
    )
    builder.connect(src.output_ports[0], pre.input_ports[0])
    builder.connect(pre.output_ports[0], interp.input_ports[0])
    diagram = builder.build()
    ctx = diagram.create_context()
    return float(np.asarray(interp.output_ports[0].eval(ctx)))


def _eval_direct_lookup(xp, yp, x_query, extrapolation):
    """Reference: a direct ``LookupTable1d`` with the same extrapolation."""
    builder = jaxonomy.DiagramBuilder()
    src = builder.add(library.Constant(jnp.asarray(x_query)))
    blk = builder.add(
        LookupTable1d(xp, yp, "linear", extrapolation=extrapolation)
    )
    builder.connect(src.output_ports[0], blk.input_ports[0])
    diagram = builder.build()
    ctx = diagram.create_context()
    return float(np.asarray(blk.output_ports[0].eval(ctx)))


# ---------------------------------------------------------------------------
# Construction-time validation: bad extrapolation strings raise.
# ---------------------------------------------------------------------------


class TestExtrapolationValidation:
    def test_prelookup_bad_mode_raises(self):
        with pytest.raises(ValueError, match="extrapolation"):
            Prelookup(jnp.array([0.0, 1.0, 2.0]), extrapolation="bogus")

    def test_interp_bad_mode_raises(self):
        with pytest.raises(ValueError, match="extrapolation"):
            InterpolationUsingPrelookup(
                jnp.array([0.0, 1.0, 2.0]), extrapolation="bogus"
            )

    def test_prelookup_records_mode(self):
        blk = Prelookup(jnp.array([0.0, 1.0, 2.0]), extrapolation="linear")
        assert blk.extrapolation == "linear"

    def test_interp_records_mode(self):
        blk = InterpolationUsingPrelookup(
            jnp.array([0.0, 1.0, 2.0]), extrapolation="nan"
        )
        assert blk.extrapolation == "nan"

    def test_default_mode_is_clip(self):
        pre = Prelookup(jnp.array([0.0, 1.0, 2.0]))
        interp = InterpolationUsingPrelookup(jnp.array([0.0, 1.0, 2.0]))
        assert pre.extrapolation == "clip"
        assert interp.extrapolation == "clip"


# ---------------------------------------------------------------------------
# Mode "clip" (default): OOB query collapses to the nearest endpoint.
# ---------------------------------------------------------------------------


class TestClipMode:
    @pytest.fixture
    def grid_and_table(self):
        xp = jnp.array([0.0, 1.0, 2.0, 3.0])
        yp = jnp.array([10.0, 20.0, 30.0, 40.0])
        return xp, yp

    def test_clip_oob_left_returns_first_value(self, grid_and_table):
        xp, yp = grid_and_table
        got = _eval_pair(xp, yp, -5.0, "clip")
        assert got == pytest.approx(10.0, abs=1e-12)

    def test_clip_oob_right_returns_last_value(self, grid_and_table):
        xp, yp = grid_and_table
        got = _eval_pair(xp, yp, 10.0, "clip")
        assert got == pytest.approx(40.0, abs=1e-12)

    def test_clip_in_range_matches_lookup_table_1d(self, grid_and_table):
        xp, yp = grid_and_table
        for q in [0.0, 0.5, 1.7, 3.0]:
            got = _eval_pair(xp, yp, q, "clip")
            want = _eval_direct_lookup(xp, yp, q, "clip")
            assert got == pytest.approx(want, abs=1e-12), (q, got, want)


# ---------------------------------------------------------------------------
# Mode "linear": OOB query extends the boundary slope linearly.
# ---------------------------------------------------------------------------


class TestLinearMode:
    @pytest.fixture
    def grid_and_table(self):
        # Uniform slope 10 on the left edge (yp[1] - yp[0]) / (xp[1] - xp[0])
        # and slope 10 on the right edge so the extrapolation arithmetic
        # is easy to verify by hand.
        xp = jnp.array([0.0, 1.0, 2.0, 3.0])
        yp = jnp.array([0.0, 10.0, 20.0, 30.0])  # affine, slope 10 everywhere
        return xp, yp

    def test_linear_oob_left_extends_slope(self, grid_and_table):
        xp, yp = grid_and_table
        # Query x = -2 on a line through (0, 0) with slope 10: y = -20.
        got = _eval_pair(xp, yp, -2.0, "linear")
        assert got == pytest.approx(-20.0, abs=1e-12)
        # And the LookupTable1d "linear" mode must agree.
        want = _eval_direct_lookup(xp, yp, -2.0, "linear")
        assert got == pytest.approx(want, abs=1e-12)

    def test_linear_oob_right_extends_slope(self, grid_and_table):
        xp, yp = grid_and_table
        # Query x = 5 on a line through (3, 30) with slope 10: y = 50.
        got = _eval_pair(xp, yp, 5.0, "linear")
        assert got == pytest.approx(50.0, abs=1e-12)
        want = _eval_direct_lookup(xp, yp, 5.0, "linear")
        assert got == pytest.approx(want, abs=1e-12)

    def test_linear_non_affine_table_matches_lookup_table_1d(self):
        """Non-affine table: the OOB extrapolation uses only the boundary
        edge slope (standard lookup-table semantics) -- agree with LookupTable1d.
        """
        xp = jnp.array([0.0, 1.0, 2.0, 3.0])
        yp = jnp.array([0.0, 1.0, 4.0, 9.0])  # x^2 sample
        # Left edge slope = (1 - 0) / 1 = 1.  Query -0.5 -> -0.5.
        # Right edge slope = (9 - 4) / 1 = 5.  Query 4.0 -> 9 + 5*1 = 14.
        for q in [-0.5, 4.0]:
            got = _eval_pair(xp, yp, q, "linear")
            want = _eval_direct_lookup(xp, yp, q, "linear")
            assert got == pytest.approx(want, abs=1e-12), (q, got, want)

    def test_linear_in_range_matches_clip(self):
        """Inside the grid, linear extrapolation makes no difference."""
        xp = jnp.array([0.0, 1.0, 2.0, 3.0])
        yp = jnp.array([0.0, 5.0, 12.0, 20.0])
        for q in [0.0, 0.3, 1.5, 2.8, 3.0]:
            got_clip = _eval_pair(xp, yp, q, "clip")
            got_linear = _eval_pair(xp, yp, q, "linear")
            assert got_linear == pytest.approx(got_clip, abs=1e-12), (
                q,
                got_clip,
                got_linear,
            )

    def test_linear_oob_is_finite_not_nan(self):
        xp = jnp.array([0.0, 1.0, 2.0, 3.0])
        yp = jnp.array([0.0, 10.0, 20.0, 30.0])
        for q in [-100.0, 100.0]:
            got = _eval_pair(xp, yp, q, "linear")
            assert np.isfinite(got), (q, got)


# ---------------------------------------------------------------------------
# Mode "nan": OOB query yields NaN.
# ---------------------------------------------------------------------------


class TestNanMode:
    @pytest.fixture
    def grid_and_table(self):
        xp = jnp.array([0.0, 1.0, 2.0, 3.0])
        yp = jnp.array([0.0, 10.0, 20.0, 30.0])
        return xp, yp

    def test_nan_oob_left_returns_nan(self, grid_and_table):
        xp, yp = grid_and_table
        got = _eval_pair(xp, yp, -0.5, "nan")
        assert np.isnan(got)

    def test_nan_oob_right_returns_nan(self, grid_and_table):
        xp, yp = grid_and_table
        got = _eval_pair(xp, yp, 3.5, "nan")
        assert np.isnan(got)

    def test_nan_in_range_is_finite(self, grid_and_table):
        xp, yp = grid_and_table
        for q in [0.0, 0.25, 1.5, 2.99, 3.0]:
            got = _eval_pair(xp, yp, q, "nan")
            assert np.isfinite(got), (q, got)

    def test_nan_at_endpoints_not_nan(self, grid_and_table):
        """Querying exactly on either endpoint is in-range -- the NaN
        mask uses strict inequalities (matches LookupTable1d).
        """
        xp, yp = grid_and_table
        for q in [0.0, 3.0]:
            got = _eval_pair(xp, yp, q, "nan")
            assert np.isfinite(got), (q, got)


# ---------------------------------------------------------------------------
# Default-off byte-equivalence: leaving the kwarg unset behaves exactly
# like the pre-existing T-114-fu-prelookup ("clip" was implicit).
# ---------------------------------------------------------------------------


class TestDefaultOffByteEquivalence:
    def test_default_pair_matches_explicit_clip(self):
        xp = jnp.array([0.0, 1.0, 2.0, 3.0])
        yp = jnp.array([5.0, 7.0, 11.0, 13.0])
        # Cover interior + both OOB sides.
        for q in [-5.0, 0.0, 0.5, 1.7, 3.0, 100.0]:
            # Implicit-default pair:
            builder = jaxonomy.DiagramBuilder()
            src = builder.add(library.Constant(jnp.asarray(q)))
            pre = builder.add(Prelookup(xp))
            interp = builder.add(InterpolationUsingPrelookup(yp))
            builder.connect(src.output_ports[0], pre.input_ports[0])
            builder.connect(pre.output_ports[0], interp.input_ports[0])
            diagram = builder.build()
            ctx = diagram.create_context()
            got_default = float(
                np.asarray(interp.output_ports[0].eval(ctx))
            )

            # Explicit "clip" pair:
            got_clip = _eval_pair(xp, yp, q, "clip")
            assert got_default == pytest.approx(got_clip, abs=1e-12), (
                q,
                got_default,
                got_clip,
            )


# ---------------------------------------------------------------------------
# Pure-JAX differentiability: gradients flow through the query coordinate
# AND the output_array for every mode (NaN mode only inside the grid --
# OOB NaN gradients are themselves NaN, which is expected and matches
# LookupTable1d).
# ---------------------------------------------------------------------------


def _prelookup_pure(xp, x_query, mode):
    """Pure-JAX equivalent of Prelookup output for a chosen mode."""
    n = xp.shape[0]
    i = jnp.clip(jnp.searchsorted(xp, x_query, side="right") - 1, 0, n - 2)
    x0 = xp[i]
    x1 = xp[i + 1]
    alpha = (x_query - x0) / (x1 - x0)
    if mode == "clip":
        alpha = jnp.clip(alpha, 0.0, 1.0)
    elif mode == "nan":
        oob = (x_query < xp[0]) | (x_query > xp[-1])
        alpha = jnp.where(oob, jnp.asarray(jnp.nan, dtype=alpha.dtype), alpha)
    # "linear": alpha left raw.
    return i, alpha


def _interp_pure(yp, idx_alpha):
    i, alpha = idx_alpha
    return (1.0 - alpha) * yp[i] + alpha * yp[i + 1]


class TestDifferentiability:
    def test_linear_grad_through_query_oob(self):
        """In ``linear`` mode the OOB derivative is the boundary slope."""
        xp = jnp.array([0.0, 1.0, 2.0, 3.0])
        yp = jnp.array([0.0, 5.0, 11.0, 14.0])

        def f(x):
            return _interp_pure(yp, _prelookup_pure(xp, x, "linear"))

        # Left edge slope = (5 - 0) / 1 = 5.  Past the left edge, derivative
        # of the linear extension is also 5.
        g = float(jax.grad(f)(jnp.asarray(-2.0)))
        assert g == pytest.approx(5.0, abs=1e-10)

        # Right edge slope = (14 - 11) / 1 = 3.  Past the right edge, also 3.
        g = float(jax.grad(f)(jnp.asarray(10.0)))
        assert g == pytest.approx(3.0, abs=1e-10)

    def test_clip_grad_through_query_oob_is_zero(self):
        """In ``clip`` mode the OOB derivative is zero (clipped alpha is
        flat past the edge).
        """
        xp = jnp.array([0.0, 1.0, 2.0, 3.0])
        yp = jnp.array([0.0, 5.0, 11.0, 14.0])

        def f(x):
            return _interp_pure(yp, _prelookup_pure(xp, x, "clip"))

        g = float(jax.grad(f)(jnp.asarray(-2.0)))
        assert g == pytest.approx(0.0, abs=1e-10)
        g = float(jax.grad(f)(jnp.asarray(10.0)))
        assert g == pytest.approx(0.0, abs=1e-10)

    def test_linear_grad_through_table_values(self):
        """In ``linear`` mode the derivative w.r.t. the table values for
        an OOB-left query depends only on the first two entries.
        """
        xp = jnp.array([0.0, 1.0, 2.0, 3.0])

        def f(yp):
            return _interp_pure(yp, _prelookup_pure(xp, jnp.asarray(-1.0), "linear"))

        yp0 = jnp.zeros(4)
        g = jax.grad(f)(yp0)
        # At x=-1, bucket i=0, alpha = (-1 - 0)/(1 - 0) = -1.
        # y = (1 - (-1)) * yp[0] + (-1) * yp[1] = 2*yp[0] - yp[1].
        # dY/dyp[0] = 2, dY/dyp[1] = -1, rest = 0.
        np.testing.assert_allclose(np.asarray(g), [2.0, -1.0, 0.0, 0.0])

    def test_nan_grad_in_range(self):
        """Inside the grid, ``nan`` mode is byte-equivalent to ``clip``
        and gradients flow normally.
        """
        xp = jnp.array([0.0, 1.0, 2.0, 3.0])
        yp = jnp.array([0.0, 5.0, 11.0, 14.0])

        def f(x):
            return _interp_pure(yp, _prelookup_pure(xp, x, "nan"))

        # Inside bucket [1, 2]: slope = (11 - 5)/1 = 6.
        g = float(jax.grad(f)(jnp.asarray(1.5)))
        assert g == pytest.approx(6.0, abs=1e-10)


# ---------------------------------------------------------------------------
# Fan-out under each mode: two InterpolationUsingPrelookup blocks sharing
# one Prelookup must both see the same extrapolation policy because the
# math is owned by the producer.
# ---------------------------------------------------------------------------


class TestFanoutUnderModes:
    @pytest.mark.parametrize("mode", ["clip", "linear", "nan"])
    def test_fanout_two_tables_share_prelookup(self, mode):
        xp = jnp.array([0.0, 1.0, 2.0, 3.0])
        yp_a = jnp.array([0.0, 10.0, 20.0, 30.0])  # slope 10
        yp_b = jnp.array([100.0, 90.0, 80.0, 70.0])  # slope -10

        builder = jaxonomy.DiagramBuilder()
        src = builder.add(library.Constant(jnp.asarray(-2.0)))  # OOB left
        pre = builder.add(Prelookup(xp, extrapolation=mode))
        ia = builder.add(InterpolationUsingPrelookup(yp_a, extrapolation=mode))
        ib = builder.add(InterpolationUsingPrelookup(yp_b, extrapolation=mode))
        builder.connect(src.output_ports[0], pre.input_ports[0])
        builder.connect(pre.output_ports[0], ia.input_ports[0])
        builder.connect(pre.output_ports[0], ib.input_ports[0])
        diagram = builder.build()
        ctx = diagram.create_context()

        got_a = float(np.asarray(ia.output_ports[0].eval(ctx)))
        got_b = float(np.asarray(ib.output_ports[0].eval(ctx)))

        if mode == "clip":
            assert got_a == pytest.approx(0.0, abs=1e-12)   # left endpoint
            assert got_b == pytest.approx(100.0, abs=1e-12)  # left endpoint
        elif mode == "linear":
            assert got_a == pytest.approx(-20.0, abs=1e-12)  # 0 + (-2)*10
            assert got_b == pytest.approx(120.0, abs=1e-12)  # 100 + (-2)*-10
        else:  # mode == "nan"
            assert np.isnan(got_a)
            assert np.isnan(got_b)
