# SPDX-License-Identifier: MIT

"""T-114-followup-prelookup-inverse -- PrelookupInverse block.

The forward :class:`Prelookup` finds ``(i, alpha)`` such that
``xp[i] + alpha * (xp[i+1] - xp[i]) ≈ x``.  The inverse counterpart
finds ``(i, alpha)`` such that
``yp[i] + alpha * (yp[i+1] - yp[i]) ≈ y`` -- i.e. it inverts a
monotonic 1-D table.

These tests exercise:

  * Strictly-increasing inverse lookup against an `x^2` table.
  * Strictly-decreasing inverse lookup against a flipped table.
  * Combined ``PrelookupInverse -> InterpolationUsingPrelookup`` (whose
    table is the breakpoint axis ``xp``) reconstructing
    ``x = sqrt(y)`` from the forward ``y = x^2`` table.
  * Construction-time validation: non-monotonic table raises a clear
    error.
  * Differentiability: ``jax.grad`` flows through the query coordinate.
  * Honest fallback: ``"linear"``/``"nan"`` extrapolation modes raise
    NotImplementedError (deeper followup).
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
    PrelookupInverse,
)


# ---------------------------------------------------------------------------
# Construction-time validation
# ---------------------------------------------------------------------------


class TestPrelookupInverseValidation:
    def test_non_1d_output_array_raises(self):
        with pytest.raises(ValueError, match="1-D"):
            PrelookupInverse(jnp.zeros((3, 3)))

    def test_too_short_output_array_raises(self):
        with pytest.raises(ValueError, match="at least 2"):
            PrelookupInverse(jnp.array([1.0]))

    def test_non_monotonic_output_array_raises(self):
        # Goes up then down -- ambiguous to invert.
        with pytest.raises(ValueError, match="monotonic"):
            PrelookupInverse(jnp.array([0.0, 2.0, 1.0]))

    def test_constant_segment_raises(self):
        # Strict monotonicity: equal neighbours not allowed.
        with pytest.raises(ValueError, match="monotonic"):
            PrelookupInverse(jnp.array([0.0, 1.0, 1.0, 2.0]))

    def test_dtype_kwarg_casts_table(self):
        yp = np.array([0.0, 1.0, 2.0], dtype=np.float64)
        blk = PrelookupInverse(yp, dtype=jnp.float32)
        assert blk.output_array.dtype == jnp.float32

    def test_extrapolation_linear_is_deferred(self):
        with pytest.raises(NotImplementedError, match="extrap"):
            PrelookupInverse(
                jnp.array([0.0, 1.0, 2.0]), extrapolation="linear"
            )

    def test_extrapolation_nan_is_deferred(self):
        with pytest.raises(NotImplementedError, match="extrap"):
            PrelookupInverse(
                jnp.array([0.0, 1.0, 2.0]), extrapolation="nan"
            )

    def test_extrapolation_bogus_raises(self):
        with pytest.raises(ValueError, match="clip"):
            PrelookupInverse(
                jnp.array([0.0, 1.0, 2.0]), extrapolation="bogus"
            )

    def test_increasing_table_records_direction(self):
        blk = PrelookupInverse(jnp.array([0.0, 1.0, 4.0, 9.0, 16.0]))
        assert blk.direction == "increasing"

    def test_decreasing_table_records_direction(self):
        blk = PrelookupInverse(jnp.array([16.0, 9.0, 4.0, 1.0, 0.0]))
        assert blk.direction == "decreasing"


# ---------------------------------------------------------------------------
# Helpers: build (Constant -> PrelookupInverse) and pull the
# (index, fraction) NamedTuple out of the output port.
# ---------------------------------------------------------------------------


def _eval_inverse_pair(yp, y_query):
    """Build (Constant -> PrelookupInverse), evaluate, return
    (int(index), float(fraction)).
    """
    builder = jaxonomy.DiagramBuilder()
    src = builder.add(library.Constant(jnp.asarray(y_query)))
    inv = builder.add(PrelookupInverse(yp))
    builder.connect(src.output_ports[0], inv.input_ports[0])
    diagram = builder.build()
    ctx = diagram.create_context()
    result = inv.output_ports[0].eval(ctx)
    return int(result.index), float(result.fraction)


def _eval_inverse_then_interp(yp, xp, y_query):
    """``PrelookupInverse(yp) -> InterpolationUsingPrelookup(xp)``:
    given y, recover x = f^{-1}(y) using xp as the table that maps
    bucket-index back into the breakpoint axis.
    """
    builder = jaxonomy.DiagramBuilder()
    src = builder.add(library.Constant(jnp.asarray(y_query)))
    inv = builder.add(PrelookupInverse(yp))
    interp = builder.add(InterpolationUsingPrelookup(xp))
    builder.connect(src.output_ports[0], inv.input_ports[0])
    builder.connect(inv.output_ports[0], interp.input_ports[0])
    diagram = builder.build()
    ctx = diagram.create_context()
    return float(np.asarray(interp.output_ports[0].eval(ctx)))


# ---------------------------------------------------------------------------
# Increasing table: known (index, fraction) values for an x^2 table.
# ---------------------------------------------------------------------------


class TestIncreasingInverseLookup:
    """Squares table ``[0, 1, 4, 9, 16]`` at xp = ``[0, 1, 2, 3, 4]``."""

    @pytest.fixture
    def yp(self):
        return jnp.array([0.0, 1.0, 4.0, 9.0, 16.0])

    def test_at_breakpoint_y4_gives_index2_fraction0(self, yp):
        idx, frac = _eval_inverse_pair(yp, 4.0)
        assert idx == 2
        assert frac == pytest.approx(0.0, abs=1e-12)

    def test_off_grid_y6p5_gives_index2_fraction0p5(self, yp):
        # 6.5 is halfway between 4 and 9.
        idx, frac = _eval_inverse_pair(yp, 6.5)
        assert idx == 2
        assert frac == pytest.approx(0.5, abs=1e-12)

    def test_at_first_breakpoint_gives_index0_fraction0(self, yp):
        idx, frac = _eval_inverse_pair(yp, 0.0)
        assert idx == 0
        assert frac == pytest.approx(0.0, abs=1e-12)

    def test_at_last_breakpoint_gives_last_bucket_fraction1(self, yp):
        # y=16 hits the right-most breakpoint; bucket clipped to n-2=3,
        # alpha = (16 - 9) / (16 - 9) = 1.0.
        idx, frac = _eval_inverse_pair(yp, 16.0)
        assert idx == 3
        assert frac == pytest.approx(1.0, abs=1e-12)

    def test_oob_left_clips(self, yp):
        # y = -5 below the min: alpha clipped to 0, bucket 0.
        idx, frac = _eval_inverse_pair(yp, -5.0)
        assert idx == 0
        assert frac == pytest.approx(0.0, abs=1e-12)

    def test_oob_right_clips(self, yp):
        # y = 100 above the max: alpha clipped to 1, last bucket.
        idx, frac = _eval_inverse_pair(yp, 100.0)
        assert idx == 3
        assert frac == pytest.approx(1.0, abs=1e-12)


# ---------------------------------------------------------------------------
# Decreasing table: the inverse must report the same (index, fraction)
# semantics relative to ``yp`` (i.e. yp[i] + alpha*(yp[i+1]-yp[i]) == y).
# ---------------------------------------------------------------------------


class TestDecreasingInverseLookup:
    """Flipped squares ``[16, 9, 4, 1, 0]``."""

    @pytest.fixture
    def yp(self):
        return jnp.array([16.0, 9.0, 4.0, 1.0, 0.0])

    def test_at_breakpoint_y4_reconstructs_exactly(self, yp):
        # yp[2] = 4.  Either (i=1, alpha=1) or (i=2, alpha=0) is a valid
        # representation -- both reconstruct y=4 exactly.  Pin the
        # reconstruction rather than the exact (index, fraction) pair so
        # we are agnostic about searchsorted side conventions on the
        # boundary.
        idx, frac = _eval_inverse_pair(yp, 4.0)
        recon = (1.0 - frac) * float(yp[idx]) + frac * float(yp[idx + 1])
        assert recon == pytest.approx(4.0, abs=1e-12)

    def test_off_grid_y6p5_gives_index1_fraction0p5(self, yp):
        # 6.5 is halfway between yp[1]=9 and yp[2]=4 ->
        # i=1, alpha = (6.5 - 9) / (4 - 9) = 0.5.
        idx, frac = _eval_inverse_pair(yp, 6.5)
        assert idx == 1
        assert frac == pytest.approx(0.5, abs=1e-12)

    def test_oob_above_max_clips(self, yp):
        # 100 above the max (yp[0]=16): pinned to nearest endpoint;
        # the (1-alpha)*y0 + alpha*y1 blend must produce yp[0]=16.
        idx, frac = _eval_inverse_pair(yp, 100.0)
        recon = (1.0 - frac) * float(yp[idx]) + frac * float(yp[idx + 1])
        assert recon == pytest.approx(16.0, abs=1e-12)

    def test_oob_below_min_clips(self, yp):
        # -5 below the min (yp[-1]=0): pinned to last bucket, alpha=1.
        idx, frac = _eval_inverse_pair(yp, -5.0)
        assert idx == 3
        assert frac == pytest.approx(1.0, abs=1e-12)

    def test_value_reconstruction_increasing(self, yp):
        # Confirm the (i, alpha) pair reconstructs the query exactly.
        for q in [0.5, 2.5, 5.0, 12.0, 15.0]:
            idx, frac = _eval_inverse_pair(yp, q)
            y0 = float(yp[idx])
            y1 = float(yp[idx + 1])
            recon = (1.0 - frac) * y0 + frac * y1
            assert recon == pytest.approx(q, abs=1e-10), (q, idx, frac)


# ---------------------------------------------------------------------------
# Headline use case: combine PrelookupInverse with an
# InterpolationUsingPrelookup whose output_array is the FORWARD
# breakpoint axis ``xp``.  The composition reconstructs the inverse
# function -- for y = x^2 this is x = sqrt(y).
# ---------------------------------------------------------------------------


class TestInverseFunctionReconstruction:
    def test_sqrt_from_squares_table(self):
        """Given the forward table (xp, yp) = (xp, xp^2), the
        composition ``PrelookupInverse(yp) -> InterpolationUsingPrelookup(xp)``
        approximates ``sqrt(y)`` (piecewise linearly between the
        squared breakpoints).
        """
        xp = jnp.array([0.0, 1.0, 2.0, 3.0, 4.0])
        yp = xp ** 2  # [0, 1, 4, 9, 16]

        # Test at a few queries that land on breakpoints (where the
        # piecewise-linear inverse is exact w.r.t. sqrt).
        for y_query, want_x in [(0.0, 0.0), (1.0, 1.0), (4.0, 2.0),
                                (9.0, 3.0), (16.0, 4.0)]:
            got = _eval_inverse_then_interp(yp, xp, y_query)
            assert got == pytest.approx(want_x, abs=1e-10), (y_query, got)

    def test_sqrt_off_breakpoint_is_piecewise_linear(self):
        """At y=6.5 (between yp[2]=4 and yp[3]=9 in the increasing
        table), the inverse is alpha = (6.5 - 4) / (9 - 4) = 0.5,
        so x = 0.5*xp[2] + 0.5*xp[3] = 0.5*2 + 0.5*3 = 2.5.

        Note this is the *piecewise-linear* inverse of x^2, not the
        true sqrt(6.5)=2.5495..., which is the right answer for the
        standard linear-interpolation lookup-table convention.
        """
        xp = jnp.array([0.0, 1.0, 2.0, 3.0, 4.0])
        yp = xp ** 2
        got = _eval_inverse_then_interp(yp, xp, 6.5)
        assert got == pytest.approx(2.5, abs=1e-10)

    def test_gain_scheduling_use_case_linear_table(self):
        """``y = a*x + b`` is the cleanest case: the piecewise-linear
        inverse is EXACT (no discretisation error from the table
        spacing because the function is linear).
        """
        a, b = 3.0, 5.0
        xp = jnp.linspace(-2.0, 2.0, 5)  # [-2, -1, 0, 1, 2]
        yp = a * xp + b  # [-1, 2, 5, 8, 11]

        for x_truth in [-1.5, -0.5, 0.3, 1.7]:
            y_query = a * x_truth + b
            got = _eval_inverse_then_interp(yp, xp, float(y_query))
            assert got == pytest.approx(x_truth, abs=1e-10), x_truth


# ---------------------------------------------------------------------------
# Differentiability: jax.grad through the query coordinate.
# ---------------------------------------------------------------------------


def _inverse_pure(yp, y_query):
    """Pure-JAX equivalent of the increasing-table PrelookupInverse
    output computation.  Mirrors the closure inside
    ``PrelookupInverse.__init__`` for the strictly-increasing branch.
    """
    n = yp.shape[0]
    i = jnp.clip(jnp.searchsorted(yp, y_query, side="right") - 1, 0, n - 2)
    y0 = yp[i]
    y1 = yp[i + 1]
    alpha = (y_query - y0) / (y1 - y0)
    alpha = jnp.clip(alpha, 0.0, 1.0)
    return i, alpha


def _interp_pure(xp, idx_alpha):
    i, alpha = idx_alpha
    return (1.0 - alpha) * xp[i] + alpha * xp[i + 1]


class TestPrelookupInverseDifferentiability:
    def test_grad_through_query_via_pure_jax(self):
        """jax.grad of (PrelookupInverse -> Interp(xp)) w.r.t. the y
        query.

        With yp = x^2 and the interp table = xp, inside the bucket
        [yp[i], yp[i+1]] we get the piecewise-linear inverse
        x = xp[i] + alpha * (xp[i+1] - xp[i]) with
        alpha = (y - yp[i]) / (yp[i+1] - yp[i]).

        So d x / d y = (xp[i+1] - xp[i]) / (yp[i+1] - yp[i]).
        For y=6.5 between yp[2]=4 and yp[3]=9:
            d/dy = (3 - 2) / (9 - 4) = 0.2
        """
        xp = jnp.array([0.0, 1.0, 2.0, 3.0, 4.0])
        yp = xp ** 2

        def f(y):
            return _interp_pure(xp, _inverse_pure(yp, y))

        g = float(jax.grad(f)(jnp.asarray(6.5)))
        assert g == pytest.approx(0.2, abs=1e-10)

        # Bucket [yp[0]=0, yp[1]=1]: d/dy = (1-0)/(1-0) = 1.0.
        g = float(jax.grad(f)(jnp.asarray(0.5)))
        assert g == pytest.approx(1.0, abs=1e-10)

    def test_grad_through_table_values(self):
        """jax.grad of the inverse-then-interp pipeline w.r.t. the
        downstream xp table: at a query inside bucket i with weight
        alpha, df/dxp[i] = 1-alpha and df/dxp[i+1] = alpha.
        """
        yp = jnp.array([0.0, 1.0, 4.0, 9.0, 16.0])

        def f(xp):
            # y=6.5 -> bucket [yp[2]=4, yp[3]=9], alpha = 0.5
            return _interp_pure(xp, _inverse_pure(yp, jnp.asarray(6.5)))

        xp0 = jnp.zeros(5)
        g = jax.grad(f)(xp0)
        assert g.shape == xp0.shape
        np.testing.assert_allclose(np.asarray(g), [0.0, 0.0, 0.5, 0.5, 0.0])

    def test_jit_traces_through_inverse_pipeline(self):
        """jax.jit must handle the inverse pipeline cleanly."""
        xp = jnp.array([0.0, 1.0, 2.0, 3.0, 4.0])
        yp = xp ** 2

        @jax.jit
        def f(y):
            return _interp_pure(xp, _inverse_pure(yp, y))

        # y=4 hits a breakpoint exactly -> x=2.0.
        out = float(f(jnp.asarray(4.0)))
        assert out == pytest.approx(2.0, abs=1e-10)


# ---------------------------------------------------------------------------
# NamedTuple pytree contract -- pin that PrelookupInverse emits a
# _PrelookupResult, same shape as Prelookup, so downstream
# InterpolationUsingPrelookup blocks accept it interchangeably.
# ---------------------------------------------------------------------------


class TestPrelookupInverseEmitsNamedTuple:
    def test_output_has_index_and_fraction_fields(self):
        yp = jnp.array([0.0, 1.0, 4.0, 9.0, 16.0])
        builder = jaxonomy.DiagramBuilder()
        src = builder.add(library.Constant(jnp.asarray(6.5)))
        inv = builder.add(PrelookupInverse(yp))
        builder.connect(src.output_ports[0], inv.input_ports[0])
        diagram = builder.build()
        ctx = diagram.create_context()
        result = inv.output_ports[0].eval(ctx)
        assert hasattr(result, "index")
        assert hasattr(result, "fraction")
        assert int(result.index) == 2
        assert float(result.fraction) == pytest.approx(0.5, abs=1e-12)

    def test_result_is_same_type_as_prelookup(self):
        """Confirm PrelookupInverse emits the same _PrelookupResult
        NamedTuple as Prelookup -- so an InterpolationUsingPrelookup
        connected downstream cannot tell them apart structurally.
        """
        from jaxonomy.library.primitives import _PrelookupResult

        yp = jnp.array([0.0, 1.0, 4.0, 9.0, 16.0])
        builder = jaxonomy.DiagramBuilder()
        src = builder.add(library.Constant(jnp.asarray(2.5)))
        inv = builder.add(PrelookupInverse(yp))
        builder.connect(src.output_ports[0], inv.input_ports[0])
        diagram = builder.build()
        ctx = diagram.create_context()
        result = inv.output_ports[0].eval(ctx)
        assert isinstance(result, _PrelookupResult)
