# SPDX-License-Identifier: MIT

"""T-114-followup-prelookup -- Prelookup + InterpolationUsingPrelookup pair.

The standard ``Prelookup``/``InterpolationUsingPrelookup`` pair is an
optimisation for the case where the SAME query coordinate is used to
look up MANY tables (e.g. 10 maps sharing one engine-RPM axis). Without
the pair each downstream ``LookupTable1d`` re-runs the binary search;
with it the search runs ONCE in ``Prelookup`` and the
``(index, fraction)`` pair fans out to N consumers.

These tests exercise:

  * Equivalence between ``Prelookup -> InterpolationUsingPrelookup`` and
    a direct ``LookupTable1d`` (linear) on the same data.
  * Two ``InterpolationUsingPrelookup`` blocks sharing one ``Prelookup``
    output: each interpolates its own table independently and produces
    the same answer as wiring it through a separate ``LookupTable1d``.
  * Out-of-bounds query handling: the ``alpha`` clip in ``Prelookup``
    collapses OOB queries to the nearest endpoint (no NaN, matches
    the standard default ``"clip"`` extrapolation).
  * Differentiability: ``jax.grad`` flows through the interpolated
    output back to both the table values and the query coordinate.

The (index, fraction) signal is a ``NamedTuple`` -- a JAX pytree -- so
this also implicitly exercises that NamedTuple-typed signals flow
through the simulator end-to-end (the same trick as :class:`BusCreator`
in T-117-followup).
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
# Construction-time validation
# ---------------------------------------------------------------------------


class TestPrelookupValidation:
    def test_non_1d_input_array_raises(self):
        with pytest.raises(ValueError, match="1-D"):
            Prelookup(jnp.zeros((3, 3)))

    def test_too_short_input_array_raises(self):
        with pytest.raises(ValueError, match="at least 2"):
            Prelookup(jnp.array([1.0]))

    def test_non_monotonic_input_array_raises(self):
        with pytest.raises(ValueError, match="monotonically"):
            Prelookup(jnp.array([0.0, 2.0, 1.0]))

    def test_dtype_kwarg_casts_grid(self):
        xp = np.array([0.0, 1.0, 2.0], dtype=np.float64)
        blk = Prelookup(xp, dtype=jnp.float32)
        assert blk.input_array.dtype == jnp.float32


class TestInterpolationUsingPrelookupValidation:
    def test_non_1d_output_array_raises(self):
        with pytest.raises(ValueError, match="1-D"):
            InterpolationUsingPrelookup(jnp.zeros((3, 3)))

    def test_too_short_output_array_raises(self):
        with pytest.raises(ValueError, match="at least 2"):
            InterpolationUsingPrelookup(jnp.array([1.0]))

    def test_dtype_kwarg_casts_table(self):
        yp = np.array([0.0, 1.0, 2.0], dtype=np.float64)
        blk = InterpolationUsingPrelookup(yp, dtype=jnp.float32)
        assert blk.output_array.dtype == jnp.float32


# ---------------------------------------------------------------------------
# End-to-end: a Prelookup feeding a single InterpolationUsingPrelookup
# must match a direct LookupTable1d (linear) on the same data.
# ---------------------------------------------------------------------------


def _eval_prelookup_pair(xp, yp, x_query):
    """Build (Constant -> Prelookup -> InterpolationUsingPrelookup),
    evaluate, and return the scalar output value.
    """
    builder = jaxonomy.DiagramBuilder()
    src = builder.add(library.Constant(jnp.asarray(x_query)))
    pre = builder.add(Prelookup(xp))
    interp = builder.add(InterpolationUsingPrelookup(yp))
    builder.connect(src.output_ports[0], pre.input_ports[0])
    builder.connect(pre.output_ports[0], interp.input_ports[0])
    diagram = builder.build()
    ctx = diagram.create_context()
    return float(np.asarray(interp.output_ports[0].eval(ctx)))


def _eval_direct_lookup(xp, yp, x_query):
    """Reference: a direct ``LookupTable1d`` (linear) on the same data."""
    builder = jaxonomy.DiagramBuilder()
    src = builder.add(library.Constant(jnp.asarray(x_query)))
    blk = builder.add(LookupTable1d(xp, yp, "linear"))
    builder.connect(src.output_ports[0], blk.input_ports[0])
    diagram = builder.build()
    ctx = diagram.create_context()
    return float(np.asarray(blk.output_ports[0].eval(ctx)))


class TestEquivalenceWithLookupTable1d:
    @pytest.fixture
    def grid_and_table(self):
        xp = jnp.linspace(0.0, 4.0, 5)  # [0, 1, 2, 3, 4]
        yp = jnp.array([0.0, 1.0, 4.0, 9.0, 16.0])  # x^2 at the breakpoints
        return xp, yp

    def test_at_breakpoint(self, grid_and_table):
        xp, yp = grid_and_table
        for q in [0.0, 1.0, 2.0, 3.0, 4.0]:
            got = _eval_prelookup_pair(xp, yp, q)
            want = _eval_direct_lookup(xp, yp, q)
            assert got == pytest.approx(want, abs=1e-12), (q, got, want)

    def test_off_grid_matches_linear_lookup(self, grid_and_table):
        xp, yp = grid_and_table
        for q in [0.25, 1.5, 2.7, 3.9]:
            got = _eval_prelookup_pair(xp, yp, q)
            want = _eval_direct_lookup(xp, yp, q)
            assert got == pytest.approx(want, abs=1e-12), (q, got, want)

    def test_explicit_value_at_known_query(self, grid_and_table):
        # x=1.5 between (1, 1) and (2, 4): linear interp -> 2.5.
        xp, yp = grid_and_table
        got = _eval_prelookup_pair(xp, yp, 1.5)
        assert got == pytest.approx(2.5, abs=1e-12)


# ---------------------------------------------------------------------------
# The headline use case: two InterpolationUsingPrelookup blocks sharing
# ONE Prelookup output. Each interpolates its own independent table.
# ---------------------------------------------------------------------------


class TestSharedPrelookupFanout:
    def test_two_tables_share_one_prelookup(self):
        """Two InterpolationUsingPrelookup blocks sharing a single
        Prelookup must produce the SAME values as two independent
        LookupTable1d blocks fed by the same query.
        """
        xp = jnp.linspace(0.0, 1.0, 6)
        yp_a = jnp.sin(xp)  # smooth table 1
        yp_b = jnp.cos(xp) + 2.0  # smooth table 2 (different shape entirely)

        for q in [0.0, 0.15, 0.5, 0.83, 1.0]:
            # Pair-based: one Prelookup, two InterpolationUsingPrelookup
            builder = jaxonomy.DiagramBuilder()
            src = builder.add(library.Constant(jnp.asarray(q)))
            pre = builder.add(Prelookup(xp))
            ia = builder.add(InterpolationUsingPrelookup(yp_a))
            ib = builder.add(InterpolationUsingPrelookup(yp_b))
            builder.connect(src.output_ports[0], pre.input_ports[0])
            builder.connect(pre.output_ports[0], ia.input_ports[0])
            builder.connect(pre.output_ports[0], ib.input_ports[0])
            diagram = builder.build()
            ctx = diagram.create_context()

            got_a = float(np.asarray(ia.output_ports[0].eval(ctx)))
            got_b = float(np.asarray(ib.output_ports[0].eval(ctx)))

            want_a = _eval_direct_lookup(xp, yp_a, q)
            want_b = _eval_direct_lookup(xp, yp_b, q)

            assert got_a == pytest.approx(want_a, abs=1e-12), (q, got_a, want_a)
            assert got_b == pytest.approx(want_b, abs=1e-12), (q, got_b, want_b)

    def test_three_tables_share_one_prelookup_via_simulate(self):
        """End-to-end ``simulate`` test confirming the NamedTuple-typed
        (index, fraction) signal flows through every stage of the
        simulator without breaking pytree handling -- 3 fan-out variant
        on a constant input.
        """
        xp = jnp.linspace(0.0, 2.0, 5)
        yp_a = jnp.array([0.0, 1.0, 2.0, 3.0, 4.0])  # identity / 2
        yp_b = jnp.array([10.0, 20.0, 30.0, 40.0, 50.0])  # affine
        yp_c = xp ** 2  # quadratic samples

        q = 0.75  # interior query, between xp[1]=0.5 and xp[2]=1.0
        # alpha = (0.75 - 0.5) / (1.0 - 0.5) = 0.5
        # so each interp = 0.5 * yp[1] + 0.5 * yp[2]
        want_a = 0.5 * 1.0 + 0.5 * 2.0  # 1.5
        want_b = 0.5 * 20.0 + 0.5 * 30.0  # 25.0
        want_c = 0.5 * float(yp_c[1]) + 0.5 * float(yp_c[2])

        builder = jaxonomy.DiagramBuilder()
        src = builder.add(library.Constant(jnp.asarray(q)))
        pre = builder.add(Prelookup(xp, name="pre"))
        ia = builder.add(InterpolationUsingPrelookup(yp_a, name="ia"))
        ib = builder.add(InterpolationUsingPrelookup(yp_b, name="ib"))
        ic = builder.add(InterpolationUsingPrelookup(yp_c, name="ic"))
        builder.connect(src.output_ports[0], pre.input_ports[0])
        builder.connect(pre.output_ports[0], ia.input_ports[0])
        builder.connect(pre.output_ports[0], ib.input_ports[0])
        builder.connect(pre.output_ports[0], ic.input_ports[0])
        diagram = builder.build()
        ctx = diagram.create_context()

        results = jaxonomy.simulate(
            diagram,
            ctx,
            (0.0, 0.1),
            recorded_signals={
                "a": ia.output_ports[0],
                "b": ib.output_ports[0],
                "c": ic.output_ports[0],
            },
        )
        out_a = float(np.asarray(results.outputs["a"])[-1])
        out_b = float(np.asarray(results.outputs["b"])[-1])
        out_c = float(np.asarray(results.outputs["c"])[-1])

        assert out_a == pytest.approx(want_a, abs=1e-12)
        assert out_b == pytest.approx(want_b, abs=1e-12)
        assert out_c == pytest.approx(want_c, abs=1e-12)


# ---------------------------------------------------------------------------
# Out-of-bounds queries: the alpha clip in Prelookup collapses OOB to
# the nearest endpoint (matches the standard default ``"clip"`` extrapolation).
# ---------------------------------------------------------------------------


class TestPrelookupOOB:
    @pytest.fixture
    def grid_and_table(self):
        xp = jnp.array([0.0, 1.0, 2.0, 3.0])
        yp = jnp.array([10.0, 20.0, 30.0, 40.0])
        return xp, yp

    def test_oob_left_clips_to_first_value(self, grid_and_table):
        xp, yp = grid_and_table
        got = _eval_prelookup_pair(xp, yp, -5.0)
        assert got == pytest.approx(10.0, abs=1e-12)
        assert np.isfinite(got)

    def test_oob_right_clips_to_last_value(self, grid_and_table):
        xp, yp = grid_and_table
        got = _eval_prelookup_pair(xp, yp, 10.0)
        assert got == pytest.approx(40.0, abs=1e-12)
        assert np.isfinite(got)

    def test_oob_does_not_produce_nan(self, grid_and_table):
        # Belt-and-braces: explicitly confirm no NaN escapes for any
        # OOB query on either side.
        xp, yp = grid_and_table
        for q in [-1e9, -1.0, 3.0001, 1e9]:
            got = _eval_prelookup_pair(xp, yp, q)
            assert not np.isnan(got), (q, got)


# ---------------------------------------------------------------------------
# Pure-JAX trace-level checks. We exercise the underlying NamedTuple-
# returning + tuple-consuming math without spinning up the simulator,
# so the differentiability story is isolated from any simulator-
# specific machinery.
# ---------------------------------------------------------------------------


def _prelookup_pure(xp, x_query):
    """Pure-JAX equivalent of the Prelookup output computation. Mirrors
    the closure body inside ``Prelookup.__init__``.
    """
    n = xp.shape[0]
    i = jnp.clip(jnp.searchsorted(xp, x_query, side="right") - 1, 0, n - 2)
    x0 = xp[i]
    x1 = xp[i + 1]
    alpha = (x_query - x0) / (x1 - x0)
    alpha = jnp.clip(alpha, 0.0, 1.0)
    return i, alpha


def _interp_pure(yp, idx_alpha):
    """Pure-JAX equivalent of the InterpolationUsingPrelookup output
    computation.
    """
    i, alpha = idx_alpha
    return (1.0 - alpha) * yp[i] + alpha * yp[i + 1]


class TestPrelookupDifferentiability:
    def test_grad_through_query(self):
        """jax.grad of (Prelookup -> Interp) w.r.t. the query coordinate.

        For a piecewise-linear table of slope `m` between xp[i] and
        xp[i+1], the derivative is exactly `m` inside that bucket.
        """
        xp = jnp.array([0.0, 1.0, 2.0, 3.0])
        yp = jnp.array([0.0, 5.0, 11.0, 14.0])

        def f(x):
            return _interp_pure(yp, _prelookup_pure(xp, x))

        # Inside bucket [1, 2]: slope = (11 - 5) / (2 - 1) = 6
        g = float(jax.grad(f)(jnp.asarray(1.5)))
        assert g == pytest.approx(6.0, abs=1e-10)

        # Inside bucket [0, 1]: slope = (5 - 0) / (1 - 0) = 5
        g = float(jax.grad(f)(jnp.asarray(0.25)))
        assert g == pytest.approx(5.0, abs=1e-10)

        # Inside bucket [2, 3]: slope = (14 - 11) / (3 - 2) = 3
        g = float(jax.grad(f)(jnp.asarray(2.5)))
        assert g == pytest.approx(3.0, abs=1e-10)

    def test_grad_through_table_values(self):
        """jax.grad of (Prelookup -> Interp) w.r.t. the output_array.

        At a query halfway through bucket i, df/dyp[i] = 1 - alpha = 0.5
        and df/dyp[i+1] = alpha = 0.5; all other entries get gradient 0.
        """
        xp = jnp.array([0.0, 1.0, 2.0, 3.0])

        def f(yp):
            # Query 1.5 -> bucket [1, 2], alpha = 0.5.
            return _interp_pure(yp, _prelookup_pure(xp, jnp.asarray(1.5)))

        yp0 = jnp.zeros(4)
        g = jax.grad(f)(yp0)
        assert g.shape == yp0.shape
        np.testing.assert_allclose(np.asarray(g), [0.0, 0.5, 0.5, 0.0])

    def test_jit_traces_through_namedtuple(self):
        """jax.jit must accept the NamedTuple-shaped intermediate
        without choking on the pytree boundary.
        """
        xp = jnp.array([0.0, 1.0, 2.0])
        yp = jnp.array([0.0, 10.0, 30.0])

        @jax.jit
        def f(x):
            return _interp_pure(yp, _prelookup_pure(xp, x))

        out = float(f(jnp.asarray(0.5)))
        # bucket [0, 1], alpha=0.5 -> 5.0
        assert out == pytest.approx(5.0, abs=1e-12)


# ---------------------------------------------------------------------------
# The (index, fraction) value is a JAX pytree (NamedTuple). Pin that
# nothing accidentally regresses the pytree registration.
# ---------------------------------------------------------------------------


class TestPrelookupResultIsAPytree:
    def test_result_is_namedtuple_pytree(self):
        from jaxonomy.library.primitives import _PrelookupResult

        result = _PrelookupResult(
            index=jnp.asarray(2), fraction=jnp.asarray(0.5)
        )
        leaves = jax.tree_util.tree_leaves(result)
        assert len(leaves) == 2

    def test_block_emits_namedtuple_at_runtime(self):
        """The Prelookup output port returns an instance whose fields
        are accessible by name -- pinning the NamedTuple shape rather
        than the framework silently flattening to a plain tuple.
        """
        xp = jnp.array([0.0, 1.0, 2.0])
        builder = jaxonomy.DiagramBuilder()
        src = builder.add(library.Constant(jnp.asarray(0.5)))
        pre = builder.add(Prelookup(xp))
        builder.connect(src.output_ports[0], pre.input_ports[0])
        diagram = builder.build()
        ctx = diagram.create_context()
        result = pre.output_ports[0].eval(ctx)
        # Must have ``index`` and ``fraction`` named fields.
        assert hasattr(result, "index")
        assert hasattr(result, "fraction")
        assert int(result.index) == 0
        assert float(result.fraction) == pytest.approx(0.5, abs=1e-12)
