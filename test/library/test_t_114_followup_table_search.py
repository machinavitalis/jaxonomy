# SPDX-License-Identifier: MIT

"""T-114-followup-table-search -- TableSearch block.

The standard "Direct Lookup" / "Search" pattern: given a strictly-
monotonic 1-D table ``xp`` and a scalar query ``x``, return the bucket
index ``i`` such that ``xp[i] <= x < xp[i+1]``.  Different from
:class:`Prelookup` in that the output is just the index (no fractional
alpha).

These tests exercise:

  * Construction-time validation: non-1-D / too-short / non-monotonic
    grids raise ``ValueError``.
  * Basic index correctness: in-range queries return the expected
    bucket.
  * OOB clamping: queries below ``xp[0]`` return ``0``; queries at or
    beyond ``xp[-1]`` return ``n - 1``.
  * Mode agreement: ``"binary"`` and ``"linear"`` modes return
    byte-identical results.
  * Gradient is stopped: ``jax.grad`` through the output returns 0 by
    construction (the index is a step function; we wrap the output in
    ``jax.lax.stop_gradient`` to make that explicit).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import jaxonomy
from jaxonomy import library
from jaxonomy.library import TableSearch


# ---------------------------------------------------------------------------
# Construction-time validation
# ---------------------------------------------------------------------------


class TestTableSearchValidation:
    def test_non_1d_xp_raises(self):
        with pytest.raises(ValueError, match="1-D"):
            TableSearch(jnp.zeros((3, 3)))

    def test_too_short_xp_raises(self):
        with pytest.raises(ValueError, match="at least 2"):
            TableSearch(jnp.array([1.0]))

    def test_non_monotonic_xp_raises(self):
        with pytest.raises(ValueError, match="monotonically"):
            TableSearch(jnp.array([0.0, 2.0, 1.0]))

    def test_non_strictly_increasing_xp_raises(self):
        # Equal-adjacent entries are not strictly increasing.
        with pytest.raises(ValueError, match="monotonically"):
            TableSearch(jnp.array([0.0, 1.0, 1.0, 2.0]))

    def test_bad_mode_raises(self):
        with pytest.raises(ValueError, match="mode"):
            TableSearch(jnp.array([0.0, 1.0, 2.0]), mode="cubic")

    def test_dtype_kwarg_casts_grid(self):
        xp = np.array([0.0, 1.0, 2.0], dtype=np.float64)
        blk = TableSearch(xp, dtype=jnp.float32)
        assert blk.xp.dtype == jnp.float32

    def test_mode_property(self):
        blk_b = TableSearch(jnp.array([0.0, 1.0, 2.0]), mode="binary")
        blk_l = TableSearch(jnp.array([0.0, 1.0, 2.0]), mode="linear")
        assert blk_b.mode == "binary"
        assert blk_l.mode == "linear"


# ---------------------------------------------------------------------------
# Helpers: build a (Constant -> TableSearch) diagram and evaluate.
# ---------------------------------------------------------------------------


def _eval_table_search(xp, x_query, mode="binary"):
    builder = jaxonomy.DiagramBuilder()
    src = builder.add(library.Constant(jnp.asarray(x_query)))
    blk = builder.add(TableSearch(xp, mode=mode))
    builder.connect(src.output_ports[0], blk.input_ports[0])
    diagram = builder.build()
    ctx = diagram.create_context()
    return float(np.asarray(blk.output_ports[0].eval(ctx)))


# ---------------------------------------------------------------------------
# Basic correctness on an in-range query.
# ---------------------------------------------------------------------------


class TestTableSearchInRange:
    @pytest.fixture
    def xp(self):
        return jnp.array([0.0, 1.0, 2.0, 3.0, 4.0])

    def test_query_1_5_returns_1(self, xp):
        # 1.5 lies between xp[1]=1 and xp[2]=2 -> bucket 1.
        got = _eval_table_search(xp, 1.5)
        assert got == pytest.approx(1.0, abs=1e-12)

    def test_query_2_5_returns_2(self, xp):
        # 2.5 lies between xp[2]=2 and xp[3]=3 -> bucket 2.
        got = _eval_table_search(xp, 2.5)
        assert got == pytest.approx(2.0, abs=1e-12)

    def test_query_at_breakpoint_returns_that_bucket(self, xp):
        # Exactly at a breakpoint: side="right" + (-1) -> the index of
        # the breakpoint itself, NOT one less. xp[2] = 2.0 -> bucket 2.
        got = _eval_table_search(xp, 2.0)
        assert got == pytest.approx(2.0, abs=1e-12)

    def test_query_at_first_breakpoint(self, xp):
        # x == xp[0]: bucket 0.
        got = _eval_table_search(xp, 0.0)
        assert got == pytest.approx(0.0, abs=1e-12)

    @pytest.mark.parametrize(
        "q, want",
        [
            (0.1, 0.0),
            (0.9, 0.0),
            (1.0, 1.0),
            (1.5, 1.0),
            (2.9, 2.0),
            (3.5, 3.0),
        ],
    )
    def test_various_queries(self, xp, q, want):
        got = _eval_table_search(xp, q)
        assert got == pytest.approx(want, abs=1e-12), (q, got, want)


# ---------------------------------------------------------------------------
# OOB clamping: left -> 0, right -> n - 1.
# ---------------------------------------------------------------------------


class TestTableSearchOOB:
    @pytest.fixture
    def xp(self):
        return jnp.array([0.0, 1.0, 2.0, 3.0, 4.0])

    def test_query_below_left_endpoint_returns_0(self, xp):
        # x < xp[0]: clamp to bucket 0.
        got = _eval_table_search(xp, -10.0)
        assert got == pytest.approx(0.0, abs=1e-12)

    def test_query_above_right_endpoint_returns_n_minus_1(self, xp):
        # x >> xp[-1]: clamp to bucket n - 1 == 4.
        got = _eval_table_search(xp, 100.0)
        assert got == pytest.approx(4.0, abs=1e-12)

    def test_query_at_right_endpoint(self, xp):
        # x == xp[-1]: side="right" of a strict ascending search puts
        # the cursor past the last entry, then -1 lands on n - 1.
        got = _eval_table_search(xp, 4.0)
        assert got == pytest.approx(4.0, abs=1e-12)


# ---------------------------------------------------------------------------
# Mode agreement: "binary" and "linear" must return identical results
# on every test query.
# ---------------------------------------------------------------------------


class TestModeAgreement:
    @pytest.fixture
    def xp(self):
        # Non-uniform grid -- both modes must agree even when the
        # spacing is irregular.
        return jnp.array([0.0, 0.5, 1.7, 3.1, 4.2, 9.0])

    @pytest.mark.parametrize(
        "q",
        [
            -5.0,    # OOB left
            0.0,     # at xp[0]
            0.25,    # bucket 0
            0.5,     # at xp[1]
            1.0,     # bucket 1
            1.7,     # at xp[2]
            2.5,     # bucket 2
            3.1,     # at xp[3]
            4.2,     # at xp[4]
            7.0,     # bucket 4
            9.0,     # at xp[-1] (right boundary)
            100.0,   # OOB right
        ],
    )
    def test_binary_and_linear_agree(self, xp, q):
        got_binary = _eval_table_search(xp, q, mode="binary")
        got_linear = _eval_table_search(xp, q, mode="linear")
        assert got_binary == got_linear, (q, got_binary, got_linear)


# ---------------------------------------------------------------------------
# Gradient is stopped: the index is piecewise-constant, so we wrap the
# output in ``jax.lax.stop_gradient`` -- ``jax.grad`` returns 0.
# ---------------------------------------------------------------------------


class TestTableSearchGradient:
    def test_stop_gradient_yields_zero(self):
        """Confirms ``jax.lax.stop_gradient`` is applied: gradient of
        the bucket index w.r.t. the query coordinate is exactly 0.
        Without ``stop_gradient`` the gradient would still be 0 almost
        everywhere (step function) but JAX might leak NaNs at the
        breakpoints -- ``stop_gradient`` makes the contract explicit.
        """
        xp = jnp.array([0.0, 1.0, 2.0, 3.0, 4.0])
        blk = TableSearch(xp)

        # Build a tiny closure around blk's compute callback.  We
        # invoke the output port directly by wiring a Constant ->
        # TableSearch and asking jax.grad to differentiate through the
        # output w.r.t. the source value.
        def f(x):
            builder = jaxonomy.DiagramBuilder()
            src = builder.add(library.Constant(x))
            ts = builder.add(TableSearch(xp))
            builder.connect(src.output_ports[0], ts.input_ports[0])
            diagram = builder.build()
            ctx = diagram.create_context()
            return ts.output_ports[0].eval(ctx)

        # At an interior query, the gradient must be exactly 0.0
        # (stop_gradient guarantees this even at finite differences).
        g = jax.grad(f)(jnp.asarray(1.5))
        assert float(np.asarray(g)) == 0.0
