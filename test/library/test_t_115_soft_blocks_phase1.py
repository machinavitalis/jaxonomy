# SPDX-License-Identifier: MIT

"""T-115 phase 1 tests: SoftSaturate (and SoftRateLimiter) smooth blocks.

These cover the differentiable-clip blocks that complement the existing
hard ``Saturate`` / ``RateLimiter`` primitives. The critical test is that
gradients flow through the saturated region, where the hard variant
returns exactly zero.
"""

import pytest

import numpy as np
import jax
import jax.numpy as jnp

import jaxonomy
from jaxonomy import library
from jaxonomy.framework.error import BlockParameterError
from jaxonomy.library.primitives import soft_saturate

pytestmark = pytest.mark.minimal


# ---------------------------------------------------------------------------
# soft_saturate functional helper
# ---------------------------------------------------------------------------


class TestSoftSaturateFunction:
    def test_midrange_near_identity(self):
        # In the middle of the range, soft_saturate is approximately identity
        # for moderate sharpness.
        y = soft_saturate(jnp.array(0.5), 0.0, 1.0, sharpness=20.0)
        assert jnp.isclose(y, 0.5, atol=1e-6)

    def test_above_upper_near_upper(self):
        y = soft_saturate(jnp.array(2.0), 0.0, 1.0, sharpness=20.0)
        # tanh(20 * 1.5 / 0.5) = tanh(60) ~ 1, so y ~ upper
        assert jnp.isclose(y, 1.0, atol=1e-6)

    def test_below_lower_near_lower(self):
        y = soft_saturate(jnp.array(-1.0), 0.0, 1.0, sharpness=20.0)
        assert jnp.isclose(y, 0.0, atol=1e-6)

    def test_monotone_increasing(self):
        # Across the active region the output must be strictly monotone.
        # Far outside the bounds tanh underflows to a constant, so we
        # only check the active band where the function actually varies.
        u = jnp.linspace(-0.5, 1.5, 101)
        y = soft_saturate(u, 0.0, 1.0, sharpness=10.0)
        assert jnp.all(jnp.diff(y) > 0)

    def test_gradient_nonzero_in_saturated_region(self):
        """The whole point of SoftSaturate: gradient flows past the limits."""

        def f(x):
            return jnp.sum(soft_saturate(x, 0.0, 1.0, sharpness=10.0))

        # Hard clip would give gradient == 0.0 here.
        g_above = jax.grad(f)(jnp.array(2.0))
        g_below = jax.grad(f)(jnp.array(-1.0))
        g_mid = jax.grad(f)(jnp.array(0.5))

        assert jnp.abs(g_above) > 0.0
        assert jnp.abs(g_below) > 0.0
        assert g_mid > 0.0
        # Gradient at midpoint should be near sharpness * (gain==1) for
        # default mid=(lo+hi)/2; the tanh slope at zero is `sharpness`.
        # Here u-mid=0 -> dy/du = sharpness * (1 - 0) = sharpness.
        # But our formula has half = (hi-lo)/2 = 0.5; check derivative
        # numerically.
        eps = 1e-4
        finite_diff = (
            soft_saturate(jnp.array(0.5 + eps), 0.0, 1.0, sharpness=10.0)
            - soft_saturate(jnp.array(0.5 - eps), 0.0, 1.0, sharpness=10.0)
        ) / (2 * eps)
        assert jnp.isclose(g_mid, finite_diff, atol=1e-3)

    def test_sharpness_low_near_identity_in_middle(self):
        # With low sharpness (and thus a gentle tanh slope of
        # ``sharpness/2`` at the midpoint) the smooth saturate is
        # approximately the identity in the immediate neighborhood of
        # the midpoint. This is the "differentiable identity in the
        # interior" property that lets gradients flow.
        sharpness = 2.0
        u = jnp.linspace(0.45, 0.55, 11)
        y = soft_saturate(u, 0.0, 1.0, sharpness=sharpness)
        # Slope at mid is `sharpness/2 = 1`, so the local linearization
        # is exactly the identity through (mid, mid). Allow modest
        # tolerance for tanh curvature.
        assert jnp.allclose(y, u, atol=5e-3)

    def test_sharpness_to_inf_matches_clip_outside(self):
        # Outside the bounds, large sharpness drives the output to the
        # nearer bound -- which is exactly what hard ``clip`` returns
        # there.
        u = jnp.array([-2.0, 2.0, 5.0])
        y_soft = soft_saturate(u, 0.0, 1.0, sharpness=200.0)
        y_hard = jnp.clip(u, 0.0, 1.0)
        assert jnp.allclose(y_soft, y_hard, atol=1e-3)


# ---------------------------------------------------------------------------
# SoftSaturate block
# ---------------------------------------------------------------------------


def _run_soft_saturate(input_value, lower=0.0, upper=1.0, sharpness=10.0):
    """Build a tiny diagram with one SoftSaturate fed by a constant."""
    builder = jaxonomy.DiagramBuilder()
    src = builder.add(library.Constant(value=float(input_value)))
    blk = builder.add(
        library.SoftSaturate(
            lower_limit=lower, upper_limit=upper, sharpness=sharpness
        )
    )
    builder.connect(src.output_ports[0], blk.input_ports[0])
    diagram = builder.build()
    context = diagram.create_context()
    return float(blk.output_ports[0].eval(context))


class TestSoftSaturateBlock:
    def test_forward_midrange(self):
        y = _run_soft_saturate(0.5, 0.0, 1.0, sharpness=20.0)
        assert np.isclose(y, 0.5, atol=1e-6)

    def test_forward_above(self):
        y = _run_soft_saturate(2.0, 0.0, 1.0, sharpness=20.0)
        assert np.isclose(y, 1.0, atol=1e-6)

    def test_forward_below(self):
        y = _run_soft_saturate(-1.0, 0.0, 1.0, sharpness=20.0)
        assert np.isclose(y, 0.0, atol=1e-6)

    def test_invalid_bounds(self):
        with pytest.raises(BlockParameterError):
            library.SoftSaturate(lower_limit=1.0, upper_limit=0.0)

    def test_invalid_sharpness(self):
        with pytest.raises(BlockParameterError):
            library.SoftSaturate(
                lower_limit=0.0, upper_limit=1.0, sharpness=-1.0
            )

    def test_infinite_bounds_rejected(self):
        with pytest.raises(BlockParameterError):
            library.SoftSaturate(lower_limit=-np.inf, upper_limit=1.0)


# ---------------------------------------------------------------------------
# Hard Saturate smoke test: existing block default-path unchanged.
# ---------------------------------------------------------------------------


class TestHardSaturateSmoke:
    def test_default_clip_unchanged(self):
        builder = jaxonomy.DiagramBuilder()
        slope = 1.0
        ramp = builder.add(library.Ramp(start_value=-2.0, slope=slope, start_time=0.0))
        sat = builder.add(library.Saturate(lower_limit=-1.0, upper_limit=1.0))
        builder.connect(ramp.output_ports[0], sat.input_ports[0])
        diagram = builder.build()
        context = diagram.create_context()
        recorded = {"y": sat.output_ports[0]}
        r = jaxonomy.simulate(diagram, context, (0.0, 4.0), recorded_signals=recorded)
        # input goes -2..2 over 4s; clipped to [-1, 1]
        expected = np.clip(-2.0 + r.time * slope, -1.0, 1.0)
        assert np.allclose(np.asarray(r.outputs["y"]), expected, atol=1e-6)


# ---------------------------------------------------------------------------
# SoftRateLimiter
# ---------------------------------------------------------------------------


class TestSoftRateLimiter:
    def test_constructs_smoke(self):
        # Smoke: build a diagram and confirm it simulates without error.
        # (Cache is initialized to the input value, so a constant input
        # never trips the rate limit; that's the same behavior as the
        # hard RateLimiter and is fine for a smoke test.)
        dt = 0.1
        builder = jaxonomy.DiagramBuilder()
        src = builder.add(library.Constant(value=0.0))
        rl = builder.add(
            library.SoftRateLimiter(
                dt=dt, upper_limit=1.0, lower_limit=-1.0, sharpness=20.0
            )
        )
        builder.connect(src.output_ports[0], rl.input_ports[0])
        diagram = builder.build()
        context = diagram.create_context()
        recorded = {"y": rl.output_ports[0]}
        r = jaxonomy.simulate(diagram, context, (0.0, 0.5), recorded_signals=recorded)
        ys = np.asarray(r.outputs["y"])
        assert np.allclose(ys, 0.0, atol=1e-6)

    def test_gradient_through_limit(self):
        """Gradient of one rate-limit step w.r.t. input is nonzero even
        when the requested step is past the rate limit. Hard
        RateLimiter would give exactly zero here."""
        dt = 0.1
        ulim = 1.0
        llim = -1.0
        sharpness = 10.0

        def step(u, y_prev):
            delta = u - y_prev
            delta_lo = dt * llim
            delta_hi = dt * ulim
            return y_prev + soft_saturate(delta, delta_lo, delta_hi, sharpness)

        # u=0.3, y_prev=0 -> requested delta=0.3 > dt*ulim=0.1, so we
        # are in the saturated region. With moderate sharpness the
        # tanh derivative is small but strictly positive in float64.
        g = jax.grad(lambda u: step(u, 0.0))(jnp.array(0.3))
        assert jnp.abs(g) > 0.0
