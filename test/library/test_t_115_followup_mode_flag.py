# SPDX-License-Identifier: MIT

"""T-115-followup-mode-flag tests.

Verifies the unified ``mode={"hard","smooth"}`` kwarg on the existing
:class:`~jaxonomy.library.Saturate` and :class:`~jaxonomy.library.RateLimiter`
blocks. Default ``mode="hard"`` must remain byte-equivalent to the legacy
phase 1 behavior (including zero-crossing event declaration). The new
``mode="smooth"`` path must match the standalone
:class:`~jaxonomy.library.SoftSaturate` /
:class:`~jaxonomy.library.SoftRateLimiter` outputs and must NOT declare
zero-crossing events.
"""

import pytest

import numpy as np
import jax.numpy as jnp

import jaxonomy
from jaxonomy import library
from jaxonomy.framework.error import BlockParameterError
from jaxonomy.library.primitives import soft_saturate

pytestmark = pytest.mark.minimal


# ---------------------------------------------------------------------------
# Saturate: mode="hard" byte-equivalence vs phase 1
# ---------------------------------------------------------------------------


def _eval_saturate(value, *, lower=0.0, upper=1.0, **kwargs):
    builder = jaxonomy.DiagramBuilder()
    src = builder.add(library.Constant(value=float(value)))
    sat = builder.add(
        library.Saturate(lower_limit=lower, upper_limit=upper, **kwargs)
    )
    builder.connect(src.output_ports[0], sat.input_ports[0])
    diagram = builder.build()
    context = diagram.create_context()
    return float(sat.output_ports[0].eval(context)), sat


class TestSaturateModeHardDefault:
    @pytest.mark.parametrize("u", [-2.0, -0.5, 0.0, 0.5, 1.0, 2.0, 3.7])
    def test_default_matches_clip(self, u):
        # Default (no mode kwarg) must be byte-equivalent to npa.clip.
        y, _ = _eval_saturate(u, lower=0.0, upper=1.0)
        assert y == float(np.clip(u, 0.0, 1.0))

    @pytest.mark.parametrize("u", [-2.0, 0.5, 2.0])
    def test_explicit_hard_matches_default(self, u):
        y_default, _ = _eval_saturate(u, lower=0.0, upper=1.0)
        y_hard, _ = _eval_saturate(u, lower=0.0, upper=1.0, mode="hard")
        assert y_default == y_hard


# ---------------------------------------------------------------------------
# Saturate: mode="smooth" matches SoftSaturate
# ---------------------------------------------------------------------------


def _eval_soft_saturate_block(value, lower=0.0, upper=1.0, sharpness=10.0):
    builder = jaxonomy.DiagramBuilder()
    src = builder.add(library.Constant(value=float(value)))
    sat = builder.add(
        library.SoftSaturate(
            lower_limit=lower, upper_limit=upper, sharpness=sharpness
        )
    )
    builder.connect(src.output_ports[0], sat.input_ports[0])
    diagram = builder.build()
    context = diagram.create_context()
    return float(sat.output_ports[0].eval(context))


class TestSaturateModeSmooth:
    @pytest.mark.parametrize("u", [-1.0, -0.3, 0.0, 0.5, 1.0, 1.7, 3.0])
    def test_smooth_mode_matches_soft_saturate_block(self, u):
        y_unified, _ = _eval_saturate(
            u, lower=0.0, upper=1.0, mode="smooth", sharpness=10.0
        )
        y_soft = _eval_soft_saturate_block(
            u, lower=0.0, upper=1.0, sharpness=10.0
        )
        assert abs(y_unified - y_soft) < 1e-12

    def test_smooth_default_sharpness_is_ten(self):
        # If sharpness is not passed, default 10.0 is used and must match
        # SoftSaturate(..., sharpness=10.0).
        y_unified, _ = _eval_saturate(
            0.5, lower=0.0, upper=1.0, mode="smooth"
        )
        y_soft = _eval_soft_saturate_block(
            0.5, lower=0.0, upper=1.0, sharpness=10.0
        )
        assert abs(y_unified - y_soft) < 1e-12

    def test_smooth_matches_soft_saturate_function(self):
        # The block-level smooth path delegates to soft_saturate(); this
        # is a tighter equality check than just "matches SoftSaturate".
        y_unified, _ = _eval_saturate(
            0.7, lower=0.0, upper=1.0, mode="smooth", sharpness=20.0
        )
        y_func = float(
            soft_saturate(jnp.array(0.7), 0.0, 1.0, sharpness=20.0)
        )
        assert abs(y_unified - y_func) < 1e-12


# ---------------------------------------------------------------------------
# Saturate: zero-crossing introspection
# ---------------------------------------------------------------------------


class TestSaturateZeroCrossings:
    def _build_with_integrator(self, *, mode="hard", **sat_kwargs):
        # The hard Saturate only declares zero-crossing events when
        # ``is_discontinuity(self.output_ports[0])`` is True, which
        # requires (a) the input to depend on time/continuous state and
        # (b) the saturate output to feed an ODE RHS. We therefore drive
        # it with a Ramp (time-dependent) and connect the saturate output
        # into an Integrator.
        builder = jaxonomy.DiagramBuilder()
        ramp = builder.add(
            library.Ramp(start_value=-2.0, slope=1.0, start_time=0.0)
        )
        sat = builder.add(
            library.Saturate(
                lower_limit=0.0, upper_limit=1.0, mode=mode, **sat_kwargs
            )
        )
        integ = builder.add(library.Integrator(0.0))
        builder.connect(ramp.output_ports[0], sat.input_ports[0])
        builder.connect(sat.output_ports[0], integ.input_ports[0])
        diagram = builder.build()
        # Triggering create_context() runs initialize_static_data which is
        # where Saturate decides whether to declare zero-crossing events.
        diagram.create_context()
        return sat

    def test_hard_mode_declares_zero_crossings(self):
        sat = self._build_with_integrator(mode="hard")
        assert sat.has_zero_crossing_events
        # The hard variant declares two events: lower- and upper-limit.
        assert len(sat.zero_crossing_events.events) == 2

    def test_smooth_mode_declares_no_zero_crossings(self):
        sat = self._build_with_integrator(mode="smooth", sharpness=10.0)
        assert not sat.has_zero_crossing_events
        assert len(sat.zero_crossing_events.events) == 0


# ---------------------------------------------------------------------------
# Saturate: validation
# ---------------------------------------------------------------------------


class TestSaturateValidation:
    def test_invalid_mode_raises(self):
        with pytest.raises(BlockParameterError, match="mode"):
            library.Saturate(
                lower_limit=0.0, upper_limit=1.0, mode="invalid"
            )

    def test_smooth_mode_requires_finite_upper(self):
        with pytest.raises(BlockParameterError):
            library.Saturate(
                lower_limit=0.0, upper_limit=np.inf, mode="smooth"
            )

    def test_smooth_mode_requires_finite_lower(self):
        with pytest.raises(BlockParameterError):
            library.Saturate(
                lower_limit=-np.inf, upper_limit=1.0, mode="smooth"
            )

    def test_smooth_mode_requires_positive_sharpness(self):
        with pytest.raises(BlockParameterError):
            library.Saturate(
                lower_limit=0.0,
                upper_limit=1.0,
                mode="smooth",
                sharpness=-1.0,
            )


# ---------------------------------------------------------------------------
# RateLimiter: mode="hard" byte-equivalence
# ---------------------------------------------------------------------------


def _simulate_rate_limiter(*, mode="hard", **kwargs):
    """Drive a rate limiter with a step input and return the recorded output."""
    dt = 0.1
    builder = jaxonomy.DiagramBuilder()
    # Step from 0 -> 1 at t=0.05 so the rate limiter sees a finite delta.
    src = builder.add(library.Step(start_value=0.0, end_value=1.0, step_time=0.05))
    rl = builder.add(
        library.RateLimiter(
            dt=dt,
            upper_limit=1.0,
            lower_limit=-1.0,
            mode=mode,
            **kwargs,
        )
    )
    builder.connect(src.output_ports[0], rl.input_ports[0])
    diagram = builder.build()
    context = diagram.create_context()
    recorded = {"y": rl.output_ports[0]}
    r = jaxonomy.simulate(
        diagram, context, (0.0, 1.0), recorded_signals=recorded
    )
    return np.asarray(r.outputs["y"])


def _simulate_soft_rate_limiter(sharpness=10.0):
    dt = 0.1
    builder = jaxonomy.DiagramBuilder()
    src = builder.add(library.Step(start_value=0.0, end_value=1.0, step_time=0.05))
    rl = builder.add(
        library.SoftRateLimiter(
            dt=dt,
            upper_limit=1.0,
            lower_limit=-1.0,
            sharpness=sharpness,
        )
    )
    builder.connect(src.output_ports[0], rl.input_ports[0])
    diagram = builder.build()
    context = diagram.create_context()
    recorded = {"y": rl.output_ports[0]}
    r = jaxonomy.simulate(
        diagram, context, (0.0, 1.0), recorded_signals=recorded
    )
    return np.asarray(r.outputs["y"])


class TestRateLimiterModeHardDefault:
    def test_default_matches_explicit_hard(self):
        ys_default = _simulate_rate_limiter()
        ys_hard = _simulate_rate_limiter(mode="hard")
        # Byte-equivalent.
        assert np.array_equal(ys_default, ys_hard)


class TestRateLimiterModeSmooth:
    def test_smooth_mode_matches_soft_rate_limiter(self):
        ys_unified = _simulate_rate_limiter(mode="smooth", sharpness=10.0)
        ys_soft = _simulate_soft_rate_limiter(sharpness=10.0)
        # Both block paths use soft_saturate with the same params -> exact
        # match modulo any solver/recording nondeterminism.
        assert np.allclose(ys_unified, ys_soft, atol=1e-12, rtol=0)

    def test_smooth_default_sharpness_is_ten(self):
        ys_unified = _simulate_rate_limiter(mode="smooth")
        ys_soft = _simulate_soft_rate_limiter(sharpness=10.0)
        assert np.allclose(ys_unified, ys_soft, atol=1e-12, rtol=0)


# ---------------------------------------------------------------------------
# RateLimiter: validation
# ---------------------------------------------------------------------------


class TestRateLimiterValidation:
    def test_invalid_mode_raises(self):
        with pytest.raises(BlockParameterError, match="mode"):
            library.RateLimiter(
                dt=0.1, upper_limit=1.0, lower_limit=-1.0, mode="invalid"
            )

    def test_smooth_mode_requires_finite_upper(self):
        with pytest.raises(BlockParameterError):
            library.RateLimiter(
                dt=0.1,
                upper_limit=np.inf,
                lower_limit=-1.0,
                mode="smooth",
            )

    def test_smooth_mode_requires_finite_lower(self):
        with pytest.raises(BlockParameterError):
            library.RateLimiter(
                dt=0.1,
                upper_limit=1.0,
                lower_limit=-np.inf,
                mode="smooth",
            )

    def test_smooth_mode_requires_positive_sharpness(self):
        with pytest.raises(BlockParameterError):
            library.RateLimiter(
                dt=0.1,
                upper_limit=1.0,
                lower_limit=-1.0,
                mode="smooth",
                sharpness=0.0,
            )
