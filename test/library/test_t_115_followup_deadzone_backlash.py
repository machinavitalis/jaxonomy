# SPDX-License-Identifier: MIT

"""T-115-followup-deadzone-backlash tests.

Verifies:

* :class:`~jaxonomy.library.DeadZone` ``mode={"hard","smooth"}`` kwarg.
  Default ``mode="hard"`` is byte-equivalent to the legacy phase 1
  behavior (including zero-crossing event declaration). The new
  ``mode="smooth"`` path dispatches to :func:`soft_dead_zone` and must
  *not* declare zero-crossing events. Smooth output approaches the hard
  output as sharpness grows.
* :class:`~jaxonomy.library.Backlash` discrete-state hysteresis block.
  Output lags input by ``width/2`` within the hysteresis band and snaps
  to the active band edge outside it. Output is differentiable w.r.t.
  ``width``.
* Validation: invalid kwargs (``mode``, ``width``, ``dt``, ``sharpness``)
  raise :class:`BlockParameterError`.
"""

import pytest

import numpy as np
import jax
import jax.numpy as jnp

import jaxonomy
from jaxonomy import library
from jaxonomy.framework.error import BlockParameterError
from jaxonomy.library.primitives import soft_dead_zone, Backlash

pytestmark = pytest.mark.minimal


# ---------------------------------------------------------------------------
# soft_dead_zone functional helper
# ---------------------------------------------------------------------------


class TestSoftDeadZoneFunction:
    def test_zero_at_origin(self):
        y = soft_dead_zone(jnp.array(0.0), 1.0, sharpness=10.0)
        assert jnp.isclose(y, 0.0, atol=1e-12)

    def test_far_outside_band_near_identity(self):
        # Far above the band -> y ~ u
        y = soft_dead_zone(jnp.array(5.0), 1.0, sharpness=20.0)
        assert jnp.isclose(y, 5.0, atol=1e-3)
        # Far below the band -> y ~ u (negative)
        y_neg = soft_dead_zone(jnp.array(-5.0), 1.0, sharpness=20.0)
        assert jnp.isclose(y_neg, -5.0, atol=1e-3)

    def test_deep_inside_band_near_zero(self):
        # Deep inside the band -> y ~ 0
        y = soft_dead_zone(jnp.array(0.1), 1.0, sharpness=20.0)
        assert jnp.abs(y) < 1e-3

    def test_high_sharpness_matches_hard(self):
        """As sharpness -> inf, smooth recovers the hard dead zone."""
        for u in [-2.0, -0.5, 0.0, 0.5, 2.0]:
            y_smooth = float(
                soft_dead_zone(jnp.array(u), 1.0, sharpness=500.0)
            )
            y_hard = 0.0 if abs(u) < 1.0 else u
            assert abs(y_smooth - y_hard) < 1e-2, (
                f"u={u}: smooth={y_smooth} hard={y_hard}"
            )

    def test_gradient_nonzero_inside_band(self):
        """The whole point: gradient flows through the dead-zone band."""

        def f(x):
            return jnp.sum(soft_dead_zone(x, 1.0, sharpness=10.0))

        # Hard dead zone would give gradient == 0 inside |u|<1.
        g_inside = jax.grad(f)(jnp.array(0.3))
        g_outside = jax.grad(f)(jnp.array(2.0))
        assert jnp.abs(g_outside) > 0.0
        # Smooth gradient inside the band is small but nonzero.
        assert jnp.abs(g_inside) > 0.0


# ---------------------------------------------------------------------------
# DeadZone: mode="hard" byte-equivalence
# ---------------------------------------------------------------------------


def _eval_deadzone(value, *, half_range=1.0, **kwargs):
    """Run a DeadZone block on a single Constant input and return the output."""
    builder = jaxonomy.DiagramBuilder()
    src = builder.add(library.Constant(value=float(value)))
    dz = builder.add(library.DeadZone(half_range=half_range, **kwargs))
    builder.connect(src.output_ports[0], dz.input_ports[0])
    diagram = builder.build()
    context = diagram.create_context()
    return float(dz.output_ports[0].eval(context)), dz


class TestDeadZoneModeHardDefault:
    @pytest.mark.parametrize("u", [-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0])
    def test_default_matches_legacy(self, u):
        y, _ = _eval_deadzone(u, half_range=1.0)
        # Legacy formula: where(|u| < hr, 0, u).
        y_ref = 0.0 if abs(u) < 1.0 else u
        assert y == y_ref

    @pytest.mark.parametrize("u", [-2.0, 0.5, 2.0])
    def test_explicit_hard_matches_default(self, u):
        y_default, _ = _eval_deadzone(u, half_range=1.0)
        y_hard, _ = _eval_deadzone(u, half_range=1.0, mode="hard")
        # Byte-equivalent.
        assert y_default == y_hard


# ---------------------------------------------------------------------------
# DeadZone: mode="smooth" approaches hard with high sharpness
# ---------------------------------------------------------------------------


class TestDeadZoneModeSmooth:
    @pytest.mark.parametrize("u", [-3.0, -1.5, 0.0, 1.5, 3.0])
    def test_smooth_matches_soft_dead_zone_function(self, u):
        y_block, _ = _eval_deadzone(
            u, half_range=1.0, mode="smooth", sharpness=20.0
        )
        y_func = float(soft_dead_zone(jnp.array(u), 1.0, sharpness=20.0))
        assert abs(y_block - y_func) < 1e-12

    def test_smooth_high_sharpness_close_to_hard(self):
        """Smooth mode with high sharpness ~ hard mode outside the band."""
        for u in [-2.0, 2.0]:
            y_smooth, _ = _eval_deadzone(
                u, half_range=1.0, mode="smooth", sharpness=200.0
            )
            y_hard, _ = _eval_deadzone(u, half_range=1.0, mode="hard")
            assert abs(y_smooth - y_hard) < 1e-2

    def test_smooth_default_sharpness_is_ten(self):
        y_explicit, _ = _eval_deadzone(
            0.5, half_range=1.0, mode="smooth", sharpness=10.0
        )
        y_default, _ = _eval_deadzone(0.5, half_range=1.0, mode="smooth")
        assert abs(y_default - y_explicit) < 1e-12


# ---------------------------------------------------------------------------
# DeadZone: zero-crossing introspection
# ---------------------------------------------------------------------------


class TestDeadZoneZeroCrossings:
    def _build_with_integrator(self, *, mode="hard", **dz_kwargs):
        builder = jaxonomy.DiagramBuilder()
        ramp = builder.add(library.Ramp(start_value=-2.0, start_time=0.0))
        dz = builder.add(
            library.DeadZone(half_range=0.5, mode=mode, **dz_kwargs)
        )
        integ = builder.add(library.Integrator(0.0))
        builder.connect(ramp.output_ports[0], dz.input_ports[0])
        builder.connect(dz.output_ports[0], integ.input_ports[0])
        diagram = builder.build()
        diagram.create_context()
        return dz

    def test_hard_mode_declares_zero_crossings(self):
        dz = self._build_with_integrator(mode="hard")
        assert dz.has_zero_crossing_events
        # The hard variant declares two events: lower- and upper-limit.
        assert len(dz.zero_crossing_events.events) == 2

    def test_smooth_mode_declares_no_zero_crossings(self):
        dz = self._build_with_integrator(mode="smooth", sharpness=10.0)
        assert not dz.has_zero_crossing_events
        assert len(dz.zero_crossing_events.events) == 0


# ---------------------------------------------------------------------------
# DeadZone: validation
# ---------------------------------------------------------------------------


class TestDeadZoneValidation:
    def test_invalid_mode_raises(self):
        with pytest.raises(BlockParameterError, match="mode"):
            library.DeadZone(half_range=1.0, mode="invalid")

    def test_smooth_mode_requires_positive_sharpness(self):
        with pytest.raises(BlockParameterError, match="sharpness"):
            library.DeadZone(
                half_range=1.0, mode="smooth", sharpness=-1.0
            )

    def test_zero_sharpness_in_smooth_raises(self):
        with pytest.raises(BlockParameterError, match="sharpness"):
            library.DeadZone(
                half_range=1.0, mode="smooth", sharpness=0.0
            )

    def test_negative_half_range_still_raises(self):
        # Existing T-001b behavior: half_range must be > 0 regardless of mode.
        with pytest.raises(BlockParameterError, match="half_range"):
            library.DeadZone(half_range=-1.0, mode="hard")
        with pytest.raises(BlockParameterError, match="half_range"):
            library.DeadZone(half_range=-1.0, mode="smooth")


# ---------------------------------------------------------------------------
# Backlash: pure kernel correctness
# ---------------------------------------------------------------------------


class TestBacklashKernel:
    def test_inside_band_holds(self):
        # |u - last| < width/2 -> output sticks at ``last``.
        y = Backlash._apply(
            last_output=jnp.array(0.0),
            u=jnp.array(0.1),
            width=jnp.array(0.5),
        )
        assert jnp.isclose(y, 0.0, atol=1e-12)

    def test_above_band_follows_upper_edge(self):
        # u > last + width/2 -> output = u - width/2.
        y = Backlash._apply(
            last_output=jnp.array(0.0),
            u=jnp.array(1.0),
            width=jnp.array(0.5),
        )
        # delta = 1.0, half = 0.25, so y = u - half = 0.75.
        assert jnp.isclose(y, 0.75, atol=1e-12)

    def test_below_band_follows_lower_edge(self):
        # u < last - width/2 -> output = u + width/2.
        y = Backlash._apply(
            last_output=jnp.array(0.0),
            u=jnp.array(-1.0),
            width=jnp.array(0.5),
        )
        # delta = -1.0, half = 0.25, so y = u + half = -0.75.
        assert jnp.isclose(y, -0.75, atol=1e-12)

    def test_zero_width_recovers_identity(self):
        # width=0: any nonzero delta crosses the band; output snaps to u.
        for last, u in [(0.0, 1.0), (1.0, 0.5), (-0.3, 0.7)]:
            y = Backlash._apply(
                last_output=jnp.array(last),
                u=jnp.array(u),
                width=jnp.array(0.0),
            )
            assert jnp.isclose(y, u, atol=1e-12)


# ---------------------------------------------------------------------------
# Backlash: end-to-end simulation with sinusoidal input
# ---------------------------------------------------------------------------


def _simulate_backlash(width=0.5, dt=0.01, t_end=2.0, freq=1.0):
    """Drive Backlash with a sinusoidal input; return (t, u, y)."""
    builder = jaxonomy.DiagramBuilder()
    sine = builder.add(library.Sine(amplitude=1.0, frequency=freq))
    bl = builder.add(library.Backlash(width=width, dt=dt, initial_output=0.0))
    builder.connect(sine.output_ports[0], bl.input_ports[0])
    diagram = builder.build()
    context = diagram.create_context()
    recorded = {
        "u": sine.output_ports[0],
        "y": bl.output_ports[0],
    }
    r = jaxonomy.simulate(
        diagram, context, (0.0, t_end), recorded_signals=recorded
    )
    return (
        np.asarray(r.time),
        np.asarray(r.outputs["u"]),
        np.asarray(r.outputs["y"]),
    )


class TestBacklashSimulation:
    def test_output_lags_input_within_band(self):
        """For a sinusoidal input, |y - u| should stay <= width/2 + slack."""
        width = 0.5
        t, u, y = _simulate_backlash(width=width, dt=0.005, t_end=2.0)
        # Allow a one-step ``dt`` of slack on top of the half-band; the
        # block output updates discretely so |u-y| can momentarily exceed
        # width/2 by one step's worth of input motion.
        max_gap_allowed = width / 2.0 + 0.05  # generous slack
        assert np.max(np.abs(u - y)) <= max_gap_allowed, (
            f"max |u-y|={np.max(np.abs(u - y))} exceeds {max_gap_allowed}"
        )

    def test_output_inside_band_when_input_changes_little(self):
        """If input never moves by more than width/2 from the last
        output, the block output stays at its initial value."""
        # Drive a low-amplitude sine with width well above the peak-to-peak
        # excursion -> output should stay at 0.
        width = 5.0  # much larger than 2 * amplitude (=2)
        t, u, y = _simulate_backlash(width=width, dt=0.01, t_end=2.0)
        # The discrete output samples should all equal the initial value
        # (== 0.0). We use a small tolerance to allow for floating-point
        # noise but the block has no continuous drift.
        assert np.allclose(y, 0.0, atol=1e-12)

    def test_output_tracks_input_with_zero_width(self):
        """width=0 -> output tracks input with one-step latency."""
        width = 0.0
        dt = 0.01
        t, u, y = _simulate_backlash(width=width, dt=dt, t_end=1.0)
        # At each discrete tick the output catches up to the last input
        # sample. Sampling at the same grid, |u(t) - y(t)| is bounded by
        # the per-step input change == |sin'(t)| * dt <= 2*pi*freq*dt.
        max_step_change = 2 * np.pi * 1.0 * dt
        # Allow a moderate slack factor (continuous sine sampled at output
        # grid; first sample may include startup transient).
        assert np.max(np.abs(u[1:] - y[1:])) < max_step_change * 3.0


# ---------------------------------------------------------------------------
# Backlash: differentiability w.r.t. width
# ---------------------------------------------------------------------------


class TestBacklashDifferentiability:
    def test_grad_through_kernel_finite(self):
        """``jax.grad`` w.r.t. ``width`` is finite for the kernel."""

        def loss(width):
            # Apply N kernel steps to track a ramp input and return the
            # final output.
            last = jnp.array(0.0)
            for u in [0.1, 0.3, 0.5, 0.8, 1.0, 0.7, 0.4, 0.0]:
                last = Backlash._apply(last, jnp.array(u), width)
            return last

        # Width inside (0, 1) -> at least one step lies outside the band
        # (e.g. u jumps by 0.3) so the gradient is well-defined.
        g = jax.grad(loss)(jnp.array(0.4))
        assert jnp.isfinite(g)

    def test_grad_through_kernel_nonzero_when_outside_band(self):
        """When the input motion exceeds width/2, the output depends on
        ``width`` via ``u - width/2`` (or ``u + width/2``), so the
        gradient is nonzero."""

        def loss(width):
            return Backlash._apply(
                jnp.array(0.0), jnp.array(1.0), width
            )

        # delta = 1.0 > width/2 for width in (0, 2); output = 1 - width/2.
        # d/dwidth = -1/2.
        g = jax.grad(loss)(jnp.array(0.5))
        assert jnp.isclose(g, -0.5, atol=1e-12)


# ---------------------------------------------------------------------------
# Backlash: validation
# ---------------------------------------------------------------------------


class TestBacklashValidation:
    def test_negative_width_raises(self):
        with pytest.raises(BlockParameterError, match="width"):
            library.Backlash(width=-0.1, dt=0.01)

    def test_zero_dt_raises(self):
        with pytest.raises(BlockParameterError, match="dt"):
            library.Backlash(width=0.5, dt=0.0)

    def test_negative_dt_raises(self):
        with pytest.raises(BlockParameterError, match="dt"):
            library.Backlash(width=0.5, dt=-0.01)
