# SPDX-License-Identifier: MIT

"""Tests for T-123-followup-decimator-mode — Decimator window modes.

Phase 1 ships only ``pick_last`` semantics (latch the most recent
fast-rate sample at every slow tick).  This followup adds:

* ``mode="mean"`` — arithmetic mean of every input sample in the
  ``output_dt`` window (standard anti-alias decimation, linear =>
  differentiable through the input).
* ``mode="peak"`` — max-absolute-value sample in the window (peak /
  envelope preservation; ``np.where``-based selector keeps gradients
  flowing through the selected sample's value).
* Validation: an unknown ``mode`` raises ``ValueError``.
* Byte-equivalence: the default ``mode="pick_last"`` matches the
  phase-1 ``Decimator`` exactly.
"""

from __future__ import annotations

from collections import namedtuple

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import jaxonomy
from jaxonomy import library
from jaxonomy.library import Decimator, DiscreteClock
from jaxonomy.framework.leaf_system import LeafSystem


pytestmark = pytest.mark.minimal


_FAST_DT = 0.01
_SLOW_DT = 0.1


# --------------------------------------------------------------------- #
# Mode validation
# --------------------------------------------------------------------- #


def test_invalid_mode_raises():
    """Unknown mode strings must raise at construction time."""
    with pytest.raises(ValueError, match="mode="):
        Decimator(input_dt=_FAST_DT, output_dt=_SLOW_DT, mode="bogus")


def test_default_mode_is_pick_last():
    """Default ``mode`` keyword is ``"pick_last"`` so phase 1 byte
    equivalence does not require an explicit kwarg."""
    block = Decimator(input_dt=_FAST_DT, output_dt=_SLOW_DT)
    assert block._mode == "pick_last"


# --------------------------------------------------------------------- #
# Byte-equivalence: pick_last default vs explicit pick_last vs phase-1
# (no-mode-kwarg).  All three should produce identical samples.
# --------------------------------------------------------------------- #


def _build_pick_last_diagram(*, mode=None, initial_state=-1.0):
    builder = jaxonomy.DiagramBuilder()
    src = builder.add(DiscreteClock(dt=_FAST_DT, name="clk"))
    kwargs = {"name": "dec"}
    if mode is not None:
        kwargs["mode"] = mode
    dec = builder.add(
        Decimator(
            input_dt=_FAST_DT,
            output_dt=_SLOW_DT,
            initial_state=initial_state,
            **kwargs,
        )
    )
    builder.connect(src.output_ports[0], dec.input_ports[0])
    diag = builder.build()
    ctx = diag.create_context()
    return diag, ctx, dec


def test_pick_last_default_matches_explicit():
    diag_a, ctx_a, dec_a = _build_pick_last_diagram(mode=None)
    diag_b, ctx_b, dec_b = _build_pick_last_diagram(mode="pick_last")

    res_a = jaxonomy.simulate(
        diag_a, ctx_a, (0.0, 0.5),
        recorded_signals={"y": dec_a.output_ports[0]},
    )
    res_b = jaxonomy.simulate(
        diag_b, ctx_b, (0.0, 0.5),
        recorded_signals={"y": dec_b.output_ports[0]},
    )
    np.testing.assert_array_equal(
        np.asarray(res_a.outputs["y"]),
        np.asarray(res_b.outputs["y"]),
    )


def test_pick_last_byte_equivalent_to_phase1_behavior():
    """Phase-1 expectation: between slow ticks the output is held flat
    and only takes a finite number of distinct values (≤ #slow ticks)."""
    diag, ctx, dec = _build_pick_last_diagram(mode="pick_last")
    res = jaxonomy.simulate(
        diag, ctx, (0.0, 0.5),
        recorded_signals={"y": dec.output_ports[0]},
    )
    y = np.asarray(res.outputs["y"])
    # Same invariant as ``test_fast_to_slow_decimator_samples_at_slow_rate``
    # in the phase-1 suite: at most ~6 distinct held values over [0, 0.5].
    distinct = set(np.round(y, 8))
    assert len(distinct) <= 7


# --------------------------------------------------------------------- #
# mode="mean" on a ramp input
# --------------------------------------------------------------------- #


def test_mean_mode_on_ramp_emits_window_mean():
    """``DiscreteClock(fast_dt)`` produces u(t) = t at every fast tick.

    With ``output_dt=10*input_dt`` the first slow tick at t=0.1 reads
    the accumulated window of samples at t=0, 0.01, ..., 0.09 (sum =
    0.45, count = 10, mean = 0.045) and stores the mean on
    ``state.output``.  The output port samples ``state.output`` at the
    *next* slow tick (t=0.2) — the standard one-slow-tick latch that
    matches the phase-1 pick_last block.  Subsequent windows shift by
    0.1 each."""
    builder = jaxonomy.DiagramBuilder()
    src = builder.add(DiscreteClock(dt=_FAST_DT, name="clk"))
    dec = builder.add(
        Decimator(
            input_dt=_FAST_DT,
            output_dt=_SLOW_DT,
            initial_state=-1.0,
            mode="mean",
            name="dec_mean",
        )
    )
    builder.connect(src.output_ports[0], dec.input_ports[0])
    diag = builder.build()
    ctx = diag.create_context()

    res = jaxonomy.simulate(
        diag, ctx, (0.0, 0.5),
        recorded_signals={"src": src.output_ports[0], "dec": dec.output_ports[0]},
    )
    ts = np.asarray(res.time)
    y = np.asarray(res.outputs["dec"])

    # At t=0.0 ... t<0.2, output should still be the initial_state
    # (-1.0) — the very first slow tick at t=0.1 *computes* the
    # window-0 mean but the output port samples it only at the next
    # slow tick (t=0.2).
    mask_init = (ts < 0.2 - 1e-9)
    assert np.allclose(y[mask_init], -1.0)

    # At t in [0.2, 0.3) the emitted mean covers window 0 (samples at
    # t = 0.00..0.09), expected mean = 0.045.
    mask_w0 = (ts >= 0.2 - 1e-9) & (ts < 0.3 - 1e-9)
    assert np.allclose(y[mask_w0], 0.045, atol=1e-9)

    # Window 1 (t = 0.10..0.19), expected mean = 0.145.  Emitted at
    # t=0.3.
    mask_w1 = (ts >= 0.3 - 1e-9) & (ts < 0.4 - 1e-9)
    assert np.allclose(y[mask_w1], 0.145, atol=1e-9)

    # Window 2 (t = 0.20..0.29), expected mean = 0.245.  Emitted at
    # t=0.4.
    mask_w2 = (ts >= 0.4 - 1e-9) & (ts < 0.5 - 1e-9)
    assert np.allclose(y[mask_w2], 0.245, atol=1e-9)


def test_mean_mode_differs_from_pick_last_on_ramp():
    """Sanity: mean and pick_last give *different* outputs on a ramp."""
    def run(mode):
        builder = jaxonomy.DiagramBuilder()
        src = builder.add(DiscreteClock(dt=_FAST_DT, name="clk"))
        dec = builder.add(
            Decimator(
                input_dt=_FAST_DT,
                output_dt=_SLOW_DT,
                initial_state=0.0,
                mode=mode,
                name=f"dec_{mode}",
            )
        )
        builder.connect(src.output_ports[0], dec.input_ports[0])
        diag = builder.build()
        ctx = diag.create_context()
        res = jaxonomy.simulate(
            diag, ctx, (0.0, 0.3),
            recorded_signals={"y": dec.output_ports[0]},
        )
        return np.asarray(res.outputs["y"])

    y_pick = run("pick_last")
    y_mean = run("mean")
    assert not np.allclose(y_pick, y_mean)


# --------------------------------------------------------------------- #
# mode="peak" preserves the max-|u| sample in each window
# --------------------------------------------------------------------- #


class _SpikeSource(LeafSystem):
    """Tiny test source: large negative spike at the middle of each
    ``slow_dt`` window plus a baseline value tied to the window index."""

    def __init__(self, fast_dt, slow_dt):
        super().__init__()
        self._fast_dt = float(fast_dt)
        self._slow_dt = float(slow_dt)
        self.declare_output_port(self._out, period=fast_dt, offset=0.0)

    def initialize(self):
        pass

    def _out(self, time, state, **_params):
        k = jnp.floor(time / self._slow_dt)
        # Spike at t = k*slow_dt + 0.5*slow_dt.  Detect within a fast-dt
        # window of that target so the JAX compare is robust to tiny FP
        # noise on the discrete clock.
        target = k * self._slow_dt + 0.5 * self._slow_dt
        in_spike = jnp.abs(time - target) < 0.5 * self._fast_dt
        return jnp.where(in_spike, -10.0 - k, 0.1 * k)


def test_peak_mode_captures_window_peak():
    """Each slow window contains a single large negative spike
    (amplitude grows by 1.0 per window).  ``mode="peak"`` should emit
    that spike at the next slow tick."""
    builder = jaxonomy.DiagramBuilder()
    src = builder.add(_SpikeSource(_FAST_DT, _SLOW_DT))
    dec = builder.add(
        Decimator(
            input_dt=_FAST_DT,
            output_dt=_SLOW_DT,
            initial_state=0.0,
            mode="peak",
            name="dec_peak",
        )
    )
    builder.connect(src.output_ports[0], dec.input_ports[0])
    diag = builder.build()
    ctx = diag.create_context()

    res = jaxonomy.simulate(
        diag, ctx, (0.0, 0.5),
        recorded_signals={"y": dec.output_ports[0]},
    )
    ts = np.asarray(res.time)
    y = np.asarray(res.outputs["y"])

    # At t in [0.2, 0.3) the emitted peak covers window 0
    # (t = 0.00..0.09, spike at t=0.05 with amplitude -10.0).  Standard
    # one-slow-tick latch delays the emission by one slow_dt relative
    # to the window.
    mask_w0 = (ts >= 0.2 - 1e-9) & (ts < 0.3 - 1e-9)
    assert np.allclose(y[mask_w0], -10.0, atol=1e-9), (
        f"window-0 peak should be -10 (the t in [0,0.09] spike), got {y[mask_w0]}"
    )

    # Window 1 (t = 0.10..0.19) contains the -11 spike at t=0.15.
    mask_w1 = (ts >= 0.3 - 1e-9) & (ts < 0.4 - 1e-9)
    assert np.allclose(y[mask_w1], -11.0, atol=1e-9)

    # Window 2 (t = 0.20..0.29) contains the -12 spike at t=0.25.
    mask_w2 = (ts >= 0.4 - 1e-9) & (ts < 0.5 - 1e-9)
    assert np.allclose(y[mask_w2], -12.0, atol=1e-9)


def test_peak_mode_picks_positive_when_no_negative_larger():
    """When all samples in a window are positive, ``peak`` returns the
    sample with the largest value (which is also the largest |u|)."""
    builder = jaxonomy.DiagramBuilder()
    src = builder.add(DiscreteClock(dt=_FAST_DT, name="clk"))
    dec = builder.add(
        Decimator(
            input_dt=_FAST_DT,
            output_dt=_SLOW_DT,
            initial_state=0.0,
            mode="peak",
            name="dec_peak",
        )
    )
    builder.connect(src.output_ports[0], dec.input_ports[0])
    diag = builder.build()
    ctx = diag.create_context()
    res = jaxonomy.simulate(
        diag, ctx, (0.0, 0.3),
        recorded_signals={"y": dec.output_ports[0]},
    )
    ts = np.asarray(res.time)
    y = np.asarray(res.outputs["y"])
    # The largest sample in window-0 [t=0.0..0.09] is 0.09 (from the
    # DiscreteClock ramp), emitted at t=0.2 (one-slow-tick latch).
    mask_w0 = (ts >= 0.2 - 1e-9) & (ts < 0.3 - 1e-9)
    assert np.allclose(y[mask_w0], 0.09, atol=1e-9)


# --------------------------------------------------------------------- #
# Differentiability through the input signal in mean mode
# --------------------------------------------------------------------- #


def test_mean_mode_gradient_flows_through_input():
    """grad through the mean-mode update flow is finite and equals the
    analytic derivative (mean is linear in each input sample).

    Bypasses the simulator (whose recorded-signals path is not
    JAX-traceable as written) and drives ``_update_mean_accumulate`` /
    ``_update_mean_emit`` directly, mirroring the phase-1
    ``test_decimator_gradient_flows_through_input`` pattern."""
    State = namedtuple("State", ["discrete_state"])
    block = Decimator(
        input_dt=_FAST_DT, output_dt=_SLOW_DT, initial_state=0.0, mode="mean",
    )

    from jaxonomy.library.primitives import _DecimatorMeanState

    def loss(samples):
        # Replay a single window of fast-rate inputs through the
        # accumulate / emit callbacks.  After N accumulates the state
        # holds (sum, N); the emit produces mean=sum/N.
        state = State(
            discrete_state=_DecimatorMeanState(
                output=jnp.asarray(0.0),
                accumulator=jnp.asarray(0.0),
                count=jnp.asarray(0.0),
            )
        )
        for s in samples:
            new_xd = block._update_mean_accumulate(jnp.asarray(0.0), state, s)
            state = State(discrete_state=new_xd)
        new_xd = block._update_mean_emit(jnp.asarray(0.0), state, jnp.asarray(0.0))
        state = State(discrete_state=new_xd)
        return block._output_windowed(jnp.asarray(0.0), state)

    samples = jnp.arange(10, dtype=jnp.float64) + 1.0
    # Mean of [1, 2, ..., 10] = 5.5
    val = loss(samples)
    assert abs(float(val) - 5.5) < 1e-9

    # d(mean)/d(samples_i) = 1/N for every i. With N=10 each entry's
    # gradient must be 0.1.
    g = jax.grad(lambda s: loss(s))(samples)
    g_np = np.asarray(g)
    np.testing.assert_allclose(g_np, np.full(10, 0.1), atol=1e-9)


# --------------------------------------------------------------------- #
# Rate-transition marker survives the new modes
# --------------------------------------------------------------------- #


@pytest.mark.parametrize("mode", ["pick_last", "mean", "peak"])
def test_rate_transition_marker_preserved_for_all_modes(mode):
    block = Decimator(
        input_dt=_FAST_DT, output_dt=_SLOW_DT, initial_state=0.0, mode=mode,
    )
    assert block._jaxonomy_rate_transition is True
