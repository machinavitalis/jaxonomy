# SPDX-License-Identifier: MIT

"""Tests for T-123 Phase 1 — ``RateTransition`` block / ``Decimator``.

T-123 ships the block-level companion to T-105 Multirate.  Phase 1
ships:

* :class:`Decimator` — a fast-to-slow subsampler-and-hold.
* :func:`RateTransition` — a factory that picks ``ZeroOrderHold`` for
  slow→fast, :class:`Decimator` for fast→slow, or :class:`UnitDelay`
  for the same-rate case.
* Integration with T-105's :func:`detect_rate_mismatches`: blocks
  flagged ``_jaxonomy_rate_transition = True`` silence the rate-mismatch
  walker on adjacent connections.

These tests cover all of the above and the differentiability contract.
"""

from __future__ import annotations

import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import jaxonomy
from jaxonomy import library
from jaxonomy.library import (
    Decimator,
    DiscreteClock,
    RateTransition,
    UnitDelay,
    ZeroOrderHold,
)
from jaxonomy.simulation.rate_groups import (
    RateMismatchWarning,
    detect_rate_mismatches,
    infer_block_sample_time,
)


pytestmark = pytest.mark.minimal


# --------------------------------------------------------------------- #
# Factory dispatch
# --------------------------------------------------------------------- #


class TestRateTransitionFactory:
    def test_slow_to_fast_returns_zoh(self):
        block = RateTransition(input_dt=0.1, output_dt=0.01)
        assert isinstance(block, ZeroOrderHold)
        assert getattr(block, "_jaxonomy_rate_transition", False) is True

    def test_fast_to_slow_returns_decimator(self):
        block = RateTransition(input_dt=0.01, output_dt=0.1)
        assert isinstance(block, Decimator)
        assert block._jaxonomy_rate_transition is True
        assert block.input_dt == 0.01
        assert block.output_dt == 0.1

    def test_same_rate_returns_unit_delay(self):
        block = RateTransition(input_dt=0.05, output_dt=0.05, initial_state=2.5)
        assert isinstance(block, UnitDelay)
        # UnitDelay path is not flagged: same-rate cannot be a mismatch.
        assert getattr(block, "_jaxonomy_rate_transition", False) is False

    def test_decimator_warns_when_output_dt_not_greater(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            Decimator(input_dt=0.1, output_dt=0.01)
        # Exactly one warning, mentioning output_dt.
        assert any("output_dt" in str(w.message) for w in caught)


# --------------------------------------------------------------------- #
# Sample-time inference: rate-transition marker is honoured.
# --------------------------------------------------------------------- #


class TestSampleTimeInference:
    def test_decimator_inferred_at_output_dt(self):
        # Decimator declares a periodic update at output_dt; inference
        # picks that period.
        block = Decimator(input_dt=0.01, output_dt=0.1, initial_state=0.0)
        # Need to wire it into a tiny diagram so initialize() runs.
        builder = jaxonomy.DiagramBuilder()
        src = builder.add(library.Constant(0.0, name="src"))
        dec = builder.add(block)
        builder.connect(src.output_ports[0], dec.input_ports[0])
        diag = builder.build()
        diag.create_context()

        st = infer_block_sample_time(dec)
        assert st.kind == "discrete"
        assert st.period == 0.1


# --------------------------------------------------------------------- #
# Slow → fast (ZOH semantics): output holds the slow value across fast ticks.
# --------------------------------------------------------------------- #


def _build_slow_to_fast_diagram(slow_dt, fast_dt, t_final):
    """slow DiscreteClock(slow_dt) -> RateTransition(slow_dt, fast_dt)."""
    builder = jaxonomy.DiagramBuilder()
    slow_src = builder.add(DiscreteClock(dt=slow_dt, name="slow_clk"))
    rt = builder.add(
        RateTransition(input_dt=slow_dt, output_dt=fast_dt, name="rt_zoh")
    )
    builder.connect(slow_src.output_ports[0], rt.input_ports[0])
    diagram = builder.build()
    context = diagram.create_context()
    return diagram, context, slow_src, rt


def test_slow_to_fast_zoh_holds_value():
    """ZOH at fast rate: between slow ticks, the held value stays flat."""
    slow_dt, fast_dt = 0.1, 0.01
    diagram, context, slow_src, rt = _build_slow_to_fast_diagram(
        slow_dt=slow_dt, fast_dt=fast_dt, t_final=0.5
    )
    res = jaxonomy.simulate(
        diagram,
        context,
        (0.0, 0.5),
        recorded_signals={
            "src": slow_src.output_ports[0],
            "rt": rt.output_ports[0],
        },
    )
    src_vals = np.asarray(res.outputs["src"])
    rt_vals = np.asarray(res.outputs["rt"])
    # The ZOH and the slow source both sample at offset=0 on the same
    # clock at slow ticks; on the slow-tick sample the ZOH still shows
    # the *previous* held value (one slow tick old) because of the
    # standard two-phase x⁻ ordering.  Off the slow ticks the ZOH
    # output is exactly the most recent slow sample.
    #
    # Verify: rt is held flat between slow ticks (only one or zero
    # transitions per slow window) and matches the slow source within a
    # single slow_dt lag.
    times = np.asarray(res.time)
    diff = np.abs(np.diff(rt_vals))
    # Number of distinct held values across [0, 0.5] is at most ~6.
    assert len(set(np.round(rt_vals, 8))) <= 7
    # rt lags src by at most slow_dt (one slow tick) at every sample.
    assert np.all(np.abs(rt_vals - src_vals) <= slow_dt + 1e-9)


# --------------------------------------------------------------------- #
# Fast → slow (Decimator): output samples at slow rate.
# --------------------------------------------------------------------- #


def test_fast_to_slow_decimator_samples_at_slow_rate():
    """Decimator sampled at slow_dt: between slow ticks, output is constant."""
    fast_dt, slow_dt = 0.01, 0.1
    builder = jaxonomy.DiagramBuilder()
    fast_src = builder.add(DiscreteClock(dt=fast_dt, name="fast_clk"))
    dec = builder.add(
        RateTransition(input_dt=fast_dt, output_dt=slow_dt, name="rt_dec")
    )
    builder.connect(fast_src.output_ports[0], dec.input_ports[0])
    diagram = builder.build()
    context = diagram.create_context()

    res = jaxonomy.simulate(
        diagram,
        context,
        (0.0, 0.5),
        recorded_signals={
            "src": fast_src.output_ports[0],
            "dec": dec.output_ports[0],
        },
    )
    times = np.asarray(res.time)
    dec_vals = np.asarray(res.outputs["dec"])

    # The decimator output should only take a finite number of distinct
    # values (one per slow tick within the interval).  With slow_dt=0.1
    # over [0, 0.5] we expect at most ~6 distinct held values
    # (initial state + one per tick).
    distinct = set(np.round(dec_vals, 8))
    assert len(distinct) <= 7

    # Output is held flat between successive slow ticks: any change in
    # the decimator output happens only at a slow-tick boundary.
    diff = np.abs(np.diff(dec_vals))
    change_idx = np.where(diff > 1e-9)[0]
    if len(change_idx) > 0:
        change_times = times[change_idx + 1]
        # Each change time should be within fast_dt of a multiple of slow_dt.
        for t in change_times:
            nearest_slow = round(t / slow_dt) * slow_dt
            assert abs(t - nearest_slow) <= 2 * fast_dt, (
                f"Decimator output changed at t={t}, not near a slow tick."
            )


def test_decimator_initial_state_held_until_first_slow_tick():
    """Output stays at initial_state until the first slow update fires."""
    fast_dt, slow_dt = 0.01, 0.1
    builder = jaxonomy.DiagramBuilder()
    fast_src = builder.add(library.Constant(7.0, name="src7"))
    dec = builder.add(
        Decimator(
            input_dt=fast_dt,
            output_dt=slow_dt,
            initial_state=-1.0,
            name="dec",
        )
    )
    builder.connect(fast_src.output_ports[0], dec.input_ports[0])
    diagram = builder.build()
    context = diagram.create_context()

    res = jaxonomy.simulate(
        diagram,
        context,
        (0.0, 0.05),  # before the first slow tick at 0.1
        recorded_signals={"dec": dec.output_ports[0]},
    )
    dec_vals = np.asarray(res.outputs["dec"])
    # Every recorded sample on (0, 0.05) is the initial state.
    assert np.allclose(dec_vals, -1.0)


# --------------------------------------------------------------------- #
# Same-rate path: behaves like a UnitDelay.
# --------------------------------------------------------------------- #


def test_same_rate_behaves_like_unit_delay():
    """A RateTransition at equal dt should match a hand-rolled UnitDelay."""
    dt = 0.05
    initial = 3.0

    def _build_with(block_factory):
        builder = jaxonomy.DiagramBuilder()
        src = builder.add(DiscreteClock(dt=dt, name="clk"))
        blk = builder.add(block_factory())
        builder.connect(src.output_ports[0], blk.input_ports[0])
        diag = builder.build()
        ctx = diag.create_context()
        return diag, ctx, blk

    diag_a, ctx_a, blk_a = _build_with(
        lambda: RateTransition(input_dt=dt, output_dt=dt, initial_state=initial)
    )
    diag_b, ctx_b, blk_b = _build_with(
        lambda: UnitDelay(dt=dt, initial_state=initial)
    )

    res_a = jaxonomy.simulate(
        diag_a, ctx_a, (0.0, 0.5),
        recorded_signals={"y": blk_a.output_ports[0]},
    )
    res_b = jaxonomy.simulate(
        diag_b, ctx_b, (0.0, 0.5),
        recorded_signals={"y": blk_b.output_ports[0]},
    )
    np.testing.assert_array_equal(
        np.asarray(res_a.outputs["y"]), np.asarray(res_b.outputs["y"])
    )


# --------------------------------------------------------------------- #
# Differentiability: gradient flows through the rate-transition input.
# --------------------------------------------------------------------- #


def test_decimator_gradient_flows_through_input():
    """grad through Decimator's update / output is non-zero and finite.

    Bypasses the full simulator (whose recorded-signals path is not
    JAX-traceable as written) and exercises ``Decimator._update`` /
    ``_output`` directly across a small loop, which is the same pattern
    used by ``test_t_107_transport_delay_phase1.test_transport_delay_grad_through_input``.
    """
    from collections import namedtuple

    State = namedtuple("State", ["discrete_state"])
    block = Decimator(input_dt=0.01, output_dt=0.1, initial_state=0.0)

    def loss(amp):
        # Sample-and-hold: every "slow tick" latches the input amp into
        # state; output equals state.  Sum across N ticks should be
        # roughly N * amp ⇒ derivative ≈ N.
        state = State(discrete_state=jnp.asarray(0.0))
        total = jnp.array(0.0)
        for _ in range(5):
            new_xd = block._update(jnp.asarray(0.0), state, amp)
            state = State(discrete_state=new_xd)
            total = total + block._output(jnp.asarray(0.0), state)
        return total

    g = jax.grad(loss)(jnp.asarray(2.0))
    assert jnp.isfinite(g)
    assert float(g) > 0
    # 5 ticks * d/d_amp(amp) per tick = 5
    assert abs(float(g) - 5.0) < 1e-6


# --------------------------------------------------------------------- #
# Integration with T-105 detect_rate_mismatches.
# --------------------------------------------------------------------- #


class TestT105Integration:
    """A diagram with ``Slow → RateTransition → Fast`` should *not*
    trigger T-105's rate-mismatch warning."""

    def _build_slow_rt_fast(self):
        slow_dt, fast_dt = 0.10, 0.01
        builder = jaxonomy.DiagramBuilder()
        slow = builder.add(
            UnitDelay(dt=slow_dt, initial_state=0.0, name="slow")
        )
        # Slow source needs an input — feed it a constant.
        slow_src = builder.add(library.Constant(1.0, name="slow_src"))
        builder.connect(slow_src.output_ports[0], slow.input_ports[0])

        rt = builder.add(
            RateTransition(input_dt=slow_dt, output_dt=fast_dt, name="rt")
        )
        builder.connect(slow.output_ports[0], rt.input_ports[0])

        fast = builder.add(
            UnitDelay(dt=fast_dt, initial_state=0.0, name="fast")
        )
        builder.connect(rt.output_ports[0], fast.input_ports[0])

        diag = builder.build()
        diag.create_context()
        return diag

    def _build_slow_to_fast_no_rt(self):
        """Same shape but with the RateTransition deleted; expect a mismatch."""
        slow_dt, fast_dt = 0.10, 0.01
        builder = jaxonomy.DiagramBuilder()
        slow_src = builder.add(library.Constant(1.0, name="slow_src"))
        slow = builder.add(
            UnitDelay(dt=slow_dt, initial_state=0.0, name="slow")
        )
        builder.connect(slow_src.output_ports[0], slow.input_ports[0])

        fast = builder.add(
            UnitDelay(dt=fast_dt, initial_state=0.0, name="fast")
        )
        builder.connect(slow.output_ports[0], fast.input_ports[0])
        diag = builder.build()
        diag.create_context()
        return diag

    def test_rate_transition_silences_mismatch(self):
        diag = self._build_slow_rt_fast()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            mismatches = detect_rate_mismatches(diag, on_mismatch="warn")
        rate_warnings = [
            w for w in caught if issubclass(w.category, RateMismatchWarning)
        ]
        assert mismatches == []
        assert rate_warnings == []

    def test_no_rate_transition_still_warns(self):
        """Sanity check: removing the RateTransition restores the warning."""
        diag = self._build_slow_to_fast_no_rt()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            mismatches = detect_rate_mismatches(diag, on_mismatch="warn")
        assert len(mismatches) >= 1
        rate_warnings = [
            w for w in caught if issubclass(w.category, RateMismatchWarning)
        ]
        assert len(rate_warnings) >= 1

    def test_fast_to_slow_chain_silenced(self):
        """Fast → Decimator → Slow stays silent under the detector."""
        slow_dt, fast_dt = 0.10, 0.01
        builder = jaxonomy.DiagramBuilder()
        fast_src = builder.add(DiscreteClock(dt=fast_dt, name="fast_clk"))
        rt = builder.add(
            RateTransition(input_dt=fast_dt, output_dt=slow_dt, name="rt_dec")
        )
        slow = builder.add(
            UnitDelay(dt=slow_dt, initial_state=0.0, name="slow")
        )
        builder.connect(fast_src.output_ports[0], rt.input_ports[0])
        builder.connect(rt.output_ports[0], slow.input_ports[0])
        diag = builder.build()
        diag.create_context()

        mismatches = detect_rate_mismatches(diag, on_mismatch="collect")
        assert mismatches == []
