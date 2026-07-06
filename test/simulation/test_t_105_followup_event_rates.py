# SPDX-License-Identifier: MIT

"""Tests for T-105-followup-event-rates — rate-mismatch detection
across event-driven blocks (ZC-triggered, etc.).

The followup extends :func:`detect_rate_mismatches` so that
event-driven blocks (e.g. :class:`ZeroCrossingTriggeredSubsystem`)
constitute their own rate category ``SampleTime.event_driven``,
distinct from continuous and from periodic:

* ``event_driven`` ↔ ``discrete`` connections raise a warning that
  event timing may not align with the discrete grid.
* ``event_driven`` ↔ ``event_driven`` passes silently (both irregular).
* ``continuous`` ↔ ``event_driven`` passes silently (events sample
  continuous state on demand).

These tests cover both the explicit ``event_driven=True`` opt-in and
the implicit ``declare_zero_crossing`` detection, plus the
compatibility table inside :meth:`SampleTime.matches`.
"""

from __future__ import annotations

import warnings

import jax.numpy as jnp
import pytest

import jaxonomy
from jaxonomy.framework import (
    LeafSystem,
    ZeroCrossingTriggeredSubsystem,
)
from jaxonomy.simulation.rate_groups import (
    RateMismatch,
    RateMismatchWarning,
    SampleTime,
    detect_rate_mismatches,
    group_blocks_by_rate,
    infer_block_sample_time,
)


pytestmark = pytest.mark.minimal


# ---------------------------------------------------------------------
# Tiny pass-through used to wire diagrams.
# ---------------------------------------------------------------------


class _PassThrough(LeafSystem):
    def __init__(self, *, name=None, **kwargs):
        super().__init__(name=name, **kwargs)
        self.declare_input_port(name="u")
        self.declare_output_port(
            self._eval,
            name="y",
            requires_inputs=True,
            prerequisites_of_calc=[self.input_ports[0].ticket],
        )

    def _eval(self, _time, _state, *inputs, **_params):
        return inputs[0]


# ---------------------------------------------------------------------
# SampleTime.event_driven algebra
# ---------------------------------------------------------------------


class TestSampleTimeEventDriven:
    def test_factory(self):
        st = SampleTime.event_driven()
        assert st.kind == "event_driven"
        assert st.is_event_driven()
        assert not st.is_discrete()
        assert not st.is_universal()

    def test_event_driven_matches_event_driven(self):
        a = SampleTime.event_driven()
        b = SampleTime.event_driven()
        assert a.matches(b)

    def test_event_driven_mismatches_discrete(self):
        ev = SampleTime.event_driven()
        disc = SampleTime.discrete(period=0.01)
        assert not ev.matches(disc)
        assert not disc.matches(ev)

    def test_event_driven_matches_continuous(self):
        ev = SampleTime.event_driven()
        cont = SampleTime.continuous()
        assert ev.matches(cont)
        assert cont.matches(ev)

    def test_event_driven_matches_universal(self):
        ev = SampleTime.event_driven()
        const = SampleTime.constant()
        inh = SampleTime.inherited()
        assert ev.matches(const)
        assert const.matches(ev)
        assert ev.matches(inh)
        assert inh.matches(ev)


# ---------------------------------------------------------------------
# infer_block_sample_time picks up event-driven blocks
# ---------------------------------------------------------------------


class TestInferEventDriven:
    def test_zc_triggered_subsystem_is_event_driven(self):
        zc = ZeroCrossingTriggeredSubsystem(
            submodel=lambda u: u + 1.0,
            n_inputs=1,
            edge="rising",
            initial_value=0.0,
            name="zc",
        )
        st = infer_block_sample_time(zc)
        assert st.kind == "event_driven"

    def test_explicit_event_driven_attribute(self):
        # Honest-fallback path: a block author can opt-in by setting
        # ``event_driven=True`` on the instance even if no ZC events
        # have been declared yet.
        from jaxonomy.library import Constant

        block = Constant(1.0, name="K")
        # Constant alone classifies as ``constant``.
        assert infer_block_sample_time(block).kind == "constant"
        # Marking ``event_driven=True`` flips the classification.
        block.event_driven = True
        assert infer_block_sample_time(block).kind == "event_driven"

    def test_periodic_event_still_wins_over_zc(self):
        # A hybrid block that has both periodic events AND declares ZC
        # events should still bucket as ``discrete`` — the periodic rate
        # is the dominant scheduling rate; the ZC is a side effect.
        from jaxonomy.library import DiscreteClock

        clk = DiscreteClock(dt=0.01, name="clk")
        # Manually splat a fake ZC event onto the leaf — the inference
        # helper must still prefer the periodic rate.
        clk._zero_crossing_events.append(object())
        st = infer_block_sample_time(clk)
        assert st.kind == "discrete"
        assert st.period == 0.01


# ---------------------------------------------------------------------
# group_blocks_by_rate buckets event-driven blocks separately
# ---------------------------------------------------------------------


class TestGroupEventDriven:
    def test_event_driven_in_own_bucket(self):
        from jaxonomy.library import Constant, DiscreteClock

        builder = jaxonomy.DiagramBuilder()
        builder.add(DiscreteClock(dt=0.01, name="clk"))
        builder.add(Constant(0.0, name="K"))
        zc = builder.add(
            ZeroCrossingTriggeredSubsystem(
                submodel=lambda u: jnp.asarray(u) + 1.0,
                n_inputs=1,
                initial_value=0.0,
                name="zc",
            )
        )
        # Wire trigger and input from the constant so create_context()
        # succeeds.
        const_for_trigger = builder.add(Constant(0.0, name="K_trig"))
        const_for_input = builder.add(Constant(1.0, name="K_in"))
        builder.connect(const_for_trigger.output_ports[0], zc.input_ports[0])
        builder.connect(const_for_input.output_ports[0], zc.input_ports[1])
        diag = builder.build()
        diag.create_context()

        groups = group_blocks_by_rate(diag)
        kinds = {k.kind for k in groups}
        assert "event_driven" in kinds
        ev_key = next(k for k in groups if k.kind == "event_driven")
        assert {leaf.name for leaf in groups[ev_key]} == {"zc"}


# ---------------------------------------------------------------------
# detect_rate_mismatches: event_driven ↔ periodic flags
# ---------------------------------------------------------------------


class TestDetectEventRateMismatches:
    def _build_event_to_periodic(self):
        """ZC-triggered (event_driven) -> UnitDelay (periodic).

        The event timing may not land on the UnitDelay's sample grid;
        ``detect_rate_mismatches`` must flag this connection.
        """
        from jaxonomy.library import Constant, UnitDelay

        builder = jaxonomy.DiagramBuilder()
        trig = builder.add(Constant(0.0, name="trig"))
        zc_in = builder.add(Constant(2.0, name="zc_in"))
        zc = builder.add(
            ZeroCrossingTriggeredSubsystem(
                submodel=lambda u: jnp.asarray(u) + 1.0,
                n_inputs=1,
                initial_value=0.0,
                name="zc",
            )
        )
        ud = builder.add(UnitDelay(dt=0.01, initial_state=0.0, name="ud"))
        builder.connect(trig.output_ports[0], zc.input_ports[0])
        builder.connect(zc_in.output_ports[0], zc.input_ports[1])
        builder.connect(zc.output_ports[0], ud.input_ports[0])
        diag = builder.build()
        diag.create_context()
        return diag

    def _build_event_to_event(self):
        """Two ZC-triggered blocks in series — no warning."""
        from jaxonomy.library import Constant

        builder = jaxonomy.DiagramBuilder()
        trig_a = builder.add(Constant(0.0, name="trig_a"))
        in_a = builder.add(Constant(1.0, name="in_a"))
        trig_b = builder.add(Constant(0.0, name="trig_b"))
        zc_a = builder.add(
            ZeroCrossingTriggeredSubsystem(
                submodel=lambda u: jnp.asarray(u) + 1.0,
                n_inputs=1,
                initial_value=0.0,
                name="zc_a",
            )
        )
        zc_b = builder.add(
            ZeroCrossingTriggeredSubsystem(
                submodel=lambda u: jnp.asarray(u) * 2.0,
                n_inputs=1,
                initial_value=0.0,
                name="zc_b",
            )
        )
        builder.connect(trig_a.output_ports[0], zc_a.input_ports[0])
        builder.connect(in_a.output_ports[0], zc_a.input_ports[1])
        builder.connect(trig_b.output_ports[0], zc_b.input_ports[0])
        builder.connect(zc_a.output_ports[0], zc_b.input_ports[1])
        diag = builder.build()
        diag.create_context()
        return diag

    def _build_continuous_to_event(self):
        """Integrator (continuous) -> ZC-triggered — no warning.

        Events sample continuous state on demand, so this connection is
        legal without any rate-transition machinery.
        """
        from jaxonomy.library import Constant, Integrator

        builder = jaxonomy.DiagramBuilder()
        const_x = builder.add(Constant(0.5, name="const_x"))
        integ = builder.add(Integrator(0.0, name="integ"))
        zc = builder.add(
            ZeroCrossingTriggeredSubsystem(
                submodel=lambda u: jnp.asarray(u) + 1.0,
                n_inputs=1,
                initial_value=0.0,
                name="zc",
            )
        )
        builder.connect(const_x.output_ports[0], integ.input_ports[0])
        # integ -> zc trigger.
        builder.connect(integ.output_ports[0], zc.input_ports[0])
        # And a constant feeds the user input (port 1).
        c_in = builder.add(Constant(2.0, name="c_in"))
        builder.connect(c_in.output_ports[0], zc.input_ports[1])
        diag = builder.build()
        diag.create_context()
        return diag

    def test_event_to_periodic_warns(self):
        diag = self._build_event_to_periodic()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            mismatches = detect_rate_mismatches(diag, on_mismatch="warn")

        assert len(mismatches) == 1
        mm = mismatches[0]
        assert isinstance(mm, RateMismatch)
        assert mm.src_system_name == "zc"
        assert mm.dst_system_name == "ud"
        assert mm.src_sample_time.is_event_driven()
        assert mm.dst_sample_time.is_discrete()

        rate_warnings = [
            w for w in caught if issubclass(w.category, RateMismatchWarning)
        ]
        assert len(rate_warnings) == 1
        msg = str(rate_warnings[0].message)
        # The specialised message must mention the event-timing concern,
        # not the generic RateTransition advice.
        assert "event timing may not align" in msg
        assert "zc" in msg and "ud" in msg

    def test_two_event_driven_blocks_silent(self):
        diag = self._build_event_to_event()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            mismatches = detect_rate_mismatches(diag, on_mismatch="warn")
        # The ZC-output → ZC-trigger wire is event_driven ↔ event_driven,
        # which is silent.  The Constant feeders are universal so they
        # don't fire either.
        assert mismatches == []
        assert not [
            w for w in caught if issubclass(w.category, RateMismatchWarning)
        ]

    def test_continuous_to_event_silent(self):
        diag = self._build_continuous_to_event()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            mismatches = detect_rate_mismatches(diag, on_mismatch="warn")
        assert mismatches == []
        assert not [
            w for w in caught if issubclass(w.category, RateMismatchWarning)
        ]

    def test_explicit_event_driven_attribute_flags_mismatch(self):
        """Honest fallback: a block author can set ``event_driven=True``
        directly without going through ``declare_zero_crossing``."""
        from jaxonomy.library import UnitDelay

        builder = jaxonomy.DiagramBuilder()
        # Build a tiny custom passthrough and tag it ``event_driven``.
        src = builder.add(_PassThrough(name="async_src"))
        src.event_driven = True
        # Feed the passthrough's input from a universal constant so
        # ``create_context()`` succeeds (the constant ↔ event_driven
        # edge is silent).
        from jaxonomy.library import Constant

        const = builder.add(Constant(0.0, name="K"))
        builder.connect(const.output_ports[0], src.input_ports[0])

        ud = builder.add(UnitDelay(dt=0.01, initial_state=0.0, name="ud"))
        builder.connect(src.output_ports[0], ud.input_ports[0])

        diag = builder.build()
        diag.create_context()

        mismatches = detect_rate_mismatches(diag, on_mismatch="collect")
        # The async_src → ud edge is event_driven → discrete, flagged.
        flagged = [
            mm for mm in mismatches
            if mm.src_system_name == "async_src" and mm.dst_system_name == "ud"
        ]
        assert len(flagged) == 1
        assert flagged[0].src_sample_time.is_event_driven()
