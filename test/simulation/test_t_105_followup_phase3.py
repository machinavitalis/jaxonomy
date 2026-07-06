# SPDX-License-Identifier: MIT

"""Tests for T-105-followup-phase3 — auto-insertion of ``RateTransition``
blocks at connect time.

Phase 3 ships an ``auto_insert_rate_transitions: bool = False`` flag on
:class:`DiagramBuilder`.  When enabled, :meth:`DiagramBuilder.connect`
synthesises a :func:`jaxonomy.library.RateTransition` block (a
``ZeroOrderHold`` for slow→fast, a :class:`Decimator` for fast→slow)
between any two adjacent leaves whose inferred discrete sample times
differ.  Default ``False`` keeps the legacy code path byte-equivalent
(the strict mode that surfaces rate mismatches rather than silently
inserting transitions).

The auto-insertion path must:

* stay completely silent when the flag is left at its default;
* produce a fast→slow :class:`Decimator` for ``src(dt=0.1) → dst(dt=0.5)``;
* produce a slow→fast :class:`ZeroOrderHold` for ``src(dt=0.5) → dst(dt=0.1)``;
* skip insertion when source and destination already match;
* compose with ``validate_rates_at_connect="warn"`` (warning fires AND
  the bridge gets inserted).
"""

from __future__ import annotations

import warnings

import pytest

import jaxonomy
from jaxonomy.framework import LeafSystem
from jaxonomy.simulation.rate_groups import (
    RateMismatchWarning,
    SampleTime,
    assert_no_rate_mismatches,
    infer_block_sample_time,
)


pytestmark = pytest.mark.minimal


# --------------------------------------------------------------------- #
# Tiny pass-through sink so the slow signal has somewhere to land in
# any ``build()`` flow we may exercise.
# --------------------------------------------------------------------- #


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


def _registered_names(builder):
    return [s.name for s in builder._registered_systems]


def _bridges(builder):
    return [
        s
        for s in builder._registered_systems
        if getattr(s, "_jaxonomy_rate_transition", False)
    ]


# =====================================================================
# Default-off path: pre-existing T-105 + T-123 behaviour preserved.
# =====================================================================


class TestDefaultOffByteEquivalent:
    def test_default_no_auto_insert(self):
        """``auto_insert_rate_transitions=False`` (default) must not insert
        any block when the rates differ.  This is the byte-equivalent
        guarantee that protects every existing diagram in the corpus."""
        from jaxonomy.library import DiscreteClock, ZeroOrderHold

        builder = jaxonomy.DiagramBuilder()  # default: off
        clk = builder.add(DiscreteClock(dt=0.1, name="clk_fast"))
        zoh = builder.add(ZeroOrderHold(dt=0.5, name="zoh_slow"))

        builder.connect(clk.output_ports[0], zoh.input_ports[0])

        # Only the two user-registered systems should be present.
        assert _registered_names(builder) == ["clk_fast", "zoh_slow"]
        assert _bridges(builder) == []

    def test_default_no_auto_insert_with_validate_warn(self):
        """Validate-only mode still warns but does not insert."""
        from jaxonomy.library import DiscreteClock, ZeroOrderHold

        builder = jaxonomy.DiagramBuilder(validate_rates_at_connect="warn")
        clk = builder.add(DiscreteClock(dt=0.1, name="clk_fast"))
        zoh = builder.add(ZeroOrderHold(dt=0.5, name="zoh_slow"))

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            builder.connect(clk.output_ports[0], zoh.input_ports[0])

        rate_warnings = [
            w for w in caught if issubclass(w.category, RateMismatchWarning)
        ]
        assert len(rate_warnings) == 1
        assert _bridges(builder) == []  # validate-only never inserts


# =====================================================================
# Auto-insertion: fast → slow yields a Decimator.
# =====================================================================


class TestAutoInsertFastToSlow:
    def test_fast_to_slow_inserts_decimator(self):
        from jaxonomy.library import Decimator, DiscreteClock, ZeroOrderHold

        builder = jaxonomy.DiagramBuilder(auto_insert_rate_transitions=True)
        clk = builder.add(DiscreteClock(dt=0.1, name="clk_fast"))
        zoh = builder.add(ZeroOrderHold(dt=0.5, name="zoh_slow"))

        builder.connect(clk.output_ports[0], zoh.input_ports[0])

        bridges = _bridges(builder)
        assert len(bridges) == 1
        bridge = bridges[0]
        # Fast → slow gets a Decimator at the slow rate.
        assert isinstance(bridge, Decimator)
        assert bridge.input_dt == 0.1
        assert bridge.output_dt == 0.5
        # Wiring: clk → bridge → zoh.
        assert builder._connection_map[bridge.input_ports[0].locator] == (
            clk.output_ports[0].locator
        )
        assert builder._connection_map[zoh.input_ports[0].locator] == (
            bridge.output_ports[0].locator
        )

    def test_fast_to_slow_diagram_builds_silently(self):
        """The auto-inserted bridge satisfies ``assert_no_rate_mismatches``
        on the resulting diagram (the bridge marker silences the walker)."""
        from jaxonomy.library import DiscreteClock, ZeroOrderHold

        builder = jaxonomy.DiagramBuilder(auto_insert_rate_transitions=True)
        clk = builder.add(DiscreteClock(dt=0.1, name="clk_fast"))
        zoh = builder.add(ZeroOrderHold(dt=0.5, name="zoh_slow"))
        sink = builder.add(_PassThrough(name="sink"))

        builder.connect(clk.output_ports[0], zoh.input_ports[0])
        builder.connect(zoh.output_ports[0], sink.input_ports[0])

        diagram = builder.build()
        # Should not raise: the auto-inserted bridge silences the walker.
        result = assert_no_rate_mismatches(diagram, on_mismatch="error")
        assert result == []


# =====================================================================
# Auto-insertion: slow → fast yields a ZeroOrderHold.
# =====================================================================


class TestAutoInsertSlowToFast:
    def test_slow_to_fast_inserts_zoh(self):
        from jaxonomy.library import DiscreteClock, ZeroOrderHold

        builder = jaxonomy.DiagramBuilder(auto_insert_rate_transitions=True)
        # Use a slow DiscreteClock (period 0.5s) feeding a fast ZOH (0.1s).
        slow = builder.add(DiscreteClock(dt=0.5, name="clk_slow"))
        fast = builder.add(ZeroOrderHold(dt=0.1, name="zoh_fast"))

        builder.connect(slow.output_ports[0], fast.input_ports[0])

        bridges = _bridges(builder)
        assert len(bridges) == 1
        bridge = bridges[0]
        # Slow → fast: RateTransition factory returns a ZeroOrderHold
        # at the fast rate, tagged with the rate-transition marker.
        assert isinstance(bridge, ZeroOrderHold)
        assert bridge._jaxonomy_rate_transition is True
        # The bridge runs at the fast rate (0.1s).
        bridge_st = infer_block_sample_time(bridge)
        assert bridge_st.kind == "discrete"
        assert bridge_st.period == 0.1


# =====================================================================
# Same rate / universal sources: no insertion.
# =====================================================================


class TestNoInsertionWhenUnnecessary:
    def test_matching_rates_no_insertion(self):
        from jaxonomy.library import DiscreteClock, ZeroOrderHold

        builder = jaxonomy.DiagramBuilder(auto_insert_rate_transitions=True)
        clk = builder.add(DiscreteClock(dt=0.1, name="clk"))
        zoh = builder.add(ZeroOrderHold(dt=0.1, name="zoh"))

        builder.connect(clk.output_ports[0], zoh.input_ports[0])

        # Matched rates: no bridge inserted; map stays direct.
        assert _bridges(builder) == []
        assert builder._connection_map[zoh.input_ports[0].locator] == (
            clk.output_ports[0].locator
        )

    def test_constant_source_no_insertion(self):
        """``Constant`` is universal-rate; no bridge is needed."""
        from jaxonomy.library import Constant, ZeroOrderHold

        builder = jaxonomy.DiagramBuilder(auto_insert_rate_transitions=True)
        k = builder.add(Constant(1.0, name="K"))
        zoh = builder.add(ZeroOrderHold(dt=0.5, name="zoh"))

        builder.connect(k.output_ports[0], zoh.input_ports[0])

        assert _bridges(builder) == []

    def test_existing_rate_transition_not_re_bridged(self):
        """If the user already inserted a RateTransition, do not re-bridge."""
        from jaxonomy.library import DiscreteClock, RateTransition, ZeroOrderHold

        builder = jaxonomy.DiagramBuilder(auto_insert_rate_transitions=True)
        clk = builder.add(DiscreteClock(dt=0.1, name="clk_fast"))
        rt = builder.add(
            RateTransition(input_dt=0.1, output_dt=0.5, name="user_rt")
        )
        zoh = builder.add(ZeroOrderHold(dt=0.5, name="zoh_slow"))

        builder.connect(clk.output_ports[0], rt.input_ports[0])
        builder.connect(rt.output_ports[0], zoh.input_ports[0])

        # Only the user-supplied bridge should be present (no extras).
        bridges = _bridges(builder)
        assert len(bridges) == 1
        assert bridges[0].name == "user_rt"


# =====================================================================
# Composition with validate_rates_at_connect="warn".
# =====================================================================


class TestComposesWithValidate:
    def test_warn_and_auto_insert_both_fire(self):
        """Both flags on: warning still fires AND bridge gets inserted."""
        from jaxonomy.library import Decimator, DiscreteClock, ZeroOrderHold

        builder = jaxonomy.DiagramBuilder(
            auto_insert_rate_transitions=True,
            validate_rates_at_connect="warn",
        )
        clk = builder.add(DiscreteClock(dt=0.1, name="clk_fast"))
        zoh = builder.add(ZeroOrderHold(dt=0.5, name="zoh_slow"))

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            builder.connect(clk.output_ports[0], zoh.input_ports[0])

        # Warning fires (validate_rates_at_connect contract preserved).
        rate_warnings = [
            w for w in caught if issubclass(w.category, RateMismatchWarning)
        ]
        assert len(rate_warnings) == 1
        assert "clk_fast" in str(rate_warnings[0].message)
        assert "zoh_slow" in str(rate_warnings[0].message)

        # And the bridge was inserted (auto-insert contract preserved).
        bridges = _bridges(builder)
        assert len(bridges) == 1
        assert isinstance(bridges[0], Decimator)
