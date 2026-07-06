# SPDX-License-Identifier: MIT

"""Tests for T-105-followup-phase2 — port-level connect-time
multirate consistency check + explicit ``sample_time`` attribute.

Phase 2 ships:

* An optional ``sample_time`` attribute on blocks that
  :func:`infer_block_sample_time` honours when present (escape hatch
  for blocks whose periodic events are configured in ``initialize()``
  rather than ``__init__``).
* :func:`check_connection_rate_compat` — a single-connection variant
  of :func:`detect_rate_mismatches` for connect-time wiring.
* A new ``DiagramBuilder(validate_rates_at_connect=...)`` flag that
  routes every :meth:`DiagramBuilder.connect` call through that
  helper.  Default ``None`` keeps the legacy code path
  byte-equivalent.
* :func:`assert_no_rate_mismatches` — post-build helper that walks
  every connection and raises (or warns) on the first incompatibility.

The connect-time diagnostic must:

* warn when two adjacent blocks have incompatible discrete rates and
  no ``RateTransition`` block sits between them;
* stay silent when a ``RateTransition`` (``ZeroOrderHold`` /
  ``Decimator`` / etc. tagged ``_jaxonomy_rate_transition=True``) is
  inserted;
* stay silent when the option is left at its default
  (byte-equivalent).
"""

from __future__ import annotations

import warnings

import pytest

import jaxonomy
from jaxonomy.framework import LeafSystem
from jaxonomy.framework.diagram_builder import BuilderError
from jaxonomy.simulation.rate_groups import (
    RateMismatch,
    RateMismatchError,
    RateMismatchWarning,
    SampleTime,
    assert_no_rate_mismatches,
    check_connection_rate_compat,
    infer_block_sample_time,
)


pytestmark = pytest.mark.minimal


# ---------------------------------------------------------------------
# Tiny pass-through sink so we have somewhere for the slow signal to
# flow.  Mirrors helpers in the Phase 1 test file but stays local.
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


# =====================================================================
# Explicit ``sample_time`` attribute escape hatch.
# =====================================================================


class _ExplicitlyDiscrete(LeafSystem):
    """A block that declares its rate via an explicit attribute.

    Used to verify that :func:`infer_block_sample_time` honours the
    explicit declaration even when no periodic events are wired up.
    """

    def __init__(self, *, period, name=None, **kwargs):
        super().__init__(name=name, **kwargs)
        self.declare_input_port(name="u")
        self.declare_output_port(
            self._eval,
            name="y",
            requires_inputs=True,
            prerequisites_of_calc=[self.input_ports[0].ticket],
        )
        # T-105 Phase 2: explicit declaration honored over event scan.
        self.sample_time = SampleTime.discrete(period=period)

    def _eval(self, _time, _state, *inputs, **_params):
        return inputs[0]


class TestExplicitSampleTimeAttribute:
    def test_explicit_sample_time_honored(self):
        block = _ExplicitlyDiscrete(period=0.05, name="ed")
        st = infer_block_sample_time(block)
        assert st.kind == "discrete"
        assert st.period == 0.05

    def test_explicit_overrides_event_inference(self):
        # If a block has both an explicit attribute *and* periodic events,
        # the explicit attribute wins (it is the authoritative declaration).
        from jaxonomy.library import DiscreteClock

        block = DiscreteClock(dt=0.01, name="clk")
        # Override: caller declares the block belongs to a 0.5s rate group.
        block.sample_time = SampleTime.discrete(period=0.5)
        st = infer_block_sample_time(block)
        assert st.kind == "discrete"
        assert st.period == 0.5

    def test_non_sample_time_attribute_ignored(self):
        # Accidental ``sample_time = "fast"`` (string, not a SampleTime
        # instance) should not crash inference; we silently skip and
        # fall back to event-based inference.
        from jaxonomy.library import DiscreteClock

        block = DiscreteClock(dt=0.01, name="clk")
        block.sample_time = "fast"  # not a SampleTime instance
        st = infer_block_sample_time(block)
        # Falls back to event-based inference.
        assert st.kind == "discrete"
        assert st.period == 0.01


# =====================================================================
# check_connection_rate_compat — single-connection helper.
# =====================================================================


class TestCheckConnectionRateCompat:
    def test_matching_rates_returns_none(self):
        from jaxonomy.library import DiscreteClock, ZeroOrderHold

        src = DiscreteClock(dt=0.1, name="clk")
        dst = ZeroOrderHold(dt=0.1, name="zoh")
        result = check_connection_rate_compat(
            src, 0, dst, 0, on_mismatch="collect"
        )
        assert result is None

    def test_mismatched_rates_returns_record(self):
        from jaxonomy.library import DiscreteClock, ZeroOrderHold

        src = DiscreteClock(dt=0.1, name="clk")
        dst = ZeroOrderHold(dt=0.5, name="zoh")
        result = check_connection_rate_compat(
            src, 0, dst, 0, on_mismatch="collect"
        )
        assert isinstance(result, RateMismatch)
        assert result.src_system_name == "clk"
        assert result.dst_system_name == "zoh"

    def test_warn_emits_warning(self):
        from jaxonomy.library import DiscreteClock, ZeroOrderHold

        src = DiscreteClock(dt=0.1, name="clk")
        dst = ZeroOrderHold(dt=0.5, name="zoh")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = check_connection_rate_compat(
                src, 0, dst, 0, on_mismatch="warn"
            )
        assert isinstance(result, RateMismatch)
        rate_warnings = [
            w for w in caught if issubclass(w.category, RateMismatchWarning)
        ]
        assert len(rate_warnings) == 1
        assert "clk" in str(rate_warnings[0].message)
        assert "zoh" in str(rate_warnings[0].message)

    def test_error_raises(self):
        from jaxonomy.library import DiscreteClock, ZeroOrderHold

        src = DiscreteClock(dt=0.1, name="clk")
        dst = ZeroOrderHold(dt=0.5, name="zoh")
        with pytest.raises(RateMismatchError) as info:
            check_connection_rate_compat(src, 0, dst, 0, on_mismatch="error")
        assert "clk" in str(info.value)
        assert "zoh" in str(info.value)

    def test_rate_transition_marker_silences_check(self):
        from jaxonomy.library import DiscreteClock, ZeroOrderHold

        src = DiscreteClock(dt=0.1, name="clk")
        dst = ZeroOrderHold(dt=0.5, name="zoh")
        # Tag the destination as a rate-transition bridge.
        dst._jaxonomy_rate_transition = True
        result = check_connection_rate_compat(
            src, 0, dst, 0, on_mismatch="error"
        )
        assert result is None


# =====================================================================
# DiagramBuilder.connect — opt-in connect-time enforcement.
# =====================================================================


class TestConnectTimeValidation:
    def test_default_off_no_warning_on_mismatch(self):
        """Default-off path: builder silently allows mismatched rates."""
        from jaxonomy.library import DiscreteClock, ZeroOrderHold

        builder = jaxonomy.DiagramBuilder()  # no flag
        clk = builder.add(DiscreteClock(dt=0.1, name="clk_fast"))
        zoh = builder.add(ZeroOrderHold(dt=0.5, name="zoh_slow"))
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            builder.connect(clk.output_ports[0], zoh.input_ports[0])
        rate_warnings = [
            w for w in caught if issubclass(w.category, RateMismatchWarning)
        ]
        assert rate_warnings == []

    def test_warn_emits_at_connect_time(self):
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
        assert "clk_fast" in str(rate_warnings[0].message)
        assert "zoh_slow" in str(rate_warnings[0].message)

    def test_true_is_sugar_for_warn(self):
        """``validate_rates_at_connect=True`` is treated like ``"warn"``."""
        from jaxonomy.library import DiscreteClock, ZeroOrderHold

        builder = jaxonomy.DiagramBuilder(validate_rates_at_connect=True)
        clk = builder.add(DiscreteClock(dt=0.1, name="clk"))
        zoh = builder.add(ZeroOrderHold(dt=0.5, name="zoh"))
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            builder.connect(clk.output_ports[0], zoh.input_ports[0])
        rate_warnings = [
            w for w in caught if issubclass(w.category, RateMismatchWarning)
        ]
        assert len(rate_warnings) == 1

    def test_error_raises_at_connect_time(self):
        from jaxonomy.library import DiscreteClock, ZeroOrderHold

        builder = jaxonomy.DiagramBuilder(validate_rates_at_connect="error")
        clk = builder.add(DiscreteClock(dt=0.1, name="clk_fast"))
        zoh = builder.add(ZeroOrderHold(dt=0.5, name="zoh_slow"))
        with pytest.raises(RateMismatchError) as info:
            builder.connect(clk.output_ports[0], zoh.input_ports[0])
        assert "clk_fast" in str(info.value)
        assert "zoh_slow" in str(info.value)

    def test_rate_transition_block_silences_warning(self):
        """clk(0.1) -> RateTransition -> zoh(0.5): no warning."""
        from jaxonomy.library import (
            DiscreteClock,
            RateTransition,
            ZeroOrderHold,
        )

        builder = jaxonomy.DiagramBuilder(validate_rates_at_connect="warn")
        clk = builder.add(DiscreteClock(dt=0.1, name="clk_fast"))
        rt = builder.add(
            RateTransition(input_dt=0.1, output_dt=0.5, name="rt")
        )
        zoh = builder.add(ZeroOrderHold(dt=0.5, name="zoh_slow"))

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            builder.connect(clk.output_ports[0], rt.input_ports[0])
            builder.connect(rt.output_ports[0], zoh.input_ports[0])

        rate_warnings = [
            w for w in caught if issubclass(w.category, RateMismatchWarning)
        ]
        assert rate_warnings == [], (
            f"expected no rate warnings; got {[str(w.message) for w in rate_warnings]}"
        )

    def test_matched_rates_no_warning(self):
        from jaxonomy.library import DiscreteClock, ZeroOrderHold

        builder = jaxonomy.DiagramBuilder(validate_rates_at_connect="warn")
        clk = builder.add(DiscreteClock(dt=0.1, name="clk"))
        zoh = builder.add(ZeroOrderHold(dt=0.1, name="zoh"))
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            builder.connect(clk.output_ports[0], zoh.input_ports[0])
        rate_warnings = [
            w for w in caught if issubclass(w.category, RateMismatchWarning)
        ]
        assert rate_warnings == []

    def test_constant_source_universal(self):
        """Constant -> any-rate: never warns."""
        from jaxonomy.library import Constant, ZeroOrderHold

        builder = jaxonomy.DiagramBuilder(validate_rates_at_connect="warn")
        k = builder.add(Constant(1.0, name="K"))
        zoh = builder.add(ZeroOrderHold(dt=0.5, name="zoh"))
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            builder.connect(k.output_ports[0], zoh.input_ports[0])
        rate_warnings = [
            w for w in caught if issubclass(w.category, RateMismatchWarning)
        ]
        assert rate_warnings == []

    def test_invalid_flag_value_rejected(self):
        with pytest.raises(BuilderError):
            jaxonomy.DiagramBuilder(validate_rates_at_connect="loud")


# =====================================================================
# assert_no_rate_mismatches — post-build helper.
# =====================================================================


class TestAssertNoRateMismatchesHelper:
    def test_raises_on_mismatched_diagram(self):
        from jaxonomy.library import DiscreteClock, ZeroOrderHold

        builder = jaxonomy.DiagramBuilder()  # off at connect time
        clk = builder.add(DiscreteClock(dt=0.1, name="clk_fast"))
        zoh = builder.add(ZeroOrderHold(dt=0.5, name="zoh_slow"))
        passthrough = builder.add(_PassThrough(name="sink"))
        builder.connect(clk.output_ports[0], zoh.input_ports[0])
        builder.connect(zoh.output_ports[0], passthrough.input_ports[0])
        diag = builder.build()

        with pytest.raises(RateMismatchError):
            assert_no_rate_mismatches(diag)

    def test_silent_with_rate_transition(self):
        from jaxonomy.library import (
            DiscreteClock,
            RateTransition,
            ZeroOrderHold,
        )

        builder = jaxonomy.DiagramBuilder()
        clk = builder.add(DiscreteClock(dt=0.1, name="clk_fast"))
        rt = builder.add(
            RateTransition(input_dt=0.1, output_dt=0.5, name="rt")
        )
        zoh = builder.add(ZeroOrderHold(dt=0.5, name="zoh_slow"))
        passthrough = builder.add(_PassThrough(name="sink"))
        builder.connect(clk.output_ports[0], rt.input_ports[0])
        builder.connect(rt.output_ports[0], zoh.input_ports[0])
        builder.connect(zoh.output_ports[0], passthrough.input_ports[0])
        diag = builder.build()

        # Should not raise — the RateTransition bridges the rates.
        result = assert_no_rate_mismatches(diag)
        assert result == []

    def test_warn_mode_returns_collected(self):
        from jaxonomy.library import DiscreteClock, ZeroOrderHold

        builder = jaxonomy.DiagramBuilder()
        clk = builder.add(DiscreteClock(dt=0.1, name="clk_fast"))
        zoh = builder.add(ZeroOrderHold(dt=0.5, name="zoh_slow"))
        passthrough = builder.add(_PassThrough(name="sink"))
        builder.connect(clk.output_ports[0], zoh.input_ports[0])
        builder.connect(zoh.output_ports[0], passthrough.input_ports[0])
        diag = builder.build()

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            mismatches = assert_no_rate_mismatches(diag, on_mismatch="warn")

        assert len(mismatches) >= 1
        rate_warnings = [
            w for w in caught if issubclass(w.category, RateMismatchWarning)
        ]
        assert len(rate_warnings) >= 1
