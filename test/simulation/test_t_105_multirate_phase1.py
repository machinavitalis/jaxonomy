# SPDX-License-Identifier: MIT

"""Tests for T-105 Phase 1 — multirate sample-time inference and
rate-mismatch detection.

These cover:

* :class:`SampleTime` algebra (``matches``, ``is_universal``, ...);
* :func:`infer_block_sample_time` correctly classifies ``Constant``,
  ``Integrator``, ``DiscreteClock``, and ``UnitDelay`` blocks;
* :func:`group_blocks_by_rate` buckets a multirate diagram into the
  expected groups;
* :func:`detect_rate_mismatches` warns / errors / collects on a 2-rate
  diagram, and stays silent on a single-rate diagram;
* default-off byte-equivalence: ``simulate`` with the new option *off*
  produces identical recorded outputs to a baseline simulation on a
  single-rate diagram.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

import jaxonomy
from jaxonomy.framework import LeafSystem, DependencyTicket
from jaxonomy.simulation.rate_groups import (
    RateMismatch,
    RateMismatchError,
    RateMismatchWarning,
    SampleTime,
    detect_rate_mismatches,
    format_rate_groups,
    group_blocks_by_rate,
    infer_block_sample_time,
    iter_rate_groups,
)
from jaxonomy.simulation.types import SimulatorOptions


pytestmark = pytest.mark.minimal


# ---------------------------------------------------------------------
# Tiny pass-through used to wire the diagrams below.  Mirrors the
# helper in T-104's tests but stays local so the two files don't
# share fixtures.
# ---------------------------------------------------------------------


class _PassThrough(LeafSystem):
    def __init__(self, *, name=None, **kwargs):
        super().__init__(name=name, **kwargs)
        self.declare_input_port(name="u")
        self._out_idx = self.declare_output_port(
            self._eval,
            name="y",
            requires_inputs=True,
            prerequisites_of_calc=[self.input_ports[0].ticket],
        )

    def _eval(self, _time, _state, *inputs, **_params):
        return inputs[0]


# =====================================================================
# SampleTime value type
# =====================================================================


class TestSampleTime:
    def test_continuous_constant_inherited_factories(self):
        assert SampleTime.continuous().kind == "continuous"
        assert SampleTime.constant().kind == "constant"
        assert SampleTime.inherited().kind == "inherited"

    def test_discrete_factory_normalises(self):
        st = SampleTime.discrete(period=0.01, offset=0.0)
        assert st.kind == "discrete"
        assert st.period == 0.01
        assert st.offset == 0.0

    def test_discrete_rejects_nonpositive_period(self):
        with pytest.raises(ValueError):
            SampleTime.discrete(period=0.0)
        with pytest.raises(ValueError):
            SampleTime.discrete(period=-1.0)

    def test_discrete_rejects_nonfinite_period(self):
        with pytest.raises(ValueError):
            SampleTime.discrete(period=float("inf"))

    def test_universal_kinds(self):
        assert SampleTime.constant().is_universal()
        assert SampleTime.inherited().is_universal()
        assert not SampleTime.continuous().is_universal()
        assert not SampleTime.discrete(period=0.01).is_universal()

    def test_matches_universal_to_anything(self):
        c = SampleTime.constant()
        d = SampleTime.discrete(period=0.01)
        cont = SampleTime.continuous()
        assert c.matches(d)
        assert d.matches(c)
        assert c.matches(cont)
        assert cont.matches(c)

    def test_matches_continuous_to_continuous(self):
        a = SampleTime.continuous()
        b = SampleTime.continuous()
        assert a.matches(b)

    def test_matches_discrete_same_period(self):
        a = SampleTime.discrete(period=0.01, offset=0.0)
        b = SampleTime.discrete(period=0.01, offset=0.0)
        assert a.matches(b)

    def test_mismatch_discrete_different_period(self):
        a = SampleTime.discrete(period=0.01)
        b = SampleTime.discrete(period=0.10)
        assert not a.matches(b)

    def test_offset_alone_is_not_a_mismatch(self):
        # Phase 1 deliberately compares periods only.  Two-phase
        # x⁻ atomicity within a rate group routinely produces same-
        # period / different-offset event pairs (see UnitDelay's
        # output port at offset=0 and state update at offset=dt).
        # Phase 2 / T-123 will refine this.
        a = SampleTime.discrete(period=0.01, offset=0.0)
        b = SampleTime.discrete(period=0.01, offset=0.005)
        assert a.matches(b)

    def test_mismatch_continuous_to_discrete(self):
        cont = SampleTime.continuous()
        disc = SampleTime.discrete(period=0.01)
        assert not cont.matches(disc)
        assert not disc.matches(cont)


# =====================================================================
# infer_block_sample_time
# =====================================================================


class TestInferBlockSampleTime:
    def test_constant_block_is_constant(self):
        from jaxonomy.library import Constant

        block = Constant(1.0, name="K")
        st = infer_block_sample_time(block)
        assert st.kind == "constant"

    def test_integrator_block_is_continuous(self):
        from jaxonomy.library import Integrator

        block = Integrator(0.0, name="I")
        st = infer_block_sample_time(block)
        assert st.kind == "continuous"

    def test_discrete_clock_is_discrete(self):
        from jaxonomy.library import DiscreteClock

        block = DiscreteClock(dt=0.01, name="clk")
        st = infer_block_sample_time(block)
        assert st.kind == "discrete"
        assert st.period == 0.01

    def test_unit_delay_is_discrete(self):
        # ``UnitDelay`` configures its periodic update inside
        # ``initialize()``, which only runs at context-creation time.
        # Build a tiny diagram (with the UnitDelay's input wired so
        # ``create_context()`` doesn't trip the disconnected-input
        # check) and then introspect.
        from jaxonomy.library import Constant, UnitDelay

        builder = jaxonomy.DiagramBuilder()
        src = builder.add(Constant(0.0, name="src"))
        block = builder.add(UnitDelay(dt=0.05, initial_state=0.0, name="ud"))
        builder.connect(src.output_ports[0], block.input_ports[0])
        diag = builder.build()
        diag.create_context()
        st = infer_block_sample_time(block)
        assert st.kind == "discrete"
        assert st.period == 0.05


# =====================================================================
# group_blocks_by_rate
# =====================================================================


class TestGroupBlocksByRate:
    def test_two_rate_diagram_buckets_correctly(self):
        from jaxonomy.library import DiscreteClock, UnitDelay

        builder = jaxonomy.DiagramBuilder()
        clk_fast = builder.add(DiscreteClock(dt=0.01, name="clk_fast"))
        ud_fast = builder.add(UnitDelay(dt=0.01, initial_state=0.0, name="ud_fast"))
        ud_slow = builder.add(UnitDelay(dt=0.10, initial_state=0.0, name="ud_slow"))
        builder.connect(clk_fast.output_ports[0], ud_fast.input_ports[0])
        # ud_slow is fed from ud_fast (rate mismatch — but build still
        # succeeds; we are only inspecting groups here).
        builder.connect(ud_fast.output_ports[0], ud_slow.input_ports[0])
        diag = builder.build()
        diag.create_context()  # triggers UnitDelay.initialize()

        groups = group_blocks_by_rate(diag)
        # Three discrete blocks total: two at 0.01s, one at 0.10s
        keys = list(groups.keys())
        periods = sorted(k.period for k in keys if k.is_discrete())
        assert periods == [0.01, 0.10]

        fast_key = next(k for k in keys if k.is_discrete() and k.period == 0.01)
        slow_key = next(k for k in keys if k.is_discrete() and k.period == 0.10)
        assert {leaf.name for leaf in groups[fast_key]} == {"clk_fast", "ud_fast"}
        assert {leaf.name for leaf in groups[slow_key]} == {"ud_slow"}

    def test_format_includes_rate_groups(self):
        from jaxonomy.library import DiscreteClock, Constant

        builder = jaxonomy.DiagramBuilder()
        builder.add(DiscreteClock(dt=0.01, name="clk"))
        builder.add(Constant(1.0, name="K"))
        diag = builder.build()
        out = format_rate_groups(diag)
        assert "clk" in out
        assert "K" in out
        assert "discrete" in out
        assert "constant" in out

    def test_iter_rate_groups_orders_discrete_first(self):
        from jaxonomy.library import Constant, DiscreteClock

        builder = jaxonomy.DiagramBuilder()
        builder.add(Constant(1.0, name="K"))
        builder.add(DiscreteClock(dt=0.10, name="clk_slow"))
        builder.add(DiscreteClock(dt=0.01, name="clk_fast"))
        diag = builder.build()
        ordered = list(iter_rate_groups(diag))
        # Discrete groups come first, smallest period first.
        assert ordered[0][0].kind == "discrete"
        assert ordered[0][0].period == 0.01
        assert ordered[1][0].kind == "discrete"
        assert ordered[1][0].period == 0.10
        # Constant group comes last.
        assert ordered[-1][0].kind == "constant"


# =====================================================================
# detect_rate_mismatches
# =====================================================================


class TestDetectRateMismatches:
    def _build_two_rate_diagram(self):
        """clk_fast(0.01s) -> ud_slow(0.10s) — mismatched rates."""
        from jaxonomy.library import DiscreteClock, UnitDelay

        builder = jaxonomy.DiagramBuilder()
        clk_fast = builder.add(DiscreteClock(dt=0.01, name="clk_fast"))
        ud_slow = builder.add(UnitDelay(dt=0.10, initial_state=0.0, name="ud_slow"))
        builder.connect(clk_fast.output_ports[0], ud_slow.input_ports[0])
        diag = builder.build()
        diag.create_context()  # triggers UnitDelay.initialize()
        return diag

    def _build_single_rate_diagram(self):
        """clk(0.01s) -> ud(0.01s) — matched rates."""
        from jaxonomy.library import DiscreteClock, UnitDelay

        builder = jaxonomy.DiagramBuilder()
        clk = builder.add(DiscreteClock(dt=0.01, name="clk"))
        ud = builder.add(UnitDelay(dt=0.01, initial_state=0.0, name="ud"))
        builder.connect(clk.output_ports[0], ud.input_ports[0])
        diag = builder.build()
        diag.create_context()
        return diag

    def _build_constant_to_discrete_diagram(self):
        """Constant -> UnitDelay — no mismatch (constant is universal)."""
        from jaxonomy.library import Constant, UnitDelay

        builder = jaxonomy.DiagramBuilder()
        k = builder.add(Constant(1.0, name="K"))
        ud = builder.add(UnitDelay(dt=0.01, initial_state=0.0, name="ud"))
        builder.connect(k.output_ports[0], ud.input_ports[0])
        diag = builder.build()
        diag.create_context()
        return diag

    def test_warn_on_mismatch(self):
        diag = self._build_two_rate_diagram()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            mismatches = detect_rate_mismatches(diag, on_mismatch="warn")

        assert len(mismatches) == 1
        mm = mismatches[0]
        assert isinstance(mm, RateMismatch)
        assert mm.src_system_name == "clk_fast"
        assert mm.dst_system_name == "ud_slow"
        # Exactly one warning of the right type was emitted.
        rate_warnings = [w for w in caught if issubclass(w.category, RateMismatchWarning)]
        assert len(rate_warnings) == 1
        assert "clk_fast" in str(rate_warnings[0].message)
        assert "ud_slow" in str(rate_warnings[0].message)

    def test_error_on_mismatch_raises(self):
        diag = self._build_two_rate_diagram()
        with pytest.raises(RateMismatchError) as info:
            detect_rate_mismatches(diag, on_mismatch="error")
        assert "clk_fast" in str(info.value)
        assert "ud_slow" in str(info.value)

    def test_collect_silent(self):
        diag = self._build_two_rate_diagram()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            mismatches = detect_rate_mismatches(diag, on_mismatch="collect")
        # No warnings emitted; mismatches still returned.
        assert not [w for w in caught if issubclass(w.category, RateMismatchWarning)]
        assert len(mismatches) == 1

    def test_single_rate_diagram_silent(self):
        diag = self._build_single_rate_diagram()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            mismatches = detect_rate_mismatches(diag, on_mismatch="warn")
        assert mismatches == []
        assert not [w for w in caught if issubclass(w.category, RateMismatchWarning)]

    def test_constant_source_treated_as_universal(self):
        diag = self._build_constant_to_discrete_diagram()
        mismatches = detect_rate_mismatches(diag, on_mismatch="collect")
        assert mismatches == []


# =====================================================================
# Default-off byte-equivalence: existing single-rate diagrams unchanged.
# =====================================================================


class TestDefaultOffByteEquivalence:
    """The new ``check_rate_transitions`` option defaults to ``None``.

    With it left untouched, ``simulate`` must produce the exact same
    recorded outputs as before this task landed.  We exercise this on a
    minimal Sine -> Integrator -> passthrough diagram.
    """

    def _build(self):
        from jaxonomy.library import Sine, Integrator

        builder = jaxonomy.DiagramBuilder()
        sine = builder.add(Sine(name="Sin_0"))
        integ = builder.add(Integrator(0.0, name="Integrator_0"))
        passthrough = builder.add(_PassThrough(name="pass"))
        builder.connect(sine.output_ports[0], integ.input_ports[0])
        builder.connect(integ.output_ports[0], passthrough.input_ports[0])
        return builder.build(), passthrough

    def test_default_off_matches_baseline(self):
        diag_a, pt_a = self._build()
        diag_b, pt_b = self._build()

        ctx_a = diag_a.create_context()
        ctx_b = diag_b.create_context()

        # A: no options at all (the truly default path).
        results_a = jaxonomy.simulate(
            diag_a, ctx_a, t_span=(0.0, 1.0),
            recorded_signals={"y": pt_a.output_ports[0]},
        )
        # B: explicit options with ``check_rate_transitions=None``.
        opts = SimulatorOptions(check_rate_transitions=None)
        results_b = jaxonomy.simulate(
            diag_b, ctx_b, t_span=(0.0, 1.0), options=opts,
            recorded_signals={"y": pt_b.output_ports[0]},
        )

        np.testing.assert_array_equal(
            np.asarray(results_a.outputs["y"]),
            np.asarray(results_b.outputs["y"]),
        )
        np.testing.assert_array_equal(
            np.asarray(results_a.time), np.asarray(results_b.time)
        )

    def test_default_simulator_options_field_is_none(self):
        # Guard against accidental flips of the default that would
        # silently turn the check on for every user.
        opts = SimulatorOptions()
        assert opts.check_rate_transitions is None


# =====================================================================
# Opt-in path: simulate() emits warnings on a multirate-mismatched
# diagram when the user asks for them.
# =====================================================================


class TestSimulateOptInWarnPath:
    def test_simulate_warns_when_check_rate_transitions_warn(self):
        from jaxonomy.library import DiscreteClock, UnitDelay

        builder = jaxonomy.DiagramBuilder()
        clk_fast = builder.add(DiscreteClock(dt=0.01, name="clk_fast"))
        ud_slow = builder.add(UnitDelay(dt=0.10, initial_state=0.0, name="ud_slow"))
        builder.connect(clk_fast.output_ports[0], ud_slow.input_ports[0])
        # Wire a sink so the diagram has somewhere for the slow output
        # to flow.  Phase 1 doesn't validate connection direction beyond
        # the existing connect-time checks.
        passthrough = builder.add(_PassThrough(name="pass"))
        builder.connect(ud_slow.output_ports[0], passthrough.input_ports[0])
        diag = builder.build()

        ctx = diag.create_context()
        opts = SimulatorOptions(check_rate_transitions="warn")

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            jaxonomy.simulate(
                diag, ctx, t_span=(0.0, 0.2), options=opts,
                recorded_signals={"y": passthrough.output_ports[0]},
            )

        rate_warnings = [
            w for w in caught if issubclass(w.category, RateMismatchWarning)
        ]
        # At least one mismatch warning fired (clk_fast -> ud_slow).
        assert any(
            "clk_fast" in str(w.message) and "ud_slow" in str(w.message)
            for w in rate_warnings
        ), [str(w.message) for w in rate_warnings]
