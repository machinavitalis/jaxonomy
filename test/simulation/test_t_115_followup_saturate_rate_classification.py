# SPDX-License-Identifier: MIT

"""Tests for T-115-followup-saturate-rate-classification.

Before the fix, :class:`Saturate` (and any block that declares pure
solver-hint zero-crossing events with no ``reset_map`` and no mode
transition) was classified by :func:`infer_block_sample_time` as
``event_driven``. The clip-boundary ZC events exist only so the ODE
solver can localise the discontinuity — they have no behavioural effect
on the block itself. Treating them as event-driven flipped the canonical
``PID(discrete) → Saturate → plant(continuous)`` pattern from a clean
single-rate flow into a noisy "discrete → event_driven" rate-mismatch
warning, with a misleading "insert a Sample-and-Hold or rebuild the
downstream block to be event-driven too" hint.

After the fix, :meth:`LeafSystem.declare_zero_crossing` keeps a
``_n_behavioral_zc_events`` counter that only increments for events with
a user-supplied ``reset_map`` or a ``start_mode`` / ``end_mode``
transition. The rate-groups classifier consults the new counter so that
solver-hint-only events (Saturate, DeadZone, IfThenElse, Comparator's
on/off auto-detect path) no longer flip the block to ``event_driven``.

These tests cover:

* The behavioural-ZC counter increments correctly per event flavour
  (pure guard / guard + reset_map / mode transition).
* :class:`Saturate` (after ``initialize_static_data``) is *not*
  classified as ``event_driven``.
* The canonical ``PID → Saturate → plant`` pattern triggers no
  ``RateMismatchWarning``.
* A block that legitimately has a reset_map (``ZeroCrossingTriggeredSubsystem``
  through container path / a StateMachine transition) is still
  classified as ``event_driven`` — the fix is targeted, not blanket.
"""

from __future__ import annotations

import warnings

import pytest

import jaxonomy
from jaxonomy import library
from jaxonomy.framework.leaf_system import LeafSystem
from jaxonomy.simulation.rate_groups import (
    RateMismatchWarning,
    detect_rate_mismatches,
    infer_block_sample_time,
)


pytestmark = pytest.mark.minimal


# ---------------------------------------------------------------------
# Counter-level unit tests
# ---------------------------------------------------------------------


class _GuardOnlyBlock(LeafSystem):
    """Test fixture: declares a single guard-only ZC event."""

    def __init__(self):
        super().__init__(name="guard_only")
        self.declare_input_port()

        def _guard(_t, _s, *inputs, **_p):
            return inputs[0]

        self.declare_zero_crossing(_guard, direction="crosses_zero")

        def _out(_t, _s, *inputs, **_p):
            return inputs[0]

        self.declare_output_port(_out)


class _GuardAndResetBlock(LeafSystem):
    """Test fixture: ZC event with a user-supplied reset_map."""

    def __init__(self):
        super().__init__(name="guard_and_reset")
        self.declare_input_port()
        self.declare_continuous_state(default_value=0.0, ode=self._ode)

        def _guard(_t, _s, *inputs, **_p):
            return inputs[0]

        def _reset(_t, state, *_inputs, **_p):
            return state.with_continuous_state(0.0)

        self.declare_zero_crossing(
            _guard, reset_map=_reset, direction="crosses_zero",
        )

        def _out(_t, state, *_inputs, **_p):
            return state.continuous_state

        self.declare_output_port(_out)

    def _ode(self, _t, state, *_inputs, **_p):
        return 1.0  # trivial integrator


class TestBehavioralCounter:
    def test_pure_guard_event_does_not_increment(self):
        block = _GuardOnlyBlock()
        assert block._n_behavioral_zc_events == 0
        assert len(block._zero_crossing_events) == 1

    def test_event_with_reset_map_increments(self):
        block = _GuardAndResetBlock()
        assert block._n_behavioral_zc_events == 1
        assert len(block._zero_crossing_events) == 1

    def test_mixed_events_count_correctly(self):
        block = _GuardOnlyBlock()
        # Hand-declare an additional reset-bearing event on the same
        # block to ensure mixed declarations are counted independently.
        def _g(_t, _s, *_i, **_p):
            return 0.0

        def _r(_t, state, *_i, **_p):
            return state

        block.declare_zero_crossing(_g, reset_map=_r, direction="crosses_zero")
        assert len(block._zero_crossing_events) == 2
        assert block._n_behavioral_zc_events == 1


# ---------------------------------------------------------------------
# Saturate-specific classification
# ---------------------------------------------------------------------


def _build_continuous_saturate():
    """Build a diagram with Saturate sitting on a *continuous* path.

    :func:`is_discontinuity` only fires Saturate's clip-boundary ZC
    declarations when the output is downstream of continuous state and
    feeds an ODE — otherwise ``initialize_static_data`` is a no-op.
    Wiring Saturate between two continuous-state blocks
    (``plant1 → sat → plant2``) guarantees the events get declared so
    the rate-classification path is exercised.
    """
    builder = jaxonomy.DiagramBuilder()
    plant1 = builder.add(library.Integrator(initial_state=0.5, name="plant1"))
    sat = builder.add(library.Saturate(
        upper_limit=1.0, lower_limit=-1.0, name="sat",
    ))
    plant2 = builder.add(library.Integrator(initial_state=0.0, name="plant2"))
    drive = builder.add(library.Constant(value=1.0, name="drive"))

    # plant1 is integrating ``drive`` (= 1.0), so its output ramps from
    # 0.5 toward 1.0 and Saturate's clip boundary actually trips.
    builder.connect(drive.output_ports[0], plant1.input_ports[0])
    builder.connect(plant1.output_ports[0], sat.input_ports[0])
    builder.connect(sat.output_ports[0], plant2.input_ports[0])

    diagram = builder.build()
    return diagram, dict(sat=sat, plant1=plant1, plant2=plant2)


def _build_pid_saturate_plant():
    """Canonical ``ref → err → PID → Saturate → plant`` loop.

    Plant is a continuous Integrator so the algebraic loop breaks at
    the plant (orthogonal to T-127-followup). This is the canonical
    pattern the follow-up finding calls out — historically it emitted
    a "discrete → event_driven" rate-mismatch warning because Saturate
    was classified as event-driven by its clip-boundary ZC events.
    """
    dt = 0.05
    builder = jaxonomy.DiagramBuilder()
    ref = builder.add(library.Constant(value=1.0, name="ref"))
    err = builder.add(library.Adder(2, operators="+-", name="err"))
    pid = builder.add(library.PIDDiscrete(
        dt=dt, kp=1.0, ki=0.5, kd=0.0, initial_state=0.0, name="pid",
    ))
    sat = builder.add(library.Saturate(
        upper_limit=1.0, lower_limit=-1.0, name="sat",
    ))
    plant = builder.add(library.Integrator(initial_state=0.0, name="plant"))

    builder.connect(ref.output_ports[0], err.input_ports[0])
    builder.connect(plant.output_ports[0], err.input_ports[1])
    builder.connect(err.output_ports[0], pid.input_ports[0])
    builder.connect(pid.output_ports[0], sat.input_ports[0])
    builder.connect(sat.output_ports[0], plant.input_ports[0])

    diagram = builder.build()
    return diagram, dict(pid=pid, sat=sat, plant=plant)


class TestSaturateClassification:
    def test_saturate_not_event_driven_after_zc_registration(self):
        """``initialize_static_data`` registers ZC events on Saturate
        only when its output sits on a continuous path that feeds an
        ODE (``is_discontinuity`` gate). The continuous
        ``plant1 → Saturate → plant2`` topology trips that gate; the
        new counter then ensures the rate classifier still infers a
        non-event-driven sample time."""
        diagram, parts = _build_continuous_saturate()
        ctx = diagram.create_context()
        # Trigger initialize_static_data → Saturate declares its ZC
        # boundary events.
        jaxonomy.simulate(diagram, ctx, (0.0, 0.1))

        sat = parts["sat"]
        assert len(sat._zero_crossing_events) == 2, (
            "Saturate should have declared its 2 clip-boundary ZC events "
            "during initialize_static_data."
        )
        assert sat._n_behavioral_zc_events == 0, (
            "Saturate's clip-boundary events are solver hints — no "
            "reset_map, no mode transition, so they must not count as "
            "behavioural."
        )

        sat_st = infer_block_sample_time(sat)
        assert not sat_st.is_event_driven(), (
            f"Saturate should NOT be classified as event_driven after "
            f"the T-115-followup-saturate-rate-classification fix; got "
            f"{sat_st!r}."
        )

    def test_no_rate_mismatch_in_pid_saturate_plant(self):
        """No ``RateMismatchWarning`` from the canonical PID loop.

        Note: in this specific topology Saturate's ZC events are *not*
        registered at all (PIDDiscrete's output is not continuous, so
        ``is_discontinuity`` short-circuits ``initialize_static_data``),
        so the test passes both pre- and post-fix. It's retained as a
        guard against future regressions that would loosen the
        ``is_discontinuity`` gate without re-checking this path.
        """
        diagram, _ = _build_pid_saturate_plant()
        ctx = diagram.create_context()
        jaxonomy.simulate(diagram, ctx, (0.0, 0.1))

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            mismatches = detect_rate_mismatches(diagram, on_mismatch="warn")
        rate_warns = [w for w in caught if issubclass(w.category, RateMismatchWarning)]
        assert mismatches == [], (
            f"detect_rate_mismatches should return no entries for the "
            f"canonical PID → Saturate → plant pattern; got: {mismatches!r}"
        )
        assert rate_warns == [], (
            f"No RateMismatchWarning expected; got {len(rate_warns)} "
            f"warnings: {[str(w.message) for w in rate_warns]}"
        )

    def test_continuous_saturate_does_not_classify_as_event_driven(self):
        """The direct repro: continuous-upstream Saturate, post-fix.

        Pre-fix, ``infer_block_sample_time(sat)`` returned ``event_driven``
        (because Saturate's clip-boundary ZC events were registered).
        Post-fix, it returns ``constant`` (universal-match), which is
        the user-visible improvement: ``print_schedule`` no longer
        labels memoryless clipping blocks as event-driven, and any
        downstream discrete consumer no longer triggers an
        event-driven ↔ discrete mismatch warning.
        """
        diagram, parts = _build_continuous_saturate()
        ctx = diagram.create_context()
        jaxonomy.simulate(diagram, ctx, (0.0, 0.1))

        sat = parts["sat"]
        assert len(sat._zero_crossing_events) == 2, (
            "Sanity: continuous → Saturate → continuous DOES register "
            "the clip-boundary ZC events."
        )
        st = infer_block_sample_time(sat)
        assert st.kind == "constant", (
            f"Saturate with only solver-hint ZC events should classify "
            f"as universal (``constant``); got {st!r}."
        )


# ---------------------------------------------------------------------
# Behavioural-ZC blocks are still event-driven
# ---------------------------------------------------------------------


class TestBehavioralBlocksUnchanged:
    def test_block_with_reset_map_still_event_driven(self):
        """A block with a user-supplied reset_map keeps its
        ``event_driven`` classification — the fix is targeted at the
        pure-guard case only."""
        block = _GuardAndResetBlock()
        st = infer_block_sample_time(block)
        # The block has continuous state alongside an event with reset,
        # so the classifier should pick the event-driven kind (step 4
        # wins over step 5 — continuous-state blocks become event-driven
        # when they also declare a behavioural ZC).
        assert st.is_event_driven(), (
            f"A guard+reset block must still classify as event_driven; "
            f"got {st!r}."
        )

    def test_relay_block_still_event_driven(self):
        """``library.Relay`` declares mode-transition ZC events;
        those are behavioural and the block must still be
        event-driven."""
        relay = library.Relay(
            on_threshold=1.0, off_threshold=0.0,
            on_value=1.0, off_value=0.0,
            initial_state=0.0,
        )
        # Wire to a constant so create_context can build.
        builder = jaxonomy.DiagramBuilder()
        builder.add(relay)
        src = builder.add(library.Constant(value=0.5))
        builder.connect(src.output_ports[0], relay.input_ports[0])
        diagram = builder.build()
        diagram.create_context()

        assert relay._n_behavioral_zc_events == 2, (
            "Relay declares 2 mode-transition ZC events (on/off)."
        )
        st = infer_block_sample_time(relay)
        assert st.is_event_driven(), (
            f"Relay with mode-transition ZC must classify as "
            f"event_driven; got {st!r}."
        )
