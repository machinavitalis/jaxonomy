# SPDX-License-Identifier: MIT

"""Tests for T-105-followup-print-schedule-feedback-cycle.

Pre-fix, :meth:`Diagram.print_schedule` rendered any cycle in the
leaf-connection graph as bare ``<cycle detected>``, even when the
cycle was closed through discrete state (a :class:`PIDDiscrete`,
:class:`UnitDelay`, etc.). The simulator runs such loops fine
because the sample-and-hold output breaks the loop at the
fundamental-step boundary, but the alarming ``<cycle detected>``
report trained users to distrust working diagrams.

Post-fix, the rate-summary payload exposes a ``cycle_kind`` slot
distinguishing two flavours:

* ``"feedback-through-discrete"`` — the cycle's edge set includes a
  block with a sample-and-hold output (a periodic event with finite
  period). The text / markdown renderers explain that the simulator
  handles this fine.
* ``"algebraic"`` — no discrete-state break in the loop. The
  renderers warn explicitly that the simulator will raise
  :class:`AlgebraicLoopError`.

Tests:
* Feedback through PIDDiscrete is rendered with the new explanatory
  message in both text and markdown formats.
* Pure algebraic loop (no discrete state on the cycle) is rendered
  with the alarming message.
* Acyclic diagram is unaffected.
* JSON payload exposes ``cycle_kind`` for programmatic consumers.
"""

from __future__ import annotations

import json

import pytest

import jaxonomy
from jaxonomy import library
from jaxonomy.simulation.rate_groups import rate_summary


pytestmark = pytest.mark.minimal


# ---------------------------------------------------------------------
# Diagram builders
# ---------------------------------------------------------------------


def _build_pid_closed_loop():
    """ref → err → PIDDiscrete → Saturate → Gain(plant) → err.

    Closed loop with PIDDiscrete (sample-and-hold output) — a real
    cycle in the leaf-connection graph that the simulator runs fine.
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
    plant = builder.add(library.Gain(gain=0.5, name="plant"))
    builder.connect(ref.output_ports[0], err.input_ports[0])
    builder.connect(plant.output_ports[0], err.input_ports[1])
    builder.connect(err.output_ports[0], pid.input_ports[0])
    builder.connect(pid.output_ports[0], sat.input_ports[0])
    builder.connect(sat.output_ports[0], plant.input_ports[0])
    return builder.build()


def _build_pure_algebraic_loop():
    """Gain → Gain → Gain → … → first Gain (no discrete state).

    A diagram where two Gain blocks feed into each other (via a Sum
    routing) with no discrete block to break the loop. Used as the
    canonical "real algebraic loop" the simulator would reject.
    """
    builder = jaxonomy.DiagramBuilder()
    g1 = builder.add(library.Gain(gain=0.5, name="g1"))
    g2 = builder.add(library.Gain(gain=0.5, name="g2"))
    # Mutual feedback: g1 → g2 → g1.
    builder.connect(g1.output_ports[0], g2.input_ports[0])
    builder.connect(g2.output_ports[0], g1.input_ports[0])
    return builder.build()


def _build_acyclic_baseline():
    """A simple feed-forward diagram for the unaffected-baseline test."""
    builder = jaxonomy.DiagramBuilder()
    src = builder.add(library.Constant(value=1.0, name="src"))
    gain = builder.add(library.Gain(gain=2.0, name="gain"))
    builder.connect(src.output_ports[0], gain.input_ports[0])
    return builder.build()


# ---------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------


class TestFeedbackThroughDiscrete:
    def test_text_renders_feedback_through_discrete_message(self):
        d = _build_pid_closed_loop()
        out = rate_summary(d, format="text")
        assert "feedback cycle through discrete state" in out, (
            f"Expected feedback-through-discrete explanation; got:\n{out}"
        )
        assert "pid" in out
        # The alarming algebraic-cycle phrasing should NOT appear.
        assert "AlgebraicLoopError" not in out

    def test_markdown_renders_feedback_through_discrete_message(self):
        d = _build_pid_closed_loop()
        out = rate_summary(d, format="markdown")
        assert "feedback cycle through discrete state" in out
        assert "pid" in out

    def test_json_payload_exposes_cycle_kind(self):
        d = _build_pid_closed_loop()
        out = rate_summary(d, format="json")
        payload = json.loads(out)
        assert payload["execution_order"] is None
        assert payload["cycle_kind"] == "feedback-through-discrete"
        assert "pid" in payload["cycle_discrete_blocks"]


class TestPureAlgebraicLoop:
    def test_text_renders_algebraic_loop_message(self):
        d = _build_pure_algebraic_loop()
        out = rate_summary(d, format="text")
        assert "algebraic cycle detected" in out, (
            f"Expected algebraic-cycle explanation; got:\n{out}"
        )
        assert "AlgebraicLoopError" in out

    def test_json_payload_marks_as_algebraic(self):
        d = _build_pure_algebraic_loop()
        out = rate_summary(d, format="json")
        payload = json.loads(out)
        assert payload["execution_order"] is None
        assert payload["cycle_kind"] == "algebraic"
        # No discrete blocks on the cycle.
        assert not payload["cycle_discrete_blocks"]


class TestAcyclicBaseline:
    def test_acyclic_diagram_unaffected(self):
        d = _build_acyclic_baseline()
        out = rate_summary(d, format="text")
        # No cycle messaging.
        assert "feedback cycle through discrete state" not in out
        assert "algebraic cycle detected" not in out
        # And the execution order renders normally.
        assert "src -> gain" in out

    def test_acyclic_json_cycle_kind_is_none(self):
        d = _build_acyclic_baseline()
        out = rate_summary(d, format="json")
        payload = json.loads(out)
        assert payload["execution_order"] == ["src", "gain"]
        assert payload["cycle_kind"] is None
        assert payload["cycle_discrete_blocks"] is None
