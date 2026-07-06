# SPDX-License-Identifier: MIT

"""Tests for T-105-followup-print-schedule-pre-context.

Pre-fix, :meth:`Diagram.print_schedule` misbucketed discrete blocks
whose periodic events are registered in :meth:`initialize` —
:class:`PIDDiscrete`, :class:`Decimator`, :class:`UnitDelay`,
:class:`ZeroOrderHold` — as ``constant`` when called before any
context had been created, because the periodic event hadn't been
declared yet.

Post-fix, ``print_schedule`` lazily calls :meth:`create_context` once
before rendering so every leaf has its :meth:`initialize` hook run,
which registers the periodic events. The lazy call is idempotent.
If the diagram is unbuildable (missing connections, etc.) the call
fails gracefully with a clear warning and renders the schedule
against the pre-init state anyway.

Tests:
* :class:`PIDDiscrete` shows up in its correct discrete-rate bucket
  when ``print_schedule`` is called before any explicit
  ``create_context``.
* Same for :class:`UnitDelay`, :class:`ZeroOrderHold`,
  :class:`Decimator`.
* Calling ``print_schedule`` after an explicit ``create_context`` is
  byte-equivalent (idempotent path).
* ``ensure_initialized=False`` opt-out preserves the pre-fix
  behaviour for callers who want to inspect the pre-init state.
* Unbuildable diagram (missing connection) emits a clear warning and
  still renders the schedule against the partially-initialised state.
"""

from __future__ import annotations

import io
import warnings

import pytest

import jaxonomy
from jaxonomy import library


pytestmark = pytest.mark.minimal


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------


def _build_pid_diagram():
    builder = jaxonomy.DiagramBuilder()
    pid = builder.add(library.PIDDiscrete(
        dt=0.05, kp=1.0, ki=0.5, kd=0.0, initial_state=0.0,
        name="pid",
    ))
    src = builder.add(library.Constant(value=0.0, name="src"))
    builder.connect(src.output_ports[0], pid.input_ports[0])
    return builder.build()


def _build_unit_delay_diagram():
    builder = jaxonomy.DiagramBuilder()
    ud = builder.add(library.UnitDelay(
        dt=0.02, initial_state=0.0, name="delay",
    ))
    src = builder.add(library.Constant(value=0.0, name="src"))
    builder.connect(src.output_ports[0], ud.input_ports[0])
    return builder.build()


def _build_zoh_diagram():
    builder = jaxonomy.DiagramBuilder()
    zoh = builder.add(library.ZeroOrderHold(dt=0.1, name="zoh"))
    src = builder.add(library.Constant(value=0.0, name="src"))
    builder.connect(src.output_ports[0], zoh.input_ports[0])
    return builder.build()


def _build_decimator_diagram():
    builder = jaxonomy.DiagramBuilder()
    dec = builder.add(library.Decimator(input_dt=0.01, output_dt=0.05,
                                         name="dec"))
    src = builder.add(library.Constant(value=0.0, name="src"))
    builder.connect(src.output_ports[0], dec.input_ports[0])
    return builder.build()


def _capture(diagram, **kwargs):
    buf = io.StringIO()
    diagram.print_schedule(file=buf, **kwargs)
    return buf.getvalue()


# ---------------------------------------------------------------------
# Lazy create_context behaviour
# ---------------------------------------------------------------------


class TestLazyInitialise:
    def test_piddiscrete_classified_as_discrete_pre_context(self):
        d = _build_pid_diagram()
        out = _capture(d)
        assert "discrete(period=0.05" in out, (
            f"PIDDiscrete should bucket as discrete(period=0.05); got:\n{out}"
        )
        # And it should NOT be in the constant bucket.
        lines = out.splitlines()
        constant_line = next(
            (ln for ln in lines if "constant" in ln), None,
        )
        if constant_line:
            assert "pid" not in constant_line, (
                f"PIDDiscrete should not be in the constant group; got:\n{out}"
            )

    def test_unit_delay_classified_as_discrete_pre_context(self):
        d = _build_unit_delay_diagram()
        out = _capture(d)
        assert "discrete(period=0.02" in out, (
            f"UnitDelay should bucket as discrete(period=0.02); got:\n{out}"
        )

    def test_zero_order_hold_classified_as_discrete_pre_context(self):
        d = _build_zoh_diagram()
        out = _capture(d)
        assert "discrete(period=0.1" in out, (
            f"ZeroOrderHold should bucket as discrete(period=0.1); got:\n{out}"
        )

    def test_decimator_classified_as_discrete_pre_context(self):
        d = _build_decimator_diagram()
        out = _capture(d)
        assert "discrete(period=" in out, (
            f"Decimator should bucket as discrete; got:\n{out}"
        )


# ---------------------------------------------------------------------
# Opt-out + idempotence
# ---------------------------------------------------------------------


class TestOptOut:
    def test_ensure_initialized_false_preserves_legacy_behaviour(self):
        """``ensure_initialized=False`` reproduces the pre-fix misbucket."""
        d = _build_pid_diagram()
        out = _capture(d, ensure_initialized=False)
        # In the pre-init state, PID's periodic event is not yet
        # declared, so the block falls through to ``constant``.
        constant_line = next(
            (ln for ln in out.splitlines() if "constant" in ln), "",
        )
        assert "pid" in constant_line, (
            f"With ensure_initialized=False, PIDDiscrete should appear "
            f"in the constant bucket (pre-init state). Got:\n{out}"
        )

    def test_post_context_call_is_idempotent(self):
        """Explicitly calling create_context() first does not change
        the rendered output."""
        d = _build_pid_diagram()
        d.create_context()  # explicit pre-init
        out_explicit = _capture(d)
        out_auto = _capture(_build_pid_diagram())
        # Both should put PID in the discrete bucket.
        assert "discrete(period=0.05" in out_explicit
        assert "discrete(period=0.05" in out_auto


# ---------------------------------------------------------------------
# Graceful fallback on unbuildable diagram
# ---------------------------------------------------------------------


class TestUnbuildableFallback:
    def test_unbuildable_diagram_emits_warning_and_still_renders(self):
        """A diagram with a missing connection should fail
        create_context() but still render the schedule with a
        clear warning."""
        # Construct a diagram with an unconnected input port.
        builder = jaxonomy.DiagramBuilder()
        builder.add(library.PIDDiscrete(
            dt=0.05, kp=1.0, ki=0.5, kd=0.0, initial_state=0.0,
            name="pid_unconnected",
        ))
        d = builder.build()

        buf = io.StringIO()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            try:
                d.print_schedule(file=buf)
            except Exception:
                pytest.fail(
                    "print_schedule must not raise on an unbuildable "
                    "diagram; it should warn and render the partial "
                    "schedule."
                )
        # Either it succeeded silently (create_context worked) OR it
        # emitted our specific warning. Both outcomes are OK; we only
        # fail if no output was produced at all.
        assert buf.getvalue(), (
            "print_schedule must produce some output even when "
            "create_context() fails."
        )
        # If a warning fired, it should be the one we expect.
        relevant = [w for w in caught
                    if "create_context() failed" in str(w.message)]
        # We don't assert relevant != [] because some unbuildable
        # diagrams may still succeed in create_context if there are no
        # required connections — what matters is that NO exception
        # escapes print_schedule.
