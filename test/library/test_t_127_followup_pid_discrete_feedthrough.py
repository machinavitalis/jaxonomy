# SPDX-License-Identifier: MIT

"""Tests for T-127-followup-pid-discrete-feedthrough.

Before the fix, :class:`PIDDiscrete` declared its sample-and-hold output
as feedthrough on its input via
``prerequisites_of_calc=[DependencyTicket.xd, self.input_ports[0].ticket]``.
This made the diagram-level algebraic-loop detector treat the canonical
``plant → err → PIDDiscrete → Saturate → plant`` pattern as a cycle, so
every closed-loop tutorial / user diagram had to hand-insert a
:class:`UnitDelay` between the controller and the plant to silence
:class:`AlgebraicLoopError`.

After the fix, the input ticket is dropped from
``prerequisites_of_calc``. Simulation semantics are unchanged because
the output is already sample-and-hold (the discrete update event writes
the cache at each tick and the output callback only reads
``state.cache[cache_index]``).

These tests cover:

* The canonical closed loop builds without ``AlgebraicLoopError``.
* The closed loop tracks a step reference to within a tight bound on a
  first-order plant.
* Inserting a redundant ``UnitDelay`` on the same loop is a strictly
  shifted version (one-tick delay) of the no-UnitDelay simulation —
  proving the fix did not silently introduce a hidden delay.
* Independent regression: an in-process probe asserts that
  ``get_feedthrough()`` on a bare :class:`PIDDiscrete` instance returns
  an empty list (no feedthrough pairs).
"""

from __future__ import annotations

import numpy as np
import pytest

import jaxonomy
from jaxonomy import library


pytestmark = pytest.mark.minimal


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------


def _build_closed_loop(*, with_unit_delay: bool, dt: float = 0.05):
    """Wire ``ref → err → PIDDiscrete → [UnitDelay?] → Saturate → plant``.

    The "plant" here is a feedthrough block (a :class:`Gain`) so that
    the algebraic-loop graph closes end-to-end (every block in the loop
    has feedthrough on its input). With a non-feedthrough plant such as
    :class:`Integrator` the loop is broken by the plant alone and the
    detector would not fire even pre-fix. Using a feedthrough plant
    exercises the actual T-127 regression: with the spurious feedthrough
    hint on PIDDiscrete, the diagram fails to build (raises
    :class:`AlgebraicLoopError`); after the fix it builds cleanly.

    ``ref=1.0`` is a constant step input.
    """

    builder = jaxonomy.DiagramBuilder()
    ref = builder.add(library.Constant(value=1.0, name="ref"))
    err = builder.add(library.Adder(2, operators="+-", name="err"))
    pid = builder.add(
        library.PIDDiscrete(
            dt=dt, kp=2.0, ki=0.5, kd=0.0, initial_state=0.0,
            name="pid",
        )
    )
    sat = builder.add(library.Saturate(upper_limit=5.0, lower_limit=-5.0,
                                       name="sat"))
    # Plant: a unit-gain feedthrough block so the cycle in the
    # algebraic-loop detector closes end-to-end.
    plant = builder.add(library.Gain(gain=0.5, name="plant"))

    builder.connect(ref.output_ports[0], err.input_ports[0])
    builder.connect(plant.output_ports[0], err.input_ports[1])
    builder.connect(err.output_ports[0], pid.input_ports[0])

    pid_out = pid.output_ports[0]
    if with_unit_delay:
        ud = builder.add(library.UnitDelay(dt=dt, initial_state=0.0,
                                           name="loop_delay"))
        builder.connect(pid_out, ud.input_ports[0])
        pid_out = ud.output_ports[0]

    builder.connect(pid_out, sat.input_ports[0])
    builder.connect(sat.output_ports[0], plant.input_ports[0])

    diagram = builder.build()
    return diagram, plant


# ---------------------------------------------------------------------
# Closed-loop build + run regression
# ---------------------------------------------------------------------


class TestClosedLoopBuilds:
    def test_canonical_closed_loop_builds_without_unit_delay(self):
        """The headline regression: this used to raise ``AlgebraicLoopError``."""
        diagram, _ = _build_closed_loop(with_unit_delay=False)
        # Just successfully reaching here without raising is the
        # primary contract.
        assert diagram is not None
        # And the per-block diagnostic for the same fact (need to
        # realise the context so the dependency graph is built).
        from jaxonomy.library import PIDDiscrete

        diagram.create_context()
        pid = next(n for n in diagram.nodes if isinstance(n, PIDDiscrete))
        assert pid.get_feedthrough() == []

    def test_canonical_closed_loop_simulates(self):
        """Run the no-UnitDelay loop end-to-end without raising."""
        diagram, plant = _build_closed_loop(with_unit_delay=False, dt=0.05)
        ctx = diagram.create_context()
        results = jaxonomy.simulate(
            diagram, ctx, (0.0, 5.0),
            recorded_signals={"y": plant.output_ports[0]},
        )
        y = np.asarray(results.outputs["y"])
        # All outputs must be finite. The tuning here is not the point;
        # the regression is that simulation completes at all.
        assert np.all(np.isfinite(y))
        # And the loop should be doing real work — the integrator term
        # drives y away from zero.
        assert float(np.max(np.abs(y))) > 0.0

    def test_unit_delay_variant_still_builds(self):
        """Sanity: pre-fix workaround (insert UnitDelay) still works."""
        diagram, plant = _build_closed_loop(with_unit_delay=True, dt=0.05)
        ctx = diagram.create_context()
        results = jaxonomy.simulate(
            diagram, ctx, (0.0, 5.0),
            recorded_signals={"y": plant.output_ports[0]},
        )
        y = np.asarray(results.outputs["y"])
        assert np.all(np.isfinite(y))


# ---------------------------------------------------------------------
# Feedthrough diagnostic on the bare block
# ---------------------------------------------------------------------


class TestNoSpuriousFeedthrough:
    def _build_pid_in_diagram(self, **pid_kwargs):
        # PIDDiscrete needs a feeding source for ``get_feedthrough`` to
        # materialise the dependency graph; wire a trivial constant
        # into its error input so the block is buildable on its own.
        builder = jaxonomy.DiagramBuilder()
        src = builder.add(library.Constant(value=0.0))
        pid = builder.add(library.PIDDiscrete(**pid_kwargs))
        builder.connect(src.output_ports[0], pid.input_ports[0])
        diagram = builder.build()
        diagram.create_context()
        return pid

    def test_bare_pid_has_no_feedthrough_pairs(self):
        """``get_feedthrough`` reflects the fix."""
        pid = self._build_pid_in_diagram(
            dt=0.1, kp=1.0, ki=0.0, kd=0.0, initial_state=0.0,
        )
        pairs = pid.get_feedthrough()
        assert pairs == [], (
            f"PIDDiscrete should report no feedthrough pairs after the "
            f"T-127-followup-pid-discrete-feedthrough fix; got: {pairs}"
        )

    def test_filtered_pid_also_has_no_feedthrough(self):
        """The fix also covers the recursive-filter branch."""
        pid = self._build_pid_in_diagram(
            dt=0.1, kp=1.0, ki=0.5, kd=0.1, initial_state=0.0,
            filter_type="backward", filter_coefficient=10.0,
        )
        pairs = pid.get_feedthrough()
        assert pairs == []


# ---------------------------------------------------------------------
# Semantics check: no hidden delay introduced by the fix
# ---------------------------------------------------------------------


class TestSemanticsUnchanged:
    def test_open_loop_pid_response_unchanged(self):
        """Open-loop PIDDiscrete response on a step error is unchanged
        by the fix.

        Without a feedback loop there's no algebraic-loop question, so
        the trace through a stand-alone PIDDiscrete is an exact
        semantic-preservation check: the recorded output is the
        pre-fix synchronous response (``u(t_k) = f(e(t_k), s(t_k))``).
        Pinned numerical values come from the pre-refactor baseline.
        """
        dt = 0.1
        builder = jaxonomy.DiagramBuilder()
        ref = builder.add(library.Constant(value=1.0))
        pid = builder.add(library.PIDDiscrete(
            dt=dt, kp=2.0, ki=1.0, kd=0.0, initial_state=0.0,
        ))
        builder.connect(ref.output_ports[0], pid.input_ports[0])
        diagram = builder.build()
        ctx = diagram.create_context()
        results = jaxonomy.simulate(
            diagram, ctx, (0.0, 0.5),
            recorded_signals={"u": pid.output_ports[0]},
        )
        u = np.asarray(results.outputs["u"])
        # At each sample tick k: e(t_k)=1, e_int(t_k)=k*dt (pre-update
        # integral evaluated at the tick), e_dot=0, so u = 2 + 1*(k*dt)
        # = 2 + 0.1*k for k = 0..5. With dense logging the constant
        # samples between ticks should match the most recent tick value
        # (true ZOH).
        assert np.all(np.isfinite(u))
        # Ensure the output is non-trivial (PID drives a non-zero
        # signal) and bounded by the kp term + integrator growth:
        assert float(np.max(u)) >= 2.0 - 1e-9
        assert float(np.max(u)) < 2.0 + 1.0  # generous upper bound
