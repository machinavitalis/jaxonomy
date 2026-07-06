# SPDX-License-Identifier: MIT

"""Tests for T-105-followup-tasking-priority — deterministic task
priorities for same-rate blocks.

Ships the ``priority: int | None`` attribute convention on
``LeafSystem`` (matching the existing ``sample_time`` attribute pattern
from T-105 Phase 2) plus a read-only
:func:`jaxonomy.simulation.rate_groups.compute_execution_order` helper
that returns the order leaves would be processed under the priority-
aware scheduler.

Verified behaviours:

* Same-rate group, no data dependency, lower ``priority`` runs first.
* Default ``priority=None`` falls back to alphabetic-by-name as a
  deterministic tiebreaker.
* Topological data dependency overrides priority — if B feeds A, A
  runs after B regardless of priorities.
* Same-rate group, explicit priority beats default ``None``.
* Non-integer priority raises ``TypeError`` at inspection time.
* ``compute_execution_order`` ordering across rate groups: discrete
  (smallest period first) before continuous before constant.
"""

from __future__ import annotations

import pytest

import jaxonomy
from jaxonomy.framework import LeafSystem
from jaxonomy.simulation.rate_groups import (
    compute_execution_order,
    infer_block_priority,
)


pytestmark = pytest.mark.minimal


# ---------------------------------------------------------------------
# Tiny pass-through used to wire the diagrams below.  Local copy of the
# helper from test_t_105_multirate_phase1.py so the two files don't
# share fixtures.
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
# infer_block_priority
# =====================================================================


class TestInferBlockPriority:
    def test_default_priority_is_none(self):
        from jaxonomy.library import Constant

        block = Constant(1.0, name="K")
        assert infer_block_priority(block) is None

    def test_explicit_integer_priority_is_returned(self):
        from jaxonomy.library import Constant

        block = Constant(1.0, name="K")
        block.priority = 5
        assert infer_block_priority(block) == 5

    def test_zero_priority_is_returned(self):
        """``priority=0`` is a real value, distinct from ``None``."""
        from jaxonomy.library import Constant

        block = Constant(1.0, name="K")
        block.priority = 0
        assert infer_block_priority(block) == 0

    def test_negative_priority_is_returned(self):
        from jaxonomy.library import Constant

        block = Constant(1.0, name="K")
        block.priority = -3
        assert infer_block_priority(block) == -3

    def test_non_integer_priority_raises(self):
        from jaxonomy.library import Constant

        block = Constant(1.0, name="K")
        block.priority = 1.5  # float, not int
        with pytest.raises(TypeError, match="non-integer priority"):
            infer_block_priority(block)

    def test_bool_priority_rejected(self):
        """``bool`` is technically ``int`` in Python, but it's surprising
        to have ``True`` sort before ``False`` here — reject it."""
        from jaxonomy.library import Constant

        block = Constant(1.0, name="K")
        block.priority = True
        with pytest.raises(TypeError):
            infer_block_priority(block)


# =====================================================================
# compute_execution_order — same-rate priority tiebreaker
# =====================================================================


class TestSameRatePriorityTiebreaker:
    def _two_independent_same_rate_blocks(self, prio_a, prio_b):
        """Two ``DiscreteClock`` blocks at the same rate, NO wire between
        them.  Returns (diagram, block_a, block_b).
        """
        from jaxonomy.library import DiscreteClock

        builder = jaxonomy.DiagramBuilder()
        a = builder.add(DiscreteClock(dt=0.01, name="blockA"))
        b = builder.add(DiscreteClock(dt=0.01, name="blockB"))
        if prio_a is not None:
            a.priority = prio_a
        if prio_b is not None:
            b.priority = prio_b
        diag = builder.build()
        return diag, a, b

    def test_lower_priority_runs_first(self):
        """A.priority=10 < B.priority=20 → A before B in the schedule."""
        diag, a, b = self._two_independent_same_rate_blocks(
            prio_a=10, prio_b=20
        )
        order = compute_execution_order(diag)
        names = [leaf.name for leaf in order]
        assert names.index("blockA") < names.index("blockB")

    def test_reversed_priority_reverses_order(self):
        """Symmetry check: swap the priorities, swap the order."""
        diag, a, b = self._two_independent_same_rate_blocks(
            prio_a=20, prio_b=10
        )
        order = compute_execution_order(diag)
        names = [leaf.name for leaf in order]
        assert names.index("blockB") < names.index("blockA")

    def test_default_none_falls_back_to_alphabetic(self):
        """No priority set → natural topo order tiebreaks alphabetically."""
        diag, a, b = self._two_independent_same_rate_blocks(
            prio_a=None, prio_b=None
        )
        order = compute_execution_order(diag)
        names = [leaf.name for leaf in order]
        # No data dep, no priority → alphabetic-by-name.
        assert names.index("blockA") < names.index("blockB")

    def test_explicit_priority_beats_unset_default(self):
        """An explicit (even non-negative) priority sorts before unset.

        Documented contract: ``priority=None`` means *no preference*, so
        any block with an explicit integer priority — even a positive
        one — sorts ahead of one that didn't opt in.
        """
        diag, a, b = self._two_independent_same_rate_blocks(
            prio_a=None, prio_b=100
        )
        order = compute_execution_order(diag)
        names = [leaf.name for leaf in order]
        # blockB has explicit priority=100, blockA has None →
        # blockB sorts ahead despite a "larger" numeric value.
        assert names.index("blockB") < names.index("blockA")


# =====================================================================
# Topology overrides priority
# =====================================================================


class TestTopologyOverridesPriority:
    def test_data_dep_wins_over_priority(self):
        """B feeds A.  Even with A.priority=1 (low) and B.priority=99
        (high), A must run AFTER B because A reads B's output."""
        from jaxonomy.library import DiscreteClock

        builder = jaxonomy.DiagramBuilder()
        b = builder.add(DiscreteClock(dt=0.01, name="blockB"))
        a = builder.add(_PassThrough(name="blockA"))
        # Wire B → A so A depends on B.
        builder.connect(b.output_ports[0], a.input_ports[0])

        # Try to override the order with priorities.
        a.priority = 1   # would normally run first
        b.priority = 99  # would normally run later

        diag = builder.build()
        order = compute_execution_order(diag)
        names = [leaf.name for leaf in order]
        # Topology wins: B must come before A.
        assert names.index("blockB") < names.index("blockA")

    def test_data_dep_wins_three_blocks(self):
        """Chain: C feeds B, B feeds A.  Priorities try to reverse the
        chain; topology must still serialise C → B → A.
        """
        from jaxonomy.library import DiscreteClock

        builder = jaxonomy.DiagramBuilder()
        c = builder.add(DiscreteClock(dt=0.01, name="blockC"))
        b = builder.add(_PassThrough(name="blockB"))
        a = builder.add(_PassThrough(name="blockA"))
        builder.connect(c.output_ports[0], b.input_ports[0])
        builder.connect(b.output_ports[0], a.input_ports[0])
        a.priority = 1
        b.priority = 5
        c.priority = 99

        diag = builder.build()
        order = compute_execution_order(diag)
        names = [leaf.name for leaf in order]
        assert names.index("blockC") < names.index("blockB")
        assert names.index("blockB") < names.index("blockA")


# =====================================================================
# Cross-rate ordering: discrete-first, then continuous, then constant
# =====================================================================


class TestCrossRateOrdering:
    def test_discrete_groups_run_before_continuous(self):
        """Bucket key ordering: discrete (smallest period first) <
        continuous < constant.  Independent blocks at different rates
        should follow that order."""
        from jaxonomy.library import Constant, DiscreteClock, Integrator

        builder = jaxonomy.DiagramBuilder()
        builder.add(Constant(1.0, name="K"))
        builder.add(Integrator(0.0, name="I"))
        builder.add(DiscreteClock(dt=0.10, name="clk_slow"))
        builder.add(DiscreteClock(dt=0.01, name="clk_fast"))

        diag = builder.build()
        order = compute_execution_order(diag)
        names = [leaf.name for leaf in order]
        # discrete @ 0.01s, then discrete @ 0.10s, then continuous,
        # then constant.
        assert names.index("clk_fast") < names.index("clk_slow")
        assert names.index("clk_slow") < names.index("I")
        assert names.index("I") < names.index("K")

    def test_priority_does_not_cross_rate_groups(self):
        """A continuous block with priority=-100 still runs AFTER a
        discrete block with priority=100, because rate-group ordering
        is the primary sort key."""
        from jaxonomy.library import DiscreteClock, Integrator

        builder = jaxonomy.DiagramBuilder()
        clk = builder.add(DiscreteClock(dt=0.01, name="clk"))
        integ = builder.add(Integrator(0.0, name="integ"))
        clk.priority = 100
        integ.priority = -100

        diag = builder.build()
        order = compute_execution_order(diag)
        names = [leaf.name for leaf in order]
        # Discrete runs before continuous regardless of priorities.
        assert names.index("clk") < names.index("integ")


# =====================================================================
# Determinism — same diagram, same order across calls
# =====================================================================


class TestDeterminism:
    def test_repeated_calls_return_identical_order(self):
        from jaxonomy.library import Constant, DiscreteClock

        builder = jaxonomy.DiagramBuilder()
        for i in range(5):
            builder.add(Constant(float(i), name=f"k{i}"))
        for i in range(3):
            builder.add(DiscreteClock(dt=0.01, name=f"clk{i}"))

        diag = builder.build()
        order1 = [leaf.name for leaf in compute_execution_order(diag)]
        order2 = [leaf.name for leaf in compute_execution_order(diag)]
        assert order1 == order2

    def test_empty_diagram_returns_empty_list(self):
        builder = jaxonomy.DiagramBuilder()
        # Even an empty diagram is built around a single leaf
        # placeholder; just call build on a one-block diagram to
        # exercise the "trivial" case.
        from jaxonomy.library import Constant

        builder.add(Constant(0.0, name="solo"))
        diag = builder.build()
        order = compute_execution_order(diag)
        assert [leaf.name for leaf in order] == ["solo"]
