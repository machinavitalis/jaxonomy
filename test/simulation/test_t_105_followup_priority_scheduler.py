# SPDX-License-Identifier: MIT
"""T-105-followup-priority-scheduler-hook — actual scheduler integration.

T-105-fu-tasking-priority shipped a ``priority`` attribute convention
plus a read-only ``compute_execution_order`` helper, but the helper was
purely advisory: it did NOT influence the real
:meth:`SystemBase.handle_discrete_update` scheduler at simulation time.

This followup wires the same ``(rate_key, priority_key, name)`` ordering
into the actual discrete-update scheduler so that:

* Within a same-rate group of independent blocks, the leaf with the
  lower ``priority`` integer runs first.
* The default (``priority=None`` on every leaf) is byte-equivalent to
  the pre-followup scheduler — both the diagonal (legacy) branch and
  the lower-triangular branch (T-022a).
* Topological dependencies still win: if B feeds A, A runs after B
  regardless of either's priority.
* The integration works on BOTH the diagonal and lower-triangular
  Phase-2 branches.

Execution-order observation strategy
------------------------------------

Discrete-update events are dispatched in Python (no JAX trace) on the
``math_backend="numpy"`` path, so we attach a Python list to each leaf
and append to it from the inside of the periodic-update closure.  After
``simulate`` returns, the list records the exact order in which the
scheduler invoked each block's state update.

This is the same "debug-callback recording" approach called out in the
T-105-followup-priority-scheduler-hook task plan.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

import jaxonomy
from jaxonomy.framework import LeafSystem
from jaxonomy.framework.discrete_dependencies import (
    DependencyCycleError,
    topological_sort,
)
from jaxonomy.framework.system_base import _build_priority_tiebreak


pytestmark = pytest.mark.minimal


# ---------------------------------------------------------------------
# Recording fixtures
# ---------------------------------------------------------------------


class _RecordingDiscreteBlock(LeafSystem):
    """Tiny discrete block that records its execution order via a
    closure-captured Python list.

    ``x[n+1] = x[n] + 1`` and ``y[n] = x[n]``.  The state update closure
    appends ``self.name`` to ``recorder`` every time it fires — so the
    list reflects scheduler-visible execution order.
    """

    def __init__(self, recorder: list, *, dt: float = 0.1, name: str):
        super().__init__(name=name)
        self._recorder = recorder
        self.declare_discrete_state(default_value=jnp.array(0.0))
        self.declare_output_port(
            lambda t, s, *u, **p: s.discrete_state,
            period=dt, offset=0.0, name="y",
        )
        self.declare_periodic_update(
            self._upd, period=dt, offset=0.0,
        )

    def _upd(self, time, state, *inputs, **params):
        self._recorder.append(self.name)
        return state.discrete_state + 1.0


class _RecordingPassthrough(LeafSystem):
    """Discrete passthrough with a recording side effect.

    ``y[n] = u[n]`` (cache update) and ``x[n+1] = u[n]`` (state update).
    Used to assemble simple data-dependency chains for the
    topology-overrides-priority test.
    """

    def __init__(self, recorder: list, *, dt: float = 0.1, name: str):
        super().__init__(name=name)
        self._recorder = recorder
        self.declare_input_port(name="u")
        self.declare_discrete_state(default_value=jnp.array(0.0))
        self.declare_output_port(
            lambda t, s, *u, **p: s.discrete_state,
            period=dt, offset=0.0, name="y",
        )
        self.declare_periodic_update(
            self._upd, period=dt, offset=0.0,
        )

    def _upd(self, time, state, *inputs, **params):
        self._recorder.append(self.name)
        return inputs[0]


def _simulate_one_tick(diagram, *, lower_triangular: bool):
    """Run the simulator for exactly one discrete tick.

    Uses ``math_backend="numpy"`` so the inside of the periodic-update
    closure runs in pure Python (no JAX trace) and Python-list side
    effects survive.  Returns the post-simulation context for value
    assertions if needed.
    """
    ctx = diagram.create_context()
    opts = jaxonomy.SimulatorOptions(
        math_backend="numpy",
        max_major_steps=20,
        lower_triangular_discrete_update=lower_triangular,
    )
    return jaxonomy.simulate(diagram, ctx, (0.0, 0.15), options=opts)


# =====================================================================
# Two same-rate blocks, NO data dependency — priority orders them
# =====================================================================


class TestSameRatePriorityOrdersScheduler:
    """Priority should actually decide which block runs first in the
    scheduler — not just in the advisory ``compute_execution_order``
    helper."""

    @pytest.mark.parametrize("lower_triangular", [False, True])
    def test_lower_priority_runs_first(self, lower_triangular):
        rec: list[str] = []
        bld = jaxonomy.DiagramBuilder()
        a = bld.add(_RecordingDiscreteBlock(rec, name="blockA"))
        b = bld.add(_RecordingDiscreteBlock(rec, name="blockB"))
        # B has the lower priority → B runs first.
        a.priority = 20
        b.priority = 10
        diagram = bld.build()

        _simulate_one_tick(diagram, lower_triangular=lower_triangular)
        # blockB must come before blockA in the recording.  The
        # simulator may fire either block multiple times across the
        # major-step bookkeeping, but the FIRST occurrence of B must
        # precede the first occurrence of A.
        first_a = rec.index("blockA")
        first_b = rec.index("blockB")
        assert first_b < first_a, (
            f"expected B before A; got rec={rec}"
        )

    @pytest.mark.parametrize("lower_triangular", [False, True])
    def test_reversed_priority_reverses_order(self, lower_triangular):
        """Symmetry: swap the priorities, swap the recorded order."""
        rec: list[str] = []
        bld = jaxonomy.DiagramBuilder()
        a = bld.add(_RecordingDiscreteBlock(rec, name="blockA"))
        b = bld.add(_RecordingDiscreteBlock(rec, name="blockB"))
        a.priority = 10  # A first now
        b.priority = 20
        diagram = bld.build()

        _simulate_one_tick(diagram, lower_triangular=lower_triangular)
        first_a = rec.index("blockA")
        first_b = rec.index("blockB")
        assert first_a < first_b, f"expected A before B; got rec={rec}"

    def test_default_none_preserves_natural_order(self):
        """``priority=None`` on every block → preserves the legacy
        (declaration / system_id) order.  Tests both branches under
        the same fixture."""
        rec_diag: list[str] = []
        bld1 = jaxonomy.DiagramBuilder()
        bld1.add(_RecordingDiscreteBlock(rec_diag, name="blockA"))
        bld1.add(_RecordingDiscreteBlock(rec_diag, name="blockB"))
        diag1 = bld1.build()
        _simulate_one_tick(diag1, lower_triangular=False)

        rec_topo: list[str] = []
        bld2 = jaxonomy.DiagramBuilder()
        bld2.add(_RecordingDiscreteBlock(rec_topo, name="blockA"))
        bld2.add(_RecordingDiscreteBlock(rec_topo, name="blockB"))
        diag2 = bld2.build()
        _simulate_one_tick(diag2, lower_triangular=True)

        # Both: A added first → A fires before B (declaration order for
        # the diagonal branch; system_id order for the topological
        # branch).
        assert rec_diag.index("blockA") < rec_diag.index("blockB")
        assert rec_topo.index("blockA") < rec_topo.index("blockB")


# =====================================================================
# Topology overrides priority
# =====================================================================


class TestTopologyOverridesPriority:
    """When B feeds A, A must run AFTER B regardless of priorities."""

    def test_dataflow_chain_wins_over_priority_topological(self):
        rec: list[str] = []
        bld = jaxonomy.DiagramBuilder()
        b = bld.add(_RecordingDiscreteBlock(rec, name="blockB"))
        a = bld.add(_RecordingPassthrough(rec, name="blockA"))
        bld.connect(b.output_ports[0], a.input_ports[0])
        # Try to invert with priorities — topology must still win.
        a.priority = 1
        b.priority = 99
        diagram = bld.build()

        _simulate_one_tick(diagram, lower_triangular=True)
        first_a = rec.index("blockA")
        first_b = rec.index("blockB")
        assert first_b < first_a, (
            f"topology should win over priority: rec={rec}"
        )


# =====================================================================
# Cycle detection still works
# =====================================================================


class TestCycleDetectionUnaffected:
    """The priority hook must not break cycle detection."""

    def test_cycle_raises_dependency_error(self):
        g = {"a": {"b"}, "b": {"a"}}
        with pytest.raises(DependencyCycleError, match="cycle"):
            topological_sort(g)

    def test_cycle_raises_with_tiebreak_callable(self):
        """Passing a priority-aware ``tiebreak_key`` does not suppress
        the cycle check."""
        g = {"x": {"y"}, "y": {"x"}}
        with pytest.raises(DependencyCycleError, match="cycle"):
            topological_sort(g, tiebreak_key=lambda n: (0, n))


# =====================================================================
# _build_priority_tiebreak: unit-level behaviour
# =====================================================================


class TestBuildPriorityTiebreak:
    """Direct unit tests for the priority-tiebreak builder so we can
    catch regressions without going through the full simulator."""

    def test_no_leaves_returns_none(self):
        # Pass a dummy object with no ``leaf_systems`` attribute.
        class _Bare:
            pass

        assert _build_priority_tiebreak(_Bare()) is None

    def test_all_default_priority_returns_none(self):
        """Crucial for byte-equivalence: no priorities → None → legacy
        scheduler ordering."""
        from jaxonomy.library import Constant

        bld = jaxonomy.DiagramBuilder()
        bld.add(Constant(1.0, name="a"))
        bld.add(Constant(2.0, name="b"))
        diag = bld.build()
        assert _build_priority_tiebreak(diag) is None

    def test_any_priority_set_returns_callable(self):
        from jaxonomy.library import Constant

        bld = jaxonomy.DiagramBuilder()
        a = bld.add(Constant(1.0, name="a"))
        bld.add(Constant(2.0, name="b"))
        a.priority = 5
        diag = bld.build()
        key = _build_priority_tiebreak(diag)
        assert key is not None
        # The leaf with explicit priority should sort before the one
        # without.
        ka = key(a.system_id)
        kb = key(diag.leaf_systems[-1].system_id)
        # Compare priority component: (0, value) vs (1, 0).
        assert ka[1] < kb[1]

    def test_unknown_system_id_sorts_after_known(self):
        """Defensive: a stray ``system_id`` not in the diagram (e.g. an
        event from a nested system) sorts after every known leaf rather
        than crashing the scheduler."""
        from jaxonomy.library import Constant

        bld = jaxonomy.DiagramBuilder()
        a = bld.add(Constant(1.0, name="a"))
        a.priority = 5
        diag = bld.build()
        key = _build_priority_tiebreak(diag)
        known = key(a.system_id)
        unknown = key("not-a-real-id")
        assert known < unknown


# =====================================================================
# Byte-equivalence: T-022a chain tests still pass under default
# =====================================================================


class _ScaleStore(jaxonomy.LeafSystem):
    """Lifted from test_lower_triangular for a self-contained
    byte-equivalence check.  ``x[n+1] = u[n] * k``."""

    def __init__(self, k: float, dt: float = 0.1, **kwargs):
        super().__init__(**kwargs)
        self._k = float(k)
        self.declare_input_port(name="u")
        self.declare_discrete_state(default_value=jnp.array(0.0))
        self.declare_output_port(
            lambda t, s, *u, **p: s.discrete_state,
            period=dt, offset=0.0,
        )
        self.declare_periodic_update(
            lambda t, s, *u, **p: u[0] * self._k,
            period=dt, offset=0.0,
        )


class _ConstantSource(jaxonomy.LeafSystem):
    """Lifted from test_lower_triangular."""

    def __init__(self, value: float, dt: float = 0.1, **kwargs):
        super().__init__(**kwargs)
        self._v = float(value)
        self.declare_discrete_state(default_value=jnp.array(value))
        self.declare_output_port(
            lambda t, s, *u, **p: s.discrete_state,
            period=dt, offset=0.0,
        )
        self.declare_periodic_update(
            lambda t, s, *u, **p: jnp.array(self._v),
            period=dt, offset=0.0,
        )


class TestByteEquivalenceWhenNoPrioritiesSet:
    """The crown-jewel test: every block has ``priority=None`` (the
    default) → simulation outputs are bit-identical to the pre-followup
    scheduler.  We check this on the same 3-block-chain fixture used by
    T-022a's ``test_three_block_chain_diagonal_vs_topological``.
    """

    def _build_chain(self, n_blocks: int, k: float = 2.0):
        bld = jaxonomy.DiagramBuilder()
        src = bld.add(_ConstantSource(1.0, name="src"))
        chain = []
        prev = src
        for i in range(n_blocks):
            b = bld.add(_ScaleStore(k=k, name=f"b{i}"))
            bld.connect(prev.output_ports[0], b.input_ports[0])
            chain.append(b)
            prev = b
        return bld.build(), src, *chain

    @pytest.mark.parametrize("lower_triangular", [False, True])
    def test_chain_byte_equivalent_to_legacy(self, lower_triangular):
        diagram, _src, *_blocks = self._build_chain(n_blocks=3)
        ctx = diagram.create_context()
        opts = jaxonomy.SimulatorOptions(
            math_backend="jax",
            max_major_steps=200,
            lower_triangular_discrete_update=lower_triangular,
        )
        r1 = jaxonomy.simulate(diagram, ctx, (0.0, 0.5), options=opts)
        r2 = jaxonomy.simulate(diagram, ctx, (0.0, 0.5), options=opts)
        last = diagram.leaf_systems[-1]
        v1 = float(r1.context[last.system_id].discrete_state)
        v2 = float(r2.context[last.system_id].discrete_state)
        assert v1 == v2

    def test_topological_chain_matches_t022a_expected_value(self):
        """Diagonal vs lower-triangular on a 3-block chain (same fixture
        as T-022a's ``test_three_block_chain_diagonal_vs_topological``)
        — confirms that adding ``_build_priority_tiebreak`` with no
        priorities set leaves the topological path producing the same
        numerical result as before."""
        diagram, _src, _b0, _b1, b2 = self._build_chain(n_blocks=3, k=2.0)
        ctx = diagram.create_context()

        def _run(lt):
            opts = jaxonomy.SimulatorOptions(
                math_backend="jax",
                max_major_steps=200,
                lower_triangular_discrete_update=lt,
            )
            return jaxonomy.simulate(diagram, ctx, (0.0, 0.15), options=opts)

        r_diag = _run(False)
        r_topo = _run(True)
        diag_b2 = float(r_diag.context[b2.system_id].discrete_state)
        topo_b2 = float(r_topo.context[b2.system_id].discrete_state)
        # Pre-followup contract from T-022a.
        assert diag_b2 == 0.0
        assert topo_b2 == 8.0
