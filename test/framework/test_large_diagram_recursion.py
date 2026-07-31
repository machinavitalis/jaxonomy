# SPDX-License-Identifier: MIT

"""Structurally large diagrams must not overflow the interpreter stack.

Two independent recursions used to scale their *stack depth* with block count,
so a long serial signal path raised ``RecursionError`` rather than a wrong
answer or a slow answer:

1. ``SystemBase.__deepcopy__`` copied ``_dependency_graph``, whose
   ``DependencyTracker.prerequisites`` form a linked chain spanning the whole
   signal path. Any copy of a *warm* system (one that had already built a
   context — which is what the documented sweep idiom does) therefore recursed
   once per block. ``Diagram.with_parameters`` starts with a ``deepcopy``, so
   the sweep / ``vmap`` / optimization idiom broke at ~200 blocks.

2. ``Diagram.check_no_algebraic_loops`` searched for cycles with a recursive
   DFS, burning one frame per feedthrough edge. It runs inside every
   ``create_context``. Because it iterates a ``set``, the traversal order —
   and so the threshold — varied run to run.

The counts here are well past both historical thresholds but stay in the fast
tier (a 1000-block chain builds and copies in well under a second).
"""

import numpy as np
import pytest

import jaxonomy
from jaxonomy.library import Constant, Gain, Integrator

pytestmark = pytest.mark.minimal


def _serial_chain(n_blocks, first_gain=1.0):
    """Constant(1) -> Gain(first_gain) -> Gain(1.0)^(n-1) -> Integrator."""
    builder = jaxonomy.DiagramBuilder()
    src = builder.add(Constant(1.0, name="src"))
    prev = src.output_ports[0]
    for k in range(n_blocks):
        gain = first_gain if k == 0 else 1.0
        g = builder.add(Gain(gain, name=f"g{k}"))
        builder.connect(prev, g.input_ports[0])
        prev = g.output_ports[0]
    integ = builder.add(Integrator(initial_state=0.0, name="out"))
    builder.connect(prev, integ.input_ports[0])
    return builder.build(name="chain"), integ


class TestWarmDiagramCopy:
    """Copying a system that has already built a context must not recurse per block."""

    @pytest.mark.parametrize("n_blocks", [400, 1000])
    def test_with_parameters_on_warm_diagram(self, n_blocks):
        diagram, _ = _serial_chain(n_blocks)
        diagram.create_context()  # warm: builds the dependency graph

        updated = diagram.with_parameters({"g0.gain": 2.0})

        assert updated is not diagram
        assert updated._dependency_graph is None, (
            "the copy's dependency graph must be reset, not copied"
        )

    def test_copy_is_shallow_in_stack_depth(self):
        """deepcopy nesting must be constant in block count, not linear."""
        import copy

        depths = {}
        for n_blocks in (30, 120):
            diagram, _ = _serial_chain(n_blocks)
            diagram.create_context()

            depth = [0]
            max_depth = [0]
            real_deepcopy = copy.deepcopy

            def traced(x, memo=None):
                depth[0] += 1
                max_depth[0] = max(max_depth[0], depth[0])
                try:
                    return real_deepcopy(x, memo)
                finally:
                    depth[0] -= 1

            copy.deepcopy = traced
            try:
                traced(diagram)
            finally:
                copy.deepcopy = real_deepcopy
            depths[n_blocks] = max_depth[0]

        assert depths[30] == depths[120], (
            f"deepcopy nesting scales with block count: {depths}"
        )

    def test_swept_parameter_still_takes_effect(self):
        """Guard the fix against over-reach: the sweep must still be exact."""
        n_blocks = 40
        diagram, _ = _serial_chain(n_blocks)
        diagram.create_context()

        for value in (1.0, 2.0, 3.5):
            swept = diagram.with_parameters({"g0.gain": value})
            integ = next(s for s in swept.nodes if s.name == "out")
            results = jaxonomy.simulate(
                swept,
                swept.create_context(),
                (0.0, 1.0),
                recorded_signals={"y": integ.output_ports[0]},
            )
            # Unit input through one Gain(value) and N-1 unity gains,
            # integrated over 1s.
            assert float(results.outputs["y"][-1]) == pytest.approx(value, rel=1e-6)


class TestAlgebraicLoopSearchDepth:
    """The cycle search is iterative, so a long feedthrough path is fine."""

    @pytest.mark.parametrize("n_blocks", [1000, 2500])
    def test_long_feedthrough_chain_builds_context(self, n_blocks):
        diagram, _ = _serial_chain(n_blocks)
        # create_context runs check_no_algebraic_loops over the whole chain.
        assert diagram.create_context() is not None

    def test_algebraic_loop_still_detected(self):
        """The rewrite must not lose the diagnostic it exists for."""
        from jaxonomy.framework.diagram import AlgebraicLoopError

        builder = jaxonomy.DiagramBuilder()
        a = builder.add(Gain(1.0, name="a"))
        b = builder.add(Gain(1.0, name="b"))
        builder.connect(a.output_ports[0], b.input_ports[0])
        builder.connect(b.output_ports[0], a.input_ports[0])

        with pytest.raises(AlgebraicLoopError):
            builder.build(name="loop").create_context()

    def test_long_chain_with_a_loop_at_the_end(self):
        """A cycle far down a long path is still found (no early bail-out)."""
        from jaxonomy.framework.diagram import AlgebraicLoopError

        builder = jaxonomy.DiagramBuilder()
        src = builder.add(Constant(1.0, name="src"))
        prev = src.output_ports[0]
        for k in range(600):
            g = builder.add(Gain(1.0, name=f"g{k}"))
            builder.connect(prev, g.input_ports[0])
            prev = g.output_ports[0]
        # Close a 2-block algebraic loop hanging off the end of the chain.
        x = builder.add(jaxonomy.library.Adder(2, name="x"))
        y = builder.add(Gain(1.0, name="y"))
        builder.connect(prev, x.input_ports[0])
        builder.connect(y.output_ports[0], x.input_ports[1])
        builder.connect(x.output_ports[0], y.input_ports[0])

        with pytest.raises(AlgebraicLoopError):
            builder.build(name="tail_loop").create_context()
