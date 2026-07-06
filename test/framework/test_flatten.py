# SPDX-License-Identifier: MIT

"""Tests for flatten_diagram utility."""

import jax.numpy as jnp
import numpy as np
import pytest

import jaxonomy
from jaxonomy import DiagramBuilder, LeafSystem, simulate
from jaxonomy.framework import flatten_diagram
from jaxonomy.framework.diagram import Diagram
from jaxonomy.library import Gain, Constant


class Adder(LeafSystem):
    """Simple 2-input adder for testing."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.declare_input_port()
        self.declare_input_port()
        self.declare_output_port(self._output)

    def _output(self, _time, _state, *inputs, **_params):
        return inputs[0] + inputs[1]


def _build_nested_diagram():
    """Build a nested diagram: root contains a sub-Diagram and a Gain.

    Structure:
        root
        ├── sub_diagram
        │   ├── const_1 (value=2.0)
        │   └── const_2 (value=3.0)
        │   └── adder (const_1 + const_2 = 5.0)
        │   [export: adder.out_0]
        └── gain (k=10.0, input from sub_diagram.out)
        [export: gain.out_0]

    Expected output: (2.0 + 3.0) * 10.0 = 50.0
    """
    # Build sub-diagram
    sub_builder = DiagramBuilder()
    const_1 = sub_builder.add(Constant(value=2.0, name="const_1"))
    const_2 = sub_builder.add(Constant(value=3.0, name="const_2"))
    adder = sub_builder.add(Adder(name="adder"))
    sub_builder.connect(const_1.output_ports[0], adder.input_ports[0])
    sub_builder.connect(const_2.output_ports[0], adder.input_ports[1])
    sub_builder.export_output(adder.output_ports[0])
    sub_diagram = sub_builder.build(name="sub_diagram")

    # Build root diagram
    root_builder = DiagramBuilder()
    root_builder.add(sub_diagram)
    gain = root_builder.add(Gain(gain=10.0, name="gain"))
    root_builder.connect(sub_diagram.output_ports[0], gain.input_ports[0])
    root_builder.export_output(gain.output_ports[0])
    root_diagram = root_builder.build(name="root")

    return root_diagram


def test_flatten_structure():
    """Verify that flatten_diagram produces a single-depth Diagram."""
    nested = _build_nested_diagram()

    # Nested diagram has a sub-Diagram as a node
    assert any(isinstance(node, Diagram) for node in nested.nodes)

    flat = flatten_diagram(nested)

    # All nodes should be leaf systems (not Diagrams)
    assert not any(isinstance(node, Diagram) for node in flat.nodes)

    # Should have 4 leaf systems: const_1, const_2, adder, gain
    assert len(flat.nodes) == 4

    # Should have 1 exported output (gain's output)
    assert flat.num_output_ports == 1


def test_flatten_noop_for_already_flat():
    """Verify that flattening a single-depth Diagram returns it unchanged."""
    builder = DiagramBuilder()
    const = builder.add(Constant(value=1.0, name="const"))
    gain = builder.add(Gain(gain=2.0, name="gain"))
    builder.connect(const.output_ports[0], gain.input_ports[0])
    builder.export_output(gain.output_ports[0])
    diagram = builder.build(name="flat")

    flat = flatten_diagram(diagram)

    # Should return the same object since it's already flat
    assert flat is diagram


def test_flatten_simulation_equivalence():
    """Verify that flattened and nested diagrams produce the same result."""
    nested = _build_nested_diagram()

    # Simulate the nested diagram
    ctx_nested = nested.create_context()
    results_nested = simulate(
        nested, ctx_nested, (0.0, 0.1),
        recorded_signals={"out": nested.output_ports[0]},
    )

    # Flatten and simulate
    flat = flatten_diagram(nested)
    ctx_flat = flat.create_context()
    results_flat = simulate(
        flat, ctx_flat, (0.0, 0.1),
        recorded_signals={"out": flat.output_ports[0]},
    )

    # Both should produce 50.0 at the output
    np.testing.assert_allclose(
        results_nested.outputs["out"][-1],
        results_flat.outputs["out"][-1],
        rtol=1e-6,
    )


def test_flatten_with_exported_inputs():
    """Verify flattening works when the sub-Diagram has exported inputs."""
    # Build sub-diagram with an exported input
    sub_builder = DiagramBuilder()
    gain_inner = sub_builder.add(Gain(gain=2.0, name="inner_gain"))
    sub_builder.export_input(gain_inner.input_ports[0])
    sub_builder.export_output(gain_inner.output_ports[0])
    sub_diagram = sub_builder.build(name="sub")

    # Build root: Constant -> sub_diagram -> outer_gain
    root_builder = DiagramBuilder()
    const = root_builder.add(Constant(value=5.0, name="const"))
    root_builder.add(sub_diagram)
    outer_gain = root_builder.add(Gain(gain=3.0, name="outer_gain"))
    root_builder.connect(const.output_ports[0], sub_diagram.input_ports[0])
    root_builder.connect(sub_diagram.output_ports[0], outer_gain.input_ports[0])
    root_builder.export_output(outer_gain.output_ports[0])
    root_diagram = root_builder.build(name="root")

    # Flatten
    flat = flatten_diagram(root_diagram)

    # Should have 3 leaf systems: const, inner_gain, outer_gain
    assert len(flat.nodes) == 3
    assert not any(isinstance(node, Diagram) for node in flat.nodes)

    # Simulate both and compare: expected = 5.0 * 2.0 * 3.0 = 30.0
    ctx_nested = root_diagram.create_context()
    results_nested = simulate(
        root_diagram, ctx_nested, (0.0, 0.1),
        recorded_signals={"out": root_diagram.output_ports[0]},
    )

    ctx_flat = flat.create_context()
    results_flat = simulate(
        flat, ctx_flat, (0.0, 0.1),
        recorded_signals={"out": flat.output_ports[0]},
    )

    np.testing.assert_allclose(
        results_nested.outputs["out"][-1],
        results_flat.outputs["out"][-1],
        rtol=1e-6,
    )
