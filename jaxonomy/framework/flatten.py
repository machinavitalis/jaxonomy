# SPDX-License-Identifier: MIT

"""Utility for flattening nested Diagram hierarchies.

Collapses a multi-level Diagram tree into a single-depth Diagram where every
node is a LeafSystem and the connection_map contains only leaf-to-leaf
connections.  Diagram-level exports (inputs/outputs) are preserved.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, List, Tuple

from .diagram import Diagram
from .diagram_builder import DiagramBuilder

if TYPE_CHECKING:
    from .port import InputPortLocator, OutputPortLocator

__all__ = ["flatten_diagram"]


def _resolve_output_locator(
    diagram: Diagram,
    locator: "OutputPortLocator",
) -> "OutputPortLocator":
    """Resolve an output port locator to its leaf-level source.

    If the locator references a sub-Diagram, walk into that sub-Diagram's
    exported output port map to find the actual leaf-level output port that
    produces the value.
    """
    system, port_index = locator
    if not isinstance(system, Diagram):
        # Already a leaf — nothing to resolve
        return locator

    # `system` is a sub-Diagram. Its output port at `port_index` is backed by
    # a subsystem output via _inv_output_port_map.
    sub_diagram: Diagram = system
    inner_locator = sub_diagram._inv_output_port_map[port_index]
    # Recurse in case the inner locator also points to a nested Diagram
    return _resolve_output_locator(sub_diagram, inner_locator)


def _resolve_input_locator(
    diagram: Diagram,
    locator: "InputPortLocator",
) -> List["InputPortLocator"]:
    """Resolve an input port locator to its leaf-level destination(s).

    If the locator references a sub-Diagram exported input, walk into that
    sub-Diagram's connection_map to find all leaf-level input ports that
    receive from that exported input.

    Returns a list because a single exported input could fan-out to multiple
    internal connections.
    """
    system, port_index = locator
    if not isinstance(system, Diagram):
        # Already a leaf — nothing to resolve
        return [locator]

    # `system` is a sub-Diagram. Its input port at `port_index` is backed by
    # one or more internal subsystem inputs via _input_port_map.
    sub_diagram: Diagram = system

    # Find all internal input locators that map to this diagram-level input index
    internal_inputs = [
        internal_loc
        for internal_loc, diag_idx in sub_diagram._input_port_map.items()
        if diag_idx == port_index
    ]

    # Recurse to resolve any nested sub-Diagrams
    result = []
    for inner_loc in internal_inputs:
        result.extend(_resolve_input_locator(sub_diagram, inner_loc))
    return result


def _collect_connections(diagram: Diagram) -> List[Tuple["InputPortLocator", "OutputPortLocator"]]:
    """Collect all leaf-to-leaf connections from the Diagram hierarchy.

    Walks every level of the Diagram tree, resolving connections that pass
    through sub-Diagram exported ports to their actual leaf-level endpoints.
    """
    connections = []

    # Process this level's connection_map
    for input_loc, output_loc in diagram.connection_map.items():
        # Resolve the output side to a leaf
        resolved_output = _resolve_output_locator(diagram, output_loc)

        # Resolve the input side to leaf(s)
        resolved_inputs = _resolve_input_locator(diagram, input_loc)

        for resolved_input in resolved_inputs:
            connections.append((resolved_input, resolved_output))

    # Recurse into sub-Diagrams to pick up their internal connections
    for node in diagram.nodes:
        if isinstance(node, Diagram):
            connections.extend(_collect_connections(node))

    return connections


def _collect_exported_inputs(
    diagram: Diagram,
) -> List[Tuple["InputPortLocator", str]]:
    """Collect diagram-level exported inputs, resolved to leaf ports."""
    exports = []
    for internal_loc, diag_port_idx in diagram._input_port_map.items():
        port_name = diagram.input_ports[diag_port_idx].name
        # Resolve to leaf level
        leaf_inputs = _resolve_input_locator(diagram, internal_loc)
        for leaf_loc in leaf_inputs:
            exports.append((leaf_loc, port_name))
    return exports


def _collect_exported_outputs(
    diagram: Diagram,
) -> List[Tuple["OutputPortLocator", str]]:
    """Collect diagram-level exported outputs, resolved to leaf ports."""
    exports = []
    for internal_loc, diag_port_idx in diagram._output_port_map.items():
        port_name = diagram.output_ports[diag_port_idx].name
        # Resolve to leaf level
        resolved = _resolve_output_locator(diagram, internal_loc)
        exports.append((resolved, port_name))
    return exports


def flatten_diagram(diagram: Diagram) -> Diagram:
    """Flatten a nested Diagram into a single-depth Diagram.

    All intermediate sub-Diagrams are dissolved. The resulting Diagram has:
    - nodes: all LeafSystem instances from the original tree
    - connection_map: remapped to only reference leaf-to-leaf connections
    - exported inputs/outputs: preserved (still reference the same leaf ports)

    Args:
        diagram: The (possibly nested) Diagram to flatten.

    Returns:
        A new single-depth Diagram with all original LeafSystems as direct
        children and all connections resolved to the leaf level.
    """
    # If there are no sub-Diagrams, nothing to flatten
    if not any(isinstance(node, Diagram) for node in diagram.nodes):
        return diagram

    # Gather all leaf systems
    leaf_systems = list(diagram.leaf_systems)

    # Collect all leaf-to-leaf connections
    connections = _collect_connections(diagram)

    # Collect diagram-level port exports
    exported_inputs = _collect_exported_inputs(diagram)
    exported_outputs = _collect_exported_outputs(diagram)

    # Build the new flat diagram
    builder = DiagramBuilder()
    for leaf in leaf_systems:
        # Reset parent so the builder doesn't complain about re-registration
        leaf.parent = None
        builder.add(leaf)

    # Wire up all connections
    for input_loc, output_loc in connections:
        input_sys, input_idx = input_loc
        output_sys, output_idx = output_loc
        builder.connect(
            output_sys.output_ports[output_idx],
            input_sys.input_ports[input_idx],
        )

    # Re-export diagram-level ports
    for input_loc, port_name in exported_inputs:
        input_sys, input_idx = input_loc
        builder.export_input(input_sys.input_ports[input_idx], name=port_name)

    for output_loc, port_name in exported_outputs:
        output_sys, output_idx = output_loc
        builder.export_output(output_sys.output_ports[output_idx], name=port_name)

    return builder.build(name=diagram.name)
