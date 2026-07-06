# SPDX-License-Identifier: MIT

"""Shared helpers for MCP tools (port resolution, diagram introspection)."""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp

from jaxonomy.framework import Diagram


def collect_blocks(diagram: Diagram, prefix: str = "") -> dict[str, Any]:
    """Map dotted block paths to subsystem instances."""
    out: dict[str, Any] = {}
    for node in diagram.nodes:
        key = f"{prefix}.{node.name}" if prefix else node.name
        out[key] = node
        if isinstance(node, Diagram):
            out.update(collect_blocks(node, key))
    return out


def resolve_output_port(diagram: Diagram, spec: str):
    """Resolve ``block.out_0`` style path to an :class:`~jaxonomy.framework.port.OutputPort`."""
    if "." not in spec:
        raise ValueError(
            f"recorded signal {spec!r} must look like 'block_name.port_name', e.g. 'integ.out_0'"
        )
    block_name, port_name = spec.split(".", 1)
    try:
        system = diagram[block_name]
    except Exception as e:
        raise KeyError(
            f"Block {block_name!r} not found on diagram {diagram.name!r}: {e}"
        ) from e
    for p in system.output_ports:
        if p.name == port_name:
            return p
    names = [p.name for p in system.output_ports]
    raise KeyError(
        f"No output port {port_name!r} on block {block_name!r}; available: {names}"
    )


def resolve_input_port(diagram: Diagram, spec: str):
    """Resolve ``block.in_0`` style path to an :class:`~jaxonomy.framework.port.InputPort`."""
    if "." not in spec:
        raise ValueError(f"input spec {spec!r} must be 'block_name.port_name'")
    block_name, port_name = spec.split(".", 1)
    system = diagram[block_name]
    for p in system.input_ports:
        if p.name == port_name:
            return p
    names = [p.name for p in system.input_ports]
    raise KeyError(
        f"No input port {port_name!r} on block {block_name!r}; available: {names}"
    )


def apply_input_values(diagram: Diagram, input_values: dict[str, float]) -> None:
    """Fix input ports to constants (mutates ports in place)."""
    for spec, val in input_values.items():
        port = resolve_input_port(diagram, spec)
        port.fix_value(jnp.asarray(val, dtype=jnp.float64))


def parse_csv_string(data_csv: str) -> tuple[list[str], jnp.ndarray]:
    """Parse CSV text; first row is header. Returns (column_names, data array (n_rows, n_cols))."""
    lines = [ln.strip() for ln in data_csv.strip().splitlines() if ln.strip()]
    if len(lines) < 2:
        raise ValueError("data_csv must have a header row and at least one data row")
    header = [h.strip() for h in lines[0].split(",")]
    rows = []
    for ln in lines[1:]:
        rows.append([float(x.strip()) for x in ln.split(",")])

    return header, jnp.array(rows, dtype=jnp.float64)
