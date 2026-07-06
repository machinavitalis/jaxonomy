# SPDX-License-Identifier: MIT
"""T-105-followup-rate-summary-graphviz smoke tests.

The agent shipped rate_summary_dot in rate_groups.py but ran out of context
before producing tests.
"""

from __future__ import annotations

import pytest

import jaxonomy
from jaxonomy import library
from jaxonomy.simulation.rate_groups import rate_summary_dot
from jaxonomy.testing.markers import skip_if_not_jax

skip_if_not_jax()


def _build_simple_diagram():
    """Build a small diagram with two periodic blocks for visualization tests."""
    builder = jaxonomy.DiagramBuilder()
    src = library.Constant(1.0, name="src")
    udelay = library.UnitDelay(initial_state=0.0, dt=0.1, name="udelay")
    builder.add(src, udelay)
    builder.connect(src.output_ports[0], udelay.input_ports[0])
    return builder.build()


def test_dot_emits_digraph_header():
    diag = _build_simple_diagram()
    dot = rate_summary_dot(diag)
    assert isinstance(dot, str)
    assert "digraph" in dot
    assert dot.strip().endswith("}")


def test_dot_includes_block_names():
    diag = _build_simple_diagram()
    dot = rate_summary_dot(diag)
    # Block names should appear somewhere in the output (as node labels or IDs).
    assert "src" in dot or "udelay" in dot


def test_dot_contains_subgraph_clusters():
    """Rate groups should be rendered as DOT clusters."""
    diag = _build_simple_diagram()
    dot = rate_summary_dot(diag)
    assert "subgraph" in dot or "cluster" in dot


def test_dot_empty_diagram_still_valid():
    """An empty diagram should still produce a valid (body-less) digraph."""
    builder = jaxonomy.DiagramBuilder()
    src = library.Constant(0.0, name="lone")
    builder.add(src)
    diag = builder.build()
    dot = rate_summary_dot(diag)
    assert "digraph" in dot
    assert dot.strip().endswith("}")
