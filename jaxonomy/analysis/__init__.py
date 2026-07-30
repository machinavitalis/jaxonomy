# SPDX-License-Identifier: MIT

"""Whole-model analysis on top of the framework's dependency structure.

The influence graph merges the model's leaf-level dependency DAG (which says
*whether* information flows) with autodiff Jacobians (which say *how much*),
giving quantitative model slicing, chain-rule path attribution, bottleneck
detection, and dead-edge diagnostics on one queryable object.
"""

from .block_jacobians import LeafJacobians, leaf_jacobians
from .influence import (
    InfluenceGraph,
    InfluenceSlice,
    PathAttribution,
    influence_graph,
)
from .influence_context import format_influence_subgraph, influence_subgraph

__all__ = [
    "InfluenceGraph",
    "InfluenceSlice",
    "LeafJacobians",
    "PathAttribution",
    "format_influence_subgraph",
    "influence_graph",
    "influence_subgraph",
    "leaf_jacobians",
]
