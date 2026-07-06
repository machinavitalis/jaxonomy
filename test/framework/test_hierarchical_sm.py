# SPDX-License-Identifier: MIT
"""
T-024 — hierarchical StateMachineBuilder tests.

Covers:

  - Backwards compatibility: flat state machines (no parents) build
    unchanged.
  - ``add_state(parent=composite)`` records parent/children correctly.
  - ``State.ancestors()`` returns the outermost-first chain.
  - At build time, a transition between two leaf states whose nearest
    common ancestor is composite_X fires:
        leaf1.on_exit → ... → leaf1's-parent.on_exit (up to but not
        including composite_X) → transition actions →
        composite_X-direct-child.on_entry → ... → leaf2.on_entry
  - The initial state's full ancestor on_entry chain runs at startup.
"""

from __future__ import annotations

import pytest

from jaxonomy.framework.state_machine_builder import (
    State,
    StateMachineBuilder,
)


# ── parent / children / ancestors structure ─────────────────────────────


def test_add_state_with_parent_links_children():
    bld = StateMachineBuilder()
    parent = bld.add_compound_state("composite")
    child_a = bld.add_state("a", parent=parent)
    child_b = bld.add_state("b", parent=parent)
    assert child_a.parent is parent
    assert child_b.parent is parent
    assert parent.children == [child_a, child_b]
    assert parent.is_leaf is False
    assert child_a.is_leaf is True


def test_unknown_parent_raises():
    bld = StateMachineBuilder()
    bogus = State(name="not_in_builder")
    with pytest.raises(ValueError, match="not added"):
        bld.add_state("a", parent=bogus)


def test_ancestors_walks_outermost_first():
    bld = StateMachineBuilder()
    grand = bld.add_compound_state("grand")
    parent = bld.add_compound_state("parent", parent=grand)
    leaf = bld.add_state("leaf", parent=parent)
    assert leaf.ancestors() == [grand, parent]
    assert parent.ancestors() == [grand]
    assert grand.ancestors() == []


# ── LCA helper ──────────────────────────────────────────────────────────


def test_lca_with_shared_ancestor():
    bld = StateMachineBuilder()
    grand = bld.add_compound_state("grand")
    pa = bld.add_compound_state("pa", parent=grand)
    pb = bld.add_compound_state("pb", parent=grand)
    a = bld.add_state("a", parent=pa)
    b = bld.add_state("b", parent=pb)
    assert StateMachineBuilder._lca(a, b) is grand


def test_lca_one_state_inside_the_other():
    bld = StateMachineBuilder()
    grand = bld.add_compound_state("grand")
    leaf = bld.add_state("leaf", parent=grand)
    assert StateMachineBuilder._lca(grand, leaf) is grand


def test_lca_no_common_ancestor():
    bld = StateMachineBuilder()
    a = bld.add_state("a")
    b = bld.add_state("b")
    assert StateMachineBuilder._lca(a, b) is None


# ── entry/exit chain folding at build time ─────────────────────────────


def test_entry_exit_chains_in_merged_actions():
    """A transition between leaf siblings under one composite should NOT
    fire the composite's on_exit / on_entry (LCA short-circuits it)."""
    bld = StateMachineBuilder()
    parent = bld.add_compound_state("parent")
    parent.on_exit = ["parent_exit = 1"]
    parent.on_entry = ["parent_entry = 1"]
    a = bld.add_state("a", parent=parent)
    a.on_exit = ["a_exit = 1"]
    b = bld.add_state("b", parent=parent)
    b.on_entry = ["b_entry = 1"]
    bld.set_initial_state(a)
    t = bld.add_transition(a, b, guard="True", actions=["mid = 1"])

    exit_chain = bld._exit_chain_from(t.source, stop_at=parent)
    entry_chain = bld._entry_chain_to(t.dest, stop_at=parent)

    assert exit_chain == ["a_exit = 1"]
    assert entry_chain == ["b_entry = 1"]


def test_transition_crossing_composite_boundary_fires_outer_chains():
    """Transition from a deeply-nested leaf to one outside the composite
    must fire all on_exit chains up to the LCA, then all on_entry chains
    down."""
    bld = StateMachineBuilder()
    grand = bld.add_compound_state("grand")
    pa = bld.add_compound_state("pa", parent=grand)
    pa.on_exit = ["pa_exit = 1"]
    a = bld.add_state("a", parent=pa)
    a.on_exit = ["a_exit = 1"]
    b = bld.add_state("b", parent=grand)
    b.on_entry = ["b_entry = 1"]

    exit_chain = bld._exit_chain_from(a, stop_at=grand)
    entry_chain = bld._entry_chain_to(b, stop_at=grand)

    # innermost-first on the way out
    assert exit_chain == ["a_exit = 1", "pa_exit = 1"]
    # outermost-first on the way in (just b since b is grand's direct child)
    assert entry_chain == ["b_entry = 1"]


def test_initial_state_entry_includes_ancestor_entries():
    bld = StateMachineBuilder()
    parent = bld.add_compound_state("parent")
    parent.on_entry = ["parent_setup = 1"]
    leaf = bld.add_state("leaf", parent=parent)
    leaf.on_entry = ["leaf_setup = 1"]
    bld.set_initial_state(leaf)

    chain = bld._entry_chain_to(leaf)
    # parent first, then leaf
    assert chain == ["parent_setup = 1", "leaf_setup = 1"]


# ── flat-builder regression: existing behaviour unchanged ──────────────


def test_flat_builder_unchanged():
    """Flat (non-hierarchical) state machines must behave exactly as
    before — no ancestor chain, on_entry/on_exit fold the legacy way."""
    bld = StateMachineBuilder()
    a = bld.add_state("a")
    b = bld.add_state("b")
    a.on_exit = ["a_x = 1"]
    b.on_entry = ["b_e = 1"]
    bld.set_initial_state(a)

    exit_chain = bld._exit_chain_from(a, stop_at=None)
    entry_chain = bld._entry_chain_to(b, stop_at=None)
    assert exit_chain == ["a_x = 1"]
    assert entry_chain == ["b_e = 1"]
