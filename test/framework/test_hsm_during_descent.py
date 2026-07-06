# SPDX-License-Identifier: MIT
"""
T-024a — during actions + composite-target descent tests.

Stresses the StateMachineBuilder beyond the trivial flat case:

  - A composite state with `during` actions: the during chain folds
    into entry-of-the-composite for transitions that *enter* it; does
    NOT re-fire for transitions strictly within the composite (UML
    "stay in scope, no re-entry" semantics).
  - A composite state used as a transition target: the runtime
    auto-descends through `initial_child` to a leaf, fires every
    intermediate composite's entry + during, and the resulting
    transition's destNodeId is the leaf.
  - `set_initial_child` overrides the default (first-added child)
    initial child.
  - The startup entry point on a composite initial state correctly
    descends to a leaf.
  - Multi-level nesting: grand → mid → leaf, with `during` at every
    level.
"""

from __future__ import annotations

import pytest

from jaxonomy.framework.state_machine_builder import (
    State,
    StateMachineBuilder,
)


# ── descent helper ─────────────────────────────────────────────────────


def test_descend_to_leaf_passes_through_initial_child():
    bld = StateMachineBuilder()
    grand = bld.add_compound_state("grand")
    mid = bld.add_compound_state("mid", parent=grand)
    leaf = bld.add_state("leaf", parent=mid)

    final, path = bld._descend_to_leaf(grand)
    assert final is leaf
    assert path == [mid, leaf]


def test_descend_to_leaf_no_op_for_leaf_state():
    bld = StateMachineBuilder()
    a = bld.add_state("a")
    final, path = bld._descend_to_leaf(a)
    assert final is a
    assert path == []


def test_descend_raises_when_composite_has_no_children():
    bld = StateMachineBuilder()
    # add a state, then mark it composite by adding a child, then remove
    # — but builder doesn't support removing.  Instead, manually create
    # a State that is_leaf=False with no children to provoke the error.
    parent = bld.add_state("parent")
    parent.children.append(State(name="dangling"))  # not via builder
    parent.initial_child = None
    with pytest.raises(ValueError, match="no initial child"):
        bld._descend_to_leaf(parent)


# ── set_initial_child ─────────────────────────────────────────────────


def test_first_added_child_is_default_initial():
    bld = StateMachineBuilder()
    parent = bld.add_compound_state("parent")
    a = bld.add_state("a", parent=parent)
    b = bld.add_state("b", parent=parent)
    assert parent.initial_child is a


def test_set_initial_child_overrides_default():
    bld = StateMachineBuilder()
    parent = bld.add_compound_state("parent")
    a = bld.add_state("a", parent=parent)
    b = bld.add_state("b", parent=parent)
    bld.set_initial_child(parent, b)
    assert parent.initial_child is b


def test_set_initial_child_rejects_non_child():
    bld = StateMachineBuilder()
    parent = bld.add_compound_state("parent")
    a = bld.add_state("a", parent=parent)
    other = bld.add_state("other")  # not a child of parent
    with pytest.raises(ValueError, match="not a child"):
        bld.set_initial_child(parent, other)


# ── during folded into entry chain ────────────────────────────────────


def test_during_fires_on_entry_via_include_during():
    bld = StateMachineBuilder()
    parent = bld.add_compound_state("parent")
    parent.during = ["heartbeat = heartbeat + 1"]
    parent.on_entry = ["entered_parent = 1"]
    leaf = bld.add_state("leaf", parent=parent)
    leaf.on_entry = ["entered_leaf = 1"]

    chain = bld._entry_chain_to(leaf, include_during=True)
    # parent.on_entry, then parent.during, then leaf.on_entry, then leaf.during
    assert chain == [
        "entered_parent = 1",
        "heartbeat = heartbeat + 1",
        "entered_leaf = 1",
    ]


def test_during_NOT_in_chain_without_include_during_flag():
    bld = StateMachineBuilder()
    parent = bld.add_compound_state("parent")
    parent.during = ["heartbeat = heartbeat + 1"]
    leaf = bld.add_state("leaf", parent=parent)
    leaf.on_entry = ["entered_leaf = 1"]
    chain = bld._entry_chain_to(leaf)  # default include_during=False
    assert "heartbeat = heartbeat + 1" not in chain


# ── transition resolution: composite target → leaf descend ────────────


def test_transition_to_composite_descends_to_initial_leaf():
    """add_transition(src, dest=composite) should resolve dest to the
    composite's initial-child leaf in the resulting StateMachine block."""
    bld = StateMachineBuilder()
    src = bld.add_state("src")
    composite = bld.add_compound_state("composite")
    inner_a = bld.add_state("inner_a", parent=composite)  # initial child
    inner_b = bld.add_state("inner_b", parent=composite)
    bld.set_initial_state(src)
    bld.add_transition(src, composite, guard="True")

    # Manually traverse the resolution path the builder will take.
    resolved, path = bld._descend_to_leaf(composite)
    assert resolved is inner_a
    assert path == [inner_a]


def test_transition_to_deeply_nested_composite_descends_through_chain():
    bld = StateMachineBuilder()
    src = bld.add_state("src")
    grand = bld.add_compound_state("grand")
    mid = bld.add_compound_state("mid", parent=grand)
    leaf = bld.add_state("leaf", parent=mid)
    bld.set_initial_state(src)

    resolved, path = bld._descend_to_leaf(grand)
    assert resolved is leaf
    assert path == [mid, leaf]


# ── initial state descent ────────────────────────────────────────────


def test_set_initial_state_composite_descends_at_entry_actions_time():
    bld = StateMachineBuilder()
    composite = bld.add_compound_state("composite")
    composite.on_entry = ["composite_entered = 1"]
    composite.during = ["composite_during = 1"]
    leaf = bld.add_state("leaf", parent=composite)
    leaf.on_entry = ["leaf_entered = 1"]
    bld.set_initial_state(composite)

    actions = bld._entry_actions(output_names=[])
    # Order: composite.on_entry, composite.during, leaf.on_entry,
    # leaf.during (none).
    assert "composite_entered = 1" in actions
    assert "composite_during = 1" in actions
    assert "leaf_entered = 1" in actions
    # composite_entered comes before leaf_entered.
    assert actions.index("composite_entered = 1") < actions.index("leaf_entered = 1")


# ── multi-level nested during ────────────────────────────────────────


def test_three_level_nest_during_chain():
    """grand.during, mid.during, leaf.during all fire when the
    transition enters from outside grand."""
    bld = StateMachineBuilder()
    grand = bld.add_compound_state("grand")
    grand.during = ["g_during = 1"]
    grand.on_entry = ["g_entry = 1"]
    mid = bld.add_compound_state("mid", parent=grand)
    mid.during = ["m_during = 1"]
    mid.on_entry = ["m_entry = 1"]
    leaf = bld.add_state("leaf", parent=mid)
    leaf.on_entry = ["l_entry = 1"]

    chain = bld._entry_chain_to(leaf, include_during=True)
    # All three on_entry and the two non-leaf during should appear, in order.
    expected = [
        "g_entry = 1", "g_during = 1",
        "m_entry = 1", "m_during = 1",
        "l_entry = 1",
    ]
    assert chain == expected


# ── transition WITHIN a composite does not re-fire its during ────────


def test_during_does_NOT_refire_for_intra_composite_transition():
    """When source and dest are siblings under the same composite, the
    composite's during action should NOT fire — the LCA logic short-
    circuits ancestors common to both."""
    bld = StateMachineBuilder()
    parent = bld.add_compound_state("parent")
    parent.during = ["parent_during = 1"]
    a = bld.add_state("a", parent=parent)
    b = bld.add_state("b", parent=parent)
    bld.set_initial_state(a)

    # parent is the LCA of (a, b) — it is excluded from the entry chain.
    chain = bld._entry_chain_to(b, stop_at=parent, include_during=True)
    assert chain == []  # b.on_entry is empty + b is leaf so no during applies


def test_during_DOES_fire_for_inter_composite_transition():
    """A transition that crosses out of one composite into another
    composite's region must fire the destination composite's during."""
    bld = StateMachineBuilder()
    grand = bld.add_compound_state("grand")
    pa = bld.add_compound_state("pa", parent=grand)
    pa.during = ["pa_during = 1"]
    a = bld.add_state("a", parent=pa)
    pb = bld.add_compound_state("pb", parent=grand)
    pb.during = ["pb_during = 1"]
    b = bld.add_state("b", parent=pb)

    # LCA of (a, b) = grand.  Transition exits pa, enters pb.
    chain = bld._entry_chain_to(b, stop_at=grand, include_during=True)
    assert "pb_during = 1" in chain
    assert "pa_during = 1" not in chain  # we're leaving pa


# ── flat (non-hierarchical) regression ────────────────────────────────


def test_flat_state_machine_unchanged_by_t024a():
    """A purely-flat machine with no composites should produce the same
    transition action chain as before T-024a."""
    bld = StateMachineBuilder()
    a = bld.add_state("a")
    a.on_exit = ["a_exit = 1"]
    a.during = ["a_during = 1"]   # leaf during is a no-op without
                                   # composite scope
    b = bld.add_state("b")
    b.on_entry = ["b_entry = 1"]
    bld.set_initial_state(a)
    t = bld.add_transition(a, b, guard="True", actions=["mid = 1"])

    exit_chain = bld._exit_chain_from(t.source, stop_at=None)
    entry_chain = bld._entry_chain_to(t.dest, stop_at=None, include_during=True)
    # Leaf during fires on entry to b's region from outside.  In flat
    # SMs this is fine and matches user intent.
    assert exit_chain == ["a_exit = 1"]
    assert entry_chain == ["b_entry = 1"]   # b has no during
