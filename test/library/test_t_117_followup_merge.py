# SPDX-License-Identifier: MIT
"""T-117-followup-bus-merge: ``merge_buses`` helper + ``BusMerge`` block.

``merge_buses(bus_a, bus_b, *, on_collision="error")`` returns a
NamedTuple-shaped bus whose fields are the union of the two inputs'
fields (in declaration order ``a ++ (b - a)``). ``BusMerge`` wraps the
helper as a LeafSystem so a merged bus can be produced inside a Diagram.

These tests pin:

  * Disjoint-fields union: ``merge_buses({x, y}, {z}) -> {x, y, z}``.
  * Collision policy: ``"error"`` raises; ``"prefer_a"`` / ``"prefer_b"``
    each take the leaf from the named source.
  * End-to-end ``BusMerge`` flow through ``simulate``.
  * Differentiability: ``jax.grad`` flows from each merged-bus field
    back to the upstream input that contributed it.
  * Construction-time validation (bad ``on_collision``, bad spec,
    invalid identifier).
"""

from __future__ import annotations

from collections import namedtuple

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import jaxonomy
from jaxonomy import library
from jaxonomy.library import BusCreator, BusMerge, BusSelector, merge_buses
from jaxonomy.testing.markers import skip_if_not_jax

skip_if_not_jax()


# ---------------------------------------------------------------------------
# Pure-helper tests for ``merge_buses``. NamedTuple inputs are built
# directly so we exercise the helper in isolation from the LeafSystem
# wrapper.
# ---------------------------------------------------------------------------


def _ntuple(name, fields, values):
    """Tiny convenience: build a NamedTuple instance from parallel lists."""
    Cls = namedtuple(name, fields)
    return Cls(*values)


def test_merge_buses_disjoint_union():
    """{x, y} U {z} -> {x, y, z} in declaration order, with the right values."""
    bus_a = _ntuple("BusA", ["x", "y"], [1.0, 2.0])
    bus_b = _ntuple("BusB", ["z"], [3.0])
    merged = merge_buses(bus_a, bus_b)
    assert merged._fields == ("x", "y", "z")
    np.testing.assert_allclose(float(merged.x), 1.0)
    np.testing.assert_allclose(float(merged.y), 2.0)
    np.testing.assert_allclose(float(merged.z), 3.0)


def test_merge_buses_disjoint_union_three_each():
    """A wider disjoint case to confirm order is ``a`` then ``b``."""
    bus_a = _ntuple("A", ["a1", "a2", "a3"], [10.0, 20.0, 30.0])
    bus_b = _ntuple("B", ["b1", "b2", "b3"], [40.0, 50.0, 60.0])
    merged = merge_buses(bus_a, bus_b)
    assert merged._fields == ("a1", "a2", "a3", "b1", "b2", "b3")


def test_merge_buses_collision_error_raises():
    bus_a = _ntuple("A", ["x", "y"], [1.0, 2.0])
    bus_b = _ntuple("B", ["y", "z"], [99.0, 3.0])
    with pytest.raises(ValueError, match="collision"):
        merge_buses(bus_a, bus_b)
    # Default policy is ``"error"``.
    with pytest.raises(ValueError, match="collision"):
        merge_buses(bus_a, bus_b, on_collision="error")


def test_merge_buses_collision_prefer_a_keeps_a_value():
    bus_a = _ntuple("A", ["x", "y"], [1.0, 2.0])
    bus_b = _ntuple("B", ["y", "z"], [99.0, 3.0])
    merged = merge_buses(bus_a, bus_b, on_collision="prefer_a")
    # Schema is independent of policy: ``a ++ (b - a)`` = (x, y, z).
    assert merged._fields == ("x", "y", "z")
    np.testing.assert_allclose(float(merged.x), 1.0)
    np.testing.assert_allclose(float(merged.y), 2.0)  # from a, not 99
    np.testing.assert_allclose(float(merged.z), 3.0)


def test_merge_buses_collision_prefer_b_keeps_b_value():
    bus_a = _ntuple("A", ["x", "y"], [1.0, 2.0])
    bus_b = _ntuple("B", ["y", "z"], [99.0, 3.0])
    merged = merge_buses(bus_a, bus_b, on_collision="prefer_b")
    assert merged._fields == ("x", "y", "z")
    np.testing.assert_allclose(float(merged.x), 1.0)
    np.testing.assert_allclose(float(merged.y), 99.0)  # from b
    np.testing.assert_allclose(float(merged.z), 3.0)


def test_merge_buses_invalid_on_collision_raises():
    bus_a = _ntuple("A", ["x"], [1.0])
    bus_b = _ntuple("B", ["y"], [2.0])
    with pytest.raises(ValueError, match="on_collision"):
        merge_buses(bus_a, bus_b, on_collision="bogus")


def test_merge_buses_non_namedtuple_input_raises():
    bus_a = _ntuple("A", ["x"], [1.0])
    with pytest.raises(TypeError, match="bus_b"):
        merge_buses(bus_a, (1.0, 2.0))  # plain tuple has no ``_fields``
    with pytest.raises(TypeError, match="bus_a"):
        merge_buses({"x": 1.0}, bus_a)


def test_merge_buses_returns_namedtuple_compatible_with_jax():
    """The merged bus is a JAX pytree (NamedTuples are auto-registered)."""
    bus_a = _ntuple("A", ["x"], [jnp.asarray(1.0)])
    bus_b = _ntuple("B", ["y"], [jnp.asarray(2.0)])
    merged = merge_buses(bus_a, bus_b)
    leaves = jax.tree_util.tree_leaves(merged)
    assert len(leaves) == 2
    np.testing.assert_allclose(float(leaves[0]), 1.0)
    np.testing.assert_allclose(float(leaves[1]), 2.0)


# ---------------------------------------------------------------------------
# Differentiability: gradient must flow back to whichever input bus
# contributed each merged-bus leaf.
# ---------------------------------------------------------------------------


def test_merge_buses_differentiable_disjoint():
    """f(a1, a2, b1) = m.a1*7 + m.a2*8 + m.b1*9 -> grads (7, 8, 9)."""
    BusA = namedtuple("BusA", ["a1", "a2"])
    BusB = namedtuple("BusB", ["b1"])

    def f(a1, a2, b1):
        merged = merge_buses(BusA(a1, a2), BusB(b1))
        return merged.a1 * 7.0 + merged.a2 * 8.0 + merged.b1 * 9.0

    g_a1, g_a2, g_b1 = jax.grad(f, argnums=(0, 1, 2))(1.0, 2.0, 3.0)
    np.testing.assert_allclose(float(g_a1), 7.0)
    np.testing.assert_allclose(float(g_a2), 8.0)
    np.testing.assert_allclose(float(g_b1), 9.0)


def test_merge_buses_differentiable_prefer_a():
    """Under ``prefer_a``, gradient on a colliding leaf flows to ``a``,
    not to ``b``.
    """
    BusA = namedtuple("BusA", ["x", "y"])
    BusB = namedtuple("BusB", ["y", "z"])

    def f(ax, ay, by, bz):
        merged = merge_buses(BusA(ax, ay), BusB(by, bz), on_collision="prefer_a")
        return merged.x + merged.y * 100.0 + merged.z

    g_ax, g_ay, g_by, g_bz = jax.grad(f, argnums=(0, 1, 2, 3))(
        1.0, 2.0, 99.0, 3.0
    )
    np.testing.assert_allclose(float(g_ax), 1.0)
    np.testing.assert_allclose(float(g_ay), 100.0)  # from a, gets the *100
    np.testing.assert_allclose(float(g_by), 0.0)  # discarded under prefer_a
    np.testing.assert_allclose(float(g_bz), 1.0)


def test_merge_buses_traces_through_jit():
    BusA = namedtuple("BusA", ["x", "y"])
    BusB = namedtuple("BusB", ["z"])

    @jax.jit
    def f(x, y, z):
        merged = merge_buses(BusA(x, y), BusB(z))
        return merged.x + merged.y + merged.z

    out = f(jnp.asarray(1.0), jnp.asarray(2.0), jnp.asarray(3.0))
    np.testing.assert_allclose(float(out), 6.0)


# ---------------------------------------------------------------------------
# ``BusMerge`` block end-to-end: feed two BusCreators into a BusMerge,
# select a field out the back, and confirm the value flows through
# ``simulate``.
# ---------------------------------------------------------------------------


def _build_merge_diagram(
    fields_a, values_a, fields_b, values_b, select, on_collision="error"
):
    """Two BusCreators -> BusMerge -> BusSelector. Returns ``(diagram, selector)``."""
    sources_a = [library.Constant(v) for v in values_a]
    sources_b = [library.Constant(v) for v in values_b]
    creator_a = BusCreator(fields_a, name="creator_a")
    creator_b = BusCreator(fields_b, name="creator_b")
    merger = BusMerge(fields_a, fields_b, on_collision=on_collision, name="merger")
    selector = BusSelector(select, name="selector")

    builder = jaxonomy.DiagramBuilder()
    builder.add(*sources_a, *sources_b, creator_a, creator_b, merger, selector)
    for src, port in zip(sources_a, creator_a.input_ports):
        builder.connect(src.output_ports[0], port)
    for src, port in zip(sources_b, creator_b.input_ports):
        builder.connect(src.output_ports[0], port)
    builder.connect(creator_a.output_ports[0], merger.input_ports[0])
    builder.connect(creator_b.output_ports[0], merger.input_ports[1])
    builder.connect(merger.output_ports[0], selector.input_ports[0])
    return builder.build(), selector


@pytest.mark.parametrize(
    ("select", "expected"),
    [
        ("x", 1.0),
        ("y", 2.0),
        ("z", 3.0),
    ],
)
def test_busmerge_end_to_end_disjoint(select, expected):
    """Merge {x, y} with {z}, select each merged field through ``simulate``."""
    diagram, selector = _build_merge_diagram(
        fields_a=("x", "y"),
        values_a=(1.0, 2.0),
        fields_b=("z",),
        values_b=(3.0,),
        select=select,
    )
    ctx = diagram.create_context()
    results = jaxonomy.simulate(
        diagram,
        ctx,
        (0.0, 0.1),
        recorded_signals={"out": selector.output_ports[0]},
    )
    out = float(np.asarray(results.outputs["out"])[-1])
    np.testing.assert_allclose(out, expected)


def test_busmerge_end_to_end_prefer_b():
    """Field collision on ``y`` resolved by ``prefer_b`` end-to-end."""
    diagram, selector = _build_merge_diagram(
        fields_a=("x", "y"),
        values_a=(1.0, 2.0),
        fields_b=("y", "z"),
        values_b=(99.0, 3.0),
        select="y",
        on_collision="prefer_b",
    )
    ctx = diagram.create_context()
    results = jaxonomy.simulate(
        diagram,
        ctx,
        (0.0, 0.1),
        recorded_signals={"out": selector.output_ports[0]},
    )
    out = float(np.asarray(results.outputs["out"])[-1])
    np.testing.assert_allclose(out, 99.0)


# ---------------------------------------------------------------------------
# ``BusMerge`` construction-time validation.
# ---------------------------------------------------------------------------


def test_busmerge_collision_error_raises_at_construction():
    with pytest.raises(ValueError, match="collision"):
        BusMerge(("x", "y"), ("y", "z"))


def test_busmerge_invalid_on_collision_raises():
    with pytest.raises(ValueError, match="on_collision"):
        BusMerge(("x",), ("y",), on_collision="bogus")


def test_busmerge_empty_spec_raises():
    with pytest.raises(ValueError, match="bus_spec_a"):
        BusMerge((), ("y",))
    with pytest.raises(ValueError, match="bus_spec_b"):
        BusMerge(("x",), ())


def test_busmerge_invalid_identifier_raises():
    with pytest.raises(ValueError, match="not a valid Python identifier"):
        BusMerge(("x", "1bad"), ("y",))


def test_busmerge_accepts_buscreator_instance_as_spec():
    """``bus_spec`` accepts a :class:`BusCreator` instance directly."""
    creator_a = BusCreator(("a", "b"))
    creator_b = BusCreator(("c",))
    merger = BusMerge(creator_a, creator_b)
    assert merger.field_names == ("a", "b", "c")


def test_busmerge_accepts_namedtuple_class_as_spec():
    BusA = namedtuple("BusA", ["a", "b"])
    BusB = namedtuple("BusB", ["c"])
    merger = BusMerge(BusA, BusB)
    assert merger.field_names == ("a", "b", "c")


def test_busmerge_collisions_property_reports_overlapping_names():
    merger = BusMerge(("x", "y"), ("y", "z"), on_collision="prefer_a")
    assert merger.collisions == ("y",)
    assert merger.field_names == ("x", "y", "z")
    assert merger.on_collision == "prefer_a"


def test_busmerge_bus_type_property_is_namedtuple():
    merger = BusMerge(("a", "b"), ("c",))
    Bus = merger.bus_type
    assert issubclass(Bus, tuple)
    assert Bus._fields == ("a", "b", "c")
    inst = Bus(1.0, 2.0, 3.0)
    assert inst.a == 1.0 and inst.b == 2.0 and inst.c == 3.0
