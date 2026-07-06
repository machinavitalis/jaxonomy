# SPDX-License-Identifier: MIT
"""T-117-followup-nested-buses: nested ``BusCreator`` buses + helpers.

The T-117-fu-bus-namedtuple :class:`BusCreator` produces a NamedTuple
pytree. NamedTuples nest natively in JAX — a ``BusCreator`` whose
inputs include other bus signals simply yields a NamedTuple whose
values are themselves NamedTuples, and ``jax.tree_util`` walks the
whole structure as a single pytree.

This file ships explicit verification that nested buses round-trip
cleanly through ``simulate`` and ``jit`` / ``grad``, plus tests for
three new helpers in ``jaxonomy.library``:

  * :func:`bus_fields(bus_signal)` — list of ``(name, value)`` pairs
    in declaration order.
  * :func:`flatten_bus(bus_signal)` — concatenates all leaves into a
    flat 1-D array (recurses into nested sub-buses).
  * :func:`unflatten_bus(flat_array, bus_spec)` — inverse round-trip.
"""

from __future__ import annotations

from collections import namedtuple

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import jaxonomy
from jaxonomy import library
from jaxonomy.library import (
    BusCreator,
    BusSelector,
    bus_fields,
    flatten_bus,
    unflatten_bus,
)
from jaxonomy.testing.markers import skip_if_not_jax

skip_if_not_jax()


# ---------------------------------------------------------------------------
# Nested-bus round-trip through ``simulate`` — the central proof that
# NamedTuple-of-NamedTuple flows through every stage of the simulator.
# ---------------------------------------------------------------------------


def test_nested_bus_roundtrip_through_simulate():
    """``BusCreator(("a", "inner"))`` where ``inner`` is itself a bus.

    Wiring (port indices in []):
        c_a [0] ────────────────────────────→ outer.a [0]
        c_x [0] ────→ inner.x [0]
        c_y [0] ────→ inner.y [1]
                        inner.bus [0] ─────→ outer.inner [1]

    Then ``BusSelector("inner")`` pulls out the sub-bus from ``outer``.
    The sub-bus is itself a NamedTuple with fields ``x`` and ``y``;
    we route it back to a second-level ``BusSelector("x")`` to recover
    the scalar.

    We record three outputs:
        out_a → outer.a   (1.0)
        sub_x → outer.inner.x   (10.0)
        sub_y → outer.inner.y   (20.0)
    """
    c_a = library.Constant(1.0)
    c_x = library.Constant(10.0)
    c_y = library.Constant(20.0)

    inner = BusCreator(("x", "y"), name="inner")
    outer = BusCreator(("a", "inner"), name="outer")

    sel_inner = BusSelector("inner", name="sel_inner")
    sel_a = BusSelector("a", name="sel_a")
    sel_inner_x = BusSelector("x", name="sel_inner_x")
    sel_inner_y = BusSelector("y", name="sel_inner_y")

    builder = jaxonomy.DiagramBuilder()
    builder.add(
        c_a, c_x, c_y,
        inner, outer,
        sel_a, sel_inner, sel_inner_x, sel_inner_y,
    )
    # Build the inner bus.
    builder.connect(c_x.output_ports[0], inner.input_ports[0])
    builder.connect(c_y.output_ports[0], inner.input_ports[1])
    # Build the outer bus: scalar "a" + nested "inner" bus.
    builder.connect(c_a.output_ports[0], outer.input_ports[0])
    builder.connect(inner.output_ports[0], outer.input_ports[1])
    # Pull "a" back out at the top.
    builder.connect(outer.output_ports[0], sel_a.input_ports[0])
    # Pull the nested bus back out, then pull "x" and "y" from it.
    builder.connect(outer.output_ports[0], sel_inner.input_ports[0])
    builder.connect(sel_inner.output_ports[0], sel_inner_x.input_ports[0])
    builder.connect(sel_inner.output_ports[0], sel_inner_y.input_ports[0])

    diagram = builder.build()
    ctx = diagram.create_context()
    results = jaxonomy.simulate(
        diagram, ctx, (0.0, 0.1),
        recorded_signals={
            "out_a": sel_a.output_ports[0],
            "sub_x": sel_inner_x.output_ports[0],
            "sub_y": sel_inner_y.output_ports[0],
        },
    )

    np.testing.assert_allclose(
        float(np.asarray(results.outputs["out_a"])[-1]), 1.0
    )
    np.testing.assert_allclose(
        float(np.asarray(results.outputs["sub_x"])[-1]), 10.0
    )
    np.testing.assert_allclose(
        float(np.asarray(results.outputs["sub_y"])[-1]), 20.0
    )


def test_nested_bus_jit_and_grad_flow_through_at_pure_jax_level():
    """Pure-JAX trace: a NamedTuple of NamedTuples is a valid pytree.

    f(a, x, y) = outer.a * 2 + outer.inner.x * 3 + outer.inner.y * 5
    expected gradient w.r.t. (a, x, y) = (2, 3, 5).
    """
    inner_cls = BusCreator(("x", "y")).bus_type
    outer_cls = BusCreator(("a", "inner")).bus_type

    @jax.jit
    def f(a, x, y):
        inner = inner_cls(x, y)
        outer = outer_cls(a, inner)
        return outer.a * 2.0 + outer.inner.x * 3.0 + outer.inner.y * 5.0

    val = f(jnp.asarray(1.0), jnp.asarray(10.0), jnp.asarray(20.0))
    np.testing.assert_allclose(float(val), 1.0 * 2.0 + 10.0 * 3.0 + 20.0 * 5.0)

    g_a, g_x, g_y = jax.grad(f, argnums=(0, 1, 2))(1.0, 10.0, 20.0)
    np.testing.assert_allclose(float(g_a), 2.0)
    np.testing.assert_allclose(float(g_x), 3.0)
    np.testing.assert_allclose(float(g_y), 5.0)


def test_nested_bus_is_a_single_jax_pytree():
    """``jax.tree_util.tree_leaves`` flattens nested buses transparently."""
    inner_cls = BusCreator(("x", "y")).bus_type
    outer_cls = BusCreator(("a", "inner")).bus_type

    bus = outer_cls(
        jnp.asarray(1.0),
        inner_cls(jnp.asarray(10.0), jnp.asarray(20.0)),
    )
    leaves = jax.tree_util.tree_leaves(bus)
    # Three leaves total: a, inner.x, inner.y.
    assert len(leaves) == 3
    np.testing.assert_allclose(float(leaves[0]), 1.0)
    np.testing.assert_allclose(float(leaves[1]), 10.0)
    np.testing.assert_allclose(float(leaves[2]), 20.0)


# ---------------------------------------------------------------------------
# bus_fields: iteration in declaration order.
# ---------------------------------------------------------------------------


def test_bus_fields_yields_pairs_in_declaration_order():
    creator = BusCreator(("position", "velocity", "acceleration"))
    Bus = creator.bus_type
    bus = Bus(1.0, 2.0, 3.0)
    fields = bus_fields(bus)
    assert fields == [
        ("position", 1.0),
        ("velocity", 2.0),
        ("acceleration", 3.0),
    ]


def test_bus_fields_returns_list_not_generator():
    """We promise a list so callers can re-iterate."""
    Bus = BusCreator(("a", "b")).bus_type
    bus = Bus(1.0, 2.0)
    fields = bus_fields(bus)
    assert isinstance(fields, list)
    # Re-iterating works (a generator would be exhausted on the second pass).
    assert list(fields) == fields


def test_bus_fields_works_on_nested_bus_without_recursing():
    """``bus_fields`` is a single-level helper — sub-buses are returned
    as their own NamedTuple values, callers recurse if they need to."""
    inner = BusCreator(("x", "y")).bus_type
    outer = BusCreator(("a", "inner")).bus_type
    bus = outer(1.0, inner(10.0, 20.0))
    fields = bus_fields(bus)
    assert [name for name, _ in fields] == ["a", "inner"]
    assert fields[0][1] == 1.0
    # The second value is itself a NamedTuple-typed bus.
    inner_value = fields[1][1]
    assert hasattr(inner_value, "_fields")
    assert inner_value.x == 10.0 and inner_value.y == 20.0


def test_bus_fields_rejects_non_bus_values():
    with pytest.raises(TypeError, match="NamedTuple-shaped bus signal"):
        bus_fields([1.0, 2.0, 3.0])
    with pytest.raises(TypeError, match="NamedTuple-shaped bus signal"):
        bus_fields({"a": 1.0, "b": 2.0})


def test_bus_fields_accepts_any_namedtuple_class():
    """The helper is structural — it does not require a BusCreator-built bus."""
    UserBus = namedtuple("UserBus", ("p", "q"))
    bus = UserBus(1.0, 2.0)
    fields = bus_fields(bus)
    assert fields == [("p", 1.0), ("q", 2.0)]


# ---------------------------------------------------------------------------
# flatten_bus / unflatten_bus: round-trip identity.
# ---------------------------------------------------------------------------


def test_flatten_bus_flat_scalar_fields():
    creator = BusCreator(("a", "b", "c"))
    bus = creator.bus_type(1.0, 2.0, 3.0)
    flat = flatten_bus(bus)
    np.testing.assert_allclose(np.asarray(flat), [1.0, 2.0, 3.0])


def test_flatten_bus_nested_in_declaration_order():
    inner = BusCreator(("x", "y")).bus_type
    outer = BusCreator(("a", "inner", "b")).bus_type
    bus = outer(1.0, inner(10.0, 20.0), 99.0)
    flat = flatten_bus(bus)
    # Declaration order: a, inner.x, inner.y, b.
    np.testing.assert_allclose(np.asarray(flat), [1.0, 10.0, 20.0, 99.0])


def test_flatten_bus_vector_leaves():
    """1-D vector leaves contribute their length to the flat output."""
    creator = BusCreator(("a", "b"))
    bus = creator.bus_type(np.asarray([1.0, 2.0]), np.asarray([3.0, 4.0, 5.0]))
    flat = flatten_bus(bus)
    np.testing.assert_allclose(
        np.asarray(flat), [1.0, 2.0, 3.0, 4.0, 5.0]
    )


def test_flatten_bus_rejects_non_bus_values():
    with pytest.raises(TypeError, match="NamedTuple-shaped bus signal"):
        flatten_bus([1.0, 2.0, 3.0])


def test_unflatten_bus_roundtrips_flat_scalar_bus():
    creator = BusCreator(("a", "b", "c"))
    bus_spec = creator.bus_type
    bus = bus_spec(1.0, 2.0, 3.0)
    flat = flatten_bus(bus)
    restored = unflatten_bus(flat, bus_spec)
    assert isinstance(restored, bus_spec)
    np.testing.assert_allclose(float(restored.a), 1.0)
    np.testing.assert_allclose(float(restored.b), 2.0)
    np.testing.assert_allclose(float(restored.c), 3.0)


def test_unflatten_bus_roundtrips_nested_bus():
    """A nested bus with a ``_field_specs`` annotation round-trips
    automatically — flatten then unflatten yields equal field-by-field."""
    inner_spec = BusCreator(("x", "y")).bus_type
    outer_spec = BusCreator(("a", "inner", "b")).bus_type
    # Tag the outer spec with its nested-bus schema so unflatten_bus can
    # find the nested NamedTuple class.
    outer_spec._field_specs = {"inner": inner_spec}

    bus = outer_spec(1.0, inner_spec(10.0, 20.0), 99.0)
    flat = flatten_bus(bus)
    restored = unflatten_bus(flat, outer_spec)

    assert isinstance(restored, outer_spec)
    assert isinstance(restored.inner, inner_spec)
    np.testing.assert_allclose(float(restored.a), 1.0)
    np.testing.assert_allclose(float(restored.inner.x), 10.0)
    np.testing.assert_allclose(float(restored.inner.y), 20.0)
    np.testing.assert_allclose(float(restored.b), 99.0)


def test_unflatten_bus_vector_leaves_via_leaf_sizes():
    """Vector leaves need explicit ``leaf_sizes`` so we know per-field length."""
    bus_spec = BusCreator(("u", "v")).bus_type
    flat = np.asarray([1.0, 2.0, 3.0, 4.0, 5.0])
    restored = unflatten_bus(flat, bus_spec, leaf_sizes=(2, 3))
    np.testing.assert_allclose(np.asarray(restored.u), [1.0, 2.0])
    np.testing.assert_allclose(np.asarray(restored.v), [3.0, 4.0, 5.0])


def test_unflatten_bus_rejects_non_namedtuple_spec():
    with pytest.raises(TypeError, match="NamedTuple class"):
        unflatten_bus(np.asarray([1.0, 2.0]), tuple)
    with pytest.raises(TypeError, match="NamedTuple class"):
        unflatten_bus(np.asarray([1.0]), list)


def test_unflatten_bus_length_mismatch_raises():
    bus_spec = BusCreator(("a", "b", "c")).bus_type
    with pytest.raises(ValueError, match="3 scalar leaves but flat_array"):
        unflatten_bus(np.asarray([1.0, 2.0]), bus_spec)


def test_unflatten_bus_leaf_sizes_shape_mismatch_raises():
    bus_spec = BusCreator(("a", "b")).bus_type
    with pytest.raises(ValueError, match="leaf_sizes shape mismatch"):
        unflatten_bus(np.asarray([1.0, 2.0, 3.0]), bus_spec, leaf_sizes=(3,))


def test_flatten_unflatten_full_roundtrip_nested_vectors():
    """Combined: nested bus with vector leaves, flatten + unflatten."""
    inner_spec = BusCreator(("u", "v")).bus_type
    outer_spec = BusCreator(("scalar", "vec_bus")).bus_type
    outer_spec._field_specs = {"vec_bus": inner_spec}

    bus = outer_spec(
        np.asarray([7.0]),
        inner_spec(np.asarray([1.0, 2.0]), np.asarray([3.0, 4.0, 5.0])),
    )
    flat = flatten_bus(bus)
    np.testing.assert_allclose(
        np.asarray(flat), [7.0, 1.0, 2.0, 3.0, 4.0, 5.0]
    )

    restored = unflatten_bus(
        flat, outer_spec, leaf_sizes=(1, (2, 3))
    )
    np.testing.assert_allclose(np.asarray(restored.scalar), [7.0])
    np.testing.assert_allclose(np.asarray(restored.vec_bus.u), [1.0, 2.0])
    np.testing.assert_allclose(np.asarray(restored.vec_bus.v), [3.0, 4.0, 5.0])


# ---------------------------------------------------------------------------
# JAX-compatibility of the helpers themselves.
# ---------------------------------------------------------------------------


def test_flatten_bus_traces_through_jit():
    """``flatten_bus`` is JAX-traceable — useful inside compiled blocks."""
    creator = BusCreator(("a", "b", "c"))
    Bus = creator.bus_type

    @jax.jit
    def flatten(a, b, c):
        return flatten_bus(Bus(a, b, c))

    out = flatten(jnp.asarray(1.0), jnp.asarray(2.0), jnp.asarray(3.0))
    np.testing.assert_allclose(np.asarray(out), [1.0, 2.0, 3.0])


def test_unflatten_bus_traces_through_jit_and_grad():
    """``unflatten_bus`` survives ``jit`` + ``grad`` because it only
    does NamedTuple construction and array indexing."""
    bus_spec = BusCreator(("a", "b", "c")).bus_type

    @jax.jit
    def loss(flat):
        bus = unflatten_bus(flat, bus_spec)
        return bus.a * 2.0 + bus.b * 3.0 + bus.c * 5.0

    g = jax.grad(loss)(jnp.asarray([1.0, 2.0, 3.0]))
    np.testing.assert_allclose(np.asarray(g), [2.0, 3.0, 5.0])
