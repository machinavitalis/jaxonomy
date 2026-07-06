# SPDX-License-Identifier: MIT
"""T-117-followup-bus-namedtuple: BusCreator / BusSelector primitives.

BusCreator(field_names) packs n inputs into a single output that is a
``collections.namedtuple``-typed pytree (one field per input port).
BusSelector(field_name) pulls one named field out of a bus signal.

The bus value is a NamedTuple, which is JAX-pytree-friendly out of the
box: it survives ``jit`` / ``vmap`` / ``grad`` without any extra
``register_pytree_node`` boilerplate. These tests exercise:

  * Round-trip identity ``BusSelector(BusCreator(a,b,c), "a") == a``
    via ``simulate``.
  * Selection of every named field from a 3-field bus.
  * Pure-JAX trace through ``jax.jit`` and ``jax.grad`` of the
    underlying ops (the constructor + ``getattr``).
  * Construction-time validation (empty / duplicate / non-identifier
    field names).

The full ``BusType`` registry, per-field units, and nested
buses are out of scope for this test module.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import jaxonomy
from jaxonomy import library
from jaxonomy.library import BusCreator, BusSelector
from jaxonomy.testing.markers import skip_if_not_jax

skip_if_not_jax()


# ---------------------------------------------------------------------------
# Construction-time validation. These are pure-Python paths that run before
# any JAX trace, so they do not need a Diagram.
# ---------------------------------------------------------------------------


def test_bus_creator_empty_field_names_raises():
    with pytest.raises(ValueError, match="at least one field name"):
        BusCreator(())


def test_bus_creator_duplicate_field_names_raises():
    with pytest.raises(ValueError, match="must be unique"):
        BusCreator(("a", "b", "a"))


def test_bus_creator_invalid_identifier_raises():
    with pytest.raises(ValueError, match="not a valid Python identifier"):
        BusCreator(("a", "b c"))


def test_bus_selector_invalid_identifier_raises():
    with pytest.raises(ValueError, match="not a valid Python identifier"):
        BusSelector("not a name")


# ---------------------------------------------------------------------------
# End-to-end ``simulate`` tests. These confirm the NamedTuple-typed bus
# signal flows through every stage of the simulator without tripping
# pytree handling.
# ---------------------------------------------------------------------------


def _build_roundtrip_diagram(field_names, values, select):
    """Build BusCreator -> BusSelector with one Constant per field.

    Returns ``(diagram, selector)`` ready for ``simulate``.
    """
    sources = [library.Constant(v) for v in values]
    creator = BusCreator(field_names, name="creator")
    selector = BusSelector(select, name="selector")

    builder = jaxonomy.DiagramBuilder()
    builder.add(*sources, creator, selector)
    for src, port in zip(sources, creator.input_ports):
        builder.connect(src.output_ports[0], port)
    builder.connect(creator.output_ports[0], selector.input_ports[0])
    diagram = builder.build()
    return diagram, selector


def test_bus_roundtrip_selects_each_field():
    """For every field, BusSelector pulls back the corresponding input."""
    field_names = ("position", "velocity", "acceleration")
    values = (1.0, 2.0, 3.0)

    for fname, expected in zip(field_names, values):
        diagram, selector = _build_roundtrip_diagram(field_names, values, fname)
        ctx = diagram.create_context()
        results = jaxonomy.simulate(
            diagram, ctx, (0.0, 0.1),
            recorded_signals={"out": selector.output_ports[0]},
        )
        out = float(np.asarray(results.outputs["out"])[-1])
        np.testing.assert_allclose(out, expected)


def test_bus_creator_field_names_property():
    """The ``field_names`` and ``bus_type`` properties expose the schema."""
    creator = BusCreator(("alpha", "beta"))
    assert creator.field_names == ("alpha", "beta")
    assert creator.bus_type._fields == ("alpha", "beta")
    # Confirm the underlying type is a real NamedTuple — the test that it
    # is a JAX pytree is implicit in the simulate() round-trip above.
    bus = creator.bus_type(1.0, 2.0)
    assert bus.alpha == 1.0
    assert bus.beta == 2.0


def test_bus_selector_field_name_property():
    sel = BusSelector("velocity")
    assert sel.field_name == "velocity"


# ---------------------------------------------------------------------------
# JAX trace-level tests. We exercise the underlying NamedTuple
# construction + getattr path under ``jit`` and ``grad`` without
# spinning up the simulator — this isolates "the bus signal flows
# through JAX cleanly" from any simulator-specific machinery.
# ---------------------------------------------------------------------------


def test_bus_signal_traces_through_jit():
    """A NamedTuple-shaped bus survives jax.jit unchanged."""
    creator = BusCreator(("x", "y", "z"))
    Bus = creator.bus_type

    @jax.jit
    def make_and_select(a, b, c):
        bus = Bus(a, b, c)
        return bus.y

    out = make_and_select(jnp.asarray(1.0), jnp.asarray(2.0), jnp.asarray(3.0))
    np.testing.assert_allclose(float(out), 2.0)


def test_bus_signal_differentiates_through_each_field():
    """Gradient flows from BusSelector output back to every BusCreator input.

    f(a, b, c) = bus.a * 4 + bus.b * 5 + bus.c * 6 (after pack + select)
    expected gradient: (4, 5, 6).
    """
    creator = BusCreator(("a", "b", "c"))
    Bus = creator.bus_type

    def f(a, b, c):
        bus = Bus(a, b, c)
        return getattr(bus, "a") * 4.0 + getattr(bus, "b") * 5.0 + getattr(bus, "c") * 6.0

    g_a, g_b, g_c = jax.grad(f, argnums=(0, 1, 2))(1.0, 2.0, 3.0)
    np.testing.assert_allclose(float(g_a), 4.0)
    np.testing.assert_allclose(float(g_b), 5.0)
    np.testing.assert_allclose(float(g_c), 6.0)


def test_bus_signal_is_a_pytree():
    """Confirm the bus type is recognized by jax.tree_util.

    NamedTuples are registered automatically; this test pins that we
    have not accidentally subclassed in a way that breaks pytree
    registration.
    """
    creator = BusCreator(("p", "q"))
    Bus = creator.bus_type
    bus = Bus(jnp.asarray(1.0), jnp.asarray(2.0))
    leaves = jax.tree_util.tree_leaves(bus)
    assert len(leaves) == 2
    np.testing.assert_allclose(float(leaves[0]), 1.0)
    np.testing.assert_allclose(float(leaves[1]), 2.0)


def test_bus_creator_underlying_type_is_namedtuple():
    """BusCreator.bus_type is a real NamedTuple subclass.

    We deliberately do NOT pass ``default_value=`` to
    ``declare_output_port`` (see the implementation note in
    primitives.py — leaf_system.py calls ``npa.array`` on the default,
    which would flatten our NamedTuple into a 1-D array). Instead the
    framework lazily evaluates ``_compute_bus`` on a dummy context,
    yielding the correct NamedTuple type at run time. This test pins
    that ``bus_type`` is itself a NamedTuple — the runtime behavior is
    covered by the simulate() round-trip and the JIT tests below.
    """
    creator = BusCreator(("p", "q", "r"))
    Bus = creator.bus_type
    assert issubclass(Bus, tuple)
    assert Bus._fields == ("p", "q", "r")
    bus = Bus(1.0, 2.0, 3.0)
    assert bus.p == 1.0 and bus.q == 2.0 and bus.r == 3.0
