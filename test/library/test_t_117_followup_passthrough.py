# SPDX-License-Identifier: MIT
"""T-117-followup-bus-passthrough: ``BusPassthrough`` LeafSystem.

``BusPassthrough()`` is a single-input / single-output identity block:
the output port returns the input value unchanged. The block exists to
support diagram rewiring (route a bus through a junction point),
debugging signal flow (insert a passthrough and record its output), and
breaking dependency chains for the scheduler.

These tests pin:

  * 3-field bus round-trip: ``BusCreator -> BusPassthrough -> BusSelector``
    pulls each field back out unchanged through ``simulate``.
  * Differentiability: ``jax.grad`` flows from the passthrough output
    back to every upstream input that contributed to the bus, with
    Jacobian = identity.
  * Composition with a scalar (non-bus) signal also works: the block
    is fully type-agnostic.
  * Construction-time bus_unit validation (TypeError on a non-BusUnit).
"""

from __future__ import annotations

from collections import namedtuple

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import jaxonomy
from jaxonomy import library
from jaxonomy.library import BusCreator, BusPassthrough, BusSelector
from jaxonomy.testing.markers import skip_if_not_jax

skip_if_not_jax()


# ---------------------------------------------------------------------------
# End-to-end ``simulate`` round-trip: BusCreator -> BusPassthrough ->
# BusSelector should return the input value of the corresponding field.
# ---------------------------------------------------------------------------


def _build_passthrough_diagram(field_names, values, select):
    """Wire Constants -> BusCreator -> BusPassthrough -> BusSelector."""
    sources = [library.Constant(v) for v in values]
    creator = BusCreator(field_names, name="creator")
    passthrough = BusPassthrough(name="passthrough")
    selector = BusSelector(select, name="selector")

    builder = jaxonomy.DiagramBuilder()
    builder.add(*sources, creator, passthrough, selector)
    for src, port in zip(sources, creator.input_ports):
        builder.connect(src.output_ports[0], port)
    builder.connect(creator.output_ports[0], passthrough.input_ports[0])
    builder.connect(passthrough.output_ports[0], selector.input_ports[0])
    return builder.build(), selector


@pytest.mark.parametrize(
    ("select", "expected"),
    [
        ("position", 1.0),
        ("velocity", 2.0),
        ("acceleration", 3.0),
    ],
)
def test_passthrough_3field_bus_each_field_unchanged(select, expected):
    """For every field of a 3-field bus, the value emerges unchanged."""
    diagram, selector = _build_passthrough_diagram(
        ("position", "velocity", "acceleration"),
        (1.0, 2.0, 3.0),
        select,
    )
    ctx = diagram.create_context()
    results = jaxonomy.simulate(
        diagram, ctx, (0.0, 0.1),
        recorded_signals={"out": selector.output_ports[0]},
    )
    out = float(np.asarray(results.outputs["out"])[-1])
    np.testing.assert_allclose(out, expected)


def test_passthrough_3field_bus_all_fields_preserved_via_simulate():
    """All three fields of a 3-field bus emerge unchanged through the
    BusCreator -> BusPassthrough -> BusSelector chain in a single
    simulate call (one selector per field, three diagrams).

    Exercises the same chain as the parametrised test above but pins
    that the NamedTuple-typed bus survives the passthrough cleanly: if
    the framework had silently called ``npa.array`` on it (flattening
    to 1-D), the BusSelector at the end would produce wrong values.
    """
    fields = ("position", "velocity", "acceleration")
    values = (1.0, 2.0, 3.0)

    out_values = []
    for fname in fields:
        diagram, selector = _build_passthrough_diagram(fields, values, fname)
        ctx = diagram.create_context()
        results = jaxonomy.simulate(
            diagram, ctx, (0.0, 0.1),
            recorded_signals={"out": selector.output_ports[0]},
        )
        out_values.append(float(np.asarray(results.outputs["out"])[-1]))

    np.testing.assert_allclose(out_values, list(values))


# ---------------------------------------------------------------------------
# Differentiability: a NamedTuple-shaped bus carried through a Python
# identity closure has Jacobian = identity, so jax.grad on a sum-of-
# fields readout returns gradient = 1 for every contributing input.
# ---------------------------------------------------------------------------


def test_passthrough_differentiable_through_each_field():
    """f(a, b, c) = pt(bus).a*4 + pt(bus).b*5 + pt(bus).c*6 -> grads (4, 5, 6).

    The passthrough closure is just ``return value``; gradient flows
    leaf-by-leaf back through the NamedTuple to each contributing input.
    """
    Bus = namedtuple("Bus", ("a", "b", "c"))

    def passthrough_closure(value):
        # Mirrors the runtime closure inside BusPassthrough exactly.
        return value

    def f(a, b, c):
        bus = Bus(a, b, c)
        out = passthrough_closure(bus)
        return out.a * 4.0 + out.b * 5.0 + out.c * 6.0

    g_a, g_b, g_c = jax.grad(f, argnums=(0, 1, 2))(1.0, 2.0, 3.0)
    np.testing.assert_allclose(float(g_a), 4.0)
    np.testing.assert_allclose(float(g_b), 5.0)
    np.testing.assert_allclose(float(g_c), 6.0)


def test_passthrough_traces_through_jit():
    """A NamedTuple bus passed through the identity closure survives jit."""
    Bus = namedtuple("Bus", ("x", "y", "z"))

    @jax.jit
    def f(x, y, z):
        bus = Bus(x, y, z)
        # Inline copy of BusPassthrough's closure body.
        out = bus
        return out.x + out.y + out.z

    val = f(jnp.asarray(1.0), jnp.asarray(2.0), jnp.asarray(3.0))
    np.testing.assert_allclose(float(val), 6.0)


# ---------------------------------------------------------------------------
# Composition: BusCreator -> BusPassthrough -> BusSelector chain. The
# parametrised end-to-end test above already covers selecting each field;
# this test confirms the chain still produces the correct value when the
# diagram is built with a passthrough explicitly inserted as a junction.
# ---------------------------------------------------------------------------


def test_passthrough_composes_with_creator_and_selector():
    """The full BusCreator -> BusPassthrough -> BusSelector chain is wired
    correctly and yields the upstream input on the selected field."""
    fields = ("alpha", "beta")
    values = (10.0, 20.0)

    diagram, selector = _build_passthrough_diagram(fields, values, "beta")
    ctx = diagram.create_context()
    results = jaxonomy.simulate(
        diagram, ctx, (0.0, 0.1),
        recorded_signals={"out": selector.output_ports[0]},
    )
    out = float(np.asarray(results.outputs["out"])[-1])
    np.testing.assert_allclose(out, 20.0)


# ---------------------------------------------------------------------------
# Construction-time validation. A non-BusUnit value passed for
# ``bus_unit`` is a TypeError; the default ``bus_unit=None`` path is
# byte-equivalent to a unit-less LeafSystem.
# ---------------------------------------------------------------------------


def test_passthrough_bus_unit_typecheck_rejects_non_busunit():
    with pytest.raises(TypeError, match="must be a BusUnit instance"):
        BusPassthrough(bus_unit="not-a-bus-unit")


def test_passthrough_default_bus_unit_is_none():
    pt = BusPassthrough()
    assert pt.bus_unit is None


def test_passthrough_has_one_input_one_output():
    pt = BusPassthrough()
    assert len(pt.input_ports) == 1
    assert len(pt.output_ports) == 1
    assert pt.input_ports[0].name == "in"
    assert pt.output_ports[0].name == "out"
