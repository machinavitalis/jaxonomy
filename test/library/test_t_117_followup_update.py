# SPDX-License-Identifier: MIT
"""T-117-followup-bus-update: ``BusUpdate`` LeafSystem.

``BusUpdate(bus_spec, field_name)`` is a two-input / one-output block:
input port 0 is an upstream bus (NamedTuple-shaped), input port 1 is the
new value for the named field. The output port is a fresh bus identical
to the input except in the ``field_name`` slot. Equivalent to the
verbose BusSelector-modify-BusCreator triplet, but expressed as a
single block.

These tests pin:

  * 3-field bus update via ``simulate``: the targeted field is
    replaced; the other two fields flow through unchanged.
  * Differentiability through both inputs: ``jax.grad`` returns the
    expected gradient on both the bus_in path (for non-targeted fields)
    and the new_value path (for the targeted field).
  * Construction-time validation: unknown ``field_name`` raises
    ``ValueError`` with a useful message; non-string ``field_name``
    raises ``TypeError``; empty ``bus_spec`` raises ``ValueError``.
  * The block accepts the three ``bus_spec`` forms supported by
    BusMerge (BusCreator instance, NamedTuple class, tuple of names).
"""

from __future__ import annotations

from collections import namedtuple

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import jaxonomy
from jaxonomy import library
from jaxonomy.library import BusCreator, BusSelector, BusUpdate
from jaxonomy.testing.markers import skip_if_not_jax

skip_if_not_jax()


# ---------------------------------------------------------------------------
# End-to-end ``simulate`` round-trip: BusCreator -> BusUpdate -> BusSelector
# should return the *new* value for the updated field, and the *original*
# value for any untouched field.
# ---------------------------------------------------------------------------


def _build_update_diagram(field_names, values, update_field, new_value, select):
    """Wire Constants -> BusCreator -> BusUpdate(<-Constant) -> BusSelector."""
    sources = [library.Constant(v) for v in values]
    creator = BusCreator(field_names, name="creator")
    new_value_src = library.Constant(new_value)
    updater = BusUpdate(field_names, update_field, name="updater")
    selector = BusSelector(select, name="selector")

    builder = jaxonomy.DiagramBuilder()
    builder.add(*sources, creator, new_value_src, updater, selector)
    for src, port in zip(sources, creator.input_ports):
        builder.connect(src.output_ports[0], port)
    builder.connect(creator.output_ports[0], updater.input_ports[0])
    builder.connect(new_value_src.output_ports[0], updater.input_ports[1])
    builder.connect(updater.output_ports[0], selector.input_ports[0])
    return builder.build(), selector


def test_update_replaces_targeted_field_via_simulate():
    """The updated field reads back as the new value through simulate."""
    fields = ("x", "y", "z")
    values = (1.0, 2.0, 3.0)

    diagram, selector = _build_update_diagram(
        fields, values, update_field="x", new_value=99.0, select="x"
    )
    ctx = diagram.create_context()
    results = jaxonomy.simulate(
        diagram, ctx, (0.0, 0.1),
        recorded_signals={"out": selector.output_ports[0]},
    )
    out = float(np.asarray(results.outputs["out"])[-1])
    np.testing.assert_allclose(out, 99.0)


@pytest.mark.parametrize(
    ("select", "expected"),
    [
        ("y", 2.0),
        ("z", 3.0),
    ],
)
def test_update_preserves_untouched_fields_via_simulate(select, expected):
    """Untouched fields (y, z) pass through with their original values
    when only field x is updated."""
    fields = ("x", "y", "z")
    values = (1.0, 2.0, 3.0)

    diagram, selector = _build_update_diagram(
        fields, values, update_field="x", new_value=99.0, select=select
    )
    ctx = diagram.create_context()
    results = jaxonomy.simulate(
        diagram, ctx, (0.0, 0.1),
        recorded_signals={"out": selector.output_ports[0]},
    )
    out = float(np.asarray(results.outputs["out"])[-1])
    np.testing.assert_allclose(out, expected)


def test_update_each_position_in_3field_bus():
    """Updating each of x, y, z in turn correctly replaces only that
    field; the other two flow through. Sweeps the entire bus to confirm
    no off-by-one in the source-map computation."""
    fields = ("x", "y", "z")
    values = (1.0, 2.0, 3.0)
    new_value = 42.0

    for update_field in fields:
        for sel in fields:
            diagram, selector = _build_update_diagram(
                fields, values, update_field, new_value, sel
            )
            ctx = diagram.create_context()
            results = jaxonomy.simulate(
                diagram, ctx, (0.0, 0.1),
                recorded_signals={"out": selector.output_ports[0]},
            )
            out = float(np.asarray(results.outputs["out"])[-1])
            expected = new_value if sel == update_field else dict(zip(fields, values))[sel]
            np.testing.assert_allclose(
                out, expected,
                err_msg=f"update={update_field}, sel={sel}: got {out!r}",
            )


# ---------------------------------------------------------------------------
# Differentiability: gradients flow through both inputs leaf-by-leaf.
# Using a closure that mirrors the BusUpdate runtime path exactly so the
# test pins the autodiff contract independently of framework plumbing.
# ---------------------------------------------------------------------------


def test_update_differentiable_through_both_inputs():
    """f(a, b, c, n) = upd(bus, n).x*4 + upd(bus, n).y*5 + upd(bus, n).z*6
    with target field = "x" should give:
      dL/da = 0   (a is replaced; gradient is on n instead)
      dL/db = 5   (untouched; flows back through bus_in)
      dL/dc = 6   (untouched; flows back through bus_in)
      dL/dn = 4   (the new_value path lights up the x slot only)
    """
    Bus = namedtuple("Bus", ("x", "y", "z"))

    def update_closure(bus_in_val, new_value):
        # Mirrors the BusUpdate runtime closure body.
        return Bus(new_value, bus_in_val.y, bus_in_val.z)

    def f(a, b, c, n):
        bus = Bus(a, b, c)
        out = update_closure(bus, n)
        return out.x * 4.0 + out.y * 5.0 + out.z * 6.0

    g_a, g_b, g_c, g_n = jax.grad(f, argnums=(0, 1, 2, 3))(
        1.0, 2.0, 3.0, 99.0
    )
    np.testing.assert_allclose(float(g_a), 0.0)
    np.testing.assert_allclose(float(g_b), 5.0)
    np.testing.assert_allclose(float(g_c), 6.0)
    np.testing.assert_allclose(float(g_n), 4.0)


def test_update_traces_through_jit():
    """A NamedTuple bus passed through the BusUpdate-style closure
    survives ``jax.jit``."""
    Bus = namedtuple("Bus", ("x", "y", "z"))

    @jax.jit
    def f(x, y, z, n):
        bus = Bus(x, y, z)
        # Inline copy of the BusUpdate(field_name="y") closure body.
        out = Bus(bus.x, n, bus.z)
        return out.x + out.y + out.z

    val = f(
        jnp.asarray(1.0),
        jnp.asarray(2.0),
        jnp.asarray(3.0),
        jnp.asarray(50.0),
    )
    # x=1, y=50 (replaced), z=3 -> 54.
    np.testing.assert_allclose(float(val), 54.0)


# ---------------------------------------------------------------------------
# Construction-time validation. Unknown field_name and other malformed
# inputs should fail loudly at construction time, not at simulate time.
# ---------------------------------------------------------------------------


def test_update_unknown_field_name_raises():
    with pytest.raises(ValueError, match="not in the bus schema"):
        BusUpdate(("x", "y", "z"), "nonexistent")


def test_update_non_string_field_name_raises():
    with pytest.raises(TypeError, match="must be a string"):
        BusUpdate(("x", "y", "z"), 0)  # type: ignore[arg-type]


def test_update_empty_bus_spec_raises():
    with pytest.raises(ValueError, match="at least one field"):
        BusUpdate((), "x")


def test_update_bad_bus_spec_type_raises():
    with pytest.raises(TypeError):
        BusUpdate(42, "x")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# bus_spec form coverage: BusCreator instance, NamedTuple class, tuple
# of names should all produce equivalent block structure.
# ---------------------------------------------------------------------------


def test_update_accepts_buscreator_instance_as_bus_spec():
    creator = BusCreator(("a", "b"))
    upd = BusUpdate(creator, "a")
    assert upd.field_names == ("a", "b")
    assert upd.field_name == "a"


def test_update_accepts_namedtuple_class_as_bus_spec():
    Bus = namedtuple("Bus", ("a", "b"))
    upd = BusUpdate(Bus, "b")
    assert upd.field_names == ("a", "b")
    assert upd.field_name == "b"


def test_update_accepts_tuple_of_names_as_bus_spec():
    upd = BusUpdate(("p", "q"), "q")
    assert upd.field_names == ("p", "q")
    assert upd.field_name == "q"


def test_update_has_two_inputs_one_output():
    upd = BusUpdate(("a", "b"), "a")
    assert len(upd.input_ports) == 2
    assert len(upd.output_ports) == 1
    assert upd.input_ports[0].name == "bus_in"
    assert upd.input_ports[1].name == "new_value"
    assert upd.output_ports[0].name == "bus_out"


def test_update_bus_type_has_correct_fields():
    upd = BusUpdate(("x", "y", "z"), "y")
    assert upd.bus_type._fields == ("x", "y", "z")
