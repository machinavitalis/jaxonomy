# SPDX-License-Identifier: MIT
"""T-117-followup-bus-units: per-field unit propagation through
``BusCreator`` / ``BusSelector``.

The T-117-followup-bus-namedtuple bus blocks pack signals into a
NamedTuple-typed pytree but originally dropped each input port's
``units=`` attribute on the floor: the output bus carried no unit
metadata and downstream :class:`BusSelector` blocks could not learn
the unit of the selected field at connect time.

This followup wires units through buses:

  * :class:`BusCreator` accepts an optional ``field_units`` mapping;
    when present each input port is declared with the matching unit
    and the output bus port carries a :class:`BusUnit` describing
    the whole compound signal.
  * :class:`BusSelector` accepts an optional ``bus_unit`` mapping;
    when present the input port is tagged with the full ``BusUnit``
    (so the connect-time check verifies the upstream bus is
    compatible) and the output port carries the scalar unit of the
    selected field.
  * The connect-time consistency check in
    :class:`DiagramBuilder.connect` is extended to compare two
    :class:`BusUnit`s field-by-field via
    :func:`assert_unit_compatible`.

When ``field_units`` / ``bus_unit`` are omitted (the default), behaviour
is byte-equivalent to T-117-fu-bus-namedtuple — verified by the
existing ``test_t_117_followup_bus.py`` suite.
"""

from __future__ import annotations

import numpy as np
import pytest

import jaxonomy
from jaxonomy import library
from jaxonomy.framework.units import (
    BusUnit,
    UnitMismatchError,
    Unit as _Unit,
    ampere,
    meter,
    second,
)
from jaxonomy.library import BusCreator, BusSelector
from jaxonomy.testing.markers import skip_if_not_jax

# ``volt`` is not in the SI-base export list — synthesise it locally so
# the test file is self-contained without touching the curated
# ``units.__all__``.
volt = _Unit(dims=(1, 2, -3, -1, 0, 0, 0), name="V")  # kg*m^2*s^-3*A^-1

skip_if_not_jax()


# ---------------------------------------------------------------------------
# Construction: BusCreator with field_units propagates units to ports.
# ---------------------------------------------------------------------------


def test_bus_creator_field_units_tags_input_ports():
    """Each input port gets the unit listed under its field name."""
    creator = BusCreator(("v", "i"), field_units={"v": volt, "i": ampere})
    # Port order matches field_names declaration order.
    assert creator.input_ports[0].units == volt
    assert creator.input_ports[1].units == ampere


def test_bus_creator_field_units_tags_output_port_as_busunit():
    """The output bus port carries a BusUnit describing the schema."""
    creator = BusCreator(("v", "i"), field_units={"v": volt, "i": ampere})
    bu = creator.output_ports[0].units
    assert isinstance(bu, BusUnit)
    assert bu.fields == {"v": volt, "i": ampere}
    assert creator.bus_unit is bu


def test_bus_creator_field_units_none_is_byte_equivalent():
    """No ``field_units`` argument leaves ports unit-less (legacy path)."""
    creator = BusCreator(("v", "i"))
    assert creator.input_ports[0].units is None
    assert creator.input_ports[1].units is None
    assert creator.output_ports[0].units is None
    assert creator.bus_unit is None


def test_bus_creator_field_units_mismatched_keys_raises():
    """Extra / missing keys in ``field_units`` are caught at construction."""
    with pytest.raises(ValueError, match="must match field_names"):
        BusCreator(("v", "i"), field_units={"v": volt})  # missing "i"
    with pytest.raises(ValueError, match="must match field_names"):
        BusCreator(
            ("v", "i"),
            field_units={"v": volt, "i": ampere, "extra": meter},
        )


def test_bus_creator_field_units_non_unit_value_raises():
    """Non-Unit values in ``field_units`` are rejected with a clear error."""
    with pytest.raises(TypeError, match="must be a Unit instance"):
        BusCreator(("v",), field_units={"v": "volts"})


# ---------------------------------------------------------------------------
# Construction: BusSelector with bus_unit learns its output unit.
# ---------------------------------------------------------------------------


def test_bus_selector_with_bus_unit_declares_field_output_unit():
    """The selector output unit is the bus_unit field for its name."""
    bu = BusUnit(fields={"v": volt, "i": ampere})
    sel_v = BusSelector("v", bus_unit=bu)
    sel_i = BusSelector("i", bus_unit=bu)
    assert sel_v.output_ports[0].units == volt
    assert sel_i.output_ports[0].units == ampere
    # Input port carries the full BusUnit so connect-time check works.
    assert sel_v.input_ports[0].units == bu


def test_bus_selector_without_bus_unit_has_no_output_unit():
    """Default (no bus_unit) keeps output port unit-less (legacy)."""
    sel = BusSelector("v")
    assert sel.output_ports[0].units is None
    assert sel.input_ports[0].units is None
    assert sel.bus_unit is None


def test_bus_selector_bus_unit_missing_field_raises():
    """Selecting a field not present in the supplied bus_unit errors out."""
    bu = BusUnit(fields={"v": volt})
    with pytest.raises(ValueError, match="not present in bus_unit fields"):
        BusSelector("i", bus_unit=bu)


def test_bus_selector_bus_unit_wrong_type_raises():
    with pytest.raises(TypeError, match="must be a BusUnit"):
        BusSelector("v", bus_unit=volt)  # passed a Unit instead of BusUnit


# ---------------------------------------------------------------------------
# BusUnit value-type semantics (equality, hashing, repr, validation).
# ---------------------------------------------------------------------------


def test_bus_unit_equality_ignores_field_order():
    a = BusUnit(fields={"v": volt, "i": ampere})
    b = BusUnit(fields={"i": ampere, "v": volt})
    assert a == b
    assert hash(a) == hash(b)


def test_bus_unit_inequality_on_field_set_mismatch():
    assert BusUnit({"v": volt}) != BusUnit({"i": ampere})
    assert BusUnit({"v": volt, "i": ampere}) != BusUnit({"v": volt})


def test_bus_unit_inequality_on_per_field_unit_mismatch():
    assert BusUnit({"v": volt}) != BusUnit({"v": meter})


def test_bus_unit_validates_fields_at_construction():
    with pytest.raises(TypeError, match="must be a str"):
        BusUnit(fields={123: volt})
    with pytest.raises(TypeError, match="must map to a Unit instance"):
        BusUnit(fields={"v": "volt"})


def test_bus_unit_field_unit_lookup():
    bu = BusUnit({"v": volt, "i": ampere})
    assert bu.field_unit("v") == volt
    assert bu.field_unit("missing") is None


# ---------------------------------------------------------------------------
# Connect-time consistency: BusUnit propagation through DiagramBuilder.
# ---------------------------------------------------------------------------


def _ports_under_default_builder(unit_conversion="auto"):
    """Helper: build a (Constant, BusCreator, BusSelector) wiring and
    return the partly-built builder.  Callers add the final
    ``connect`` themselves and inspect the result.
    """
    return jaxonomy.DiagramBuilder(unit_conversion=unit_conversion)


def test_bus_creator_input_port_unit_mismatch_raises_at_connect():
    """Connecting a meter-port to a volt-typed BusCreator slot errors out."""
    src_bad = library.Constant(1.0)
    src_bad.output_ports[0].units = meter  # tag as metres
    src_good = library.Constant(2.0)
    creator = BusCreator(("v", "i"), field_units={"v": volt, "i": ampere})

    builder = _ports_under_default_builder()
    builder.add(src_bad, src_good, creator)
    # 0th creator input expects volt; we're handing it metres.
    with pytest.raises(UnitMismatchError):
        builder.connect(src_bad.output_ports[0], creator.input_ports[0])


def test_bus_creator_to_bus_selector_connect_passes_with_matching_busunit():
    """Matched BusUnits flow through without error and the round-trip
    selector output carries the right scalar unit."""
    src_v = library.Constant(3.0)
    src_i = library.Constant(5.0)
    src_v.output_ports[0].units = volt
    src_i.output_ports[0].units = ampere

    bus_units = {"v": volt, "i": ampere}
    creator = BusCreator(("v", "i"), field_units=bus_units, name="creator")
    sel = BusSelector(
        "v", bus_unit=BusUnit(fields=bus_units), name="selector"
    )

    builder = _ports_under_default_builder()
    builder.add(src_v, src_i, creator, sel)
    builder.connect(src_v.output_ports[0], creator.input_ports[0])
    builder.connect(src_i.output_ports[0], creator.input_ports[1])
    # The actual BusUnit -> BusUnit connect under test:
    builder.connect(creator.output_ports[0], sel.input_ports[0])
    diagram = builder.build()

    # Selector output is tagged volt (the field's scalar unit).
    assert sel.output_ports[0].units == volt

    # End-to-end simulate: BusSelector("v") returns the v-source value.
    ctx = diagram.create_context()
    results = jaxonomy.simulate(
        diagram, ctx, (0.0, 0.1),
        recorded_signals={"v_out": sel.output_ports[0]},
    )
    out = float(np.asarray(results.outputs["v_out"])[-1])
    np.testing.assert_allclose(out, 3.0)


def test_bus_creator_to_bus_selector_connect_mismatched_busunit_raises():
    """A BusSelector tagged with a different BusUnit schema fails to
    connect to the creator's output."""
    creator = BusCreator(
        ("v", "i"), field_units={"v": volt, "i": ampere}, name="creator"
    )
    # Selector declares the bus has units (volt, second) — incompatible.
    sel = BusSelector(
        "v",
        bus_unit=BusUnit({"v": volt, "i": second}),
        name="selector",
    )

    builder = _ports_under_default_builder()
    builder.add(creator, sel)
    with pytest.raises(UnitMismatchError):
        builder.connect(creator.output_ports[0], sel.input_ports[0])


def test_bus_creator_to_bus_selector_field_set_mismatch_raises():
    """BusUnits with different field-name sets are incompatible."""
    creator = BusCreator(
        ("v", "i"), field_units={"v": volt, "i": ampere}, name="creator"
    )
    sel = BusSelector(
        "v",
        bus_unit=BusUnit({"v": volt}),  # missing "i"
        name="selector",
    )
    builder = _ports_under_default_builder()
    builder.add(creator, sel)
    with pytest.raises(UnitMismatchError):
        builder.connect(creator.output_ports[0], sel.input_ports[0])


def test_bus_unit_wildcards_against_none_on_other_side():
    """Tagged BusCreator connecting into an untagged BusSelector is OK
    (default-off byte-equivalence with the unit-less bus path)."""
    creator = BusCreator(
        ("v", "i"), field_units={"v": volt, "i": ampere}, name="creator"
    )
    sel = BusSelector("v", name="selector")  # no bus_unit
    builder = _ports_under_default_builder()
    builder.add(creator, sel)
    # Should not raise — None on the input wildcards the BusUnit.
    builder.connect(creator.output_ports[0], sel.input_ports[0])


def test_bus_unit_to_scalar_unit_rejected():
    """BusUnit on one side and a plain scalar Unit on the other side
    is unambiguously a wiring bug; the connect-time check rejects it."""
    creator = BusCreator(
        ("v", "i"), field_units={"v": volt, "i": ampere}, name="creator"
    )
    # Manually fabricate an input port tagged with a scalar Unit.
    sink = library.IOPort(name="sink")
    sink.input_ports[0].units = volt  # scalar — not a BusUnit
    builder = _ports_under_default_builder()
    builder.add(creator, sink)
    with pytest.raises(UnitMismatchError):
        builder.connect(creator.output_ports[0], sink.input_ports[0])


# ---------------------------------------------------------------------------
# Strict unit_conversion="error" mode also handles BusUnit.
# ---------------------------------------------------------------------------


def test_busunit_strict_error_mode_still_works():
    """The Phase-1 strict-equal path (unit_conversion='error') also
    routes BusUnit comparisons through assert_unit_compatible."""
    creator = BusCreator(
        ("v", "i"), field_units={"v": volt, "i": ampere}, name="creator"
    )
    sel = BusSelector(
        "v", bus_unit=BusUnit({"v": volt, "i": ampere}), name="selector"
    )
    builder = jaxonomy.DiagramBuilder(unit_conversion="error")
    builder.add(creator, sel)
    builder.connect(creator.output_ports[0], sel.input_ports[0])
    # No exception: matched BusUnits pass strict-mode check too.

    creator2 = BusCreator(
        ("v",), field_units={"v": volt}, name="creator2"
    )
    sel2 = BusSelector(
        "v", bus_unit=BusUnit({"v": meter}), name="selector2"
    )
    builder2 = jaxonomy.DiagramBuilder(unit_conversion="error")
    builder2.add(creator2, sel2)
    with pytest.raises(UnitMismatchError):
        builder2.connect(creator2.output_ports[0], sel2.input_ports[0])
