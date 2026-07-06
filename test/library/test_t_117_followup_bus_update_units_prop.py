# SPDX-License-Identifier: MIT

"""Tests for T-117-followup-bus-update-units-prop.

Pre-fix, :class:`BusUpdate` dropped :class:`BusUnit` field metadata
when constructing the output bus — every closed-loop topology that
mutated a bus mid-stream lost the unit-consistency guarantees
T-117-followup-bus-units provides everywhere else. The pre-fix code
had an explicit "do not propagate units here" comment with a deferred
follow-up.

Post-fix, :class:`BusUpdate` accepts an optional ``bus_unit=`` kwarg
mirroring :class:`BusSelector` and :class:`BusPassthrough`:

* The ``bus_in`` input port advertises the full BusUnit so the
  connect-time check verifies the upstream BusCreator produced a
  compatible schema.
* The ``new_value`` input port advertises ``bus_unit.fields[
  field_name]`` so the connect-time check enforces unit compatibility
  on the replacement value (composes with T-104 Phase 2 behaviour on
  Sum / Product / Integrator).
* The ``bus_out`` output port re-exports the same BusUnit so
  downstream consumers see the schema preserved through the update.

Tests:
* Default-off byte-equivalence (no ``bus_unit=`` kwarg, no behaviour
  change).
* Unit propagation: input + output ports advertise the BusUnit;
  ``new_value`` port advertises the per-field unit.
* Constructor rejects a ``bus_unit`` whose field set does not match
  the bus schema (with a clear message).
* Constructor rejects non-:class:`BusUnit` types passed to ``bus_unit``.
* End-to-end ``simulate`` with a BusUnit-tagged bus produces the
  same numerical output as the legacy unit-less path.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

import jaxonomy
from jaxonomy import library
from jaxonomy.framework.units import BusUnit, Unit


pytestmark = pytest.mark.minimal


_VOLT = Unit(dims=(1, 2, -3, -1, 0, 0, 0), name="V")
_AMP = Unit(dims=(0, 0, 0, 1, 0, 0, 0), name="A")


# ---------------------------------------------------------------------
# Default-off byte-equivalence
# ---------------------------------------------------------------------


class TestDefaultOff:
    def test_legacy_construction_unchanged(self):
        """``BusUpdate(spec, field)`` without ``bus_unit`` is unchanged."""
        blk = library.BusUpdate(("v", "i"), "v")
        assert blk.bus_unit is None
        assert blk.input_ports[0].units is None
        assert blk.input_ports[1].units is None
        assert blk.output_ports[0].units is None


# ---------------------------------------------------------------------
# Unit propagation
# ---------------------------------------------------------------------


class TestUnitPropagation:
    def test_bus_unit_applied_to_input_and_output_ports(self):
        bu = BusUnit(fields={"v": _VOLT, "i": _AMP})
        blk = library.BusUpdate(("v", "i"), "v", bus_unit=bu)
        assert blk.bus_unit is bu
        assert blk.input_ports[0].units == bu, (
            "bus_in port should advertise the full BusUnit so the "
            "connect-time check verifies the upstream schema."
        )
        assert blk.output_ports[0].units == bu, (
            "bus_out port should re-export the BusUnit so downstream "
            "consumers see the schema preserved through the update."
        )

    def test_new_value_port_carries_per_field_unit(self):
        bu = BusUnit(fields={"v": _VOLT, "i": _AMP})
        blk_v = library.BusUpdate(("v", "i"), "v", bus_unit=bu)
        blk_i = library.BusUpdate(("v", "i"), "i", bus_unit=bu)
        assert blk_v.input_ports[1].units == _VOLT
        assert blk_i.input_ports[1].units == _AMP


# ---------------------------------------------------------------------
# Construction-time validation
# ---------------------------------------------------------------------


class TestConstructionValidation:
    def test_non_bus_unit_value_rejected(self):
        with pytest.raises(TypeError, match="bus_unit must be a BusUnit"):
            library.BusUpdate(("v",), "v", bus_unit={"v": _VOLT})

    def test_bus_unit_missing_schema_field_rejected(self):
        bu = BusUnit(fields={"v": _VOLT})  # missing "i"
        with pytest.raises(ValueError, match="do not match the bus schema"):
            library.BusUpdate(("v", "i"), "v", bus_unit=bu)

    def test_bus_unit_extra_field_rejected(self):
        bu = BusUnit(fields={"v": _VOLT, "i": _AMP, "extra": _VOLT})
        with pytest.raises(ValueError, match="do not match the bus schema"):
            library.BusUpdate(("v", "i"), "v", bus_unit=bu)


# ---------------------------------------------------------------------
# End-to-end simulate
# ---------------------------------------------------------------------


class TestEndToEnd:
    def test_simulate_with_bus_unit_matches_unitless_path(self):
        """Numerical output is unchanged by adding bus_unit propagation."""
        bu = BusUnit(fields={"v": _VOLT, "i": _AMP})

        def _build(with_unit):
            builder = jaxonomy.DiagramBuilder()
            # Build a 2-field bus from two constants.
            v_src = builder.add(library.Constant(value=1.5, name="v_src"))
            i_src = builder.add(library.Constant(value=2.5, name="i_src"))
            new_v = builder.add(library.Constant(value=9.0, name="new_v"))
            if with_unit:
                creator = builder.add(library.BusCreator(
                    ("v", "i"),
                    field_units={"v": _VOLT, "i": _AMP},
                    name="creator",
                ))
            else:
                creator = builder.add(library.BusCreator(
                    ("v", "i"), name="creator",
                ))
            update = builder.add(library.BusUpdate(
                ("v", "i"), "v",
                bus_unit=bu if with_unit else None,
                name="update",
            ))
            sel = builder.add(library.BusSelector("v", name="sel_v"))
            builder.connect(v_src.output_ports[0], creator.input_ports[0])
            builder.connect(i_src.output_ports[0], creator.input_ports[1])
            builder.connect(creator.output_ports[0], update.input_ports[0])
            builder.connect(new_v.output_ports[0], update.input_ports[1])
            builder.connect(update.output_ports[0], sel.input_ports[0])
            diagram = builder.build()
            ctx = diagram.create_context()
            results = jaxonomy.simulate(
                diagram, ctx, (0.0, 0.05),
                recorded_signals={"out_v": sel.output_ports[0]},
            )
            return np.asarray(results.outputs["out_v"])

        y_unit = _build(with_unit=True)
        y_nounit = _build(with_unit=False)
        np.testing.assert_array_equal(y_unit, y_nounit)
        # And the replacement value flows through: out_v == new_v = 9.0.
        np.testing.assert_allclose(y_unit, 9.0)
