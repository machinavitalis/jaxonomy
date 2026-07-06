# SPDX-License-Identifier: MIT

"""T-104 phase 2 (acausal extension) — canonical units per domain port.

Every standard acausal domain (electrical, rotational, translational,
thermal, hydraulic, fluid) now exposes ``flow_units`` / ``pot_units``
class attributes on its ``PortBase`` subclass. The values are
:class:`jaxonomy.framework.units.Unit` instances tagged with the
conventional SI dimensions for that domain's flow + potential variables.

These attributes are class-level (not per-instance) so they're cheap to
look up and don't disturb the existing component constructors. Tests
just assert the table.
"""

from __future__ import annotations

import pytest


def test_imports_resolve():
    """Acausal port classes import without disturbing anything."""
    from jaxonomy.acausal.component_library.component_base import (
        ElecPort,
        FluidPort,
        HydraulicPort,
        RotationalPort,
        ThermalPort,
        TranslationalPort,
    )

    for cls in (ElecPort, FluidPort, HydraulicPort, RotationalPort,
                ThermalPort, TranslationalPort):
        assert cls.flow_units is not None, f"{cls.__name__} missing flow_units"
        assert cls.pot_units is not None, f"{cls.__name__} missing pot_units"


def test_electrical_canonical_units():
    """ElecPort: flow=Amp (current), pot=Volt."""
    from jaxonomy.acausal.component_library.component_base import ElecPort

    assert ElecPort.flow_units.name == "A"
    assert ElecPort.flow_units.dims == (0, 0, 0, 1, 0, 0, 0)
    assert ElecPort.pot_units.name == "V"
    # Volt: kg·m^2 / (s^3·A)
    assert ElecPort.pot_units.dims == (1, 2, -3, -1, 0, 0, 0)


def test_rotational_canonical_units_tagged_to_avoid_torque_energy_confusion():
    """RotationalPort flow is N·m as TORQUE, not energy — the
    physical_quantity tag disambiguates."""
    from jaxonomy.acausal.component_library.component_base import RotationalPort

    assert RotationalPort.flow_units.dims == (1, 2, -2, 0, 0, 0, 0)  # N·m
    assert RotationalPort.flow_units.physical_quantity == "torque"
    # ang.velocity in 1/s with the disambiguation tag
    assert RotationalPort.pot_units.dims == (0, 0, -1, 0, 0, 0, 0)
    assert RotationalPort.pot_units.physical_quantity == "angular_velocity"


def test_translational_canonical_units():
    from jaxonomy.acausal.component_library.component_base import TranslationalPort

    assert TranslationalPort.flow_units.name == "N"
    assert TranslationalPort.flow_units.dims == (1, 1, -2, 0, 0, 0, 0)
    assert TranslationalPort.pot_units.name == "m/s"
    assert TranslationalPort.pot_units.dims == (0, 1, -1, 0, 0, 0, 0)


def test_thermal_canonical_units():
    from jaxonomy.acausal.component_library.component_base import ThermalPort

    # heat-flow in Watt = kg·m^2 / s^3
    assert ThermalPort.flow_units.dims == (1, 2, -3, 0, 0, 0, 0)
    # temperature in Kelvin
    assert ThermalPort.pot_units.dims == (0, 0, 0, 0, 1, 0, 0)


def test_hydraulic_canonical_units():
    from jaxonomy.acausal.component_library.component_base import HydraulicPort

    # mass-flow in kg/s
    assert HydraulicPort.flow_units.dims == (1, 0, -1, 0, 0, 0, 0)
    # pressure in Pa = kg / (m·s^2)
    assert HydraulicPort.pot_units.dims == (1, -1, -2, 0, 0, 0, 0)


def test_fluid_canonical_units_match_hydraulic():
    """Fluid and hydraulic share the same canonical (mass-flow, pressure)
    pair — both transport mass through a pressure-driven domain."""
    from jaxonomy.acausal.component_library.component_base import (
        FluidPort,
        HydraulicPort,
    )

    assert FluidPort.flow_units == HydraulicPort.flow_units
    assert FluidPort.pot_units == HydraulicPort.pot_units


def test_port_base_default_is_none_for_custom_subclasses():
    """Custom domains can subclass PortBase without inheriting unit
    constants — the base class explicitly leaves flow_units / pot_units
    at None so unknown domains are flagged for the user to fill in."""
    from jaxonomy.acausal.component_library.component_base import PortBase

    assert PortBase.flow_units is None
    assert PortBase.pot_units is None


def test_units_are_class_level_not_per_instance():
    """Class-level attribute → no per-instance overhead, no need to
    update existing component constructors."""
    from jaxonomy.acausal.component_library.component_base import ElecPort

    # Two different "instances" (via __dict__ tricks since the class
    # constructor wants sym args) share the same class attribute.
    a = object.__new__(ElecPort)
    b = object.__new__(ElecPort)
    assert a.flow_units is b.flow_units
    assert a.flow_units is ElecPort.flow_units
