# SPDX-License-Identifier: MIT

"""T-104 phase 2 — unit propagation rules + diagram walker.

Two layers:

* The pure per-block algebra functions (``units_for_adder`` etc.) are
  tested in isolation against the rule docstrings.
* :func:`propagate_diagram_units` walks a built diagram, applies the
  registered rules, and stamps output-port units. Tested end-to-end
  against small fixtures (Sine source with declared output unit
  flowing into a Gain / Adder / Integrator chain).
"""

from __future__ import annotations

import pytest

import jaxonomy
from jaxonomy.framework.units import (
    Unit,
    UnitMismatchError,
    dimensionless,
)
from jaxonomy.framework.unit_propagation import (
    UNIT_OF_TIME,
    get_unit_rule,
    propagate_diagram_units,
    register_unit_rule,
    units_for_adder,
    units_for_derivative,
    units_for_gain,
    units_for_integrator,
    units_for_passthrough,
    units_for_product,
    units_for_reciprocal,
)
from jaxonomy.library import (
    Adder,
    Constant,
    Gain,
    Integrator,
    Sine,
)


# ---------------------------------------------------------------------------
# Unit helpers used throughout the tests.
# ---------------------------------------------------------------------------


METER = Unit(dims=(0, 1, 0, 0, 0, 0, 0), name="m")
SECOND = Unit(dims=(0, 0, 1, 0, 0, 0, 0), name="s")
NEWTON = Unit(dims=(1, 1, -2, 0, 0, 0, 0), name="N")
MPS = Unit(dims=(0, 1, -1, 0, 0, 0, 0), name="m/s")
MPS2 = Unit(dims=(0, 1, -2, 0, 0, 0, 0), name="m/s^2")


# ---------------------------------------------------------------------------
# Pure algebra — Adder.
# ---------------------------------------------------------------------------


def test_adder_same_unit_passes_through():
    assert units_for_adder([METER, METER, METER]) == METER


def test_adder_raises_on_mismatch():
    with pytest.raises(UnitMismatchError):
        units_for_adder([METER, NEWTON])


def test_adder_all_none_returns_none():
    assert units_for_adder([None, None]) is None


def test_adder_mixed_none_treats_as_wildcard():
    """Inputs with no declared unit are treated as wildcards (consistent
    with the connect-time policy)."""
    assert units_for_adder([METER, None, METER]) == METER


# ---------------------------------------------------------------------------
# Pure algebra — Gain / Product / Reciprocal / Integrator / Derivative.
# ---------------------------------------------------------------------------


def test_gain_dimensionless_passes_input():
    """Default Gain (no gain_units in params) is dimensionless."""
    assert units_for_gain([METER]) == METER


def test_gain_with_unit_multiplies():
    # gain in N/m (spring constant) applied to a displacement → force
    spring = Unit(dims=(1, 0, -2, 0, 0, 0, 0), name="N/m")
    out = units_for_gain([METER], params={"gain_units": spring})
    assert out == NEWTON


def test_product_multiplies_inputs():
    out = units_for_product([NEWTON, METER])
    # N·m
    assert out.dims == (1, 2, -2, 0, 0, 0, 0)


def test_reciprocal_inverts():
    out = units_for_reciprocal([SECOND])
    # 1/s = Hz
    assert out.dims == (0, 0, -1, 0, 0, 0, 0)


def test_integrator_multiplies_by_seconds():
    """integ(velocity) = position; integ(m/s) → m."""
    out = units_for_integrator([MPS])
    assert out == METER


def test_derivative_divides_by_seconds():
    """d/dt(position) = velocity; d/dt(m) → m/s."""
    out = units_for_derivative([METER])
    assert out == MPS


def test_double_integrate_velocity_gives_position_times_seconds():
    """integ(integ(m/s)) → m·s, which is what the algebra dictates."""
    once = units_for_integrator([MPS])           # m
    twice = units_for_integrator([once])         # m·s
    assert twice.dims == (0, 1, 1, 0, 0, 0, 0)


def test_passthrough_returns_input():
    assert units_for_passthrough([METER]) == METER
    assert units_for_passthrough([]) is None
    assert units_for_passthrough([None]) is None


# ---------------------------------------------------------------------------
# Registry — register_unit_rule / get_unit_rule.
# ---------------------------------------------------------------------------


def test_registry_returns_none_for_unknown_block():
    assert get_unit_rule("NoSuchBlockName") is None


def test_registry_returns_preregistered_math_rules():
    assert get_unit_rule("Adder") is units_for_adder
    assert get_unit_rule("Gain") is units_for_gain
    assert get_unit_rule("Integrator") is units_for_integrator


def test_register_unit_rule_can_override():
    """Last write wins — downstream libs can override built-ins."""
    original = get_unit_rule("Gain")
    sentinel = lambda inputs, params=None: METER  # noqa: E731
    register_unit_rule("Gain", sentinel)
    try:
        assert get_unit_rule("Gain") is sentinel
    finally:
        register_unit_rule("Gain", original)  # restore


# ---------------------------------------------------------------------------
# Diagram walker — end-to-end.
# ---------------------------------------------------------------------------


def _wire_source_unit(block, port_index, unit):
    """Stamp ``unit`` on an output port (the framework doesn't expose a
    setter; the attribute is documented as plain-attr at port-decl
    time, see system_base.py phase-1 comment)."""
    block.output_ports[port_index].units = unit


def test_walker_propagates_through_gain():
    """Sine (m) → Gain (default) should leave the output as m."""
    b = jaxonomy.DiagramBuilder()
    src = b.add(Sine(amplitude=1.0, frequency=1.0))
    gain = b.add(Gain(2.0))
    b.connect(src.output_ports[0], gain.input_ports[0])
    b.export_output(gain.output_ports[0], name="y")
    diagram = b.build(name="root")

    _wire_source_unit(src, 0, METER)
    stamped = propagate_diagram_units(diagram)
    assert stamped >= 1
    assert gain.output_ports[0].units == METER


def test_walker_propagates_through_integrator():
    """Sine (m/s) → Integrator → output unit should be m."""
    b = jaxonomy.DiagramBuilder()
    src = b.add(Sine(amplitude=1.0, frequency=1.0))
    integ = b.add(Integrator(0.0))
    b.connect(src.output_ports[0], integ.input_ports[0])
    diagram = b.build(name="root")

    _wire_source_unit(src, 0, MPS)
    propagate_diagram_units(diagram)
    assert integ.output_ports[0].units == METER


def test_walker_propagates_through_adder_with_matched_inputs():
    b = jaxonomy.DiagramBuilder()
    s1 = b.add(Sine(amplitude=1.0, frequency=1.0, name="s1"))
    s2 = b.add(Sine(amplitude=1.0, frequency=2.0, name="s2"))
    add = b.add(Adder(2, operators="++"))
    b.connect(s1.output_ports[0], add.input_ports[0])
    b.connect(s2.output_ports[0], add.input_ports[1])
    diagram = b.build(name="root")

    _wire_source_unit(s1, 0, METER)
    _wire_source_unit(s2, 0, METER)
    propagate_diagram_units(diagram)
    assert add.output_ports[0].units == METER


def test_walker_raises_on_adder_mismatch_at_propagate_time():
    """The phase-2 walker raises with the same machinery the connect
    check would — the user discovers the bug at propagate time even if
    they wired m to N without realising."""
    b = jaxonomy.DiagramBuilder()
    s1 = b.add(Sine(amplitude=1.0, frequency=1.0, name="s1"))
    s2 = b.add(Sine(amplitude=1.0, frequency=2.0, name="s2"))
    add = b.add(Adder(2, operators="++"))
    b.connect(s1.output_ports[0], add.input_ports[0])
    b.connect(s2.output_ports[0], add.input_ports[1])
    diagram = b.build(name="root")

    _wire_source_unit(s1, 0, METER)
    _wire_source_unit(s2, 0, NEWTON)
    with pytest.raises(UnitMismatchError):
        propagate_diagram_units(diagram)


def test_walker_preserves_existing_output_units_by_default():
    """Default ``overwrite=False`` respects user-supplied annotations."""
    b = jaxonomy.DiagramBuilder()
    src = b.add(Sine(amplitude=1.0, frequency=1.0))
    gain = b.add(Gain(2.0))
    b.connect(src.output_ports[0], gain.input_ports[0])
    diagram = b.build(name="root")

    _wire_source_unit(src, 0, METER)
    # User pre-annotates gain output as N (overriding what the rule
    # would compute, which is m).
    gain.output_ports[0].units = NEWTON
    propagate_diagram_units(diagram)
    assert gain.output_ports[0].units == NEWTON


def test_walker_overwrite_flag_replaces_existing_units():
    b = jaxonomy.DiagramBuilder()
    src = b.add(Sine(amplitude=1.0, frequency=1.0))
    gain = b.add(Gain(2.0))
    b.connect(src.output_ports[0], gain.input_ports[0])
    diagram = b.build(name="root")

    _wire_source_unit(src, 0, METER)
    gain.output_ports[0].units = NEWTON
    propagate_diagram_units(diagram, overwrite=True)
    assert gain.output_ports[0].units == METER


def test_walker_skips_blocks_with_no_registered_rule():
    """A block whose class has no rule entry has its output port left
    untouched; the walker quietly skips it (no exception)."""
    b = jaxonomy.DiagramBuilder()
    src = b.add(Sine(amplitude=1.0, frequency=1.0))
    c = b.add(Constant(0.0))   # Constant has no registered rule
    add = b.add(Adder(2, operators="++"))
    b.connect(src.output_ports[0], add.input_ports[0])
    b.connect(c.output_ports[0], add.input_ports[1])
    diagram = b.build(name="root")

    _wire_source_unit(src, 0, METER)
    # Adder sees (m, None) — treats None as wildcard, output stays m.
    propagate_diagram_units(diagram)
    assert add.output_ports[0].units == METER


def test_walker_reaches_fixed_point_through_chain():
    """Sine (m/s) → Gain → Integrator → Gain should propagate units
    end-to-end (m/s → m/s → m → m) in <8 passes."""
    b = jaxonomy.DiagramBuilder()
    src = b.add(Sine(amplitude=1.0, frequency=1.0))
    g1 = b.add(Gain(2.0, name="g1"))
    integ = b.add(Integrator(0.0))
    g2 = b.add(Gain(0.5, name="g2"))
    b.connect(src.output_ports[0], g1.input_ports[0])
    b.connect(g1.output_ports[0], integ.input_ports[0])
    b.connect(integ.output_ports[0], g2.input_ports[0])
    diagram = b.build(name="root")

    _wire_source_unit(src, 0, MPS)
    propagate_diagram_units(diagram)

    assert g1.output_ports[0].units == MPS
    assert integ.output_ports[0].units == METER
    assert g2.output_ports[0].units == METER


def test_walker_on_non_diagram_returns_zero():
    """Defensive: passing a non-Diagram input is a no-op."""
    assert propagate_diagram_units(object()) == 0


def test_unit_of_time_is_seconds():
    """Sanity: the canonical time unit is plain s, not s with scale."""
    assert UNIT_OF_TIME.dims == (0, 0, 1, 0, 0, 0, 0)
    assert UNIT_OF_TIME == SECOND
