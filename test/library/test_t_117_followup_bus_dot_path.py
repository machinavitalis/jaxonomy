# SPDX-License-Identifier: MIT

"""Tests for T-117-followup-bus-dot-path.

Pre-fix, :class:`BusSelector` only supported single-level field names
— extracting a deeply nested field like
``chassis.suspension.spring_force`` required three cascaded selectors.
Post-fix, the dot-path string is resolved via
:func:`operator.attrgetter`, so a single selector handles arbitrary
nesting depth.

Tests:
* Dot-path through a 2-level nested bus returns the leaf value.
* Dot-path through a 3-level nested bus also works.
* Invalid segment in a dot path raises ValueError at construction.
* Single-segment field_name path is unchanged (byte-equivalent).
* Output port name is the leaf segment, not the full dotted string.
* ``bus_unit`` validation: top-level segment is verified to exist in
  ``bus_unit.fields``; output unit is silently None for dotted paths
  because :class:`BusUnit` is flat.
* End-to-end simulate with a nested bus.
"""

from __future__ import annotations

from collections import namedtuple

import jax.numpy as jnp
import numpy as np
import pytest

import jaxonomy
from jaxonomy import library
from jaxonomy.framework.units import BusUnit, Unit


_VOLT = Unit(dims=(1, 2, -3, -1, 0, 0, 0), name="V")
_AMP = Unit(dims=(0, 0, 0, 1, 0, 0, 0), name="A")


pytestmark = pytest.mark.minimal


# ---------------------------------------------------------------------
# Bus shapes used across tests
# ---------------------------------------------------------------------


_Inner = namedtuple("_Inner", ["x", "y"])
_Outer = namedtuple("_Outer", ["a", "inner"])
_Deeper = namedtuple("_Deeper", ["leaf"])
_DeepOuter = namedtuple("_DeepOuter", ["chassis"])
_Chassis = namedtuple("_Chassis", ["suspension"])
_Suspension = namedtuple("_Suspension", ["spring_force"])


# ---------------------------------------------------------------------
# Construction-time validation
# ---------------------------------------------------------------------


class TestDotPathValidation:
    def test_simple_dot_path_accepted(self):
        sel = library.BusSelector("inner.x")
        assert sel.field_name == "inner.x"
        # Output port name is the leaf segment, not the full dotted
        # string (NamedTuple field constraint).
        assert sel.output_ports[0].name == "x"

    def test_three_level_dot_path_accepted(self):
        sel = library.BusSelector("chassis.suspension.spring_force")
        assert sel.field_name == "chassis.suspension.spring_force"
        assert sel.output_ports[0].name == "spring_force"

    def test_empty_segment_in_dot_path_rejected(self):
        with pytest.raises(ValueError, match="not a valid Python identifier"):
            library.BusSelector("inner..x")

    def test_invalid_segment_in_dot_path_rejected(self):
        with pytest.raises(
            ValueError, match="not a valid Python identifier"
        ):
            library.BusSelector("inner.has space")

    def test_leading_dot_rejected(self):
        with pytest.raises(ValueError, match="not a valid Python identifier"):
            library.BusSelector(".x")

    def test_empty_field_name_rejected(self):
        with pytest.raises(ValueError, match="must be non-empty"):
            library.BusSelector("")

    def test_single_segment_still_works(self):
        """Byte-equivalent path: single-segment field_name unchanged."""
        sel = library.BusSelector("x")
        assert sel.field_name == "x"
        assert sel.output_ports[0].name == "x"


# ---------------------------------------------------------------------
# Runtime extraction
# ---------------------------------------------------------------------


def _make_nested_bus_source(bus_value, name="src"):
    """Return a LeafSystem whose sole output port emits ``bus_value``."""

    class _NestedBusSource(jaxonomy.LeafSystem):
        def __init__(self):
            super().__init__(name=name)

            def _emit(_t, _s, *_inputs, **_p):
                return bus_value

            self.declare_output_port(_emit, requires_inputs=False)

    return _NestedBusSource()


def _run_dot_path_selector(bus_value, field_name, **sel_kwargs):
    """Helper: wire a nested-bus source into BusSelector and simulate."""
    builder = jaxonomy.DiagramBuilder()
    src = builder.add(_make_nested_bus_source(bus_value))
    sel = builder.add(library.BusSelector(field_name, **sel_kwargs))
    builder.connect(src.output_ports[0], sel.input_ports[0])
    diagram = builder.build()
    ctx = diagram.create_context()
    results = jaxonomy.simulate(
        diagram, ctx, (0.0, 0.05),
        recorded_signals={"y": sel.output_ports[0]},
    )
    return np.asarray(results.outputs["y"])


class TestDotPathExtraction:
    def test_two_level_nested_bus_extraction(self):
        """``BusSelector("inner.y")`` pulls a leaf from a 2-level bus."""
        bus = _Outer(a=jnp.array(1.0), inner=_Inner(x=jnp.array(2.0),
                                                    y=jnp.array(3.0)))
        y = _run_dot_path_selector(bus, "inner.y")
        np.testing.assert_allclose(y, 3.0)

    def test_three_level_nested_bus_extraction(self):
        """The canonical ``chassis.suspension.spring_force`` case."""
        force = jnp.array(42.0)
        bus = _DeepOuter(
            chassis=_Chassis(
                suspension=_Suspension(spring_force=force),
            ),
        )
        y = _run_dot_path_selector(bus, "chassis.suspension.spring_force")
        np.testing.assert_allclose(y, 42.0)

    def test_dot_path_with_slice_idx(self):
        """``slice_idx`` indexes into the leaf array after the dot-path."""
        bus = _Outer(
            a=jnp.array(0.0),
            inner=_Inner(x=jnp.array([10.0, 20.0, 30.0]), y=jnp.array(0.0)),
        )
        y = _run_dot_path_selector(bus, "inner.x", slice_idx=2)
        np.testing.assert_allclose(y, 30.0)


# ---------------------------------------------------------------------
# BusUnit interaction
# ---------------------------------------------------------------------


class TestDotPathWithBusUnit:
    def test_dotted_path_validates_top_segment(self):
        """Top segment must exist in bus_unit.fields; deeper segments
        are not validated (BusUnit is flat)."""
        bu = BusUnit(fields={"inner": _VOLT,
                              "a": _VOLT})
        # Top segment present → OK.
        sel = library.BusSelector("inner.x", bus_unit=bu)
        # Output unit is None for dotted paths (no leaf-unit lookup).
        assert sel.output_ports[0].units is None

    def test_dotted_path_rejects_unknown_top_segment(self):
        bu = BusUnit(fields={"a": _VOLT})
        with pytest.raises(ValueError, match="top-level segment"):
            library.BusSelector("inner.x", bus_unit=bu)

    def test_single_segment_still_propagates_unit(self):
        """Byte-equivalent: single-segment with bus_unit gives the
        leaf unit on the output port."""
        u = _VOLT
        bu = BusUnit(fields={"a": u})
        sel = library.BusSelector("a", bus_unit=bu)
        assert sel.output_ports[0].units == u


# ---------------------------------------------------------------------
# End-to-end simulate
# ---------------------------------------------------------------------


class TestDotPathEndToEnd:
    def test_dot_path_in_a_diagram_simulates(self):
        """Dot-path selector composes with the simulator."""
        builder = jaxonomy.DiagramBuilder()
        # Wrap a Constant-emitted bus through a CustomJaxBlock that
        # builds the nested NamedTuple. Easiest path: just expose a
        # bus directly via a leaf system.
        class _NestedBusSource(jaxonomy.LeafSystem):
            def __init__(self):
                super().__init__(name="src")

                def _emit(_t, _s, *_inputs, **_p):
                    return _Outer(a=jnp.array(1.0),
                                  inner=_Inner(x=jnp.array(7.0),
                                               y=jnp.array(11.0)))

                self.declare_output_port(_emit, requires_inputs=False)

        src = builder.add(_NestedBusSource())
        sel = builder.add(library.BusSelector("inner.y", name="sel_y"))
        builder.connect(src.output_ports[0], sel.input_ports[0])
        diagram = builder.build()
        ctx = diagram.create_context()
        results = jaxonomy.simulate(
            diagram, ctx, (0.0, 0.1),
            recorded_signals={"y": sel.output_ports[0]},
        )
        y = np.asarray(results.outputs["y"])
        # Constant bus, constant selection → constant output = 11.0.
        np.testing.assert_allclose(y, 11.0)
