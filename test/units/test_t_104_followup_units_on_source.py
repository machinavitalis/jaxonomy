# SPDX-License-Identifier: MIT

"""Tests for T-104-followup-units-on-source-blocks.

Verifies that the four most common source blocks (``Constant``, ``Sine``,
``Step``, ``Ramp``) honour an optional ``units=`` kwarg that tags their
sole output port with the requested :class:`Unit`.

Coverage:

* Each block, when constructed with ``units=meter``, produces an output
  port whose ``units`` attribute is ``meter``.
* Without a ``units=`` kwarg the output port's ``units`` is ``None``
  (default-off / byte-equivalence with pre-T-104 diagrams).
* Connecting a unit-tagged source to a destination port declaring a
  different unit raises :class:`UnitMismatchError` at
  :meth:`DiagramBuilder.connect` time.
* Connecting to a downstream port with no unit declared (or with the
  matching unit) succeeds.
"""

from __future__ import annotations

import math

import pytest

import jaxonomy
from jaxonomy.framework import LeafSystem
from jaxonomy.framework.units import (
    UnitMismatchError,
    meter,
    second,
)
from jaxonomy.library import Constant, Ramp, Sine, Step


pytestmark = pytest.mark.minimal


# ---------------------------------------------------------------------
# Tiny LeafSystem used by the connect-time tests below.  Mirrors the
# helper used in ``test_t_104_units_phase1.py``: one input port whose
# unit the test can freely set, and a no-op forwarding output.
# ---------------------------------------------------------------------


class _PassThrough(LeafSystem):
    def __init__(self, *, in_units=None, name=None, **kwargs):
        super().__init__(name=name, **kwargs)
        self.declare_input_port(name="u", units=in_units)
        self._out_idx = self.declare_output_port(
            self._eval,
            name="y",
            requires_inputs=True,
            prerequisites_of_calc=[self.input_ports[0].ticket],
        )

    def _eval(self, _time, _state, *inputs, **_params):
        return inputs[0]


# =====================================================================
# Output-port unit attribute, per source block.
# =====================================================================


class TestOutputPortCarriesUnit:
    def test_constant_carries_declared_unit(self):
        block = Constant(value=1.0, units=meter, name="K")
        assert block.output_ports[0].units == meter

    def test_sine_carries_declared_unit(self):
        block = Sine(
            amplitude=1.0,
            frequency=2.0 * math.pi,
            units=meter,
            name="Sin",
        )
        assert block.output_ports[0].units == meter

    def test_step_carries_declared_unit(self):
        block = Step(
            start_value=0.0,
            end_value=1.0,
            step_time=1.0,
            units=meter,
            name="Stp",
        )
        assert block.output_ports[0].units == meter

    def test_ramp_carries_declared_unit(self):
        block = Ramp(
            start_value=0.0,
            slope=1.0,
            start_time=0.0,
            units=meter,
            name="Rmp",
        )
        assert block.output_ports[0].units == meter


# =====================================================================
# Default-off: omitting the ``units=`` kwarg leaves the output port
# with ``units=None`` (byte-equivalence with pre-T-104 diagrams).
# =====================================================================


class TestDefaultOffByteEquivalent:
    def test_constant_default_units_none(self):
        block = Constant(value=1.0, name="K")
        assert block.output_ports[0].units is None

    def test_sine_default_units_none(self):
        block = Sine(name="Sin")
        assert block.output_ports[0].units is None

    def test_step_default_units_none(self):
        block = Step(name="Stp")
        assert block.output_ports[0].units is None

    def test_ramp_default_units_none(self):
        block = Ramp(name="Rmp")
        assert block.output_ports[0].units is None


# =====================================================================
# Connect-time enforcement: a unit-tagged source connected to a
# destination port carrying an incompatible unit must raise.
# =====================================================================


class TestConnectTimeEnforcement:
    @pytest.mark.parametrize(
        "make_source",
        [
            pytest.param(
                lambda: Constant(value=1.0, units=meter, name="src"),
                id="Constant",
            ),
            pytest.param(
                lambda: Sine(units=meter, name="src"),
                id="Sine",
            ),
            pytest.param(
                lambda: Step(units=meter, name="src"),
                id="Step",
            ),
            pytest.param(
                lambda: Ramp(units=meter, name="src"),
                id="Ramp",
            ),
        ],
    )
    def test_mismatched_unit_raises(self, make_source):
        builder = jaxonomy.DiagramBuilder()
        source = builder.add(make_source())
        sink = builder.add(_PassThrough(in_units=second, name="sink"))
        with pytest.raises(UnitMismatchError) as info:
            builder.connect(source.output_ports[0], sink.input_ports[0])
        msg = str(info.value)
        # The DiagramBuilder names both ports in the message.
        assert "src" in msg
        assert "sink" in msg

    @pytest.mark.parametrize(
        "make_source",
        [
            pytest.param(
                lambda: Constant(value=1.0, units=meter, name="src"),
                id="Constant",
            ),
            pytest.param(
                lambda: Sine(units=meter, name="src"),
                id="Sine",
            ),
            pytest.param(
                lambda: Step(units=meter, name="src"),
                id="Step",
            ),
            pytest.param(
                lambda: Ramp(units=meter, name="src"),
                id="Ramp",
            ),
        ],
    )
    def test_matching_unit_connects(self, make_source):
        builder = jaxonomy.DiagramBuilder()
        source = builder.add(make_source())
        sink = builder.add(_PassThrough(in_units=meter, name="sink"))
        # Identical units: must not raise.
        builder.connect(source.output_ports[0], sink.input_ports[0])

    @pytest.mark.parametrize(
        "make_source",
        [
            pytest.param(
                lambda: Constant(value=1.0, units=meter, name="src"),
                id="Constant",
            ),
            pytest.param(
                lambda: Sine(units=meter, name="src"),
                id="Sine",
            ),
            pytest.param(
                lambda: Step(units=meter, name="src"),
                id="Step",
            ),
            pytest.param(
                lambda: Ramp(units=meter, name="src"),
                id="Ramp",
            ),
        ],
    )
    def test_unit_tagged_source_connects_to_bare_sink(self, make_source):
        # A destination port with no declared unit (``units=None``)
        # accepts any source — preserves the backward-compat invariant.
        builder = jaxonomy.DiagramBuilder()
        source = builder.add(make_source())
        sink = builder.add(_PassThrough(name="sink"))
        builder.connect(source.output_ports[0], sink.input_ports[0])
