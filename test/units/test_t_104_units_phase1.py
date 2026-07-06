# SPDX-License-Identifier: MIT

"""Tests for T-104 Phase 1 — port-level units with consistency checking.

These cover:

* the :class:`Unit` algebra (multiply / divide / equality / dimensionless
  detection);
* default-off behaviour: ports without an explicit ``units=`` connect to
  anything;
* connect-time errors raised by :meth:`DiagramBuilder.connect` when units
  are incompatible, with both ports named in the message;
* numerical byte-equivalence of a simple diagram simulated with vs. without
  declared units.
"""

from __future__ import annotations

import pytest
import jax.numpy as jnp
import numpy as np

import jaxonomy
from jaxonomy.framework import LeafSystem, DependencyTicket
from jaxonomy.framework.units import (
    Unit,
    UnitMismatchError,
    are_units_compatible,
    assert_unit_compatible,
    dimensionless,
    dimensionless_unit,
    meter,
    second,
    kilogram,
    newton,
)


pytestmark = pytest.mark.minimal


# ---------------------------------------------------------------------
# Tiny LeafSystem used by the connect-time tests below.  It declares
# one input and one output, optionally with units, and simply forwards
# the input value (or returns a constant when fixed via fix_value).
# ---------------------------------------------------------------------


class _PassThrough(LeafSystem):
    def __init__(self, *, in_units=None, out_units=None, name=None, **kwargs):
        super().__init__(name=name, **kwargs)
        self.declare_input_port(name="u", units=in_units)
        self._out_idx = self.declare_output_port(
            self._eval,
            name="y",
            units=out_units,
            requires_inputs=True,
            prerequisites_of_calc=[self.input_ports[0].ticket],
        )

    def _eval(self, _time, _state, *inputs, **_params):
        return inputs[0]


# =====================================================================
# Algebra
# =====================================================================


class TestUnitAlgebra:
    def test_meter_times_second_over_second_equals_meter(self):
        # The exact identity called out in the task spec.
        result = meter * second / second
        assert result == meter

    def test_division_yields_velocity(self):
        velocity = meter / second
        assert velocity.dims == (0, 1, -1, 0, 0, 0, 0)

    def test_multiplication_accumulates_dims(self):
        area = meter * meter
        assert area.dims == (0, 2, 0, 0, 0, 0, 0)
        assert (area / meter) == meter

    def test_power(self):
        assert (meter ** 2).dims == (0, 2, 0, 0, 0, 0, 0)
        assert (meter ** -1).dims == (0, -1, 0, 0, 0, 0, 0)
        assert (meter ** 0) == dimensionless

    def test_newton_decomposition(self):
        # N == kg * m / s^2
        derived = kilogram * meter / (second ** 2)
        assert derived == newton

    def test_dimensionless_is_identity_under_multiplication(self):
        assert (meter * dimensionless) == meter
        assert (dimensionless * second) == second

    def test_dimensionless_constant_aliases(self):
        assert dimensionless is dimensionless_unit

    def test_is_dimensionless_property(self):
        assert dimensionless.is_dimensionless
        assert not meter.is_dimensionless
        # m / m collapses to dimensionless via algebra
        assert (meter / meter).is_dimensionless

    def test_unit_equality_ignores_name(self):
        labelled = Unit(dims=meter.dims, name="length")
        assert labelled == meter

    def test_repr_is_human_readable(self):
        # We don't pin the exact format, just that something useful comes out.
        assert "m" in repr(meter)
        assert "dimensionless" in repr(dimensionless)


# =====================================================================
# are_units_compatible / assert_unit_compatible
# =====================================================================


class TestCompatibilityHelper:
    def test_matching_units_compatible(self):
        assert are_units_compatible(meter, meter)

    def test_dimensionless_connects_to_anything(self):
        assert are_units_compatible(dimensionless, meter)
        assert are_units_compatible(meter, dimensionless)
        # None side is also treated as dimensionless / inheriting
        assert are_units_compatible(None, meter)
        assert are_units_compatible(meter, None)
        assert are_units_compatible(None, None)

    def test_mismatch_raises(self):
        with pytest.raises(UnitMismatchError) as info:
            assert_unit_compatible(meter, second, src_label="A", dst_label="B")
        msg = str(info.value)
        # Message must name both sides per the task spec.
        assert "A" in msg
        assert "B" in msg


# =====================================================================
# Connect-time enforcement
# =====================================================================


class TestConnectTimeEnforcement:
    def test_compatible_connection_succeeds(self):
        builder = jaxonomy.DiagramBuilder()
        src = builder.add(_PassThrough(out_units=meter, name="src"))
        dst = builder.add(_PassThrough(in_units=meter, name="dst"))
        # Should not raise.
        builder.connect(src.output_ports[0], dst.input_ports[0])

    def test_incompatible_connection_raises(self):
        builder = jaxonomy.DiagramBuilder()
        src = builder.add(_PassThrough(out_units=meter, name="src"))
        dst = builder.add(_PassThrough(in_units=second, name="dst"))
        with pytest.raises(UnitMismatchError) as info:
            builder.connect(src.output_ports[0], dst.input_ports[0])
        msg = str(info.value)
        # Both ports named in the message.
        assert "src" in msg
        assert "dst" in msg

    def test_dimensionless_default_connects_to_anything(self):
        # A port without any declared unit (the default) connects to a
        # port with units, in either direction.  This is the
        # backwards-compat invariant.
        builder = jaxonomy.DiagramBuilder()
        bare_src = builder.add(_PassThrough(name="bare_src"))
        meter_dst = builder.add(_PassThrough(in_units=meter, name="meter_dst"))
        builder.connect(bare_src.output_ports[0], meter_dst.input_ports[0])

        builder2 = jaxonomy.DiagramBuilder()
        meter_src = builder2.add(_PassThrough(out_units=meter, name="meter_src"))
        bare_dst = builder2.add(_PassThrough(name="bare_dst"))
        builder2.connect(meter_src.output_ports[0], bare_dst.input_ports[0])

    def test_explicit_dimensionless_connects(self):
        builder = jaxonomy.DiagramBuilder()
        src = builder.add(
            _PassThrough(out_units=dimensionless, name="src")
        )
        dst = builder.add(_PassThrough(in_units=meter, name="dst"))
        # dimensionless is an explicit "wildcard" too.
        builder.connect(src.output_ports[0], dst.input_ports[0])


# =====================================================================
# Numerical byte-equivalence: a simple Diagram with units produces the
# same simulated outputs as the same Diagram without units declared.
# =====================================================================


class TestNumericalByteEquivalence:
    def _build(self, *, with_units: bool):
        from jaxonomy.library import Sine, Integrator

        builder = jaxonomy.DiagramBuilder()
        sine = builder.add(Sine(name="Sin_0"))
        integ = builder.add(Integrator(0.0, name="Integrator_0"))
        builder.connect(sine.output_ports[0], integ.input_ports[0])

        # Wrap in a passthrough whose input/output ports optionally
        # carry units.  This exercises the units-aware connect path
        # without depending on a unit-aware standard-library block.
        passthrough = builder.add(
            _PassThrough(
                in_units=meter if with_units else None,
                out_units=meter if with_units else None,
                name="pass",
            )
        )
        builder.connect(integ.output_ports[0], passthrough.input_ports[0])

        return builder.build(), passthrough

    def test_default_path_byte_equivalent(self):
        diag_a, passthrough_a = self._build(with_units=False)
        diag_b, passthrough_b = self._build(with_units=True)

        ctx_a = diag_a.create_context()
        ctx_b = diag_b.create_context()

        results_a = jaxonomy.simulate(
            diag_a, ctx_a, t_span=(0.0, 1.0),
            recorded_signals={"y": passthrough_a.output_ports[0]},
        )
        results_b = jaxonomy.simulate(
            diag_b, ctx_b, t_span=(0.0, 1.0),
            recorded_signals={"y": passthrough_b.output_ports[0]},
        )

        # Bit-for-bit identical recorded signal.
        np.testing.assert_array_equal(
            np.asarray(results_a.outputs["y"]),
            np.asarray(results_b.outputs["y"]),
        )
        np.testing.assert_array_equal(
            np.asarray(results_a.time), np.asarray(results_b.time)
        )
