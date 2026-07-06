# SPDX-License-Identifier: MIT

"""Tests for T-104 followup — scalar unit conversion + ``unit_conversion`` flag.

Phase 1 of T-104 only allowed strict-equal units; this followup adds:

* a :func:`assert_units_compatible_with_scale` helper that returns the
  multiplicative conversion factor when two units share base-dim
  exponents but differ by ``scale``;
* a ``unit_conversion="auto" | "warn" | "error"`` flag on
  :class:`DiagramBuilder` controlling connect-time behaviour for
  scaled-but-compatible units;
* automatic insertion of the conversion factor into the destination
  input port's evaluation callback under ``"auto"`` / ``"warn"``;
* a default-off byte-equivalence guarantee: a diagram without any
  ``units=`` declarations is unchanged regardless of the flag.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

import jaxonomy
from jaxonomy.framework import LeafSystem
from jaxonomy.framework.units import (
    UnitMismatchError,
    assert_units_compatible_with_scale,
    conversion_factor,
    dimensionless,
    gram,
    hour,
    kilogram,
    kilometer,
    meter,
    millimeter,
    millisecond,
    minute,
    second,
)


pytestmark = pytest.mark.minimal


# ---------------------------------------------------------------------
# Tiny LeafSystem that forwards its (single) input to its (single)
# output, with optional unit declarations on each side.
# ---------------------------------------------------------------------


class _PassThrough(LeafSystem):
    def __init__(self, *, in_units=None, out_units=None, name=None, **kwargs):
        super().__init__(name=name, **kwargs)
        self.declare_input_port(name="u", units=in_units)
        self.declare_output_port(
            self._eval,
            name="y",
            units=out_units,
            requires_inputs=True,
            prerequisites_of_calc=[self.input_ports[0].ticket],
        )

    def _eval(self, _time, _state, *inputs, **_params):
        return inputs[0]


# =====================================================================
# conversion_factor / assert_units_compatible_with_scale helpers
# =====================================================================


class TestConversionFactorHelper:
    def test_meter_to_kilometer(self):
        assert conversion_factor(meter, kilometer) == pytest.approx(1e-3)

    def test_kilometer_to_meter(self):
        assert conversion_factor(kilometer, meter) == pytest.approx(1e3)

    def test_second_to_millisecond(self):
        assert conversion_factor(second, millisecond) == pytest.approx(1e3)

    def test_minute_to_hour(self):
        assert conversion_factor(minute, hour) == pytest.approx(60.0 / 3600.0)

    def test_kilogram_to_gram(self):
        assert conversion_factor(kilogram, gram) == pytest.approx(1e3)

    def test_matched_units_factor_is_one(self):
        assert conversion_factor(meter, meter) == 1.0
        assert conversion_factor(second, second) == 1.0

    def test_dimensionless_wildcard_factor_is_one(self):
        assert conversion_factor(None, meter) == 1.0
        assert conversion_factor(meter, None) == 1.0
        assert conversion_factor(dimensionless, kilometer) == 1.0

    def test_dimension_mismatch_raises(self):
        with pytest.raises(UnitMismatchError):
            conversion_factor(meter, second)

    def test_assert_returns_factor(self):
        f = assert_units_compatible_with_scale(
            meter, kilometer, src_label="src", dst_label="dst"
        )
        assert f == pytest.approx(1e-3)

    def test_assert_dimension_mismatch_names_both_ports(self):
        with pytest.raises(UnitMismatchError) as info:
            assert_units_compatible_with_scale(
                meter, second, src_label="A", dst_label="B"
            )
        msg = str(info.value)
        assert "A" in msg
        assert "B" in msg


# =====================================================================
# Connect-time behaviour under each ``unit_conversion`` mode.
# We exercise the wired-up factor by reading the destination port's
# evaluation through a fixed source value.
# =====================================================================


def _eval_pass_through_input(diag, dest_block, source_value: float) -> float:
    """Build a context for ``diag`` and read ``dest_block``'s input port,
    after fixing the upstream value via a Constant block.  Returns the
    scalar value the destination port sees (post-conversion).
    """
    ctx = diag.create_context()
    return float(np.asarray(dest_block.eval_input(ctx, 0)))


class TestConnectModeAuto:
    def test_meter_to_kilometer_applies_factor_silently(self):
        from jaxonomy.library import Constant

        builder = jaxonomy.DiagramBuilder()  # default unit_conversion="auto"
        src = builder.add(
            _PassThrough(in_units=meter, out_units=meter, name="src")
        )
        dst = builder.add(
            _PassThrough(in_units=kilometer, out_units=kilometer, name="dst")
        )
        const = builder.add(Constant(value=1500.0, name="const"))

        # Wire the constant through `src` (a same-unit hop) into `dst`,
        # which has differently-scaled units on its input.  The connect
        # should auto-insert the 1e-3 factor on `dst`'s input.
        builder.connect(const.output_ports[0], src.input_ports[0])
        with warnings.catch_warnings():
            warnings.simplefilter("error")  # ensure NO UserWarning fires
            builder.connect(src.output_ports[0], dst.input_ports[0])

        diag = builder.build()
        # `src` passes through 1500 m; the conversion factor 1/1000 then
        # applies on `dst`'s input port, so `dst` should see 1.5 km.
        seen = _eval_pass_through_input(diag, dst, 1500.0)
        assert seen == pytest.approx(1.5)

    def test_dest_carries_recorded_conversion_factor(self):
        builder = jaxonomy.DiagramBuilder()
        src = builder.add(_PassThrough(out_units=meter, name="src"))
        dst = builder.add(_PassThrough(in_units=kilometer, name="dst"))
        builder.connect(src.output_ports[0], dst.input_ports[0])
        # Diagnostic attribute pinned by `_install_unit_conversion`.
        assert dst.input_ports[0]._unit_conversion_factor == pytest.approx(1e-3)

    def test_matched_units_no_factor_attached(self):
        builder = jaxonomy.DiagramBuilder()
        src = builder.add(_PassThrough(out_units=meter, name="src"))
        dst = builder.add(_PassThrough(in_units=meter, name="dst"))
        builder.connect(src.output_ports[0], dst.input_ports[0])
        # No conversion needed -> attribute should be absent (default-off).
        assert not hasattr(dst.input_ports[0], "_unit_conversion_factor")


class TestConnectModeWarn:
    def test_warn_mode_emits_userwarning_and_applies_factor(self):
        from jaxonomy.library import Constant

        builder = jaxonomy.DiagramBuilder(unit_conversion="warn")
        src = builder.add(_PassThrough(in_units=meter, out_units=meter, name="src"))
        dst = builder.add(_PassThrough(in_units=kilometer, out_units=kilometer, name="dst"))
        const = builder.add(Constant(value=2000.0, name="const"))
        builder.connect(const.output_ports[0], src.input_ports[0])
        with pytest.warns(UserWarning, match="Unit conversion"):
            builder.connect(src.output_ports[0], dst.input_ports[0])
        diag = builder.build()
        seen = _eval_pass_through_input(diag, dst, 2000.0)
        assert seen == pytest.approx(2.0)

    def test_warn_mode_no_warning_for_matched_units(self):
        builder = jaxonomy.DiagramBuilder(unit_conversion="warn")
        src = builder.add(_PassThrough(out_units=meter, name="src"))
        dst = builder.add(_PassThrough(in_units=meter, name="dst"))
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            builder.connect(src.output_ports[0], dst.input_ports[0])


class TestConnectModeError:
    def test_error_mode_refuses_scaled_connection(self):
        builder = jaxonomy.DiagramBuilder(unit_conversion="error")
        src = builder.add(_PassThrough(out_units=meter, name="src"))
        dst = builder.add(_PassThrough(in_units=kilometer, name="dst"))
        with pytest.raises(UnitMismatchError):
            builder.connect(src.output_ports[0], dst.input_ports[0])

    def test_error_mode_allows_matched_units(self):
        builder = jaxonomy.DiagramBuilder(unit_conversion="error")
        src = builder.add(_PassThrough(out_units=meter, name="src"))
        dst = builder.add(_PassThrough(in_units=meter, name="dst"))
        builder.connect(src.output_ports[0], dst.input_ports[0])

    def test_error_mode_allows_dimensionless_wildcard(self):
        # An undeclared (None) port still acts as a wildcard regardless of mode.
        builder = jaxonomy.DiagramBuilder(unit_conversion="error")
        src = builder.add(_PassThrough(out_units=meter, name="src"))
        dst = builder.add(_PassThrough(name="dst"))
        builder.connect(src.output_ports[0], dst.input_ports[0])


class TestDimensionMismatchAlwaysRaises:
    @pytest.mark.parametrize("mode", ["auto", "warn", "error"])
    def test_meter_to_second_raises_in_every_mode(self, mode):
        builder = jaxonomy.DiagramBuilder(unit_conversion=mode)
        src = builder.add(_PassThrough(out_units=meter, name="src"))
        dst = builder.add(_PassThrough(in_units=second, name="dst"))
        with pytest.raises(UnitMismatchError):
            builder.connect(src.output_ports[0], dst.input_ports[0])


class TestUnitConversionFlagValidation:
    def test_invalid_flag_rejected_at_construction(self):
        from jaxonomy.framework.diagram_builder import BuilderError
        with pytest.raises(BuilderError, match="unit_conversion"):
            jaxonomy.DiagramBuilder(unit_conversion="bogus")


# =====================================================================
# Default-off byte-equivalence: a diagram with no units= declarations
# is bit-for-bit identical regardless of unit_conversion mode.
# =====================================================================


class TestDefaultOffByteEquivalence:
    def _build(self, mode):
        from jaxonomy.library import Sine, Integrator

        builder = jaxonomy.DiagramBuilder(unit_conversion=mode)
        sine = builder.add(Sine(name="Sin_0"))
        integ = builder.add(Integrator(0.0, name="Integrator_0"))
        builder.connect(sine.output_ports[0], integ.input_ports[0])
        passthrough = builder.add(_PassThrough(name="pass"))
        builder.connect(integ.output_ports[0], passthrough.input_ports[0])
        return builder.build(), passthrough

    @pytest.mark.parametrize("mode", ["auto", "warn", "error"])
    def test_no_units_declared_is_byte_equivalent(self, mode):
        diag_default, pt_default = self._build("auto")
        diag_other, pt_other = self._build(mode)

        ctx_a = diag_default.create_context()
        ctx_b = diag_other.create_context()

        results_a = jaxonomy.simulate(
            diag_default, ctx_a, t_span=(0.0, 1.0),
            recorded_signals={"y": pt_default.output_ports[0]},
        )
        results_b = jaxonomy.simulate(
            diag_other, ctx_b, t_span=(0.0, 1.0),
            recorded_signals={"y": pt_other.output_ports[0]},
        )
        np.testing.assert_array_equal(
            np.asarray(results_a.outputs["y"]),
            np.asarray(results_b.outputs["y"]),
        )
        np.testing.assert_array_equal(
            np.asarray(results_a.time), np.asarray(results_b.time)
        )
