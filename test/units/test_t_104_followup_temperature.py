# SPDX-License-Identifier: MIT

"""Tests for T-104 followup — offset-aware temperature conversion.

T-104 phase 1 and the T-104-fu-conversion follow-up only handle units
that differ by a *scalar* factor (length, mass, time, ...). This
follow-up extends ``Unit`` with an ``offset`` attribute so the canonical
affine temperature scales (Celsius, Fahrenheit) round-trip correctly
through :func:`jaxonomy.framework.units.convert_offset_aware`.

Coverage:

* Kelvin <-> Celsius and Celsius <-> Fahrenheit on the standard
  textbook reference points (freezing / boiling water, absolute zero).
* Round-trip exactness for K -> C -> K and C -> F -> C.
* Algebraic refusal: ``celsius * celsius`` (and similar) raises
  :class:`UnitMismatchError` with a message naming the affine units.
* Scalar-factor helpers (``conversion_factor`` /
  ``assert_units_compatible_with_scale``) refuse offsetted units with a
  clear error that points the caller at ``convert_offset_aware``.
* Default-off byte-equivalence: existing offset=0 units (meter, second,
  kilometer, ...) compare and hash identically to their phase-1 values
  and produce the same ``conversion_factor`` results.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from jaxonomy.framework.units import (
    Unit,
    UnitMismatchError,
    assert_units_compatible_with_scale,
    celsius,
    conversion_factor,
    convert_offset_aware,
    dimensionless,
    fahrenheit,
    gram,
    hour,
    kelvin,
    kilogram,
    kilometer,
    meter,
    millimeter,
    millisecond,
    minute,
    newton,
    second,
)


pytestmark = pytest.mark.minimal


# =====================================================================
# Canonical reference points: K <-> C <-> F
# =====================================================================


class TestKelvinCelsius:
    def test_freezing_point_kelvin_to_celsius(self):
        assert convert_offset_aware(kelvin, celsius, 273.15) == pytest.approx(0.0)

    def test_boiling_point_kelvin_to_celsius(self):
        assert convert_offset_aware(kelvin, celsius, 373.15) == pytest.approx(100.0)

    def test_absolute_zero_kelvin_to_celsius(self):
        assert convert_offset_aware(kelvin, celsius, 0.0) == pytest.approx(-273.15)

    def test_celsius_to_kelvin_round_values(self):
        assert convert_offset_aware(celsius, kelvin, 0.0) == pytest.approx(273.15)
        assert convert_offset_aware(celsius, kelvin, 100.0) == pytest.approx(373.15)
        assert convert_offset_aware(celsius, kelvin, -273.15) == pytest.approx(0.0)


class TestCelsiusFahrenheit:
    def test_freezing_point_celsius_to_fahrenheit(self):
        assert convert_offset_aware(celsius, fahrenheit, 0.0) == pytest.approx(32.0)

    def test_boiling_point_celsius_to_fahrenheit(self):
        assert convert_offset_aware(celsius, fahrenheit, 100.0) == pytest.approx(212.0)

    def test_fahrenheit_to_celsius(self):
        assert convert_offset_aware(fahrenheit, celsius, 32.0) == pytest.approx(0.0)
        assert convert_offset_aware(fahrenheit, celsius, 212.0) == pytest.approx(100.0)

    def test_absolute_zero_fahrenheit_to_kelvin(self):
        # -459.67 F is absolute zero by definition.
        assert convert_offset_aware(fahrenheit, kelvin, -459.67) == pytest.approx(
            0.0, abs=1e-9
        )

    def test_kelvin_to_fahrenheit_freezing(self):
        assert convert_offset_aware(kelvin, fahrenheit, 273.15) == pytest.approx(
            32.0, abs=1e-9
        )


# =====================================================================
# Round-trip exactness
# =====================================================================


class TestRoundTrip:
    @pytest.mark.parametrize("value", [0.0, 100.0, 273.15, 373.15, 1234.5])
    def test_kelvin_celsius_round_trip(self, value):
        round_tripped = convert_offset_aware(
            celsius, kelvin, convert_offset_aware(kelvin, celsius, value)
        )
        assert round_tripped == pytest.approx(value)

    @pytest.mark.parametrize("value", [-40.0, 0.0, 25.0, 100.0])
    def test_celsius_fahrenheit_round_trip(self, value):
        round_tripped = convert_offset_aware(
            fahrenheit, celsius, convert_offset_aware(celsius, fahrenheit, value)
        )
        assert round_tripped == pytest.approx(value)

    def test_minus_forty_is_invariant(self):
        # The textbook fixed point: -40 C == -40 F.
        assert convert_offset_aware(celsius, fahrenheit, -40.0) == pytest.approx(-40.0)
        assert convert_offset_aware(fahrenheit, celsius, -40.0) == pytest.approx(-40.0)

    def test_numpy_array_round_trip(self):
        values = np.array([0.0, 100.0, 273.15, 373.15])
        out = convert_offset_aware(
            celsius, kelvin, convert_offset_aware(kelvin, celsius, values)
        )
        np.testing.assert_allclose(out, values)


# =====================================================================
# Algebraic refusal for offsetted units
# =====================================================================


class TestAffineAlgebraRefused:
    def test_celsius_times_celsius_raises(self):
        with pytest.raises(UnitMismatchError) as info:
            _ = celsius * celsius
        msg = str(info.value).lower()
        # Message should mention what went wrong (offset / affine).
        assert "offset" in msg or "affine" in msg

    def test_celsius_times_second_raises(self):
        with pytest.raises(UnitMismatchError):
            _ = celsius * second

    def test_second_times_celsius_raises(self):
        with pytest.raises(UnitMismatchError):
            _ = second * celsius

    def test_celsius_divided_by_second_raises(self):
        with pytest.raises(UnitMismatchError):
            _ = celsius / second

    def test_fahrenheit_squared_raises(self):
        with pytest.raises(UnitMismatchError):
            _ = fahrenheit ** 2

    def test_celsius_to_first_power_is_identity_in_dims(self):
        # ``unit ** 1`` is a no-op even on affine units (no algebra needed).
        powered = celsius ** 1
        assert powered.dims == celsius.dims

    def test_conversion_factor_refuses_offsetted(self):
        with pytest.raises(UnitMismatchError) as info:
            conversion_factor(kelvin, celsius)
        msg = str(info.value).lower()
        assert "convert_offset_aware" in msg or "offset" in msg

    def test_assert_units_compatible_with_scale_refuses_offsetted(self):
        with pytest.raises(UnitMismatchError):
            assert_units_compatible_with_scale(celsius, fahrenheit)


# =====================================================================
# Convert helper: dimensional checks + wildcard behaviour
# =====================================================================


class TestConvertOffsetAwareEdges:
    def test_dimension_mismatch_raises(self):
        with pytest.raises(UnitMismatchError):
            convert_offset_aware(celsius, second, 100.0)

    def test_dimensionless_wildcard_passes_through(self):
        # When either side is dimensionless / None, the helper is a no-op.
        assert convert_offset_aware(None, celsius, 42.0) == 42.0
        assert convert_offset_aware(celsius, None, 42.0) == 42.0
        assert convert_offset_aware(dimensionless, celsius, 42.0) == 42.0

    def test_same_unit_is_identity(self):
        assert convert_offset_aware(celsius, celsius, 25.0) == pytest.approx(25.0)
        assert convert_offset_aware(kelvin, kelvin, 300.0) == pytest.approx(300.0)

    def test_non_offsetted_scalar_still_works(self):
        # The helper should subsume conversion_factor for offset=0 pairs.
        assert convert_offset_aware(meter, kilometer, 1500.0) == pytest.approx(1.5)
        assert convert_offset_aware(kilometer, meter, 1.5) == pytest.approx(1500.0)


# =====================================================================
# Default-off byte-equivalence: existing offset=0 units unchanged
# =====================================================================


class TestDefaultOffByteEquivalence:
    @pytest.mark.parametrize(
        "unit",
        [meter, second, kilogram, newton, kilometer, millimeter,
         millisecond, minute, hour, gram, kelvin],
    )
    def test_existing_units_have_zero_offset(self, unit):
        assert unit.offset == 0.0

    def test_existing_unit_equality_unchanged(self):
        # Hand-construct a Unit with the same dims+scale and confirm
        # it still compares equal to the canonical singleton.
        clone = Unit(dims=meter.dims, scale=1.0)
        assert clone == meter
        assert hash(clone) == hash(meter)

    def test_kilometer_meter_conversion_unchanged(self):
        # Pre-existing scalar conversion path must be byte-identical.
        assert conversion_factor(meter, kilometer) == pytest.approx(1e-3)
        assert conversion_factor(kilometer, meter) == pytest.approx(1e3)

    def test_unit_algebra_unchanged_for_zero_offset(self):
        # The existing m * s / s == m identity from phase 1 still holds.
        assert meter * second / second == meter
        assert (meter / second).dims == (0, 1, -1, 0, 0, 0, 0)
        assert (meter ** 2).dims == (0, 2, 0, 0, 0, 0, 0)

    def test_repr_omits_offset_when_zero(self):
        # Spot-check: the new offset field must not leak into reprs of
        # offset-free units (otherwise downstream snapshot tests would
        # break).
        assert "+" not in repr(meter)
        assert "+" not in repr(kilometer)
        assert "-" not in repr(meter)


# =====================================================================
# Temperature-unit metadata
# =====================================================================


class TestTemperatureUnitMetadata:
    def test_celsius_dims_match_kelvin(self):
        assert celsius.dims == kelvin.dims

    def test_fahrenheit_dims_match_kelvin(self):
        assert fahrenheit.dims == kelvin.dims

    def test_celsius_offset_is_positive_273_15(self):
        # Convention: offset takes raw->base SI, so 0 C must map to
        # 273.15 K via `scale * raw + offset`.
        assert celsius.offset == pytest.approx(273.15)
        assert celsius.scale == pytest.approx(1.0)

    def test_fahrenheit_offset_and_scale(self):
        # T_K = (5/9) * T_F + (5/9) * 459.67
        assert fahrenheit.scale == pytest.approx(5.0 / 9.0)
        assert fahrenheit.offset == pytest.approx((5.0 / 9.0) * 459.67)

    def test_celsius_and_fahrenheit_are_not_equal(self):
        # Both share dims with kelvin but differ in scale/offset.
        assert celsius != kelvin
        assert fahrenheit != kelvin
        assert celsius != fahrenheit

    def test_celsius_repr_shows_offset(self):
        # Doesn't pin exact format; just confirms the new field surfaces
        # when nonzero.
        text = repr(celsius)
        assert "273.15" in text or "degC" in text
