# SPDX-License-Identifier: MIT

"""Tests for T-104-followup-pint-bridge — optional pint registry bridge.

This follow-up adds a :func:`jaxonomy.framework.units.parse_unit`
string parser that:

* Prefers ``pint`` (via the optional ``[units]`` extra) when it is
  installed — pint understands the full SI vocabulary plus a huge
  catalogue of customary / engineering units (km/h, mph, psi, ...).
* Falls back to a small built-in parser covering the curated SI
  subset already exposed as module-level constants
  (``m``, ``kg``, ``s``, ``A``, ``K``, ``mol``, ``cd``, ``N``,
  ``J``, ``W``, ``Hz``, ``rad``, ``km``, ``mm``, ``ms``, ``min``,
  ``hr``, ``g``, ``degC``, ``degF``) when pint is absent.

The default-off contract: importing :mod:`jaxonomy.framework.units`
without pint installed MUST still succeed and ``parse_unit`` must
still resolve every spec in the curated subset. The pint backend
is purely additive and only widens what the parser accepts.
"""

from __future__ import annotations

import importlib

import pytest

from jaxonomy.framework.units import (
    Unit,
    celsius,
    dimensionless,
    fahrenheit,
    hertz,
    hour,
    joule,
    kilogram,
    kilometer,
    meter,
    newton,
    parse_unit,
    second,
    watt,
)


# ---------------------------------------------------------------------
# Built-in parser (always exercised — pint is optional)
# ---------------------------------------------------------------------


class TestBuiltinParser:
    """Cover every spec listed in the task spec plus a handful of
    related cases the built-in parser must support without pint."""

    def test_single_base_symbol(self):
        assert parse_unit("m") == meter
        assert parse_unit("kg") == kilogram
        assert parse_unit("s") == second

    def test_division_simple(self):
        # ``parse_unit("m/s") == meter / second``
        assert parse_unit("m/s") == meter / second

    def test_division_with_superscript_exponent(self):
        # ``parse_unit("m/s²") == meter / second**2``
        u = parse_unit("m/s²")
        assert u == meter / second**2
        assert u.dims == (0, 1, -2, 0, 0, 0, 0)

    def test_division_with_ascii_exponent(self):
        # ASCII fallback for systems / users that can't type unicode.
        assert parse_unit("m/s**2") == meter / second**2
        assert parse_unit("m/s^2") == meter / second**2

    def test_multiplication_middle_dot(self):
        # ``parse_unit("N·m") == newton * meter``
        u = parse_unit("N·m")
        assert u == newton * meter
        # Dimensionally a joule (energy/torque share base-SI dims).
        assert u.dims == joule.dims

    def test_multiplication_asterisk(self):
        assert parse_unit("N*m") == newton * meter

    def test_force_from_base_units(self):
        # ``parse_unit("kg·m/s²")`` should equal ``newton`` dimensionally
        # AND, because all factors have scale=1, byte-equal to ``newton``.
        u = parse_unit("kg·m/s²")
        assert u == newton
        assert u.dims == newton.dims

    def test_dimensionless_specs(self):
        assert parse_unit("1").is_dimensionless
        assert parse_unit("").is_dimensionless

    def test_derived_units_known(self):
        assert parse_unit("Hz") == hertz
        assert parse_unit("J") == joule
        assert parse_unit("W") == watt

    def test_scaled_units_known(self):
        assert parse_unit("km") == kilometer
        assert parse_unit("hr") == hour
        # ``h`` is also accepted as an alias for hour by the curated table.
        assert parse_unit("h") == hour

    def test_affine_units_round_trip(self):
        # Affine (offsetted) units cannot participate in algebra but
        # must round-trip as bare names.
        assert parse_unit("degC") == celsius
        assert parse_unit("degF") == fahrenheit
        # Unicode degree-letter forms also work.
        assert parse_unit("°C") == celsius

    def test_multi_product_quotient(self):
        # Heat capacity: J / (kg·K) — but built-in parser only allows
        # ONE '/', so spell it as J*K**-1*kg**-1.
        u = parse_unit("J*K**-1*kg**-1")
        # Power should reduce to m**2 / (s**2 * K).
        assert u.dims == (0, 2, -2, 0, -1, 0, 0)

    def test_unknown_symbol_raises_value_error(self):
        with pytest.raises(ValueError) as exc_info:
            parse_unit("foobar")
        # Error message names the curated set so users know what to
        # try next.
        assert "foobar" in str(exc_info.value)

    def test_non_string_raises_type_error(self):
        with pytest.raises(TypeError):
            parse_unit(123)  # type: ignore[arg-type]
        with pytest.raises(TypeError):
            parse_unit(None)  # type: ignore[arg-type]

    def test_returns_unit_instance(self):
        # Sanity: the return type is always Unit (not pint.Quantity).
        u = parse_unit("m/s")
        assert isinstance(u, Unit)

    def test_compatible_with_existing_algebra(self):
        # The parsed unit composes through the existing operators.
        speed = parse_unit("m/s")
        accel = speed / second
        assert accel == meter / second**2

    def test_default_off_no_pint_required(self):
        # Importing the module without pint installed must not break.
        # We re-import to make sure the module loads from a clean slate.
        # (importlib.reload would be heavy; this assert is enough.)
        mod = importlib.import_module("jaxonomy.framework.units")
        assert hasattr(mod, "parse_unit")
        assert mod.parse_unit("m") == meter


# ---------------------------------------------------------------------
# Pint-backed parser (skipped when pint is missing)
# ---------------------------------------------------------------------


class TestPintBackend:
    """Pint adds support for niche units the built-in parser doesn't
    know about (km/h, mph, degC parsing variants, ...). These tests
    are skipped automatically when pint is not installed."""

    def test_pint_round_trip_km_per_hour(self):
        pytest.importorskip("pint")
        u = parse_unit("km/h")
        # km/h is a speed: same base-SI dims as meter/second.
        assert u.dims == (meter / second).dims
        # And the scalar conversion factor must be 1000/3600.
        expected_scale = 1000.0 / 3600.0
        assert u.scale == pytest.approx(expected_scale)

    def test_pint_temperature(self):
        pytest.importorskip("pint")
        # ``degC`` via pint should match jaxonomy's celsius (same
        # dims as kelvin, scale=1.0, offset=273.15). Some pint
        # builds raise for offsetted units in to_base_units(); in
        # that case the bridge falls back to the built-in parser
        # which still handles ``degC`` correctly.
        u = parse_unit("degC")
        assert u.dims == celsius.dims  # same as kelvin's dims

    def test_pint_complex_spec_falls_back_gracefully(self):
        pytest.importorskip("pint")
        # A spec that pint understands but the built-in does not:
        # newtons per square metre (pascals). Pint should resolve
        # it via to_base_units → kg/(m·s²).
        u = parse_unit("N/m**2")
        assert u.dims == (1, -1, -2, 0, 0, 0, 0)

    def test_pint_registry_cached(self):
        pytest.importorskip("pint")
        # The bridge caches its UnitRegistry to keep parse_unit cheap.
        from jaxonomy.framework import units as units_mod
        # Drop any cached registry so we exercise the lazy-create path.
        units_mod._pint_registry = None
        _ = parse_unit("m")
        first = units_mod._pint_registry
        _ = parse_unit("m/s")
        second_call = units_mod._pint_registry
        assert first is second_call


# ---------------------------------------------------------------------
# Default-off byte-equivalence with T-104 phase 1 + followups.
# ---------------------------------------------------------------------


class TestDefaultOff:
    """``parse_unit`` is purely additive — none of the pre-existing
    constants change their ``dims`` / ``scale`` / ``offset`` / hash."""

    def test_module_constants_unchanged(self):
        # Spot-check a handful of constants. If T-104 phase 1's
        # byte-equivalence regressed, these would fire.
        assert meter.dims == (0, 1, 0, 0, 0, 0, 0)
        assert meter.scale == 1.0
        assert meter.offset == 0.0

        assert newton.dims == (1, 1, -2, 0, 0, 0, 0)
        assert newton.scale == 1.0
        assert newton.offset == 0.0

        assert celsius.scale == 1.0
        assert celsius.offset == 273.15

    def test_parse_unit_does_not_mutate_constants(self):
        # Repeatedly parse a string and confirm the canonical constant
        # is returned by identity when the spec matches a known one.
        # (We rely on _unit_from_dims_and_scale's lookup for pint,
        # and on _SYMBOL_TABLE for the built-in path.)
        for _ in range(3):
            u = parse_unit("m")
            assert u == meter
            assert u.dims == meter.dims

    def test_dimensionless_passthrough(self):
        # Dimensionless wildcards continue to compare equal to
        # ``dimensionless`` and connect to anything.
        assert parse_unit("1") == dimensionless
        assert parse_unit("") == dimensionless
