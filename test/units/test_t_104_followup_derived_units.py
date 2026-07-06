# SPDX-License-Identifier: MIT

"""Tests for T-104 followup — derived/composite SI units + ``derived_unit``.

Covers the extra derived units shipped on top of T-104 phase 1
(``coulomb``, ``volt``, ``ohm``, ``farad``, ``weber``, ``henry``,
``pascal``, ``tesla``) and the :func:`derived_unit` helper that lets
users alias a composed :class:`Unit` expression with a friendly name.

The defining algebraic identities (volt * ampere == watt, etc.) are
the same ones documented in the SI brochure; if any of these regress
we want to know loudly.
"""

from __future__ import annotations

import pytest

from jaxonomy.framework.units import (
    Unit,
    UnitMismatchError,
    derived_unit,
    # base units
    ampere,
    meter,
    second,
    kilogram,
    # phase-1 derived
    newton,
    joule,
    watt,
    hertz,
    radian,
    # new T-104-followup-derived-units
    coulomb,
    volt,
    ohm,
    farad,
    weber,
    henry,
    pascal,
    tesla,
    # offset-aware (only used for negative test)
    celsius,
)


pytestmark = pytest.mark.minimal


# =====================================================================
# Defining identities for the new derived units.
# =====================================================================


class TestDefiningIdentities:
    def test_coulomb_is_ampere_second(self):
        assert coulomb == ampere * second
        assert coulomb.dims == (0, 0, 1, 1, 0, 0, 0)

    def test_volt_from_joule_per_coulomb(self):
        assert volt == joule / coulomb

    def test_volt_from_watt_per_ampere(self):
        assert volt == watt / ampere

    def test_volt_times_ampere_equals_watt(self):
        # Task spec identity.
        assert volt * ampere == watt

    def test_ohm_from_volt_per_ampere(self):
        assert ohm == volt / ampere

    def test_ohm_times_ampere_equals_volt(self):
        # Task spec identity.
        assert ohm * ampere == volt

    def test_farad_from_coulomb_per_volt(self):
        assert farad == coulomb / volt

    def test_farad_times_volt_equals_coulomb(self):
        # Task spec identity.
        assert farad * volt == coulomb

    def test_weber_from_volt_times_second(self):
        assert weber == volt * second

    def test_henry_from_weber_per_ampere(self):
        assert henry == weber / ampere

    def test_pascal_from_newton_per_meter_squared(self):
        assert pascal == newton / (meter ** 2)

    def test_pascal_times_meter_squared_equals_newton(self):
        # Task spec identity.
        assert pascal * (meter ** 2) == newton

    def test_tesla_from_weber_per_meter_squared(self):
        assert tesla == weber / (meter ** 2)


# =====================================================================
# Cross-checks: dimensional independence of the new units.
# =====================================================================


class TestDimensionalSanity:
    def test_volt_dims(self):
        # V = kg * m^2 / (s^3 * A)
        assert volt.dims == (1, 2, -3, -1, 0, 0, 0)

    def test_ohm_dims(self):
        # Ω = kg * m^2 / (s^3 * A^2)
        assert ohm.dims == (1, 2, -3, -2, 0, 0, 0)

    def test_farad_dims(self):
        # F = s^4 * A^2 / (kg * m^2)
        assert farad.dims == (-1, -2, 4, 2, 0, 0, 0)

    def test_pascal_dims(self):
        # Pa = kg / (m * s^2)
        assert pascal.dims == (1, -1, -2, 0, 0, 0, 0)

    def test_henry_dims(self):
        # H = kg * m^2 / (s^2 * A^2)
        assert henry.dims == (1, 2, -2, -2, 0, 0, 0)

    def test_tesla_dims(self):
        # T = kg / (s^2 * A)
        assert tesla.dims == (1, 0, -2, -1, 0, 0, 0)

    def test_units_are_distinct(self):
        # No two distinct named derived units collapse to the same
        # (dims, scale) — guards against an off-by-one in any of the
        # tuples above.
        candidates = [
            coulomb, volt, ohm, farad, weber, henry, pascal, tesla,
            newton, joule, watt, hertz,
        ]
        for i, a in enumerate(candidates):
            for b in candidates[i + 1:]:
                assert a != b, (
                    f"Distinct derived units collapsed: {a!r} == {b!r}"
                )

    def test_all_have_no_offset(self):
        # Every new derived unit is purely multiplicative (no affine
        # shift). Important for the unit-conversion / pint-bridge paths
        # which would otherwise refuse to compute a scalar factor.
        for u in (coulomb, volt, ohm, farad, weber, henry, pascal, tesla):
            assert u.offset == 0.0
            assert u.scale == 1.0


# =====================================================================
# derived_unit() helper.
# =====================================================================


class TestDerivedUnitHelper:
    def test_named_composite(self):
        # Spec example: ``derived_unit("my_unit", "mu", meter * newton)``
        # creates a custom unit.
        my_unit = derived_unit(
            name="my_unit",
            symbol="mu",
            components=meter * newton,
        )
        assert isinstance(my_unit, Unit)
        # Dimensions match the composition.
        assert my_unit.dims == (meter * newton).dims
        # Equal under Unit.__eq__ to the un-aliased composition.
        assert my_unit == meter * newton
        # Friendly label round-trips through repr.
        assert "mu" in repr(my_unit)

    def test_symbol_overrides_name_in_repr(self):
        # When both ``name`` and ``symbol`` are given, ``symbol`` is the
        # display label (matching SI convention: N vs newton).
        torque = derived_unit("torque", "N·m", meter * newton)
        assert "N·m" in repr(torque)

    def test_name_used_when_symbol_omitted(self):
        torque = derived_unit("torque", components=meter * newton)
        # No symbol → name is the display label.
        assert "torque" in repr(torque)

    def test_dimensions_preserved_through_alias(self):
        # Aliasing a derived expression must not alter its dims, scale
        # or offset — only the ``name`` field changes.
        velocity = derived_unit("speed", "v", meter / second)
        assert velocity.dims == (meter / second).dims
        assert velocity.scale == (meter / second).scale
        assert velocity.offset == 0.0

    def test_participates_in_algebra(self):
        # A unit defined via ``derived_unit`` plays normally in the
        # multiplicative algebra: it can be further composed and the
        # result equals what you'd get from the underlying expression.
        torque = derived_unit("torque", "N·m", meter * newton)
        # Torque * angular velocity (1/s) = power (W)
        ang_vel = second ** -1
        assert torque * ang_vel == watt

    def test_requires_components(self):
        with pytest.raises(TypeError):
            # Missing ``components`` → not a Unit.
            derived_unit("oops")

    def test_rejects_non_unit_components(self):
        with pytest.raises(TypeError):
            derived_unit("oops", "x", components="meter * newton")  # type: ignore[arg-type]

    def test_rejects_affine_components(self):
        # Affine (offsetted) units cannot be re-aliased — they don't
        # participate in the multiplicative algebra.
        with pytest.raises(UnitMismatchError):
            derived_unit("hot", "T", components=celsius)


# =====================================================================
# Default-off byte-equivalence: pre-existing T-104 phase-1 constants.
# =====================================================================


class TestPreExistingUnitsUnchanged:
    """Ensure the additions don't perturb the pre-existing phase-1
    constants (their tuples must be byte-equivalent so users importing
    them get the identical dims / scale / offset and ``parse_unit("N")``
    still returns the canonical instance)."""

    def test_meter(self):
        assert meter.dims == (0, 1, 0, 0, 0, 0, 0)
        assert meter.scale == 1.0
        assert meter.offset == 0.0

    def test_newton(self):
        assert newton.dims == (1, 1, -2, 0, 0, 0, 0)

    def test_joule(self):
        assert joule.dims == (1, 2, -2, 0, 0, 0, 0)

    def test_watt(self):
        assert watt.dims == (1, 2, -3, 0, 0, 0, 0)

    def test_hertz(self):
        assert hertz.dims == (0, 0, -1, 0, 0, 0, 0)

    def test_radian_is_dimensionless_under_equality(self):
        # Radians are dimensionally dimensionless (zero exponents,
        # unity scale, zero offset), so ``radian.is_dimensionless`` is
        # True even though the canonical name is preserved for prints.
        assert radian.is_dimensionless


# =====================================================================
# Smoke test: the new units round-trip through parse_unit() (which we
# extended to recognise the new symbols).
# =====================================================================


class TestParseUnitBridge:
    def test_parse_volt(self):
        from jaxonomy.framework.units import parse_unit
        assert parse_unit("V") == volt

    def test_parse_coulomb(self):
        from jaxonomy.framework.units import parse_unit
        assert parse_unit("C") == coulomb

    def test_parse_pascal(self):
        from jaxonomy.framework.units import parse_unit
        assert parse_unit("Pa") == pascal

    def test_parse_ohm_ascii(self):
        from jaxonomy.framework.units import parse_unit
        assert parse_unit("ohm") == ohm
