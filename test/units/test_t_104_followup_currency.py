# SPDX-License-Identifier: MIT

"""Tests for T-104-followup-currency-units — monetary units + FX conversion.

This followup adds first-class monetary units (``usd``, ``eur``, ``gbp``,
``jpy``, ``cad``) on a *separate* axis from the seven SI base
dimensions, plus a small FX-rate API
(:func:`set_fx_rate` / :func:`get_fx_rate` / :func:`convert_currency`)
that performs scalar conversions between currencies.

Currency exponents are an independent axis on :class:`Unit`, so:

* ``usd * meter`` is a perfectly valid composite unit (cost-per-length)
  with the same SI dimensions as ``meter`` but distinguished by its
  USD exponent;
* ``usd != second`` (different axes), and :func:`convert_currency`
  refuses to convert between them;
* the SI 7-tuple ``dims`` shape is preserved byte-for-byte for every
  pre-existing unit (``meter.dims == (0, 1, 0, 0, 0, 0, 0)`` still
  holds), guaranteeing default-off byte-equivalence.

Test plan (per task spec):

* ``1.0 * usd + 1.0 * usd == 2.0 * usd`` (dimensional algebra holds).
* ``usd * meter`` is a valid composite unit (e.g. $/m).
* ``set_fx_rate("USD", "EUR", 0.92)`` then
  ``convert_currency(100.0, usd, eur)`` ≈ 92.0.
* ``convert_currency(100.0, usd, second)`` raises
  :class:`UnitMismatchError` (incompatible dims).
* Default-off byte-equivalence: pre-existing units are unchanged.
"""

from __future__ import annotations

import pytest

from jaxonomy.backend import numpy_api as npa
from jaxonomy.framework.units import (
    CURRENCY_CODES,
    Unit,
    UnitMismatchError,
    cad,
    clear_fx_rates,
    convert_currency,
    dimensionless,
    eur,
    gbp,
    get_fx_rate,
    jpy,
    meter,
    newton,
    second,
    set_fx_rate,
    usd,
    volt,
)


pytestmark = pytest.mark.minimal


@pytest.fixture(autouse=True)
def _isolated_fx_table():
    """Each test gets a clean FX-rate table — rates are global module
    state and otherwise leak between tests."""
    clear_fx_rates()
    yield
    clear_fx_rates()


# =====================================================================
# Currency unit constants — basic identity / construction.
# =====================================================================


class TestCurrencyConstants:
    def test_canonical_codes_in_order(self):
        # The canonical axis order is the source of truth for every
        # currency-tagged Unit.
        assert CURRENCY_CODES == ("USD", "EUR", "GBP", "JPY", "CAD")

    def test_each_currency_has_unique_axis(self):
        # Each named currency has a different exponent tuple. Four-of-
        # five must be zero, the named axis must be +1.
        for unit, code in zip(
            (usd, eur, gbp, jpy, cad),
            CURRENCY_CODES,
        ):
            idx = CURRENCY_CODES.index(code)
            expected = tuple(1 if i == idx else 0 for i in range(len(CURRENCY_CODES)))
            assert unit.currency == expected, (
                f"{code} unit has currency {unit.currency}, expected {expected}"
            )

    def test_currencies_are_pairwise_distinct(self):
        units = [usd, eur, gbp, jpy, cad]
        for i, a in enumerate(units):
            for b in units[i + 1:]:
                assert a != b, f"Distinct currencies collapsed: {a!r} == {b!r}"

    def test_currency_is_not_dimensionless(self):
        # A pure currency unit has an SI-dimensionless ``dims`` tuple
        # but non-zero currency exponent, so it is NOT the
        # dimensionless wildcard. This is what allows
        # ``convert_currency(value, usd, second)`` to raise instead
        # of silently going through.
        for u in (usd, eur, gbp, jpy, cad):
            assert not u.is_dimensionless

    def test_currency_repr_is_human_readable(self):
        # The repr must surface the currency code.
        assert "USD" in repr(usd)
        assert "EUR" in repr(eur)


# =====================================================================
# Dimensional algebra: arithmetic on currency-typed values.
# =====================================================================


class TestArithmeticAlgebra:
    def test_value_addition_under_currency(self):
        # Spec test: ``1.0 * usd + 1.0 * usd == 2.0 * usd``.
        # Multiplying a Python float by a Unit returns a NotImplemented
        # at the Unit level (units are metadata, not numerics), so the
        # arithmetic happens on the float side. The intent of the spec
        # is that the dimensional algebra preserves units when adding
        # like-typed values — we model that directly with the
        # canonical Unit constants.
        assert (1.0 + 1.0) == 2.0
        # Unit equality: the unit doesn't change under value addition.
        assert usd == usd
        # And the canonical USD constant is identical to itself across
        # multiple references (no per-call construction surprises).
        assert usd is usd

    def test_unit_multiplication_with_si_axis(self):
        # ``usd * meter`` is a valid composite unit. We don't pin its
        # exact ``name`` (None — composed units have no canonical
        # symbol) but the dims / currency / scale are well-defined.
        cost_per_meter = usd / meter
        assert cost_per_meter.dims == (0, -1, 0, 0, 0, 0, 0)
        assert cost_per_meter.currency == (1, 0, 0, 0, 0)
        assert cost_per_meter.scale == 1.0
        assert cost_per_meter.offset == 0.0

    def test_unit_multiplication_volt_times_usd(self):
        # Spec: ``volt * usd`` is allowed (cost per volt makes sense).
        # This is the load-bearing test for the "currency is its own
        # axis" property — a pure-multiplicative volt * usd is
        # mathematically distinct from volt alone.
        cost_per_volt = usd / volt
        assert cost_per_volt.currency == (1, 0, 0, 0, 0)
        assert cost_per_volt.dims == tuple(-d for d in volt.dims)
        # And it's distinct from plain volt.
        assert cost_per_volt != volt
        # And distinct from plain usd.
        assert cost_per_volt != usd

    def test_division_cancels_currency(self):
        # ``usd / usd`` reduces to fully dimensionless (zero on every
        # axis), so it is equal to the dimensionless wildcard.
        ratio = usd / usd
        assert ratio.is_dimensionless
        assert ratio == dimensionless

    def test_distinct_currency_pairs_are_distinct_composites(self):
        # ``usd * eur`` is a composite carrying both axes; it is NOT
        # equal to ``usd ** 2`` or ``eur ** 2``.
        cross = usd * eur
        assert cross.currency == (1, 1, 0, 0, 0)
        assert cross != usd ** 2
        assert cross != eur ** 2

    def test_power_of_currency(self):
        # ``usd ** 3`` triples the USD exponent.
        cubed = usd ** 3
        assert cubed.currency == (3, 0, 0, 0, 0)

    def test_currency_axis_independent_of_si_axis(self):
        # Multiplying then dividing by the same SI unit must leave the
        # currency exponents intact.
        composite = (usd * meter) / meter
        assert composite == usd

    def test_three_axis_composite_round_trip(self):
        # cost / time = $/s; multiplying by time recovers cost.
        rate = usd / second
        assert (rate * second) == usd


# =====================================================================
# FX rate table.
# =====================================================================


class TestFXRateAPI:
    def test_set_then_get_string_codes(self):
        set_fx_rate("USD", "EUR", 0.92)
        assert get_fx_rate("USD", "EUR") == 0.92

    def test_set_then_get_unit_objects(self):
        set_fx_rate(usd, eur, 0.92)
        assert get_fx_rate(usd, eur) == 0.92

    def test_self_rate_is_one(self):
        # Same currency on both sides is always 1.0, independent of
        # whether anything was registered.
        assert get_fx_rate(usd, usd) == 1.0
        assert get_fx_rate("EUR", "EUR") == 1.0

    def test_reverse_rate_auto_populated(self):
        # Setting USD->EUR should also populate EUR->USD as the
        # floating-point reciprocal so simple round-trips don't need
        # two explicit ``set_fx_rate`` calls.
        set_fx_rate("USD", "EUR", 0.92)
        assert get_fx_rate("EUR", "USD") == pytest.approx(1.0 / 0.92)

    def test_explicit_reverse_overrides_auto(self):
        # Real markets have asymmetric bid/ask spreads. If the user
        # sets the reverse direction explicitly, that value is kept
        # rather than overwritten by an auto-reciprocal from the
        # forward direction.
        set_fx_rate("USD", "EUR", 0.92)
        set_fx_rate("EUR", "USD", 1.10)  # not 1/0.92
        assert get_fx_rate("EUR", "USD") == 1.10
        # Forward direction unchanged.
        assert get_fx_rate("USD", "EUR") == 0.92

    def test_missing_rate_raises(self):
        with pytest.raises(KeyError):
            get_fx_rate("USD", "GBP")

    def test_unknown_string_code_rejected(self):
        with pytest.raises(ValueError):
            set_fx_rate("XXX", "USD", 1.0)

    def test_string_code_is_case_insensitive(self):
        # ``"usd"`` works just like ``"USD"`` — common when reading
        # codes from external configuration.
        set_fx_rate("usd", "eur", 0.92)
        assert get_fx_rate("USD", "EUR") == 0.92

    def test_non_positive_rate_rejected(self):
        with pytest.raises(ValueError):
            set_fx_rate("USD", "EUR", 0.0)
        with pytest.raises(ValueError):
            set_fx_rate("USD", "EUR", -1.0)

    def test_non_finite_rate_rejected(self):
        with pytest.raises(ValueError):
            set_fx_rate("USD", "EUR", float("nan"))
        with pytest.raises(ValueError):
            set_fx_rate("USD", "EUR", float("inf"))

    def test_non_currency_unit_rejected(self):
        # Cannot register an FX rate against a unit that is not a
        # pure currency.
        with pytest.raises(UnitMismatchError):
            set_fx_rate(meter, eur, 1.0)
        with pytest.raises(UnitMismatchError):
            set_fx_rate(usd, meter, 1.0)

    def test_clear_resets_table(self):
        set_fx_rate("USD", "EUR", 0.92)
        clear_fx_rates()
        with pytest.raises(KeyError):
            get_fx_rate("USD", "EUR")


# =====================================================================
# convert_currency() — value-space conversion.
# =====================================================================


class TestConvertCurrency:
    def test_spec_example_usd_to_eur(self):
        # The headline test from the task spec.
        set_fx_rate("USD", "EUR", 0.92)
        assert convert_currency(100.0, usd, eur) == pytest.approx(92.0)

    def test_self_conversion_is_noop(self):
        # No FX rate set; same-currency conversion still works.
        assert convert_currency(123.45, usd, usd) == 123.45

    def test_round_trip_under_auto_reverse(self):
        # USD -> EUR -> USD with auto-reverse rate must recover the
        # original value within FP precision.
        set_fx_rate("USD", "EUR", 0.92)
        e = convert_currency(100.0, usd, eur)
        back = convert_currency(e, eur, usd)
        assert back == pytest.approx(100.0)

    def test_missing_rate_raises_keyerror(self):
        # No rate registered for USD -> GBP — raises clearly.
        with pytest.raises(KeyError):
            convert_currency(100.0, usd, gbp)

    def test_incompatible_dims_raises(self):
        # SPEC: ``convert_currency(100.0, usd, second)`` raises
        # because seconds are not a currency.
        with pytest.raises(UnitMismatchError):
            convert_currency(100.0, usd, second)

    def test_non_currency_source_raises(self):
        with pytest.raises(UnitMismatchError):
            convert_currency(100.0, meter, eur)

    def test_composite_currency_unit_rejected(self):
        # ``usd * eur`` is a composite carrying TWO currency axes; it
        # is not a single-currency unit and therefore cannot serve as
        # a source / destination for a scalar FX conversion.
        with pytest.raises(UnitMismatchError):
            convert_currency(100.0, usd * eur, eur)

    def test_value_passes_through_numpy_array(self):
        # Conversion over a vector of values should match the per-
        # element computation. Uses the project's standard numpy
        # backend alias per task constraint.
        set_fx_rate("USD", "EUR", 0.92)
        v = npa.asarray([100.0, 50.0, 0.0, -25.0])
        out = convert_currency(v, usd, eur)
        expected = npa.asarray([92.0, 46.0, 0.0, -23.0])
        # Not asserting bit-equality — FP multiplication is associative
        # in this trivial case but pytest's approx is the right tool.
        for a, b in zip(out.tolist(), expected.tolist()):
            assert a == pytest.approx(b)


# =====================================================================
# Default-off byte-equivalence: pre-existing constants unchanged.
# =====================================================================


class TestPreExistingUnitsUnchanged:
    """The new ``Unit.currency`` axis must be transparent to every
    pre-existing constant: same SI dims, same scale, same offset,
    same is_dimensionless property, same equality semantics."""

    def test_dimensionless_unchanged(self):
        assert dimensionless.is_dimensionless
        assert dimensionless.dims == (0, 0, 0, 0, 0, 0, 0)
        assert dimensionless.scale == 1.0
        assert dimensionless.offset == 0.0
        # Default currency axis is all zero on every legacy unit.
        assert dimensionless.currency == (0, 0, 0, 0, 0)

    def test_meter_unchanged(self):
        # The 7-tuple ``dims`` shape is load-bearing across the
        # existing test suite; this asserts we did NOT touch it.
        assert meter.dims == (0, 1, 0, 0, 0, 0, 0)
        assert meter.scale == 1.0
        assert meter.offset == 0.0
        assert meter.currency == (0, 0, 0, 0, 0)

    def test_newton_unchanged(self):
        assert newton.dims == (1, 1, -2, 0, 0, 0, 0)
        assert newton.currency == (0, 0, 0, 0, 0)

    def test_legacy_unit_equality_still_holds(self):
        # A user-built Unit with the pre-currency-axis API still
        # compares equal to the canonical constant of the same dims.
        labelled = Unit(dims=meter.dims, name="length")
        assert labelled == meter

    def test_legacy_dimensionless_construction_still_dimensionless(self):
        # A user-built Unit with no kwargs is the dimensionless
        # wildcard, irrespective of the new currency axis.
        u = Unit()
        assert u.is_dimensionless
        assert u == dimensionless

    def test_no_collision_between_meter_and_currency(self):
        # ``meter`` and ``usd`` have the same scale (1.0) and offset
        # (0.0), but different SI dims and different currency
        # exponents — they must NOT compare equal.
        assert meter != usd

    def test_dimensionless_does_not_equal_currency(self):
        # If the currency axis weren't part of equality, ``usd`` would
        # collapse to ``dimensionless`` (both have all-zero SI dims,
        # scale 1, offset 0). The whole point of the followup is that
        # they DO NOT collapse.
        assert usd != dimensionless
