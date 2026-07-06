# SPDX-License-Identifier: MIT

"""T-104 phase 3 — Unit JSON serialization + summary + physical_quantity tag.

Covers the three pieces of the phase-3 list:

* ``Unit.to_dict`` / ``from_dict`` / ``to_json`` / ``from_json``: lossless
  round-trip serialisation that drops default fields for compactness.
* ``Unit.summary``: human-readable one-line display.
* ``physical_quantity``: disambiguates two units with identical SI
  dimensions (e.g. ``N·m`` as torque vs. energy).  Participates in
  equality/hash and in :func:`are_units_compatible`; algebraic ops
  drop it (the user must re-tag the result).
"""

from __future__ import annotations

import json

import pytest

from jaxonomy.framework.units import (
    Unit,
    UnitMismatchError,
    are_units_compatible,
    assert_unit_compatible,
)


# ---------------------------------------------------------------------------
# physical_quantity field — equality, hash, compatibility.
# ---------------------------------------------------------------------------


def test_physical_quantity_default_is_none():
    u = Unit(dims=(1, 0, 0, 0, 0, 0, 0))
    assert u.physical_quantity is None


def test_physical_quantity_participates_in_equality():
    torque = Unit(dims=(1, 2, -2, 0, 0, 0, 0), physical_quantity="torque")
    energy = Unit(dims=(1, 2, -2, 0, 0, 0, 0), physical_quantity="energy")
    assert torque != energy


def test_physical_quantity_participates_in_hash():
    torque = Unit(dims=(1, 2, -2, 0, 0, 0, 0), physical_quantity="torque")
    energy = Unit(dims=(1, 2, -2, 0, 0, 0, 0), physical_quantity="energy")
    assert hash(torque) != hash(energy)


def test_physical_quantity_compat_requires_match_when_both_set():
    torque = Unit(dims=(1, 2, -2, 0, 0, 0, 0), physical_quantity="torque")
    energy = Unit(dims=(1, 2, -2, 0, 0, 0, 0), physical_quantity="energy")
    assert not are_units_compatible(torque, energy)
    with pytest.raises(UnitMismatchError):
        assert_unit_compatible(torque, energy)


def test_physical_quantity_unset_is_wildcard():
    """One side ``None`` is a wildcard so legacy code keeps working."""
    torque = Unit(dims=(1, 2, -2, 0, 0, 0, 0), physical_quantity="torque")
    untagged = Unit(dims=(1, 2, -2, 0, 0, 0, 0))
    assert are_units_compatible(torque, untagged)
    assert are_units_compatible(untagged, torque)


def test_same_physical_quantity_is_compatible():
    a = Unit(dims=(1, 2, -2, 0, 0, 0, 0), physical_quantity="torque")
    b = Unit(dims=(1, 2, -2, 0, 0, 0, 0), physical_quantity="torque")
    assert are_units_compatible(a, b)


def test_legacy_default_unit_equality_unchanged_after_phase3():
    """No-args Unit() instances must remain byte-equivalent under
    equality / hash so existing data structures aren't churned."""
    a = Unit()
    b = Unit()
    assert a == b
    assert hash(a) == hash(b)


# ---------------------------------------------------------------------------
# to_dict / from_dict round-trip.
# ---------------------------------------------------------------------------


def test_default_unit_serialises_to_empty_dict():
    """Compact default-omission: an all-defaults Unit emits ``{}``."""
    assert Unit().to_dict() == {}


def test_default_dict_round_trips_to_default_unit():
    assert Unit.from_dict({}) == Unit()


def test_round_trip_preserves_every_field():
    u = Unit(
        dims=(1, 2, -2, 0, 0, 0, 0),
        scale=2.5,
        offset=273.15,
        currency=(1, 0, 0, 0, 0),
        name="custom",
        physical_quantity="torque",
    )
    # to_dict drops nothing this time; from_dict reconstructs identically.
    loaded = Unit.from_dict(u.to_dict())
    assert loaded == u
    assert loaded.name == u.name  # name is informational, still preserved


def test_round_trip_via_json_string():
    u = Unit(
        dims=(0, 1, 0, 0, 0, 0, 0),
        name="m",
        physical_quantity="length",
    )
    loaded = Unit.from_json(u.to_json())
    assert loaded == u
    assert loaded.physical_quantity == "length"


def test_to_json_pretty_indent_emits_multiline():
    u = Unit(dims=(0, 1, 0, 0, 0, 0, 0), name="m")
    s = u.to_json(indent=2)
    assert "\n" in s


def test_from_json_rejects_non_object():
    with pytest.raises(ValueError, match="JSON object"):
        Unit.from_json("[1, 2, 3]")


def test_to_dict_omits_default_only_fields():
    """Only non-default fields appear; the dict stays compact for
    diff-friendly storage of large model files."""
    u = Unit(dims=(0, 1, 0, 0, 0, 0, 0), name="m")
    d = u.to_dict()
    # dims is non-default (length axis), name is set; nothing else.
    assert set(d.keys()) == {"dims", "name"}


# ---------------------------------------------------------------------------
# summary() — human-readable display.
# ---------------------------------------------------------------------------


def test_summary_dimensionless():
    s = Unit().summary()
    assert "dimensionless" in s


def test_summary_includes_dim_labels():
    u = Unit(dims=(1, 2, -2, 0, 0, 0, 0), name="N*m")
    s = u.summary()
    assert "kg" in s
    assert "m^2" in s
    assert "s^-2" in s
    assert 'name="N*m"' in s


def test_summary_includes_physical_quantity_when_set():
    u = Unit(dims=(1, 2, -2, 0, 0, 0, 0), physical_quantity="torque")
    assert 'physical_quantity="torque"' in u.summary()


def test_summary_renders_currency_axis():
    u = Unit(currency=(1, 0, 0, 0, 0), name="USD")
    assert "USD" in u.summary()


def test_summary_renders_scale_and_offset_only_when_non_default():
    u = Unit(dims=(0, 1, 0, 0, 0, 0, 0), scale=1000.0, name="km")
    s = u.summary()
    assert "scale=1000.0" in s
    # offset stays at 0 — must NOT appear (keeps the summary clean).
    assert "offset" not in s


# ---------------------------------------------------------------------------
# repr — physical_quantity disambiguator suffix.
# ---------------------------------------------------------------------------


def test_repr_appends_physical_quantity_tag():
    u = Unit(dims=(1, 2, -2, 0, 0, 0, 0), physical_quantity="torque")
    assert "@torque" in repr(u)


# ---------------------------------------------------------------------------
# Serialised forms remain stable across versions (smoke test).
# ---------------------------------------------------------------------------


def test_serialised_dict_keys_are_documented():
    """The set of keys :meth:`to_dict` can emit is fixed; new fields
    must be added with defaults so older serialised forms continue to
    load. This test asserts the current key set to catch accidental
    additions without an explicit migration plan."""
    rich = Unit(
        dims=(1, 2, -2, 0, 0, 0, 0),
        scale=2.0,
        offset=0.5,
        currency=(1, 0, 0, 0, 0),
        name="X",
        physical_quantity="Y",
    )
    keys = set(rich.to_dict().keys())
    assert keys == {
        "dims", "scale", "offset", "currency", "name", "physical_quantity",
    }
