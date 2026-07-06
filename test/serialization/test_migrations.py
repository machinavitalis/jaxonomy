# SPDX-License-Identifier: MIT
"""
T-016 — JSON schema migration registry tests.

Uses synthetic schema transitions so the test does not depend on the
real registered migrations (which can be empty in the steady state).
Each test isolates the global ``_MIGRATIONS`` dict via a fixture.
"""

from __future__ import annotations

import pytest

from jaxonomy.dashboard.serialization.migrations import (
    MigrationError,
    migrate_to_current,
    register_migration,
    registered_versions,
)
from jaxonomy.dashboard.serialization import migrations as migrations_module


@pytest.fixture(autouse=True)
def _isolate_registry():
    """Snapshot and restore the global migration registry for each test."""
    saved = dict(migrations_module._MIGRATIONS)
    migrations_module._MIGRATIONS.clear()
    yield
    migrations_module._MIGRATIONS.clear()
    migrations_module._MIGRATIONS.update(saved)


# ── basic registration ────────────────────────────────────────────────────


def test_register_and_apply_single_migration():
    """v0 → v1 migration is applied to bring a v0 model to v1."""

    @register_migration(from_version=0)
    def _v0_to_v1(d: dict) -> dict:
        d["new_field"] = "added_at_v1"
        return d

    assert registered_versions() == [0]

    model = {"schema_version": 0, "blocks": []}
    out = migrate_to_current(model, current_version=1)
    assert out["schema_version"] == 1
    assert out["new_field"] == "added_at_v1"


def test_chained_migrations_are_applied_in_order():
    """v0 → v1 → v2 chain advances both fields."""

    @register_migration(from_version=0)
    def _v0_to_v1(d: dict) -> dict:
        d["v1_added"] = True
        return d

    @register_migration(from_version=1)
    def _v1_to_v2(d: dict) -> dict:
        d["v2_added"] = True
        return d

    assert registered_versions() == [0, 1]

    model = {"schema_version": 0}
    out = migrate_to_current(model, current_version=2)
    assert out["schema_version"] == 2
    assert out["v1_added"] is True
    assert out["v2_added"] is True


# ── identity / no-op cases ────────────────────────────────────────────────


def test_already_at_current_returns_unchanged():
    """A model already at the target version passes through."""

    @register_migration(from_version=0)
    def _v0_to_v1(d: dict) -> dict:  # pragma: no cover - should not run
        d["should_not_appear"] = True
        return d

    model = {"schema_version": 1, "value": 42}
    out = migrate_to_current(model, current_version=1)
    assert out["schema_version"] == 1
    assert "should_not_appear" not in out
    assert out["value"] == 42


def test_no_migrations_registered_with_v0_input():
    """Empty registry + v0 model with target v0 → identity."""
    model = {"schema_version": 0}
    out = migrate_to_current(model, current_version=0)
    assert out is model


# ── error paths ──────────────────────────────────────────────────────────


def test_missing_intermediate_migration_raises():
    """Registry has v0→v1 and v2→v3 but a v0 model needs v1→v2 too."""

    @register_migration(from_version=0)
    def _v0_to_v1(d: dict) -> dict:
        return d

    @register_migration(from_version=2)
    def _v2_to_v3(d: dict) -> dict:
        return d

    with pytest.raises(MigrationError, match="no migration registered from schema version 1"):
        migrate_to_current({"schema_version": 0}, current_version=3)


def test_duplicate_registration_raises():
    @register_migration(from_version=0)
    def _first(d: dict) -> dict:
        return d

    with pytest.raises(MigrationError, match="already registered"):
        @register_migration(from_version=0)
        def _second(d: dict) -> dict:  # noqa: F811
            return d


def test_negative_from_version_rejected():
    with pytest.raises(MigrationError, match="non-negative int"):
        register_migration(from_version=-1)


def test_migration_returning_non_dict_raises():
    @register_migration(from_version=0)
    def _bad(d: dict):
        return "not a dict"

    with pytest.raises(MigrationError, match="returned str"):
        migrate_to_current({"schema_version": 0}, current_version=1)


# ── string / missing schema_version handling ─────────────────────────────


def test_missing_schema_version_treated_as_v0():
    """A model without schema_version is treated as v0 and migrated."""

    @register_migration(from_version=0)
    def _v0_to_v1(d: dict) -> dict:
        d["migrated"] = True
        return d

    model = {"value": 1}  # no schema_version
    out = migrate_to_current(model, current_version=1)
    assert out["schema_version"] == 1
    assert out["migrated"] is True


def test_string_schema_version_treated_as_v0():
    """Legacy string schema_version values are treated as v0 (consistent
    with the existing legacy-format warning in from_model_json)."""

    @register_migration(from_version=0)
    def _v0_to_v1(d: dict) -> dict:
        d["migrated"] = True
        return d

    model = {"schema_version": "3", "value": 1}
    out = migrate_to_current(model, current_version=1)
    assert out["schema_version"] == 1
    assert out["migrated"] is True
