# SPDX-License-Identifier: MIT
"""
JSON model schema migration registry (T-016).

When the on-disk model.json schema evolves, older models still need to
load.  This module provides a forward-only migration mechanism: each
migration takes a dict at schema version ``N`` and returns a dict at
version ``N+1``.  Migrations are applied in sequence by
:func:`migrate_to_current` so a v0 model becomes v1 → v2 → ... →
``CURRENT_SCHEMA_VERSION``.

Usage::

    @register_migration(from_version=1)
    def _v1_to_v2(model_dict: dict) -> dict:
        # Rename "old_key" to "new_key" everywhere it appears.
        ...
        return model_dict

The registry is keyed on the source version so duplicate registrations
fail loudly.  Migrations may not skip versions; if a v1→v3 transform
is needed, register v1→v2 and v2→v3 separately.

Backwards compatibility: models with ``schema_version`` already at
``CURRENT_SCHEMA_VERSION`` pass through unchanged.  Models with no
``schema_version`` field are treated as version 0; if a v0→v1
migration is registered they go through it, otherwise they reach the
loader as-is (legacy behaviour preserved).
"""

from __future__ import annotations

from typing import Callable, Dict


__all__ = [
    "register_migration",
    "migrate_to_current",
    "registered_versions",
    "MigrationError",
]


_MIGRATIONS: Dict[int, Callable[[dict], dict]] = {}


class MigrationError(Exception):
    """Raised on a malformed migration registration or apply failure."""


def register_migration(from_version: int) -> Callable[[Callable], Callable]:
    """Register a migration from schema version ``from_version`` to
    ``from_version + 1``.

    Decorator form::

        @register_migration(from_version=1)
        def v1_to_v2(model_dict: dict) -> dict:
            return ...

    The decorated function must take a ``dict`` and return a ``dict``.
    Mutations in-place are permitted but the function should still
    return the (possibly same) dict — callers rely on the return value.
    """
    if not isinstance(from_version, int) or from_version < 0:
        raise MigrationError(
            f"register_migration: from_version must be a non-negative int, "
            f"got {from_version!r}."
        )

    def _decorator(func: Callable[[dict], dict]) -> Callable[[dict], dict]:
        if from_version in _MIGRATIONS:
            raise MigrationError(
                f"register_migration: a migration from version "
                f"{from_version} is already registered "
                f"({_MIGRATIONS[from_version].__name__!r}).  Each version "
                "transition may have at most one registered migration."
            )
        _MIGRATIONS[from_version] = func
        return func

    return _decorator


def registered_versions() -> list[int]:
    """Sorted list of source versions currently registered."""
    return sorted(_MIGRATIONS)


def migrate_to_current(
    model_dict: dict,
    *,
    current_version: int,
) -> dict:
    """Apply every registered migration in sequence to bring
    ``model_dict`` from its declared version up to ``current_version``.

    Args:
        model_dict: The parsed JSON object (a dict).  Must include a
            ``schema_version`` field; missing or non-int values are
            treated as version 0.
        current_version: The target version (typically
            ``CURRENT_SCHEMA_VERSION`` from ``from_model_json``).

    Returns:
        The migrated dict.  ``schema_version`` is updated as each
        transition is applied so partial chains are observable in the
        debugger.

    Raises:
        MigrationError: if a migration transition is missing
            (e.g. the model is at v1 but only v3→v4 is registered) or
            if a migration function returns a non-dict.
    """
    sv_raw = model_dict.get("schema_version", 0)
    if isinstance(sv_raw, str):
        # Legacy Collimator-era models stored schema_version as a string
        # ("3", "4", etc.).  Match `from_model_json`'s policy and treat
        # every such value as version 0.  A registered v0→v1 migration
        # will run on these.
        sv = 0
    else:
        try:
            sv = int(sv_raw) if sv_raw not in (None,) else 0
        except (TypeError, ValueError):
            sv = 0

    if sv >= current_version:
        return model_dict

    while sv < current_version:
        if sv not in _MIGRATIONS:
            raise MigrationError(
                f"migrate_to_current: no migration registered from "
                f"schema version {sv} (target {current_version}).  "
                f"Registered transitions: "
                f"{', '.join(f'{v}→{v + 1}' for v in registered_versions())}."
            )
        out = _MIGRATIONS[sv](model_dict)
        if not isinstance(out, dict):
            raise MigrationError(
                f"migrate_to_current: migration {sv}→{sv + 1} returned "
                f"{type(out).__name__}, expected dict."
            )
        model_dict = out
        sv += 1
        model_dict["schema_version"] = sv

    return model_dict
