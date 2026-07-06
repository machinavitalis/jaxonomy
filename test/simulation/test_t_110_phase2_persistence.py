# SPDX-License-Identifier: MIT

"""T-110 phase 2 — persistence + verification helpers for ProvenanceManifest.

Covers the two pieces of the original T-110 phasing list that close the
reproducibility-workflow loop:

* :meth:`ProvenanceManifest.save` / :func:`load_manifest` — pin a
  manifest to disk so a release tag, a CI artifact, or a notebook can
  ship "this is the run we promised."
* :func:`compare_manifests` / :func:`verify_manifest` — diff two
  manifests and surface every drifted field; raise
  :class:`ManifestMismatch` (an ``AssertionError`` subclass) on any drift.

The third sub-item of phase 2 (a CI workflow that bumps a published
manifest per release tag) is deferred under
``T-110-followup-ci-manifest-bump`` — it depends on the CI convention
and reference-corpus selection rather than on the code surfaced here.
"""

from __future__ import annotations

import json

import pytest

from jaxonomy.simulation import (
    ManifestMismatch,
    ProvenanceManifest,
    compare_manifests,
    load_manifest,
    verify_manifest,
)


# ---------------------------------------------------------------------------
# Fixture.
# ---------------------------------------------------------------------------


def _make_manifest(
    *,
    timestamp: str = "2026-05-16T00:00:00+00:00",
    git_head: str | None = "abc123",
    config_hash: str = "deadbeef",
    options=None,
) -> ProvenanceManifest:
    """Construct a minimal but complete ProvenanceManifest fixture."""
    return ProvenanceManifest(
        jaxonomy_version="2.3.0",
        jax_version="0.6.0",
        numpy_version="2.0.0",
        precision_info={"dtype": "float64", "x64_enabled": True},
        options=options if options is not None else {"max_major_steps": 100},
        system={"name": "test", "leaves": 3},
        timestamp=timestamp,
        git_head=git_head,
        git_head_sha=git_head,
        git_branch="main",
        git_dirty=False,
        git_head_commit_time="2026-05-15T12:00:00+00:00",
        config_hash=config_hash,
    )


# ---------------------------------------------------------------------------
# Persistence: save -> load round-trip.
# ---------------------------------------------------------------------------


def test_save_then_load_round_trips_every_field(tmp_path):
    manifest = _make_manifest()
    path = tmp_path / "manifest.json"

    manifest.save(path)
    loaded = load_manifest(path)

    # Every persisted field comes back equal.
    assert loaded == manifest


def test_save_pretty_prints_by_default(tmp_path):
    """Default indent=2 yields a multi-line, diff-friendly file."""
    path = tmp_path / "manifest.json"
    _make_manifest().save(path)
    text = path.read_text()
    assert "\n" in text
    # JSON indent=2 produces lines starting with two spaces.
    assert "  " in text


def test_save_indent_none_emits_compact_json(tmp_path):
    path = tmp_path / "manifest.json"
    _make_manifest().save(path, indent=None)
    text = path.read_text()
    # No internal newlines in compact form.
    assert text.count("\n") == 0
    # And still valid JSON.
    assert json.loads(text)["jaxonomy_version"] == "2.3.0"


def test_save_creates_parent_directories(tmp_path):
    """``save`` doesn't require the caller to pre-create the directory."""
    path = tmp_path / "nested" / "subdir" / "manifest.json"
    _make_manifest().save(path)
    assert path.exists()


def test_load_manifest_missing_file_raises_file_not_found(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_manifest(tmp_path / "does_not_exist.json")


def test_load_manifest_invalid_json_raises(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text("{not valid json")
    with pytest.raises(json.JSONDecodeError):
        load_manifest(path)


# ---------------------------------------------------------------------------
# compare_manifests — returns the diff list.
# ---------------------------------------------------------------------------


def test_compare_identical_manifests_returns_empty():
    a = _make_manifest()
    b = _make_manifest()
    assert compare_manifests(a, b) == []


def test_compare_default_ignores_timestamp_drift():
    """Timestamp always differs; default ignore set treats it as noise."""
    a = _make_manifest(timestamp="2026-05-16T00:00:00+00:00")
    b = _make_manifest(timestamp="2026-05-16T13:14:15+00:00")
    assert compare_manifests(a, b) == []


def test_compare_can_be_told_to_include_timestamp():
    a = _make_manifest(timestamp="2026-05-16T00:00:00+00:00")
    b = _make_manifest(timestamp="2026-05-16T13:14:15+00:00")
    diffs = compare_manifests(a, b, ignore_fields=set())
    paths = {d[0] for d in diffs}
    assert "timestamp" in paths


def test_compare_surfaces_top_level_scalar_drift():
    a = _make_manifest(config_hash="aaaa")
    b = _make_manifest(config_hash="bbbb")
    diffs = compare_manifests(a, b)
    assert diffs == [("config_hash", "aaaa", "bbbb")]


def test_compare_surfaces_nested_dict_drift_with_dotted_path():
    a = _make_manifest(options={"max_major_steps": 100, "rtol": 1e-6})
    b = _make_manifest(options={"max_major_steps": 200, "rtol": 1e-6})
    diffs = compare_manifests(a, b)
    assert diffs == [("options.max_major_steps", 100, 200)]


def test_compare_detects_missing_keys_in_either_direction():
    a = _make_manifest(options={"max_major_steps": 100, "extra": "foo"})
    b = _make_manifest(options={"max_major_steps": 100})
    diffs = compare_manifests(a, b)
    # Extra key on the actual side, missing on the expected side.
    assert ("options.extra", "foo", "<missing>") in diffs


def test_compare_ignore_fields_can_drop_git_metadata():
    """Cross-machine comparison frequently wants to ignore git metadata."""
    a = _make_manifest(git_head="abc")
    b = _make_manifest(git_head="def")
    diffs = compare_manifests(a, b)  # default ignores only timestamp
    paths = {d[0] for d in diffs}
    assert "git_head" in paths
    # Now widen the ignore set.
    diffs = compare_manifests(
        a, b, ignore_fields={"timestamp", "git_head", "git_head_sha"}
    )
    assert diffs == []


# ---------------------------------------------------------------------------
# verify_manifest — raises on drift.
# ---------------------------------------------------------------------------


def test_verify_identical_manifests_passes_silently():
    a = _make_manifest()
    b = _make_manifest()
    verify_manifest(a, b)  # no exception


def test_verify_drift_raises_manifest_mismatch_carrying_differences():
    a = _make_manifest(config_hash="aaaa")
    b = _make_manifest(config_hash="bbbb")
    with pytest.raises(ManifestMismatch) as exc_info:
        verify_manifest(a, b)
    # The exception carries the diff list for programmatic introspection.
    assert exc_info.value.differences == [("config_hash", "aaaa", "bbbb")]
    # And the message names the drifted field.
    assert "config_hash" in str(exc_info.value)


def test_manifest_mismatch_is_assertion_error_subclass():
    """Composes naturally with pytest / standard assert-style flows."""
    a = _make_manifest(config_hash="aaaa")
    b = _make_manifest(config_hash="bbbb")
    with pytest.raises(AssertionError):
        verify_manifest(a, b)


def test_verify_round_trip_with_load_manifest(tmp_path):
    """Save -> load -> verify is the canonical CI shape."""
    manifest = _make_manifest()
    path = tmp_path / "reference.json"
    manifest.save(path)

    # Equivalent to a CI step that re-runs the simulation, captures a
    # fresh manifest, and verifies it against the pinned reference.
    fresh = _make_manifest(timestamp="2026-05-17T08:00:00+00:00")
    reference = load_manifest(path)
    verify_manifest(fresh, reference)  # passes — timestamp ignored by default
