# SPDX-License-Identifier: MIT

"""Tests for T-110-followup-git-revision — extended git metadata in the
:class:`ProvenanceManifest`.

Covers:

* :func:`gather_git_info` returns a dict with the four expected keys.
* Inside a git checkout, ``git_head_sha`` / ``git_branch`` are populated
  on the manifest.
* The ``git_dirty`` flag tracks the working-tree state of a tempdir git
  repo (clean → ``False``; modify a file → ``True``).
* Outside a git checkout (and when ``git`` is missing on PATH), every
  ``git_*`` field is ``None`` and no exception bubbles up.
* :meth:`ProvenanceManifest.to_dict` / :meth:`to_json` round-trip
  preserves all new fields.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest

from jaxonomy.simulation import provenance as prov_mod
from jaxonomy.simulation.provenance import (
    ProvenanceManifest,
    compute_provenance,
    gather_git_info,
)


pytestmark = pytest.mark.minimal


# Short SHAs from ``git rev-parse --short`` are at least 4 hex chars and
# at most 40.  Accept a permissive lowercase-hex match.
_SHORT_SHA_RE = re.compile(r"^[0-9a-f]{4,40}$")


def _git_available() -> bool:
    return shutil.which("git") is not None


# ---------------------------------------------------------------------
# Helpers — build an isolated tempdir git repo so dirty-flag toggling
# doesn't depend on the surrounding worktree's state.
# ---------------------------------------------------------------------


def _init_tempdir_repo(root: Path) -> None:
    """Initialise a minimal git repo at ``root`` with one commit."""
    env = {
        **os.environ,
        # Disable any user-global git hooks / templates / signing that
        # might fight the test runner.
        "GIT_TERMINAL_PROMPT": "0",
        "GIT_AUTHOR_NAME": "Test",
        "GIT_AUTHOR_EMAIL": "test@example.com",
        "GIT_COMMITTER_NAME": "Test",
        "GIT_COMMITTER_EMAIL": "test@example.com",
    }

    def run(args: list[str]) -> None:
        subprocess.run(
            args, cwd=root, env=env, check=True,
            capture_output=True, text=True, timeout=5.0,
        )

    run(["git", "init", "--initial-branch=main", "."])
    run(["git", "config", "commit.gpgsign", "false"])
    (root / "hello.txt").write_text("hello\n")
    run(["git", "add", "hello.txt"])
    run(["git", "commit", "-m", "initial"])


# =====================================================================
# gather_git_info — direct unit tests on the seam
# =====================================================================


@pytest.mark.skipif(not _git_available(), reason="git not on PATH")
class TestGatherGitInfoInTempdirRepo:
    def test_keys_present(self, tmp_path: Path):
        _init_tempdir_repo(tmp_path)
        info = gather_git_info(str(tmp_path))
        assert set(info.keys()) == {"sha", "branch", "dirty", "commit_time"}

    def test_sha_shape(self, tmp_path: Path):
        _init_tempdir_repo(tmp_path)
        info = gather_git_info(str(tmp_path))
        assert isinstance(info["sha"], str)
        assert _SHORT_SHA_RE.match(info["sha"]), info["sha"]

    def test_branch_name(self, tmp_path: Path):
        _init_tempdir_repo(tmp_path)
        info = gather_git_info(str(tmp_path))
        # Some git installs ignore ``--initial-branch`` on older versions
        # and fall back to ``master``; either is acceptable.
        assert info["branch"] in ("main", "master")

    def test_dirty_flag_toggles(self, tmp_path: Path):
        _init_tempdir_repo(tmp_path)
        info_clean = gather_git_info(str(tmp_path))
        assert info_clean["dirty"] is False

        # Modify a tracked file — porcelain output now non-empty.
        (tmp_path / "hello.txt").write_text("hello world\n")
        info_dirty = gather_git_info(str(tmp_path))
        assert info_dirty["dirty"] is True

    def test_commit_time_iso(self, tmp_path: Path):
        _init_tempdir_repo(tmp_path)
        info = gather_git_info(str(tmp_path))
        assert isinstance(info["commit_time"], str)
        # ISO-8601-ish: contains a 'T' between date and time, and a TZ
        # suffix (we emit UTC, so '+00:00').
        assert "T" in info["commit_time"]
        assert "+00:00" in info["commit_time"]


# =====================================================================
# gather_git_info — fallback behaviour outside a repo / no git
# =====================================================================


class TestGatherGitInfoFallback:
    def test_outside_repo_all_none(self, tmp_path: Path):
        # tmp_path has no .git in it (and no parent .git either, in CI).
        # On a developer machine running this from a worktree, the
        # nearest parent might still be a git repo — guard the assertion
        # by running git from within an isolated tmp_path and discovering
        # via subprocess whether git considers it a repo.
        not_a_repo_dir = tmp_path / "not_a_repo"
        not_a_repo_dir.mkdir()
        # Force git to treat this dir as a leaf so it cannot ascend.
        env = {**os.environ, "GIT_CEILING_DIRECTORIES": str(tmp_path)}
        try:
            subprocess.run(
                ["git", "-C", str(not_a_repo_dir), "rev-parse", "HEAD"],
                env=env, capture_output=True, text=True, timeout=5.0, check=True,
            )
            pytest.skip("Test environment has an unexpected enclosing git repo")
        except (subprocess.CalledProcessError, FileNotFoundError):
            pass

        # Monkeypatch the helper to pin the CEILING env so the production
        # call replicates the same isolation.
        saved = os.environ.get("GIT_CEILING_DIRECTORIES")
        os.environ["GIT_CEILING_DIRECTORIES"] = str(tmp_path)
        try:
            info = gather_git_info(str(not_a_repo_dir))
        finally:
            if saved is None:
                os.environ.pop("GIT_CEILING_DIRECTORIES", None)
            else:
                os.environ["GIT_CEILING_DIRECTORIES"] = saved

        assert info == {
            "sha": None,
            "branch": None,
            "dirty": None,
            "commit_time": None,
        }

    def test_no_exception_when_git_missing(self, monkeypatch, tmp_path: Path):
        # Force ``subprocess.run`` inside the provenance module to raise
        # ``FileNotFoundError`` — simulates ``git`` not on PATH.  The
        # ``_run_git`` wrapper must swallow it and ``gather_git_info``
        # must still return the all-None dict (best-effort capture).
        def _no_git(*args, **kwargs):
            raise FileNotFoundError("git not on PATH")

        monkeypatch.setattr(prov_mod.subprocess, "run", _no_git)
        info = gather_git_info(str(tmp_path))
        assert info["sha"] is None
        assert info["branch"] is None
        assert info["dirty"] is None
        assert info["commit_time"] is None


# =====================================================================
# compute_provenance — wiring into the manifest
# =====================================================================


class TestComputeProvenanceGitFields:
    def test_include_git_false_disables_all_git_fields(self):
        manifest = compute_provenance(None, None, include_git=False)
        assert manifest.git_head is None
        assert manifest.git_head_sha is None
        assert manifest.git_branch is None
        assert manifest.git_dirty is None
        assert manifest.git_head_commit_time is None

    def test_stubbed_gather_populates_manifest(self, monkeypatch):
        stub = {
            "sha": "abc1234",
            "branch": "feature/x",
            "dirty": True,
            "commit_time": "2026-05-09T12:34:56+00:00",
        }
        monkeypatch.setattr(prov_mod, "gather_git_info", lambda *a, **k: stub)
        manifest = compute_provenance(None, None)
        assert manifest.git_head == "abc1234"
        assert manifest.git_head_sha == "abc1234"
        assert manifest.git_branch == "feature/x"
        assert manifest.git_dirty is True
        assert manifest.git_head_commit_time == "2026-05-09T12:34:56+00:00"

    def test_stubbed_gather_all_none(self, monkeypatch):
        stub = {"sha": None, "branch": None, "dirty": None, "commit_time": None}
        monkeypatch.setattr(prov_mod, "gather_git_info", lambda *a, **k: stub)
        manifest = compute_provenance(None, None)
        assert manifest.git_head is None
        assert manifest.git_head_sha is None
        assert manifest.git_branch is None
        assert manifest.git_dirty is None
        assert manifest.git_head_commit_time is None


# =====================================================================
# Serialisation round-trip
# =====================================================================


class TestSerializationRoundtrip:
    def _stub_manifest(self) -> ProvenanceManifest:
        return ProvenanceManifest(
            jaxonomy_version="0.0.test",
            jax_version="0.0.test",
            numpy_version="0.0.test",
            precision_info={"x64_enabled": True},
            options={"rtol": 1e-6},
            system={"system_id": 0, "type": "X", "parameter_names": [], "hash": "h"},
            timestamp="2026-05-09T00:00:00+00:00",
            git_head="abc1234",
            git_head_sha="abc1234",
            git_branch="main",
            git_dirty=False,
            git_head_commit_time="2026-05-08T12:00:00+00:00",
        )

    def test_to_dict_includes_new_fields(self):
        m = self._stub_manifest()
        d = m.to_dict()
        assert d["git_head_sha"] == "abc1234"
        assert d["git_branch"] == "main"
        assert d["git_dirty"] is False
        assert d["git_head_commit_time"] == "2026-05-08T12:00:00+00:00"

    def test_dict_roundtrip_preserves_new_fields(self):
        m = self._stub_manifest()
        rebuilt = ProvenanceManifest.from_dict(m.to_dict())
        assert rebuilt.git_head_sha == m.git_head_sha
        assert rebuilt.git_branch == m.git_branch
        assert rebuilt.git_dirty == m.git_dirty
        assert rebuilt.git_head_commit_time == m.git_head_commit_time
        assert rebuilt.to_dict() == m.to_dict()

    def test_json_roundtrip_preserves_new_fields(self):
        m = self._stub_manifest()
        encoded = m.to_json()
        decoded = json.loads(encoded)
        assert decoded["git_head_sha"] == "abc1234"
        assert decoded["git_branch"] == "main"
        assert decoded["git_dirty"] is False
        assert decoded["git_head_commit_time"] == "2026-05-08T12:00:00+00:00"
        rebuilt = ProvenanceManifest.from_dict(decoded)
        assert rebuilt.to_dict() == m.to_dict()

    def test_dirty_none_roundtrips(self):
        m = self._stub_manifest()
        # Re-build with git_dirty=None.
        m = ProvenanceManifest(
            jaxonomy_version=m.jaxonomy_version,
            jax_version=m.jax_version,
            numpy_version=m.numpy_version,
            precision_info=m.precision_info,
            options=m.options,
            system=m.system,
            timestamp=m.timestamp,
            git_head=None,
            git_head_sha=None,
            git_branch=None,
            git_dirty=None,
            git_head_commit_time=None,
        )
        rebuilt = ProvenanceManifest.from_dict(m.to_dict())
        assert rebuilt.git_dirty is None
        assert rebuilt.to_dict() == m.to_dict()
