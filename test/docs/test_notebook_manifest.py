# SPDX-License-Identifier: MIT

"""Anti-drift tests for the shipped-notebook manifest.

These are pure bookkeeping checks — no notebook is executed — so they cost
nothing and run in the default tier on every pull request. They are what stops
the execution gate from quietly rotting: a notebook added to ``docs/`` without a
manifest entry turns this red, rather than being silently untested until someone
notices the docs site is broken.
"""

from __future__ import annotations

import binascii
import json
import re

import pytest

from .notebook_manifest import (
    MANIFEST,
    REPO_ROOT,
    SMOKE,
    TIERS,
    discovered_notebooks,
    manifest_paths,
)


def test_manifest_is_not_empty():
    # Guard against a refactor emptying the manifest, which would make every
    # parametrized execution test below vacuously pass.
    assert len(MANIFEST) >= 90, "expected the manifest to carry the shipped notebooks"


def test_every_notebook_on_disk_is_in_the_manifest():
    missing = sorted(set(discovered_notebooks()) - set(manifest_paths()))
    assert not missing, (
        "notebooks under docs/ with no manifest entry — add them to "
        f"test/docs/notebook_manifest.py: {missing}"
    )


def test_every_manifest_entry_exists_on_disk():
    stale = sorted(set(manifest_paths()) - set(discovered_notebooks()))
    assert not stale, f"manifest entries with no notebook on disk: {stale}"


def test_no_duplicate_manifest_entries():
    paths = [nb.path for nb in MANIFEST]
    dupes = sorted({p for p in paths if paths.count(p) > 1})
    assert not dupes, f"duplicate manifest entries: {dupes}"


@pytest.mark.parametrize("nb", MANIFEST, ids=lambda nb: nb.path)
def test_manifest_entry_is_well_formed(nb):
    assert nb.tier in TIERS
    assert nb.timeout > 0, f"{nb.path}: timeout must be positive"
    assert nb.full_path.exists(), f"{nb.path}: missing"
    assert nb.run_dir.is_dir(), f"{nb.path}: run_dir {nb.run_dir} is not a directory"


def test_smoke_tier_stays_small():
    # The smoke tier runs on every PR. Its whole value is being cheap enough
    # that nobody is tempted to delete it; a bloated tier gets disabled.
    smoke = [nb for nb in MANIFEST if nb.tier == SMOKE]
    assert 4 <= len(smoke) <= 12, f"smoke tier has {len(smoke)} notebooks; keep it 4-12"
    budget = sum(nb.timeout for nb in smoke)
    assert budget <= 900, f"smoke tier timeout budget {budget}s is too generous"


@pytest.mark.parametrize("nb", MANIFEST, ids=lambda nb: nb.path)
def test_committed_image_outputs_decode(nb):
    """Every embedded image in a committed output must be valid base64.

    The execution gate deliberately ignores committed outputs, so a corrupt
    stored image passes it — the notebook still runs fine. mkdocs-jupyter does
    not: nbconvert's ExtractOutput preprocessor decodes every embedded image
    when rendering, so one bad payload fails the whole docs build and takes the
    published site down. That is not hypothetical; it happened, and went
    unnoticed for two weeks because nothing checked.

    Cheap enough for the default tier: parsing the corpus is well under a
    second and nothing is executed.
    """
    document = json.loads(nb.full_path.read_text(encoding="utf-8"))

    corrupt = []
    for index, cell in enumerate(document.get("cells", [])):
        for output in cell.get("outputs", []):
            for mime, payload in output.get("data", {}).items():
                if not mime.startswith("image/") or mime.endswith("svg+xml"):
                    continue
                encoded = payload if isinstance(payload, str) else "".join(payload)
                try:
                    binascii.a2b_base64(encoded)
                except (binascii.Error, ValueError) as exc:
                    corrupt.append(f"cell {index} [{mime}]: {exc}")

    assert not corrupt, (
        f"{nb.path} has undecodable embedded image data, which will fail the "
        "docs build:\n  " + "\n  ".join(corrupt)
    )


def test_mkdocs_nav_notebooks_exist():
    # A nav entry pointing at a deleted notebook builds a broken docs page.
    # Matched by regex rather than parsed: mkdocs.yml carries custom !ENV and
    # !!python/name: tags that a plain SafeLoader rejects, and pyyaml is not a
    # declared dependency of the test extra.
    text = (REPO_ROOT / "mkdocs.yml").read_text(encoding="utf-8")
    referenced = sorted(set(re.findall(r"[\w./-]+\.ipynb", text)))
    assert referenced, "expected mkdocs nav to reference example notebooks"

    missing = sorted(p for p in referenced if not (REPO_ROOT / "docs" / p).exists())
    assert not missing, f"mkdocs.yml nav references missing notebooks: {missing}"
