# SPDX-License-Identifier: MIT

"""Anti-drift tests for the top-level CLAIMS.md / KNOWN_GAPS.md contract docs.

CLAIMS.md is the internal evidence ledger (what we claim, mapped to proof);
KNOWN_GAPS.md is its public inverse (what we do not yet claim). They are
symmetric. These tests catch the mechanical drift modes — a file gets deleted,
a cross-reference breaks, the public README stops pointing at the gaps doc, the
internal/public split gets blurred, or the README's "150+ blocks" claim drifts
away from the actual library size.

Note the asymmetry vs. KNOWN_GAPS: CLAIMS.md is explicitly *internal, not
published*, so the README points at the public KNOWN_GAPS.md only — not at
CLAIMS.md.
"""

from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
CLAIMS = REPO / "CLAIMS.md"
KNOWN_GAPS = REPO / "KNOWN_GAPS.md"
README = REPO / "README.md"


def test_claims_file_exists():
    assert CLAIMS.exists(), "CLAIMS.md missing at repo root"


def test_known_gaps_file_exists():
    assert KNOWN_GAPS.exists(), "KNOWN_GAPS.md missing at repo root"


def test_claims_cross_references_known_gaps():
    assert "KNOWN_GAPS.md" in CLAIMS.read_text(), (
        "CLAIMS.md must cross-reference KNOWN_GAPS.md as its public inverse."
    )


def test_known_gaps_cross_references_claims():
    assert "CLAIMS.md" in KNOWN_GAPS.read_text(), (
        "KNOWN_GAPS.md must cross-reference CLAIMS.md so readers see both "
        "halves of the contract."
    )


def test_claims_declares_itself_internal_and_not_published():
    """The internal/public split is load-bearing: CLAIMS is the evidence
    ledger we keep private, KNOWN_GAPS is what we publish. If a future edit
    blurs that, this catches it."""
    text = CLAIMS.read_text()
    assert "Internal" in text, "CLAIMS.md must declare itself internal"
    assert "Not published" in text or "not published" in text, (
        "CLAIMS.md must state it is not published (the public inverse is "
        "KNOWN_GAPS.md)."
    )


def test_readme_points_at_public_known_gaps():
    """The public README must point readers at KNOWN_GAPS.md. It must NOT be
    required to link the internal CLAIMS.md."""
    assert "KNOWN_GAPS.md" in README.read_text(), (
        "README should point at the public KNOWN_GAPS.md."
    )


def test_known_gaps_keeps_gap_anchor_phrases():
    """KNOWN_GAPS uses canonical status phrases as grep anchors. Losing all of
    them means the file eroded into untracked prose."""
    text = KNOWN_GAPS.read_text().lower()
    canonical = ("not yet implemented", "partial", "known limitation", "experimental")
    assert any(phrase in text for phrase in canonical), (
        "KNOWN_GAPS.md lost every canonical gap-status phrase."
    )


def test_readme_block_count_claim_stays_conservative():
    """The README claims "150+ library blocks". That number must stay at or
    below the real export count so the claim never over-states. If the library
    shrinks below 150, either the README comes down or the count is restored.
    """
    import jaxonomy.library as lib

    exported = getattr(lib, "__all__", None)
    assert exported, "jaxonomy.library must define __all__"
    count = len(exported)
    assert count >= 150, (
        f"jaxonomy.library.__all__ has {count} entries (< 150); the README's "
        '"150+ library blocks" claim no longer holds — restore the blocks or '
        "lower the README number."
    )
    assert "150+" in README.read_text(), (
        "README should state the conservative '150+' block count that this "
        "test guards."
    )
