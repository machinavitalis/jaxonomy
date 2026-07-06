# SPDX-License-Identifier: MIT

"""T-110-followup-stable-fingerprint — cross-process-stable config_hash.

Pre-fix behaviour: `ProvenanceManifest.config_hash` folded the
per-process ``system_id`` counter into the system fingerprint, so two
structurally-equivalent runs in different Python processes (or even
in the same process after a fresh ``LeafSystem`` construction) got
different hashes. The headline T-110 "notarized receipt" claim
implies the *converse*: byte-equivalent runs should share a hash.

Post-fix: ``_system_fingerprint`` hashes only ``(type_name, sorted
parameter_names)``. ``system_id`` is preserved in the returned dict
for in-process debugging but is excluded from the digest.

We can't easily fork a subprocess inside the test runner, but we
*can* exercise the same failure mode: build the diagram twice (the
two builds get different ``system_id`` counter values) and assert
their ``config_hash`` is identical. Pre-fix this test would fail;
post-fix it must pass.
"""

from __future__ import annotations

import jax.numpy as jnp
import pytest

import jaxonomy
from jaxonomy.simulation.provenance import (
    _compute_config_hash,
    _system_fingerprint,
    compute_provenance,
)
from jaxonomy.library import Constant, Integrator


def _build_minimal_diagram():
    """``Constant(1.0) -> Integrator(0.0)``, exported as ``y``.

    Trivial structural fixture — what matters is that re-building
    the diagram bumps the per-process ``system_id`` counter on each
    leaf, exercising the cross-process-equivalent failure mode."""
    b = jaxonomy.DiagramBuilder()
    src = b.add(Constant(1.0, name="src"))
    integ = b.add(Integrator(0.0, name="integ"))
    b.connect(src.output_ports[0], integ.input_ports[0])
    b.export_output(integ.output_ports[0], name="y")
    return b.build(name="root")


# ---------------------------------------------------------------------------
# Fingerprint stability — the core T-110-followup contract.
# ---------------------------------------------------------------------------


def test_system_fingerprint_hash_stable_across_rebuilds():
    """Two independent builds of the same diagram structure produce
    identical ``hash`` fields even though their ``system_id`` counters
    differ."""
    a = _build_minimal_diagram()
    b = _build_minimal_diagram()

    fa = _system_fingerprint(a)
    fb = _system_fingerprint(b)

    # The diagnostic system_id should differ (proves the rebuild
    # actually bumped the counter — i.e. we're exercising the
    # failure mode the fix targets).
    assert fa["system_id"] != fb["system_id"], (
        "test fixture is not exercising the failure mode — both builds "
        "got the same system_id"
    )
    # The hash must be identical (the post-fix contract).
    assert fa["hash"] == fb["hash"], (
        f"system fingerprint hash differs across rebuilds: "
        f"{fa['hash']!r} vs {fb['hash']!r}"
    )


def test_system_fingerprint_includes_system_id_in_dict():
    """``system_id`` is kept in the returned dict for debugging even
    though it's not in the hash."""
    diag = _build_minimal_diagram()
    fp = _system_fingerprint(diag)
    assert "system_id" in fp
    assert fp["system_id"] is not None


def test_system_fingerprint_none_unchanged():
    """``_system_fingerprint(None)`` keeps its zero-payload shape."""
    fp = _system_fingerprint(None)
    assert fp == {
        "system_id": None,
        "type": None,
        "parameter_names": [],
        "hash": None,
    }


def test_system_fingerprint_differs_when_structure_differs():
    """A different diagram structure must produce a different hash —
    the fix must NOT collapse genuine differences."""
    a = _build_minimal_diagram()
    b = jaxonomy.DiagramBuilder()
    src = b.add(Constant(1.0, name="src"))
    b.export_output(src.output_ports[0], name="y")  # no Integrator
    different = b.build(name="root_alt")

    fa = _system_fingerprint(a)
    fb = _system_fingerprint(different)
    assert fa["hash"] != fb["hash"]


# ---------------------------------------------------------------------------
# config_hash stability — the user-facing manifest field.
# ---------------------------------------------------------------------------


def test_config_hash_stable_across_rebuilds():
    """The end-to-end ``ProvenanceManifest.config_hash`` (what users
    actually compare) is stable across two rebuilds of the same
    diagram + options."""
    a = _build_minimal_diagram()
    b = _build_minimal_diagram()
    options = jaxonomy.SimulatorOptions(rtol=1e-6, atol=1e-8)

    pa = compute_provenance(a, options, include_git=False,
                            timestamp="2026-01-01T00:00:00+00:00")
    pb = compute_provenance(b, options, include_git=False,
                            timestamp="2026-01-01T00:00:00+00:00")

    assert pa.config_hash == pb.config_hash, (
        f"config_hash differs across rebuilds: {pa.config_hash!r} vs "
        f"{pb.config_hash!r}"
    )


def test_config_hash_differs_when_options_differ():
    """Sanity: changing options that affect the kernel does flip the hash."""
    diag = _build_minimal_diagram()
    p1 = compute_provenance(
        diag,
        jaxonomy.SimulatorOptions(rtol=1e-6, atol=1e-8),
        include_git=False,
        timestamp="2026-01-01T00:00:00+00:00",
    )
    p2 = compute_provenance(
        diag,
        jaxonomy.SimulatorOptions(rtol=1e-9, atol=1e-11),
        include_git=False,
        timestamp="2026-01-01T00:00:00+00:00",
    )
    assert p1.config_hash != p2.config_hash


def test_config_hash_simulates_cross_process_replay():
    """Canonical end-to-end use case the T-110-followup unblocks:
    persist a manifest from "process A", re-build the diagram in
    "process B", and verify the hashes match — even though the
    rebuilt diagram has fresh ``system_id`` counters."""
    # Process A — capture and "save".
    diag_a = _build_minimal_diagram()
    options = jaxonomy.SimulatorOptions(rtol=1e-6, atol=1e-8)
    saved = compute_provenance(diag_a, options, include_git=False,
                               timestamp="2026-01-01T00:00:00+00:00")

    # ... (would be json.dumps + write to disk + read back in real code;
    # ProvenanceManifest persistence is covered by the phase-2 tests) ...

    # Process B — fresh rebuild + recompute.
    diag_b = _build_minimal_diagram()
    fresh = compute_provenance(diag_b, options, include_git=False,
                               timestamp="2026-01-02T12:34:56+00:00")

    # Timestamp differs (different "process" / wall-clock) but
    # config_hash matches (the run-identity contract).
    assert saved.timestamp != fresh.timestamp
    assert saved.config_hash == fresh.config_hash
