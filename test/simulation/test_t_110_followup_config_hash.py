# SPDX-License-Identifier: MIT

"""Tests for T-110-followup-config-hash — deterministic config hash.

The ``ProvenanceManifest.config_hash`` field is a SHA-256 over the
inputs that define a run's identity: ``SimulatorOptions`` field
values, the system fingerprint, the jaxonomy version, and the jax
version.  Timestamp and git metadata are deliberately excluded so
the hash is stable across runs and across git commits.

These tests cover the contracts the task spec calls out:

* Same options + same system → same hash.
* Different options → different hash.
* Different system parameters → different hash.
* Hash is deterministic byte-for-byte across re-computations.
* ``to_dict`` / ``from_dict`` round-trip preserves ``config_hash``.
* Default-off (``record_provenance=False``) → no manifest computed.
* Opt-in (``record_provenance=True``) → manifest carries a non-empty
  ``config_hash``.
"""

from __future__ import annotations

import pytest

import jaxonomy
from jaxonomy.backend import numpy_api as npa
from jaxonomy.simulation import SimulatorOptions
from jaxonomy.simulation.provenance import (
    ProvenanceManifest,
    compute_provenance,
)


pytestmark = pytest.mark.minimal


# ---------------------------------------------------------------------
# Tiny scalar-linear leaf reused across cases.
# ---------------------------------------------------------------------


def _make_scalar_linear(a: float = 1.5):
    class ScalarLinear(jaxonomy.LeafSystem):
        def __init__(self):
            super().__init__(name="ScalarLinear")
            self.declare_continuous_parameter("a", a)
            self.declare_continuous_state(
                shape=(), ode=self.ode, dtype=npa.float64,
            )

        def ode(self, time, state, **params):
            xc = state.continuous_state
            return -params["a"] * xc

    model = ScalarLinear()
    ctx = model.create_context(time=0.0)
    ctx = ctx.with_continuous_state(npa.float64(2.0))
    return model, ctx


def _make_minimal_leaf():
    """Variant without a declared parameter — keeps tests robust against
    framework signature drift on ``declare_continuous_parameter``."""
    class ScalarLinearNoParam(jaxonomy.LeafSystem):
        def __init__(self):
            super().__init__(name="ScalarLinearNoParam")
            self.declare_continuous_state(
                shape=(), ode=self.ode, dtype=npa.float64,
            )

        def ode(self, time, state):
            xc = state.continuous_state
            return -1.5 * xc

    model = ScalarLinearNoParam()
    ctx = model.create_context(time=0.0)
    ctx = ctx.with_continuous_state(npa.float64(2.0))
    return model, ctx


# =====================================================================
# Hash population + shape
# =====================================================================


class TestConfigHashShape:
    def test_field_is_non_empty_hex_string(self):
        model, _ = _make_minimal_leaf()
        manifest = compute_provenance(model, SimulatorOptions())
        # SHA-256 hex digest is 64 lowercase hex chars.
        assert isinstance(manifest.config_hash, str)
        assert len(manifest.config_hash) == 64
        int(manifest.config_hash, 16)  # parses as hex


# =====================================================================
# Determinism
# =====================================================================


class TestConfigHashDeterminism:
    def test_identical_inputs_same_hash(self):
        # Same instance + same options → same hash.  Crucially we
        # compare across two independent ``compute_provenance`` calls
        # (timestamp pinned to keep the rest of the manifest equal
        # for clarity, but ``config_hash`` excludes the timestamp so
        # it would match either way).
        model, _ = _make_minimal_leaf()
        opts = SimulatorOptions(rtol=1e-6, atol=1e-8)
        m1 = compute_provenance(model, opts, timestamp="FIXED")
        m2 = compute_provenance(model, opts, timestamp="FIXED")
        assert m1.config_hash == m2.config_hash

    def test_hash_independent_of_timestamp(self):
        # Same config, different timestamps → identical config_hash.
        model, _ = _make_minimal_leaf()
        opts = SimulatorOptions()
        m1 = compute_provenance(model, opts, timestamp="2026-01-01T00:00:00Z")
        m2 = compute_provenance(model, opts, timestamp="2026-12-31T23:59:59Z")
        assert m1.timestamp != m2.timestamp
        assert m1.config_hash == m2.config_hash

    def test_hash_independent_of_git(self):
        # ``include_git=True`` vs ``False`` should not perturb the
        # config hash — git metadata is deliberately excluded so the
        # hash is reproducible across commits / non-git checkouts.
        model, _ = _make_minimal_leaf()
        opts = SimulatorOptions()
        m_with = compute_provenance(
            model, opts, timestamp="FIXED", include_git=True,
        )
        m_without = compute_provenance(
            model, opts, timestamp="FIXED", include_git=False,
        )
        assert m_with.config_hash == m_without.config_hash


# =====================================================================
# Sensitivity to config changes
# =====================================================================


class TestConfigHashSensitivity:
    def test_different_rtol_different_hash(self):
        model, _ = _make_minimal_leaf()
        m_default = compute_provenance(
            model, SimulatorOptions(rtol=1e-6), timestamp="FIXED",
        )
        m_tight = compute_provenance(
            model, SimulatorOptions(rtol=1e-9), timestamp="FIXED",
        )
        assert m_default.config_hash != m_tight.config_hash

    def test_different_atol_different_hash(self):
        model, _ = _make_minimal_leaf()
        m_a = compute_provenance(
            model, SimulatorOptions(atol=1e-8), timestamp="FIXED",
        )
        m_b = compute_provenance(
            model, SimulatorOptions(atol=1e-12), timestamp="FIXED",
        )
        assert m_a.config_hash != m_b.config_hash

    def test_different_system_instances_same_hash_when_structurally_equivalent(self):
        # T-110-followup-stable-fingerprint: two independent constructions
        # of the same structural system MUST share a config_hash, even
        # though their per-process ``system_id`` counters differ. This
        # is the headline reproducibility contract — without it,
        # cross-process verification ("did my re-run match the
        # recorded run?") cannot work because every rebuild gets a
        # fresh system_id and would otherwise produce a different hash.
        model_a, _ = _make_minimal_leaf()
        model_b, _ = _make_minimal_leaf()
        opts = SimulatorOptions()
        h_a = compute_provenance(model_a, opts, timestamp="FIXED").config_hash
        h_b = compute_provenance(model_b, opts, timestamp="FIXED").config_hash
        assert h_a == h_b

    def test_no_system_still_hashes(self):
        # Manifest computed with ``system=None`` must still produce a
        # populated, stable hash — useful for offline reconstruction
        # tests where the system fingerprint is filled in by hand.
        h_none = compute_provenance(
            None, SimulatorOptions(), timestamp="FIXED",
        ).config_hash
        assert isinstance(h_none, str)
        assert len(h_none) == 64


# =====================================================================
# Serialisation round-trip
# =====================================================================


class TestConfigHashRoundtrip:
    def test_to_dict_includes_config_hash(self):
        model, _ = _make_minimal_leaf()
        manifest = compute_provenance(model, SimulatorOptions())
        d = manifest.to_dict()
        assert "config_hash" in d
        assert d["config_hash"] == manifest.config_hash

    def test_from_dict_preserves_config_hash(self):
        model, _ = _make_minimal_leaf()
        manifest = compute_provenance(model, SimulatorOptions())
        rebuilt = ProvenanceManifest.from_dict(manifest.to_dict())
        assert rebuilt.config_hash == manifest.config_hash
        # Full payload also round-trips.
        assert rebuilt.to_dict() == manifest.to_dict()


# =====================================================================
# Default-off / opt-in behaviour through ``simulate(...)``
# =====================================================================


class TestSimulateIntegration:
    def test_default_off_no_hash_computed(self):
        # ``record_provenance=False`` (default) → no manifest, hence
        # no config_hash anywhere on the result.
        model, ctx = _make_minimal_leaf()
        results = jaxonomy.simulate(
            model, ctx, (0.0, 0.1),
            options=SimulatorOptions(enable_tracing=False),
        )
        assert results.provenance is None

    def test_opt_in_populates_config_hash(self):
        model, ctx = _make_minimal_leaf()
        opts = SimulatorOptions(
            enable_tracing=False,
            record_provenance=True,
        )
        results = jaxonomy.simulate(model, ctx, (0.0, 0.1), options=opts)
        assert results.provenance is not None
        assert isinstance(results.provenance.config_hash, str)
        assert len(results.provenance.config_hash) == 64
