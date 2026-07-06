# SPDX-License-Identifier: MIT

"""Tests for T-110 Phase 1 — provenance/reproducibility manifest.

Covers:

* :func:`compute_provenance` returns a populated
  :class:`ProvenanceManifest` containing library versions, precision
  info, options, and a system fingerprint.
* ``to_dict()`` round-trips through ``json.dumps`` / ``json.loads``.
* Two manifests for the same system + options have identical hashes
  (modulo the timestamp).
* Default ``record_provenance=False`` → ``results.provenance is None``
  and the recorded outputs are byte-equivalent to a baseline run.
* Opt-in ``record_provenance=True`` → ``results.provenance`` is a
  populated :class:`ProvenanceManifest`.
"""

from __future__ import annotations

import json

import numpy as np
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
# Tiny scalar-linear leaf used as the smoke system for byte-equivalence
# and provenance-population checks.
# ---------------------------------------------------------------------


def _make_scalar_linear():
    a = 1.5

    class ScalarLinear(jaxonomy.LeafSystem):
        def __init__(self):
            super().__init__(name="ScalarLinear")
            self.declare_continuous_state(
                shape=(), ode=self.ode, dtype=npa.float64,
            )

        def ode(self, time, state):
            xc = state.continuous_state
            return -a * xc

    model = ScalarLinear()
    ctx = model.create_context(time=0.0)
    ctx = ctx.with_continuous_state(npa.float64(2.0))
    return model, ctx


# =====================================================================
# Direct ``compute_provenance`` API
# =====================================================================


class TestComputeProvenance:
    def test_basic_fields_populated(self):
        model, _ = _make_scalar_linear()
        opts = SimulatorOptions()
        manifest = compute_provenance(model, opts)

        assert isinstance(manifest, ProvenanceManifest)
        assert manifest.jaxonomy_version
        assert manifest.jax_version
        assert manifest.numpy_version
        assert "default_float_dtype" in manifest.precision_info
        assert "x64_enabled" in manifest.precision_info
        assert manifest.timestamp  # non-empty ISO string
        assert manifest.system["type"] == "ScalarLinear"
        assert isinstance(manifest.system["parameter_names"], list)
        assert manifest.system["hash"] is not None

    def test_options_snapshot_includes_known_fields(self):
        model, _ = _make_scalar_linear()
        opts = SimulatorOptions(rtol=1e-7, atol=1e-9, max_major_steps=42)
        manifest = compute_provenance(model, opts)

        assert manifest.options["rtol"] == pytest.approx(1e-7)
        assert manifest.options["atol"] == pytest.approx(1e-9)
        assert manifest.options["max_major_steps"] == 42
        # The new flag itself is recorded so a downstream auditor can
        # tell whether the manifest was opt-in.
        assert "record_provenance" in manifest.options

    def test_to_dict_roundtrips_through_json(self):
        model, _ = _make_scalar_linear()
        manifest = compute_provenance(model, SimulatorOptions())

        encoded = json.dumps(manifest.to_dict())
        decoded = json.loads(encoded)
        assert decoded["jaxonomy_version"] == manifest.jaxonomy_version
        assert decoded["jax_version"] == manifest.jax_version
        assert decoded["numpy_version"] == manifest.numpy_version
        assert decoded["timestamp"] == manifest.timestamp
        assert decoded["precision_info"]["default_float_dtype"] == (
            manifest.precision_info["default_float_dtype"]
        )

    def test_to_json_helper(self):
        model, _ = _make_scalar_linear()
        manifest = compute_provenance(model, SimulatorOptions())

        encoded = manifest.to_json()
        decoded = json.loads(encoded)
        assert decoded == manifest.to_dict()

    def test_from_dict_roundtrip(self):
        model, _ = _make_scalar_linear()
        manifest = compute_provenance(model, SimulatorOptions())
        rebuilt = ProvenanceManifest.from_dict(manifest.to_dict())
        assert rebuilt.to_dict() == manifest.to_dict()

    def test_determinism_same_inputs_same_hash(self):
        # Two identical systems built via the same constructor get
        # different ``system_id``s (the framework's auto-counter), so
        # we need the SAME instance for the hash to match.  This is
        # the operational guarantee a user wants: re-running a
        # simulation with the SAME system + options yields identical
        # provenance hashes (modulo timestamp).
        model, _ = _make_scalar_linear()
        opts = SimulatorOptions(rtol=1e-6, atol=1e-8)
        m1 = compute_provenance(model, opts, timestamp="FIXED")
        m2 = compute_provenance(model, opts, timestamp="FIXED")

        assert m1.system["hash"] == m2.system["hash"]
        # With timestamp pinned, the entire payload should match.
        assert m1.to_dict() == m2.to_dict()

    def test_structurally_equivalent_systems_have_same_hash(self):
        # T-110-followup-stable-fingerprint: two structurally-equivalent
        # systems share a ``hash`` even though their ``system_id``s
        # differ (per-process auto-incrementing counter). The post-fix
        # contract is that the hash captures structural identity, not
        # instance identity — required for cross-process reproducibility.
        m1, _ = _make_scalar_linear()
        m2, _ = _make_scalar_linear()
        h1 = compute_provenance(m1, SimulatorOptions()).system["hash"]
        h2 = compute_provenance(m2, SimulatorOptions()).system["hash"]
        assert h1 == h2

    def test_no_git_skip(self):
        model, _ = _make_scalar_linear()
        manifest = compute_provenance(
            model, SimulatorOptions(), include_git=False,
        )
        assert manifest.git_head is None

    def test_no_system_safe(self):
        # ``compute_provenance(None, ...)`` should not crash; useful
        # for offline reconstruction tests.
        manifest = compute_provenance(None, SimulatorOptions())
        assert manifest.system["system_id"] is None
        assert manifest.system["hash"] is None


# =====================================================================
# ``simulate(...)`` integration
# =====================================================================


class TestSimulateIntegration:
    def test_default_off_results_provenance_none(self):
        model, ctx = _make_scalar_linear()
        results = jaxonomy.simulate(
            model, ctx, (0.0, 0.5),
            options=SimulatorOptions(math_backend="numpy", enable_tracing=False),
        )
        assert results.provenance is None

    def test_opt_in_populates_results_provenance(self):
        model, ctx = _make_scalar_linear()
        opts = SimulatorOptions(
            math_backend="numpy",
            enable_tracing=False,
            record_provenance=True,
        )
        results = jaxonomy.simulate(model, ctx, (0.0, 0.5), options=opts)
        assert isinstance(results.provenance, ProvenanceManifest)
        # to_dict round-trips through json.
        d = results.provenance.to_dict()
        assert d["jaxonomy_version"]
        assert d["jax_version"]
        assert d["numpy_version"]
        assert "precision_info" in d
        assert d["timestamp"]
        json.loads(json.dumps(d))  # smoke serialise

    def test_default_off_byte_equivalent_outputs(self):
        # The outputs of two otherwise-identical simulations must match
        # bit-for-bit when ``record_provenance`` is off (control) vs
        # off (re-run).  Provenance must not perturb numerics either
        # way: but our primary guarantee is the default-off path is
        # untouched.
        model_a, ctx_a = _make_scalar_linear()
        model_b, ctx_b = _make_scalar_linear()
        opts = SimulatorOptions(math_backend="numpy", enable_tracing=False)
        r_a = jaxonomy.simulate(model_a, ctx_a, (0.0, 0.5), options=opts)
        r_b = jaxonomy.simulate(model_b, ctx_b, (0.0, 0.5), options=opts)

        assert r_a.provenance is None
        assert r_b.provenance is None
        # Final continuous state should match exactly — no provenance
        # path leaked into the trace.
        x_a = np.asarray(r_a.context.continuous_state)
        x_b = np.asarray(r_b.context.continuous_state)
        np.testing.assert_array_equal(x_a, x_b)

    def test_provenance_matches_python_compute(self):
        # The manifest attached by ``simulate`` should agree with what
        # ``compute_provenance`` produces for the same system on the
        # *post-_check_options* options (``simulate`` resolves things
        # like ``math_backend`` and auto-estimates ``max_major_steps``
        # before recording provenance).  We compare the deterministic
        # subset that is invariant under the resolution step.
        model, ctx = _make_scalar_linear()
        opts = SimulatorOptions(
            math_backend="numpy",
            enable_tracing=False,
            record_provenance=True,
        )
        results = jaxonomy.simulate(model, ctx, (0.0, 0.1), options=opts)
        attached = results.provenance
        rebuilt = compute_provenance(
            model, opts, timestamp=attached.timestamp,
        )
        # System fingerprint and version metadata are insensitive to
        # the internal options-resolution step.
        assert attached.system == rebuilt.system
        assert attached.precision_info == rebuilt.precision_info
        assert attached.jaxonomy_version == rebuilt.jaxonomy_version
        assert attached.jax_version == rebuilt.jax_version
        assert attached.numpy_version == rebuilt.numpy_version
        # ``record_provenance`` must round-trip into the captured options.
        assert attached.options["record_provenance"] is True
