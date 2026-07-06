# SPDX-License-Identifier: MIT

"""Tests for T-110-followup-attach-on-batch.

Covers attaching a single shared :class:`ProvenanceManifest` to the
:class:`BatchSimulationResults` returned by :func:`simulate_batch` and
:func:`simulate_distributed` when ``SimulatorOptions.record_provenance=True``.

* Default-off path (``record_provenance=False``) → ``results.provenance is None``.
* Opt-in path → manifest populated with library versions, system fingerprint,
  options snapshot.
* The manifest is one shared object — running the same diagram + options
  twice yields manifests with identical system fingerprints (modulo
  timestamp).
* :func:`attach_provenance_to_batch` standalone helper populates an
  existing :class:`BatchSimulationResults` whose ``provenance`` is ``None``.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

import jaxonomy
from jaxonomy import (
    DiagramBuilder,
    LeafSystem,
    SimulatorOptions,
    simulate_batch,
    simulate_distributed,
)
from jaxonomy.simulation.batch import (
    BatchSimulationResults,
    attach_provenance_to_batch,
)
from jaxonomy.simulation.provenance import ProvenanceManifest


pytestmark = pytest.mark.minimal


# ---------------------------------------------------------------------------
# Tiny decay diagram used by every test in this file.  Identical in shape to
# the diagram in ``test_distributed.py`` so the batch / distributed paths
# share semantics.
# ---------------------------------------------------------------------------


class _Decay(LeafSystem):
    """Scalar exponential decay xdot = -k * x."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.declare_dynamic_parameter("k", 1.0)
        self.declare_continuous_state(
            default_value=jnp.array(1.0), ode=self._ode
        )
        self.declare_continuous_state_output(name="x")

    def _ode(self, t, state, **p):
        return -p["k"] * state.continuous_state


def _build_diagram():
    db = DiagramBuilder()
    db.add(_Decay(name="leaf"))
    return db.build(name="root")


def _opts(*, record_provenance: bool = False) -> SimulatorOptions:
    return SimulatorOptions(
        math_backend="jax",
        ode_solver_method="dopri5",
        max_major_steps=64,
        return_context=False,
        record_provenance=record_provenance,
    )


# ---------------------------------------------------------------------------
# simulate_batch
# ---------------------------------------------------------------------------


class TestSimulateBatchProvenance:
    def test_default_off_provenance_is_none(self):
        sys = _build_diagram()
        rec = {"x": sys["leaf"].output_ports[0]}
        ks = jnp.linspace(0.5, 2.0, 4)
        result = simulate_batch(
            sys,
            t_span=(0.0, 0.5),
            param_batches={"leaf.k": ks},
            options=_opts(record_provenance=False),
            recorded_signals=rec,
        )
        assert result.provenance is None

    def test_opt_in_populates_manifest(self):
        sys = _build_diagram()
        rec = {"x": sys["leaf"].output_ports[0]}
        ks = jnp.linspace(0.5, 2.0, 4)
        result = simulate_batch(
            sys,
            t_span=(0.0, 0.5),
            param_batches={"leaf.k": ks},
            options=_opts(record_provenance=True),
            recorded_signals=rec,
        )
        assert isinstance(result.provenance, ProvenanceManifest)
        assert result.provenance.jaxonomy_version
        assert result.provenance.jax_version
        assert result.provenance.numpy_version
        assert "default_float_dtype" in result.provenance.precision_info
        # System fingerprint is recorded.
        assert result.provenance.system["hash"] is not None
        # The options dict captures ``record_provenance=True`` so an
        # auditor can tell the manifest was opt-in.
        assert result.provenance.options["record_provenance"] is True

    def test_same_params_same_provenance_system_hash(self):
        """Running the same diagram + options twice yields manifests with
        identical system fingerprints (modulo timestamp).  Provenance is
        about reproducibility, not run-uniqueness."""
        sys = _build_diagram()
        rec = {"x": sys["leaf"].output_ports[0]}
        ks = jnp.linspace(0.5, 2.0, 4)

        r1 = simulate_batch(
            sys,
            t_span=(0.0, 0.5),
            param_batches={"leaf.k": ks},
            options=_opts(record_provenance=True),
            recorded_signals=rec,
        )
        r2 = simulate_batch(
            sys,
            t_span=(0.0, 0.5),
            param_batches={"leaf.k": ks},
            options=_opts(record_provenance=True),
            recorded_signals=rec,
        )
        assert r1.provenance is not None
        assert r2.provenance is not None
        assert r1.provenance.system["hash"] == r2.provenance.system["hash"]
        assert r1.provenance.options == r2.provenance.options

    def test_loop_path_opt_in_populates_manifest(self):
        """Force the safe Python-loop path and verify provenance still
        attaches exactly once (no per-replica overhead, single manifest)."""
        sys = _build_diagram()
        rec = {"x": sys["leaf"].output_ports[0]}
        ks = jnp.linspace(0.5, 2.0, 3)
        result = simulate_batch(
            sys,
            t_span=(0.0, 0.5),
            param_batches={"leaf.k": ks},
            options=_opts(record_provenance=True),
            recorded_signals=rec,
            _force_loop=True,
        )
        assert isinstance(result.provenance, ProvenanceManifest)


# ---------------------------------------------------------------------------
# simulate_distributed (1-device degenerate path always exercised)
# ---------------------------------------------------------------------------


class TestSimulateDistributedProvenance:
    def test_default_off_provenance_is_none(self):
        sys = _build_diagram()
        rec = {"x": sys["leaf"].output_ports[0]}
        ks = jnp.linspace(0.5, 2.0, 4)
        result = simulate_distributed(
            sys,
            t_span=(0.0, 0.5),
            param_batches={"leaf.k": ks},
            options=_opts(record_provenance=False),
            recorded_signals=rec,
            devices=[jax.devices()[0]],
        )
        assert result.provenance is None

    def test_opt_in_populates_manifest(self):
        sys = _build_diagram()
        rec = {"x": sys["leaf"].output_ports[0]}
        ks = jnp.linspace(0.5, 2.0, 4)
        result = simulate_distributed(
            sys,
            t_span=(0.0, 0.5),
            param_batches={"leaf.k": ks},
            options=_opts(record_provenance=True),
            recorded_signals=rec,
            devices=[jax.devices()[0]],
        )
        assert isinstance(result.provenance, ProvenanceManifest)
        assert result.provenance.jaxonomy_version
        assert result.provenance.jax_version
        assert result.provenance.numpy_version
        assert result.provenance.system["hash"] is not None
        assert result.provenance.options["record_provenance"] is True


# ---------------------------------------------------------------------------
# attach_provenance_to_batch standalone helper
# ---------------------------------------------------------------------------


class TestAttachProvenanceToBatch:
    def test_attach_populates_provenance(self):
        sys = _build_diagram()
        # A bare result (no provenance) like one could get from a batch run
        # done before provenance was recorded.
        bare = BatchSimulationResults(time=None, outputs={}, used_vmap=False)
        assert bare.provenance is None

        attached = attach_provenance_to_batch(bare, sys, _opts())
        # In-place + returns same instance for convenience.
        assert attached is bare
        assert isinstance(bare.provenance, ProvenanceManifest)
        assert bare.provenance.system["type"] is not None
        assert bare.provenance.jaxonomy_version
