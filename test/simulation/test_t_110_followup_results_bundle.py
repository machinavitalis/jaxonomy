# SPDX-License-Identifier: MIT

"""Tests for T-110-followup-results-bundle — ``ResultsWithProvenance``.

Covers:

* :func:`bundle_results` returns a :class:`ResultsWithProvenance` when
  the underlying results object carries a populated ``provenance`` field.
* :func:`bundle_results` returns the *original* results object unchanged
  when ``provenance`` is None or absent — preserves the byte-equivalent
  default-off path.
* The wrapper forwards attribute access to the underlying results
  (``wrapped.outputs[name]`` works, ``wrapped.time`` works, etc.).
* ``__repr__`` mentions both sides so the wrapper is self-describing.
* End-to-end with ``simulate(...)`` + ``record_provenance=True``: bundle
  the result, exercise both ``.results`` / ``.provenance`` accessors and
  the forwarded-attribute path on a real ``SimulationResults`` instance.
"""

from __future__ import annotations

import numpy as np
import pytest

import jaxonomy
from jaxonomy.backend import numpy_api as npa
from jaxonomy.simulation import (
    bundle_results,
    ProvenanceManifest,
    ResultsWithProvenance,
    SimulationResults,
    SimulatorOptions,
)
from jaxonomy.simulation.provenance import compute_provenance


pytestmark = pytest.mark.minimal


# ---------------------------------------------------------------------
# Tiny scalar-linear leaf reused for the simulate-integration test.
# Mirrors the helper from ``test_t_110_provenance_phase1.py`` so that
# the integration test path exercises the same byte-equivalent surface
# that the rest of the T-110 suite already validates.
# ---------------------------------------------------------------------


def _make_scalar_linear():
    a = 1.5

    class ScalarLinear(jaxonomy.LeafSystem):
        def __init__(self):
            super().__init__(name="ScalarLinear")
            self.declare_continuous_state(
                shape=(), ode=self.ode, dtype=npa.float64,
            )
            self.declare_continuous_state_output(name="x")

        def ode(self, time, state):
            xc = state.continuous_state
            return -a * xc

    model = ScalarLinear()
    ctx = model.create_context(time=0.0)
    ctx = ctx.with_continuous_state(npa.float64(2.0))
    return model, ctx


# =====================================================================
# Direct ``bundle_results`` API — no simulate dependency
# =====================================================================


class TestBundleResultsDirect:
    """Exercise ``bundle_results`` with hand-built results stand-ins so
    the wrapper logic is testable in isolation from the simulator."""

    def test_passthrough_when_provenance_none(self):
        # A bare ``SimulationResults`` with provenance left as the
        # default ``None`` should round-trip through bundle_results
        # unchanged — the helper must not wrap the no-provenance case.
        results = SimulationResults(
            context=None,
            time=np.array([0.0, 1.0]),
            outputs={"x": np.array([2.0, 0.5])},
        )
        bundled = bundle_results(results)
        assert bundled is results

    def test_passthrough_when_no_provenance_attribute(self):
        # An object that doesn't even *have* a ``provenance`` attribute
        # (e.g. a third-party results-like type) should also round-trip
        # unchanged — ``getattr(..., None)`` keeps the helper safe.
        class BareResults:
            outputs = {"x": np.array([1.0, 2.0])}

        results = BareResults()
        bundled = bundle_results(results)
        assert bundled is results

    def test_wraps_when_provenance_present(self):
        manifest = compute_provenance(
            None, SimulatorOptions(), include_git=False,
        )
        results = SimulationResults(
            context=None,
            time=np.array([0.0, 1.0]),
            outputs={"x": np.array([2.0, 0.5])},
            provenance=manifest,
        )
        bundled = bundle_results(results)

        assert isinstance(bundled, ResultsWithProvenance)
        # ``.results`` / ``.provenance`` accessors expose both sides.
        assert bundled.results is results
        assert bundled.provenance is manifest

    def test_attribute_forwarding(self):
        manifest = compute_provenance(
            None, SimulatorOptions(), include_git=False,
        )
        outputs = {"x": np.array([2.0, 0.5])}
        time = np.array([0.0, 1.0])
        results = SimulationResults(
            context=None,
            time=time,
            outputs=outputs,
            provenance=manifest,
        )
        bundled = bundle_results(results)

        # Forwarded attribute lookup — exactly the underlying object.
        assert bundled.outputs is outputs
        np.testing.assert_array_equal(bundled.time, time)
        np.testing.assert_array_equal(bundled.outputs["x"], outputs["x"])
        # The wrapper itself does not shadow ``provenance`` — both the
        # field and the forwarded attribute resolve to the same object.
        assert bundled.provenance is manifest

    def test_attribute_error_on_missing(self):
        # Lookups for genuinely-missing attributes should raise
        # ``AttributeError`` (NOT swallow), and the message should
        # mention the wrapper class so the user knows where to look.
        manifest = compute_provenance(
            None, SimulatorOptions(), include_git=False,
        )
        results = SimulationResults(
            context=None,
            time=np.array([0.0, 1.0]),
            outputs={"x": np.array([2.0, 0.5])},
            provenance=manifest,
        )
        bundled = bundle_results(results)

        with pytest.raises(AttributeError) as excinfo:
            _ = bundled.does_not_exist_on_either_side
        assert "ResultsWithProvenance" in str(excinfo.value)

    def test_repr_mentions_both_sides(self):
        manifest = compute_provenance(
            None, SimulatorOptions(), include_git=False,
        )
        results = SimulationResults(
            context=None,
            time=np.array([0.0, 1.0]),
            outputs={"x": np.array([2.0, 0.5])},
            provenance=manifest,
        )
        bundled = bundle_results(results)

        text = repr(bundled)
        assert "ResultsWithProvenance" in text
        assert "results=" in text
        assert "provenance=" in text

    def test_wrapper_is_frozen(self):
        # The wrapper is a frozen dataclass; mutating either side
        # should raise ``FrozenInstanceError`` so the (results,
        # provenance) pairing can't drift apart silently.
        import dataclasses

        manifest = compute_provenance(
            None, SimulatorOptions(), include_git=False,
        )
        results = SimulationResults(
            context=None,
            time=np.array([0.0, 1.0]),
            outputs={"x": np.array([2.0, 0.5])},
            provenance=manifest,
        )
        bundled = bundle_results(results)

        with pytest.raises(dataclasses.FrozenInstanceError):
            bundled.results = None  # type: ignore[misc]


# =====================================================================
# End-to-end: ``simulate(...)`` → ``bundle_results``
# =====================================================================


class TestBundleResultsSimulateIntegration:
    """Exercise the wrapper against the real ``SimulationResults`` that
    ``simulate(...)`` produces with ``record_provenance=True``."""

    def test_simulate_then_bundle(self):
        model, ctx = _make_scalar_linear()
        opts = SimulatorOptions(
            enable_tracing=False,
            record_provenance=True,
        )
        results = jaxonomy.simulate(
            model, ctx, (0.0, 0.5),
            options=opts,
            recorded_signals={"x": model.output_ports[0]},
        )
        # Sanity: legacy ``.provenance`` field path still works.
        assert isinstance(results.provenance, ProvenanceManifest)

        bundled = bundle_results(results)
        assert isinstance(bundled, ResultsWithProvenance)
        # The wrapper exposes both sides + forwards attribute access.
        assert bundled.results is results
        assert bundled.provenance is results.provenance
        assert bundled.outputs is results.outputs
        # Forwarded ``.outputs[...]`` indexing is the headline ergonomic
        # the wrapper exists for — exercise it explicitly.
        np.testing.assert_array_equal(
            bundled.outputs["x"], results.outputs["x"],
        )

    def test_simulate_default_off_passthrough(self):
        # When the simulator was run with the default
        # ``record_provenance=False``, ``bundle_results`` must return
        # the original results unchanged so legacy callers can adopt
        # the helper without changing their default-off code paths.
        model, ctx = _make_scalar_linear()
        results = jaxonomy.simulate(
            model, ctx, (0.0, 0.5),
            options=SimulatorOptions(enable_tracing=False),
            recorded_signals={"x": model.output_ports[0]},
        )
        assert results.provenance is None
        bundled = bundle_results(results)
        assert bundled is results
