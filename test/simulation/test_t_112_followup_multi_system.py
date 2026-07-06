# SPDX-License-Identifier: MIT

"""Tests for T-112-followup-multi-system.

Layered on top of T-112-followup-stateful, this followup adds a
*pool-of-diagrams* kernel cache to :class:`FastRestartSimulator`:

* ``run_with_diagram(diagram, parameters=..., initial_state=...,
  recorded_signals=...)`` keys the compiled kernel by ``id(diagram)``
  so users holding a ``dict[str, Diagram]`` (e.g. controller-variant
  pool) can rapidly switch between them without recompiling on every
  swap.  The first call for each distinct Diagram instance compiles;
  subsequent calls for the same instance reuse the cached kernel.

Default-off byte-equivalence: the existing :meth:`run` and
:meth:`reset` APIs are untouched (verified by the phase-1 +
stateful-followup test files still passing).
"""

from __future__ import annotations

import time

import jax.numpy as jnp
import pytest

import jaxonomy
from jaxonomy.library import Gain, Integrator, Sine
from jaxonomy.simulation import (
    FastRestartSimulator,
    SimulatorOptions,
    simulate,
)

pytestmark = pytest.mark.minimal


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _build_diagram(k: float = 1.0, name: str = "t112fu_multi_diag"):
    """Sine -> Gain(k) -> Integrator (single recorded output)."""
    builder = jaxonomy.DiagramBuilder()
    sine = builder.add(Sine(amplitude=1.0, frequency=1.0, name="sine"))
    gain = builder.add(Gain(gain=k, name="gain"))
    integ = builder.add(Integrator(initial_state=0.0, name="integ"))
    builder.connect(sine.output_ports[0], gain.input_ports[0])
    builder.connect(gain.output_ports[0], integ.input_ports[0])
    return builder.build(name=name)


def _opts(max_major_steps: int = 200) -> SimulatorOptions:
    return SimulatorOptions(
        math_backend="jax",
        max_major_steps=max_major_steps,
    )


def _new_sim(diag) -> FastRestartSimulator:
    """A FastRestartSimulator with diag's port as the default recorded signal."""
    return FastRestartSimulator(
        diag,
        t_span=(0.0, 0.5),
        options=_opts(),
        recorded_signals={"y": diag["integ"].output_ports[0]},
    )


# ---------------------------------------------------------------------------
# Per-diagram-identity kernel cache
# ---------------------------------------------------------------------------


class TestPerDiagramCache:
    def test_first_call_compiles_second_call_warm(self):
        """First run_with_diagram(diag_a) compiles; second is a cache hit."""
        diag_a = _build_diagram(k=1.0, name="diag_a")
        sim = _new_sim(diag_a)
        # First call: builds the kernel and caches it under id(diag_a).
        assert id(diag_a) not in sim._diagram_kernel_cache
        sim.run_with_diagram(
            diag_a,
            recorded_signals={"y": diag_a["integ"].output_ports[0]},
        )
        assert id(diag_a) in sim._diagram_kernel_cache
        kernel_a_first = sim._diagram_kernel_cache[id(diag_a)]["kernel"]
        # Second call: should reuse the exact same kernel object.
        sim.run_with_diagram(
            diag_a,
            recorded_signals={"y": diag_a["integ"].output_ports[0]},
        )
        kernel_a_second = sim._diagram_kernel_cache[id(diag_a)]["kernel"]
        assert kernel_a_first is kernel_a_second, (
            "second run_with_diagram(diag_a) must not recompile — the "
            "cached kernel object should be reused"
        )

    def test_distinct_diagrams_compile_separately(self):
        """run_with_diagram(diag_b) builds a *new* kernel for diag_b."""
        diag_a = _build_diagram(k=1.0, name="diag_a")
        diag_b = _build_diagram(k=10.0, name="diag_b")
        sim = _new_sim(diag_a)
        sim.run_with_diagram(
            diag_a,
            recorded_signals={"y": diag_a["integ"].output_ports[0]},
        )
        sim.run_with_diagram(
            diag_b,
            recorded_signals={"y": diag_b["integ"].output_ports[0]},
        )
        # Both cached, with distinct kernel objects.
        assert id(diag_a) in sim._diagram_kernel_cache
        assert id(diag_b) in sim._diagram_kernel_cache
        kernel_a = sim._diagram_kernel_cache[id(diag_a)]["kernel"]
        kernel_b = sim._diagram_kernel_cache[id(diag_b)]["kernel"]
        assert kernel_a is not kernel_b, (
            "diag_a and diag_b must compile to distinct kernels"
        )

    def test_round_trip_diag_a_is_warm(self):
        """Going A -> B -> A reuses the original cached kernel for A."""
        diag_a = _build_diagram(k=1.0, name="diag_a")
        diag_b = _build_diagram(k=10.0, name="diag_b")
        sim = _new_sim(diag_a)
        # Burn the compile for A.
        sim.run_with_diagram(
            diag_a,
            recorded_signals={"y": diag_a["integ"].output_ports[0]},
        )
        kernel_a_orig = sim._diagram_kernel_cache[id(diag_a)]["kernel"]
        # Now B.
        sim.run_with_diagram(
            diag_b,
            recorded_signals={"y": diag_b["integ"].output_ports[0]},
        )
        # Back to A — must still be the original kernel.
        sim.run_with_diagram(
            diag_a,
            recorded_signals={"y": diag_a["integ"].output_ports[0]},
        )
        kernel_a_after = sim._diagram_kernel_cache[id(diag_a)]["kernel"]
        assert kernel_a_orig is kernel_a_after, (
            "round-trip A -> B -> A must reuse the originally-cached "
            "kernel for A"
        )

    def test_round_trip_is_substantially_faster_than_recompile(self):
        """A round-trip swap via run_with_diagram is faster than a recompile.

        We compare:
        * the *first* call for diag_b (cold compile),
        * a *return* call for diag_a after a diag_b detour (cache hit).

        The cache-hit call should be much faster than the cold compile.
        """
        diag_a = _build_diagram(k=1.0, name="diag_a")
        diag_b = _build_diagram(k=10.0, name="diag_b")
        sim = _new_sim(diag_a)

        # Pre-warm A so its compile cost isn't measured.
        sim.run_with_diagram(
            diag_a,
            recorded_signals={"y": diag_a["integ"].output_ports[0]},
        )
        # Cold compile for B.
        t0 = time.perf_counter()
        rb = sim.run_with_diagram(
            diag_b,
            recorded_signals={"y": diag_b["integ"].output_ports[0]},
        )
        _ = float(rb.time[-1])  # device sync
        cold_b = time.perf_counter() - t0

        # Cache-hit return to A.
        t0 = time.perf_counter()
        ra = sim.run_with_diagram(diag_a)
        _ = float(ra.time[-1])  # device sync
        warm_a = time.perf_counter() - t0

        assert cold_b > 1e-3, (
            f"cold compile for diag_b too fast to time reliably: {cold_b:.6f}s"
        )
        assert warm_a < 0.5 * cold_b, (
            f"return-to-A cache hit not faster than B's cold compile: "
            f"cold_b={cold_b:.4f}s, warm_a={warm_a:.4f}s, "
            f"ratio={warm_a / cold_b:.3f}"
        )


# ---------------------------------------------------------------------------
# Numerical equivalence with simulate()
# ---------------------------------------------------------------------------


class TestNumericalEquivalence:
    def test_run_with_diagram_matches_simulate(self):
        """run_with_diagram(diag) outputs equal a cold simulate(diag, ...)."""
        diag = _build_diagram(k=1.7, name="numeq_diag")
        recorded = {"y": diag["integ"].output_ports[0]}

        # Reference: cold simulate().
        ctx = diag.create_context()
        ref = simulate(
            diag,
            ctx,
            (0.0, 0.5),
            options=_opts(),
            recorded_signals=recorded,
        )

        # Warm path via run_with_diagram.
        sim = _new_sim(diag)
        got = sim.run_with_diagram(diag, recorded_signals=recorded)

        assert got.outputs["y"].shape == ref.outputs["y"].shape
        assert jnp.allclose(got.outputs["y"], ref.outputs["y"], atol=1e-6), (
            f"run_with_diagram diverges from simulate: max diff = "
            f"{float(jnp.max(jnp.abs(got.outputs['y'] - ref.outputs['y']))):.3e}"
        )

    def test_pool_distinguishes_diagrams(self):
        """Two pool diagrams with different gains produce different peaks."""
        diag_a = _build_diagram(k=0.5, name="pool_small")
        diag_b = _build_diagram(k=2.0, name="pool_big")
        sim = _new_sim(diag_a)
        ra = sim.run_with_diagram(
            diag_a,
            recorded_signals={"y": diag_a["integ"].output_ports[0]},
        )
        rb = sim.run_with_diagram(
            diag_b,
            recorded_signals={"y": diag_b["integ"].output_ports[0]},
        )
        peak_a = float(jnp.max(jnp.abs(ra.outputs["y"])))
        peak_b = float(jnp.max(jnp.abs(rb.outputs["y"])))
        assert peak_b > peak_a, (
            f"diag_b (k=2) should integrate to a larger peak than "
            f"diag_a (k=0.5); got peak_a={peak_a:.3f}, peak_b={peak_b:.3f}"
        )


# ---------------------------------------------------------------------------
# Interaction with parameters= and initial_state=
# ---------------------------------------------------------------------------


class TestPerCallOverrides:
    def test_initial_state_override_on_cached_diagram(self):
        """initial_state= still works when going through run_with_diagram."""
        diag = _build_diagram(k=1.0, name="ic_diag")
        sim = FastRestartSimulator(
            diag,
            t_span=(0.0, 0.001),  # short — first sample == initial state
            options=_opts(max_major_steps=4),
            recorded_signals={"y": diag["integ"].output_ports[0]},
        )
        # Warm the cache.
        sim.run_with_diagram(diag)
        r = sim.run_with_diagram(diag, initial_state=jnp.float64(5.0))
        assert abs(float(r.outputs["y"][0]) - 5.0) < 1e-6


# ---------------------------------------------------------------------------
# Default-off / non-regression
# ---------------------------------------------------------------------------


class TestDefaultOff:
    def test_existing_run_api_unchanged(self):
        """The original .run() path still works and doesn't touch the new cache."""
        diag = _build_diagram(k=1.0, name="defoff_diag")
        sim = _new_sim(diag)
        # The new diagram-keyed cache must start empty.
        assert sim._diagram_kernel_cache == {}
        assert sim._diagram_pool == {}
        sim.run(parameters={"gain.gain": jnp.float64(1.0)})
        # ``run()`` populates the *legacy* kernel slot, not the
        # per-diagram cache.
        assert sim._kernel is not None
        assert sim._diagram_kernel_cache == {}
        assert sim._diagram_pool == {}

    def test_close_clears_diagram_cache(self):
        """close() also drops the per-diagram cache (housekeeping)."""
        diag_a = _build_diagram(k=1.0, name="a")
        diag_b = _build_diagram(k=2.0, name="b")
        sim = _new_sim(diag_a)
        sim.run_with_diagram(
            diag_a,
            recorded_signals={"y": diag_a["integ"].output_ports[0]},
        )
        sim.run_with_diagram(
            diag_b,
            recorded_signals={"y": diag_b["integ"].output_ports[0]},
        )
        assert len(sim._diagram_kernel_cache) == 2
        sim.close()
        assert sim._diagram_kernel_cache == {}
        assert sim._diagram_pool == {}

    def test_missing_recorded_signals_on_first_call_raises(self):
        """First call for a brand-new diagram with no recorded_signals raises.

        Specifically: if ``self.recorded_signals`` is bound to a
        *different* diagram's ports (the common pool case), the user
        must supply recorded_signals on the first run_with_diagram for
        each pool member.  We don't auto-detect the mismatch — but we
        do require *some* recorded_signals dict to exist.
        """
        diag_a = _build_diagram(k=1.0, name="raise_a")
        sim = _new_sim(diag_a)
        # Manually clear ``self.recorded_signals`` to simulate the
        # "no fallback available" case.
        sim.recorded_signals = {}
        diag_b = _build_diagram(k=2.0, name="raise_b")
        with pytest.raises(ValueError, match="recorded_signals"):
            sim.run_with_diagram(diag_b)
