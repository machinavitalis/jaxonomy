# SPDX-License-Identifier: MIT

"""Tests for T-112 Phase 1 — Fast Restart (warm-start single-simulation loop).

Covers:

* :class:`FastRestartSimulator` reuses the JIT cache across calls — second
  ``run()`` is markedly faster than the first.
* Warm-restart numerical results match a cold :func:`simulate` call on
  the same parameters.
* Different parameter dicts produce correctly-differentiated results
  across runs of the same simulator instance.
* Default :func:`simulate` path is untouched (smoke check).
* The :func:`fast_restart` generator helper yields one
  :class:`SimulationResults` per parameter dict.
* Context-manager protocol works (``__enter__`` / ``__exit__``).
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
    fast_restart,
    simulate,
)

pytestmark = pytest.mark.minimal


# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------


def _build_diagram(k: float = 1.0):
    """Sine -> Gain(k) -> Integrator (single recorded output)."""
    builder = jaxonomy.DiagramBuilder()
    sine = builder.add(Sine(amplitude=1.0, frequency=1.0, name="sine"))
    gain = builder.add(Gain(gain=k, name="gain"))
    integ = builder.add(Integrator(initial_state=0.0, name="integ"))
    builder.connect(sine.output_ports[0], gain.input_ports[0])
    builder.connect(gain.output_ports[0], integ.input_ports[0])
    return builder.build(name="t112_diag")


def _opts(max_major_steps: int = 200) -> SimulatorOptions:
    return SimulatorOptions(
        math_backend="jax",
        max_major_steps=max_major_steps,
    )


# ---------------------------------------------------------------------------
# Construction & validation
# ---------------------------------------------------------------------------


class TestConstruction:
    def test_requires_recorded_signals(self):
        diag = _build_diagram()
        with pytest.raises(ValueError, match="recorded_signals"):
            FastRestartSimulator(diag, t_span=(0.0, 1.0), options=_opts())

    def test_rejects_non_jax_backend(self):
        diag = _build_diagram()
        bad_opts = SimulatorOptions(math_backend="numpy", max_major_steps=100)
        with pytest.raises(ValueError, match="math_backend"):
            FastRestartSimulator(
                diag,
                t_span=(0.0, 1.0),
                options=bad_opts,
                recorded_signals={"y": diag["integ"].output_ports[0]},
            )

    def test_rejects_disabled_tracing(self):
        diag = _build_diagram()
        bad_opts = SimulatorOptions(
            math_backend="jax", max_major_steps=100, enable_tracing=False,
        )
        with pytest.raises(ValueError, match="enable_tracing"):
            FastRestartSimulator(
                diag,
                t_span=(0.0, 1.0),
                options=bad_opts,
                recorded_signals={"y": diag["integ"].output_ports[0]},
            )


# ---------------------------------------------------------------------------
# Core warm-start mechanics
# ---------------------------------------------------------------------------


class TestWarmStart:
    def test_runs_increment_counter(self):
        diag = _build_diagram()
        sim = FastRestartSimulator(
            diag,
            t_span=(0.0, 0.5),
            options=_opts(),
            recorded_signals={"y": diag["integ"].output_ports[0]},
        )
        assert sim.n_runs == 0
        sim.run()
        assert sim.n_runs == 1
        sim.run(parameters={"gain.gain": jnp.float64(2.0)})
        assert sim.n_runs == 2

    def test_warm_call_faster_than_cold(self):
        """Second run() should be substantially faster than the first.

        The first call pays JIT-compile cost; the second hits the JAX
        cache.  We assert a generous ratio (warm < 0.5 * cold) to avoid
        flakiness — in practice the speedup is much larger (10x+) on a
        cold cache.
        """
        diag = _build_diagram()
        sim = FastRestartSimulator(
            diag,
            t_span=(0.0, 1.0),
            options=_opts(),
            recorded_signals={"y": diag["integ"].output_ports[0]},
        )

        # Warm run: includes JIT-compile of advance_to.
        t0 = time.perf_counter()
        r_cold = sim.run(parameters={"gain.gain": jnp.float64(1.0)})
        # Block on the result by inspecting the time vector (forces
        # device sync — JAX ops are async by default).
        _ = float(r_cold.time[-1])
        cold = time.perf_counter() - t0

        # Warm run: kernel cached, only context patch + dispatch.
        t0 = time.perf_counter()
        r_warm = sim.run(parameters={"gain.gain": jnp.float64(1.5)})
        _ = float(r_warm.time[-1])
        warm = time.perf_counter() - t0

        # The first call must have actually done meaningful work
        # (otherwise the ratio is meaningless).
        assert cold > 1e-3, f"cold run too fast to time reliably: {cold:.6f}s"
        assert warm < 0.5 * cold, (
            f"warm restart not faster than cold: cold={cold:.4f}s, "
            f"warm={warm:.4f}s, ratio={warm / cold:.3f}"
        )

    def test_warm_run_matches_cold_simulate(self):
        """A warm restart's outputs must equal a cold simulate() on the same params.

        This is the correctness guarantee: kernel reuse must not perturb
        numerics — the JIT cache key is the abstract pytree shape, not
        the values.
        """
        diag = _build_diagram()
        recorded = {"y": diag["integ"].output_ports[0]}
        opts = _opts()

        # Cold reference via simulate(): rebuild a parameterised diagram
        # and call simulate from scratch.
        diag_cold = diag.with_parameters({"gain.gain": jnp.float64(1.7)})
        ctx_cold = diag_cold.create_context()
        ref = simulate(
            diag_cold,
            ctx_cold,
            (0.0, 1.0),
            options=opts,
            recorded_signals={"y": diag_cold["integ"].output_ports[0]},
        )

        # Warm path via FastRestartSimulator on the original (k=1) diagram,
        # patched per-run to gain=1.7.
        sim = FastRestartSimulator(
            diag, t_span=(0.0, 1.0), options=opts, recorded_signals=recorded,
        )
        # Burn one call to compile the kernel.
        _ = sim.run(parameters={"gain.gain": jnp.float64(1.0)})
        # Warm call with the test parameter value.
        warm = sim.run(parameters={"gain.gain": jnp.float64(1.7)})

        # Time grids should match exactly (same kernel, same advance_to
        # decisions for the same context shape — the only difference is
        # the gain value, which doesn't affect the step schedule for
        # this linear system at this tolerance).
        assert warm.outputs["y"].shape == ref.outputs["y"].shape
        # Numerical match (allow a small tolerance for floating-point
        # reordering of independent ops).
        assert jnp.allclose(warm.outputs["y"], ref.outputs["y"], atol=1e-6), (
            f"warm restart diverges from cold simulate: max diff = "
            f"{float(jnp.max(jnp.abs(warm.outputs['y'] - ref.outputs['y']))):.3e}"
        )

    def test_each_run_applies_its_own_params(self):
        """Successive runs must reflect their own parameter values, not
        leak state from the previous call."""
        diag = _build_diagram()
        sim = FastRestartSimulator(
            diag,
            t_span=(0.0, 1.0),
            options=_opts(),
            recorded_signals={"y": diag["integ"].output_ports[0]},
        )

        results = []
        for k in (jnp.float64(0.5), jnp.float64(1.0), jnp.float64(1.5)):
            r = sim.run(parameters={"gain.gain": k})
            results.append(r)

        y_small = results[0].outputs["y"]
        y_med = results[1].outputs["y"]
        y_large = results[2].outputs["y"]

        # The integrator's amplitude scales with gain — so runs with
        # larger gain should have larger absolute peaks.
        peak_small = float(jnp.max(jnp.abs(y_small)))
        peak_med = float(jnp.max(jnp.abs(y_med)))
        peak_large = float(jnp.max(jnp.abs(y_large)))

        assert peak_small < peak_med < peak_large, (
            f"runs did not apply distinct parameters: peaks "
            f"({peak_small:.3f}, {peak_med:.3f}, {peak_large:.3f})"
        )

    def test_no_params_runs_with_base_context(self):
        """Calling run() without parameters uses the base context (gain=1.0)."""
        diag = _build_diagram(k=1.0)
        sim = FastRestartSimulator(
            diag,
            t_span=(0.0, 0.5),
            options=_opts(),
            recorded_signals={"y": diag["integ"].output_ports[0]},
        )
        r = sim.run()  # no parameters arg
        assert r.outputs["y"] is not None
        assert r.outputs["y"].shape[0] > 1


# ---------------------------------------------------------------------------
# Context-manager protocol
# ---------------------------------------------------------------------------


class TestContextManager:
    def test_with_block(self):
        diag = _build_diagram()
        with FastRestartSimulator(
            diag,
            t_span=(0.0, 0.5),
            options=_opts(),
            recorded_signals={"y": diag["integ"].output_ports[0]},
        ) as sim:
            r = sim.run(parameters={"gain.gain": jnp.float64(1.0)})
            assert r.outputs["y"].shape[0] > 1
        # After __exit__, internal kernel reference is dropped; calling
        # run() again should still work (re-build via JAX cache).
        assert sim._kernel is None

    def test_close_then_rerun(self):
        diag = _build_diagram()
        sim = FastRestartSimulator(
            diag,
            t_span=(0.0, 0.5),
            options=_opts(),
            recorded_signals={"y": diag["integ"].output_ports[0]},
        )
        sim.run(parameters={"gain.gain": jnp.float64(1.0)})
        sim.close()
        assert sim._kernel is None
        # run() rebuilds.
        sim.run(parameters={"gain.gain": jnp.float64(1.5)})
        assert sim._kernel is not None


# ---------------------------------------------------------------------------
# Functional helper
# ---------------------------------------------------------------------------


class TestFastRestartGenerator:
    def test_yields_one_per_dict(self):
        diag = _build_diagram()
        grid = [
            {"gain.gain": jnp.float64(0.5)},
            {"gain.gain": jnp.float64(1.0)},
            {"gain.gain": jnp.float64(1.5)},
        ]
        results = list(
            fast_restart(
                diag,
                t_span=(0.0, 0.5),
                parameter_grid=grid,
                options=_opts(),
                recorded_signals={"y": diag["integ"].output_ports[0]},
            )
        )
        assert len(results) == 3
        for r in results:
            assert r.outputs["y"].shape[0] > 1
        # Distinct parameter values produce distinct trajectories.
        peaks = [float(jnp.max(jnp.abs(r.outputs["y"]))) for r in results]
        assert peaks[0] < peaks[1] < peaks[2]


# ---------------------------------------------------------------------------
# Default-off / non-regression smoke
# ---------------------------------------------------------------------------


class TestDefaultOffPath:
    def test_simulate_unchanged(self):
        """Importing FastRestartSimulator does not perturb simulate()."""
        diag = _build_diagram(k=1.5)
        ctx = diag.create_context()
        r = simulate(
            diag,
            ctx,
            (0.0, 0.5),
            options=_opts(),
            recorded_signals={"y": diag["integ"].output_ports[0]},
        )
        assert r.outputs["y"] is not None
        assert r.outputs["y"].shape[0] > 1
        # No new fields populated (provenance defaulted off, etc.).
        assert r.provenance is None
