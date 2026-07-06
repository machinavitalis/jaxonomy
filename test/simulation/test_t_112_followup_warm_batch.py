# SPDX-License-Identifier: MIT

"""Tests for T-112-followup-warm-batch.

Layered on top of phase 1 + the multi-system / stateful followups, this
followup adds :meth:`FastRestartSimulator.run_batch` — a batched
parameter-sweep API that vmaps the warm-cached kernel.

Coverage:

* ``run_batch`` with N distinct parameter values produces N distinct
  trajectories.
* A single-row batch (``{"K": jnp.array([1.0])}``) matches a scalar
  :meth:`run` (``parameters={"K": 1.0}``) within tolerance.
* The cached kernel is reused — calling :meth:`run` first and then
  :meth:`run_batch` does not rebuild ``_kernel``.
* ``initial_states_batch`` is honored on a per-row basis.
* Importing FastRestartSimulator does not perturb the default
  :func:`simulate` path (byte-equivalence smoke).
"""

from __future__ import annotations

import jax.numpy as jnp
import pytest

import jaxonomy
from jaxonomy.library import Gain, Integrator, Sine
from jaxonomy.simulation import (
    BatchSimulationResults,
    FastRestartSimulator,
    SimulatorOptions,
    simulate,
)

pytestmark = pytest.mark.minimal


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _build_diagram(k: float = 1.0, x0: float = 0.0):
    """Sine -> Gain(k) -> Integrator(x0) (single recorded output)."""
    builder = jaxonomy.DiagramBuilder()
    sine = builder.add(Sine(amplitude=1.0, frequency=1.0, name="sine"))
    gain = builder.add(Gain(gain=k, name="gain"))
    integ = builder.add(Integrator(initial_state=x0, name="integ"))
    builder.connect(sine.output_ports[0], gain.input_ports[0])
    builder.connect(gain.output_ports[0], integ.input_ports[0])
    return builder.build(name="t112fu_warm_batch_diag")


def _opts(max_major_steps: int = 200) -> SimulatorOptions:
    return SimulatorOptions(
        math_backend="jax",
        max_major_steps=max_major_steps,
    )


# ---------------------------------------------------------------------------
# Core run_batch mechanics
# ---------------------------------------------------------------------------


class TestRunBatch:
    def test_distinct_params_produce_distinct_results(self):
        """Three batch rows with three different gains produce three trajectories."""
        diag = _build_diagram()
        sim = FastRestartSimulator(
            diag,
            t_span=(0.0, 1.0),
            options=_opts(),
            recorded_signals={"y": diag["integ"].output_ports[0]},
        )

        ks = jnp.asarray([0.5, 1.0, 1.5])
        result = sim.run_batch({"gain.gain": ks})

        assert isinstance(result, BatchSimulationResults)
        assert "y" in result.outputs
        # Shape: (N=3, T)
        assert result.outputs["y"].shape[0] == 3
        assert result.outputs["y"].shape[1] > 1

        # Peaks should be monotonically increasing in gain.
        peaks = [float(jnp.max(jnp.abs(result.outputs["y"][i]))) for i in range(3)]
        assert peaks[0] < peaks[1] < peaks[2], (
            f"batch rows did not differ by gain: peaks={peaks}"
        )

    def test_n_runs_counter_bumps_by_batch_size(self):
        """``n_runs`` increments by N on a batch call."""
        diag = _build_diagram()
        sim = FastRestartSimulator(
            diag,
            t_span=(0.0, 0.5),
            options=_opts(),
            recorded_signals={"y": diag["integ"].output_ports[0]},
        )
        assert sim.n_runs == 0
        sim.run_batch({"gain.gain": jnp.asarray([0.5, 1.0, 1.5, 2.0])})
        assert sim.n_runs == 4

    def test_rejects_empty_batch(self):
        diag = _build_diagram()
        sim = FastRestartSimulator(
            diag,
            t_span=(0.0, 0.5),
            options=_opts(),
            recorded_signals={"y": diag["integ"].output_ports[0]},
        )
        with pytest.raises(ValueError, match="non-empty"):
            sim.run_batch({})


# ---------------------------------------------------------------------------
# Single-row batch matches scalar run()
# ---------------------------------------------------------------------------


class TestSingleRowMatchesScalar:
    def test_singleton_batch_matches_scalar_run(self):
        """run_batch with N=1 produces the same trajectory as a scalar run()."""
        diag = _build_diagram()
        recorded = {"y": diag["integ"].output_ports[0]}

        # Scalar warm-restart run.
        sim_scalar = FastRestartSimulator(
            diag,
            t_span=(0.0, 1.0),
            options=_opts(),
            recorded_signals=recorded,
        )
        scalar = sim_scalar.run(parameters={"gain.gain": jnp.float64(1.7)})

        # Batch warm-restart run (single row).
        sim_batch = FastRestartSimulator(
            diag,
            t_span=(0.0, 1.0),
            options=_opts(),
            recorded_signals=recorded,
        )
        batch = sim_batch.run_batch({"gain.gain": jnp.asarray([1.7])})

        # Shapes must line up.
        assert batch.outputs["y"].shape[0] == 1
        assert batch.outputs["y"].shape[1] == scalar.outputs["y"].shape[0]

        # Numerical equivalence (small tolerance for vmap reordering).
        max_diff = float(
            jnp.max(jnp.abs(batch.outputs["y"][0] - scalar.outputs["y"]))
        )
        assert max_diff < 1e-6, (
            f"singleton batch diverges from scalar run: max diff = {max_diff:.3e}"
        )


# ---------------------------------------------------------------------------
# Kernel re-use across run() and run_batch()
# ---------------------------------------------------------------------------


class TestKernelReuseAcrossRunAndBatch:
    def test_run_first_then_batch_reuses_kernel(self):
        """Warming with run() before run_batch() must keep _kernel intact."""
        diag = _build_diagram()
        sim = FastRestartSimulator(
            diag,
            t_span=(0.0, 0.5),
            options=_opts(),
            recorded_signals={"y": diag["integ"].output_ports[0]},
        )
        # Cold warm-up.
        sim.run(parameters={"gain.gain": jnp.float64(1.0)})
        kernel_after_warmup = sim._kernel
        assert kernel_after_warmup is not None

        # Batch call should use the same kernel object (no rebuild).
        sim.run_batch({"gain.gain": jnp.asarray([0.5, 1.0, 1.5])})
        assert sim._kernel is kernel_after_warmup


# ---------------------------------------------------------------------------
# Batched initial-state override
# ---------------------------------------------------------------------------


class TestBatchedInitialState:
    def test_initial_states_batch_per_row(self):
        """Each row's IC is honored — first samples track the provided x0's."""
        diag = _build_diagram(k=1.0, x0=0.0)
        sim = FastRestartSimulator(
            diag,
            t_span=(0.0, 0.001),  # very short — sample[0] ≈ initial state
            options=_opts(max_major_steps=4),
            recorded_signals={"y": diag["integ"].output_ports[0]},
        )
        # One run to warm the kernel (and freeze the t-grid shape).
        sim.run()

        x0s = jnp.asarray([0.0, 2.0, -3.0])
        # Use a constant parameter batch matching N=3.
        r = sim.run_batch(
            {"gain.gain": jnp.asarray([1.0, 1.0, 1.0])},
            initial_states_batch=x0s,
        )
        # First sample per row should match the provided IC.
        for i, x0 in enumerate([0.0, 2.0, -3.0]):
            assert abs(float(r.outputs["y"][i][0]) - x0) < 1e-6, (
                f"row {i} initial sample {float(r.outputs['y'][i][0])} "
                f"does not match IC {x0}"
            )


# ---------------------------------------------------------------------------
# Default-off / non-regression smoke
# ---------------------------------------------------------------------------


class TestDefaultOffPath:
    def test_simulate_unchanged_after_run_batch_added(self):
        """run_batch landing does not perturb the default simulate() path."""
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
        assert r.provenance is None

    def test_no_run_batch_no_state_change(self):
        """Constructing a FastRestartSimulator without calling run_batch
        leaves it in the same lazy state as before the followup."""
        diag = _build_diagram()
        sim = FastRestartSimulator(
            diag,
            t_span=(0.0, 0.5),
            options=_opts(),
            recorded_signals={"y": diag["integ"].output_ports[0]},
        )
        assert sim._kernel is None
        assert sim.n_runs == 0
