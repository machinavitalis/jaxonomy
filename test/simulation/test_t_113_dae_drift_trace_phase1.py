# SPDX-License-Identifier: MIT
"""Tests for T-113 phase 1 — per-major-step DAE constraint drift trace.

T-113 (originally tracked as T-MW-110) extends the existing T-003a
projection / T-003b drift-threshold-warning machinery with a diagnostic
*trace*: when ``SimulatorOptions.record_dae_drift=True``, the
simulator captures ``||f_a||_∞`` at each major step and surfaces the
``(time, residual)`` arrays via ``SimulationResults.dae_drift_trace``.

Phase 1 ships only the trace itself.

These tests verify:

* Default-off byte-equivalence smoke — ``record_dae_drift`` defaults
  to ``False`` and ``dae_drift_trace`` defaults to ``None`` on
  ``SimulationResults``; the option's absence does not perturb the
  recorded outputs of an existing DAE simulation (so the default-off
  hot path is byte-equivalent).
* Pure-ODE no-op — enabling the option on a non-DAE system yields
  ``dae_drift_trace is None`` (no mass matrix → no monitor → no trace).
* Trace populated on a planar pendulum — running with projection +
  trace produces a populated dict with monotonically-bounded residuals.
* Determinism — two identical runs produce identical traces.
"""

from __future__ import annotations

import numpy as np
import jax.numpy as jnp

import jaxonomy
from jaxonomy.testing.markers import skip_if_not_jax, requires_jax

skip_if_not_jax()


# Index-2 holonomic-constraint test bed (T-032).  Verbatim copy from
# test/simulation/test_dae_projection.py — same fixture so the
# T-003a/T-003b/T-113 stack is exercised against a single canonical
# nonlinear DAE.
class PlanarPendulum(jaxonomy.LeafSystem):
    def __init__(self, L=1.0, g0=9.8, name=None):
        super().__init__(name=name)
        x0 = np.array(
            [0.0, 0.8660254037844386, 0.0, -4.9, -0.5,
             -4.243524478543744, -7.35, -7.35, 0.0]
        )
        self.declare_dynamic_parameter("L", L)
        self.declare_dynamic_parameter("g0", g0)
        self.nx, self.nz = 2, 7
        M = np.concatenate([np.ones(self.nx), np.zeros(self.nz)])
        self.declare_continuous_state(default_value=x0, mass_matrix=M, ode=self.ode)
        self.declare_continuous_state_output(name="x")

    def ode(self, time, state, **parameters):
        L, g0 = parameters["L"], parameters["g0"]
        x = state.continuous_state[:2]
        z = state.continuous_state[2:]
        f = jnp.array([z[3], x[0]])
        g = jnp.array([
            -(L**2) + x[1] ** 2 + z[2] ** 2,
            2 * z[0] * z[2] + 2 * x[1] * x[0],
            z[0] - z[6],
            2 * z[3] * x[1] + 2 * z[4] * z[2] + 2 * z[0] ** 2 + 2 * x[0] ** 2,
            z[4] - z[5],
            z[5] + g0 - z[1] * z[2],
            -z[1] * x[1] + z[3],
        ])
        return jnp.concatenate([f, g])


def test_default_off_no_drift_trace():
    """``SimulatorOptions()`` defaults to ``record_dae_drift=False`` and
    ``SimulationResults.dae_drift_trace`` is ``None`` for both pure-ODE
    and DAE runs."""
    assert jaxonomy.SimulatorOptions().record_dae_drift is False

    model = PlanarPendulum()
    ctx = model.create_context()
    res = jaxonomy.simulate(
        model, ctx, (0.0, 0.1),
        options=jaxonomy.SimulatorOptions(
            math_backend="jax", ode_solver_method="bdf",
        ),
    )
    assert res.dae_drift_trace is None


@requires_jax()
def test_default_off_byte_equivalent_outputs():
    """Adding an unset ``record_dae_drift`` does not perturb the
    recorded continuous-state output trajectory.  Baseline run vs.
    run with the option explicitly defaulted both produce the same
    final continuous state."""
    model = PlanarPendulum()
    ctx = model.create_context()

    rec = {"x": model.output_ports[0]}
    base_opts = jaxonomy.SimulatorOptions(
        math_backend="jax", ode_solver_method="bdf",
        rtol=1e-6, atol=1e-8,
    )
    res_baseline = jaxonomy.simulate(
        model, ctx, (0.0, 0.5), recorded_signals=rec, options=base_opts,
    )
    # Same options but the new field explicitly defaulted to False.
    same_opts = jaxonomy.SimulatorOptions(
        math_backend="jax", ode_solver_method="bdf",
        rtol=1e-6, atol=1e-8,
        record_dae_drift=False,
    )
    res_default = jaxonomy.simulate(
        model, ctx, (0.0, 0.5), recorded_signals=rec, options=same_opts,
    )
    np.testing.assert_array_equal(
        np.asarray(res_baseline.outputs["x"]),
        np.asarray(res_default.outputs["x"]),
    )
    assert res_baseline.dae_drift_trace is None
    assert res_default.dae_drift_trace is None


@requires_jax()
def test_pure_ode_record_is_noop():
    """Setting ``record_dae_drift=True`` on a pure-ODE system yields no
    trace (no mass matrix → no algebraic constraints to monitor)."""

    class Decay(jaxonomy.LeafSystem):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.declare_continuous_state(
                default_value=jnp.array(1.0), ode=self._ode,
            )
            self.declare_continuous_state_output(name="x")

        def _ode(self, time, state, **params):
            return -state.continuous_state

    sys = Decay()
    ctx = sys.create_context()
    res = jaxonomy.simulate(
        sys, ctx, (0.0, 1.0),
        recorded_signals={"x": sys.output_ports[0]},
        options=jaxonomy.SimulatorOptions(
            math_backend="jax", ode_solver_method="dopri5",
            record_dae_drift=True,
        ),
    )
    assert res.dae_drift_trace is None
    assert float(res.outputs["x"][-1]) < 0.5


@requires_jax()
def test_drift_trace_populated_on_pendulum():
    """``record_dae_drift=True`` on the pendulum populates a
    ``{"time": ndarray, "residual": ndarray}`` dict with at least one
    sample, monotonic times, and finite non-negative residuals.  With
    projection enabled and a tight tol, every sample stays well below
    a generous 1e-3 ceiling over a 5 s sim."""
    model = PlanarPendulum()
    ctx = model.create_context()
    opts = jaxonomy.SimulatorOptions(
        math_backend="jax", ode_solver_method="bdf",
        rtol=1e-8, atol=1e-10,
        dae_projection_enabled=True,
        dae_projection_tol=1e-12,
        dae_projection_max_iter=4,
        record_dae_drift=True,
    )
    res = jaxonomy.simulate(model, ctx, (0.0, 5.0), options=opts)

    trace = res.dae_drift_trace
    assert trace is not None, "projection-enabled run should produce a trace"
    assert isinstance(trace, dict)
    assert set(trace) == {"time", "residual"}

    times = np.asarray(trace["time"])
    residuals = np.asarray(trace["residual"])
    assert times.shape == residuals.shape
    assert times.shape[0] >= 1, "expected at least one major-step sample"

    # Times monotonic non-decreasing (each sample comes from one major
    # step in chronological order).  Allow equality for back-to-back
    # zero-length steps (e.g. event resets).
    assert np.all(np.diff(times) >= -1e-12)

    # Residuals finite and non-negative (||f_a||_∞ is by construction).
    assert np.all(np.isfinite(residuals))
    assert np.all(residuals >= 0.0)

    # Generous bound — projection holds residual far below 1e-3 even at
    # the loose end of the sim.  Tight projection tol (1e-12) is the
    # actual ceiling but we keep the assertion loose so it stays robust
    # against minor solver/backend noise.
    assert np.max(residuals) < 1e-3, (
        f"projection failed to hold residual: max={np.max(residuals):.3e}"
    )


@requires_jax()
def test_drift_trace_deterministic():
    """Two identical runs produce byte-equal drift traces."""
    model = PlanarPendulum()
    ctx = model.create_context()
    opts = jaxonomy.SimulatorOptions(
        math_backend="jax", ode_solver_method="bdf",
        rtol=1e-6, atol=1e-8,
        dae_projection_enabled=True,
        dae_projection_tol=1e-10,
        dae_projection_max_iter=4,
        record_dae_drift=True,
    )
    res_a = jaxonomy.simulate(model, ctx, (0.0, 1.0), options=opts)
    res_b = jaxonomy.simulate(model, ctx, (0.0, 1.0), options=opts)
    assert res_a.dae_drift_trace is not None
    assert res_b.dae_drift_trace is not None
    np.testing.assert_array_equal(
        res_a.dae_drift_trace["time"],
        res_b.dae_drift_trace["time"],
    )
    np.testing.assert_array_equal(
        res_a.dae_drift_trace["residual"],
        res_b.dae_drift_trace["residual"],
    )


@requires_jax()
def test_trace_without_projection_still_populates():
    """``record_dae_drift=True`` without projection still captures the
    trace — the trace is a diagnostic surface independent of the
    correction step.  Useful for users who want to *measure* drift
    before deciding whether to enable projection."""
    model = PlanarPendulum()
    ctx = model.create_context()
    opts = jaxonomy.SimulatorOptions(
        math_backend="jax", ode_solver_method="bdf",
        rtol=1e-6, atol=1e-8,
        record_dae_drift=True,
    )
    res = jaxonomy.simulate(model, ctx, (0.0, 0.2), options=opts)
    trace = res.dae_drift_trace
    assert trace is not None
    assert trace["time"].shape[0] >= 1
    assert np.all(np.isfinite(trace["residual"]))
