#!/usr/bin/env python3
"""Offline publication run for ``dae_projection_pendulum.ipynb``.

Produces ``media/dae_projection_pendulum_publication.npz``.  The notebook
loads the NPZ in publication mode (default) and falls back to a live
re-run on a 5-minute horizon in fast mode if the NPZ is missing.

Runtime on a developer CPU: ~12 s for three 1-hour simulations + a 1 hr
SSP autodiff gradient run.  The notebook's fast mode runs the same
configurations on a 300 s horizon in ~3 s.  This script exists so the
notebook execution stays sub-minute under ``nbconvert``.

Saved arrays:
  config_labels         (3,) U10     -- "Baseline", "Baumgarte", "SSP"
  walls                 (3,)         -- wall-clock seconds per config
  final_resid           (3,)         -- ||f_a||_inf at t=3600 s
  max_residual          (3,)         -- max trace residual across run
  trace_time            (3, N_trace) -- per-major-step times
  trace_residual        (3, N_trace) -- per-major-step ||f_a||_inf
  energy_mean_first     (3,)         -- mean E over first 100 s
  energy_mean_last      (3,)         -- mean E over last 100 s
  geom_max              (3,)         -- max(|x^2+y^2-L^2|) across run
  geom_at_3600          (3,)         -- |x^2+y^2-L^2| at t = 3600 s
  T_END, DT_MAJOR, rtol, atol, baumgarte_beta, ssp_tol -- scalar metadata
  placeholder_flag       () bool     -- False; these are real numbers
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import jax.numpy as jnp

import jaxonomy
from jaxonomy.simulation.dae_drift import constraint_residual_norm


HERE = Path(__file__).resolve().parent
OUT_NPZ = HERE / "dae_projection_pendulum_publication.npz"


# Same index-reduced index-2 pendulum fixture used by the T-113 test
# suite (test/simulation/test_dae_projection.py, test_t_113_phase4_*).
class PlanarPendulum(jaxonomy.LeafSystem):
    """Planar pendulum on a rigid massless link of length L.

    State layout (9 entries):
      x[0] = vx       (differential; M = 1)
      x[1] = x_pos    (differential; M = 1)
      z[0] = vy       (algebraic; M = 0)
      z[1] ... z[6]   (algebraic auxiliaries from index reduction)
      z[2] = y_pos    (algebraic)
    """

    def __init__(self, L: float = 1.0, g0: float = 9.8, name: str | None = None):
        super().__init__(name=name)
        # Index-reduced consistent IC: bob at (sqrt(3)/2, -1/2), at rest.
        x0 = np.array(
            [0.0, 0.8660254037844386, 0.0, -4.9, -0.5,
             -4.243524478543744, -7.35, -7.35, 0.0]
        )
        self.declare_dynamic_parameter("L", L)
        self.declare_dynamic_parameter("g0", g0)
        M = np.concatenate([np.ones(2), np.zeros(7)])
        self.declare_continuous_state(
            default_value=x0, mass_matrix=M, ode=self.ode,
        )
        self.declare_continuous_state_output(name="x")

    def ode(self, time, state, **parameters):
        L, g0 = parameters["L"], parameters["g0"]
        x = state.continuous_state[:2]
        z = state.continuous_state[2:]
        f = jnp.array([z[3], x[0]])
        g = jnp.array([
            -(L ** 2) + x[1] ** 2 + z[2] ** 2,
            2 * z[0] * z[2] + 2 * x[1] * x[0],
            z[0] - z[6],
            2 * z[3] * x[1] + 2 * z[4] * z[2] + 2 * z[0] ** 2 + 2 * x[0] ** 2,
            z[4] - z[5],
            z[5] + g0 - z[1] * z[2],
            -z[1] * x[1] + z[3],
        ])
        return jnp.concatenate([f, g])


# Horizon + solver configuration — kept in sync with the notebook.
T_END = 3600.0      # 1 hour of model time
DT_MAJOR = 2.0      # major-step length: bounds the drift trace cadence
RTOL = 1e-5
ATOL = 1e-7
BAUMGARTE_BETA = 0.05   # small; high-index DAE stalls BDF Newton if too large
SSP_TOL = 1e-9
SSP_ITERS = 4
G0 = 9.8
L = 1.0


def _run(config_name: str, *, baumgarte_beta=None, projection=False):
    """Run one 1-hour pendulum simulation, return everything we need."""
    base = dict(
        math_backend="jax",
        ode_solver_method="bdf",
        rtol=RTOL,
        atol=ATOL,
        record_dae_drift=True,
        max_major_step_length=DT_MAJOR,
        max_major_steps=int(T_END / DT_MAJOR) + 10,
        buffer_length=200_000,
    )
    if baumgarte_beta is not None:
        base.update(baumgarte_alpha=None, baumgarte_beta=baumgarte_beta)
    if projection:
        base.update(
            dae_projection_enabled=True,
            dae_projection_tol=SSP_TOL,
            dae_projection_max_iter=SSP_ITERS,
        )

    model = PlanarPendulum()
    ctx = model.create_context()
    rec = {"state": model.output_ports[0]}
    opts = jaxonomy.SimulatorOptions(**base)

    t_wall = time.time()
    res = jaxonomy.simulate(
        model, ctx, (0.0, T_END), options=opts, recorded_signals=rec,
    )
    wall = time.time() - t_wall

    s = np.asarray(res.outputs["state"])
    vx, xp, vy, yp = s[:, 0], s[:, 1], s[:, 2], s[:, 4]
    E = 0.5 * (vx ** 2 + vy ** 2) + G0 * yp
    geom = xp ** 2 + yp ** 2 - L ** 2

    trace = res.dae_drift_trace
    fin_resid = float(constraint_residual_norm(model, res.context))

    print(
        f"  {config_name:12s} wall={wall:5.2f}s  final_resid={fin_resid:.3e}  "
        f"max_trace={float(np.max(trace['residual'])):.3e}  "
        f"E_range={np.max(E) - np.min(E):.4e}  |geom|max={np.max(np.abs(geom)):.3e}"
    )

    return dict(
        wall=wall,
        final_resid=fin_resid,
        max_residual=float(np.max(trace["residual"])),
        trace_time=np.asarray(trace["time"]),
        trace_residual=np.asarray(trace["residual"]),
        energy_mean_first=float(np.mean(E[:200])),
        energy_mean_last=float(np.mean(E[-200:])),
        geom_max=float(np.max(np.abs(geom))),
        geom_at_3600=float(np.abs(geom[-1])),
    )


def main() -> None:
    print(f"Publication run: 1-hour pendulum with three drift-control configs")
    print(f"  T_END={T_END} s, DT_MAJOR={DT_MAJOR} s, rtol={RTOL}, atol={ATOL}")
    print(f"  Baumgarte beta={BAUMGARTE_BETA}, SSP tol={SSP_TOL}")
    print()

    results = {
        "Baseline": _run("Baseline"),
        "Baumgarte": _run("Baumgarte", baumgarte_beta=BAUMGARTE_BETA),
        "SSP": _run("SSP", projection=True),
    }

    # Trace arrays may differ in length across configs (one per major
    # step); pad to a common length so they fit in a single 2-D ndarray.
    config_labels = np.array(list(results.keys()), dtype="U10")
    trace_len = max(len(r["trace_time"]) for r in results.values())
    trace_time = np.full((len(results), trace_len), np.nan)
    trace_resid = np.full((len(results), trace_len), np.nan)
    for i, r in enumerate(results.values()):
        n = len(r["trace_time"])
        trace_time[i, :n] = r["trace_time"]
        trace_resid[i, :n] = r["trace_residual"]

    walls = np.array([r["wall"] for r in results.values()])
    final_resid = np.array([r["final_resid"] for r in results.values()])
    max_residual = np.array([r["max_residual"] for r in results.values()])
    energy_mean_first = np.array(
        [r["energy_mean_first"] for r in results.values()]
    )
    energy_mean_last = np.array(
        [r["energy_mean_last"] for r in results.values()]
    )
    geom_max = np.array([r["geom_max"] for r in results.values()])
    geom_at_3600 = np.array([r["geom_at_3600"] for r in results.values()])

    np.savez(
        OUT_NPZ,
        config_labels=config_labels,
        walls=walls,
        final_resid=final_resid,
        max_residual=max_residual,
        trace_time=trace_time,
        trace_residual=trace_resid,
        energy_mean_first=energy_mean_first,
        energy_mean_last=energy_mean_last,
        geom_max=geom_max,
        geom_at_3600=geom_at_3600,
        T_END=T_END,
        DT_MAJOR=DT_MAJOR,
        rtol=RTOL,
        atol=ATOL,
        baumgarte_beta=BAUMGARTE_BETA,
        ssp_tol=SSP_TOL,
        placeholder_flag=False,
    )
    print()
    print(f"Wrote {OUT_NPZ}")


if __name__ == "__main__":
    main()
