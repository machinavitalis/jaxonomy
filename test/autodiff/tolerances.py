# SPDX-License-Identifier: MIT
"""
Tolerance policy for gradient-correctness property tests (T-001).

The thresholds below are enforced by ``assert_grad_matches_fd`` in
``_framework.py``. Changing any value requires a test re-run and a note in
``TOLERANCES.md`` with the justification.

Entries are indexed as ``TOL[(solver, dtype)]`` → dict with:
  - ``rtol``: relative tolerance for the AD-vs-FD comparison
  - ``atol``: absolute tolerance for the AD-vs-FD comparison
  - ``fd_eps``: central-difference step size used to compute FD ground truth
  - ``sim_rtol`` / ``sim_atol``: solver tolerances passed to SimulatorOptions

Rationale for per-solver/dtype differences is documented in TOLERANCES.md.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp


@dataclass(frozen=True)
class GradTolerance:
    rtol: float
    atol: float
    fd_eps: float
    sim_rtol: float
    sim_atol: float


# (solver, dtype) → tolerance bundle
TOL: dict[tuple[str, str], GradTolerance] = {
    ("rk4", "float32"):    GradTolerance(rtol=5e-3, atol=1e-4, fd_eps=3e-3, sim_rtol=1e-6, sim_atol=1e-8),
    ("rk4", "float64"):    GradTolerance(rtol=1e-3, atol=1e-5, fd_eps=1e-5, sim_rtol=1e-8, sim_atol=1e-10),
    ("dopri5", "float32"): GradTolerance(rtol=5e-3, atol=1e-4, fd_eps=3e-3, sim_rtol=1e-6, sim_atol=1e-8),
    ("dopri5", "float64"): GradTolerance(rtol=5e-4, atol=1e-5, fd_eps=1e-5, sim_rtol=1e-8, sim_atol=1e-10),
    ("bdf", "float32"):    GradTolerance(rtol=1e-2, atol=5e-4, fd_eps=3e-3, sim_rtol=1e-5, sim_atol=1e-7),
    ("bdf", "float64"):    GradTolerance(rtol=5e-3, atol=1e-4, fd_eps=1e-5, sim_rtol=1e-6, sim_atol=1e-8),
}

# Stateless / non-simulation gradient checks (pure feedthrough, reduce, source
# blocks). No solver involved; tolerance depends only on dtype.
STATELESS_TOL: dict[str, GradTolerance] = {
    "float32": GradTolerance(rtol=1e-3, atol=1e-5, fd_eps=3e-3, sim_rtol=0.0, sim_atol=0.0),
    "float64": GradTolerance(rtol=1e-6, atol=1e-8, fd_eps=1e-5, sim_rtol=0.0, sim_atol=0.0),
}

DTYPES: tuple[str, ...] = ("float32", "float64")

# All solvers exposed through ``SimulatorOptions.ode_solver_method``.
SOLVERS: tuple[str, ...] = ("rk4", "dopri5", "bdf")


def dtype_of(name: str):
    return jnp.float32 if name == "float32" else jnp.float64


def get_tol(solver: str, dtype: str) -> GradTolerance:
    try:
        return TOL[(solver.lower(), dtype)]
    except KeyError as err:
        raise KeyError(
            f"No tolerance policy for (solver={solver}, dtype={dtype}); "
            f"known keys: {sorted(TOL)}"
        ) from err
