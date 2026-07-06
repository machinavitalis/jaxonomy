# SPDX-License-Identifier: MIT

"""Fixed-step Runge-Kutta 4 solver exposed through ``SimulatorOptions``.

``Dopri5Solver`` and ``BDFSolver`` both adapt their step size to the user's
``rtol``/``atol``.  Some workloads — real-time MPC loops, fixed-rate
co-simulation, deterministic ``simulate_batch`` traces under ``vmap`` — need a
step count that is statically known.  ``RK4Solver`` wraps the existing
``runge_kutta_step`` in ``rk4.py`` behind the same ``ODESolverImpl`` contract
the adaptive solvers satisfy, so it participates in the hybrid simulator's
event loop, adjoint machinery, and zero-crossing localisation.

Fixed-step specifics:

- The step size is taken from ``ODESolverOptions.max_step_size`` (the
  ``SimulatorOptions.max_minor_step_size`` field).  If unset, a default of
  ``0.01`` is used — a conservative value chosen for a typical 1 Hz–100 Hz
  signal; users should set a step size explicitly for any real workload.
- ``rtol`` / ``atol`` are accepted (from the base class) but ignored — fixed
  step by definition.
- Zero-crossing localisation calls ``eval_interpolant(t)`` to bisect within a
  major step.  RK4 has no native dense output formula, so a linear
  interpolant between ``(t_prev, y_prev)`` and ``(t, y)`` is used.  Accuracy
  is limited by the step size rather than the solver; if a user needs tight
  event times they should reduce ``max_minor_step_size`` accordingly.
- Mass matrices (DAE form) are not supported — use BDF for those.
"""

from __future__ import annotations

import dataclasses
from functools import partial
from typing import TYPE_CHECKING, Callable

import jax
import jax.numpy as jnp
from jax import lax

from jaxonomy.backend.typing import Array
from .ode_solver_impl import ODESolverImpl, ODESolverState
from .rk4 import runge_kutta_step

if TYPE_CHECKING:
    from ...framework.state import StateComponent

__all__ = ["RK4State", "RK4Solver"]


_DEFAULT_STEP = 0.01


@dataclasses.dataclass
class RK4State(ODESolverState):
    """Solver state for fixed-step RK4.

    Keeps the (t_prev, y_prev) pair around for linear interpolation during
    zero-crossing bisection.  No adaptive-step bookkeeping is needed.
    """

    y_prev: Array = None
    unravel: Callable = None

    def __post_init__(self):
        if self.t_prev is None:
            self.t_prev = self.t
        if self.y_prev is None:
            self.y_prev = self.y

    def eval_interpolant(self, t_eval: float) -> Array:
        if self.unravel is None:
            raise ValueError("Unravel function not set: cannot evaluate interpolant.")
        # Linear interpolation between (t_prev, y_prev) and (t, y). Cast alpha
        # to the state dtype so the result keeps y's precision (otherwise the
        # zero-crossing bisection's lax.cond branches see mismatched dtypes).
        span = self.t - self.t_prev
        alpha = jnp.where(span == 0, 0.0, (t_eval - self.t_prev) / span)
        alpha = alpha.astype(self.y.dtype)
        y_interp = self.y_prev + alpha * (self.y - self.y_prev)
        return self.unravel(y_interp)

    @property
    def unraveled_state(self) -> "StateComponent":
        return self.unravel(self.y)

    def ravel(self, x: "StateComponent") -> Array:
        return jnp.concatenate([jnp.ravel(e) for e in jax.tree_util.tree_leaves(x)])

    def update(self, next_y: Array, next_t: float, next_f: Array) -> "RK4State":
        return RK4State(
            y=next_y,
            t=next_t,
            f=next_f,
            dt=self.dt,
            t_prev=self.t,
            y_prev=self.y,
            unravel=self.unravel,
        )

    def tree_flatten(self):
        children = (self.y, self.t, self.f, self.dt, self.t_prev, self.y_prev)
        aux_data = (self.unravel,)
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        (unravel,) = aux_data
        return cls(*children, unravel=unravel)


jax.tree_util.register_pytree_node(
    RK4State,
    lambda state: state.tree_flatten(),
    RK4State.tree_unflatten,
)


class RK4Solver(ODESolverImpl):
    """Fixed-step classic 4th-order Runge-Kutta solver.

    Suitable for real-time / MPC / fixed-rate co-simulation workloads where a
    statically-known step count is preferable to adaptive accuracy control.
    For general-purpose simulation prefer ``dopri5`` (non-stiff) or ``bdf``
    (stiff / DAE).
    """

    supports_mass_matrix: bool = False

    def _initialize_state(self, func, t0, xc0, _mass, unravel, *args, dt=None):
        if dt is None:
            dt = self.max_step_size or _DEFAULT_STEP
        # Match dt's dtype to t0 so downstream arithmetic (t + dt, t_eval - t_prev)
        # stays in one float width; the zero-crossing bisection cond is dtype-strict.
        dt = jnp.asarray(dt, dtype=jnp.asarray(t0).dtype)
        f0 = func(xc0, t0, *args)
        return RK4State(
            y=xc0, t=t0, f=f0, dt=dt, t_prev=t0, y_prev=xc0, unravel=unravel,
        )

    def step(self, func, boundary_time, solver_state):
        """Advance by ``dt`` (clipped to the remaining interval to the boundary).

        RK4 is single-attempt; no retry/reject logic is needed.
        """
        y = solver_state.y
        t = solver_state.t
        dt = solver_state.dt
        # Clip so we do not overshoot the boundary time.
        dt_eff = jnp.minimum(dt, boundary_time - t).astype(dt.dtype)

        def _rhs(y_, t_):
            return func(y_, t_)

        next_y = runge_kutta_step(_rhs, y, t, dt_eff)
        next_t = (t + dt_eff).astype(t.dtype)
        next_f = func(next_y, next_t).astype(solver_state.f.dtype)
        return solver_state.update(next_y, next_t, next_f)
