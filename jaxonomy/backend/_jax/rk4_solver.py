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
import numpy as np
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
        self._substep_groups = self._collect_substep_groups()
        f0 = func(xc0, t0, *args)
        return RK4State(
            y=xc0, t=t0, f=f0, dt=dt, t_prev=t0, y_prev=xc0, unravel=unravel,
        )

    def _collect_substep_groups(self):
        """T-133: static ``(N, mask)`` groups for multirate substepping.

        Reads the system's declared per-block substep factors (aligned
        with the flattened continuous state exactly like ``mass_matrix``
        — leaves-concatenation over ``leaf_systems`` order) and groups
        the flat state entries by distinct factor > 1. Everything here
        is static Python/numpy computed once at solver init, so the
        traced ``step`` closes over concrete masks and loop counts.
        Returns ``[]`` for single-rate systems (the byte-equivalent
        default path).
        """
        self._substep_ny = 0
        self._substep_groups_aug = {}
        if not getattr(self.system, "has_multirate_substeps", False):
            return []
        vec_tree = self.system.continuous_substep_vector
        leaves = [
            np.ravel(leaf)
            for leaf in jax.tree_util.tree_leaves(vec_tree)
        ]
        if not leaves:
            return []
        factors = np.concatenate(leaves)
        self._substep_ny = int(factors.size)
        groups = []
        for n in sorted(set(int(f) for f in factors)):
            if n <= 1:
                continue
            groups.append((n, factors == n))
        return groups

    def _groups_for_size(self, n_flat: int):
        """Substep groups matching the flat vector actually being stepped.

        The forward pass integrates the raw continuous state (size
        ``ny``). The checkpointed reverse pass (``simulator._odeint_adj``)
        reuses this solver on the augmented vector
        ``[y, y_bar, t_bar, param_adjoints]``: the primal block and its
        costate block inherit the forward factors (the adjoint of a fast
        mode is equally fast), while the trailing quadrature entries
        (``t_bar`` and parameter adjoints — pure integrals with no
        self-dynamics) stay single-rate. Any other size (defensive) gets
        no multirate treatment.
        """
        groups = self._substep_groups
        if not groups:
            return []
        ny = self._substep_ny
        if n_flat == ny:
            return groups
        if n_flat >= 2 * ny + 1:
            cached = self._substep_groups_aug.get(n_flat)
            if cached is None:
                cached = [
                    (
                        n,
                        np.concatenate(
                            [m, m, np.zeros(n_flat - 2 * ny, dtype=bool)]
                        ),
                    )
                    for n, m in groups
                ]
                self._substep_groups_aug[n_flat] = cached
            return cached
        return []

    def step(self, func, boundary_time, solver_state):
        """Advance by ``dt`` (clipped to the remaining interval to the boundary).

        RK4 is single-attempt; no retry/reject logic is needed.

        T-133 multirate: when any block declared
        ``declare_continuous_state(substeps=N)``, its state entries are
        advanced with ``N`` RK4 substeps of ``dt/N`` while the rest of
        the diagram takes the single ``dt`` step. Coupling uses the
        classical frozen (zero-order-hold) scheme — during the slow pass
        the fast entries are held at their start-of-step values, and
        during each fast group's substep loop all other entries are held
        at start-of-step values. This bounds the coupling error at
        O(dt) across the interface (the same contract as a hand-rolled
        substep loop inside the block) while giving the fast dynamics
        the stability of the finer step. The substep loop has a static
        trip count, so the whole step stays jit-, vmap-, and
        reverse-AD-compatible.
        """
        y = solver_state.y
        t = solver_state.t
        dt = solver_state.dt
        # Clip so we do not overshoot the boundary time.
        dt_eff = jnp.minimum(dt, boundary_time - t).astype(dt.dtype)

        def _rhs(y_, t_):
            return func(y_, t_)

        groups = (
            self._groups_for_size(int(y.shape[0]))
            if getattr(self, "_substep_groups", [])
            else []
        )
        if not groups:
            next_y = runge_kutta_step(_rhs, y, t, dt_eff)
        else:
            fast_any = jnp.asarray(
                np.logical_or.reduce([mask for _, mask in groups])
            )

            # Slow pass: advance everything one dt step with the fast
            # entries frozen at their start-of-step values, so an
            # unstable-at-dt fast mode cannot pollute the slow states'
            # stage derivatives. The fast entries of this result are
            # overwritten below.
            def _slow_rhs(y_, t_):
                return func(jnp.where(fast_any, y, y_), t_)

            next_y = runge_kutta_step(_slow_rhs, y, t, dt_eff)

            for n_sub, mask_np in groups:
                mask = jnp.asarray(mask_np)
                h_sub = (dt_eff / n_sub).astype(dt.dtype)

                def _fast_rhs(y_, t_, mask=mask):
                    # All non-group entries held at start-of-step values
                    # (ZOH coupling), including other fast groups.
                    return func(jnp.where(mask, y_, y), t_)

                def _substep(i, y_f, mask=mask, h_sub=h_sub, rhs=_fast_rhs):
                    t_i = (t + i * h_sub).astype(t.dtype)
                    y_new = runge_kutta_step(rhs, y_f, t_i, h_sub)
                    return jnp.where(mask, y_new, y_f)

                y_fast = lax.fori_loop(0, n_sub, _substep, y)
                next_y = jnp.where(mask, y_fast, next_y)

        next_t = (t + dt_eff).astype(t.dtype)
        next_f = func(next_y, next_t).astype(solver_state.f.dtype)
        return solver_state.update(next_y, next_t, next_f)
