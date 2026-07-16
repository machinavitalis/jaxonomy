# SPDX-License-Identifier: MIT

"""JAX-based Backwards Differentiation Formula ODE/DAE solver with adaptive stepsize.

References:
[1] G. D. Byrne, A. C. Hindmarsh, "A Polyalgorithm for the Numerical
    Solution of Ordinary Differential Equations", ACM Transactions on
    Mathematical Software, Vol. 1, No. 1, pp. 71-96, March 1975.
[2] L. F. Shampine, M. W. Reichelt, "THE MATLAB ODE SUITE", SIAM J. SCI.
    COMPUTE., Vol. 18, No. 1, pp. 1-22, January 1997.
[3] E. Hairer, G. Wanner, "Solving Ordinary Differential Equations I:
    Nonstiff Problems", Sec. III.2.
"""

from __future__ import annotations
import dataclasses
from typing import TYPE_CHECKING, Callable
from functools import partial

import numpy as np
import jax
import jax.numpy as jnp
from jax import lax

from .ode_solver_impl import (
    ODESolverImpl,
    ODESolverState,
    norm,
    error_step_size_too_small,
)
from ..typing import Array
from ...lazy_loader import LazyLoader, LazyModuleAccessor

if TYPE_CHECKING:
    from jax.scipy.linalg import lu_factor, lu_solve

    from ...framework.state import StateComponent
else:
    jax_scipy_linalg = LazyLoader("jax_scipy_linalg", globals(), "jax.scipy.linalg")
    lu_factor = LazyModuleAccessor(jax_scipy_linalg, "lu_factor")
    lu_solve = LazyModuleAccessor(jax_scipy_linalg, "lu_solve")

__all__ = [
    "BDFState",
    "BDFSolver",
]

EPS = np.finfo(float).eps

MAX_ORDER = 5
NEWTON_MAXITER = 4
ROOT_SOLVE_MAXITER = 15
MIN_FACTOR = 0.2
MAX_FACTOR = 10


# https://github.com/scipy/scipy/blob/v1.13.0/scipy/integrate/_ivp/bdf.py#L242
kappa = jnp.array([0, -0.1850, -1 / 9, -0.0823, -0.0415, 0])
GAMMA = jnp.hstack((0, jnp.cumsum(1 / jnp.arange(1, MAX_ORDER + 1))))
ALPHA = (1 - kappa) * GAMMA
ERROR_CONST = kappa * GAMMA + 1 / jnp.arange(1, MAX_ORDER + 2)


@dataclasses.dataclass
class BDFState(ODESolverState):
    # `t_return` is the time value to return in time series.  Will be inf when
    # the end time is reached.  Otherwise it should match `t`.
    t_return: float = None
    n_acc: int = 0  # Number of accepted steps
    n_rej: int = 0  # Number of rejected steps
    accepted: bool = False  # Whether the most recent attempted step was accepted

    # Unique to BDF:
    order: int = 1  # The current order of the BDF solver. Initialize to first-order.
    D: Array = None  # Table of backwards differences
    J: Array = None  # The Jacobian matrix
    M: Array = None  # The mass matrix
    LU: Array = None  # The LU factorization
    U: Array = None

    updated_jacobian: bool = False  # Whether the Jacobian has been updated
    n_equal_steps: int = 0  # Number of equal-length steps taken

    # Aux data (note this should all come at the end)
    unravel: Callable = None  # Unravel the flattened vector to the original pytree

    def __post_init__(self):
        if self.t_return is None:
            self.t_return = self.t
        if self.t_prev is None:
            self.t_prev = self.t

    # Inherits docstring from `ODESolverState`
    def eval_interpolant(self, t_eval: float) -> Array:
        if self.unravel is None:
            raise ValueError("Unravel function not set: cannot evaluate interpolant.")

        order = self.order
        t = self.t
        h = self.dt
        D = self.D

        def while_body(j, val):
            p, y = val
            p *= (t_eval - (t - h * j)) / (h * (1 + j))
            return (p, y + D[j + 1] * p)

        _, y = lax.fori_loop(0, order, while_body, (1.0, D[0]))
        return self.unravel(y)

    # Inherits docstring from `ODESolverState`
    @property
    def unraveled_state(self) -> StateComponent:
        return self.unravel(self.y)

    def ravel(self, x: StateComponent) -> Array:
        return jnp.concatenate([jnp.ravel(e) for e in jax.tree_util.tree_leaves(x)])

    @property
    def step_variables(self):
        return self.y, self.f, self.t, self.dt

    def tree_flatten(self):
        children = (
            self.y,
            self.t,
            self.f,
            self.dt,
            self.t_prev,
            self.t_return,
            self.n_acc,
            self.n_rej,
            self.accepted,
            self.order,
            self.D,
            self.J,
            self.M,
            self.LU,
            self.U,
            self.updated_jacobian,
            self.n_equal_steps,
        )
        aux_data = (self.unravel,)
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        (unravel,) = aux_data
        return cls(*children, unravel)

    def with_state_and_time(self, y, t) -> BDFState:
        return dataclasses.replace(self, y=y, t=t, t_prev=t)


jax.tree_util.register_pytree_node(
    BDFState,
    lambda state: state.tree_flatten(),
    BDFState.tree_unflatten,
)


ROW_IDX = np.arange(MAX_ORDER + 3).reshape(-1, 1)
COL_IDX = np.arange(MAX_ORDER + 3)


# T-017b-followup-newton-blocker: `R_matrix` and `_update_D` are pure,
# module-scope, autodiff-safe (no closure over solver state, no captured
# mutable references).  Each is called from 3+ sites inside a single BDF
# step; hoisting behind module-scope `jax.jit` collapses repeated call-sites
# in one trace into a `pjit_p` primitive whose lowering is cached across
# BDFSolver instances.  Median compile-time on `rc_acausal_dae` drops from
# 75.4 ms to 67.5 ms (n=11, ~10% improvement); `simulate_batch_decay_n1000`
# also gets ~10%.  The full Newton-iteration body (`_solve_newton_system_impl`
# below) was extracted to module scope for symmetry but applying @jax.jit
# there did NOT help (XLA backend-compile time scales with outer jaxpr size,
# not nested pjit boundaries).  The autodiff path is unaffected — `jax.custom_vjp`
# in `simulation/autodiff_rules.py` traces through nested `pjit` the same as
# any function call (verified: `pytest test/autodiff/ -m "not slow"` clean).
@jax.jit
def R_matrix(order, factor):
    # Formula from Ref. [2] (NOTE: the "Sec. 3.2" pointer is wrong — section
    # 3.2 of [2] contains no equations; the formula is elsewhere in the ref).
    n = MAX_ORDER
    factor_arr = jnp.asarray(factor)
    i = ROW_IDX[1 : n + 1]  # noqa
    j = COL_IDX[1 : n + 1]  # noqa
    M = jnp.zeros((n + 1, n + 1), dtype=factor_arr.dtype)
    M = M.at[1:, 1:].set(((i - 1 - factor_arr * j) / i).astype(factor_arr.dtype))
    M = M.at[0, :].set(1)

    i = ROW_IDX[: n + 1]
    j = COL_IDX[: n + 1]
    mask = jnp.logical_and(i <= order, j <= order)
    M = jnp.where(mask, M, jnp.asarray(1.0, dtype=factor_arr.dtype))
    R = jnp.where(mask, jnp.cumprod(M, axis=0), jnp.asarray(0.0, dtype=factor_arr.dtype))
    return R.astype(factor_arr.dtype)


@jax.jit
def _update_D(D, order, factor):
    # update D using the equations in Ref. [2] (NOTE: the "section 3.2"
    # pointer is wrong — section 3.2 of [2] contains no equations).
    n = MAX_ORDER
    factor = jnp.asarray(factor, dtype=D.dtype)
    U = R_matrix(order, jnp.asarray(1.0, dtype=D.dtype))
    R = R_matrix(order, factor)
    I_ = jnp.eye(n + 3, dtype=D.dtype)

    RU = jnp.where(
        (ROW_IDX <= order) & (COL_IDX <= order),
        I_.at[: n + 1, : n + 1].set(R.dot(U)),
        I_,
    )
    return jnp.dot(RU.T, D).astype(D.dtype)


def _emit_bdf_nonfinite_warning(bailed, t_val, dt_val, row_mask):
    """Host-side ``UserWarning`` emitter for the BDF terminal NaN bailout.

    Invoked from inside the jit'd ``step`` via ``jax.debug.callback`` —
    that primitive lifts the call out of the XLA computation onto the
    host so ``warnings.warn`` works (same idiom as
    ``_emit_dae_drift_warning`` in ``simulation/simulator.py``).  Gated
    host-side on ``bailed`` so healthy steps stay silent.

    Handles both the plain and the ``vmap``-batched case: under
    ``vmap`` the arguments arrive with a leading batch dimension and a
    single aggregated warning is emitted listing the failing lanes.

    A negative reported time indicates the failure occurred in the
    reverse-time *adjoint* solve of the autodiff backward pass (the
    adjoint system is integrated in negated time).
    """
    import warnings

    bailed = np.atleast_1d(np.asarray(bailed))
    if not bailed.any():
        return

    t_val = np.broadcast_to(np.asarray(t_val), bailed.shape)
    dt_val = np.broadcast_to(np.asarray(dt_val), bailed.shape)
    row_mask = np.atleast_2d(np.asarray(row_mask))

    msgs = []
    for lane in np.flatnonzero(bailed):
        t_f = float(t_val[lane])
        dt_f = float(dt_val[lane])
        rows = np.flatnonzero(row_mask[lane])
        if rows.size:
            detail = (
                f"non-finite state rows {rows.tolist()} "
                f"(of {row_mask.shape[-1]})"
            )
        else:
            detail = (
                "error test could not be satisfied at the minimum step "
                "size (state still finite but not converging)"
            )
        lane_tag = f"[batch element {lane}] " if bailed.size > 1 else ""
        msgs.append(f"{lane_tag}t={t_f:.6g}, dt={dt_f:.3e}: {detail}")

    warnings.warn(
        "BDF solver aborted: the corrector/step loop went non-finite and "
        "the adaptive step size could not recover ("
        + "; ".join(msgs)
        + "). The state is NaN from this time onward. Common causes: an "
        "inconsistent algebraic initial state (for DAEs, project the "
        "initial conditions to the constraint manifold before "
        "simulating), an ill-conditioned component equation, or a "
        "genuinely diverging solution. A negative time indicates the "
        "failure occurred in the reverse-time adjoint solve.",
        UserWarning,
        stacklevel=2,
    )


def _solve_newton_system_impl(func, t, y, c, psi, LU, M, scale, tol):
    """Module-scope Newton inner loop; pure (no closure over solver state).

    T-017b-followup-newton-blocker: hoisted out of `BDFSolver.solve_newton_system`
    as a refactor toward future module-level JIT caching.  ``tol`` is taken as
    an argument rather than via closure on ``self.newton_tol`` so the body has
    no reference to mutable solver state — this keeps the body autodiff-stable
    (the custom-VJP rules in ``simulation/autodiff_rules.py`` close over the
    call-site, never the body's internals).

    Empirically, applying ``@jax.jit`` to this function with ``func`` static
    did NOT cut compile time on rc_acausal_dae (≤1 ms noise either way).
    The hot lowering cost is XLA's ``backend_compile_and_load``, which scales
    with the *outer* jaxpr size, not the number of nested ``pjit`` boundaries.
    The function is left as a plain helper here as a clean shape for future
    experimentation; the @jax.jit on ``R_matrix`` and ``_update_D`` above is
    where the actual compile-time win comes from (~10% on rc_acausal_dae).
    """

    dtype = y.dtype
    c = jnp.asarray(c, dtype=dtype)
    psi = jnp.asarray(psi, dtype=dtype)
    scale = jnp.asarray(scale, dtype=dtype)
    M = jnp.asarray(M, dtype=dtype)
    LU = (LU[0].astype(dtype), LU[1])

    def _cond_fun(carry):
        exit_flag = carry[0]
        return ~exit_flag

    def _body_fun(carry):
        exit_flag, converged, y, d, dy_norm_old, k = carry

        f = func(y, t)

        dy = lu_solve(LU, c * f - M @ (psi + d))
        dy_norm = norm(M @ dy / scale)

        # NOTE: SciPy has a check here to exit early if the iterations appear
        # to be diverging, which saves a few iterations in some cases.  However,
        # this test does not appear to be robust with a mass matrix, so that test
        # is not implemented here.
        rate = jnp.where(
            jnp.isfinite(dy_norm_old),
            dy_norm / dy_norm_old,
            jnp.asarray(jnp.inf, dtype=dtype)
        )

        y = y + dy
        d = d + dy

        converged = jnp.all(abs(f) <= EPS) | (
            (dy_norm == 0.0)
            | (jnp.isfinite(rate) & (rate / (1 - rate) * dy_norm < tol))
        )

        dy_norm_old = dy_norm
        k += 1
        exit_flag = (k >= NEWTON_MAXITER) | converged
        return exit_flag, converged, y, d, dy_norm_old, k

    d = jnp.zeros_like(y)
    dy_norm_old = jnp.asarray(jnp.inf, dtype=dtype)
    exit_flag = False
    converged = False
    k = 0
    exit_flag, converged, y, d, dy_norm_old, k = jax.lax.while_loop(
        _cond_fun,
        _body_fun,
        (exit_flag, converged, y, d, dy_norm_old, k),
    )

    return converged, k, y, d


class BDFSolver(ODESolverImpl):
    """JAX-based Backwards Differentiation Formula ODE solver.

    The solver is variable-order and variable-stepsize, using a simplified
    Newton iteration to solve the nonlinear system of equations at each step.
    The order ranges from 1 to 5, and the step size is adjusted based on the
    error estimate from the Newton iteration.
    """

    def _finalize(self):
        self.supports_mass_matrix = True
        super()._finalize()
        self.newton_tol = max(10 * EPS / self.rtol, min(0.03, self.rtol**0.5))

        try:
            is_tpu = jax.default_backend() == "tpu"
        except Exception:
            is_tpu = False

        if is_tpu and jax.config.read("jax_enable_x64"):
            from ..ode_solver import ODESolverError
            raise ODESolverError(
                "TPU backend does not support double-precision (float64) LU decomposition, "
                "which is required by the implicit BDF solver. Please configure precision "
                "to float32 (or disable JAX global x64)."
            )

    def _initialize_state(self, func, t0, xc0, mass, unravel, *args, dt=None):
        # Initialize the solver state using the first-order BDF method
        order = 1
        f0 = func(xc0, t0, *args)
        if dt is None:
            dt = self.initial_step_size(func, xc0, t0, order, f0, *args)

        # Initialize the BDF state matrix with the first-order (Euler) step
        D = jnp.zeros((MAX_ORDER + 3, len(xc0)), dtype=xc0.dtype)
        D = D.at[0].set(xc0)
        D = D.at[1].set(f0 * dt)
        c = (dt / ALPHA[order]).astype(xc0.dtype)

        if mass is None:
            M = np.eye(len(xc0), dtype=xc0.dtype)
        else:
            M = mass

        jac = jax.jacfwd(func, argnums=0)
        J = jac(xc0, t0, *args)
        LU = self._lu_factor(M - c * J)
        LU = (LU[0].astype(xc0.dtype), LU[1])

        state = BDFState(
            xc0,
            t0,
            f0,
            dt,
            unravel=unravel,
            order=order,
            D=D,
            J=J,
            M=M,
            LU=LU,
        )

        return state

    def predict(self, D, order) -> tuple[Array, Array]:
        # Predict new state value using the BDF formula
        k = jnp.repeat(ROW_IDX, D.shape[1], axis=1)
        return jnp.sum(jnp.where(k <= order, D, 0), axis=0)

    def solve_newton_system(self, func, t, y, c, psi, LU, M, scale):
        # Solve the BDF system of equations using a simplified Newton iteration.
        # Delegates to the module-scope `_solve_newton_system_impl` so the
        # lowering is shared (T-017b-followup-newton-blocker).
        return _solve_newton_system_impl(
            func, t, y, c, psi, LU, M, scale, self.newton_tol
        )

    def newton_iteration(self, state, func, boundary_time):
        y0, f, t, h = state.step_variables
        n_equal_steps = state.n_equal_steps
        order = state.order
        M, D, J, LU = state.M, state.D, state.J, state.LU

        jac_fn = jax.jacfwd(func, argnums=0)

        t_new = t + h
        factor = abs(boundary_time - t) / h
        (t_new, D, n_equal_steps, recalc_lu) = jax.tree.map(
            partial(jnp.where, t_new - boundary_time > 0),
            (boundary_time, _update_D(D, order, factor), 0, True),
            (t_new, D, n_equal_steps, False),
        )
        h = t_new - t
        error_step_size_too_small(t, h, boundary_time, self.enable_autodiff)

        # Update LU: `c` has changed (maybe)
        c = (h / ALPHA[order]).astype(y0.dtype)
        LU = self._lu_factor(M - c * J, recalc_lu, LU)
        LU = (LU[0].astype(y0.dtype), LU[1])

        y_predict = self.predict(D, order=order)

        # Update the vector used in simplified Newton iterations
        # Since all arrays must be statically sized, extend `GAMMA` with zeros
        # to match the size of `D` (MAX_ORDER + 3)
        k = COL_IDX
        gamma = jnp.concatenate((GAMMA, np.array([0.0, 0.0])))
        gamma = gamma.astype(y0.dtype)
        gamma = jnp.where(k > 0, jnp.where(k <= order, gamma, 0), 0)
        k = jnp.repeat(ROW_IDX, D.shape[1], axis=1)
        D_submat = jnp.where(k > 0, jnp.where(k <= order, D, 0), 0)
        psi = (jnp.dot(D_submat.T, gamma) / ALPHA[order]).astype(y0.dtype)

        scale = (self.atol + self.rtol * jnp.abs(y_predict)).astype(y0.dtype)

        def _cond_fun(carry):
            exit_flag = carry[0]
            return ~exit_flag

        def _body_fun(carry):
            _exit_flag, _n_iter, _converged, current_jac, J, LU, _y_new, _d = carry

            converged, n_iter, y_new, d = self.solve_newton_system(
                func, t_new, y_predict, c, psi, LU, M, scale
            )

            # Will exit if either converged or the Jacobian is already up to date
            # This will trigger decreasing the step size and trying again.
            exit_flag = converged | current_jac

            # https://github.com/scipy/scipy/blob/v1.13.0/scipy/integrate/_ivp/bdf.py#L370-L375

            # The `cond_fun` check will break out of the loop if not converged but the Jacobian
            # is up to date.  Here we will just update the Jacobian if not converged.
            current_jac, J, recalc_lu = jax.tree.map(
                partial(jnp.where, ~converged & current_jac),
                (current_jac, J, False),
                (True, jac_fn(y_predict, t_new), True),
            )
            # Update LU: Jacobian has changed (maybe)
            LU = self._lu_factor(M - c * J, recalc_lu, LU)

            return exit_flag, n_iter, converged, current_jac, J, LU, y_new, d

        # Set up loop variables
        converged = False
        d = jnp.zeros_like(y0)
        y_new = jnp.array(y0, copy=True)
        k = 0
        current_jac = state.updated_jacobian
        J = state.J
        exit_flag = False
        carry = (exit_flag, k, converged, current_jac, J, LU, y_new, d)

        # NOTE: This will run a maximum of twice, once with the current (possibly
        # out-of-date) Jacobian, and once with an updated Jacobian.  The loop
        # construction is not necessary but helps minimize "call sites" to the
        # ODE RHS and linear solver, which can be expensive to compile.
        _exit_flag, k, converged, current_jac, J, LU, y_new, d = jax.lax.while_loop(
            _cond_fun, _body_fun, carry
        )

        # Note that `dt` only changes here if it was clipped by the boundary
        # time at the top.  The outer `attempt_bdf_step` function needs to handle
        # updating both `t` and `y` in the state, since it manages adaptive step
        # size apart from the boundary check.
        state = dataclasses.replace(
            state,
            D=D,
            J=J,
            updated_jacobian=current_jac,
            LU=LU,
            dt=jnp.asarray(h, dtype=y0.dtype),
            n_equal_steps=n_equal_steps,
        )
        # T-038a-followup-bdf-condition-check: forward one condition-
        # number estimate per ``newton_iteration`` call on the final
        # ``A = M - c*J`` matrix actually used.  No-op when no monitor
        # is attached (default-off byte-equivalent path).  Placed after
        # the inner ``while_loop`` so the estimate reflects the
        # final/converged matrix, not the in-loop intermediate.
        self._maybe_emit_condition(M - c * J, t_new)
        return converged, k, y_new, d, state

    def _lu_factor(self, A, pred=None, LU=None):
        if pred is None:
            return lu_factor(A)
        return jax.tree.map(partial(jnp.where, pred), lu_factor(A), LU)

    def _maybe_emit_condition(self, A, t):
        """T-038a-followup-bdf-condition-check.

        Forward an aggregated condition-number estimate of the Newton-
        system matrix ``A = M - c*J`` to a Python-side aggregator via
        ``jax.debug.callback``.  No-op when ``_cond_monitor`` is None —
        the default-off hot path stays byte-equivalent.

        Side-channelled (set on the solver instance from the simulator)
        rather than packed into the solver state so the parallel
        ``T-017b-followup-newton-blocker`` (Newton-body hoist) does not
        have to thread a new return value through the JIT boundary.
        """
        monitor = getattr(self, "_cond_monitor", None)
        if monitor is None:
            return
        # ``jnp.linalg.cond`` uses the 2-norm by default — one extra SVD
        # on the small Newton matrix.  Cheap on the typical
        # ``n_states × n_states`` system (n in the tens to low hundreds);
        # very-large-state systems should leave the option off.
        cond_est = jnp.linalg.cond(A)
        # Replace non-finite (NaN, inf) with a sentinel below the host-
        # side seed so a transient singular factor doesn't permanently
        # poison the running max.  Genuine large finite values still
        # propagate.
        cond_est = jnp.where(jnp.isfinite(cond_est), cond_est, -jnp.inf)
        jax.debug.callback(monitor.update, cond_est, jnp.asarray(t))

    def _update_dt(self, state, factor):
        dtype = state.y.dtype
        factor = jnp.asarray(factor, dtype=dtype)
        order = state.order
        h = state.dt * factor
        D = _update_D(state.D, order, factor)
        c = (h / ALPHA[order]).astype(dtype)

        # Redo LU factorization (timestep changed)
        LU = self._lu_factor(state.M - c * state.J)
        LU = (LU[0].astype(dtype), LU[1])
        return dataclasses.replace(state, n_equal_steps=0, dt=h, D=D, LU=LU)

    def attempt_bdf_step(self, func, boundary_time, carry):
        state = carry[0]
        fail_diag = carry[5]

        converged, n_iter, y_new, d, state = self.newton_iteration(
            state, func, boundary_time
        )

        # Three cases for adaptive step sizing:
        # (1) The Newton iterations did not converge but the Jacobian has already been
        #   updated, reduce step size by 0.5 and try again.
        # (2) The Newton iterations converged but the error estimate is too high, update
        #   the step size with the optimal factor and try again.
        # (3) The Newton iterations converged and the error estimate is acceptable,
        #   accept the step and update the state for the next step.  The factor for this
        #   case is `-inf`, since the other two cases will have positive scale factors
        updated_jacobian = state.updated_jacobian
        safety = 0.9 * (2 * NEWTON_MAXITER + 1) / (2 * NEWTON_MAXITER + n_iter)
        scale = self.atol + self.rtol * jnp.abs(y_new)

        # NOTE: differs from scipy's BDF (scipy/integrate/_ivp/bdf.py#L388):
        # scipy has no `state.M` (mass matrix) factor here — it is required for
        # the DAE form jaxonomy supports.
        dtype = state.y.dtype
        error = (ERROR_CONST[state.order] * (state.M @ d)).astype(dtype)

        error_norm = norm(error / scale)
        opt_factor = jnp.maximum(
            jnp.asarray(MIN_FACTOR, dtype=dtype),
            safety * error_norm ** (-1 / (state.order + 1))
        )

        factor = jnp.where(
            ~converged & updated_jacobian,
            jnp.asarray(0.5, dtype=dtype),
            jnp.where(error_norm > 1, opt_factor, jnp.asarray(-jnp.inf, dtype=dtype)),
        )

        # Termination guards (T-005/T-008), mirroring the dopri5 ones — the
        # ``~accepted`` loop in ``step`` only exits when a step is accepted:
        # - A NaN state/RHS (e.g. a NaN parameter) never converges the Newton
        #   iteration and never passes the error test, so the retry factor
        #   stays positive forever.
        # - A diverging solution keeps failing the error test and halves dt
        #   geometrically without bound (there is no hmin floor by default).
        # Gated on ``would_retry`` so a healthy step (which the controller
        # accepts on its own, e.g. a tiny final sliver before the interval
        # boundary) can never be affected.  On a terminal step: poison the
        # state with NaN (it is by construction garbage — the error test
        # failed and the controller could not shrink dt further), force-accept
        # it, and jump dt to the remaining interval so the enclosing
        # simulation loop terminates promptly with the non-finite state
        # visible to the caller, instead of hanging.
        #
        # NOTE ``would_retry`` is NOT ``factor > 0``: a NaN error norm fails
        # ``error_norm > 1`` and would flow to the ``-inf`` (accept) branch —
        # i.e. without this guard a NaN step is silently *accepted* at the
        # current (often already collapsed) dt, and the simulation crawls to
        # tf in billions of tiny NaN steps.  A NaN error norm must count as
        # a retry so the bailout below can fire instead.
        would_retry = (~converged & updated_jacobian) | ~(error_norm <= 1.0)
        nonfinite = ~jnp.all(jnp.isfinite(y_new)) | jnp.isnan(error_norm)
        dt_floor = (
            jnp.finfo(dtype).eps
            * jnp.maximum(jnp.abs(state.t), jnp.abs(boundary_time))
        ).astype(dtype)
        # T-134: a non-finite trial step at a healthy dt is a *failed*
        # step, not a terminal one — a Newton blowup on a hard-switching
        # transition (e.g. a diode turning on) is routinely recoverable at
        # half the step. Route it through the normal rejection path
        # (retry at dt/2) and terminate only once dt has collapsed to the
        # floor. This preserves the T-005/T-008 no-hang guarantee: a
        # genuinely diverging solution fails every retry and reaches the
        # floor after ~60 geometric halvings, then bails out below.
        #
        # NaN-dt hazard: a NaN RHS at t0 makes ``initial_step_size``
        # return dt = NaN, and every comparison against a NaN dt is
        # False — a naive ``dt <= dt_floor`` bailout can then NEVER fire
        # and the retry loop spins forever in-kernel. ``dt_above_floor``
        # is therefore the ONLY dt predicate used here: NaN dt reads as
        # "not above the floor", which routes straight to the terminal
        # bailout.
        dt_above_floor = state.dt > dt_floor
        retry_nonfinite = nonfinite & dt_above_floor
        factor = jnp.where(
            retry_nonfinite, jnp.asarray(0.5, dtype=dtype), factor
        )
        force_accept = (would_retry | nonfinite) & ~dt_above_floor
        factor = jnp.where(
            force_accept, jnp.asarray(-jnp.inf, dtype=dtype), factor
        )
        # Capture failure diagnostics BEFORE the trial state is poisoned
        # with NaN below (which erases the which-rows information).  The
        # bailout force-accepts and exits the retry loop, so the values
        # written on the bailing attempt are the ones that survive the
        # loop; ``step`` forwards them to a host-side warning emitter.
        _fail_bailed, fail_t, fail_dt, fail_rows = fail_diag
        fail_diag = (
            force_accept,
            jnp.where(force_accept, jnp.asarray(state.t, dtype=fail_t.dtype), fail_t),
            jnp.where(force_accept, jnp.asarray(state.dt, dtype=fail_dt.dtype), fail_dt),
            jnp.where(force_accept, ~jnp.isfinite(y_new), fail_rows),
        )
        y_new = jnp.where(force_accept, jnp.full_like(y_new, jnp.nan), y_new)
        bailout_state = dataclasses.replace(
            state, dt=jnp.asarray(boundary_time - state.t, dtype=dtype)
        )
        state = jax.tree.map(
            partial(jnp.where, force_accept), bailout_state, state
        )

        # If the factor is negative, then the step is accepted.  Otherwise, we have to
        # update the step size and LU factorization for the next iteration.
        (state, accepted) = jax.tree.map(
            partial(jnp.where, factor > 0),
            (self._update_dt(state, factor), False),
            (state, True),
        )

        return state, accepted, y_new, d, n_iter, fail_diag

    def _update_difference_matrix(self, state, d):
        D, order = state.D, state.order
        D = D.at[order + 2].set(d - D[order + 1])
        D = D.at[order + 1].set(d)

        def body_fun(j, D):
            i = order - j
            return D.at[i].add(D[i + 1])

        D = lax.fori_loop(0, order + 1, body_fun, D)
        return dataclasses.replace(state, D=D)

    def _update_difference_matrix_order_change(self, state, d, y, n_iter):
        D, order = state.D, state.order
        state = self._update_difference_matrix(state, d)
        D = state.D
        dtype = state.y.dtype

        scale = (self.atol + self.rtol * jnp.abs(y)).astype(dtype)
        error = (ERROR_CONST[order] * d).astype(dtype)
        error_norm = norm(error / scale)
        safety = 0.9 * (2 * NEWTON_MAXITER + 1) / (2 * NEWTON_MAXITER + n_iter)

        # Optimal step size factor for order k-1 and k+1
        error_m_norm = jnp.where(
            order > 1,
            norm((ERROR_CONST[order - 1] * D[order] / scale).astype(dtype)),
            jnp.asarray(jnp.inf, dtype=dtype),
        )
        error_p_norm = jnp.where(
            order < MAX_ORDER,
            norm((ERROR_CONST[order + 1] * D[order + 2] / scale).astype(dtype)),
            jnp.asarray(jnp.inf, dtype=dtype),
        )

        error_norms = jnp.array([error_m_norm, error_norm, error_p_norm])
        exponent = (-1 / (jnp.arange(3) + order)).astype(dtype)
        factors = error_norms ** exponent

        # Select new order to maximize resulting step size, then scale
        # by the corresponding factor
        max_index = jnp.argmax(factors)
        order += max_index - 1

        opt_factor = jnp.minimum(jnp.asarray(MAX_FACTOR, dtype=dtype), safety * factors[max_index])
        state = dataclasses.replace(state, D=D, order=order)
        return self._update_dt(state, opt_factor)

    # Inherits docstring from `ODESolverBase`
    def step(self, func, boundary_time, solver_state):
        # https://github.com/scipy/scipy/blob/v1.13.0/scipy/integrate/_ivp/bdf.py#L310-L324
        h = solver_state.dt
        D = solver_state.D
        order = solver_state.order
        n_equal_steps = solver_state.n_equal_steps
        hmax = self.hmax
        hmin = self.hmin
        (h, D, n_equal_steps) = jax.tree.map(
            partial(jnp.where, h > hmax),
            (hmax, _update_D(D, order, hmax / h), 0),
            (h, D, n_equal_steps),
        )
        (h, D, n_equal_steps) = jax.tree.map(
            partial(jnp.where, h < hmin),
            (hmin, _update_D(D, order, hmin / h), 0),
            (h, D, n_equal_steps),
        )
        solver_state = dataclasses.replace(
            solver_state,
            dt=h,
            D=D,
            n_equal_steps=n_equal_steps,
        )

        def cond_fun(carry):
            accepted = carry[1]
            return ~accepted

        y = jnp.zeros_like(solver_state.y)
        d = jnp.zeros_like(solver_state.y)
        # Failure-diagnostic carry slot: (bailed, t_fail, dt_fail, row_mask).
        # Written by ``attempt_bdf_step`` on the terminal NaN/step-underflow
        # bailout (T-005/T-008/T-134) and forwarded to a host-side
        # ``UserWarning`` below.  ``dtype`` matches the state so the pytree
        # structure is loop-invariant.
        dtype = solver_state.y.dtype
        fail_diag = (
            False,
            jnp.asarray(jnp.nan, dtype=dtype),
            jnp.asarray(jnp.nan, dtype=dtype),
            jnp.zeros_like(solver_state.y, dtype=bool),
        )
        (solver_state, _accepted, y_new, d, n_iter, fail_diag) = lax.while_loop(
            cond_fun,
            partial(self.attempt_bdf_step, func, boundary_time),
            (solver_state, False, y, d, -1, fail_diag),
        )

        # T-005/T-008/T-134 diagnosability: if the retry loop exited through
        # the terminal bailout (state poisoned with NaN), surface a host-side
        # ``UserWarning`` with the failure time, the collapsed step size, and
        # which state rows went non-finite.  ``jax.debug.callback`` lifts the
        # emitter onto the host; wrapping it in ``lax.cond`` keeps the healthy
        # path free of host round-trips (an unconditional per-step callback
        # measured ~17% wall overhead on a stiff Van der Pol run; the cond
        # branch is ~free).  jit/vmap-safe on current JAX — the historical
        # "IO effect not supported in vmap-of-cond" limitation (T-002b) that
        # removed ``error_step_size_too_small`` no longer applies (verified
        # on jax 0.9.2); under ``vmap`` the cond lowers to a form where the
        # callback may run for non-bailing batches too, so the emitter
        # re-checks ``bailed`` host-side and stays silent for those.
        # The cond-gated host callback is runtime-free but costs ~0.3 s of
        # XLA compile per BDF model (measured on the rc_acausal_dae compile
        # gate), so it is only compiled in when
        # ``SimulatorOptions(bdf_nonfinite_diagnostics=True)`` stamped the
        # flag on this solver (same wiring as ``_cond_monitor``).  The
        # default path still gets a free post-run non-finite check in
        # ``simulate()`` that points at the flag.
        if getattr(self, "_nonfinite_diagnostics", False):
            def _warn_branch(diag):
                jax.debug.callback(_emit_bdf_nonfinite_warning, *diag)
                return 0

            lax.cond(fail_diag[0], _warn_branch, lambda diag: 0, fail_diag)

        # Occasionally floating point precision loss will mean that the next
        # time step will be less than machine epsilon from the boundary time.
        # In this case it's safe to assume that the boundary time has been reached.
        t_new = solver_state.t + solver_state.dt
        t_new = jnp.where(abs(t_new - boundary_time) < 10 * EPS, boundary_time, t_new)

        solver_state = dataclasses.replace(
            solver_state,
            t=t_new,
            y=y_new,
            n_equal_steps=solver_state.n_equal_steps + 1,
        )

        return jax.tree.map(
            partial(jnp.where, n_equal_steps < solver_state.order + 1),
            self._update_difference_matrix(solver_state, d),
            self._update_difference_matrix_order_change(solver_state, d, y_new, n_iter),
        )
