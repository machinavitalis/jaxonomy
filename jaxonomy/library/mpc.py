# SPDX-License-Identifier: MIT

import warnings
from functools import partial
from typing import TYPE_CHECKING

import numpy as np
import jax
import jax.numpy as jnp

from ..framework import LeafSystem
from ..backend import cond
from ..lazy_loader import LazyLoader

if TYPE_CHECKING:
    from jax.scipy import linalg
else:
    linalg = LazyLoader("linalg", globals(), "jax.scipy.linalg")


_load_error_msg = (
    "OSQP is not installed. You can install it with:\n"
    "pip install cmake\n"
    "pip install jaxonomy[nmpc]"
)

osqp = LazyLoader(
    "osqp",
    globals(),
    "osqp",
    error_message=_load_error_msg,
)


def _osqp_warm_start_kwarg():
    """Name of OSQP's warm-start setting for the installed version.

    OSQP renamed the ``warm_start`` setting to ``warm_starting`` in 1.0, so the
    same code must pick the right keyword to run on both the 0.6.x line and the
    1.x line (the block otherwise raises ``TypeError`` at ``setup`` on whichever
    version it wasn't written for).
    """
    version = getattr(osqp, "__version__", "") or ""
    try:
        major = int(version.split(".")[0])
    except (ValueError, IndexError):
        major = 0
    return "warm_starting" if major >= 1 else "warm_start"

__all__ = [
    "LinearDiscreteTimeMPC",
    "LinearDiscreteTimeMPC_OSQP",
]


class LinearDiscreteTimeMPC(LeafSystem):
    """Model predictive control for a linear discrete-time system.

    Solves a constrained quadratic program at each time step using OSQP
    via ``jax.pure_callback``, making it compatible with JAX's JIT compiler.

    Notes:
        This block is *feedthrough*: the QP solver runs every time the output
        port is evaluated.  Pair with a zero-order hold so the solver is
        invoked only once per MPC step.

    Args:
        lin_sys: Linearized system (continuous-time A, B matrices).
        Q: State cost matrix (n×n).
        R: Input cost matrix (m×m).
        N: Prediction horizon (number of steps).
        dt: Sampling period for Euler discretization.
        x_ref: Terminal state reference (length-n array).
        lbu: Lower bound on control input (scalar or length-m array).
        ubu: Upper bound on control input (scalar or length-m array).
        warm_start: Whether to warm-start the OSQP solver between solves.
    """

    def __init__(
        self,
        lin_sys,
        Q,
        R,
        N,
        dt,
        x_ref,
        lbu=-np.inf,
        ubu=np.inf,
        name=None,
        warm_start=False,
    ):
        super().__init__(name=name)
        lin_sys.create_context()
        self.n = lin_sys.A.shape[0]
        self.m = lin_sys.B.shape[1]
        self.N = N
        self.warm_start = warm_start

        # Euler discretization
        A = jnp.eye(self.n) + dt * lin_sys.A
        B = dt * lin_sys.B

        self.declare_input_port()

        # Shape of the full primal solution vector x = [x_0, u_0, ..., x_{N-1}, u_{N-1}]
        self._result_template = jnp.zeros((self.n + self.m) * self.N)

        self._make_solver(A, B, Q, R, lbu, ubu, N, x_ref)

        # Wrap the non-JAX OSQP solve in a pure_callback so it is JIT-compatible
        self._jax_solve = partial(
            jax.pure_callback, self._np_solve, self._result_template
        )

        self.declare_output_port(
            self._output,
            requires_inputs=True,
            period=dt,
            offset=0.0,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _make_solver(self, A, B, Q, R, lbu, ubu, N, xf):
        from scipy import sparse

        n, m = self.n, self.m
        I_A = jnp.eye(n)
        I_B = jnp.eye(m)

        def e(k):
            return jnp.zeros(N).at[k].set(1.0)

        # Block-diagonal cost matrix P = diag(Q, R, Q, R, ...) of size (n+m)*N
        P_dense = linalg.block_diag(*([Q, R] * N))

        # Equality / dynamics constraints
        L0 = jnp.eye(n, N * (n + m))
        L_defect = jnp.vstack(
            [
                jnp.kron(e(k), jnp.hstack([A, B]))
                + jnp.kron(e(k + 1), jnp.hstack([-I_A, 0 * B]))
                for k in range(N - 1)
            ]
        )
        Lf = jnp.kron(e(N - 1), jnp.hstack([I_A, 0 * B]))
        L_input = jnp.vstack(
            [jnp.kron(e(k), jnp.hstack([0 * B.T, I_B])) for k in range(N)]
        )
        L_dense = jnp.vstack([L0, L_defect, Lf, L_input])

        self._L_defect_rows = L_defect.shape[0]
        self._xf = xf
        self._lbu = lbu
        self._ubu = ubu

        # Precompute JIT-compiled bounds helper (used inside pure_callback)
        def _get_bounds(x0):
            lb = jnp.hstack(
                [x0, jnp.zeros(L_defect.shape[0]), xf, jnp.full(N * m, lbu)]
            )
            ub = jnp.hstack(
                [x0, jnp.zeros(L_defect.shape[0]), xf, jnp.full(N * m, ubu)]
            )
            return lb, ub

        self._get_bounds = jax.jit(_get_bounds)

        # Set up OSQP with dummy initial bounds; updated before each solve
        lb0, ub0 = _get_bounds(jnp.zeros(n))
        self.solver = osqp.OSQP()
        self.solver.setup(
            P=sparse.csc_matrix(np.array(P_dense)),
            q=np.zeros(N * (n + m)),
            A=sparse.csc_matrix(np.array(L_dense)),
            l=np.array(lb0),
            u=np.array(ub0),
            verbose=False,
            **{_osqp_warm_start_kwarg(): self.warm_start},
        )

    def _np_solve(self, time, state, x0):
        """Non-JAX solve called via pure_callback. Inputs are concrete numpy arrays."""
        lb, ub = self._get_bounds(x0)
        self.solver.update(l=np.array(lb), u=np.array(ub))
        sol = self.solver.solve()
        return sol.x

    def _dummy_solve(self, _time, _state, *_inputs, **_params):
        """Return inf when time is inf (minor ODE steps guarding against OSQP errors)."""
        return jnp.full(self._result_template.shape, jnp.inf)

    def _output(self, time, state, *inputs):
        args = (time, state, *inputs)
        xu_flat = cond(jnp.isinf(time), self._dummy_solve, self._jax_solve, *args)
        xu_traj = xu_flat.reshape((self.n + self.m, self.N), order="F")
        return xu_traj[self.n:, 0]


class LinearDiscreteTimeMPC_OSQP(LinearDiscreteTimeMPC):
    """Deprecated alias for :class:`LinearDiscreteTimeMPC`.

    Both classes now use OSQP via ``jax.pure_callback``.
    Use :class:`LinearDiscreteTimeMPC` directly.
    """

    def __init__(self, *args, **kwargs):
        warnings.warn(
            "LinearDiscreteTimeMPC_OSQP is deprecated and will be removed in a "
            "future release. Use LinearDiscreteTimeMPC instead — both classes now "
            "use OSQP via jax.pure_callback.",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__(*args, **kwargs)
