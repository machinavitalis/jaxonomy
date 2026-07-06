# SPDX-License-Identifier: MIT

from __future__ import annotations
from typing import TYPE_CHECKING
from functools import partial

import numpy as np

import jax
import jax.numpy as jnp
from jax._src.numpy.util import promote_dtypes_inexact
from jax.flatten_util import ravel_pytree
from jax.experimental.ode import ravel_first_arg

from ..ode_solver import ODESolverBase, ODESolverState

from jaxonomy.lazy_loader import LazyLoader, LazyModuleAccessor


if TYPE_CHECKING:
    import equinox as eqx
    from scipy.linalg import block_diag

    from ...framework import ContextBase
else:
    eqx = LazyLoader("eqx", globals(), "equinox")
    scipy_linalg = LazyLoader("scipy_linalg", globals(), "scipy.linalg")
    block_diag = LazyModuleAccessor(scipy_linalg, "block_diag")


__all__ = [
    "ODESolverImpl",
    "ODESolverState",
    "norm",
    "error_step_size_too_small",
]


def norm(x):
    return (jnp.linalg.norm(x) / x.size**0.5).astype(x.dtype)


def error_step_size_too_small(t, h, tf, disable):
    """Adaptive-solver step-size-too-small diagnostic.

    Historically this routine used ``jax.debug.callback`` to raise a
    descriptive ``ODESolverError`` when the adaptive step size collapsed
    toward machine epsilon.  That was helpful in single-simulation runs
    but broke ``simulate_batch(use_vmap=True)`` with
    ``NotImplementedError: IO effect not supported in vmap-of-cond`` —
    the callback sat inside an adaptive-step ``lax.cond`` that ``vmap``
    then tried to lift (T-002b).

    The callback has been removed.  A runaway solve now manifests as
    either a NaN-valued state (the adaptive step size clips to ``hmin``
    and the error ratio is force-accepted) or, more commonly, the
    simulator hitting its ``max_major_steps`` budget with a descriptive
    ``max_major_steps exceeded`` error from the main loop.  Both paths
    surface the underlying problem without requiring an IO effect
    inside the ODE solver's inner loop.

    The function signature is preserved for backward compatibility with
    existing call sites; it is now a pure no-op.
    """
    del t, h, tf, disable  # unused; kept for signature compatibility


class _StableUnravel:
    """Hashable, structurally-comparable unravel callable for pytree restoration.

    Unlike the closure returned by :func:`jax.flatten_util.ravel_pytree`, instances
    of this class implement ``__eq__`` and ``__hash__`` by *value* (based on the
    pytree structure, leaf shapes, and leaf dtypes).  Two instances representing the
    same pytree structure will therefore compare *equal*, which is required when the
    unravel is stored as JAX pytree ``aux_data`` (as in ``Dopri5State`` /
    ``BDFState``).

    Background
    ----------
    JAX's :func:`~jax.custom_vjp` mechanism traces the decorated function once to
    discover the expected output pytree structure, then traces the ``fwd`` rule a
    second time to obtain the actual primal + residuals.  Both traces independently
    call ``ravel_pytree``, producing two **different** ``HashablePartial`` closure
    objects that compare *unequal* (because ``HashablePartial.__eq__`` uses
    ``is``-identity for the wrapped function).  The resulting pytree structure
    mismatch raises ``TypeError: Custom VJP fwd rule … must produce a pair…``.

    ``_StableUnravel`` avoids this by comparing on structure rather than identity,
    so the two traces produce instances that are ``==`` to each other.
    """

    __slots__ = ("_treedef", "_leaf_shapes", "_leaf_dtypes")

    def __init__(self, example_pytree):
        leaves, treedef = jax.tree_util.tree_flatten(example_pytree)
        self._treedef = treedef
        self._leaf_shapes = tuple(
            np.asarray(leaf).shape if not hasattr(leaf, "shape") else leaf.shape
            for leaf in leaves
        )
        self._leaf_dtypes = tuple(
            np.asarray(leaf).dtype if not hasattr(leaf, "dtype") else leaf.dtype
            for leaf in leaves
        )

    # ------------------------------------------------------------------
    # Core callable
    # ------------------------------------------------------------------

    def __call__(self, flat):
        """Unravel a flat JAX array back to the original pytree structure."""
        n_leaves = len(self._leaf_shapes)
        if n_leaves == 0:
            return self._treedef.unflatten([])

        sizes = [int(np.prod(s)) if s else 1 for s in self._leaf_shapes]
        splits = list(np.cumsum(sizes[:-1]))
        parts = jnp.split(flat, splits) if splits else [flat]

        leaves = [
            part.reshape(shape).astype(dtype)
            for part, shape, dtype in zip(parts, self._leaf_shapes, self._leaf_dtypes)
        ]
        return self._treedef.unflatten(leaves)

    # ------------------------------------------------------------------
    # Value-based equality (the key property for pytree aux_data safety)
    # ------------------------------------------------------------------

    def __eq__(self, other):
        if not isinstance(other, _StableUnravel):
            return False
        return (
            self._treedef == other._treedef
            and self._leaf_shapes == other._leaf_shapes
            and self._leaf_dtypes == other._leaf_dtypes
        )

    def __hash__(self):
        return hash((
            self._treedef,
            self._leaf_shapes,
            tuple(str(d) for d in self._leaf_dtypes),
        ))

    def __repr__(self):
        return (
            f"_StableUnravel(treedef={self._treedef}, "
            f"shapes={self._leaf_shapes}, dtypes={self._leaf_dtypes})"
        )


class ODESolverImpl(ODESolverBase):
    """Common implementation for JAX ODE solvers.

    This class includes common functionality for JAX ODE solvers, such as
    overriding the `initialize` method to provide custom VJP definitions
    and a method for computing the initial step size.
    """

    def _finalize(self):
        self.hmin = self.min_step_size or 0.0
        self.hmax = self.max_step_size or jnp.inf
        self.initialize = self._override_initialize_vjp()

    def initialize(self, context: ContextBase, dt: float = None) -> ODESolverImpl:
        # The abstract base class requires an implementation of this method.
        # However, in order to provide a custom VJP definition, the class will
        # override this method with a custom VJP definition in `_override_initialize_vjp`.
        # Hence, this is a dummy implementation that should never actually be called.
        raise RuntimeError(
            "Default method should have been overridden in __post_init__"
        )

    def _initialize_state(
        self, func, t0, xc0, mass, unravel, *args, dt=None
    ) -> ODESolverState:
        raise NotImplementedError

    # Inherits docstring from `ODESolverBase`
    def _initialize(self, context: ContextBase, dt: float = None) -> ODESolverState:
        xc0 = context.continuous_state
        t0 = context.time

        # Use _StableUnravel instead of the closure returned by ravel_pytree.
        # ravel_pytree creates a new HashablePartial on every call, and two
        # HashablePartials wrapping different-identity closures compare *unequal*.
        # When JAX traces advance_to twice (once for structure discovery, once for
        # the custom_vjp fwd rule), the two Dopri5State / BDFState objects would
        # have different unravel objects in aux_data, causing a pytree structure
        # mismatch.  _StableUnravel has value-based __eq__ / __hash__, so two
        # instances representing the same pytree shape compare equal.
        stable_unravel = _StableUnravel(xc0)
        xc0, _ = ravel_pytree(xc0)  # get the flat array; discard JAX's unravel

        # Note that the mass matrix is known here statically, so it's
        # okay to use the scipy.linalg.block_diag function rather than the
        # jax.scipy version.
        self.mass = None
        if self.system.has_mass_matrix:
            self.mass = block_diag(*jax.tree.leaves(self.system.mass_matrix))

        from jax.api_util import debug_info
        self.flat_ode_rhs = ravel_first_arg(self.ode_rhs, stable_unravel, debug_info("flat_ode_rhs", self.ode_rhs, (), {}))
        return self._initialize_state(
            self.flat_ode_rhs, t0, xc0, self.mass, stable_unravel, context, dt=dt
        )

    def _override_initialize_vjp(self):
        if not self.enable_autodiff:
            return self._initialize

        # if self.system.has_mass_matrix:
        #     # TODO: See "The Adjoint DAE System and Its Numerical Solution"
        #     # by Cao, Li, Petzold, and Serban for a discussion of how to handle
        #     # adjoint sensitivity analysis for DAEs.
        #     # https://www.researchgate.net/publication/230872722_Adjoint_Sensitivity_Analysis_for_Differential-Algebraic_Equations_The_Adjoint_DAE_System_and_Its_Numerical_Solution
        #     raise NotImplementedError(
        #         "Automatic differentiation is not currently supported for systems "
        #         "with non-trivial mass matrices."
        #     )

        def _wrapped_initialize(self: ODESolverImpl, context, dt=None):
            return self._initialize(context, dt=dt)

        def _wrapped_initialize_fwd(self: ODESolverImpl, context, dt):
            # Need to correctly initialize the time step if it is not
            # provided.  This is probably not the most efficient
            # implementation, since it results in multiple call sites
            # to the RHS evaluation.  However, it will not typically end
            # up in the JIT computation graph unless differentiating through
            # reset maps. From some simple timing, the overhead seems to be pretty
            # minimal, at least.

            if dt is None:
                state = self._initialize(context, dt)
                dt = state.dt

            primals, vjp_fun = jax.vjp(partial(self._initialize, dt=dt), context)
            residuals = (vjp_fun,)
            return primals, residuals

        def _wrapped_initialize_adj(self, dt, residuals, adjoints):
            (vjp_fun,) = residuals
            (context_adj,) = vjp_fun(adjoints)
            return (context_adj,)

        initialize = jax.custom_vjp(_wrapped_initialize, nondiff_argnums=(0, 2))
        initialize.defvjp(_wrapped_initialize_fwd, _wrapped_initialize_adj)

        # Copy docstring and type hints
        initialize.__doc__ = super().initialize.__doc__
        initialize.__annotations__ = self._initialize.__annotations__

        return partial(initialize, self)

    def initialize_adjoint(self, func, init_adj_state, tf, context):
        """Initialize the solver configured for the adjoint reverse-time solve."""

        def adj_dynamics(aug_state, neg_t, context):
            """Original system augmented with vjp_y, vjp_t and vjp_args."""
            y, y_bar, *_ = aug_state
            # `neg_t` here is negative time, so we need to negate again to get back to
            # normal time.  The VJP is filtered to only differentiable arguments
            y_dot, vjpfun = eqx.filter_vjp(func, y, -neg_t, context)
            return (-y_dot, *vjpfun(y_bar))

        n = len(init_adj_state[0])  # Number of states in the original system

        # The adjoint solver state is used with checkpoint=False (no nested VJP),
        # so we don't need _StableUnravel here — JAX's standard unravel is fine.
        init_adj_state, unravel = ravel_pytree(init_adj_state)
        from jax.api_util import debug_info
        adj_dynamics = ravel_first_arg(adj_dynamics, unravel, debug_info("adj_dynamics", adj_dynamics, (), {}))
        if self.mass is None:
            adj_mass = np.eye(len(init_adj_state))
        else:
            # The first two blocks of the mass matrix are [M, M.T], followed by
            # the identity matrix for the rest of the state.
            adj_mass = block_diag(
                self.mass, self.mass.T, np.eye(len(init_adj_state) - 2 * n)
            )
        adj_solver_state = self._initialize_state(
            adj_dynamics, -tf, init_adj_state, adj_mass, unravel, context
        )
        return adj_solver_state, adj_dynamics

    def initial_step_size(self, func, y0, t0, order, f0, *args):
        # Algorithm from:
        # E. Hairer, S. P. Norsett G. Wanner,
        # Solving Ordinary Differential Equations I: Nonstiff Problems, Sec. II.4.
        y0, f0 = promote_dtypes_inexact(y0, f0)
        dtype = y0.dtype

        scale = self.atol + jnp.abs(y0) * self.rtol
        scale = scale.astype(dtype)
        d0 = norm(y0 / scale)
        d1 = norm(f0 / scale)

        h0 = jnp.where((d0 < 1e-5) | (d1 < 1e-5), 1e-6, 0.01 * d0 / d1)
        y1 = y0 + h0.astype(dtype) * f0
        f1 = func(y1, t0 + h0, *args)
        d2 = norm((f1 - f0) / scale) / h0

        h1 = jnp.where(
            (d1 <= 1e-15) & (d2 <= 1e-15),
            jnp.maximum(1e-6, h0 * 1e-3),
            (0.01 / jnp.maximum(d1, d2)) ** (1.0 / (order + 1.0)),
        )

        dt = jnp.minimum(100.0 * h0, h1)
        return jnp.clip(dt, min=self.hmin, max=self.hmax)
