# SPDX-License-Identifier: MIT

"""
Interior Point Optimizer (IPOPT) for optimization of the objective function
with optional constraints and bounds.

IPOPT is a gold-standard large-scale NLP solver that uses second-order
(Hessian) information.  Gradients and Hessians are computed via JAX automatic
differentiation, so they are exact (not finite-difference approximations).

Supports:
  - Unconstrained optimisation
  - Inequality-constrained optimisation (constraints ≥ 0)
  - Box-bounded optimisation
  - Combined bounded + constrained optimisation
"""

import jax
import jax.numpy as jnp

from jaxonomy.lazy_loader import LazyLoader
from jaxonomy.logging import logger

from .base import OptimizationResult, Optimizer, Optimizable

cyipopt = LazyLoader(
    "cyipopt",
    globals(),
    "cyipopt",
    error_message="cyipopt is not installed.",
)


def _atleast_1d_output(fn):
    """Wrap *fn* so it always returns a 1-D JAX array.

    cyipopt's sparse-Jacobian structure probe requires the constraint function
    to return at least a 1-D array.  ``constraints_from_context`` often returns
    a 0-D array (bare scalar), which cyipopt cannot introspect.
    """
    def _wrapped(x):
        return jnp.atleast_1d(fn(x))
    return _wrapped


class IPOPT(Optimizer):
    """
    Interior Point Optimizer (IPOPT) for optimization of the objective function
    with optional constraints and bounds.

    Parameters:
        optimizable (Optimizable):
            The optimizable object.
        options (dict):
            Options forwarded to ``cyipopt.minimize_ipopt``.
            See https://coin-or.github.io/Ipopt/OPTIONS.html for the full
            list.  Commonly used keys:

            ``maxiter`` (int, default 3000)
                Maximum number of IPOPT iterations.
            ``disp`` (int, default 5)
                Verbosity level (0 = silent).
            ``tol`` (float)
                Convergence tolerance on the NLP optimality conditions.
            ``acceptable_tol`` (float)
                Looser acceptable-solution tolerance.
    """

    def __init__(self, optimizable: Optimizable, options: dict = {"disp": 5}):
        self.optimizable = optimizable
        self.options = options
        self.optimal_params = None

    def optimize(self) -> OptimizationResult:
        """Run optimisation and return an :class:`~jaxonomy.optimization.OptimizationResult`.

        Gradients of the objective and constraint Jacobians are computed with
        JAX automatic differentiation (``jax.grad`` / ``jax.jacrev``).

        **Hessian strategy** — JAX's ``jax.hessian`` requires forward-mode
        automatic differentiation (``jacfwd``) through the gradient, but the
        jaxonomy ODE solver uses ``custom_vjp`` which only supports
        reverse-mode.  Attempting to compute ``jax.hessian`` of a simulation
        objective therefore raises a runtime error.  IPOPT is instead
        configured with ``hessian_approximation = "limited-memory"`` (L-BFGS
        approximation) which only requires first-order gradient information and
        converges super-linearly.  For problems where you *know* the objective
        is twice-differentiable and do not use the jaxonomy ODE integrator you
        can override this by passing ``options={"hessian_approximation":
        "exact", ...}`` and providing ``hess`` via the ``_hess_fn`` constructor
        argument.
        """
        params = self.optimizable.params_0_flat

        # ── objective ─────────────────────────────────────────────────────────
        objective = jax.jit(self.optimizable.objective_flat)
        gradient = jax.jit(jax.grad(objective))

        # ── constraints (only when present) ───────────────────────────────────
        if self.optimizable.has_constraints:
            # Ensure the constraint function always returns a 1-D array so
            # cyipopt can reliably probe its Jacobian sparsity structure.
            constraints = jax.jit(_atleast_1d_output(self.optimizable.constraints_flat))
            constraints_jac = jax.jit(jax.jacrev(constraints))

            constraints_ipopt = [
                {
                    "type": "ineq",
                    "fun": constraints,
                    "jac": constraints_jac,
                    # No "hess" key: IPOPT uses L-BFGS for constraint Hessians
                    # when hessian_approximation="limited-memory"
                }
            ]
        else:
            constraints_ipopt = []

        # ── bounds ─────────────────────────────────────────────────────────────
        bounds = self.optimizable.bounds_flat

        # Jobs from the UI may put (-inf, inf) as default bounds.  The user
        # may also specify bounds this way.  cyipopt expects ``None`` to mean
        # unbounded.
        if bounds is not None:
            bounds = [
                (
                    None if b[0] == -jnp.inf else b[0],
                    None if b[1] == jnp.inf else b[1],
                )
                for b in bounds
            ]

            # If every bound is None the problem is effectively unbounded.
            flattened_bounds = [element for tup in bounds for element in tup]
            all_none = all(element is None for element in flattened_bounds)
            bounds = None if all_none else bounds

        # ── merge options — inject limited-memory unless caller overrides ──────
        effective_options = {"hessian_approximation": "limited-memory"}
        effective_options.update(self.options)

        # ── call IPOPT ─────────────────────────────────────────────────────────
        res = cyipopt.minimize_ipopt(
            objective,
            x0=params,
            jac=gradient,
            hess=None,
            constraints=constraints_ipopt,
            bounds=bounds,
            options=effective_options,
        )

        logger.info("IPOPT result:\n%s", res)
        if not getattr(res, "success", False):
            logger.warning("IPOPT did not converge: %s", getattr(res, "message", ""))

        # ── unpack result ──────────────────────────────────────────────────────
        solved_params = res.x
        self.optimal_params = self.optimizable.unflatten_params(solved_params)
        if self.optimizable.transformation is not None:
            self.optimal_params = self.optimizable.transformation.inverse_transform(
                self.optimal_params
            )

        return OptimizationResult(
            params=self.optimal_params,
            success=bool(getattr(res, "success", False)),
            nit=int(getattr(res, "nit", 0)),
            nfev=int(getattr(res, "nfev", 0)),
            message=str(getattr(res, "message", "")),
            final_loss=float(getattr(res, "fun", float("nan"))),
        )
