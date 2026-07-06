# SPDX-License-Identifier: MIT

"""High-level autodiff-based parameter tuning for jaxonomy diagrams.

This module wraps the lower-level `Optimizable` framework into a single
function-call API for the common case: "I have a diagram with some scalar
parameters (controller gains, set-points, time constants, etc.) and I want
to find values that minimize an objective evaluated over a simulation."

The simulator is differentiable, so the gradient of the objective with
respect to the parameters is computed automatically via JAX, and a
standard nonlinear optimizer (Scipy L-BFGS-B by default, or any Optax
optimizer) is used to find a local minimum.

Why this exists
---------------
Without this API, tuning controller gains in a closed-loop diagram is
typically done by hand: edit a constant, re-run the simulation, look at
a plot, repeat. Each iteration is wall-clock-bound by simulation time
and exploits exactly zero of the gradient information that JAX already
computes for free. With this API the same loop runs at simulation speed
times a small constant overhead per gradient step, and finds locally
optimal parameter values without human intervention.

Example
-------
    >>> import jax.numpy as jnp
    >>> import jaxonomy
    >>> from jaxonomy.optimization import tune_parameters
    >>>
    >>> # Build a closed-loop diagram once. The controller exposes some
    >>> # parameters via the jaxonomy `Parameter` mechanism (or via context
    >>> # manipulation in `set_params`).
    >>> diagram, plant, controller = build_my_loop(K_p_init=1.0)
    >>> ctx = diagram.create_context()
    >>>
    >>> def set_params(context, params):
    ...     # Write parameter values into the context.
    ...     return context.with_parameter(controller, "K_p", params["K_p"])
    >>>
    >>> def objective(results_context):
    ...     # Final continuous state, want it near zero (regulation).
    ...     x_f = results_context.continuous_state
    ...     return jnp.sum(x_f ** 2)
    >>>
    >>> result = tune_parameters(
    ...     diagram=diagram,
    ...     base_context=ctx,
    ...     sim_t_span=(0.0, 10.0),
    ...     params_0={"K_p": 1.0},
    ...     set_params=set_params,
    ...     objective_fn=objective,
    ...     bounds={"K_p": (0.0, 100.0)},
    ...     optimizer="scipy-lbfgs",
    ...     n_iter=50,
    ... )
    >>> print("Optimal K_p:", float(result.params["K_p"]))
    >>> print("Final objective:", float(result.objective))
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional, Tuple, Union

import jax
import jax.numpy as jnp
import numpy as np

from jaxonomy.logging import logger
from jaxonomy.optimization.framework import (
    Optimizable,
    Scipy,
    OptimizationResult,
)
from jaxonomy.simulation import SimulatorOptions, estimate_max_major_steps


__all__ = [
    "tune_parameters",
    "TuningResult",
]


# ---------------------------------------------------------------------------
# Public types
# ---------------------------------------------------------------------------


@dataclass
class TuningResult:
    """Result of a `tune_parameters` call.

    Attributes:
        params: Optimal parameter values as a dict {name: jax.Array}.
        objective: Final objective value (scalar).
        history: List of (iteration, objective) tuples if tracking enabled.
        success: True if the optimizer reported successful convergence.
        message: Human-readable status from the underlying optimizer.
        raw: Underlying `OptimizationResult` from the optimizer framework
            (for inspection of optimizer-specific fields).
    """

    params: Dict[str, jax.Array]
    objective: float
    history: list = field(default_factory=list)
    success: bool = True
    message: str = ""
    raw: Optional[OptimizationResult] = None


# ---------------------------------------------------------------------------
# Internal Optimizable subclass — constructed dynamically per call
# ---------------------------------------------------------------------------


class _CallableOptimizable(Optimizable):
    """An Optimizable whose `prepare_context` and `objective_from_context`
    are user-supplied callables.

    This is the lightweight bridge between the high-level `tune_parameters`
    surface and the existing Optimizable / Optimizer framework. We do not
    expose this class publicly — users get one-call simplicity through
    `tune_parameters`.
    """

    def __init__(
        self,
        diagram,
        base_context,
        sim_t_span,
        params_0,
        bounds,
        set_params_fn,
        objective_fn,
        sim_options,
    ):
        self._set_params_fn = set_params_fn
        self._objective_fn = objective_fn
        self._params_init = dict(params_0)
        super().__init__(
            diagram=diagram,
            base_context=base_context,
            sim_t_span=sim_t_span,
            params_0=params_0,
            bounds=bounds,
            sim_options=sim_options,
        )

    def prepare_context(self, context, params: dict):
        return self._set_params_fn(context, params)

    def objective_from_context(self, context):
        return self._objective_fn(context)

    def optimizable_params(self, context) -> dict:
        # The framework calls this when params_0 was not supplied; in our
        # API params_0 is always passed explicitly, so we just echo it.
        # Each value is wrapped in a JAX-compatible array.
        return {k: jnp.asarray(v) for k, v in self._params_init.items()}


# ---------------------------------------------------------------------------
# Public function
# ---------------------------------------------------------------------------


# Optimizer aliases supported as strings. Anything not in this map is
# interpreted as a scipy method name and routed to the Scipy optimizer.
# Scipy method names are case-sensitive in the underlying framework.
_OPTIMIZER_ALIASES = {
    "scipy": ("scipy", "L-BFGS-B"),
    "scipy-lbfgs": ("scipy", "L-BFGS-B"),
    "scipy-lbfgsb": ("scipy", "L-BFGS-B"),
    "scipy-bfgs": ("scipy", "BFGS"),
    "scipy-slsqp": ("scipy", "SLSQP"),
    "scipy-trust-constr": ("scipy", "trust-constr"),
    "scipy-nelder-mead": ("scipy", "Nelder-Mead"),
    "scipy-nm": ("scipy", "Nelder-Mead"),
    "optax": ("optax", "adam"),
    "optax-adam": ("optax", "adam"),
    "optax-sgd": ("optax", "sgd"),
}

# Set of accepted scipy method names (mixed case). Lower-case aliases are
# converted to these canonical forms.
_SCIPY_METHODS = {
    "nelder-mead": "Nelder-Mead",
    "powell": "Powell",
    "cg": "CG",
    "bfgs": "BFGS",
    "newton-cg": "Newton-CG",
    "l-bfgs-b": "L-BFGS-B",
    "tnc": "TNC",
    "cobyla": "COBYLA",
    "slsqp": "SLSQP",
    "trust-constr": "trust-constr",
    "dogleg": "dogleg",
    "trust-ncg": "trust-ncg",
    "trust-krylov": "trust-krylov",
    "trust-exact": "trust-exact",
}


def tune_parameters(
    diagram,
    base_context,
    sim_t_span: Tuple[float, float],
    params_0: Dict[str, Any],
    set_params: Callable[[Any, Dict[str, jax.Array]], Any],
    objective_fn: Callable[[Any], jax.Array],
    bounds: Optional[Dict[str, Tuple[float, float]]] = None,
    optimizer: str = "scipy-lbfgs",
    n_iter: int = 100,
    learning_rate: float = 0.05,
    sim_options: Optional[SimulatorOptions] = None,
    verbose: bool = True,
) -> TuningResult:
    """Tune scalar parameters of a jaxonomy diagram to minimize an objective.

    The simulator is differentiated through using JAX autodiff; the gradient
    of `objective_fn` with respect to each entry of `params_0` is computed
    automatically, and an optimizer minimizes the objective.

    Args:
        diagram: A built jaxonomy diagram.
        base_context: A `Context` created from the diagram. The optimizer
            calls `set_params(base_context, params)` each iteration to inject
            the current parameter values, then advances the simulator over
            `sim_t_span`, then evaluates `objective_fn` on the final context.
        sim_t_span: `(t0, tf)` simulation interval.
        params_0: Dict of initial parameter values. Each value must be a
            scalar or a JAX-compatible array.
        set_params: Callback `(context, params_dict) -> updated_context`.
            Typical implementations write parameter values into LeafSystem
            parameters via `context.with_parameter(...)`, or modify initial
            states.
        objective_fn: Callback `(results_context) -> scalar`. The optimizer
            minimizes this. Must be differentiable through JAX.
        bounds: Optional dict `{param_name: (lb, ub)}` for box constraints.
            Only honoured by box-constrained optimizers (l-bfgs-b, slsqp,
            trust-constr); ignored otherwise.
        optimizer: Optimizer alias or scipy method name. Defaults to
            "scipy-lbfgs". See `_OPTIMIZER_ALIASES` for shortcuts.
        n_iter: Maximum number of optimizer iterations.
        learning_rate: Learning rate for optax optimizers (ignored by scipy).
        sim_options: Optional `SimulatorOptions`. If None, a default with
            autodiff enabled is used.
        verbose: If True, log progress to the jaxonomy logger.

    Returns:
        A `TuningResult` with optimal parameter values, the final objective,
        and a reference to the raw optimizer result.

    Notes:
        - **Discrete parameters** (e.g., a horizon length `N`, a state-machine
          guard threshold) are not differentiable through the simulator and
          should be left as fixed hyperparameters. If you need to sweep them,
          wrap `tune_parameters` in an outer loop or grid search.
        - **Saturation regions** (`jnp.clip`, `jax.lax.cond` with hard
          switches) have zero gradient. Tuning parameters whose value
          determines a saturation region may be impossible from a starting
          point already saturated; consider warm-starting away from the
          saturation boundary.
        - **Bounds enforcement**: with `scipy-lbfgs` / `slsqp` /
          `trust-constr`, bounds are honoured by the solver. With `optax`
          optimizers, bounds are not enforced; clip parameters yourself in
          `set_params` if you need them.

    See also:
        `jaxonomy.optimization.Optimizable` — the lower-level interface this
        function wraps. Use it directly if you need stochastic variables,
        constraints, or batched evaluations.
    """
    if sim_options is None:
        sim_options = SimulatorOptions(enable_autodiff=True)

    # Default max_major_steps if user didn't provide it
    if sim_options.max_major_steps is None:
        sim_options = dataclasses.replace(
            sim_options,
            max_major_steps=estimate_max_major_steps(diagram, sim_t_span),
        )

    # Normalize bounds: the Optimizable framework expects a dict of (lb, ub)
    # tuples; missing keys mean unbounded. We pass through as-is.
    optimizable = _CallableOptimizable(
        diagram=diagram,
        base_context=base_context,
        sim_t_span=sim_t_span,
        params_0=params_0,
        bounds=bounds,
        set_params_fn=set_params,
        objective_fn=objective_fn,
        sim_options=sim_options,
    )

    # Resolve optimizer choice
    opt_kind, opt_method = _resolve_optimizer(optimizer)

    if verbose:
        logger.info(
            f"tune_parameters: optimizer={opt_kind}/{opt_method}, "
            f"params={list(params_0.keys())}, n_iter={n_iter}"
        )

    raw_result: OptimizationResult
    if opt_kind == "scipy":
        opt = Scipy(
            optimizable=optimizable,
            opt_method=opt_method,
            opt_method_config={"maxiter": int(n_iter), "disp": bool(verbose)},
        )
        raw_result = opt.optimize()
    elif opt_kind == "optax":
        # The Optax runner in jaxonomy is wired for stochastic-variable
        # optimization. Tuning deterministic diagram parameters via optax is
        # not yet exposed at this top-level API. Users who need it should
        # build an OptimizableWithStochasticVars directly.
        raise NotImplementedError(
            "Optax-based tuning for deterministic diagrams is not yet supported "
            "by `tune_parameters`. Use optimizer='scipy-lbfgs' (default) for now, "
            "or drop down to OptimizableWithStochasticVars + Optax directly."
        )
    else:
        raise ValueError(f"Unknown optimizer kind: {opt_kind!r}")

    # Reassemble result
    optimal_params = raw_result.params
    final_objective = (
        float(raw_result.final_loss)
        if raw_result.final_loss is not None
        else float("nan")
    )
    success = bool(getattr(raw_result, "success", True))
    message = str(getattr(raw_result, "message", ""))

    if verbose:
        logger.info(
            f"tune_parameters: done. final objective={final_objective:.6g}, "
            f"success={success}"
        )

    return TuningResult(
        params=optimal_params,
        objective=final_objective,
        history=list(getattr(raw_result, "loss_history", []) or []),
        success=success,
        message=message,
        raw=raw_result,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _resolve_optimizer(spec: str) -> Tuple[str, str]:
    """Resolve an optimizer alias to (kind, method)."""
    if spec in _OPTIMIZER_ALIASES:
        return _OPTIMIZER_ALIASES[spec]
    # Bare scipy method name passthrough — case-insensitive matching against
    # canonical names.
    canonical = _SCIPY_METHODS.get(spec.lower())
    if canonical is not None:
        return ("scipy", canonical)
    raise ValueError(
        f"Unknown optimizer {spec!r}. Use one of {sorted(_OPTIMIZER_ALIASES)} "
        f"or a scipy method name."
    )


