# SPDX-License-Identifier: MIT

"""
Multi-start wrapper for any jaxonomy optimizer.

Runs N optimizations from different initial points and returns all results
plus the best one.  Handles non-convex problems where gradient-based methods
may get stuck in local minima.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Callable

import jax.numpy as jnp
import numpy as np

from .framework.base.optimizable import Optimizable
from .framework.base.optimizer import OptimizationResult


@dataclass
class MultiStartResult:
    """
    Results from a multi-start optimization run.

    Attributes:
        results: All ``OptimizationResult`` objects — one per start.
        best_result: The result with the lowest ``final_loss`` among
            successful runs.
        best_start_index: Index into ``results`` of the best run.
        n_starts: Total number of starts attempted.
        n_successful: Number of starts that reported ``success=True``.
    """

    results: list[OptimizationResult]
    best_result: OptimizationResult
    best_start_index: int
    n_starts: int
    n_successful: int

    def summary(self) -> str:
        lines = [
            f"MultiStartResult: {self.n_successful}/{self.n_starts} starts converged.",
            f"Best start: #{self.best_start_index}  "
            f"final_loss={self.best_result.final_loss:.6g}  "
            f"params={self.best_result.params}",
        ]
        for i, r in enumerate(self.results):
            marker = " ← best" if i == self.best_start_index else ""
            loss_str = f"{r.final_loss:.6g}" if r.final_loss is not None else "N/A"
            lines.append(
                f"  [{i}] success={r.success}  loss={loss_str}  "
                f"nit={r.nit}{marker}"
            )
        return "\n".join(lines)

    def __repr__(self) -> str:
        return self.summary()


def _uniform_sampler(
    n_starts: int,
    params_0_flat: np.ndarray,
    scale: float,
    bounds_flat: list | None,
    rng: np.random.Generator,
) -> np.ndarray:
    """Default sampling: uniform in [p0 - scale*|p0|, p0 + scale*|p0|]."""
    magnitude = np.maximum(np.abs(params_0_flat), 1.0)
    lo = params_0_flat - scale * magnitude
    hi = params_0_flat + scale * magnitude
    samples = rng.uniform(lo, hi, size=(n_starts, len(params_0_flat)))
    if bounds_flat is not None:
        for i, (lb, ub) in enumerate(bounds_flat):
            if lb is not None and lb != -jnp.inf:
                samples[:, i] = np.maximum(samples[:, i], float(lb))
            if ub is not None and ub != jnp.inf:
                samples[:, i] = np.minimum(samples[:, i], float(ub))
    return samples


class MultiStart:
    """
    Multi-start wrapper for any jaxonomy optimizer.

    Runs ``n_starts`` optimizations from different initial points and returns
    all results as well as the best one (lowest ``final_loss``).

    Parameters
    ----------
    optimizable : Optimizable
        The problem to optimize.  Must be a jaxonomy ``Optimizable`` instance.
    optimizer_factory : Callable[[Optimizable], Optimizer]
        A *factory* function that takes an ``Optimizable`` (potentially with
        different initial parameters) and returns a ready-to-run optimizer.
        Example::

            factory = lambda opt: Scipy(opt, "L-BFGS-B",
                                        opt_method_config={"maxiter": 40},
                                        use_autodiff_grad=True)
            ms = MultiStart(optimizable, factory, n_starts=8, seed=0)
            result = ms.run()

    n_starts : int
        Number of random restarts (default 10).
    init_sampler : Callable or None
        Custom sampling function with signature
        ``(n_starts: int, params_0_flat: np.ndarray) -> np.ndarray``
        returning an array of shape ``(n_starts, n_params)``.
        Row 0 is always replaced with the original ``params_0_flat`` when
        ``include_initial=True``.  If ``None`` (default), uniform sampling
        around ``params_0`` is used.
    sample_scale : float
        Scale factor for the default uniform sampler.  The search window
        for each parameter is
        ``[p0 ± sample_scale * max(|p0|, 1)]`` (default 1.0).
    seed : int or None
        Random seed for reproducibility.
    include_initial : bool
        When ``True`` (default), the first start always uses the original
        ``params_0``, regardless of the sampler output.
    """

    def __init__(
        self,
        optimizable: Optimizable,
        optimizer_factory: Callable,
        n_starts: int = 10,
        init_sampler: Callable | None = None,
        sample_scale: float = 1.0,
        seed: int | None = None,
        include_initial: bool = True,
    ):
        self.optimizable = optimizable
        self.optimizer_factory = optimizer_factory
        self.n_starts = n_starts
        self.init_sampler = init_sampler
        self.sample_scale = sample_scale
        self.include_initial = include_initial
        self.rng = np.random.default_rng(seed)
        self._results: list[OptimizationResult] = []

    def _generate_starts(self) -> np.ndarray:
        """Return (n_starts, n_params) array of initial parameter vectors."""
        params_0 = np.array(self.optimizable.params_0_flat)

        if self.init_sampler is not None:
            starts = np.array(self.init_sampler(self.n_starts, params_0))
        else:
            starts = _uniform_sampler(
                self.n_starts,
                params_0,
                self.sample_scale,
                self.optimizable.bounds_flat,
                self.rng,
            )

        if self.include_initial:
            starts[0] = params_0

        return starts

    def run(self) -> MultiStartResult:
        """
        Execute all starts sequentially and return a :class:`MultiStartResult`.

        Each start clones the optimizable with a new ``params_0_flat``, calls
        ``optimizer_factory(clone)`` to get a fresh optimizer, and runs
        ``optimizer.optimize()``.  Failed starts (exceptions) are recorded as
        unsuccessful ``OptimizationResult`` entries with ``success=False``.

        Returns
        -------
        MultiStartResult
        """
        starts = self._generate_starts()
        results: list[OptimizationResult] = []

        for i, p0 in enumerate(starts):
            # Shallow-copy the optimizable and override its initial params.
            opt_clone = copy.copy(self.optimizable)
            opt_clone.params_0_flat = jnp.array(p0)

            optimizer = self.optimizer_factory(opt_clone)

            try:
                result = optimizer.optimize()
                if not isinstance(result, OptimizationResult):
                    # Wrap legacy plain-dict return for compatibility
                    result = OptimizationResult(params=dict(result))
            except Exception as exc:  # noqa: BLE001
                nan_params = {
                    k: float("nan")
                    for k in self.optimizable.unflatten_params(
                        jnp.array(p0)
                    )
                }
                result = OptimizationResult(
                    params=nan_params,
                    success=False,
                    message=f"Start {i} raised {type(exc).__name__}: {exc}",
                    final_loss=float("inf"),
                )

            results.append(result)

        self._results = results

        # Pick best: lowest finite final_loss among successful runs.
        successful = [
            (i, r)
            for i, r in enumerate(results)
            if r.success
            and r.final_loss is not None
            and np.isfinite(float(r.final_loss))
        ]

        if successful:
            best_idx, best = min(successful, key=lambda ir: float(ir[1].final_loss))
        else:
            # All failed — fall back to last result
            best_idx = len(results) - 1
            best = results[best_idx]

        n_successful = sum(1 for r in results if r.success)

        return MultiStartResult(
            results=results,
            best_result=best,
            best_start_index=best_idx,
            n_starts=self.n_starts,
            n_successful=n_successful,
        )

    @property
    def results(self) -> list[OptimizationResult]:
        """Results from the last :meth:`run` call (empty before first run)."""
        return self._results
