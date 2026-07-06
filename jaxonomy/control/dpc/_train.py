# SPDX-License-Identifier: MIT

"""Optax-driven training loop for DPC policies."""

from __future__ import annotations

from typing import Any, Callable, NamedTuple

import jax
import jax.numpy as jnp


class TrainResults(NamedTuple):
    """Return value of :func:`train_policy`.

    Attributes:
        params: Optimised policy parameters (same PyTree shape as the
            initial ``params``).
        loss_history: 1-D array of length ``n_iters`` with the loss at
            each iteration.
    """

    params: Any
    loss_history: jnp.ndarray


def train_policy(
    loss_fn: Callable[[Any, jnp.ndarray, jnp.ndarray], jnp.ndarray],
    params_init: Any,
    x0_batch: jnp.ndarray,
    ref_batch: jnp.ndarray,
    *,
    n_iters: int = 200,
    learning_rate: float = 1e-3,
    optimizer_name: str = "adam",
    verbose: bool = False,
) -> TrainResults:
    """Train a DPC policy by gradient descent on ``loss_fn`` with optax.

    Args:
        loss_fn: Output of :func:`dpc_loss` (or any callable with the
            same signature) — ``loss_fn(params, x0_batch, ref_batch)
            -> scalar``.
        params_init: Initial policy parameters (PyTree).
        x0_batch: Batch of initial states forwarded to ``loss_fn`` at
            every iteration. Use a fresh resample per epoch outside the
            loop if stochastic-DAgger-style training is desired.
        ref_batch: Batch of reference trajectories matching ``x0_batch``.
        n_iters: Number of gradient steps.
        learning_rate: Optax LR.
        optimizer_name: ``"adam"`` (default) or ``"sgd"`` — anything
            else is delegated to ``optax.<name>(learning_rate)``.
        verbose: Print loss every ``n_iters // 10`` steps when ``True``.

    Returns:
        :class:`TrainResults` with the optimised parameters and the
        per-iteration loss history.

    Notes:
        Implements a vanilla single-batch optax training loop. For
        stochastic DPC / receding-horizon variants where ``x0_batch``
        or ``ref_batch`` should resample per epoch, drive this function
        in your own outer loop with smaller ``n_iters`` per call.
    """
    try:
        import optax  # type: ignore
    except ImportError as e:
        raise ImportError(
            "jaxonomy.control.dpc.train_policy requires optax — install via "
            "`pip install optax`."
        ) from e

    opt_factory = getattr(optax, optimizer_name, None)
    if opt_factory is None:
        raise ValueError(
            f"train_policy: unknown optimizer {optimizer_name!r}; expected an "
            f"optax constructor name (e.g. 'adam', 'sgd', 'adamw')."
        )
    optimizer = opt_factory(learning_rate)
    opt_state = optimizer.init(params_init)

    grad_fn = jax.jit(jax.value_and_grad(loss_fn))

    @jax.jit
    def _step(params, opt_state, x0, ref):
        loss, grads = grad_fn(params, x0, ref)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss

    params = params_init
    loss_history = []
    for i in range(int(n_iters)):
        params, opt_state, loss = _step(params, opt_state, x0_batch, ref_batch)
        loss_history.append(float(loss))
        if verbose and (i == 0 or (i + 1) % max(1, n_iters // 10) == 0):
            print(f"[dpc.train_policy] iter {i + 1}/{n_iters}: loss = {float(loss):.4e}")

    return TrainResults(
        params=params,
        loss_history=jnp.asarray(loss_history),
    )
