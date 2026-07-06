# SPDX-License-Identifier: MIT

"""Cost composition for DPC: stage + terminal + penalties."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

import jax.numpy as jnp


@dataclass(frozen=True)
class Penalty:
    """A constraint expressed as a callable + a flavour (soft / barrier).

    DPC constraints (state limits, control saturation, "stay above zero"
    requirements) are added to the loss as differentiable surrogates so
    the policy training can find solutions that satisfy them rather than
    relying on a hard projection step. Two surrogates ship out of the
    box:

    - ``mode="soft"`` (default): a one-sided quadratic penalty
      ``weight * max(0, violation) ** 2``. Zero gradient at the boundary,
      so the policy learns to back away from the constraint smoothly.
    - ``mode="barrier"``: a log-barrier
      ``-weight * log(-violation)``. Diverges as the boundary is
      approached from the feasible side; requires the initial trajectory
      to be strictly feasible.

    The constraint convention is ``violation > 0`` means "violated" — i.e.
    a state limit ``x <= x_max`` is encoded as
    ``constraint_fn(x, u, ref) = x - x_max``.

    Args:
        constraint_fn: Callable ``(x, u, ref) -> array``. Each element
            of the returned array is one scalar constraint; ``> 0`` means
            violated.
        weight: Multiplicative weight on the surrogate. Higher weights
            push harder against the constraint at the cost of harder
            optimisation.
        mode: ``"soft"`` (default) or ``"barrier"``.
        name: Optional label for diagnostics.
    """

    constraint_fn: Callable[[jnp.ndarray, jnp.ndarray, jnp.ndarray], jnp.ndarray]
    weight: float = 1.0
    mode: str = "soft"
    name: str = ""

    def __post_init__(self):
        if self.mode not in ("soft", "barrier"):
            raise ValueError(
                f"Penalty.mode must be 'soft' or 'barrier', got {self.mode!r}"
            )
        if self.weight < 0:
            raise ValueError(f"Penalty.weight must be >= 0, got {self.weight}")

    def evaluate(
        self,
        x: jnp.ndarray,
        u: jnp.ndarray,
        ref: jnp.ndarray,
    ) -> jnp.ndarray:
        """Evaluate the penalty surrogate; returns a scalar."""
        violation = self.constraint_fn(x, u, ref)
        if self.mode == "soft":
            return self.weight * jnp.sum(jnp.maximum(violation, 0.0) ** 2)
        # barrier: log-barrier on the feasible side. Use a small epsilon
        # to keep the gradient finite on numerical violations.
        eps = 1e-9
        feasible = -violation - eps
        # If the constraint is currently violated (feasible <= 0), fall
        # back to a soft-quadratic so the gradient still points outward.
        soft_fallback = self.weight * jnp.sum(jnp.maximum(violation + eps, 0.0) ** 2)
        barrier_value = -self.weight * jnp.sum(jnp.log(jnp.maximum(feasible, eps)))
        return jnp.where(jnp.any(feasible <= 0.0), soft_fallback, barrier_value)


def dpc_loss(
    rollout,
    stage_cost: Callable[[jnp.ndarray, jnp.ndarray, jnp.ndarray], jnp.ndarray],
    terminal_cost: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray] | None = None,
    penalties: Sequence[Penalty] = (),
) -> Callable[[Any, jnp.ndarray, jnp.ndarray], jnp.ndarray]:
    """Compose a DPC objective from a rollout + stage / terminal / penalty terms.

    Returns a callable ``loss_fn(params, x0_batch, ref_batch) -> scalar``
    suitable for ``jax.grad`` / ``optax`` minimisation.

    Args:
        rollout: A :class:`ClosedLoopRollout` instance (or any callable
            with the same ``(params, x0, ref) -> RolloutResults`` signature).
        stage_cost: ``stage_cost(x_k, u_k, ref_k) -> scalar`` evaluated
            at every step ``k = 0..horizon-1`` and summed.
        terminal_cost: Optional ``terminal_cost(x_terminal, ref_terminal)
            -> scalar`` added once at the trajectory end. ``None``
            (default) skips it.
        penalties: Sequence of :class:`Penalty` instances, each
            accumulated over every step (including terminal).

    Returns:
        A scalar loss function suitable for ``jax.grad`` over ``params``.
    """

    def loss_fn(params, x0, ref):
        results = rollout(params, x0, ref)
        x = results.x  # shape (B, H+1, n_x)
        u = results.u  # shape (B, H, n_u)
        r = results.ref  # shape (B, H+1, ...)

        # Sum stage cost across all (batch, step) pairs.
        per_step = jnp.sum(
            jnp.stack(
                [
                    stage_cost(x[:, k, :], u[:, k, :], r[:, k, :])
                    for k in range(u.shape[1])
                ]
            )
        )
        total = per_step
        if terminal_cost is not None:
            total = total + terminal_cost(x[:, -1, :], r[:, -1, :])
        for pen in penalties:
            # Accumulate penalty across all (batch, step) pairs.
            pen_total = jnp.asarray(0.0)
            for k in range(u.shape[1]):
                pen_total = pen_total + pen.evaluate(
                    x[:, k, :], u[:, k, :], r[:, k, :]
                )
            # Terminal-step penalty too (no control at the final state).
            pen_total = pen_total + pen.evaluate(
                x[:, -1, :], jnp.zeros_like(u[:, 0, :]), r[:, -1, :]
            )
            total = total + pen_total
        # Average over the batch so the gradient magnitude doesn't scale
        # with batch size.
        return total / x.shape[0]

    return loss_fn
