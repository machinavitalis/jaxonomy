# SPDX-License-Identifier: MIT

"""Closed-loop rollout for differentiable predictive control.

The rollout is a fixed-step RK4 integration of the plant under a
parameterised policy. Implemented as a pure-functional ``jax.lax.scan``
so that ``jax.grad`` flows cleanly through the trajectory back to the
policy parameters.
"""

from __future__ import annotations

from typing import Any, Callable, NamedTuple

import jax
import jax.numpy as jnp


class RolloutResults(NamedTuple):
    """Container for the trajectory returned by :class:`ClosedLoopRollout`.

    Attributes:
        time: ``(horizon + 1,)`` time grid (starts at 0.0, step = ``dt``).
        x: ``(batch_dim, horizon + 1, n_x)`` state trajectory.
        u: ``(batch_dim, horizon, n_u)`` control trajectory (one fewer
            entry than ``x`` because the last state has no following
            control).
        ref: ``(batch_dim, horizon + 1, ...)`` reference trajectory
            (broadcast or reshaped from the user input).
    """

    time: jnp.ndarray
    x: jnp.ndarray
    u: jnp.ndarray
    ref: jnp.ndarray


def _rk4_step(plant_ode, t, x, u, dt):
    """One RK4 step of ``dx/dt = plant_ode(t, x, u)``."""
    k1 = plant_ode(t, x, u)
    k2 = plant_ode(t + 0.5 * dt, x + 0.5 * dt * k1, u)
    k3 = plant_ode(t + 0.5 * dt, x + 0.5 * dt * k2, u)
    k4 = plant_ode(t + dt, x + dt * k3, u)
    return x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


class ClosedLoopRollout:
    """Roll out a plant under a parameterised policy for ``horizon`` steps.

    The rollout is the standard MPC pattern:

    .. code-block:: text

        for k in range(horizon):
            u_k = policy(params, x_k, ref_k)
            x_{k+1} = plant(x_k, u_k)

    Both ``plant_ode`` and ``policy_fn`` must be JAX-traceable pure
    functions so the whole rollout is differentiable through the policy
    parameters.

    Args:
        plant_ode: Callable ``plant_ode(time, x, u) -> dx/dt``. Should
            be vmappable on the leading batch axis of ``x`` and ``u``.
        policy_fn: Callable ``policy_fn(params, x, ref) -> u``. Vectorised
            over the leading batch axis of ``x`` and ``ref``.
        horizon: Number of RK4 steps to take. The returned ``x``
            trajectory has length ``horizon + 1``; ``u`` has length
            ``horizon``.
        dt: Fixed time step in seconds.

    Notes:
        Fixed-step RK4 (rather than an adaptive solver) is the standard
        DPC choice: adaptive step counts vary across batch elements,
        which breaks vmap, and the policy-training loop benefits more
        from deterministic compute per iteration than from local-error
        adaptivity. For systems with stiff dynamics, decrease ``dt``
        rather than introducing an adaptive integrator.
    """

    def __init__(
        self,
        plant_ode: Callable[[Any, jnp.ndarray, jnp.ndarray], jnp.ndarray],
        policy_fn: Callable[[Any, jnp.ndarray, jnp.ndarray], jnp.ndarray],
        horizon: int,
        dt: float,
    ):
        if horizon <= 0:
            raise ValueError(f"horizon must be > 0, got {horizon}")
        if dt <= 0:
            raise ValueError(f"dt must be > 0, got {dt}")
        self.plant_ode = plant_ode
        self.policy_fn = policy_fn
        self.horizon = int(horizon)
        self.dt = float(dt)

    def __call__(
        self,
        params: Any,
        x0: jnp.ndarray,
        ref: jnp.ndarray,
    ) -> RolloutResults:
        """Run the rollout.

        Args:
            params: Policy parameters (PyTree).
            x0: Initial state, shape ``(batch_dim, n_x)`` or ``(n_x,)``.
                A single-state input is promoted to a length-1 batch.
            ref: Reference trajectory, shape
                ``(batch_dim, horizon + 1, ...)``. Must be broadcastable
                against ``x0``'s batch dim.

        Returns:
            :class:`RolloutResults`.
        """
        x0 = jnp.asarray(x0)
        ref = jnp.asarray(ref)
        if x0.ndim == 1:
            x0 = x0[None, :]
            squeeze_batch = True
        else:
            squeeze_batch = False
        if ref.ndim == 2:
            ref = ref[None, :, :]

        if ref.shape[1] != self.horizon + 1:
            raise ValueError(
                f"ref.shape[1] must equal horizon+1={self.horizon + 1}, "
                f"got {ref.shape[1]}"
            )

        horizon = self.horizon
        dt = self.dt
        plant_ode = self.plant_ode
        policy_fn = self.policy_fn

        def _step(carry, scan_input):
            x, t = carry
            k_idx, ref_k = scan_input
            u = policy_fn(params, x, ref_k)
            x_next = _rk4_step(plant_ode, t, x, u, dt)
            return (x_next, t + dt), (x, u)

        # Per-batch scan with vmap over the batch leading axis.
        def _scan_one(x0_i, ref_i):
            time_grid = jnp.arange(horizon + 1, dtype=jnp.float32) * dt
            ks = jnp.arange(horizon)
            (x_final, _), (xs, us) = jax.lax.scan(
                _step,
                (x0_i, jnp.asarray(0.0, dtype=time_grid.dtype)),
                (ks, ref_i[:horizon]),
            )
            # Append the final state to ``xs`` to match the "(horizon+1, n_x)" shape.
            x_traj = jnp.concatenate([xs, x_final[None, :]], axis=0)
            return time_grid, x_traj, us

        time_grid, x_traj, u_traj = jax.vmap(_scan_one)(x0, ref)
        # ``time_grid`` is identical across batch elements; collapse.
        time_grid = time_grid[0]

        if squeeze_batch:
            x_traj = x_traj[0]
            u_traj = u_traj[0]
            ref = ref[0]

        return RolloutResults(time=time_grid, x=x_traj, u=u_traj, ref=ref)
