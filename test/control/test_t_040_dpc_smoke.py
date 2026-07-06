# SPDX-License-Identifier: MIT

"""T-040 DPC scaffolding smoke test.

Trains a tiny linear policy ``u = K @ (x_ref - x)`` on a first-order plant
``dx/dt = -x + u`` to track a constant reference. Confirms that:

1. :class:`ClosedLoopRollout` produces a finite trajectory.
2. :func:`dpc_loss` composes stage + terminal + penalty terms without
   nan/inf.
3. :func:`train_policy` reduces the loss monotonically (within noise)
   over 200 optax-adam iterations.
4. The trained policy drives the plant toward the reference.

This is a *scaffolding* test — it exercises the public API surface
listed in T-040 but does not validate against the Neuromancer two-tank
acceptance benchmark (that lives in a downstream tutorial).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest


optax = pytest.importorskip("optax")

from jaxonomy.control.dpc import (
    ClosedLoopRollout,
    Penalty,
    dpc_loss,
    train_policy,
)


def _first_order_plant(time, x, u):
    """dx/dt = -x + u (1-D)."""
    return -x + u


def _linear_policy(params, x, ref):
    """u = K * (ref - x). ``params`` is the scalar gain K wrapped in a
    1-element array."""
    K = params["K"]
    return K * (ref - x)


def _build_rollout(horizon=20, dt=0.05):
    return ClosedLoopRollout(
        plant_ode=_first_order_plant,
        policy_fn=_linear_policy,
        horizon=horizon,
        dt=dt,
    )


def test_rollout_produces_finite_trajectory():
    rollout = _build_rollout()
    params = {"K": jnp.asarray(0.5)}
    x0 = jnp.zeros((4, 1))  # batch of 4, 1-D state
    ref = jnp.ones((4, 21, 1))
    res = rollout(params, x0, ref)
    assert res.x.shape == (4, 21, 1)
    assert res.u.shape == (4, 20, 1)
    assert jnp.all(jnp.isfinite(res.x))
    assert jnp.all(jnp.isfinite(res.u))


def test_dpc_loss_composes_cleanly():
    rollout = _build_rollout()

    def stage(x, u, ref):
        return jnp.mean((x - ref) ** 2) + 0.01 * jnp.mean(u ** 2)

    def terminal(x, ref):
        return jnp.mean((x - ref) ** 2)

    # Penalty: keep |u| <= 2.0.
    pen = Penalty(
        constraint_fn=lambda x, u, ref: jnp.abs(u) - 2.0,
        weight=0.1,
        mode="soft",
    )

    loss_fn = dpc_loss(rollout, stage, terminal, penalties=[pen])

    params = {"K": jnp.asarray(0.5)}
    x0 = jnp.zeros((2, 1))
    ref = jnp.ones((2, 21, 1))
    loss = loss_fn(params, x0, ref)
    assert jnp.isfinite(loss)


def test_train_policy_reduces_loss_and_tracks_reference():
    rollout = _build_rollout(horizon=40, dt=0.05)

    def stage(x, u, ref):
        return jnp.mean((x - ref) ** 2) + 0.01 * jnp.mean(u ** 2)

    def terminal(x, ref):
        return jnp.mean((x - ref) ** 2)

    loss_fn = dpc_loss(rollout, stage, terminal)

    params_init = {"K": jnp.asarray(0.0)}
    x0 = jnp.zeros((1, 1))
    ref = jnp.ones((1, 41, 1))

    results = train_policy(
        loss_fn,
        params_init,
        x0,
        ref,
        n_iters=200,
        learning_rate=1e-1,
    )

    losses = np.asarray(results.loss_history)
    # Last 20% of training should be substantially lower than the first 20%.
    n = len(losses)
    early = float(np.mean(losses[: max(1, n // 5)]))
    late = float(np.mean(losses[-max(1, n // 5) :]))
    assert late < early, (
        f"training loss did not decrease: early={early:.4f} late={late:.4f}"
    )

    # The trained policy should drive x materially toward the reference (1.0).
    # The proportional policy ``u = K(ref - x)`` on ``dx/dt = -x + u`` has a
    # steady-state error of ``1/(1+K)`` so even an optimal K leaves some
    # residual — anything above ~0.5 demonstrates the policy is learning.
    final_traj = rollout(results.params, x0, ref)
    final_x = float(final_traj.x[0, -1, 0])
    assert final_x > 0.5, (
        f"trained policy did not move x toward reference; final x = {final_x:.3f}"
    )


def test_penalty_modes_validate():
    Penalty(constraint_fn=lambda x, u, ref: u, weight=1.0, mode="soft")
    Penalty(constraint_fn=lambda x, u, ref: u, weight=1.0, mode="barrier")
    with pytest.raises(ValueError, match="mode"):
        Penalty(constraint_fn=lambda x, u, ref: u, weight=1.0, mode="bogus")
    with pytest.raises(ValueError, match="weight"):
        Penalty(constraint_fn=lambda x, u, ref: u, weight=-1.0)
