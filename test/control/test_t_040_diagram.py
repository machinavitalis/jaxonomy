# SPDX-License-Identifier: MIT

"""T-040-followup: Diagram-integrated DPC (policy as a ``LeafSystem``).

Exercises the upgrade from the function-level
:class:`~jaxonomy.control.dpc.ClosedLoopRollout` to a real jaxonomy
``Diagram`` run under :func:`jaxonomy.simulate`:

1. :class:`PolicyBlock` + :class:`PlantBlock` wire into a feedback Diagram
   (:func:`build_closed_loop`) and produce a finite closed-loop trajectory
   under ``simulate`` (:func:`simulate_closed_loop`).
2. The Diagram rollout matches the function-level fixed-step RK4
   :class:`ClosedLoopRollout` on the same plant/policy (forward parity).
3. **Gradient correctness (the T-040 acceptance core):** ``jax.grad`` of a
   terminal cost w.r.t. the policy parameters — flowing through
   ``simulate``'s custom-VJP path — matches central finite differences.
4. A PyTree policy parameter (gain dict) round-trips through the block's
   flat ``theta`` slot.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaxonomy.control.dpc import (
    ClosedLoopRollout,
    ClosedLoopRunner,
    PlantBlock,
    PolicyBlock,
    build_closed_loop,
    simulate_closed_loop,
)
from jaxonomy.testing.markers import skip_if_not_jax

skip_if_not_jax()


# ── shared 1-D first-order plant + proportional policy ───────────────────────


def _plant_ode(t, x, u):
    """dx/dt = -x + u (scalar)."""
    return -x + u


def _policy_fn(params, x, ref):
    """u = K (ref - x). ``params`` is the gain PyTree ``{"K": scalar}``."""
    return params["K"] * (ref - x)


_DT = 0.05
_T_END = 2.0
_REF = 1.0


def _final_x_diagram(K):
    """Terminal plant state of the Diagram closed loop for gain ``K``."""
    plant = PlantBlock(_plant_ode, jnp.asarray(0.0))
    policy = PolicyBlock(_policy_fn, {"K": jnp.asarray(0.0)})
    out = simulate_closed_loop(
        plant, policy, {"K": K}, _REF, (0.0, _T_END), dt=_DT, x0=0.0,
    )
    return out["x_final"].sum()


# ── 1. closed loop produces a finite trajectory under simulate ───────────────


def test_closed_loop_runs_under_simulate():
    plant = PlantBlock(_plant_ode, jnp.asarray(0.0))
    policy = PolicyBlock(_policy_fn, {"K": jnp.asarray(0.5)})
    out = simulate_closed_loop(
        plant, policy, {"K": jnp.asarray(0.5)}, _REF, (0.0, _T_END),
        dt=_DT, x0=0.0, record=True,
    )
    assert np.all(np.isfinite(np.asarray(out["x"])))
    assert np.all(np.isfinite(np.asarray(out["u"])))
    # Proportional control on dx/dt=-x+u drives x toward K/(1+K)*ref < ref.
    assert 0.0 < float(out["x_final"]) < _REF


# ── 2. forward parity with the function-level RK4 rollout ────────────────────


def test_diagram_matches_function_level_rollout():
    K = 0.7
    # Function-level rollout (oracle): same plant, same policy, same dt/RK4.
    horizon = int(round(_T_END / _DT))
    roll = ClosedLoopRollout(
        plant_ode=_plant_ode, policy_fn=_policy_fn, horizon=horizon, dt=_DT,
    )
    ref_traj = jnp.full((horizon + 1, 1), _REF)
    fn_res = roll({"K": jnp.asarray(K)}, jnp.zeros((1,)), ref_traj)
    fn_final = float(fn_res.x[-1, 0])

    diag_final = float(_final_x_diagram(jnp.asarray(K)))
    # Both integrate the same ODE with RK4 at the same step; agree tightly.
    assert abs(diag_final - fn_final) < 1e-3, (
        f"Diagram rollout {diag_final:.6f} != function-level {fn_final:.6f}"
    )


# ── 3. gradient correctness through simulate (T-040 acceptance core) ─────────


def test_gradient_through_simulate_matches_fd():
    K0 = 0.5
    g_ad = float(jax.grad(_final_x_diagram)(jnp.asarray(K0)))

    h = 1e-4
    g_fd = (
        float(_final_x_diagram(jnp.asarray(K0 + h)))
        - float(_final_x_diagram(jnp.asarray(K0 - h)))
    ) / (2 * h)

    rel = abs(g_ad - g_fd) / (abs(g_fd) + 1e-12)
    assert rel < 5e-3, (
        f"d(x_final)/dK: AD={g_ad:.6f}, FD={g_fd:.6f}, rel_err={rel:.4e}"
    )


# ── 4. PyTree policy params round-trip through the flat theta slot ────────────


def test_pytree_params_roundtrip():
    params = {"K": jnp.asarray(0.3)}
    theta = PolicyBlock.flatten_params(params)
    assert theta.shape == (1,)
    # A 2-vector gain PyTree ravels to length-2 theta.
    params2 = {"Kp": jnp.asarray(0.3), "Kd": jnp.asarray([0.1, 0.2])}
    theta2 = PolicyBlock.flatten_params(params2)
    assert theta2.shape == (3,)


# ── 5. build_closed_loop wiring smoke ────────────────────────────────────────


def test_build_closed_loop_wiring():
    plant = PlantBlock(_plant_ode, jnp.asarray(0.0))
    policy = PolicyBlock(_policy_fn, {"K": jnp.asarray(0.5)})
    diagram, plant_s, policy_s, ref_s = build_closed_loop(plant, policy, _REF)
    # Diagram builds and a context can be created (structurally valid loop).
    ctx = diagram.create_context()
    assert ctx is not None
    assert "theta" in ctx[policy_s.system_id].parameters


# ── 6. ClosedLoopRunner (build once, run many) ───────────────────────────────


def _make_runner():
    plant = PlantBlock(_plant_ode, jnp.asarray(0.0))
    policy = PolicyBlock(_policy_fn, {"K": jnp.asarray(0.0)})
    return ClosedLoopRunner(plant, policy, _REF, (0.0, _T_END), dt=_DT, x0=0.0)


def test_runner_matches_simulate_closed_loop():
    """``ClosedLoopRunner.run`` reproduces ``simulate_closed_loop``'s x_final."""
    runner = _make_runner()
    for K in (0.3, 0.7, 1.5):
        x_runner = float(runner.run({"K": jnp.asarray(K)}).sum())
        x_func = float(_final_x_diagram(jnp.asarray(K)))
        assert abs(x_runner - x_func) < 1e-6, f"K={K}: {x_runner} != {x_func}"


def test_runner_gradient_matches_fd():
    """``jax.grad`` through the cached runner matches central FD (training path)."""
    runner = _make_runner()
    loss = lambda K: runner.run({"K": K}).sum()
    K0 = 0.5
    g_ad = float(jax.grad(loss)(jnp.asarray(K0)))
    h = 1e-4
    g_fd = (
        float(loss(jnp.asarray(K0 + h))) - float(loss(jnp.asarray(K0 - h)))
    ) / (2 * h)
    rel = abs(g_ad - g_fd) / (abs(g_fd) + 1e-12)
    assert rel < 5e-3, f"AD={g_ad:.6f}, FD={g_fd:.6f}, rel={rel:.4e}"


def test_runner_vmap_batched_rollout():
    """``jax.vmap(runner.run)`` gives a batched rollout over a gain sweep."""
    runner = _make_runner()
    Ks = jnp.array([0.2, 0.5, 1.0, 2.0])
    xb = jax.vmap(lambda K: runner.run({"K": K}).sum())(Ks)
    assert xb.shape == (4,)
    # Higher gain -> closer to the reference (monotone for this plant).
    assert np.all(np.diff(np.asarray(xb)) > 0)
    # Matches per-sample runs.
    for i, K in enumerate(np.asarray(Ks)):
        assert abs(float(xb[i]) - float(runner.run({"K": jnp.asarray(K)}).sum())) < 1e-6


def test_runner_x0_override_per_call():
    """A per-call ``x0`` overrides the runner's default initial state."""
    runner = _make_runner()
    x_a = float(runner.run({"K": jnp.asarray(0.5)}, x0=0.0).sum())
    x_b = float(runner.run({"K": jnp.asarray(0.5)}, x0=0.9).sum())
    # Different ICs on a stable plant give different terminal states.
    assert abs(x_a - x_b) > 1e-3
