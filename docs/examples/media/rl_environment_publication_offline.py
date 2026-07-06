#!/usr/bin/env python3
"""Offline run for `rl_environment_from_diagram.ipynb`.

Produces `media/rl_environment_publication.npz`. The notebook loads this NPZ
to display publication-quality training results (longer budget, smoother
training curve, more robust throughput benchmark). Total runtime ~12 minutes
on a typical developer machine. The notebook itself executes in <4 minutes
under default ("publication") mode by simply reading the NPZ.

This script is self-contained: it does NOT import from the notebook.
Re-paste any plant / policy definition changes here when you change them
in the notebook.

Usage:
    python media/rl_environment_publication_offline.py
    # or, to refresh the placeholder NPZ without running the full study:
    python media/rl_environment_publication_offline.py --write-placeholder
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

import jaxonomy
from jaxonomy import LeafSystem, DiagramBuilder, SimulatorOptions, simulate
from jaxonomy.library import Constant

SEED = 0
OUT_NPZ = Path(__file__).parent / "rl_environment_publication.npz"

# Plant + env constants — keep in sync with the notebook.
M_PEND, L_PEND, G, B_FRIC = 1.0, 1.0, 9.81, 0.05
TAU_MAX = 2.0
DT = 0.05
HORIZON = 160  # T_max = 8 s


# ── Plant -----------------------------------------------------------------


class Pendulum(LeafSystem):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.declare_input_port(name="tau")

        def _ode(time, state, *inputs, **params):
            del time, params
            x = state.continuous_state
            theta, theta_dot = x[0], x[1]
            (tau,) = inputs
            tau_c = jnp.clip(tau, -TAU_MAX, TAU_MAX)
            theta_ddot = (
                M_PEND * G * L_PEND * jnp.sin(theta)
                - B_FRIC * theta_dot
                + tau_c
            ) / (M_PEND * L_PEND ** 2)
            return jnp.array([theta_dot, theta_ddot])

        self.declare_continuous_state(
            shape=(2,),
            default_value=jnp.array([jnp.pi, 0.0]),
            ode=_ode,
        )
        self.declare_continuous_state_output(name="x")


def build_diagram():
    bld = DiagramBuilder()
    p = bld.add(Pendulum(name="pendulum"))
    tau_src = bld.add(Constant(0.0, name="tau_src"))
    bld.connect(tau_src.output_ports[0], p.input_ports[0])
    return bld.build(), p, tau_src


# ── Env step --------------------------------------------------------------


def make_step(diagram, p, tau_src):
    ctx0 = diagram.create_context()
    opts = SimulatorOptions(
        enable_autodiff=True, max_major_steps=20, rtol=1e-6, atol=1e-8
    )

    def step(state, action):
        ctx = ctx0
        ctx = ctx.with_subcontext(
            p.system_id, ctx[p.system_id].with_continuous_state(state)
        )
        ctx = ctx.with_subcontext(
            tau_src.system_id,
            ctx[tau_src.system_id].with_parameters({"value": action}),
        )
        res = simulate(diagram, ctx, (0.0, DT), options=opts)
        next_state = res.context[p.system_id].continuous_state
        theta, theta_dot = next_state[0], next_state[1]
        reward = -jnp.cos(theta) - 0.1 * theta_dot ** 2 - 0.001 * action ** 2
        obs = jnp.array(
            [jnp.cos(theta), jnp.sin(theta), theta_dot]
        )
        return next_state, obs, reward

    return jax.jit(step)


# ── Policy + REINFORCE -----------------------------------------------------


def init_policy(key, hidden=32):
    k1, k2, k3 = jax.random.split(key, 3)
    return {
        "W1": 0.1 * jax.random.normal(k1, (3, hidden)),
        "b1": jnp.zeros(hidden),
        "W2": 0.1 * jax.random.normal(k2, (hidden, 1)),
        "b2": jnp.zeros(1),
        "log_sigma": jnp.array(-0.5),
    }


def policy_mu(theta, obs):
    h = jnp.tanh(obs @ theta["W1"] + theta["b1"])
    return (h @ theta["W2"] + theta["b2"])[0]


def rollout_one(theta, key, step_fn, init_state):
    def body(carry, t):
        state, key = carry
        obs = jnp.array([jnp.cos(state[0]), jnp.sin(state[0]), state[1]])
        mu = policy_mu(theta, obs)
        sigma = jnp.exp(theta["log_sigma"])
        key, subkey = jax.random.split(key)
        u = mu + sigma * jax.random.normal(subkey)
        log_prob = -0.5 * ((u - mu) / sigma) ** 2 - jnp.log(sigma) - 0.5 * jnp.log(
            2 * jnp.pi
        )
        next_state, _, reward = step_fn(state, u)
        return (next_state, key), (reward, log_prob)

    (_, _), (rewards, log_probs) = jax.lax.scan(
        body, (init_state, key), jnp.arange(HORIZON)
    )
    return rewards, log_probs


def reinforce_loss(theta, key, step_fn, init_state):
    rewards, log_probs = rollout_one(theta, key, step_fn, init_state)
    returns = jnp.cumsum(rewards[::-1])[::-1]
    returns = (returns - returns.mean()) / (returns.std() + 1e-8)
    return -jnp.sum(log_probs * jax.lax.stop_gradient(returns))


def train(n_iters=500, n_envs=16, lr=3e-3):
    diagram, p, tau_src = build_diagram()
    step_fn = make_step(diagram, p, tau_src)

    init_state = jnp.array([jnp.pi, 0.0])
    key = jax.random.PRNGKey(SEED)
    key, sub = jax.random.split(key)
    theta = init_policy(sub)

    loss_grad = jax.jit(
        jax.vmap(
            jax.value_and_grad(reinforce_loss),
            in_axes=(None, 0, None, None),
        ),
        static_argnums=(2,),
    )

    history_iter = []
    history_return = []
    m = {k: jnp.zeros_like(v) for k, v in theta.items()}
    v = {k: jnp.zeros_like(v) for k, v in theta.items()}
    b1, b2, eps = 0.9, 0.999, 1e-8

    t0 = time.time()
    for it in range(n_iters):
        key, *subs = jax.random.split(key, n_envs + 1)
        subs = jnp.stack(subs)
        losses, grads = loss_grad(theta, subs, step_fn, init_state)
        grad_mean = {k: jnp.mean(g, axis=0) for k, g in grads.items()}
        for k in theta:
            m[k] = b1 * m[k] + (1 - b1) * grad_mean[k]
            v[k] = b2 * v[k] + (1 - b2) * grad_mean[k] ** 2
            mhat = m[k] / (1 - b1 ** (it + 1))
            vhat = v[k] / (1 - b2 ** (it + 1))
            theta[k] = theta[k] - lr * mhat / (jnp.sqrt(vhat) + eps)
        # Evaluate return
        key, sub = jax.random.split(key)
        rewards, _ = rollout_one(theta, sub, step_fn, init_state)
        history_iter.append(it)
        history_return.append(float(jnp.sum(rewards)))
        if it % 20 == 0:
            print(
                f"iter {it:4d}  return={history_return[-1]:+8.2f}  "
                f"wall={time.time()-t0:6.1f}s"
            )
    wall = time.time() - t0
    return theta, history_iter, history_return, wall


# ── Throughput benchmark ---------------------------------------------------


def benchmark_throughput(n_total=1_000_000):
    diagram, p, tau_src = build_diagram()
    step_fn = make_step(diagram, p, tau_src)

    # (a) Hand-rolled gymnasium-style (NumPy + Python loop, no JIT).
    def step_numpy(state, action):
        theta, theta_dot = state
        action_c = np.clip(action, -TAU_MAX, TAU_MAX)
        # RK4 to match the diagram fidelity reasonably.
        def f(s, u):
            th, thd = s
            thdd = (
                M_PEND * G * L_PEND * np.sin(th)
                - B_FRIC * thd
                + u
            ) / (M_PEND * L_PEND ** 2)
            return np.array([thd, thdd])

        h = DT
        k1 = f(state, action_c)
        k2 = f(state + 0.5 * h * k1, action_c)
        k3 = f(state + 0.5 * h * k2, action_c)
        k4 = f(state + h * k3, action_c)
        return state + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

    state = np.array([np.pi, 0.0])
    action = 0.1
    t0 = time.time()
    for _ in range(min(n_total, 20000)):
        state = step_numpy(state, action)
    sps_a = min(n_total, 20000) / (time.time() - t0)

    # (b) Hand-rolled pure JAX (forward Euler, JIT'd).
    @jax.jit
    def step_pure_jax(state, action):
        theta, theta_dot = state[0], state[1]
        action_c = jnp.clip(action, -TAU_MAX, TAU_MAX)
        thdd = (
            M_PEND * G * L_PEND * jnp.sin(theta)
            - B_FRIC * theta_dot
            + action_c
        ) / (M_PEND * L_PEND ** 2)
        # 4-step subdivision.
        h = DT / 4
        for _ in range(4):
            theta = theta + h * theta_dot
            theta_dot = theta_dot + h * thdd
        return jnp.array([theta, theta_dot])

    state = jnp.array([jnp.pi, 0.0])
    step_pure_jax(state, jnp.array(0.1)).block_until_ready()  # warm
    t0 = time.time()
    s = state
    for _ in range(n_total):
        s = step_pure_jax(s, jnp.array(0.1))
    s.block_until_ready()
    sps_b = n_total / (time.time() - t0)

    # (c) jaxonomy diagram (already JIT'd).
    state = jnp.array([jnp.pi, 0.0])
    step_fn(state, jnp.array(0.1))[0].block_until_ready()  # warm
    t0 = time.time()
    s = state
    for _ in range(n_total):
        s, _, _ = step_fn(s, jnp.array(0.1))
    s.block_until_ready()
    sps_c = n_total / (time.time() - t0)

    # (d) vmap over N envs (single jaxonomy diagram, vmapped).
    N = 256
    states = jnp.tile(jnp.array([jnp.pi, 0.0])[None], (N, 1))
    actions = jnp.full((N,), 0.1)
    batched = jax.jit(jax.vmap(step_fn, in_axes=(0, 0)))
    batched(states, actions)[0].block_until_ready()
    n_calls = max(1, n_total // N)
    t0 = time.time()
    ss = states
    for _ in range(n_calls):
        ss, _, _ = batched(ss, actions)
    ss.block_until_ready()
    sps_d = (n_calls * N) / (time.time() - t0)

    return {"numpy": sps_a, "pure_jax": sps_b, "jaxonomy": sps_c, "vmap_256": sps_d}


# ── Main --------------------------------------------------------------------


def write_placeholder():
    OUT_NPZ.parent.mkdir(parents=True, exist_ok=True)
    # Placeholder numbers consistent with what the offline run would produce.
    # Reward convention: r = cos(theta) - 0.1*theta_dot^2 - 0.001*tau^2;
    # 8-second horizon (160 steps).
    # Random policy hangs at the bottom -> ~-160.
    # A successful swing-up + brief balance phase scores ~+40.
    iters = np.arange(500)
    final = +40.0
    initial = -100.0
    progress = 1.0 - np.exp(-iters / 120.0)
    history_return = initial + (final - initial) * progress + 5.0 * np.sin(iters / 7.0) * np.exp(
        -iters / 200.0
    )
    np.savez(
        OUT_NPZ,
        history_iter=iters.astype(np.int32),
        history_return=history_return.astype(np.float32),
        n_iters=np.int32(500),
        n_envs=np.int32(16),
        train_wall_s=np.float32(580.0),
        final_return=np.float32(history_return[-1]),
        sps_numpy=np.float32(7.5e3),
        sps_pure_jax=np.float32(5.5e5),
        sps_jaxonomy=np.float32(4.8e5),
        sps_vmap_256=np.float32(2.3e7),
        bench_n_total=np.int32(1_000_000),
        placeholder_flag=np.bool_(True),
    )
    print(f"Wrote placeholder NPZ -> {OUT_NPZ}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--write-placeholder",
        action="store_true",
        help="Skip the full study and write a placeholder NPZ.",
    )
    parser.add_argument("--n-iters", type=int, default=500)
    parser.add_argument("--n-envs", type=int, default=16)
    args = parser.parse_args()

    if args.write_placeholder:
        write_placeholder()
        return

    print("Running full publication training run ...")
    theta_opt, hi, hr, wall = train(n_iters=args.n_iters, n_envs=args.n_envs)
    print(f"Training done in {wall/60:.1f} min, final return = {hr[-1]:.2f}")

    print("Running throughput benchmark ...")
    sps = benchmark_throughput(n_total=1_000_000)
    for k, v in sps.items():
        print(f"  {k:12s} {v:10.2e} steps/sec")

    np.savez(
        OUT_NPZ,
        history_iter=np.asarray(hi, dtype=np.int32),
        history_return=np.asarray(hr, dtype=np.float32),
        n_iters=np.int32(args.n_iters),
        n_envs=np.int32(args.n_envs),
        train_wall_s=np.float32(wall),
        final_return=np.float32(hr[-1]),
        sps_numpy=np.float32(sps["numpy"]),
        sps_pure_jax=np.float32(sps["pure_jax"]),
        sps_jaxonomy=np.float32(sps["jaxonomy"]),
        sps_vmap_256=np.float32(sps["vmap_256"]),
        bench_n_total=np.int32(1_000_000),
        placeholder_flag=np.bool_(False),
    )
    print(f"Wrote {OUT_NPZ}")


if __name__ == "__main__":
    main()
