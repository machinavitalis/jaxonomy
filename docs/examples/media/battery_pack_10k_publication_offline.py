#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Offline publication run for ``battery_pack_10k_scaling.ipynb``.

Runs the full scaling sweep up to N=100_000 cells and writes
``media/battery_pack_10k_publication.npz``.  The notebook ingests this
NPZ; the in-notebook live cells re-derive the small-N rows so the reader
can verify the architecture.

Runtime on a developer laptop (Apple M2, 32 GB RAM):

    N=     8     wall ~0.0003 s   RSS ~  235 MB
    N=    32     wall ~0.0004 s   RSS ~  253 MB
    N=   128     wall ~0.0008 s   RSS ~  273 MB
    N=   512     wall ~0.0026 s   RSS ~  291 MB
    N=  2048     wall ~0.012  s   RSS ~  326 MB
    N= 10000     wall ~0.044  s   RSS ~  415 MB
    N= 50000     wall ~0.14   s   RSS ~  795 MB
    N=100000     wall ~0.31   s   RSS ~ 1500 MB

The same kernel JITs once per N (shape-keyed); subsequent calls reuse it.
Total wall time including JIT compile: ~30 s.
"""
from __future__ import annotations

import gc
import os
import time

import jax
import jax.numpy as jnp
import numpy as np
import psutil

# ----------------------------------------------------------------------
# Pack model (copied verbatim from the notebook so the offline script is
# self-contained — by design).
# ----------------------------------------------------------------------

OCV_BREAKS = jnp.array([0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0])
OCV_VALS = jnp.array([3.0, 3.45, 3.6, 3.7, 3.85, 4.05, 4.20])
R_NEIGH = 2.0  # K/W between adjacent cells (1-D chain)


def pack_rhs(state, current, params, T_amb):
    SOC = state[:, 0]
    V_RC = state[:, 1]
    T = state[:, 2]
    R0 = params[:, 0]
    R1 = params[:, 1]
    C1 = params[:, 2]
    capAh = params[:, 3]
    R_cool = params[:, 4]
    C_th = params[:, 5]
    Ah_to_As = 3600.0
    dSOC = current / (Ah_to_As * capAh)
    dV_RC = current / C1 - V_RC / (R1 * C1)
    P_heat = current ** 2 * R0 + V_RC ** 2 / R1
    Q_amb = (T_amb - T) / R_cool
    T_left = jnp.concatenate([T[:1], T[:-1]])
    T_right = jnp.concatenate([T[1:], T[-1:]])
    Q_neigh = (T_left + T_right - 2 * T) / R_NEIGH
    dT = (P_heat + Q_amb + Q_neigh) / C_th
    return jnp.stack([dSOC, dV_RC, dT], axis=-1)


def rk4(state, current, params, T_amb, dt):
    k1 = pack_rhs(state, current, params, T_amb)
    k2 = pack_rhs(state + 0.5 * dt * k1, current, params, T_amb)
    k3 = pack_rhs(state + 0.5 * dt * k2, current, params, T_amb)
    k4 = pack_rhs(state + dt * k3, current, params, T_amb)
    return state + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


@jax.jit
def run_pack(state0, params, I_profile, T_amb, dt):
    def step(s, I_t):
        s_next = rk4(s, I_t, params, T_amb, dt)
        return s_next, (s_next[:, 0], s_next[:, 2])

    _, traces = jax.lax.scan(step, state0, I_profile)
    return traces


def build_driving_current(T_steps: int, dt: float, seed: int = 0) -> jnp.ndarray:
    """Synthetic US06-flavoured current profile — pack-level Amps drawn."""
    rng = np.random.default_rng(seed)
    t_arr = np.arange(T_steps) * dt
    base = -5.0 * (1.0 + 0.3 * np.sin(2 * np.pi * t_arr / 120.0))
    spike = -10.0 * (rng.random(T_steps) > 0.85)
    regen = +6.0 * (rng.random(T_steps) > 0.92)
    return jnp.asarray(base + spike + regen, dtype=jnp.float32)


def main():
    T_END = 600.0
    dt = 1.0
    T_steps = int(T_END / dt)
    I_profile = build_driving_current(T_steps, dt)

    N_grid = (8, 32, 128, 512, 2048, 10_000, 50_000, 100_000)
    walls, rss_peaks, soc_mu, soc_sd, T_mu, T_sd = [], [], [], [], [], []
    SOC_final_10k = None
    T_final_10k = None
    key = jax.random.PRNGKey(42)

    proc = psutil.Process(os.getpid())

    for N in N_grid:
        key, sub = jax.random.split(key)
        cap_pert = 1.0 + 0.02 * jax.random.normal(sub, (N,))
        params = jnp.stack(
            [
                jnp.full((N,), 0.025, dtype=jnp.float32),
                jnp.full((N,), 0.015, dtype=jnp.float32),
                jnp.full((N,), 80.0, dtype=jnp.float32),
                (2.5 * cap_pert).astype(jnp.float32),
                jnp.full((N,), 8.0, dtype=jnp.float32),
                jnp.full((N,), 50.0, dtype=jnp.float32),
            ],
            axis=-1,
        )
        state0 = jnp.stack(
            [
                jnp.full((N,), 0.9, dtype=jnp.float32),
                jnp.zeros((N,), dtype=jnp.float32),
                jnp.full((N,), 298.15, dtype=jnp.float32),
            ],
            axis=-1,
        )
        gc.collect()
        # Warmup compile (shape-keyed JIT)
        SOCs, Ts = run_pack(state0, params, I_profile, 298.15, dt)
        SOCs.block_until_ready()
        gc.collect()
        rss_pre = proc.memory_info().rss / 1e6
        tA = time.time()
        SOCs, Ts = run_pack(state0, params, I_profile, 298.15, dt)
        SOCs.block_until_ready()
        tB = time.time()
        rss_post = proc.memory_info().rss / 1e6
        SOC_final = np.asarray(SOCs[-1])
        T_final = np.asarray(Ts[-1])
        walls.append(tB - tA)
        rss_peaks.append(max(rss_post, rss_pre))
        soc_mu.append(float(SOC_final.mean()))
        soc_sd.append(float(SOC_final.std()))
        T_mu.append(float(T_final.mean()))
        T_sd.append(float(T_final.std()))
        if N == 10_000:
            SOC_final_10k = SOC_final.astype(np.float32)
            T_final_10k = T_final.astype(np.float32)
        print(
            f"N={N:>7d}  wall={walls[-1]:.4f}s  rss={rss_peaks[-1]:.1f} MB  "
            f"SOC mu={soc_mu[-1]:.4f} sd={soc_sd[-1]:.4f}  T mu={T_mu[-1]:.2f}"
        )

    out_path = os.path.join(os.path.dirname(__file__), "battery_pack_10k_publication.npz")
    np.savez(
        out_path,
        N_grid=np.array(N_grid, dtype=np.int64),
        wall_s=np.array(walls, dtype=np.float64),
        rss_mb=np.array(rss_peaks, dtype=np.float64),
        soc_mu=np.array(soc_mu, dtype=np.float64),
        soc_sd=np.array(soc_sd, dtype=np.float64),
        T_mu=np.array(T_mu, dtype=np.float64),
        T_sd=np.array(T_sd, dtype=np.float64),
        SOC_final_10k=SOC_final_10k,
        T_final_10k=T_final_10k,
        placeholder_flag=np.array(False),
    )
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
