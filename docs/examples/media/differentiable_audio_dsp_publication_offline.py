#!/usr/bin/env python3
"""Offline run for differentiable_audio_dsp.ipynb.

Produces media/differentiable_audio_dsp_publication.npz containing the LUFS
auto-tune trajectory: per-iteration LUFS, loss, and the four compressor
parameters. The notebook loads this NPZ in publication mode (default) and
falls back to a live re-run in fast mode if the NPZ is missing.

Runtime on a developer CPU: ~30 s (the L-BFGS-B loop calls jax.grad through
5 s of 44.1 kHz audio ~20 times). The notebook's fast mode runs the same
loop with the same maxiter and is essentially equivalent — this script
exists so the notebook execution stays sub-minute on `nbconvert`.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import minimize

# Resolve paths relative to this script so it works from any cwd.
HERE = Path(__file__).resolve().parent
OUT_NPZ = HERE / "differentiable_audio_dsp_publication.npz"

# Sample rates (kept in sync with the notebook).
FS_AUDIO = 44_100.0
FS_CTRL = 1_000.0
DT_AUDIO = 1.0 / FS_AUDIO
DT_CTRL = 1.0 / FS_CTRL

T_CLIP = 5.0
N_AUDIO = int(T_CLIP * FS_AUDIO)

SEED = 0
TARGET_LUFS = -14.0

EQ_NOMINAL = dict(
    low_f0=80.0, low_Q=0.7, low_G=+3.0,
    mid_f0=1500.0, mid_Q=1.0, mid_G=+4.0,
    hi_f0=8000.0, hi_Q=0.7, hi_G=+2.0,
)


def synthesize_test_clip(n_samples: int, fs: float, seed: int = SEED) -> jnp.ndarray:
    rng = np.random.default_rng(seed)
    white = rng.standard_normal(n_samples) * 0.5
    b = np.array([0.99886, 0.99332, 0.96900, 0.86650, 0.55000, -0.7616])
    a = np.array([0.0555179, 0.0750759, 0.1538520, 0.3104856, 0.5329522, 0.0168980])
    state = np.zeros(6)
    pink = np.zeros(n_samples)
    for k in range(n_samples):
        state = b * state + a * white[k]
        pink[k] = state.sum() + white[k] * 0.5362
    pink *= 0.18
    t = np.arange(n_samples) / fs
    kick_start = 0.5
    kick_duration = 0.10
    kick_env = np.exp(-(t - kick_start) / 0.020) * (t >= kick_start) * (t < kick_start + kick_duration)
    kick = 0.8 * kick_env * np.sin(2 * np.pi * 60.0 * (t - kick_start))
    tone = 0.15 * np.sin(2 * np.pi * 220.0 * t)
    x = pink + kick + tone
    x = x * (0.9 / max(np.max(np.abs(x)), 1e-9))
    return jnp.asarray(x)   # default float (float64 with jaxonomy's x64 on)


def biquad_peaking(f0, Q, G_db, fs):
    A = 10.0 ** (G_db / 40.0)
    w0 = 2.0 * jnp.pi * f0 / fs
    cos_w0 = jnp.cos(w0)
    alpha = jnp.sin(w0) / (2.0 * Q)
    b0 = 1.0 + alpha * A
    b1 = -2.0 * cos_w0
    b2 = 1.0 - alpha * A
    a0 = 1.0 + alpha / A
    a1 = -2.0 * cos_w0
    a2 = 1.0 - alpha / A
    return b0 / a0, b1 / a0, b2 / a0, a1 / a0, a2 / a0


def biquad_lowshelf(f0, Q, G_db, fs):
    A = 10.0 ** (G_db / 40.0)
    w0 = 2.0 * jnp.pi * f0 / fs
    cos_w0 = jnp.cos(w0)
    alpha = jnp.sin(w0) / (2.0 * Q)
    sqA = jnp.sqrt(A)
    b0 = A * ((A + 1) - (A - 1) * cos_w0 + 2 * sqA * alpha)
    b1 = 2.0 * A * ((A - 1) - (A + 1) * cos_w0)
    b2 = A * ((A + 1) - (A - 1) * cos_w0 - 2 * sqA * alpha)
    a0 = (A + 1) + (A - 1) * cos_w0 + 2 * sqA * alpha
    a1 = -2.0 * ((A - 1) + (A + 1) * cos_w0)
    a2 = (A + 1) + (A - 1) * cos_w0 - 2 * sqA * alpha
    return b0 / a0, b1 / a0, b2 / a0, a1 / a0, a2 / a0


def biquad_highshelf(f0, Q, G_db, fs):
    A = 10.0 ** (G_db / 40.0)
    w0 = 2.0 * jnp.pi * f0 / fs
    cos_w0 = jnp.cos(w0)
    alpha = jnp.sin(w0) / (2.0 * Q)
    sqA = jnp.sqrt(A)
    b0 = A * ((A + 1) + (A - 1) * cos_w0 + 2 * sqA * alpha)
    b1 = -2.0 * A * ((A - 1) + (A + 1) * cos_w0)
    b2 = A * ((A + 1) + (A - 1) * cos_w0 - 2 * sqA * alpha)
    a0 = (A + 1) - (A - 1) * cos_w0 + 2 * sqA * alpha
    a1 = 2.0 * ((A - 1) - (A + 1) * cos_w0)
    a2 = (A + 1) - (A - 1) * cos_w0 - 2 * sqA * alpha
    return b0 / a0, b1 / a0, b2 / a0, a1 / a0, a2 / a0


def smooth_max_zero(u, width_db):
    return 0.5 * (u + jnp.sqrt(u * u + width_db * width_db))


def compressor_gain_db(env_lin, threshold_db, ratio, knee_db, makeup_db):
    env_db = 20.0 * jnp.log10(jnp.maximum(env_lin, 1e-9))
    over_thresh = smooth_max_zero(env_db - threshold_db, knee_db)
    gain_reduction_db = -over_thresh * (1.0 - 1.0 / ratio)
    return gain_reduction_db + makeup_db


@jax.jit
def chain_scan(x_audio, eq_params, comp_params):
    fs_audio = FS_AUDIO
    fs_ctrl = FS_CTRL
    decim = int(round(fs_audio / fs_ctrl))

    cL = biquad_lowshelf(eq_params["low_f0"], eq_params["low_Q"], eq_params["low_G"], fs_audio)
    cM = biquad_peaking(eq_params["mid_f0"], eq_params["mid_Q"], eq_params["mid_G"], fs_audio)
    cH = biquad_highshelf(eq_params["hi_f0"], eq_params["hi_Q"], eq_params["hi_G"], fs_audio)

    alpha_a = 1.0 - jnp.exp(-1.0 / (comp_params["tau_a"] * fs_ctrl))
    alpha_r = 1.0 - jnp.exp(-1.0 / (comp_params["tau_r"] * fs_ctrl))

    def biquad_step(coef, x, s):
        b0, b1, b2, a1, a2 = coef
        y = b0 * x + b1 * s[0] + b2 * s[1] - a1 * s[2] - a2 * s[3]
        return y, jnp.stack([x, s[0], y, s[2]])

    init_carry = (jnp.zeros(4), jnp.zeros(4), jnp.zeros(4),
                  jnp.float64(0.0), jnp.float64(1.0), jnp.int32(0))

    def step(carry, x_k):
        sL, sM, sH, env_prev, gain_held, k = carry
        y1, sL_new = biquad_step(cL, x_k, sL)
        y2, sM_new = biquad_step(cM, y1, sM)
        y3, sH_new = biquad_step(cH, y2, sH)
        is_ctrl_tick = (k % decim) == 0
        abs_y3 = jnp.abs(y3)
        attacking = abs_y3 > env_prev
        alpha = jnp.where(attacking, alpha_a, alpha_r)
        env_new_candidate = alpha * abs_y3 + (1.0 - alpha) * env_prev
        env_new = jnp.where(is_ctrl_tick, env_new_candidate, env_prev)
        g_db_candidate = compressor_gain_db(env_new, comp_params["threshold_db"],
                                            comp_params["ratio"], comp_params["knee_db"],
                                            comp_params["makeup_db"])
        gain_candidate = 10.0 ** (g_db_candidate / 20.0)
        gain = jnp.where(is_ctrl_tick, gain_candidate, gain_held)
        y_out = y3 * gain
        carry_new = (sL_new, sM_new, sH_new, env_new, gain, k + 1)
        return carry_new, (y_out, env_new, gain)

    _, ys = jax.lax.scan(step, init_carry, x_audio)
    y_audio, env, gain = ys
    return {"y": y_audio, "env": env, "gain": gain}


def lufs(y):
    return -0.691 + 10.0 * jnp.log10(jnp.mean(y * y) + 1e-12)


@jax.jit
def loss_lufs(comp_vec, x_audio_arr):
    T_db, R, tau_a, tau_r = comp_vec
    comp = dict(threshold_db=T_db, ratio=R, tau_a=tau_a, tau_r=tau_r,
                knee_db=6.0, makeup_db=0.0)
    out = chain_scan(x_audio_arr, EQ_NOMINAL, comp)
    return (lufs(out["y"]) - TARGET_LUFS) ** 2


def main():
    print("Synthesising test clip...")
    x_audio = synthesize_test_clip(N_AUDIO, FS_AUDIO)
    print(f"  clip shape={x_audio.shape}, peak |x|={float(jnp.max(jnp.abs(x_audio))):.3f}")

    grad_loss = jax.jit(jax.value_and_grad(loss_lufs))

    # Warm-up JIT.  Baseline = aggressive compression (T=-30 dB, R=6:1):
    # over-compressed at ~-30 LUFS, 16 LU away from the -14 LUFS target so
    # the optimiser has clear ground to cover.
    print("Warming up jax.grad through 5 s of audio...")
    comp_baseline = jnp.array([-30.0, 6.0, 0.010, 0.080])
    v0, g0 = grad_loss(comp_baseline, x_audio)
    _ = v0.block_until_ready()
    print(f"  baseline loss = {float(v0):.4f}")
    print(f"  baseline grad = {np.asarray(g0)}")

    history_iter, history_lufs, history_loss, history_params = [], [], [], []

    def callback(xk):
        out = chain_scan(x_audio, EQ_NOMINAL, dict(
            threshold_db=xk[0], ratio=xk[1], tau_a=xk[2], tau_r=xk[3],
            knee_db=6.0, makeup_db=0.0))
        L = float(lufs(out["y"]))
        history_iter.append(len(history_iter))
        history_lufs.append(L)
        history_loss.append((L - TARGET_LUFS) ** 2)
        history_params.append(np.array(xk, dtype=np.float64))

    def f_and_g_np(xk):
        v, g = grad_loss(jnp.asarray(xk), x_audio)
        return float(v), np.asarray(g, dtype=np.float64)

    bounds = [(-40.0, 0.0), (1.1, 20.0), (0.001, 0.050), (0.005, 0.500)]
    callback(np.asarray(comp_baseline))

    print("Running L-BFGS-B (maxiter=20)...")
    t0 = time.time()
    res = minimize(
        f_and_g_np, np.asarray(comp_baseline, dtype=np.float64),
        method="L-BFGS-B", jac=True, bounds=bounds, callback=callback,
        options=dict(maxiter=20, ftol=1e-9, gtol=1e-7, disp=False),
    )
    wall_s = time.time() - t0
    comp_opt = res.x
    out_opt = chain_scan(x_audio, EQ_NOMINAL, dict(
        threshold_db=comp_opt[0], ratio=comp_opt[1],
        tau_a=comp_opt[2], tau_r=comp_opt[3], knee_db=6.0, makeup_db=0.0))
    lufs_opt = float(lufs(out_opt["y"]))
    print(f"  wall time   = {wall_s:.2f} s")
    print(f"  n_iter      = {len(history_iter)}")
    print(f"  optimised LUFS = {lufs_opt:.4f}  (target = {TARGET_LUFS:.1f})")
    print(f"  optimised |LUFS - target| = {abs(lufs_opt - TARGET_LUFS):.4f}")
    print(f"  optimised params = {comp_opt}")

    history_iter = np.asarray(history_iter)
    history_lufs = np.asarray(history_lufs)
    history_loss = np.asarray(history_loss)
    history_params = np.stack(history_params)

    OUT_NPZ.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        OUT_NPZ,
        history_iter=history_iter,
        history_lufs=history_lufs,
        history_loss=history_loss,
        history_params=history_params,
        comp_opt=np.asarray(comp_opt, dtype=np.float32),
        lufs_opt=np.float32(lufs_opt),
        n_iter=np.int32(len(history_iter)),
        pub_wall_s=np.float32(wall_s),
        placeholder_flag=np.bool_(False),
    )
    print(f"Wrote {OUT_NPZ}")


if __name__ == "__main__":
    main()
