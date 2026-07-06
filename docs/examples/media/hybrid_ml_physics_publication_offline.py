#!/usr/bin/env python3
"""Offline publication-quality run for ``hybrid_ml_physics_predictor.ipynb``.

Produces ``media/hybrid_ml_physics_publication.npz`` containing:
  - the trained MLP-residual parameters (Equinox-MLP weight pytree, flattened),
  - the three closed-loop simulation trajectories (pure-physics, pure-ML,
    hybrid) on the in-distribution test scenario,
  - the OOD test trajectories,
  - the training-loss curve.

The notebook in default ``publication`` mode loads from this NPZ and plots
in seconds. In ``fast`` mode (or when the NPZ is missing) the notebook
re-runs a smaller training step inline.

Run from the repo root::

    JAXONOMY_DISABLE_PROFILING=1 \
        python docs/examples/media/hybrid_ml_physics_publication_offline.py

Expected wall-time: ~30 s on a developer machine.

The wedge: in a production setting steps 1-3 would happen in a PyTorch /
TensorFlow training pipeline that the user already owns; the trained
checkpoint would land here as ``hybrid_residual_model.pt`` and the notebook
would load it via ``jaxonomy.library.PyTorch(file_name=...)``. Neither
torch nor tensorflow is installed in CI, so we ship the JAX/Equinox
equivalent and document the swap in the notebook.
"""
from __future__ import annotations

from pathlib import Path
import time

import numpy as np
import jax
import jax.numpy as jnp
import equinox as eqx
import optax


HERE = Path(__file__).resolve().parent
NPZ_OUT = HERE / "hybrid_ml_physics_publication.npz"

# ---------------------------------------------------------------------
# Top-level constants — must match the notebook.
# ---------------------------------------------------------------------
G = 9.81           # m/s^2
L = 1.0            # m  (pendulum length)
M = 1.0            # kg
B_VISC = 0.10      # N·m·s   viscous damping (known to the engineer)
F_COUL = 0.30      # N·m     Coulomb friction (the residual the MLP must learn)
COUL_EPS = 0.05    # rad/s   sign smoothing for the "true" Coulomb term

DT_SIM = 0.01      # s       integration timestep
T_TRAIN = 30.0     # s       training-data trajectory length
T_TEST = 12.0      # s       in-distribution test length
T_OOD = 8.0        # s       out-of-distribution test length

N_HIDDEN = 32
N_LAYERS = 2
N_EPOCHS = 4000
LR = 5e-3
NOISE_THETA_DDOT = 0.05  # rad/s^2

SEED = 7


# ---------------------------------------------------------------------
# Physics: true vs known plants. We integrate both with a hand-rolled
# fixed-step RK4 here in the offline script for portability and speed;
# the notebook does the same kinematics inside jaxonomy LeafSystems so
# the reader sees the framework patterns.
# ---------------------------------------------------------------------
def smooth_sign(v: jnp.ndarray, eps: float = COUL_EPS) -> jnp.ndarray:
    """tanh-based smoothed sign for differentiability through the Coulomb term."""
    return jnp.tanh(v / eps)


def true_accel(theta: jnp.ndarray, omega: jnp.ndarray, tau: jnp.ndarray) -> jnp.ndarray:
    """True plant ddot{theta} including Coulomb friction (engineer doesn't know F_COUL)."""
    grav = -(G / L) * jnp.sin(theta)
    visc = -(B_VISC / (M * L * L)) * omega
    coul = -(F_COUL / (M * L * L)) * smooth_sign(omega)
    drive = tau / (M * L * L)
    return grav + visc + coul + drive


def known_accel(theta: jnp.ndarray, omega: jnp.ndarray, tau: jnp.ndarray) -> jnp.ndarray:
    """Known-physics ddot{theta}: gravity + viscous damping + torque (no Coulomb)."""
    grav = -(G / L) * jnp.sin(theta)
    visc = -(B_VISC / (M * L * L)) * omega
    drive = tau / (M * L * L)
    return grav + visc + drive


def rk4_step(state, tau, accel_fn, dt):
    """Fixed-step RK4 for a (theta, omega) two-state pendulum under accel_fn."""
    def deriv(s):
        th, om = s
        return jnp.array([om, accel_fn(th, om, tau)])
    k1 = deriv(state)
    k2 = deriv(state + 0.5 * dt * k1)
    k3 = deriv(state + 0.5 * dt * k2)
    k4 = deriv(state + dt * k3)
    return state + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


def rollout(state0, taus, accel_fn, dt):
    """Roll out a fixed-step trajectory of length len(taus) under accel_fn."""
    def step(carry, tau):
        s_next = rk4_step(carry, tau, accel_fn, dt)
        return s_next, s_next
    _, traj = jax.lax.scan(step, state0, taus)
    return jnp.concatenate([state0[None, :], traj], axis=0)


# ---------------------------------------------------------------------
# Training data: drive the TRUE plant with a low-frequency random torque
# excitation, sample at 100 Hz, build (theta, omega) -> residual targets.
# ---------------------------------------------------------------------
def make_training_excitation(key, n):
    """Smoothed-noise torque excitation: bounded, low-frequency, mean-zero."""
    raw = jax.random.normal(key, (n,)) * 0.6  # ~0.6 N·m std
    # smooth with a 50-sample moving average for low-freq character
    kernel = jnp.ones(50) / 50.0
    smoothed = jnp.convolve(raw, kernel, mode="same")
    return smoothed


def build_training_set(key):
    """Generate a (theta, omega, theta_ddot_residual) supervised dataset."""
    n_steps = int(T_TRAIN / DT_SIM)
    k_drv, k_init, k_noise = jax.random.split(key, 3)
    taus = make_training_excitation(k_drv, n_steps)
    theta0 = jax.random.uniform(k_init, minval=-0.5, maxval=0.5)
    omega0 = 0.0
    state0 = jnp.array([theta0, omega0])
    traj = rollout(state0, taus, true_accel, DT_SIM)  # (n_steps+1, 2)
    thetas = traj[:-1, 0]
    omegas = traj[:-1, 1]
    # ground-truth residual = true - known, evaluated under the true (theta, omega)
    true_a = jax.vmap(true_accel)(thetas, omegas, taus)
    known_a = jax.vmap(known_accel)(thetas, omegas, taus)
    residual = true_a - known_a  # what the MLP must learn
    # add measurement noise on the residual target (the engineer would measure
    # theta_ddot via finite differences on noisy theta, which contaminates the
    # residual estimate)
    noise = jax.random.normal(k_noise, residual.shape) * NOISE_THETA_DDOT
    return thetas, omegas, residual + noise, taus


# ---------------------------------------------------------------------
# MLP residual model: a small Equinox MLP. The notebook documents how to
# swap this for a `jaxonomy.library.PyTorch` / `TensorFlow` predictor.
# ---------------------------------------------------------------------
def make_mlp(key):
    return eqx.nn.MLP(
        in_size=2,
        out_size=1,
        width_size=N_HIDDEN,
        depth=N_LAYERS,
        activation=jax.nn.tanh,
        key=key,
    )


def train(thetas, omegas, residuals, key):
    mlp = make_mlp(key)
    inputs = jnp.stack([thetas, omegas], axis=-1)         # (N, 2)
    targets = residuals[:, None]                          # (N, 1)

    params, static = eqx.partition(mlp, eqx.is_array)
    opt = optax.adam(LR)
    opt_state = opt.init(params)

    @jax.jit
    def loss_fn(params, x, y):
        m = eqx.combine(params, static)
        pred = jax.vmap(m)(x)
        return jnp.mean((pred - y) ** 2)

    @jax.jit
    def step(params, opt_state, x, y):
        loss, grads = jax.value_and_grad(loss_fn)(params, x, y)
        upd, opt_state = opt.update(grads, opt_state)
        params = optax.apply_updates(params, upd)
        return params, opt_state, loss

    loss_hist = []
    for epoch in range(N_EPOCHS):
        params, opt_state, loss = step(params, opt_state, inputs, targets)
        if epoch % 100 == 0:
            loss_hist.append(float(loss))
    return params, static, jnp.array(loss_hist)


# ---------------------------------------------------------------------
# Closed-loop comparison: pure-physics vs pure-ML vs hybrid under a
# setpoint-tracking PD controller, on the SAME plant the engineer doesn't
# know exactly (the true plant). The three predictors differ in how they
# compute the simulated ddot{theta}; ground truth is always the true plant.
# ---------------------------------------------------------------------
KP, KD = 6.0, 1.5
THETA_REF = 0.8  # rad — a modest setpoint inside training distribution

def control_torque(theta, omega, theta_ref):
    return KP * (theta_ref - theta) - KD * omega


def closed_loop(state0, accel_fn, t_end, theta_ref=THETA_REF):
    n_steps = int(t_end / DT_SIM)
    def step(carry, _):
        th, om = carry
        tau = control_torque(th, om, theta_ref)
        s_next = rk4_step(carry, tau, accel_fn, DT_SIM)
        return s_next, jnp.array([s_next[0], s_next[1], tau])
    _, packed = jax.lax.scan(step, state0, jnp.arange(n_steps))
    times = jnp.arange(n_steps) * DT_SIM
    return times, packed  # packed[:, 0]=theta, [:,1]=omega, [:,2]=tau


def hybrid_accel_factory(params, static):
    mlp = eqx.combine(params, static)
    def hybrid_accel(theta, omega, tau):
        x = jnp.array([theta, omega])
        residual = mlp(x)[0]
        return known_accel(theta, omega, tau) + residual
    return hybrid_accel


def pure_ml_accel_factory(params, static, taus_for_baseline=None):
    """Pure-ML predicts the FULL ddot{theta} as f(theta, omega), ignoring tau.

    This is the deliberately-naive comparison point: an ML model that learned
    the unforced dynamics. We train it the same way as the residual MLP but
    target = true ddot{theta} (under tau=0) so the model has no notion of the
    applied torque. (See `pure_ml_train` below.)
    """
    mlp = eqx.combine(params, static)
    def ml_accel(theta, omega, tau):
        x = jnp.array([theta, omega])
        # ML is unaware of tau by construction — that's the honest weakness
        # we want to demonstrate.
        return mlp(x)[0] + (tau / (M * L * L))
    return ml_accel


def pure_ml_train(key):
    """Train an MLP to predict ddot{theta} | tau = 0 under the true plant,
    using the same (theta, omega) coverage as the residual training set.
    """
    k1, k2 = jax.random.split(key)
    thetas, omegas, residuals_unused, _ = build_training_set(k1)
    # rebuild targets: true acceleration at zero torque (no driving)
    true_a_no_tau = jax.vmap(lambda th, om: true_accel(th, om, jnp.array(0.0)))(thetas, omegas)
    noise = jax.random.normal(k2, true_a_no_tau.shape) * NOISE_THETA_DDOT
    targets = true_a_no_tau + noise
    return train(thetas, omegas, targets, jax.random.PRNGKey(SEED + 1))


def main():
    t0 = time.time()
    key = jax.random.PRNGKey(SEED)
    k_train, k_pure_ml = jax.random.split(key)

    print("--- Stage 1: build training data ---")
    thetas, omegas, residuals, train_taus = build_training_set(k_train)
    print(f"  N samples = {thetas.shape[0]}, residual stats: mean={float(jnp.mean(residuals)):+.4f}, "
          f"std={float(jnp.std(residuals)):.4f}")

    print("--- Stage 2: train residual MLP ---")
    params, static, loss_hist = train(thetas, omegas, residuals, jax.random.PRNGKey(SEED + 2))
    final_loss = float(loss_hist[-1])
    print(f"  Trained {N_EPOCHS} epochs, final MSE = {final_loss:.4e}")

    print("--- Stage 3: train pure-ML baseline ---")
    pure_ml_params, pure_ml_static, pure_ml_loss_hist = pure_ml_train(k_pure_ml)
    print(f"  Pure-ML final MSE = {float(pure_ml_loss_hist[-1]):.4e}")

    print("--- Stage 4: closed-loop comparison (in-distribution) ---")
    state0 = jnp.array([0.0, 0.0])
    times_true, traj_true = closed_loop(state0, true_accel, T_TEST)
    times_phys, traj_phys = closed_loop(state0, known_accel, T_TEST)
    times_hyb, traj_hyb = closed_loop(state0, hybrid_accel_factory(params, static), T_TEST)
    times_ml, traj_ml = closed_loop(state0, pure_ml_accel_factory(pure_ml_params, pure_ml_static), T_TEST)

    def rmse(traj, ref):
        return float(jnp.sqrt(jnp.mean((traj[:, 0] - ref[:, 0]) ** 2)))

    print(f"  RMSE vs true:  pure-physics={rmse(traj_phys, traj_true):.4f} rad")
    print(f"                 pure-ML     ={rmse(traj_ml, traj_true):.4f} rad")
    print(f"                 hybrid      ={rmse(traj_hyb, traj_true):.4f} rad")

    print("--- Stage 5: OOD test (large initial angle, no setpoint) ---")
    state0_ood = jnp.array([2.5, 0.0])  # well outside the training range of (-0.5, 0.5)
    times_true_ood, traj_true_ood = closed_loop(state0_ood, true_accel, T_OOD, theta_ref=0.0)
    times_phys_ood, traj_phys_ood = closed_loop(state0_ood, known_accel, T_OOD, theta_ref=0.0)
    times_hyb_ood, traj_hyb_ood = closed_loop(state0_ood, hybrid_accel_factory(params, static), T_OOD, theta_ref=0.0)
    times_ml_ood, traj_ml_ood = closed_loop(state0_ood, pure_ml_accel_factory(pure_ml_params, pure_ml_static), T_OOD, theta_ref=0.0)
    print(f"  OOD RMSE vs true: pure-physics={rmse(traj_phys_ood, traj_true_ood):.4f} rad")
    print(f"                    pure-ML     ={rmse(traj_ml_ood, traj_true_ood):.4f} rad")
    print(f"                    hybrid      ={rmse(traj_hyb_ood, traj_true_ood):.4f} rad")

    print("--- Stage 6: residual grid for heatmap visualization ---")
    th_grid = jnp.linspace(-1.0, 1.0, 41)
    om_grid = jnp.linspace(-2.0, 2.0, 41)
    TH, OM = jnp.meshgrid(th_grid, om_grid, indexing="xy")
    pts = jnp.stack([TH.ravel(), OM.ravel()], axis=-1)
    mlp_eval = eqx.combine(params, static)
    pred_res = jax.vmap(mlp_eval)(pts).reshape(TH.shape)
    true_res = jax.vmap(lambda th, om: -(F_COUL / (M * L * L)) * smooth_sign(om))(TH.ravel(), OM.ravel()).reshape(TH.shape)

    wall = time.time() - t0
    print(f"Total wall-time = {wall:.2f} s; writing NPZ to {NPZ_OUT}")

    # Flatten the MLP parameter pytree for portable storage.
    flat_params = jax.tree.leaves(params)
    flat_pure_ml = jax.tree.leaves(pure_ml_params)

    np.savez(
        NPZ_OUT,
        # training-data set
        train_thetas=np.asarray(thetas),
        train_omegas=np.asarray(omegas),
        train_residuals=np.asarray(residuals),
        train_taus=np.asarray(train_taus),
        # loss curves
        loss_hist=np.asarray(loss_hist),
        pure_ml_loss_hist=np.asarray(pure_ml_loss_hist),
        # in-distribution closed-loop
        time_id=np.asarray(times_true),
        traj_true=np.asarray(traj_true),
        traj_phys=np.asarray(traj_phys),
        traj_hyb=np.asarray(traj_hyb),
        traj_ml=np.asarray(traj_ml),
        # OOD closed-loop
        time_ood=np.asarray(times_true_ood),
        traj_true_ood=np.asarray(traj_true_ood),
        traj_phys_ood=np.asarray(traj_phys_ood),
        traj_hyb_ood=np.asarray(traj_hyb_ood),
        traj_ml_ood=np.asarray(traj_ml_ood),
        # residual heatmap data
        th_grid=np.asarray(th_grid),
        om_grid=np.asarray(om_grid),
        pred_res=np.asarray(pred_res),
        true_res=np.asarray(true_res),
        # MLP params (pytree-flattened; the notebook reconstructs alongside identical static)
        n_mlp_leaves=len(flat_params),
        n_pure_ml_leaves=len(flat_pure_ml),
        wall_time=wall,
        seed=SEED,
        **{f"mlp_leaf_{i}": np.asarray(p) for i, p in enumerate(flat_params)},
        **{f"pure_ml_leaf_{i}": np.asarray(p) for i, p in enumerate(flat_pure_ml)},
    )
    print(f"Done. NPZ size = {NPZ_OUT.stat().st_size/1024:.1f} KiB.")


if __name__ == "__main__":
    main()
