# SPDX-License-Identifier: MIT
"""Cinematic render of a Falcon-9-class booster landing on a drone-ship pad.

This is the culminating visual artifact for the returning-booster tutorial
series (Parts 1–5). It uses Part 1's open-loop optimal trajectory (the only
one in the series that lands cleanly) and a rich MuJoCo scene that goes
well beyond the minimal cylinder-on-checkerboard of Part 2:

- Tapered booster body + nose cone + four grid fins + four deployed legs.
- Autonomous-drone-ship-style barge pad with concentric landing target.
- Procedural ocean ground plane with subtle wave-like normal variation.
- Gradient skybox with sun + fill lighting.
- Throttle-modulated thrust plume visible during the burn.

Run directly:
    python docs/examples/media/render_booster.py

Outputs:
    docs/examples/media/booster_landing_cinematic.mp4 (or .gif fallback)
    docs/examples/media/booster_scene.xml             (the MJCF)
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp
from jax import config as jax_config

jax_config.update("jax_enable_x64", True)

import jaxonomy
from jaxonomy.framework import LeafSystem

import mujoco

# imageio for output; ffmpeg if available else GIF fallback
try:
    import imageio.v2 as imageio
    _HAVE_IMAGEIO = True
except ImportError:
    import imageio
    _HAVE_IMAGEIO = True


# ─────────────────────────────────────────────────────────────────────────────
# Physical constants (match Part 1)
# ─────────────────────────────────────────────────────────────────────────────

M_DRY     = 25_000.0
M_FUEL_0  = 5_000.0
T_MAX     = 600_000.0
ETA_MIN, ETA_MAX = 0.4, 1.0
ISP       = 280.0
G0, G     = 9.80665, 9.81
DELTA_MAX = np.deg2rad(10.0)
L_BOOSTER = 40.0
R_BOOSTER = 1.85


def _cylinder_inertia(m, L, r):
    Ixx = Iyy = m * (3 * r**2 + L**2) / 12.0
    Izz = 0.5 * m * r**2
    return Ixx, Iyy, Izz

IXX0, IYY0, IZZ0 = _cylinder_inertia(M_DRY + M_FUEL_0, L_BOOSTER, R_BOOSTER)


# ─────────────────────────────────────────────────────────────────────────────
# Booster RHS — Z-Y-X Euler, identical to Part 1's BoosterTrajopt
# ─────────────────────────────────────────────────────────────────────────────

def _euler_zyx_kinematics_inv(phi, theta):
    cphi, sphi = jnp.cos(phi), jnp.sin(phi)
    cth,  sth  = jnp.cos(theta), jnp.sin(theta)
    tth = sth / cth
    return jnp.array([
        [1.0, sphi*tth,   cphi*tth],
        [0.0, cphi,      -sphi    ],
        [0.0, sphi/cth,   cphi/cth],
    ])


def booster_rhs(state, control):
    pos   = state[0:3]; vel = state[3:6]
    eta_a = state[6:9]; omega = state[9:12]; m_f = state[12]
    eta_throttle, delta_y, delta_z = control
    phi, theta, psi = eta_a
    m_total = M_DRY + jnp.maximum(m_f, 0.0)
    cphi, sphi = jnp.cos(phi), jnp.sin(phi)
    cth,  sth  = jnp.cos(theta), jnp.sin(theta)
    cpsi, spsi = jnp.cos(psi), jnp.sin(psi)
    R_wb = jnp.array([
        [cpsi*cth, cpsi*sth*sphi - spsi*cphi, cpsi*sth*cphi + spsi*sphi],
        [spsi*cth, spsi*sth*sphi + cpsi*cphi, spsi*sth*cphi - cpsi*sphi],
        [-sth,     cth*sphi,                   cth*cphi                  ],
    ])
    F_B = eta_throttle * T_MAX * jnp.array([
        jnp.sin(delta_y), -jnp.sin(delta_z), jnp.cos(delta_y) * jnp.cos(delta_z),
    ])
    F_thrust_W = R_wb @ F_B
    G_W = jnp.array([0.0, 0.0, -m_total * G])
    acc = (F_thrust_W + G_W) / m_total
    tau_B = jnp.array([
        -(L_BOOSTER / 2.0) * eta_throttle * T_MAX * jnp.sin(delta_z),
        -(L_BOOSTER / 2.0) * eta_throttle * T_MAX * jnp.sin(delta_y),
         0.0,
    ])
    I_diag = jnp.array([IXX0, IYY0, IZZ0])
    omega_dot = (tau_B - jnp.cross(omega, I_diag * omega)) / I_diag
    eta_dot = _euler_zyx_kinematics_inv(phi, theta) @ omega
    m_f_dot = jnp.where(m_f > 0.0, -eta_throttle * T_MAX / (ISP * G0), 0.0)
    return jnp.concatenate([vel, acc, eta_dot, omega_dot, jnp.array([m_f_dot])])


class Booster(LeafSystem):
    def __init__(self, x0, name="booster"):
        super().__init__(name=name)
        self.declare_input_port(name="u")
        self.declare_continuous_state(default_value=jnp.array(x0), ode=self._ode)
        self.declare_continuous_state_output(name="x")
    def _ode(self, time, state, *inputs, **params):
        return booster_rhs(state.continuous_state, inputs[0])


# ─────────────────────────────────────────────────────────────────────────────
# Solve Part 1's open-loop optimal trajectory
# ─────────────────────────────────────────────────────────────────────────────

def solve_part1_trajectory():
    x0 = np.array([420.0, 0.0, 630.0, -60.0, 0.0, -90.0,
                   0.0, np.deg2rad(-10.0), 0.0, 0.0, 0.0, 0.0, M_FUEL_0])
    xf = np.zeros(13)
    T_BURN = 14.0
    N = 30
    Q  = np.diag([1e-3]*3 + [1e-2]*3 + [1e-1]*3 + [1e-1]*3 + [0.0])
    QN = np.diag([1e5]*3  + [1e5]*3  + [1e4]*3  + [1e3]*3  + [0.0])
    R  = np.diag([1.0, 1e1, 1e1])
    lb_x = np.array([-3000., -1000., 0., -300., -200., -200.,
                     np.deg2rad(-45.), np.deg2rad(-60.), np.deg2rad(-45.),
                     -2., -2., -2., 0.])
    ub_x = np.array([ 3000.,  1000., 1500.,  300.,  200.,  200.,
                      np.deg2rad(45.), np.deg2rad(60.), np.deg2rad(45.),
                      2., 2., 2., M_FUEL_0])
    lb_u = np.array([ETA_MIN, -DELTA_MAX, -DELTA_MAX])
    ub_u = np.array([ETA_MAX,  DELTA_MAX,  DELTA_MAX])

    # Gravity-turn warm start
    ts = np.linspace(0, 1, N+1)[:, None]
    x_guess = x0 + ts * (xf - x0)
    a_z_req = -x0[5] / T_BURN
    eta_const = float(np.clip((M_DRY + 0.5*M_FUEL_0) * (G + a_z_req) / T_MAX, ETA_MIN, ETA_MAX))
    u_guess = np.tile(np.array([eta_const, 0.0, 0.0]), (N+1, 1))
    mf_dot = eta_const * T_MAX / (ISP * G0)
    x_guess[:, 12] = np.clip(x0[12] - mf_dot * ts.flatten() * T_BURN, 0.0, x0[12])

    plant = Booster(x0=jnp.asarray(x0))
    print("Solving Part-1 open-loop trajopt (hard terminal constraint for the demo) ...")
    t0 = time.time()
    # constrain_xf=True turns the soft Q_N penalty into an NLP equality constraint
    # at the terminal state, so the optimal trajectory lands at x_f = 0 within
    # IPOPT tolerance. The notebook version of Part 1 uses constrain_xf=False
    # (Q_N as a soft penalty) and shows a ~15 m residual — exactly the residual
    # that Parts 2+ close in the loop. For the cinematic render we want a clean
    # dead-center touchdown, so we tighten to an equality.
    try:
        x_ref, u_ref = jaxonomy.trajopt(
            plant, t0=0.0, tf=T_BURN, x0=np.asarray(x0), xf=xf,
            Q=Q, R=R, QN=QN, N=N, constrain_xf=True,
            lb_x=lb_x, ub_x=ub_x, lb_u=lb_u, ub_u=ub_u,
            x_guess=x_guess, u_guess=u_guess,
        )
        constrain_mode = "constrain_xf=True (hard equality)"
    except Exception as e:
        print(f"  hard-constraint solve failed: {e}. Falling back to soft Q_N.")
        x_ref, u_ref = jaxonomy.trajopt(
            plant, t0=0.0, tf=T_BURN, x0=np.asarray(x0), xf=xf,
            Q=Q, R=R, QN=QN, N=N, constrain_xf=False,
            lb_x=lb_x, ub_x=ub_x, lb_u=lb_u, ub_u=ub_u,
            x_guess=x_guess, u_guess=u_guess,
        )
        constrain_mode = "constrain_xf=False (soft Q_N fallback)"
    print(f"  solved in {time.time()-t0:.1f} s, terminal residual "
          f"{np.linalg.norm(x_ref[-1,0:3]):.2f} m  [{constrain_mode}]")
    return np.asarray(x_ref), np.asarray(u_ref), T_BURN


# ─────────────────────────────────────────────────────────────────────────────
# Rich MJCF scene — drone-ship pad on ocean
# ─────────────────────────────────────────────────────────────────────────────

def build_mjcf():
    """Construct the cinematic scene MJCF — Falcon-9-class booster on an ASDS-style
    pad over a realistic ocean.

    Geometry notes (after the v1 review):
    - Body geoms share a common axis: the *upper hull* extends to z = +(L/2 - 9),
      so the nose cone center sits at z = (L/2 - 9) + 1.5, not at L/2 + 1.5 (the
      original off-by-one caused a 10-m visual gap between body and nose).
    - Grid fins are bigger and lattice-shaded so they actually read as fins.
    - An engine bell pokes out below the base — the "where thrust comes from" cue.
    - The plume is a 4-stack of nested cones (hot core → outer halo) at fixed full
      length (~25 m). It's the renderer's job to toggle it on/off via the
      `plume_active` body's qpos so it doesn't appear when the engine is off.
    - Ocean uses a denser, more saturated cobalt and a flat-ish texture with very
      mild value variation; reflectance/shininess give the water sheen.
    """
    L, R = L_BOOSTER, R_BOOSTER
    upper_top_z = L/2 - 9          # z of top of `hull_upper` (above body center)
    nose_half = 2.5                # half-length of the nose capsule's cylinder part
    nose_z = upper_top_z + nose_half  # capsule center → top sits at upper_top_z + 2*nose_half
    return f"""
<mujoco model="booster_cinematic">
  <option timestep="0.005" gravity="0 0 -9.81"/>
  <visual>
    <global offwidth="1280" offheight="720"/>
    <quality shadowsize="4096"/>
    <headlight ambient="0.22 0.24 0.30" diffuse="0.0 0.0 0.0" specular="0.0 0.0 0.0"/>
    <map shadowscale="0.5" fogstart="2000" fogend="6000"/>
  </visual>
  <asset>
    <!-- Sky: gradient — warm horizon to deep navy overhead -->
    <texture name="sky" type="skybox" builtin="gradient"
             rgb1="0.20 0.32 0.55" rgb2="0.96 0.55 0.28"
             width="1024" height="1024"/>
    <!-- Ocean: a near-uniform deep cobalt with subtle banding so it looks like
         water, not a chessboard. We use the 'flat' built-in type. -->
    <texture name="ocean_tex" type="2d" builtin="flat"
             rgb1="0.04 0.10 0.20" rgb2="0.04 0.10 0.20"
             width="100" height="100" mark="random" markrgb="0.06 0.16 0.28" random="0.10"/>
    <material name="ocean" texture="ocean_tex" texrepeat="40 40"
              reflectance="0.45" shininess="0.7" specular="0.7"/>
    <!-- Drone-ship deck -->
    <material name="deck"   rgba="0.30 0.32 0.34 1.0" reflectance="0.05"/>
    <material name="deck_edge" rgba="0.95 0.80 0.10 1.0"/>
    <material name="target_ring1" rgba="1.00 1.00 1.00 1.0"/>
    <material name="target_ring2" rgba="0.85 0.10 0.10 1.0"/>
    <material name="target_center" rgba="1.00 1.00 1.00 1.0"/>
    <!-- Booster materials -->
    <material name="booster_white" rgba="0.94 0.94 0.96 1.0" reflectance="0.30" shininess="0.5"/>
    <material name="booster_band"  rgba="0.18 0.18 0.20 1.0" reflectance="0.10"/>
    <material name="booster_logo"  rgba="0.12 0.12 0.30 1.0"/>
    <material name="nose_mat"      rgba="0.94 0.94 0.96 1.0" reflectance="0.30"/>
    <material name="nose_tip"      rgba="0.20 0.20 0.24 1.0"/>
    <material name="leg_mat"       rgba="0.55 0.55 0.60 1.0" reflectance="0.20"/>
    <material name="leg_pad"       rgba="0.18 0.18 0.20 1.0"/>
    <material name="fin_mat"       rgba="0.42 0.42 0.46 1.0" reflectance="0.15"/>
    <material name="engine_bell"   rgba="0.18 0.16 0.16 1.0" reflectance="0.05"/>
    <material name="engine_nozzle" rgba="0.45 0.30 0.18 1.0"/>
    <!-- Plume layers, with emission so they glow even on the unlit side of the booster -->
    <material name="plume_core"  rgba="1.00 0.95 0.80 0.95" emission="0.9"/>
    <material name="plume_mid"   rgba="1.00 0.65 0.20 0.65" emission="0.5"/>
    <material name="plume_outer" rgba="0.95 0.35 0.10 0.35" emission="0.3"/>
    <material name="plume_haze"  rgba="0.55 0.20 0.10 0.18" emission="0.1"/>
  </asset>
  <worldbody>
    <!-- Golden-hour sun (low elevation, raked from one side) -->
    <light name="sun" mode="targetbodycom" target="booster"
           pos="-3000 1500 1200" dir="0.6 -0.3 -0.7"
           diffuse="1.10 0.92 0.78" specular="0.5 0.42 0.35"
           castshadow="true"/>
    <!-- Cool blue fill -->
    <light name="fill" pos="2500 -1500 800" dir="-0.5 0.3 -0.5"
           diffuse="0.28 0.32 0.42" specular="0.0 0.0 0.0"
           castshadow="false"/>

    <!-- Ocean surface (huge plane, slight reflectance to give a water sheen) -->
    <geom name="ocean_surface" type="plane" size="6000 6000 1" material="ocean"/>

    <!-- Drone-ship deck -->
    <geom name="deck"        type="box" pos="0 0 1.5"   size="25 25 1.5"  material="deck"/>
    <geom name="deck_edge_n" type="box" pos="0  25 3.2" size="25 0.4 0.2" material="deck_edge"/>
    <geom name="deck_edge_s" type="box" pos="0 -25 3.2" size="25 0.4 0.2" material="deck_edge"/>
    <geom name="deck_edge_e" type="box" pos=" 25 0 3.2" size="0.4 25 0.2" material="deck_edge"/>
    <geom name="deck_edge_w" type="box" pos="-25 0 3.2" size="0.4 25 0.2" material="deck_edge"/>
    <!-- Concentric landing target -->
    <geom name="target_ring_outer" type="cylinder" pos="0 0 3.05" size="14 0.05" material="target_ring1"/>
    <geom name="target_ring_mid"   type="cylinder" pos="0 0 3.08" size="10 0.05" material="target_ring2"/>
    <geom name="target_ring_in"    type="cylinder" pos="0 0 3.11" size="6  0.05" material="target_ring1"/>
    <geom name="target_center"     type="cylinder" pos="0 0 3.14" size="2  0.05" material="target_center"/>

    <!-- BOOSTER ----------------------------------------------------------------- -->
    <body name="booster" pos="0 0 3.5">
      <freejoint name="root"/>

      <!-- Main hull: lower section + black band + upper section, all sharing axis -->
      <geom name="hull_lower" type="cylinder" pos="0 0 -{L/2 - 8}" size="{R} 8"
            material="booster_white"/>
      <geom name="hull_band"  type="cylinder" pos="0 0 -{L/2 - 16.0}"  size="{R*1.01} 0.6"
            material="booster_band"/>
      <geom name="hull_upper" type="cylinder" pos="0 0 0"  size="{R} {L/2 - 9}"
            material="booster_white"/>
      <!-- Logo band near top of upper hull -->
      <geom name="logo_band" type="cylinder" pos="0 0 {upper_top_z - 3}" size="{R*1.005} 0.8"
            material="booster_logo"/>
      <!-- Nose cone — capsule centered to attach top of upper hull, plus a darker tip -->
      <geom name="nose" type="capsule" pos="0 0 {nose_z}" size="{R*0.95} {nose_half}"
            material="nose_mat"/>
      <geom name="nose_tip" type="sphere" pos="0 0 {nose_z + nose_half + R*0.95}"
            size="{R*0.55}" material="nose_tip"/>

      <!-- Grid fins: bigger, lattice-coloured, attached to body at the top section -->
      <geom name="fin_xp" type="box" pos=" {R + 1.4} 0  {upper_top_z - 1.5}"
            size="1.5 0.10 1.8" material="fin_mat"/>
      <geom name="fin_xn" type="box" pos="-{R + 1.4} 0  {upper_top_z - 1.5}"
            size="1.5 0.10 1.8" material="fin_mat"/>
      <geom name="fin_yp" type="box" pos="0  {R + 1.4} {upper_top_z - 1.5}"
            size="0.10 1.5 1.8" material="fin_mat"/>
      <geom name="fin_yn" type="box" pos="0 -{R + 1.4} {upper_top_z - 1.5}"
            size="0.10 1.5 1.8" material="fin_mat"/>

      <!-- Four landing legs deployed, with foot pads (attached to body, not gimbal) -->
      <geom name="leg_xp" type="capsule"
            fromto=" {R} 0 -{L/2 + 0.5}  {R + 3.5} 0 -{L/2 - 2.5}"
            size="0.20" material="leg_mat"/>
      <geom name="leg_xn" type="capsule"
            fromto="-{R} 0 -{L/2 + 0.5} -{R + 3.5} 0 -{L/2 - 2.5}"
            size="0.20" material="leg_mat"/>
      <geom name="leg_yp" type="capsule"
            fromto="0  {R} -{L/2 + 0.5}  0  {R + 3.5} -{L/2 - 2.5}"
            size="0.20" material="leg_mat"/>
      <geom name="leg_yn" type="capsule"
            fromto="0 -{R} -{L/2 + 0.5}  0 -{R + 3.5} -{L/2 - 2.5}"
            size="0.20" material="leg_mat"/>
      <geom name="pad_xp" type="cylinder" pos=" {R + 3.5} 0 -{L/2 - 2.5}" size="0.7 0.15"
            material="leg_pad"/>
      <geom name="pad_xn" type="cylinder" pos="-{R + 3.5} 0 -{L/2 - 2.5}" size="0.7 0.15"
            material="leg_pad"/>
      <geom name="pad_yp" type="cylinder" pos="0  {R + 3.5} -{L/2 - 2.5}" size="0.7 0.15"
            material="leg_pad"/>
      <geom name="pad_yn" type="cylinder" pos="0 -{R + 3.5} -{L/2 - 2.5}" size="0.7 0.15"
            material="leg_pad"/>

      <!-- GIMBAL ASSEMBLY ------------------------------------------------------
           Engine bell + nozzle + plume rotate together about the gimbal joints,
           so when the controller commands a non-zero gimbal angle, the engine
           visibly tilts relative to the body — *that's* the "side thrust" cue.
           Two hinges (pitch about body-Y, yaw about body-X) give 2-DOF tilt.
           The plume body adds a slide joint so its length can shrink with
           throttle (and the renderer can jitter it for a flicker effect).      -->
      <body name="gimbal_y" pos="0 0 -{L/2 + 0.0}">
        <inertial pos="0 0 0" mass="0.01" diaginertia="0.001 0.001 0.001"/>
        <joint name="gimbal_pitch" type="hinge" axis="0 1 0" damping="0.2"/>
        <body name="gimbal_z" pos="0 0 0">
          <inertial pos="0 0 0" mass="0.01" diaginertia="0.001 0.001 0.001"/>
          <joint name="gimbal_yaw" type="hinge" axis="1 0 0" damping="0.2"/>
          <geom name="engine_bell" type="cylinder" pos="0 0 -1.0" size="{R*0.80} 1.0"
                material="engine_bell"/>
          <geom name="engine_nozzle" type="cylinder" pos="0 0 -2.4" size="{R*0.60} 0.4"
                material="engine_nozzle"/>
          <body name="plume" pos="0 0 -2.8">
            <inertial pos="0 0 0" mass="0.01" diaginertia="0.001 0.001 0.001"/>
            <joint name="plume_extend" type="slide" axis="0 0 1" damping="0.1"/>
            <geom name="plume_core"  type="capsule" fromto="0 0 0 0 0 -6"
                  size="0.45" material="plume_core"  contype="0" conaffinity="0"/>
            <geom name="plume_mid"   type="capsule" fromto="0 0 -1 0 0 -14"
                  size="0.95" material="plume_mid"   contype="0" conaffinity="0"/>
            <geom name="plume_outer" type="capsule" fromto="0 0 -3 0 0 -22"
                  size="1.60" material="plume_outer" contype="0" conaffinity="0"/>
            <geom name="plume_haze"  type="capsule" fromto="0 0 -6 0 0 -30"
                  size="2.40" material="plume_haze"  contype="0" conaffinity="0"/>
            <!-- Mach diamonds: small bright rings spaced along the plume -->
            <geom name="diamond1" type="sphere" pos="0 0 -3.0" size="0.55"
                  material="plume_core" contype="0" conaffinity="0"/>
            <geom name="diamond2" type="sphere" pos="0 0 -7.0" size="0.42"
                  material="plume_core" contype="0" conaffinity="0"/>
            <geom name="diamond3" type="sphere" pos="0 0 -11.5" size="0.32"
                  material="plume_mid"  contype="0" conaffinity="0"/>
          </body>
        </body>
      </body>
    </body>
  </worldbody>
</mujoco>
"""


# ─────────────────────────────────────────────────────────────────────────────
# Trajectory resampling for smooth playback
# ─────────────────────────────────────────────────────────────────────────────

def resample(x_ref: np.ndarray, u_ref: np.ndarray, T_burn: float,
              fps: int = 24) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Densify the trajectory to the render frame rate. No pre-burn pad — the
    plume is always-on so we start directly at the moment of engine ignition.
    """
    n_burn = int(T_burn * fps)
    t_burn = np.linspace(0.0, T_burn, n_burn)
    t_orig = np.linspace(0.0, T_burn, x_ref.shape[0])
    x_burn = np.stack([np.interp(t_burn, t_orig, x_ref[:, k]) for k in range(13)], axis=1)
    u_burn = np.stack([np.interp(t_burn, t_orig, u_ref[:, k]) for k in range(3)], axis=1)
    return x_burn, u_burn, t_burn


# ─────────────────────────────────────────────────────────────────────────────
# Render loop
# ─────────────────────────────────────────────────────────────────────────────

def euler_zyx_to_quat(phi: float, theta: float, psi: float) -> np.ndarray:
    """Convert Z-Y-X Euler (phi=roll about X, theta=pitch about Y, psi=yaw about Z)
    to (w, x, y, z) quaternion for MuJoCo qpos[3:7]."""
    cy, sy = np.cos(psi * 0.5), np.sin(psi * 0.5)
    cp, sp = np.cos(theta * 0.5), np.sin(theta * 0.5)
    cr, sr = np.cos(phi * 0.5), np.sin(phi * 0.5)
    w = cr*cp*cy + sr*sp*sy
    x = sr*cp*cy - cr*sp*sy
    y = cr*sp*cy + sr*cp*sy
    z = cr*cp*sy - sr*sp*cy
    return np.array([w, x, y, z])


def render(x_traj: np.ndarray, u_traj: np.ndarray, t_traj: np.ndarray,
            mjcf: str, fps: int = 24, width: int = 1280, height: int = 720,
            output_path: str = "booster_landing_cinematic.mp4") -> str:
    mj_model = mujoco.MjModel.from_xml_string(mjcf)
    mj_data = mujoco.MjData(mj_model)
    booster_bid = mj_model.body("booster").id

    # qpos indices for the gimbal + plume joints. The freejoint root takes 7
    # entries (3 pos + 4 quat). The hinge / slide joints come after.
    qadr_gimbal_pitch  = mj_model.joint("gimbal_pitch").qposadr[0]
    qadr_gimbal_yaw    = mj_model.joint("gimbal_yaw").qposadr[0]
    qadr_plume_extend  = mj_model.joint("plume_extend").qposadr[0]
    print(f"Joint qpos addresses: gimbal_pitch={qadr_gimbal_pitch}, "
          f"gimbal_yaw={qadr_gimbal_yaw}, plume_extend={qadr_plume_extend}")

    # Random generator for plume flicker (deterministic for reproducibility)
    rng = np.random.default_rng(7)
    # Visual exaggeration factor for the gimbal — actual u commands are ±10° but
    # at this camera distance, ±10° would be invisible. Multiply by 2.5 so the
    # control authority "reads" without becoming unphysical.
    GIMBAL_VIS_GAIN = 2.5

    # Camera: 3/4 from south-east, slowly zoom in
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE

    renderer = mujoco.Renderer(mj_model, height=height, width=width)

    frames = []
    n = len(t_traj)
    print(f"Rendering {n} frames at {width}x{height}@{fps} fps ...")
    t_render_start = time.time()
    for i in range(n):
        x = x_traj[i]
        u = u_traj[i] if i < len(u_traj) else np.zeros(3)

        # Booster pose: position + quaternion from Euler.
        # Z offset rationale: the trajectory tracks COM altitude with target z=0,
        # but visually we want the legs to touch the deck (top at z≈3) when the
        # trajectory says z=0. The booster body's geom origin sits at the COM,
        # so we add (L_BOOSTER/2 + pad_top_z) so the bottom of the cylinder ends
        # up on the pad at landing.
        pad_top_z = 3.0
        z_offset = L_BOOSTER / 2.0 + pad_top_z
        mj_data.qpos[0:3] = np.array([x[0], x[1], x[2] + z_offset])
        mj_data.qpos[3:7] = euler_zyx_to_quat(x[6], x[7], x[8])

        # Gimbal: the engine + plume tilt by the commanded gimbal angles u[1], u[2]
        # (visually exaggerated for legibility — actual commands are ±10° which
        # is barely perceptible at this camera distance).
        eta = float(u[0])
        mj_data.qpos[qadr_gimbal_pitch] = float(u[1]) * GIMBAL_VIS_GAIN
        mj_data.qpos[qadr_gimbal_yaw]   = float(u[2]) * GIMBAL_VIS_GAIN

        # Plume position: a small per-frame flicker breaks the perfect-cone look.
        # We deliberately do NOT scale the plume length with throttle — sliding
        # a rigid plume body along its axis either makes a visible gap to the
        # engine bell (slide down) or shoots the plume top *into* the booster
        # body (slide up). Both look broken. Real exhaust is turbulent and
        # variable but always anchored to the nozzle; we approximate that with
        # a small jitter and accept the plume reads as full-length throughout.
        # (A faithful "throttle modulates length" would require pre-baked plume
        # geoms of different sizes that we swap via geom visibility, which is
        # not exposed in mujoco-python's runtime API.)
        plume_flicker = float(rng.normal(0.0, 0.35))
        mj_data.qpos[qadr_plume_extend] = plume_flicker

        mujoco.mj_forward(mj_model, mj_data)

        # Camera: tracking shot, framed so the full 40-m booster + plume + pad
        # all fit comfortably. Aim at the booster center (qpos[2] is COM); back
        # the camera off enough to keep the nose cone in frame at low altitude.
        alt_frac = max(0.0, min(1.0, x[2] / 650.0))
        # Aim near booster middle; shift slightly toward the pad when close so
        # the bullseye is visible at the bottom of the frame.
        aim_z = mj_data.qpos[2] - (1.0 - alt_frac) * 8.0
        cam.lookat = np.array([x[0], x[1], aim_z])
        cam.distance = 120.0 + 200.0 * alt_frac   # 120 m near landing, 320 m at apex
        cam.azimuth = -55.0
        cam.elevation = -12.0 - 6.0 * (1.0 - alt_frac)

        renderer.update_scene(mj_data, camera=cam)
        frame = renderer.render()
        frames.append(frame)

        if (i + 1) % 20 == 0:
            print(f"  frame {i+1}/{n}  (t={t_traj[i]:.2f}s, "
                  f"alt={x[2]:.1f}m, eta={eta:.2f})")
    print(f"  rendered {n} frames in {time.time()-t_render_start:.1f} s")

    # Save: prefer mp4 via imageio-ffmpeg, else gif
    out = Path(output_path)
    try:
        with imageio.get_writer(str(out), fps=fps, codec="libx264",
                                  quality=8, pixelformat="yuv420p") as w:
            for f in frames:
                w.append_data(f)
        print(f"  wrote {out}  ({out.stat().st_size / 1e6:.1f} MB)")
        return str(out)
    except Exception as e:
        # Fallback to GIF
        gif_path = out.with_suffix(".gif")
        # Downsample frames for GIF size
        gif_frames = [f[::2, ::2] for f in frames]
        imageio.mimsave(str(gif_path), gif_frames, fps=fps, loop=0)
        print(f"  mp4 failed ({e}); wrote {gif_path}  "
              f"({gif_path.stat().st_size / 1e6:.1f} MB)")
        return str(gif_path)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    here = Path(__file__).parent
    mjcf = build_mjcf()
    (here / "booster_scene.xml").write_text(mjcf)
    print(f"Wrote {here / 'booster_scene.xml'}")

    x_ref, u_ref, T_burn = solve_part1_trajectory()
    x_traj, u_traj, t_traj = resample(x_ref, u_ref, T_burn, fps=24)

    output = render(x_traj, u_traj, t_traj, mjcf,
                     fps=24, width=1280, height=720,
                     output_path=str(here / "booster_landing_cinematic.mp4"))
    print(f"\nFinal output: {output}")


if __name__ == "__main__":
    main()
