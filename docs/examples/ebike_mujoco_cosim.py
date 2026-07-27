"""3-D MuJoCo cargo e-bike co-simulation — high-fidelity companion to the
planar jaxonomy e-bike (``ebike_hybrid_simulation.py``).

The jaxonomy model is a 3-DOF *bird's-eye* planar vehicle (longitudinal,
lateral, yaw): it can tell you speed and battery state, but it structurally
cannot see the *sagittal* plane — how the bike squats and pitches under drive
torque, how the suspension travels over a bump, how load transfers between the
front and rear contact patches, or what the real tyre-ground contact does. This
script puts exactly those dynamics under a proper contact solver.

MuJoCo is the *truth model* (Mission B / decision-tree path 2): a Python control
loop reads state, applies a jaxonomy-style rear-wheel assist torque plus an
aerodynamic drag force MuJoCo does not model, and steps the physics. Roll is not
solved — the frame rides a sagittal-plane base (x-slide + z-heave + pitch), so
the bike is upright by construction (see the MJCF header comment).

Run headless (records traces + NPZ)::

    python docs/examples/ebike_mujoco_cosim.py

Run with a rendered fly-by (writes mp4 + gif)::

    python docs/examples/ebike_mujoco_cosim.py --render

Outputs land in ``docs/examples/media/``:
    ebike_mujoco_cosim.npz  — traces (t, speed_mps, pitch_rad, wheel_omega,
                              susp_travel_m, drive_torque_Nm, plus extras)
    ebike_mujoco_cosim.mp4 / .gif  — 3/4 follow-camera render (with --render)
"""
from __future__ import annotations

import argparse
import os
import time

import numpy as np
import mujoco

HERE = os.path.dirname(os.path.abspath(__file__))
MEDIA = os.path.join(HERE, "media")
XML = os.path.join(MEDIA, "ebike_mujoco.xml")


# ---------------------------------------------------------------------------
# Assist torque law — mirrors the jaxonomy assist policy: constant human pedal
# torque plus a speed-faded motor assist, cut at the EU legal limit (25 km/h).
# Returned value is torque AT THE WHEEL [Nm].
# ---------------------------------------------------------------------------
V_CUT = 6.94  # 25 km/h in m/s


def wheel_drive_torque(v_mps):
    fade = np.clip((V_CUT - v_mps) / 0.55, 0.0, 1.0)  # assist fades out by 25 km/h
    human = 25.0          # Nm at the wheel (rider)
    assist = 30.0 * fade  # Nm at the wheel (motor), cut at the legal limit
    return human + assist


# Aerodynamic drag (MuJoCo models no fluid here): F = -0.5*rho*CdA*v*|v|
RHO_AIR = 1.2
CDA = 0.8


# ---------------------------------------------------------------------------
# Terrain: flat accel -> speed bump -> descent (pushes speed past the legal
# cutoff so assist visibly cuts) -> climb. A 1-D height profile h(x) tiled
# across the heightfield rows. Descents mean h(x) can go negative; the loader
# shifts/normalises so the flat baseline sits at world z=0.
# ---------------------------------------------------------------------------
BUMP_X = 11.0       # speed-bump centre [m]
BUMP_HALFWIDTH = 0.5
BUMP_HEIGHT = 0.09  # crest height [m]

DESC_X0 = 23.0      # descent foot [m]
DESC_LEN = 10.0
DESC_DROP = 1.40    # drop over the descent [m] (~6% grade)

INC_X0 = 39.0       # climb foot [m]
INC_LEN = 10.0
INC_RISE = 0.70     # climb height [m] (~6.5% grade, matches jaxonomy grade_max)

X_CENTER = 30.0     # hfield geom x-centre (see MJCF pos)


def _smoothstep(s):
    s = np.clip(s, 0.0, 1.0)
    return s * s * (3.0 - 2.0 * s)


def road_height(x):
    """Road elevation [m] vs world x (vectorised). Flat baseline == 0."""
    x = np.asarray(x, dtype=float)
    h = np.zeros_like(x)

    # 1-cosine speed bump (compact, smooth)
    d = x - BUMP_X
    in_bump = np.abs(d) <= BUMP_HALFWIDTH
    h = np.where(in_bump,
                 0.5 * BUMP_HEIGHT * (1.0 + np.cos(np.pi * d / BUMP_HALFWIDTH)),
                 h)

    # descent then climb (smoothstep ramps)
    h = h - DESC_DROP * _smoothstep((x - DESC_X0) / DESC_LEN)
    h = h + INC_RISE * _smoothstep((x - INC_X0) / INC_LEN)
    return h


def load_model(cargo_mass=None):
    """Load the MJCF and inject the heightfield road profile.

    The heightfield stores normalised heights in [0,1]; we set its elevation
    size and z-offset programmatically so the true profile (including the
    below-baseline descent) is reproduced and the flat start sits at world z=0.

    ``cargo_mass`` (kg), if given, overrides the rear payload mass (default 58 kg)
    so callers can study weight transfer vs load.
    """
    if cargo_mass is None:
        model = mujoco.MjModel.from_xml_path(XML)
    else:
        with open(XML) as _f:
            _xml = _f.read().replace('mass="58"', f'mass="{float(cargo_mass)}"')
        model = mujoco.MjModel.from_xml_string(_xml)
    hid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_HFIELD, "road")
    gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    nrow = int(model.hfield_nrow[hid])
    ncol = int(model.hfield_ncol[hid])
    size_x = model.hfield_size[hid, 0]
    xs = np.linspace(X_CENTER - size_x, X_CENTER + size_x, ncol)
    prof = road_height(xs)

    # elevation scale is fixed by the XML (compiled collision hull); do NOT
    # resize it here or the flat baseline falls outside the hull and wheels sink.
    elev = float(model.hfield_size[hid, 2])
    base = float(prof.min())
    relief = float(prof.max() - base)
    if relief > elev:
        raise ValueError(f"road relief {relief:.2f} m exceeds hfield envelope "
                         f"{elev:.2f} m; enlarge <hfield size> z in the MJCF")
    model.geom_pos[gid, 2] = base                      # so world z == prof(x)
    data_n = (prof - base) / elev
    model.hfield_data[:] = np.tile(data_n, nrow).astype(np.float64)
    return model


# ---------------------------------------------------------------------------
# Co-simulation
# ---------------------------------------------------------------------------
def run(t_end=15.0, sample_hz=100.0, settle=0.5, seed_speed=0.0, cargo_mass=None):
    """Run the closed-loop co-sim; return a dict of numpy traces."""
    model = load_model(cargo_mass=cargo_mass)
    data = mujoco.MjData(model)

    # index helpers -------------------------------------------------------
    def jadr(name):
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        return model.jnt_qposadr[jid], model.jnt_dofadr[jid]

    qx, vx = jadr("slide_x")
    qpitch, vpitch = jadr("pitch")
    qrs, vrs = jadr("rear_spin")
    qrsusp, _ = jadr("rear_susp")
    qfsusp, _ = jadr("front_susp")
    frame_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "frame")
    act_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "rear_drive")

    # let the bike settle onto its suspension under gravity (no drive) -----
    n_settle = int(settle / model.opt.timestep)
    for _ in range(n_settle):
        mujoco.mj_step(model, data)
    # zero out any settling drift in the longitudinal coordinate/velocity
    data.qpos[qx] = 0.0
    data.qvel[vx] = seed_speed
    mujoco.mj_forward(model, data)
    # record the rest-pose suspension positions so travel is relative to rest
    rear_rest = data.qpos[qrsusp]
    front_rest = data.qpos[qfsusp]

    dt = model.opt.timestep
    n_steps = int(t_end / dt)
    sample_every = max(1, int(round(1.0 / (sample_hz * dt))))

    t_log, v_log, pitch_log, omega_log = [], [], [], []
    rear_travel_log, front_travel_log, torque_log = [], [], []
    x_log, road_log, assist_on_log = [], [], []

    for k in range(n_steps):
        v = float(data.qvel[vx])  # longitudinal speed [m/s]

        # drive torque from the assist law -> actuator
        tau = float(wheel_drive_torque(v))
        data.ctrl[act_id] = tau

        # aerodynamic drag: external horizontal force on the frame body,
        # opposing motion (world frame; reset every step)
        data.xfrc_applied[frame_bid, :] = 0.0
        data.xfrc_applied[frame_bid, 0] = -0.5 * RHO_AIR * CDA * v * abs(v)

        if k % sample_every == 0:
            t_log.append(k * dt)
            v_log.append(v)
            pitch_log.append(float(data.qpos[qpitch]))
            omega_log.append(float(data.qvel[vrs]))
            rear_travel_log.append(float(data.qpos[qrsusp] - rear_rest))
            front_travel_log.append(float(data.qpos[qfsusp] - front_rest))
            torque_log.append(tau)
            x_log.append(float(data.qpos[qx]))
            road_log.append(float(road_height(data.qpos[qx])))
            assist_on_log.append(1.0 if v < V_CUT else 0.0)

        mujoco.mj_step(model, data)

    return {
        "t": np.array(t_log),
        "speed_mps": np.array(v_log),
        "pitch_rad": np.array(pitch_log),
        "wheel_omega": np.array(omega_log),
        "susp_travel_m": np.array(rear_travel_log),      # rear (driven) suspension
        "front_travel_m": np.array(front_travel_log),
        "drive_torque_Nm": np.array(torque_log),
        "x_m": np.array(x_log),
        "road_h_m": np.array(road_log),
        "assist_on": np.array(assist_on_log),
    }


# ---------------------------------------------------------------------------
# Rendering — side-chase tracking camera, MuJoCo-in-the-loop (re-runs the same
# law). A programmatic free camera that tracks the bike's centre of mass gives
# a clean near-side-profile 3/4 view (both wheels read as full circles; the
# pitch/heave/suspension are read directly), avoiding a front-facing angle
# where the coarse heightfield mesh can occlude the near wheel.
# ---------------------------------------------------------------------------
def render(out_stem, t_end=15.0, fps=30, width=1280, height=720,
           settle=0.5, azimuth=100.0, elevation=-13.0, distance=5.2):
    import imageio.v2 as imageio

    model = load_model()
    data = mujoco.MjData(model)

    def jadr(name):
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        return model.jnt_qposadr[jid], model.jnt_dofadr[jid]

    qx, vx = jadr("slide_x")
    frame_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "frame")
    act_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "rear_drive")

    n_settle = int(settle / model.opt.timestep)
    for _ in range(n_settle):
        mujoco.mj_step(model, data)
    data.qpos[qx] = 0.0
    data.qvel[vx] = 0.0
    mujoco.mj_forward(model, data)

    dt = model.opt.timestep
    n_steps = int(t_end / dt)
    frame_every = max(1, int(round(1.0 / (fps * dt))))

    renderer = mujoco.Renderer(model, height=height, width=width)
    scene_opt = mujoco.MjvOption()
    scene_opt.flags[mujoco.mjtVisFlag.mjVIS_PERTFORCE] = True  # draw the drag arrow
    cam = mujoco.MjvCamera()
    cam.azimuth, cam.elevation, cam.distance = azimuth, elevation, distance

    frames = []
    try:
        for k in range(n_steps):
            v = float(data.qvel[vx])
            data.ctrl[act_id] = float(wheel_drive_torque(v))
            data.xfrc_applied[frame_bid, :] = 0.0
            data.xfrc_applied[frame_bid, 0] = -0.5 * RHO_AIR * CDA * v * abs(v)
            if k % frame_every == 0:
                com = data.subtree_com[frame_bid]
                cam.lookat[:] = (com[0], com[1], 0.42)  # steady height, follow x
                renderer.update_scene(data, camera=cam, scene_option=scene_opt)
                frames.append(renderer.render())
            mujoco.mj_step(model, data)
    finally:
        renderer.close()

    mp4 = out_stem + ".mp4"
    imageio.mimsave(mp4, frames, fps=fps, quality=8, macro_block_size=None)
    print(f"wrote {mp4}  ({len(frames)} frames, {os.path.getsize(mp4)/1e6:.1f} MB)")

    # a compact downsampled gif for quick preview (kept under ~5 MB)
    gif = out_stem + ".gif"
    gif_frames = [f[::3, ::3] for f in frames[::3]]   # ~427 px wide, 10 fps
    imageio.mimsave(gif, gif_frames, fps=max(1, fps // 3), loop=0)
    print(f"wrote {gif}  ({len(gif_frames)} frames, {os.path.getsize(gif)/1e6:.1f} MB)")
    return frames


def _print_summary(tr):
    v = tr["speed_mps"]
    print("\n--- co-sim summary ---")
    print(f"  duration            : {tr['t'][-1]:.1f} s, {len(tr['t'])} samples")
    print(f"  top speed           : {v.max():.2f} m/s ({v.max()*3.6:.1f} km/h)")
    print(f"  distance travelled  : {tr['x_m'][-1]:.1f} m")
    print(f"  pitch range         : [{np.degrees(tr['pitch_rad'].min()):+.2f},"
          f" {np.degrees(tr['pitch_rad'].max()):+.2f}] deg")
    print(f"  rear susp travel    : [{tr['susp_travel_m'].min()*1e3:+.0f},"
          f" {tr['susp_travel_m'].max()*1e3:+.0f}] mm")
    print(f"  wheel omega max     : {tr['wheel_omega'].max():.1f} rad/s"
          f" (=> {tr['wheel_omega'].max()*0.29:.2f} m/s surface)")
    idx_cut = np.argmax(v >= V_CUT) if (v >= V_CUT).any() else -1
    if idx_cut > 0:
        print(f"  reaches 25 km/h cut : t={tr['t'][idx_cut]:.2f} s,"
              f" x={tr['x_m'][idx_cut]:.1f} m (assist off after)")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--render", action="store_true", help="render mp4 + gif fly-by")
    ap.add_argument("--t-end", type=float, default=15.0)
    ap.add_argument("--azimuth", type=float, default=100.0,
                    help="chase-camera azimuth [deg]")
    args = ap.parse_args()

    t0 = time.time()
    tr = run(t_end=args.t_end)
    print(f"co-sim: {time.time()-t0:.2f} s wall for {args.t_end:.0f} s sim")
    _print_summary(tr)

    npz = os.path.join(MEDIA, "ebike_mujoco_cosim.npz")
    np.savez(npz, **tr)
    print(f"wrote {npz}")

    if args.render:
        t0 = time.time()
        render(os.path.join(MEDIA, "ebike_mujoco_cosim"),
               t_end=args.t_end, azimuth=args.azimuth)
        print(f"render: {time.time()-t0:.2f} s wall")


if __name__ == "__main__":
    main()
