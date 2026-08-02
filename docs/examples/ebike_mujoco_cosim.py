"""3-D MuJoCo cargo e-bike — sagittal-plane companion to the planar jaxonomy
e-bike (``ebike_hybrid_simulation.py``).

The jaxonomy model is a 3-DOF *bird's-eye* planar vehicle (longitudinal,
lateral, yaw): it can tell you speed and battery state, but it structurally
cannot see the *sagittal* plane — how the bike squats and pitches under drive
torque, how the suspension travels over a bump, how load transfers between the
front and rear contact patches. This script puts exactly those dynamics under a
proper contact solver.

This is **controller-in-the-loop simulation**, not co-simulation: one solver
(MuJoCo), with a Python control law evaluated inside its stepping loop at
500 Hz. The controller reads the model's *named sensors* — the wiring harness a
hardware-in-the-loop controller would be limited to — applies a rear-wheel
assist torque mirroring the planar model's drivetrain, plus an aerodynamic drag
force MuJoCo does not model. Roll is not solved: the frame rides a
sagittal-plane base (x-slide + z-heave + pitch), so the bike is upright by
construction (see the MJCF header comment).

Sign conventions, fixed once here so no caption can invert them: **pitch is
nose-up positive** (the raw MJCF hinge is nose-down positive; the logging layer
negates it), and **suspension travel is positive in compression** (jounce).

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
# Assist torque law — a quantitative mirror of the planar model's drivetrain,
# reflected to the wheel. Returned value is torque AT THE WHEEL [Nm].
#
#   motor: Part 1 caps assist at 12 Nm at the motor shaft; the 2.5:1 chain
#     puts 30 Nm at the wheel, faded over the same 2 km/h band to the cutoff.
#   human: Part 1's rider produces ~15-18 Nm at the crank with a cadence
#     taper; the crank turns at half wheel speed, so that is ~8 Nm at the
#     wheel (~110 W at cruise -- a real rider, not the 25 Nm/600 W
#     track-sprinter an earlier version of this file assumed).
# ---------------------------------------------------------------------------
V_CUT = 6.94  # 25 km/h in m/s


def wheel_drive_torque(v_mps):
    fade = np.clip((V_CUT - v_mps) / 0.55, 0.0, 1.0)  # assist fades out by 25 km/h
    human = 8.0           # Nm at the wheel (rider; see header note)
    assist = 30.0 * fade  # Nm at the wheel (motor), cut at the legal limit
    return human + assist


# Aerodynamic drag (MuJoCo models no fluid here): F = -0.5*rho*CdA*v*|v|
RHO_AIR = 1.2
CDA = 0.8


# ---------------------------------------------------------------------------
# Terrain: flat accel -> speed bump -> 6% descent (pushes speed past the legal
# cutoff so assist visibly cuts) -> 6% climb. Built from flat/ramp BOX
# segments plus a transverse half-buried capsule as the speed bump (see the
# MJCF road comment: a heightfield's triangulated surface gives a rolling
# wheel multiple disagreeing contact normals whose micro-slip dissipates ~5x
# the intended rolling resistance; box faces are exact planes and roll clean).
# Ramps are piecewise-LINEAR at a constant 6% -- the same grade as the planar
# model's route (grade_max = 0.06) -- with kinks at the joints, like real
# pavement transitions.
# ---------------------------------------------------------------------------
# Speed hump: a circle-segment profile ~3 m wide x 9 cm high (the standard
# neighborhood traffic-calming hump geometry). A narrow parking-lot *bump*
# (1 m wide) at this height lofts the driven wheel at ~11 km/h -- 0.7 s of
# airborne wheelspin under full assist torque -- which is real physics but a
# suspension-response demo needs ground contact.
#
# It is built from short tilted BOXES following the arc, not from one buried
# capsule: the capsule that would expose a 9 cm crest over a 3 m chord needs a
# 12.55 m radius, i.e. a 25 m-wide black cylinder that swallows the camera and
# hides the bike in every render. Boxes keep the geometry the size of the
# feature it represents.
BUMP_X = 11.0         # hump centre [m]
BUMP_HEIGHT = 0.09    # crest height above road [m]
BUMP_HALFWIDTH = 1.5  # half-chord [m] -> 3 m long hump
# Radius of the circular arc through (+/-halfwidth, 0) and (0, height)
BUMP_RADIUS = (BUMP_HALFWIDTH**2 + BUMP_HEIGHT**2) / (2 * BUMP_HEIGHT)
BUMP_NSEG = 12        # boxes approximating the arc (sagitta error < 1 mm)

GRADE = 0.06         # constant ramp grade (matches planar grade_max)
DESC_X0 = 23.0       # descent foot [m]
DESC_LEN = 25.0
DESC_DROP = GRADE * DESC_LEN   # 1.5 m

INC_X0 = 55.0        # climb foot [m]
INC_LEN = 15.0
INC_RISE = GRADE * INC_LEN     # 0.9 m

X_MIN, X_MAX = -30.0, 95.0     # road extent


def road_height(x, terrain=True):
    """Road elevation [m] vs world x (vectorised, piecewise linear + bump).
    Flat baseline == 0. ``terrain=False`` returns the flat road used for the
    like-for-like longitudinal comparison against the planar model."""
    x = np.asarray(x, dtype=float)
    h = np.zeros_like(x)
    if not terrain:
        return h

    # ramps (piecewise linear, constant 6%)
    h = h - GRADE * np.clip(x - DESC_X0, 0.0, DESC_LEN)
    h = h + GRADE * np.clip(x - INC_X0, 0.0, INC_LEN)

    # speed hump: circular arc through (+/-halfwidth, 0) and (0, height),
    # zero outside the chord
    d = np.abs(x - BUMP_X)
    r2 = BUMP_RADIUS**2 - d**2
    arc = np.sqrt(np.clip(r2, 0.0, None)) - (BUMP_RADIUS - BUMP_HEIGHT)
    h = h + np.where(d <= BUMP_HALFWIDTH, np.clip(arc, 0.0, None), 0.0)
    return h


def road_slope(x, terrain=True, dx=0.05):
    """dh/dx by central difference (point slope of the road surface)."""
    return (road_height(np.asarray(x) + dx, terrain)
            - road_height(np.asarray(x) - dx, terrain)) / (2 * dx)


WHEELBASE = 1.24  # front/rear axle separation [m] (forks at +/-0.62 in the MJCF)


def road_pitch(x, terrain=True):
    """The road's *geometric* pitch as the bike experiences it: the angle of
    the chord between the two contact patches, atan((h_front - h_rear)/L).
    This -- not the point slope dh/dx -- is the right reference for splitting
    total chassis pitch into "terrain following" and "suspension response":
    over the speed bump the point slope is far steeper than what a 1.24 m
    wheelbase can geometrically pitch."""
    x = np.asarray(x, dtype=float)
    h_f = road_height(x + WHEELBASE / 2, terrain)
    h_r = road_height(x - WHEELBASE / 2, terrain)
    return np.arctan((h_f - h_r) / WHEELBASE)


def _road_geoms_xml(terrain=True):
    """MJCF geoms for the road: flat/ramp boxes (+ the bump capsule)."""
    fr = 'friction="1.3 0.01 0.0023" condim="6"'
    thick = 0.5   # box half-thickness [m]
    half_w = 9.0  # road half-width [m]
    if not terrain:
        return (f'<geom name="road_flat" type="box" material="roadmat" {fr} '
                f'pos="{(X_MIN+X_MAX)/2} 0 {-thick}" size="{(X_MAX-X_MIN)/2} {half_w} {thick}"/>')

    segs = []   # (x0, x1, h0, h1)
    segs.append((X_MIN, DESC_X0, 0.0, 0.0))
    segs.append((DESC_X0, DESC_X0 + DESC_LEN, 0.0, -DESC_DROP))
    segs.append((DESC_X0 + DESC_LEN, INC_X0, -DESC_DROP, -DESC_DROP))
    segs.append((INC_X0, INC_X0 + INC_LEN, -DESC_DROP, -DESC_DROP + INC_RISE))
    segs.append((INC_X0 + INC_LEN, X_MAX, -DESC_DROP + INC_RISE, -DESC_DROP + INC_RISE))

    out = []
    for i, (x0, x1, h0, h1) in enumerate(segs):
        L = np.hypot(x1 - x0, h1 - h0)
        ang = np.arctan2(h1 - h0, x1 - x0)       # slope angle about +y (nose frame)
        cx, cz = (x0 + x1) / 2, (h0 + h1) / 2
        # push the box centre half a thickness along the downward surface normal
        nx, nz = np.sin(ang), -np.cos(ang)       # unit normal pointing down
        cx += nx * thick
        cz += nz * thick
        # MuJoCo euler about y is nose-down positive for +x travel: angle -ang
        out.append(f'<geom name="road_seg{i}" type="box" material="roadmat" {fr} '
                   f'pos="{cx:.4f} 0 {cz:.4f}" size="{L/2:.4f} {half_w} {thick}" '
                   f'euler="0 {-np.degrees(ang):.4f} 0"/>')
    # speed hump: short tilted boxes following the arc (see the constants note
    # -- one buried capsule of the required radius would be 25 m across)
    xs = np.linspace(BUMP_X - BUMP_HALFWIDTH, BUMP_X + BUMP_HALFWIDTH, BUMP_NSEG + 1)
    hs = road_height(xs, terrain=True)
    for i in range(BUMP_NSEG):
        x0, x1, h0, h1 = xs[i], xs[i + 1], hs[i], hs[i + 1]
        L = np.hypot(x1 - x0, h1 - h0)
        ang = np.arctan2(h1 - h0, x1 - x0)
        cx, cz = (x0 + x1) / 2, (h0 + h1) / 2
        nx, nz = np.sin(ang), -np.cos(ang)
        cx += nx * thick
        cz += nz * thick
        out.append(f'<geom name="road_hump{i}" type="box" material="darkmat" {fr} '
                   f'pos="{cx:.4f} 0 {cz:.4f}" size="{L/2:.4f} {half_w} {thick}" '
                   f'euler="0 {-np.degrees(ang):.4f} 0"/>')
    return "\n    ".join(out)


def load_model(cargo_mass=None, terrain=True):
    """Load the MJCF and inject the road as box/capsule geoms.

    ``cargo_mass`` (kg), if given, overrides the rear payload mass (default 58 kg)
    so callers can study weight transfer vs load. ``terrain=False`` builds a
    flat road (for like-for-like comparison runs against the planar model).

    Consistency guard: the analytic ``road_height`` used for logging/pitch
    decomposition and the injected geoms come from the same constants, and a
    probe check below verifies the compiled model's surface matches the
    analytic profile at several x (so the two can never drift apart).
    """
    with open(XML) as _f:
        _xml = _f.read()
    if cargo_mass is not None:
        _xml = _xml.replace('mass="58"', f'mass="{float(cargo_mass)}"')
    placeholder = next(l for l in _xml.splitlines() if 'road_placeholder' in l and '<geom' in l)
    # the placeholder is a two-line geom element; replace from '<geom name="road_placeholder"'
    start = _xml.index('<geom name="road_placeholder"')
    end = _xml.index('/>', start) + 2
    _xml = _xml[:start] + _road_geoms_xml(terrain=terrain) + _xml[end:]
    model = mujoco.MjModel.from_xml_string(_xml)

    # probe: drop a ray at several x and compare the road surface to the
    # analytic profile (excluding the bump capsule's material difference)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    for x_probe in (-5.0, 30.0, 47.9, 62.0, 80.0):
        pnt = np.array([x_probe, 0.0, 5.0])
        geomid = np.zeros(1, dtype=np.int32)
        frac = mujoco.mj_ray(model, data, pnt, np.array([0.0, 0.0, -1.0]),
                             None, 1, -1, geomid)
        z_hit = 5.0 - frac
        z_ref = float(road_height(x_probe, terrain))
        if abs(z_hit - z_ref) > 2e-3:
            raise AssertionError(f"road geom/profile mismatch at x={x_probe}: "
                                 f"surface {z_hit:.4f} vs analytic {z_ref:.4f}")
    return model


# ---------------------------------------------------------------------------
# Co-simulation
# ---------------------------------------------------------------------------
def run(t_end=15.0, sample_hz=100.0, settle=3.0, seed_speed=0.0, cargo_mass=None,
        terrain=True):
    """Run the closed-loop co-sim; return a dict of numpy traces.

    Conventions: every pitch value returned is **nose-up positive** (the MJCF
    hinge about +y is nose-down positive by the right-hand rule; the sensor
    read is negated here, once, so no downstream caption can get it backwards).
    ``pitch_rel_rad`` = total pitch minus the road's geometric pitch
    atan(dh/dx) -- the chassis's own squat/dive/suspension response, which is
    the part a planar model actually lacks.

    ``settle`` defaults to 3 s: the rear suspension mode is ~2.4 Hz at
    zeta ~ 0.24, so 0.5 s of settling left a ~16% residual transient inside
    the "rest pose" reference. All signals are read from the MJCF's named
    sensors (the wiring harness a HIL controller would see), not raw qpos.
    """
    model = load_model(cargo_mass=cargo_mass, terrain=terrain)
    data = mujoco.MjData(model)

    def sens(name):
        return float(data.sensor(name).data[0])

    frame_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "frame")
    act_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "rear_drive")
    jid_x = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "slide_x")
    qx, vx = model.jnt_qposadr[jid_x], model.jnt_dofadr[jid_x]

    # let the bike settle onto its suspension under gravity (no drive) -----
    n_settle = int(settle / model.opt.timestep)
    for _ in range(n_settle):
        mujoco.mj_step(model, data)
    # zero out any settling drift in the longitudinal coordinate/velocity
    data.qpos[qx] = 0.0
    data.qvel[vx] = seed_speed
    mujoco.mj_forward(model, data)
    # Record the rest-pose suspension positions so *travel* is relative to
    # rest. Keep the absolute values too: travel-from-rest deliberately
    # cancels static sag (each payload settles to its own rest pose), so a
    # payload study that only reads travel measures nothing about sag -- it
    # needs these absolute positions.
    rear_rest = sens("rear_travel")
    front_rest = sens("front_travel")
    heave_rest = sens("heave")

    dt = model.opt.timestep
    n_steps = int(t_end / dt)
    sample_every = max(1, int(round(1.0 / (sample_hz * dt))))

    t_log, v_log, pitch_log, omega_log = [], [], [], []
    rear_travel_log, front_travel_log, torque_log = [], [], []
    x_log, road_log, slope_log, heave_log, assist_on_log = [], [], [], [], []
    E_drive = 0.0   # \int tau * omega dt   (wheel drive energy)
    E_aero = 0.0    # \int F_drag * v dt    (aero dissipation)

    for k in range(n_steps):
        v = sens("v_long")  # longitudinal speed [m/s]

        # drive torque from the assist law -> actuator
        tau = float(wheel_drive_torque(v))
        data.ctrl[act_id] = tau

        # aerodynamic drag: external horizontal force on the frame body,
        # opposing motion (world frame; reset every step). Applied at the
        # frame CoM -- so it carries no aero pitch moment (a limitation the
        # walkthrough notes).
        F_drag = -0.5 * RHO_AIR * CDA * v * abs(v)
        data.xfrc_applied[frame_bid, :] = 0.0
        data.xfrc_applied[frame_bid, 0] = F_drag

        w_rear = sens("rear_wvel")
        E_drive += tau * w_rear * dt
        E_aero += -F_drag * v * dt

        if k % sample_every == 0:
            x = sens("x_pos")
            t_log.append(k * dt)
            v_log.append(v)
            pitch_log.append(-sens("pitch_pos"))          # nose-up positive
            omega_log.append(w_rear)
            rear_travel_log.append(sens("rear_travel") - rear_rest)
            front_travel_log.append(sens("front_travel") - front_rest)
            torque_log.append(tau)
            x_log.append(x)
            road_log.append(float(road_height(x, terrain)))
            slope_log.append(float(road_slope(x, terrain)))
            heave_log.append(sens("heave") - heave_rest)
            assist_on_log.append(1.0 if v < V_CUT else 0.0)

        mujoco.mj_step(model, data)

    pitch = np.array(pitch_log)
    slope = np.array(slope_log)
    x_arr = np.array(x_log)
    rp = road_pitch(x_arr, terrain)
    return {
        "t": np.array(t_log),
        "speed_mps": np.array(v_log),
        "pitch_rad": pitch,                              # nose-up positive
        "road_pitch_rad": rp,                            # wheelbase-chord road pitch
        "pitch_rel_rad": pitch - rp,                     # chassis response only
        "wheel_omega": np.array(omega_log),
        "susp_travel_m": np.array(rear_travel_log),      # rear (driven); + = compression
        "front_travel_m": np.array(front_travel_log),
        "heave_m": np.array(heave_log),
        "drive_torque_Nm": np.array(torque_log),
        "x_m": x_arr,
        "road_h_m": np.array(road_log),
        "road_slope": slope,
        "assist_on": np.array(assist_on_log),
        "E_drive_J": np.array(E_drive),
        "E_aero_J": np.array(E_aero),
        "total_mass_kg": np.array(float(model.body_subtreemass[frame_bid])),
        # absolute settled positions (static sag lives here, not in *_travel_m)
        "rear_static_m": np.array(rear_rest),
        "front_static_m": np.array(front_rest),
        "heave_static_m": np.array(heave_rest),
    }


# ---------------------------------------------------------------------------
# Rendering — side-chase tracking camera, MuJoCo-in-the-loop (re-runs the same
# law). A programmatic free camera that tracks the bike's centre of mass gives
# a clean near-side-profile 3/4 view (both wheels read as full circles; the
# pitch/heave/suspension are read directly), avoiding a front-facing angle
# where the coarse heightfield mesh can occlude the near wheel.
# ---------------------------------------------------------------------------
def render(out_stem, t_end=15.0, fps=30, width=1280, height=720,
           settle=3.0, azimuth=100.0, elevation=-13.0, distance=5.2):
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
                # Follow the ROAD height, not a fixed world height: the route
                # drops 1.5 m on the descent, and a camera pinned to z = 0.42
                # would leave the bike sliding out of the bottom of the frame.
                cam.lookat[:] = (com[0], com[1],
                                 float(road_height(com[0])) + 0.42)
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


def _print_summary(tr, label="terrain"):
    v = tr["speed_mps"]
    print(f"\n--- co-sim summary ({label}) ---")
    print(f"  duration            : {tr['t'][-1]:.1f} s, {len(tr['t'])} samples")
    print(f"  total mass          : {float(tr['total_mass_kg']):.1f} kg")
    print(f"  top speed           : {v.max():.2f} m/s ({v.max()*3.6:.1f} km/h)")
    print(f"  distance travelled  : {tr['x_m'][-1]:.1f} m")
    print(f"  pitch (nose-up +)   : total [{np.degrees(tr['pitch_rad'].min()):+.2f},"
          f" {np.degrees(tr['pitch_rad'].max()):+.2f}] deg;"
          f"  road [{np.degrees(tr['road_pitch_rad'].min()):+.2f},"
          f" {np.degrees(tr['road_pitch_rad'].max()):+.2f}] deg;"
          f"  chassis-rel [{np.degrees(tr['pitch_rel_rad'].min()):+.2f},"
          f" {np.degrees(tr['pitch_rel_rad'].max()):+.2f}] deg")
    print(f"  rear susp travel    : [{tr['susp_travel_m'].min()*1e3:+.0f},"
          f" {tr['susp_travel_m'].max()*1e3:+.0f}] mm  (+ = compression/jounce)")
    print(f"  wheel omega max     : {tr['wheel_omega'].max():.1f} rad/s"
          f" (=> {tr['wheel_omega'].max()*0.29:.2f} m/s surface)")
    # Energy budget sanity: drive must decompose into aero + rolling + stored
    # KE/PE, with a modest residual (contact slip, suspension damping, bump).
    T = tr["t"][-1]
    m = float(tr["total_mass_kg"])
    d = tr["x_m"][-1]
    E_roll_est = 0.008 * m * 9.81 * d          # Crr-equivalent from contact
    dKE = 0.5 * m * (v[-1] ** 2 - v[0] ** 2)
    dPE = m * 9.81 * (tr["road_h_m"][-1] - tr["road_h_m"][0])
    resid = float(tr["E_drive_J"]) - float(tr["E_aero_J"]) - E_roll_est - dKE - dPE
    print(f"  energy budget       : drive {float(tr['E_drive_J']):7.0f} J = "
          f"aero {float(tr['E_aero_J']):5.0f} + roll(Crr .008) {E_roll_est:5.0f}"
          f" + dKE {dKE:6.0f} + dPE {dPE:6.0f} + resid {resid:5.0f} J"
          f"  (mean drive power {float(tr['E_drive_J'])/T:.0f} W)")
    idx_cut = np.argmax(v >= V_CUT) if (v >= V_CUT).any() else -1
    if idx_cut > 0:
        print(f"  reaches 25 km/h cut : t={tr['t'][idx_cut]:.2f} s,"
              f" x={tr['x_m'][idx_cut]:.1f} m (assist off after)")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--render", action="store_true", help="render mp4 + gif fly-by")
    ap.add_argument("--t-end", type=float, default=18.0)
    ap.add_argument("--azimuth", type=float, default=100.0,
                    help="chase-camera azimuth [deg]")
    args = ap.parse_args()

    t0 = time.time()
    tr = run(t_end=args.t_end)
    print(f"co-sim: {time.time()-t0:.2f} s wall for {args.t_end:.0f} s sim")
    _print_summary(tr, "terrain")

    # Flat-road twin run: the like-for-like longitudinal comparison against
    # the planar model (same drive law, same losses, no terrain effects).
    tr_flat = run(t_end=args.t_end, terrain=False)
    _print_summary(tr_flat, "flat")

    npz = os.path.join(MEDIA, "ebike_mujoco_cosim.npz")
    np.savez(npz, **tr, **{f"flat_{k}": v for k, v in tr_flat.items()})
    print(f"wrote {npz}")

    if args.render:
        t0 = time.time()
        render(os.path.join(MEDIA, "ebike_mujoco_cosim"),
               t_end=args.t_end, azimuth=args.azimuth)
        print(f"render: {time.time()-t0:.2f} s wall")


if __name__ == "__main__":
    main()
