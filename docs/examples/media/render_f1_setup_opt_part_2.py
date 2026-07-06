# SPDX-License-Identifier: MIT
"""Post-hoc MuJoCo rendering of an F1 setup-optimisation before/after.

Reads `f1_part2_traj.npz` (saved by Part 2 of the notebook), builds a two-car
MJCF scene (red baseline + yellow optimised, sharing the same stylised F1
chassis), and renders a side-by-side run through the first corner (the fast
right-hander C1) with a tracking camera that follows the midpoint between
the two cars.

Purpose: render the before/after comparison to an MP4 for visualization.
No physics — `mj_forward` only; both cars are kinematically driven from the
recorded pose trajectories.

Run:
    python docs/examples/media/render_f1_setup_opt_part_2.py

Output:
    docs/examples/media/f1_setup_opt_part_2.mp4
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import mujoco

try:
    import imageio.v2 as imageio
except ImportError:  # pragma: no cover
    import imageio


HERE = Path(__file__).resolve().parent
TRAJ = HERE / "f1_part2_traj.npz"
SCENE = HERE / "f1_scene_part2.xml"
OUT = HERE / "f1_setup_opt_part_2.mp4"

# Car geometry (matches Part 1).
L_CAR = 5.50
W_CAR = 1.80
H_CAR = 0.60
WHEEL_R = 0.34
WHEEL_W = 0.36
TRACK_W = 1.60
WHEELBASE = 3.25

WHEEL_Z = WHEEL_R + 0.02  # body z-origin so wheels sit on the ground

# Render config.
FPS = 30
WIDTH = 1280
HEIGHT = 720

# Clip window around corner 0: buffer seconds before/after entry/exit.
BUFFER_S = 2.0

# Camera framing. The two cars stay within ~20 m of each other through the
# corner; we sit ~32 m back and ~18 deg above so both cars read clearly at
# 1280x720. `CAM_AZ_SMOOTH_TAU` low-pass-filters the heading-derived azimuth
# so the camera doesn't snake every frame.
CAM_BACK_DIST = 32.0
CAM_ELEVATION_DEG = -18.0
CAM_BEHIND_OFFSET = 10.0  # "lookat" pushed forward of midpoint along heading
CAM_AZ_SMOOTH_TAU = 0.30  # seconds — exponential smoothing time constant

# Corner index to render.
CORNER_IDX = 0


# ─────────────────────────────────────────────────────────────────────────────
# MJCF construction
# ─────────────────────────────────────────────────────────────────────────────
def _car_body_xml(name: str, material: str, wing_material: str) -> str:
    """One stylised F1 chassis as a free body. Wings/halo/wheels share the
    same geometry; only the chassis colour material differs between cars."""
    return f"""
    <body name="{name}" pos="0 0 {WHEEL_Z}">
      <joint name="{name}_free" type="free"/>
      <inertial pos="0 0 0" mass="830" diaginertia="200 1350 1500"/>
      <!-- Chassis box -->
      <geom name="{name}_chassis" type="box"
            size="{L_CAR/2:.3f} {W_CAR/2:.3f} {H_CAR/2:.3f}"
            pos="0 0 {H_CAR/2 + 0.05}"
            material="{material}"/>
      <!-- Halo arch (carbon) -->
      <geom name="{name}_halo" type="capsule"
            fromto="0 {-W_CAR/2 + 0.1} {H_CAR + 0.10}
                    0 { W_CAR/2 - 0.1} {H_CAR + 0.10}"
            size="0.05" material="halo"/>
      <!-- Rear wing -->
      <geom name="{name}_rear_wing" type="box"
            size="0.1 {W_CAR/2 + 0.3:.3f} 0.04"
            pos="{-L_CAR/2 + 0.20:.3f} 0 {H_CAR + 0.45}"
            material="{wing_material}"/>
      <geom name="{name}_rear_wing_endL" type="box" size="0.1 0.04 0.30"
            pos="{-L_CAR/2 + 0.20:.3f} { W_CAR/2 + 0.30:.3f} {H_CAR + 0.20}"
            material="{wing_material}"/>
      <geom name="{name}_rear_wing_endR" type="box" size="0.1 0.04 0.30"
            pos="{-L_CAR/2 + 0.20:.3f} {-W_CAR/2 - 0.30:.3f} {H_CAR + 0.20}"
            material="{wing_material}"/>
      <!-- Front wing -->
      <geom name="{name}_front_wing" type="box"
            size="0.08 {W_CAR/2 + 0.30:.3f} 0.04"
            pos="{L_CAR/2 - 0.05:.3f} 0 0.10"
            material="{wing_material}"/>
      <!-- 4 wheels -->
      <geom name="{name}_wheel_FL" type="cylinder" size="{WHEEL_R:.3f} {WHEEL_W/2:.3f}"
            pos="{WHEELBASE/2:.3f}  {TRACK_W/2:.3f} {WHEEL_R:.3f}"
            quat="0.7071 0.7071 0 0" material="tyre"/>
      <geom name="{name}_wheel_FR" type="cylinder" size="{WHEEL_R:.3f} {WHEEL_W/2:.3f}"
            pos="{WHEELBASE/2:.3f} {-TRACK_W/2:.3f} {WHEEL_R:.3f}"
            quat="0.7071 0.7071 0 0" material="tyre"/>
      <geom name="{name}_wheel_RL" type="cylinder" size="{WHEEL_R:.3f} {WHEEL_W/2:.3f}"
            pos="{-WHEELBASE/2:.3f}  {TRACK_W/2:.3f} {WHEEL_R:.3f}"
            quat="0.7071 0.7071 0 0" material="tyre"/>
      <geom name="{name}_wheel_RR" type="cylinder" size="{WHEEL_R:.3f} {WHEEL_W/2:.3f}"
            pos="{-WHEELBASE/2:.3f} {-TRACK_W/2:.3f} {WHEEL_R:.3f}"
            quat="0.7071 0.7071 0 0" material="tyre"/>
    </body>"""


def build_mjcf(
    road_X: np.ndarray,
    road_Y: np.ndarray,
    cars_X: np.ndarray,
    cars_Y: np.ndarray,
) -> str:
    """Build the MJCF for the two-car scene.

    The asphalt strip is laid down following `(road_X, road_Y)` — typically
    the mid-line between the two recorded car trajectories over the clip
    window. The recorded trajectories drift from the track centerline by
    up to ~250 m (the trajectory model in Part 2 is a placeholder), so
    rendering the centerline directly leaves the cars apparently driving
    in the grass. Laying the road under the cars instead is a small
    geometric lie that produces a coherent visual.

    Args:
        road_X, road_Y: sampled XY of the road centerline to lay down.
        cars_X, cars_Y: concatenation of both cars' XY over the clip
            window — used to size the ground plane.
    """
    # Decimate to ≤120 segments along the road; we already have only the
    # corner-relevant span.
    n_seg = min(120, len(road_X))
    idx = np.linspace(0, len(road_X) - 1, n_seg, dtype=int)
    cx = road_X[idx]
    cy = road_Y[idx]

    # Ground-plane bounding box from cars + road.
    x_min = float(min(cars_X.min(), cx.min())) - 200.0
    x_max = float(max(cars_X.max(), cx.max())) + 200.0
    y_min = float(min(cars_Y.min(), cy.min())) - 200.0
    y_max = float(max(cars_Y.max(), cy.max())) + 200.0

    # Lay down the asphalt strip — wide enough (half-width 10 m) that both
    # cars fit on it through the corner.
    road_xml = []
    for i in range(len(cx) - 1):
        x_mid = 0.5 * (cx[i] + cx[i + 1])
        y_mid = 0.5 * (cy[i] + cy[i + 1])
        dx = cx[i + 1] - cx[i]
        dy = cy[i + 1] - cy[i]
        length = float(np.hypot(dx, dy))
        if length < 0.01:
            continue
        ang_rad = float(np.arctan2(dy, dx))
        qw = np.cos(ang_rad / 2.0)
        qz = np.sin(ang_rad / 2.0)
        road_xml.append(
            f'<geom type="box" pos="{x_mid:.2f} {y_mid:.2f} 0.02" '
            f'size="{length/2 + 0.5:.2f} 10.0 0.02" '
            f'quat="{qw:.4f} 0 0 {qz:.4f}" '
            f'material="asphalt" contype="0" conaffinity="0"/>'
        )
    centerline_block = "\n      ".join(road_xml)

    # Ground plane centered on the cars' bbox so we have grass underfoot.
    cx_mid = 0.5 * (x_min + x_max)
    cy_mid = 0.5 * (y_min + y_max)
    half_span = max(x_max - x_min, y_max - y_min) * 0.7 + 250.0

    baseline_body = _car_body_xml("car_baseline", "chassis_red", "wing")
    optimised_body = _car_body_xml("car_optimised", "chassis_yellow", "wing")

    mjcf = f"""
<mujoco model="f1_setup_opt_part_2">
  <option gravity="0 0 -9.81" timestep="0.005"/>
  <visual>
    <headlight diffuse=".9 .9 .9" ambient=".25 .25 .25" specular="0 0 0"/>
    <rgba haze="0.30 0.36 0.45 1"/>
    <global offwidth="{WIDTH}" offheight="{HEIGHT}"/>
    <quality shadowsize="4096"/>
  </visual>
  <asset>
    <texture type="skybox" builtin="gradient" rgb1="0.55 0.7 0.92" rgb2="0.15 0.20 0.35"
             width="800" height="800"/>
    <texture name="grass_tex" type="2d" builtin="checker"
             rgb1="0.20 0.35 0.18" rgb2="0.13 0.25 0.13"
             width="200" height="200"/>
    <material name="grass" texture="grass_tex" texrepeat="60 60"
              reflectance="0.0" specular="0.05"/>
    <material name="asphalt" rgba="0.28 0.28 0.30 1" specular="0.20" shininess="0.20"/>
    <material name="chassis_red"    rgba="0.90 0.10 0.10 1" specular="0.75" shininess="0.7"/>
    <material name="chassis_yellow" rgba="0.95 0.80 0.10 1" specular="0.75" shininess="0.7"/>
    <material name="halo" rgba="0.05 0.05 0.05 1" specular="0.4" shininess="0.5"/>
    <material name="tyre" rgba="0.05 0.05 0.05 1" specular="0.2" shininess="0.2"/>
    <material name="wing" rgba="0.10 0.10 0.12 1" specular="0.4" shininess="0.5"/>
  </asset>
  <worldbody>
    <light name="sun" diffuse="1.0 0.95 0.85" pos="500 -500 600" dir="-0.6 0.6 -0.8"
           castshadow="true"/>
    <light name="fill" diffuse="0.30 0.35 0.50" pos="-300 300 400" dir="0.5 -0.5 -0.6"
           castshadow="false"/>
    <!-- Ground plane (grass) -->
    <geom name="ground" type="plane" pos="{cx_mid:.1f} {cy_mid:.1f} 0"
          size="{half_span:.0f} {half_span:.0f} 1" material="grass"/>
    <!-- Asphalt centerline strip (corner-region only) -->
    {centerline_block}
    {baseline_body}
    {optimised_body}
  </worldbody>
</mujoco>
"""
    return mjcf


# ─────────────────────────────────────────────────────────────────────────────
# Pose helpers
# ─────────────────────────────────────────────────────────────────────────────
def yaw_to_quat(psi: float) -> np.ndarray:
    """MuJoCo quaternion (w, x, y, z) for a rotation `psi` about +Z."""
    return np.array([np.cos(psi / 2.0), 0.0, 0.0, np.sin(psi / 2.0)])


def arc_length(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    ds = np.hypot(np.diff(X), np.diff(Y))
    return np.concatenate([[0.0], np.cumsum(ds)])


# ─────────────────────────────────────────────────────────────────────────────
# HUD overlay (matplotlib post-process)
# ─────────────────────────────────────────────────────────────────────────────
def _compose_hud(
    rgb: np.ndarray,
    t_clip: float,
    v_b_kmh: float,
    v_o_kmh: float,
    lap_b: float,
    lap_o: float,
    lap_delta: float,
) -> np.ndarray:
    """Burn HUD text into the rendered frame using matplotlib.

    Top-left: BASELINE (red) lap time.
    Top-right: OPTIMISED (yellow) lap time + delta.
    Bottom-left: baseline speed. Bottom-right: optimised speed.
    Bottom-center: clip time.
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    fig, ax = plt.subplots(figsize=(WIDTH / 100.0, HEIGHT / 100.0), dpi=100)
    ax.imshow(rgb)
    ax.set_axis_off()
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

    # Top semi-transparent strip (full-width) for high-contrast HUD text.
    ax.add_patch(Rectangle((0, 0), WIDTH, 70, facecolor="black", alpha=0.55, zorder=2))
    # Bottom semi-transparent strip for speed/clip-time.
    ax.add_patch(Rectangle((0, HEIGHT - 50), WIDTH, 50,
                           facecolor="black", alpha=0.55, zorder=2))

    common = dict(family="DejaVu Sans", fontweight="bold", zorder=3)

    # BASELINE (red) top-left.
    ax.text(20, 28, "BASELINE",
            color=(0.95, 0.20, 0.20), fontsize=14, **common)
    ax.text(20, 56, f"lap = {lap_b:6.3f} s",
            color="white", fontsize=18, **common)

    # OPTIMISED (yellow) top-right.
    sign = "+" if lap_delta >= 0 else ""
    ax.text(WIDTH - 20, 28, "OPTIMISED",
            color=(0.95, 0.80, 0.15), fontsize=14, ha="right", **common)
    ax.text(WIDTH - 20, 56,
            f"lap = {lap_o:6.3f} s   ({sign}{lap_delta:+.3f} s)",
            color="white", fontsize=18, ha="right", **common)

    # Bottom: live speed readouts + clip time.
    ax.text(20, HEIGHT - 18, f"v = {v_b_kmh:3.0f} km/h",
            color=(0.95, 0.50, 0.50), fontsize=14, **common)
    ax.text(WIDTH - 20, HEIGHT - 18, f"v = {v_o_kmh:3.0f} km/h",
            color=(0.95, 0.85, 0.40), fontsize=14, ha="right", **common)
    ax.text(WIDTH / 2, HEIGHT - 18,
            f"jaxonomy F1 part 2  -  corner C{CORNER_IDX + 1}   t = {t_clip:4.2f} s",
            color="white", fontsize=13, ha="center", **common)

    fig.canvas.draw()
    img = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
    plt.close(fig)
    return img


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    print(f"Loading trajectory: {TRAJ}")
    d = np.load(TRAJ)
    t_b = d["t_baseline"]
    X_b = d["X_baseline"]; Y_b = d["Y_baseline"]; psi_b = d["psi_baseline"]
    v_b_kmh = d["v_kmh_baseline"]
    t_o = d["t_optimised"]
    X_o = d["X_optimised"]; Y_o = d["Y_optimised"]; psi_o = d["psi_optimised"]
    v_o_kmh = d["v_kmh_optimised"]
    center_X = d["center_X"]; center_Y = d["center_Y"]
    lap_baseline = float(d["lap_baseline"])
    lap_opt = float(d["lap_opt"])
    lap_delta = float(d["lap_delta_seconds"])
    corner_s_starts = d["corner_s_starts"]
    corner_s_ends = d["corner_s_ends"]

    print(
        f"  baseline:  {len(t_b)} frames over {t_b[-1]:.2f} s, "
        f"speed {v_b_kmh.min():.0f}-{v_b_kmh.max():.0f} km/h"
    )
    print(
        f"  optimised: {len(t_o)} frames over {t_o[-1]:.2f} s, "
        f"speed {v_o_kmh.min():.0f}-{v_o_kmh.max():.0f} km/h"
    )
    print(
        f"  lap_baseline = {lap_baseline:.3f}s, "
        f"lap_opt = {lap_opt:.3f}s, delta = {lap_delta:+.3f}s"
    )

    # Find clip window: from `BUFFER_S` before either car enters corner CORNER_IDX
    # to `BUFFER_S` after either car exits.
    s_b = arc_length(X_b, Y_b)
    s_o = arc_length(X_o, Y_o)
    c_start = float(corner_s_starts[CORNER_IDX])
    c_end = float(corner_s_ends[CORNER_IDX])
    i_enter = min(int(np.searchsorted(s_b, c_start)),
                  int(np.searchsorted(s_o, c_start)))
    i_exit = max(int(np.searchsorted(s_b, c_end)),
                 int(np.searchsorted(s_o, c_end)))

    buf = int(BUFFER_S * FPS)
    i0 = max(0, i_enter - buf)
    i1 = min(len(t_b), i_exit + buf)
    n_frames = i1 - i0
    clip_dur = n_frames / FPS
    print(
        f"  corner C{CORNER_IDX + 1}: s in [{c_start:.0f}, {c_end:.0f}], "
        f"frame window [{i0}, {i1}) = {clip_dur:.2f} s @ {FPS} fps"
    )

    # Build MJCF. We lay the road under the cars (midline between the two
    # trajectories, extended by ~80 m before/after the clip so the road
    # doesn't terminate visibly in-frame). The recorded trajectories sit
    # ~90 m off the world-coords centerline in this dataset (placeholder
    # plant), so painting the world centerline directly would leave the
    # cars apparently driving on grass.
    cars_X = np.concatenate([X_b[i0:i1], X_o[i0:i1]])
    cars_Y = np.concatenate([Y_b[i0:i1], Y_o[i0:i1]])

    # Extend the road a bit beyond the clip window in both directions so
    # the asphalt strip never visibly truncates near the camera frustum.
    pad = int(2.5 * FPS)  # ~2.5 s of trajectory padding on each side
    j0 = max(0, i0 - pad)
    j1 = min(len(t_b), i1 + pad)
    road_X = 0.5 * (X_b[j0:j1] + X_o[j0:j1])
    road_Y = 0.5 * (Y_b[j0:j1] + Y_o[j0:j1])

    print("Building MJCF ...")
    mjcf = build_mjcf(road_X, road_Y, cars_X, cars_Y)
    SCENE.write_text(mjcf)
    print(f"  wrote {SCENE}  ({SCENE.stat().st_size / 1024:.1f} kB)")

    print("Loading MuJoCo model ...")
    model = mujoco.MjModel.from_xml_string(mjcf)
    data = mujoco.MjData(model)

    # Find each car's free-joint qpos slice.
    jid_b = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "car_baseline_free")
    jid_o = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "car_optimised_free")
    qadr_b = model.jnt_qposadr[jid_b]
    qadr_o = model.jnt_qposadr[jid_o]
    print(f"  baseline qpos slice = [{qadr_b}:{qadr_b + 7}]")
    print(f"  optimised qpos slice = [{qadr_o}:{qadr_o + 7}]")

    print(f"Rendering {n_frames} frames @ {WIDTH}x{HEIGHT} ...")
    t0 = time.time()
    renderer = mujoco.Renderer(model, height=HEIGHT, width=WIDTH)
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE

    # Exponential-moving-average azimuth so the camera doesn't jerk frame to
    # frame as the cars yaw through the corner.
    alpha = float(np.clip(1.0 / (1.0 + CAM_AZ_SMOOTH_TAU * FPS), 0.0, 1.0))
    cam_az_state = None  # initialised on first frame

    raw_frames = []
    try:
        for k in range(n_frames):
            i = i0 + k
            # Drive both cars kinematically.
            data.qpos[qadr_b : qadr_b + 3] = (X_b[i], Y_b[i], WHEEL_Z)
            data.qpos[qadr_b + 3 : qadr_b + 7] = yaw_to_quat(float(psi_b[i]))
            data.qpos[qadr_o : qadr_o + 3] = (X_o[i], Y_o[i], WHEEL_Z)
            data.qpos[qadr_o + 3 : qadr_o + 7] = yaw_to_quat(float(psi_o[i]))
            mujoco.mj_forward(model, data)

            # Tracking camera: aim at midpoint, biased forward in the direction
            # of motion so the cars sit slightly low-center in frame. Use the
            # mean yaw of the two cars for the camera azimuth, smoothed in
            # the complex plane to avoid wrap discontinuities at ±180 deg.
            mid_x = 0.5 * (X_b[i] + X_o[i])
            mid_y = 0.5 * (Y_b[i] + Y_o[i])
            mid_psi = 0.5 * (float(psi_b[i]) + float(psi_o[i]))
            target_az = float(np.rad2deg(mid_psi)) + 180.0

            if cam_az_state is None:
                cam_az_state = np.exp(1j * np.deg2rad(target_az))
            else:
                z_target = np.exp(1j * np.deg2rad(target_az))
                cam_az_state = (1 - alpha) * cam_az_state + alpha * z_target
            cam_az = float(np.rad2deg(np.angle(cam_az_state)))

            cam.lookat = np.array([
                mid_x + CAM_BEHIND_OFFSET * np.cos(mid_psi),
                mid_y + CAM_BEHIND_OFFSET * np.sin(mid_psi),
                1.5,
            ])
            cam.distance = CAM_BACK_DIST
            cam.azimuth = cam_az
            cam.elevation = CAM_ELEVATION_DEG

            renderer.update_scene(data, camera=cam)
            rgb = renderer.render()
            raw_frames.append(rgb.copy())

            if (k + 1) % 30 == 0:
                print(
                    f"  frame {k + 1}/{n_frames}  "
                    f"(t_clip={k / FPS:.2f}s, "
                    f"v_b={v_b_kmh[i]:.0f}, v_o={v_o_kmh[i]:.0f} km/h, "
                    f"sep={np.hypot(X_b[i] - X_o[i], Y_b[i] - Y_o[i]):.1f}m)"
                )
    finally:
        renderer.close()
    print(f"  rendered in {time.time() - t0:.1f} s")

    # Burn-in HUD with matplotlib.
    print("Composing HUD overlay ...")
    t1 = time.time()
    final_frames = []
    for k, rgb in enumerate(raw_frames):
        i = i0 + k
        img = _compose_hud(
            rgb,
            t_clip=k / FPS,
            v_b_kmh=float(v_b_kmh[i]),
            v_o_kmh=float(v_o_kmh[i]),
            lap_b=lap_baseline,
            lap_o=lap_opt,
            lap_delta=lap_delta,
        )
        final_frames.append(img)
        if (k + 1) % 30 == 0:
            print(f"  HUD frame {k + 1}/{n_frames}")
    print(f"  composed in {time.time() - t1:.1f} s")

    # Save the first frame for quick visual inspection by humans.
    preview_path = HERE / "_f1_setup_opt_part_2_preview.png"
    imageio.imwrite(str(preview_path), final_frames[0])
    print(f"  preview frame written to {preview_path}")

    # Write MP4 (fallback to GIF if ffmpeg/libx264 unavailable).
    print(f"Writing MP4: {OUT}")
    try:
        with imageio.get_writer(
            str(OUT), fps=FPS, codec="libx264",
            quality=8, pixelformat="yuv420p",
        ) as w:
            for f in final_frames:
                w.append_data(f)
        size_mb = OUT.stat().st_size / 1e6
        dur_s = len(final_frames) / FPS
        print(f"  wrote {OUT}  ({size_mb:.2f} MB, {dur_s:.2f} s)")
    except Exception as e:  # pragma: no cover
        print(f"  mp4 failed: {e}; writing GIF instead")
        gif_path = OUT.with_suffix(".gif")
        small = [f[::2, ::2] for f in final_frames]
        imageio.mimsave(str(gif_path), small, fps=FPS, loop=0)
        print(f"  wrote {gif_path}")


if __name__ == "__main__":
    main()
