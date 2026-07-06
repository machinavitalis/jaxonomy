# SPDX-License-Identifier: MIT
"""Post-hoc MuJoCo rendering of an F1 lap.

Reads `f1_part1_traj.npz` (saved by the notebook), builds a stylised F1-chassis
scene + the synthetic track centerline drawn as a strip on the ground plane,
and renders an MP4 from a chase camera. The car is a colored chassis box +
4 wheels (no aero kit) and the racing line is painted in behind it.

Purpose: render the trajectory to an MP4 for visualization.
No physics — `mj_forward` only.

Run:
    python docs/examples/media/render_f1_lap_part_1.py

Output:
    docs/examples/media/f1_lap_part_1.mp4
"""

from __future__ import annotations

import os
import time
from pathlib import Path
import numpy as np
import mujoco

try:
    import imageio.v2 as imageio
except ImportError:
    import imageio


HERE = Path(__file__).resolve().parent
TRAJ = HERE / "f1_part1_traj.npz"
OUT  = HERE / "f1_lap_part_1.mp4"

# ─────────────────────────────────────────────────────────────────────────────
# Car geometry
# ─────────────────────────────────────────────────────────────────────────────
L_CAR  = 5.50       # chassis length [m]
W_CAR  = 1.80       # chassis width  [m]
H_CAR  = 0.60       # chassis height [m] (low — ground-effect car)
WHEEL_R = 0.34      # wheel radius   [m]
WHEEL_W = 0.36      # wheel width    [m]
TRACK_W = 1.60      # track width    [m]
WHEELBASE = 3.25    # wheelbase      [m]


def build_mjcf(center_X: np.ndarray, center_Y: np.ndarray) -> str:
    """Build the MJCF for the F1 scene.

    The track centerline is rendered as a sequence of small grey ground geoms
    so the racing-line marker (the moving car) has a visible reference.
    """
    # Build a sequence of small box geoms tracing the centerline. Decimate
    # to ~200 segments so MuJoCo doesn't choke on hundreds of bodies.
    n_seg = 200
    idx = np.linspace(0, len(center_X) - 1, n_seg, dtype=int)
    cx = center_X[idx]; cy = center_Y[idx]

    centerline_xml = []
    for i in range(len(cx) - 1):
        x_mid = 0.5 * (cx[i] + cx[i+1])
        y_mid = 0.5 * (cy[i] + cy[i+1])
        dx = cx[i+1] - cx[i]; dy = cy[i+1] - cy[i]
        length = float(np.hypot(dx, dy))
        if length < 0.01:
            continue
        ang_rad = float(np.arctan2(dy, dx))
        # rotation about z by ang_rad
        qw = np.cos(ang_rad / 2.0); qz = np.sin(ang_rad / 2.0)
        centerline_xml.append(
            f'<geom type="box" pos="{x_mid:.2f} {y_mid:.2f} 0.02" '
            f'size="{length/2 + 0.5:.2f} 8.0 0.02" '
            f'quat="{qw:.4f} 0 0 {qz:.4f}" '
            f'material="asphalt" contype="0" conaffinity="0"/>'
        )
    centerline_block = "\n      ".join(centerline_xml)

    # Center the camera scene around the centerline bounding box.
    xc_min, xc_max = float(center_X.min()), float(center_X.max())
    yc_min, yc_max = float(center_Y.min()), float(center_Y.max())
    cx_mid = 0.5 * (xc_min + xc_max)
    cy_mid = 0.5 * (yc_min + yc_max)
    half_span = max(xc_max - xc_min, yc_max - yc_min) * 0.6 + 200.0

    mjcf = f"""
<mujoco model="f1_lap">
  <option gravity="0 0 -9.81" timestep="0.005"/>
  <visual>
    <headlight diffuse=".9 .9 .9" ambient=".25 .25 .25" specular="0 0 0"/>
    <rgba haze="0.30 0.36 0.45 1"/>
    <global offwidth="1280" offheight="720"/>
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
    <material name="chassis" rgba="0.85 0.10 0.10 1" specular="0.7" shininess="0.7"/>
    <material name="halo"    rgba="0.05 0.05 0.05 1" specular="0.4" shininess="0.5"/>
    <material name="tyre"    rgba="0.05 0.05 0.05 1" specular="0.2" shininess="0.2"/>
    <material name="wing"    rgba="0.10 0.10 0.12 1" specular="0.4" shininess="0.5"/>
    <material name="line"    rgba="0.85 0.85 0.85 1"/>
  </asset>
  <worldbody>
    <light name="sun" diffuse="1.0 0.95 0.85" pos="500 -500 600" dir="-0.6 0.6 -0.8"
           castshadow="true"/>
    <light name="fill" diffuse="0.30 0.35 0.50" pos="-300 300 400" dir="0.5 -0.5 -0.6"
           castshadow="false"/>
    <!-- Big ground plane (grass) -->
    <geom name="ground" type="plane" pos="{cx_mid:.1f} {cy_mid:.1f} 0"
          size="{half_span:.0f} {half_span:.0f} 1" material="grass"/>
    <!-- Centerline strip (asphalt road) -->
    {centerline_block}

    <!-- The car as a free body. Chassis box + 4 wheels + a token rear wing. -->
    <body name="car" pos="0 0 {WHEEL_R + 0.02}">
      <joint name="free" type="free"/>
      <inertial pos="0 0 0" mass="830" diaginertia="200 1350 1500"/>
      <!-- Chassis box: -->
      <geom name="chassis" type="box"
            size="{L_CAR/2:.3f} {W_CAR/2:.3f} {H_CAR/2:.3f}"
            pos="0 0 {H_CAR/2 + 0.05}"
            material="chassis"/>
      <!-- Halo arch -->
      <geom name="halo_main" type="capsule"
            fromto="0 {-W_CAR/2 + 0.1} {H_CAR + 0.10}
                    0 { W_CAR/2 - 0.1} {H_CAR + 0.10}"
            size="0.05" material="halo"/>
      <!-- Rear wing -->
      <geom name="rear_wing" type="box"
            size="0.1 {W_CAR/2 + 0.3:.3f} 0.04"
            pos="{-L_CAR/2 + 0.20:.3f} 0 {H_CAR + 0.45}"
            material="wing"/>
      <geom name="rear_wing_endL" type="box" size="0.1 0.04 0.30"
            pos="{-L_CAR/2 + 0.20:.3f} { W_CAR/2 + 0.30:.3f} {H_CAR + 0.20}"
            material="wing"/>
      <geom name="rear_wing_endR" type="box" size="0.1 0.04 0.30"
            pos="{-L_CAR/2 + 0.20:.3f} {-W_CAR/2 - 0.30:.3f} {H_CAR + 0.20}"
            material="wing"/>
      <!-- Front wing -->
      <geom name="front_wing" type="box"
            size="0.08 {W_CAR/2 + 0.30:.3f} 0.04"
            pos="{L_CAR/2 - 0.05:.3f} 0 0.10"
            material="wing"/>
      <!-- 4 wheels at the corners -->
      <geom name="wheel_FL" type="cylinder" size="{WHEEL_R:.3f} {WHEEL_W/2:.3f}"
            pos="{WHEELBASE/2:.3f}  {TRACK_W/2:.3f} {WHEEL_R:.3f}"
            quat="0.7071 0.7071 0 0" material="tyre"/>
      <geom name="wheel_FR" type="cylinder" size="{WHEEL_R:.3f} {WHEEL_W/2:.3f}"
            pos="{WHEELBASE/2:.3f} {-TRACK_W/2:.3f} {WHEEL_R:.3f}"
            quat="0.7071 0.7071 0 0" material="tyre"/>
      <geom name="wheel_RL" type="cylinder" size="{WHEEL_R:.3f} {WHEEL_W/2:.3f}"
            pos="{-WHEELBASE/2:.3f}  {TRACK_W/2:.3f} {WHEEL_R:.3f}"
            quat="0.7071 0.7071 0 0" material="tyre"/>
      <geom name="wheel_RR" type="cylinder" size="{WHEEL_R:.3f} {WHEEL_W/2:.3f}"
            pos="{-WHEELBASE/2:.3f} {-TRACK_W/2:.3f} {WHEEL_R:.3f}"
            quat="0.7071 0.7071 0 0" material="tyre"/>
    </body>
  </worldbody>
</mujoco>
"""
    return mjcf


def yaw_to_quat(psi: float) -> np.ndarray:
    """ZYX yaw-only quaternion in MuJoCo (w, x, y, z) order."""
    return np.array([np.cos(psi/2), 0.0, 0.0, np.sin(psi/2)])


def main(time_compression: float = 7.0, fps: int = 30):
    """Render the F1 lap MP4.

    Args:
        time_compression: factor to speed up the lap for the video so the
            ~60s lap fits in ~9-10 s of viewer time.
        fps: target render frame rate.
    """
    print(f"Loading trajectory: {TRAJ}")
    d = np.load(TRAJ)
    t = d["t"];  X = d["X"];  Y = d["Y"];  psi = d["psi"]
    v_kmh = d["v_kmh"];  gear = d["gear"]
    center_X = d["center_X"];  center_Y = d["center_Y"]
    lap_time = float(d["lap_time"])
    print(f"  {len(t)} frames over {t[-1]:.2f} s, lap_time = {lap_time:.2f} s")
    print(f"  speed range: {v_kmh.min():.0f}-{v_kmh.max():.0f} km/h")

    # Compress the trajectory: keep every Nth frame so the final video is
    # ~ lap_time / time_compression seconds long.
    target_video_dur = lap_time / time_compression
    n_frames = int(target_video_dur * fps)
    sample_idx = np.linspace(0, len(t) - 1, n_frames, dtype=int)
    t_v   = t[sample_idx]
    X_v   = X[sample_idx]
    Y_v   = Y[sample_idx]
    psi_v = psi[sample_idx]
    v_v   = v_kmh[sample_idx]
    g_v   = gear[sample_idx]
    print(f"  rendering {n_frames} frames @ {fps} fps "
          f"-> {n_frames/fps:.1f} s of video "
          f"(time-compression {time_compression:.1f}x)")

    print("Building MJCF ...")
    mjcf = build_mjcf(center_X, center_Y)
    (HERE / "f1_scene.xml").write_text(mjcf)
    print(f"  wrote {HERE / 'f1_scene.xml'}")

    print("Loading MuJoCo model ...")
    model = mujoco.MjModel.from_xml_string(mjcf)
    data  = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=720, width=1280)

    # Chase camera that follows the car
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE

    # Initialize the car off the visible field
    mujoco.mj_forward(model, data)

    print(f"Rendering {n_frames} frames ...")
    t0 = time.time()
    frames = []
    for i, idx in enumerate(range(n_frames)):
        # Car pose
        data.qpos[0:3] = np.array([X_v[i], Y_v[i], WHEEL_R + 0.02])
        data.qpos[3:7] = yaw_to_quat(float(psi_v[i]))
        mujoco.mj_forward(model, data)

        # Chase camera: 40 m behind + 12 m up, looking forward.
        cam_back_dist = 32.0
        cam_height    = 12.0
        cam_az = float(np.rad2deg(psi_v[i])) + 180.0  # behind the car
        cam.lookat = np.array([X_v[i] + 12.0 * np.cos(psi_v[i]),
                               Y_v[i] + 12.0 * np.sin(psi_v[i]),
                               1.5])
        cam.distance = cam_back_dist
        cam.azimuth  = cam_az
        cam.elevation = -18.0

        renderer.update_scene(data, camera=cam)
        rgb = renderer.render()

        # Burn-in HUD: lap time, speed, gear  (top-left, white text on dark strip)
        # We do this in Python rather than MuJoCo since MuJoCo has no native text.
        # Add a black bar at the top-left for legibility.
        bar_h, bar_w = 70, 360
        rgb = rgb.copy()
        rgb[:bar_h, :bar_w] = (rgb[:bar_h, :bar_w] * 0.30).astype(np.uint8)
        frames.append((rgb, t_v[i], v_v[i], g_v[i]))

        if (i + 1) % 50 == 0:
            print(f"  frame {i+1}/{n_frames}  (t={t_v[i]:.2f}s, "
                  f"v={v_v[i]:.0f} km/h, g={int(round(g_v[i]))+1})")
    print(f"  rendered in {time.time()-t0:.1f} s")

    # Burn in text overlays using matplotlib (avoids requiring PIL)
    print("Burning in HUD overlay ...")
    import matplotlib.pyplot as plt
    final_frames = []
    for rgb, ts, vs, gs in frames:
        fig, ax = plt.subplots(figsize=(12.8, 7.2), dpi=100)
        ax.imshow(rgb)
        ax.set_axis_off()
        ax.text(15, 25, f"t = {ts:5.2f} s",  color="white", fontsize=11,
                family="monospace", fontweight="bold")
        ax.text(15, 45, f"v = {vs:3.0f} km/h", color="white", fontsize=11,
                family="monospace", fontweight="bold")
        ax.text(15, 65, f"gear = {int(round(gs))+1}", color="white", fontsize=11,
                family="monospace", fontweight="bold")
        ax.text(1265, 25, f"jaxonomy F1 part 1  |  lap = {lap_time:.2f} s",
                color="white", fontsize=11, ha="right",
                family="monospace", fontweight="bold")
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
        fig.canvas.draw()
        img = np.asarray(fig.canvas.buffer_rgba())[..., :3]
        plt.close(fig)
        final_frames.append(img)
    print(f"  composed {len(final_frames)} overlay frames")

    print(f"Writing MP4: {OUT}")
    try:
        with imageio.get_writer(str(OUT), fps=fps, codec="libx264",
                                  quality=8, pixelformat="yuv420p") as w:
            for f in final_frames:
                w.append_data(f)
        print(f"  wrote {OUT}  ({OUT.stat().st_size / 1e6:.1f} MB)")
    except Exception as e:
        # Fallback to GIF (smaller frames)
        print(f"  mp4 failed: {e}; writing GIF instead")
        gif_path = OUT.with_suffix(".gif")
        small = [f[::2, ::2] for f in final_frames]
        imageio.mimsave(str(gif_path), small, fps=fps, loop=0)
        print(f"  wrote {gif_path}")


if __name__ == "__main__":
    main()
