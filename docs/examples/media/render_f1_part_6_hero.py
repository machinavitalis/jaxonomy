# SPDX-License-Identifier: MIT
"""F1 Part 6 hero MP4 — placeholder renderer.

This produces the marketing-climax MP4 that closes the 6-part F1 series.
It is a PLACEHOLDER artifact:

  - It uses MuJoCo to render a stylised F1 chassis (the same stylised car
    body from Parts 1+2) rotating against a sky background.
  - The rear wing is coloured with a SYNTHETIC pressure-coefficient field
    that animates across the three "design iterations" (init, mid, opt).
  - A lap-time clock counts down from 89.4 s (baseline) to 87.6 s (opt) in
    the corner.
  - The pressure colours are NOT from real CFD — they are a closed-form
    analytic Cp model parametrised so the leading edge is high-Cp and the
    upper rear is low-Cp (downforce signal). Direction-correct; magnitude
    not real.

The real publication pipeline (media/f1_part_6_publication_offline.sh)
replaces this with per-iteration ParaView surface stills composited with
Blender Cycles ray-tracing. That pipeline requires SU2 + OpenVSP + Blender
+ ParaView installed locally.

Purpose: render the hero MP4 for visualization.
No physics — mj_forward only.

Run:
    python docs/examples/media/render_f1_part_6_hero.py

Output:
    docs/examples/media/f1_part_6_hero.mp4
"""
from __future__ import annotations

from pathlib import Path
import time

import numpy as np
import mujoco

try:
    import imageio.v2 as imageio
except ImportError:
    import imageio

HERE = Path(__file__).resolve().parent
OUT = HERE / "f1_part_6_hero.mp4"

# ──────────────────────────────────────────────────────────────────────────
# Car geometry constants (matched to Parts 1+2's stylised F1)
# ──────────────────────────────────────────────────────────────────────────
L_CAR = 5.50
W_CAR = 1.80
H_CAR = 0.60
WHEEL_R = 0.34
WHEEL_W = 0.36
TRACK_W = 1.60
WHEELBASE = 3.25

# Per-iteration wing colour and geometry signature.
# Iter 0: small flap, modest gurney (baseline)
# Iter 4: mid flap, mid gurney
# Iter 8: large flap, large gurney (lap-opt)
ITER_LABELS = ["iter 0 (baseline)", "iter 4 (mid)", "iter 8 (lap-opt)"]
# Lap times from the placeholder NPZ
LAP_HISTORY = np.array([89.4, 88.4, 87.6])
# Wing geom signature (flap_deflection_deg, gurney_height_mm)
WING_GEOMS = np.array([(5.0, 8.0), (12.0, 12.0), (22.0, 18.0)])


def synthetic_Cp(x, y, z):
    """Synthetic pressure-coefficient field on a wing surface.

    Leading edge (-y direction in the rear-wing-local frame) is high Cp;
    trailing-edge underside is low Cp (downforce signal). Returns a value
    in [-1.5, +1.2].
    """
    # Normalise z to [0, 1] over the rear-wing region
    z_norm = np.clip((z - H_CAR - 0.30) / 0.30, 0.0, 1.0)
    # Linear front-to-back gradient + dip on the rear
    leading = np.exp(-3.0 * np.abs(x + L_CAR / 2 - 0.4))
    Cp = -0.6 + 1.4 * leading - 0.4 * z_norm
    return float(np.clip(Cp, -1.5, +1.2))


def cp_to_rgba(Cp_val):
    """Map Cp in [-1.5, 1.2] to a coolwarm-ish RGBA. Cp<0 = blue (downforce);
    Cp>0 = red (high pressure)."""
    t = (Cp_val + 1.5) / 2.7  # to [0, 1]
    t = np.clip(t, 0, 1)
    # Coolwarm-ish manual interp: blue -> white -> red
    if t < 0.5:
        s = 2 * t
        r = 0.20 + 0.80 * s
        g = 0.30 + 0.70 * s
        b = 0.85
    else:
        s = 2 * (t - 0.5)
        r = 0.95
        g = 0.85 - 0.55 * s
        b = 0.85 - 0.80 * s
    return f"{r:.2f} {g:.2f} {b:.2f} 1"


def build_mjcf(iter_idx: int) -> str:
    """Build the MJCF for the hero scene with iter-specific wing colouring."""
    flap_deg, gurney_mm = WING_GEOMS[iter_idx]
    # Wing position: rotate the wing element by the flap-angle so the
    # geometry visibly changes between iterations
    flap_rad = np.deg2rad(flap_deg)
    cq, sq = np.cos(flap_rad / 2), np.sin(flap_rad / 2)
    wing_quat_y = f"{cq:.4f} 0 {sq:.4f} 0"  # rotate about y-axis
    # Gurney height as a small visible lip on the trailing edge
    gurney_h = gurney_mm / 1000.0

    # Per-segment Cp -> colour. Six wing segments along x.
    n_seg = 6
    seg_xs = np.linspace(-L_CAR / 2 + 0.10, -L_CAR / 2 + 0.45, n_seg)
    wing_segs_xml = []
    for ix, xs in enumerate(seg_xs):
        Cp = synthetic_Cp(xs, 0.0, H_CAR + 0.45)
        # Shift Cp range per iteration: higher-downforce iter -> more negative Cp
        Cp -= 0.2 * iter_idx
        rgba = cp_to_rgba(Cp)
        wing_segs_xml.append(
            f'<geom name="wing_seg_{ix}" type="box" '
            f'size="0.025 {W_CAR/2 + 0.30:.3f} 0.03" '
            f'pos="{xs:.3f} 0 {H_CAR + 0.45:.3f}" '
            f'quat="{wing_quat_y}" '
            f'rgba="{rgba}"/>'
        )
    wing_block = "\n      ".join(wing_segs_xml)

    return f"""
<mujoco model="f1_part6_hero">
  <option gravity="0 0 -9.81" timestep="0.005"/>
  <visual>
    <headlight diffuse=".9 .9 .9" ambient=".30 .30 .30" specular="0.05 0.05 0.05"/>
    <rgba haze="0.30 0.36 0.45 1"/>
    <quality shadowsize="4096"/>
    <global offwidth="1280" offheight="720"/>
  </visual>
  <asset>
    <texture type="skybox" builtin="gradient"
             rgb1="0.55 0.70 0.92" rgb2="0.15 0.20 0.35"
             width="800" height="800"/>
    <texture name="floor_tex" type="2d" builtin="checker"
             rgb1="0.30 0.30 0.32" rgb2="0.18 0.18 0.20"
             width="300" height="300"/>
    <material name="floor" texture="floor_tex" texrepeat="40 40"
              reflectance="0.10" specular="0.20" shininess="0.20"/>
    <material name="chassis" rgba="0.80 0.10 0.10 1" specular="0.7" shininess="0.7"/>
    <material name="halo"    rgba="0.05 0.05 0.05 1" specular="0.4" shininess="0.5"/>
    <material name="tyre"    rgba="0.05 0.05 0.05 1" specular="0.2" shininess="0.2"/>
    <material name="wing_dark"  rgba="0.10 0.10 0.12 1" specular="0.4" shininess="0.5"/>
    <material name="endplate" rgba="0.18 0.18 0.20 1" specular="0.5" shininess="0.6"/>
    <material name="gurney"   rgba="0.95 0.80 0.10 1" specular="0.7" shininess="0.8"/>
  </asset>
  <worldbody>
    <light name="sun"  diffuse="1.0 0.95 0.85" pos="20 -20 25" dir="-0.6 0.6 -0.8"
           castshadow="true"/>
    <light name="fill" diffuse="0.35 0.40 0.55" pos="-15 15 18" dir="0.5 -0.5 -0.6"
           castshadow="false"/>
    <geom name="ground" type="plane" pos="0 0 0" size="30 30 1" material="floor"/>

    <body name="car" pos="0 0 {WHEEL_R + 0.02}">
      <joint name="free" type="free"/>
      <inertial pos="0 0 0" mass="830" diaginertia="200 1350 1500"/>
      <geom name="chassis" type="box"
            size="{L_CAR/2:.3f} {W_CAR/2:.3f} {H_CAR/2:.3f}"
            pos="0 0 {H_CAR/2 + 0.05}"
            material="chassis"/>
      <geom name="halo" type="capsule"
            fromto="0 {-W_CAR/2 + 0.1} {H_CAR + 0.10}
                    0 { W_CAR/2 - 0.1} {H_CAR + 0.10}"
            size="0.05" material="halo"/>
      <!-- Rear-wing segments coloured by synthetic Cp -->
      {wing_block}
      <!-- Wing endplates -->
      <geom name="rear_wing_endL" type="box" size="0.18 0.04 0.30"
            pos="{-L_CAR/2 + 0.275:.3f} { W_CAR/2 + 0.30:.3f} {H_CAR + 0.20}"
            material="endplate"/>
      <geom name="rear_wing_endR" type="box" size="0.18 0.04 0.30"
            pos="{-L_CAR/2 + 0.275:.3f} {-W_CAR/2 - 0.30:.3f} {H_CAR + 0.20}"
            material="endplate"/>
      <!-- Gurney flap on the trailing edge (visible bright strip) -->
      <geom name="gurney_strip" type="box"
            size="0.012 {W_CAR/2 + 0.28:.3f} {gurney_h:.4f}"
            pos="{-L_CAR/2 + 0.10:.3f} 0 {H_CAR + 0.48 + gurney_h:.4f}"
            material="gurney"/>
      <!-- Front wing -->
      <geom name="front_wing" type="box"
            size="0.08 {W_CAR/2 + 0.30:.3f} 0.04"
            pos="{L_CAR/2 - 0.05:.3f} 0 0.10"
            material="wing_dark"/>
      <!-- Four wheels -->
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


def yaw_to_quat(psi: float) -> np.ndarray:
    return np.array([np.cos(psi / 2), 0.0, 0.0, np.sin(psi / 2)])


def main(fps: int = 30, n_seconds: float = 24.0):
    """Render the hero MP4 — car rotating with synthetic-pressure wing colouring.

    Three segments of n_seconds/3 each, one per design iteration (baseline, mid, opt).
    Each segment: car rotates 360° on a stage, wing colour shows the synth Cp field,
    flap and gurney visibly grow across segments.
    """
    n_frames_per_seg = int(n_seconds / 3 * fps)
    n_frames_total = 3 * n_frames_per_seg
    print(f"Will render {n_frames_total} frames @ {fps} fps -> {n_frames_total/fps:.1f} s of MP4")

    # ────────────────────────────────────────────────────────────────────────
    # Render each segment with the right MJCF, store frames.
    # ────────────────────────────────────────────────────────────────────────
    all_frames = []
    seg_lap_overlay = []  # which iter + which lap-time the overlay shows

    for seg_idx in range(3):
        mjcf_str = build_mjcf(seg_idx)
        if seg_idx == 0:
            (HERE / "f1_part_6_scene.xml").write_text(mjcf_str)
        try:
            model = mujoco.MjModel.from_xml_string(mjcf_str)
        except Exception as e:
            print(f"MJCF parse failed at seg {seg_idx}: {e}")
            raise
        data = mujoco.MjData(model)
        renderer = mujoco.Renderer(model, height=720, width=1280)

        cam = mujoco.MjvCamera()
        cam.type = mujoco.mjtCamera.mjCAMERA_FREE

        mujoco.mj_forward(model, data)

        t0 = time.time()
        for f_idx in range(n_frames_per_seg):
            # Rotate the *camera* around the stationary car
            phase = f_idx / n_frames_per_seg
            cam_az = -90.0 + 360.0 * phase  # one full rotation per segment
            cam_el = -18.0 + 6.0 * np.sin(2 * np.pi * phase)  # small vertical bobble

            cam.lookat = np.array([0.0, 0.0, 0.7])
            cam.distance = 7.5
            cam.azimuth = cam_az
            cam.elevation = cam_el

            renderer.update_scene(data, camera=cam)
            rgb = renderer.render().copy()
            all_frames.append(rgb)
            seg_lap_overlay.append((seg_idx, phase))

        print(f"  seg {seg_idx} ({ITER_LABELS[seg_idx]}) rendered in {time.time()-t0:.1f}s")

    # ────────────────────────────────────────────────────────────────────────
    # Burn-in overlays — lap-time clock, iter label, marketing footer.
    # ────────────────────────────────────────────────────────────────────────
    print("Burning in HUD overlays via matplotlib ...")
    import matplotlib.pyplot as plt
    final_frames = []
    for f_idx, rgb in enumerate(all_frames):
        seg_idx, phase = seg_lap_overlay[f_idx]
        # Smooth lap-time interp between segments
        if seg_idx == 0:
            lap_shown = LAP_HISTORY[0] - (LAP_HISTORY[0] - LAP_HISTORY[1]) * phase
        elif seg_idx == 1:
            lap_shown = LAP_HISTORY[1] - (LAP_HISTORY[1] - LAP_HISTORY[2]) * phase
        else:
            lap_shown = LAP_HISTORY[2]

        fig, ax = plt.subplots(figsize=(12.8, 7.2), dpi=100)
        ax.imshow(rgb)
        ax.set_axis_off()
        # Iter label (top-left)
        ax.text(
            18, 30, ITER_LABELS[seg_idx],
            color="white", fontsize=15, fontweight="bold",
            family="monospace",
            bbox=dict(facecolor="black", alpha=0.55, pad=4, edgecolor="none"),
        )
        # Lap clock (top-right)
        ax.text(
            1262, 30, f"lap = {lap_shown:5.2f} s",
            color="#ffe89c", fontsize=18, fontweight="bold",
            ha="right", family="monospace",
            bbox=dict(facecolor="black", alpha=0.65, pad=6, edgecolor="none"),
        )
        # Marketing footer (bottom-center)
        ax.text(
            640, 695, "jaxonomy F1 part 6  |  DrivAerML + OpenVSP + SU2 + lifting-line surrogate",
            color="white", fontsize=12, ha="center",
            family="monospace", fontweight="bold",
            bbox=dict(facecolor="black", alpha=0.50, pad=4, edgecolor="none"),
        )
        # Synthetic-Cp legend strip (bottom-left)
        ax.text(
            18, 670, "rear wing colour = synthetic Cp  (red=high pressure, blue=downforce)",
            color="white", fontsize=10, family="monospace",
            bbox=dict(facecolor="black", alpha=0.45, pad=3, edgecolor="none"),
        )
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
        fig.canvas.draw()
        img = np.asarray(fig.canvas.buffer_rgba())[..., :3]
        plt.close(fig)
        final_frames.append(img)

    print(f"Composed {len(final_frames)} overlay frames")

    print(f"Writing MP4: {OUT}")
    try:
        with imageio.get_writer(
            str(OUT), fps=fps, codec="libx264",
            quality=8, pixelformat="yuv420p",
        ) as w:
            for f in final_frames:
                w.append_data(f)
        print(f"  wrote {OUT}  ({OUT.stat().st_size / 1e6:.1f} MB)")
    except Exception as e:
        print(f"  mp4 failed: {e}; writing GIF instead")
        gif_path = OUT.with_suffix(".gif")
        small = [f[::2, ::2] for f in final_frames]
        imageio.mimsave(str(gif_path), small, fps=fps, loop=0)
        print(f"  wrote {gif_path}")


if __name__ == "__main__":
    main()
