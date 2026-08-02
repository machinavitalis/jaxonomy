"""Compare the 3-D MuJoCo cargo e-bike simulation against the 2-D planar
Jaxonomy model — what the extra out-of-plane degrees of freedom actually add.

The comparison is split the only honest way:

1. **Like-for-like longitudinal check, flat road.** Both models get the same
   drive law, mass (180 kg), CdA, and Crr on flat ground; the speed traces are
   compared and the RMS/max deviation is *printed*, not asserted. Remaining
   differences are real modeling differences (drivetrain compliance and motor
   dynamics in the planar model; contact and suspension kinematics in MuJoCo).
2. **What only 3-D resolves, terrain run.** Total chassis pitch is decomposed
   into the road's geometric pitch (the wheelbase chord angle — an *input*,
   which the planar model also knows via its grade signal) and the
   chassis-relative pitch (squat under drive, suspension response over the
   speed hump) — the part a planar model structurally lacks, and honestly ~1-2
   degrees, not the road-slope-sized angles a naive reading of total pitch
   would suggest.

Run (loads the MuJoCo NPZ written by ebike_mujoco_cosim.py, runs the Jaxonomy
planar model on flat ground):
    python docs/examples/ebike_mujoco_vs_planar.py
"""

from __future__ import annotations

import os
import sys

import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import jaxonomy
from jaxonomy.simulation import SimulatorOptions
from ebike_hybrid_simulation import make_ebike_diagram, EbikeConfig

HERE = os.path.dirname(os.path.abspath(__file__))
MJ_NPZ = os.path.join(HERE, "media", "ebike_mujoco_cosim.npz")
PNG = os.path.join(HERE, "media", "ebike_mujoco_vs_planar.png")


def run_planar_flat(tf):
    """Jaxonomy planar model, flat ground, same time window."""
    cfg = EbikeConfig(grade_hold=0.0, tf=tf)
    diagram = make_ebike_diagram(cfg)
    opts = SimulatorOptions(enable_autodiff=False, rtol=5e-4, atol=5e-6, buffer_length=260000)
    rec = {p.name: p for p in diagram.output_ports if p.name in ("speed",)}
    res = jaxonomy.simulate(diagram, diagram.create_context(), (0.0, tf), options=opts,
                            recorded_signals=rec)
    return (np.asarray(res.time).squeeze(),
            np.asarray(res.outputs["speed"]).squeeze(), cfg)


def main():
    mj = np.load(MJ_NPZ)
    t_mj = mj["flat_t"]
    v_mj = mj["flat_speed_mps"]
    tf = float(t_mj[-1])

    print("Running Jaxonomy planar model on flat ground for comparison...")
    t_pl, v_pl, cfg = run_planar_flat(tf)

    # ---- quantified agreement on the flat like-for-like run ---------------
    v_pl_i = np.interp(t_mj, t_pl, v_pl)
    dev = (v_mj - v_pl_i) * 3.6
    rms_kmh = float(np.sqrt(np.mean(dev**2)))
    max_kmh = float(np.max(np.abs(dev)))
    m_mj = float(mj["flat_total_mass_kg"])
    print(f"  masses            : planar {cfg.m_total:.1f} kg, MuJoCo {m_mj:.1f} kg")
    print(f"  flat-road speeds  : planar max {v_pl.max()*3.6:.1f} km/h, "
          f"MuJoCo max {v_mj.max()*3.6:.1f} km/h")
    print(f"  speed deviation   : RMS {rms_kmh:.2f} km/h, max {max_kmh:.2f} km/h "
          f"(over {tf:.0f} s; drivetrain-model differences, quantified not asserted)")

    # ---- figure -----------------------------------------------------------
    fig, axs = plt.subplots(1, 3, figsize=(16, 4.5))

    # 1) Longitudinal speed — both models, flat road
    axs[0].plot(t_mj, v_mj * 3.6, color="tab:blue", lw=1.8, label="MuJoCo 3-D (flat)")
    axs[0].plot(t_pl, v_pl * 3.6, color="tab:orange", lw=1.8, ls="--", label="Jaxonomy planar (flat)")
    axs[0].axhline(25.0, color="tab:red", ls=":", lw=1.0, label="25 km/h cutoff")
    axs[0].set_xlabel("time (s)"); axs[0].set_ylabel("speed (km/h)")
    axs[0].set_title(f"Flat-road speed — RMS dev {rms_kmh:.1f} km/h, max {max_kmh:.1f}")
    axs[0].legend(fontsize=8)

    # 2) Pitch decomposition — terrain run (nose-up positive)
    axs[1].plot(mj["t"], np.degrees(mj["pitch_rad"]), color="tab:blue", lw=1.4,
                label="total chassis pitch")
    axs[1].plot(mj["t"], np.degrees(mj["road_pitch_rad"]), color="tab:gray", lw=1.2, ls=":",
                label="road pitch (wheelbase chord)")
    axs[1].plot(mj["t"], np.degrees(mj["pitch_rel_rad"]), color="tab:red", lw=1.6,
                label="chassis-relative (3-D-only content)")
    axs[1].axhline(0.0, color="k", lw=0.6)
    axs[1].set_xlabel("time (s)"); axs[1].set_ylabel("pitch, nose-up + (deg)")
    axs[1].set_title("Terrain run: total pitch = road + chassis response")
    axs[1].legend(fontsize=8)

    # 3) Suspension travel — terrain run
    axs[2].plot(mj["t"], mj["susp_travel_m"] * 1000.0, color="tab:blue", lw=1.4,
                label="rear (driven)")
    axs[2].plot(mj["t"], mj["front_travel_m"] * 1000.0, color="tab:green", lw=1.2,
                label="front")
    axs[2].axhline(0.0, color="tab:orange", lw=1.6, ls="--",
                   label="Jaxonomy planar (no vertical DOF)")
    axs[2].set_xlabel("time (s)"); axs[2].set_ylabel("suspension travel (mm, + = compression)")
    axs[2].set_title("Suspension over hump/ramps — 3-D only")
    axs[2].legend(fontsize=8)

    fig.suptitle("3-D MuJoCo vs 2-D planar Jaxonomy — agreement quantified, additions decomposed",
                 fontsize=13, weight="bold")
    fig.tight_layout()
    fig.savefig(PNG, dpi=130)

    print(f"  terrain pitch     : total [{np.degrees(mj['pitch_rad']).min():+.1f}, "
          f"{np.degrees(mj['pitch_rad']).max():+.1f}] deg = road "
          f"[{np.degrees(mj['road_pitch_rad']).min():+.1f}, "
          f"{np.degrees(mj['road_pitch_rad']).max():+.1f}] + chassis-rel "
          f"[{np.degrees(mj['pitch_rel_rad']).min():+.1f}, "
          f"{np.degrees(mj['pitch_rel_rad']).max():+.1f}]")
    print(f"  rear susp travel  : [{mj['susp_travel_m'].min()*1000:.0f}, "
          f"{mj['susp_travel_m'].max()*1000:.0f}] mm (planar: identically 0)")
    print(f"Wrote {PNG}")


if __name__ == "__main__":
    main()
