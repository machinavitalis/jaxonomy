"""Compare the 3-D MuJoCo cargo e-bike co-simulation against the 2-D planar
Jaxonomy model — what the extra out-of-plane degrees of freedom actually add.

The Jaxonomy ``PlanarVehicleDynamics`` is a bird's-eye 3-DOF vehicle
(longitudinal, lateral, yaw). It has *no vertical, pitch, or contact* dynamics.
The MuJoCo companion (``ebike_mujoco_cosim.py``) resolves exactly that sagittal
plane. Driven by the same assist-torque law + cutoff, both produce consistent
longitudinal behaviour; only the 3-D model shows the pitch and suspension
response.

Run (loads the MuJoCo NPZ, runs the Jaxonomy planar model on flat ground):
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
    """Jaxonomy planar model, flat ground, same ~15 s window."""
    cfg = EbikeConfig(grade_hold=0.0, tf=tf)
    diagram = make_ebike_diagram(cfg)
    opts = SimulatorOptions(enable_autodiff=False, rtol=5e-4, atol=5e-6, buffer_length=120000)
    rec = {p.name: p for p in diagram.output_ports if p.name in ("speed",)}
    res = jaxonomy.simulate(diagram, diagram.create_context(), (0.0, tf), options=opts,
                            recorded_signals=rec)
    return np.asarray(res.time), np.asarray(res.outputs["speed"]).squeeze()


def main():
    mj = np.load(MJ_NPZ)
    t_mj = mj["t"]
    tf = float(t_mj[-1])

    print("Running Jaxonomy planar model on flat ground for comparison...")
    t_pl, v_pl = run_planar_flat(tf)

    fig, axs = plt.subplots(1, 3, figsize=(16, 4.5))

    # 1) Longitudinal speed — both models
    axs[0].plot(t_mj, mj["speed_mps"] * 3.6, color="tab:blue", lw=1.8, label="MuJoCo 3-D")
    axs[0].plot(t_pl, v_pl * 3.6, color="tab:orange", lw=1.8, ls="--", label="Jaxonomy planar")
    axs[0].axhline(25.0, color="tab:red", ls=":", lw=1.0, label="25 km/h cutoff")
    axs[0].set_xlabel("time (s)"); axs[0].set_ylabel("speed (km/h)")
    axs[0].set_title("Longitudinal speed — consistent"); axs[0].legend(fontsize=8)

    # 2) Pitch — MuJoCo only
    axs[1].plot(t_mj, np.degrees(mj["pitch_rad"]), color="tab:blue", lw=1.6, label="MuJoCo 3-D")
    axs[1].axhline(0.0, color="tab:orange", lw=1.8, ls="--", label="Jaxonomy planar (no pitch DOF)")
    axs[1].set_xlabel("time (s)"); axs[1].set_ylabel("pitch angle (deg)")
    axs[1].set_title("Chassis pitch — 3-D only"); axs[1].legend(fontsize=8)

    # 3) Rear suspension travel — MuJoCo only
    axs[2].plot(t_mj, mj["susp_travel_m"] * 1000.0, color="tab:blue", lw=1.6, label="MuJoCo 3-D")
    axs[2].axhline(0.0, color="tab:orange", lw=1.8, ls="--", label="Jaxonomy planar (no vertical DOF)")
    axs[2].set_xlabel("time (s)"); axs[2].set_ylabel("rear suspension travel (mm)")
    axs[2].set_title("Suspension over the bump — 3-D only"); axs[2].legend(fontsize=8)

    fig.suptitle("3-D MuJoCo co-sim vs 2-D planar Jaxonomy model — the out-of-plane dynamics",
                 fontsize=13, weight="bold")
    fig.tight_layout()
    fig.savefig(PNG, dpi=130)

    print(f"  planar max speed  : {v_pl.max()*3.6:.1f} km/h")
    print(f"  MuJoCo max speed  : {mj['speed_mps'].max()*3.6:.1f} km/h")
    print(f"  MuJoCo pitch range: [{np.degrees(mj['pitch_rad']).min():.1f}, "
          f"{np.degrees(mj['pitch_rad']).max():.1f}] deg  (planar: identically 0)")
    print(f"  MuJoCo rear susp  : [{mj['susp_travel_m'].min()*1000:.0f}, "
          f"{mj['susp_travel_m'].max()*1000:.0f}] mm  (planar: identically 0)")
    print(f"Wrote {PNG}")


if __name__ == "__main__":
    main()
