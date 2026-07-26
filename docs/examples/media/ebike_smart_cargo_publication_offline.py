#!/usr/bin/env python3
"""Offline publication run for ebike_smart_cargo.ipynb.

Runs the full reference drive cycle (event-enabled) plus a small assist-cap
sweep at full fidelity, and writes a compact checkpoint
``media/ebike_smart_cargo_publication.npz`` (< 100 KB) plus the hero telemetry
figure ``media/ebike_smart_cargo_telemetry.png``. The notebook loads the NPZ in
< 1 s so the reader's experience stays fast (a single reference rollout is
~40-60 s of CPU).

Run from docs/examples:
    python media/ebike_smart_cargo_publication_offline.py
"""

import os
import sys
import time

import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import jaxonomy
from jaxonomy.simulation import SimulatorOptions
from ebike_hybrid_simulation import (
    make_ebike_diagram, simulate_ebike, energy_audit, EbikeConfig,
)

HERE = os.path.dirname(os.path.abspath(__file__))
NPZ = os.path.join(HERE, "ebike_smart_cargo_publication.npz")
PNG = os.path.join(HERE, "ebike_smart_cargo_telemetry.png")

N_DS = 2000  # downsample the recorded traces to this many points for the NPZ


def _ds(arr, idx):
    return np.asarray(arr).squeeze()[idx]


def main():
    cfg = EbikeConfig()

    print("Running reference drive cycle (event-enabled)...")
    t0 = time.time()
    res = simulate_ebike(cfg)
    wall = time.time() - t0
    audit = energy_audit(res, cfg)
    print(f"  done in {wall:.1f} s;  energy closure = {audit['closure_error_pct']:.2f} %")

    t = np.asarray(res.time)
    idx = np.unique(np.linspace(0, len(t) - 1, N_DS).astype(int))
    o = res.outputs

    traces = {
        "t": t[idx],
        "speed_kmh": _ds(o["speed"], idx) * 3.6,
        "soc": _ds(o["soc"], idx),
        "T_stator_C": _ds(o["T_stator"], idx) - 273.15,
        "bat_temp_C": _ds(o["bat_temp"], idx) - 273.15,
        "cadence_rpm": _ds(o["cadence"], idx) * 60.0 / (2 * np.pi),
        "assist_enable": _ds(o["assist_enable"], idx),
        "pos_x": _ds(o["pos_x"], idx),
        "pos_y": _ds(o["pos_y"], idx),
        "v_dc": _ds(o["v_dc"], idx),
        "iq_curr": _ds(o["iq_curr"], idx),
        "human_trq": _ds(o["human_trq"], idx),
    }

    # Assist cutoff event time (first 1->0 transition of assist_enable)
    en = np.asarray(o["assist_enable"]).squeeze()
    spd = np.asarray(o["speed"]).squeeze() * 3.6
    tr = np.where(np.abs(np.diff(en)) > 0.5)[0]
    cutoff_t = float(t[tr[0] + 1]) if len(tr) else float("nan")
    cutoff_v = float(spd[tr[0] + 1]) if len(tr) else float("nan")

    audit_terms = {k: float(v) for k, v in audit.items()}

    # Small assist-cap sweep (for the optimization figure), full fidelity.
    print("Assist-cap sweep for the optimization figure...")
    diagram, handles = make_ebike_diagram(cfg, return_handles=True, enable_speed_event=True)
    ctx = diagram.create_context()
    aid = handles["assist_policy"].system_id
    ports = {p.name: p for p in diagram.output_ports}
    opts = SimulatorOptions(enable_autodiff=False, rtol=5e-4, atol=5e-6, buffer_length=120000)
    caps = np.array([4.0, 8.0, 12.0, 16.0, 20.0])
    TF_OPT = 30.0
    E_batt, v_mean = [], []
    import jax.numpy as jnp
    for cap in caps:
        c = ctx.with_subcontext(aid, ctx[aid].with_parameter("max_assist_torque", jnp.array(float(cap))))
        r = jaxonomy.simulate(diagram, c, (0.0, TF_OPT), options=opts,
                              recorded_signals={"E": ports["E_batt_term"], "d": ports["distance"]})
        E_batt.append(float(np.asarray(r.outputs["E"])[-1]))
        v_mean.append(float(np.asarray(r.outputs["d"])[-1]) / TF_OPT * 3.6)
        print(f"  cap={cap:4.1f}  E_batt={E_batt[-1]:8.1f} J  v_mean={v_mean[-1]:5.2f} km/h")

    np.savez(
        NPZ,
        wall_time_s=wall,
        cutoff_t=cutoff_t, cutoff_v=cutoff_v,
        sweep_caps=caps, sweep_E_batt=np.array(E_batt), sweep_v_mean=np.array(v_mean),
        sweep_tf=TF_OPT,
        **{f"trace_{k}": v for k, v in traces.items()},
        **{f"audit_{k}": v for k, v in audit_terms.items()},
    )
    sz = os.path.getsize(NPZ) / 1024
    print(f"Wrote {NPZ}  ({sz:.1f} KB)")

    # ---- Hero telemetry figure -------------------------------------------
    fig, axs = plt.subplots(2, 3, figsize=(15, 8))
    T = traces["t"]

    ax = axs[0, 0]
    ax.plot(T, traces["speed_kmh"], color="tab:blue", lw=1.8, label="vehicle speed")
    ax.axhline(25.0, color="tab:red", ls="--", lw=1.0, label="25 km/h legal cutoff")
    ax.fill_between(T, 0, traces["speed_kmh"].max() * 1.05,
                    where=traces["assist_enable"] > 0.5, color="tab:green", alpha=0.12,
                    label="motor assist ON")
    ax.axvline(cutoff_t, color="tab:red", ls=":", lw=1.0)
    ax.set_xlabel("time (s)"); ax.set_ylabel("speed (km/h)")
    ax.set_title("Speed & assist-cutoff event"); ax.legend(fontsize=8, loc="lower right")

    ax = axs[0, 1]
    ax.plot(T, traces["soc"], color="tab:green", lw=1.8)
    ax.set_xlabel("time (s)"); ax.set_ylabel("state of charge")
    ax.set_title("Battery SOC")

    ax = axs[0, 2]
    ax.plot(T, traces["T_stator_C"], color="tab:red", lw=1.6, label="motor stator")
    ax.plot(T, traces["bat_temp_C"], color="tab:blue", lw=1.6, label="battery")
    ax.set_xlabel("time (s)"); ax.set_ylabel("temperature (°C)")
    ax.set_title("Thermal response"); ax.legend(fontsize=8)

    ax = axs[1, 0]
    ax.plot(T, traces["cadence_rpm"], color="tab:purple", lw=1.6)
    ax.set_xlabel("time (s)"); ax.set_ylabel("cadence (rpm)")
    ax.set_title("Pedalling cadence (W′-balance rider)")

    ax = axs[1, 1]
    labels = ["human", "battery", "ΔKE", "grade", "aero", "rolling", "motor heat"]
    vals = [audit_terms["E_human"], audit_terms["E_batt_term"], audit_terms["dKE"],
            audit_terms["E_climb"], audit_terms["E_aero"], audit_terms["E_roll"],
            audit_terms["E_motor_heat"]]
    colors = ["tab:green", "tab:green", "tab:gray", "tab:gray", "tab:orange", "tab:orange", "tab:red"]
    ax.bar(range(len(labels)), np.array(vals) / 1000.0, color=colors)
    ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, rotation=40, ha="right", fontsize=8)
    ax.set_ylabel("energy (kJ)")
    ax.set_title(f"Energy audit (closes to {audit_terms['closure_error_pct']:.1f} %)")

    ax = axs[1, 2]
    ax.plot(traces["pos_x"], traces["pos_y"], color="tab:cyan", lw=1.8)
    ax.plot(traces["pos_x"][0], traces["pos_y"][0], "go", ms=6, label="start")
    ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
    ax.set_title("Ground track"); ax.legend(fontsize=8); ax.axis("equal")

    fig.suptitle("Smart Cargo E-Bike — reference drive cycle telemetry", fontsize=15, weight="bold")
    fig.tight_layout()
    fig.savefig(PNG, dpi=130)
    print(f"Wrote {PNG}")


if __name__ == "__main__":
    main()
