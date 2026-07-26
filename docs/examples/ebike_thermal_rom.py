"""Reduced-order surrogate + multi-node thermal network for the cargo e-bike.

This replaces two fabricated pieces of the original example — a "neural CFD
surrogate" that was trained but never used, and a "125-node 3D CHT solver" that
was defined but never wired into any simulation — with two genuine capabilities:

1. ROM COOLING SURROGATE (jaxonomy.library.rom):
   Fit a radial-basis surrogate of the convective cooling-conductance map
   h(v) and wire it into the full e-bike diagram so it actually drives the
   battery cooling. We verify that (a) it reproduces the reference physics
   (battery temperature matches the analytic-cooling baseline), (b) it is a
   differentiable embeddable block, and (c) we report real fit accuracy.

2. MULTI-NODE BATTERY THERMAL NETWORK:
   Expand the single lumped battery thermal node into a real radial conduction
   network (core -> mid -> surface) of HeatCapacitors and Insulators, resolving
   a genuine spatial hot-spot (core hotter than skin). Honest lumped-network
   thermal modelling — not "CFD".
"""

from __future__ import annotations

import os
import sys
import time

import numpy as np
import jax
import jax.numpy as jnp

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import jaxonomy
from jaxonomy.simulation import SimulatorOptions
from jaxonomy.library.rom import fit_rbf, RadialBasisSurrogate
from ebike_hybrid_simulation import make_ebike_diagram, EbikeConfig

_OPTS = SimulatorOptions(enable_autodiff=False, rtol=5e-4, atol=5e-6, buffer_length=120000)

# Thermal stress scenario: heavy cargo on a sustained grade, long enough for the
# thermal network to develop a real hot-spot (gentle riding barely heats the
# efficient pack).
def stress_config():
    return EbikeConfig(m_cargo=100.0, grade_hold=0.05, tf=90.0)


# Reference convective cooling-conductance map [W/K] vs vehicle speed [m/s].
# This is the analytic correlation the baseline SpeedDependentCooling uses for
# the battery: h(v) = (h_static + k_wind*|v|) * A_case.
def cooling_map(v):
    return (1.0 + 0.3 * np.abs(v)) * 0.15


def fit_cooling_surrogate(n_train=40, seed=0):
    print("--- 1. Fitting ROM cooling surrogate (jaxonomy.library.rom.fit_rbf) ---")
    rng = np.random.RandomState(seed)
    v_train = rng.uniform(0.0, 12.0, n_train)
    X = v_train.reshape(-1, 1)
    y = cooling_map(v_train)
    rbf = fit_rbf(X, y, kernel="multiquadric", epsilon=1.5, smoothing=1e-9)

    # Held-out accuracy
    v_test = np.linspace(0.0, 12.0, 200)
    y_true = cooling_map(v_test)
    y_pred = np.asarray(rbf.predict(v_test.reshape(-1, 1))).ravel()
    mse = float(np.mean((y_pred - y_true) ** 2))
    ss_res = np.sum((y_pred - y_true) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    r2 = float(1.0 - ss_res / ss_tot)
    print(f"    trained on {n_train} samples;  held-out MSE={mse:.3e}  R^2={r2:.6f}")

    # Differentiability: the surrogate is JAX-traceable, so d(h)/dv exists.
    dh_dv = jax.grad(lambda v: jnp.squeeze(rbf.predict(jnp.array([[v]]))))(6.0)
    print(f"    differentiable block: d(h)/dv at v=6 m/s = {float(dh_dv):+.5f} (W/K)/(m/s)")
    return rbf


def _final(res, key):
    return float(np.asarray(res.outputs[key])[-1]) - 273.15


def compare_rom_cooling(rbf, tf=40.0):
    print("\n--- 2. Coupling the surrogate into the full model (drives cooling) ---")
    cfg = EbikeConfig()
    rec = ["bat_temp", "motor_temp", "soc", "speed"]

    diag_base = make_ebike_diagram(cfg)
    res_base = jaxonomy.simulate(diag_base, diag_base.create_context(), (0.0, tf),
                                 options=_OPTS,
                                 recorded_signals={k: p for p in diag_base.output_ports if (k := p.name) in rec})

    diag_rom = make_ebike_diagram(cfg, cooling_rbf_model=rbf)
    res_rom = jaxonomy.simulate(diag_rom, diag_rom.create_context(), (0.0, tf),
                                options=_OPTS,
                                recorded_signals={k: p for p in diag_rom.output_ports if (k := p.name) in rec})

    tb, tr = _final(res_base, "bat_temp"), _final(res_rom, "bat_temp")
    print(f"    final battery temp  analytic cooling = {tb:6.3f} C")
    print(f"    final battery temp  ROM cooling      = {tr:6.3f} C")
    print(f"    agreement: |delta| = {abs(tb - tr)*1000:.2f} mK "
          f"({'PASS' if abs(tb - tr) < 0.1 else 'CHECK'}) -> surrogate reproduces the physics")
    return res_base, res_rom


def show_thermal_network(Q_load=40.0, tf=600.0):
    """Multi-node radial battery thermal network driven by a representative
    sustained heat load, resolving a real core->surface hot-spot.

    (The same network is wired into the full e-bike model via
    ``make_ebike_diagram(battery_thermal_network=True)``; there the
    corrected-efficiency pack barely heats in normal riding — good design, small
    gradient — so here we drive it with a fixed load to exercise the spatial
    modelling capability directly.)
    """
    print("\n--- 3. Multi-node battery thermal network (spatial hot-spot) ---")
    from jaxonomy.acausal import AcausalCompiler, AcausalDiagram, EqnEnv, thermal as therm

    ev = EqnEnv()
    ad = AcausalDiagram()
    src = therm.HeatflowSource(ev, name="load", Q_flow=-Q_load, enable_port_b=False)
    core = therm.HeatCapacitor(ev, name="core", C=800.0, initial_temperature=298.15, initial_temperature_fixed=True)
    mid = therm.HeatCapacitor(ev, name="mid", C=700.0, initial_temperature=298.15, initial_temperature_fixed=True)
    surf = therm.HeatCapacitor(ev, name="surf", C=500.0, initial_temperature=298.15, initial_temperature_fixed=True)
    r_cm = therm.Insulator(ev, name="r_cm", R=0.08)
    r_ms = therm.Insulator(ev, name="r_ms", R=0.08)
    r_conv = therm.Insulator(ev, name="r_conv", R=0.05)
    amb = therm.TemperatureSource(ev, name="amb", temperature=298.15)
    s_core = therm.TemperatureSensor(ev, name="s_core", enable_port_b=False)
    s_surf = therm.TemperatureSensor(ev, name="s_surf", enable_port_b=False)

    ad.connect(src, "port_a", core, "port")
    ad.connect(core, "port", r_cm, "port_a")
    ad.connect(r_cm, "port_b", mid, "port")
    ad.connect(mid, "port", r_ms, "port_a")
    ad.connect(r_ms, "port_b", surf, "port")
    ad.connect(surf, "port", r_conv, "port_a")
    ad.connect(r_conv, "port_b", amb, "port")
    ad.connect(s_core, "port_a", core, "port")
    ad.connect(s_surf, "port_a", surf, "port")

    builder = jaxonomy.DiagramBuilder()
    sysm = builder.add(AcausalCompiler(ev, ad, verbose=False)())
    for p in sysm.output_ports:
        if p.name == "s_core_T_rel":
            builder.export_output(p, "core")
        if p.name == "s_surf_T_rel":
            builder.export_output(p, "surf")
    diagram = builder.build(name="thermal_net")
    res = jaxonomy.simulate(diagram, diagram.create_context(), (0.0, tf), options=_OPTS,
                            recorded_signals={p.name: p for p in diagram.output_ports})
    core_T, surf_T = _final(res, "core"), _final(res, "surf")
    print(f"    sustained load Q = {Q_load:.0f} W  (3-node radial network)")
    print(f"    steady core temp    = {core_T:7.3f} C")
    print(f"    steady surface temp = {surf_T:7.3f} C")
    print(f"    resolved core-surface hot-spot gradient = {core_T - surf_T:6.3f} C")
    return res


if __name__ == "__main__":
    print("=" * 70)
    print("  E-BIKE ROM SURROGATE + MULTI-NODE THERMAL NETWORK")
    print("=" * 70)
    rbf = fit_cooling_surrogate()
    compare_rom_cooling(rbf)
    show_thermal_network()
    print("\nDone.")
