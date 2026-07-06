#!/usr/bin/env python3
"""Offline publication-quality run for ``fmi_export_roundtrip.ipynb``.

Produces ``media/fmi_export_publication.npz`` from which the notebook
loads its headline numbers (closed-loop trajectories from the FMU- vs
in-process controller, manual-orchestration bit-comparison, FMU step
profile) when the live FMU build or import path is unavailable
(missing ``pythonfmu`` / ``fmpy`` / ``libpythonfmu-export.dylib``).

The notebook itself **always tries the live path first**. The
publication NPZ is only consulted if the live build fails — in which
case the prose cells point at this script's headline numbers so the
reader can still see the wedge.

Run from the repo root:

.. code-block:: bash

    JAXONOMY_DISABLE_PROFILING=1 python docs/examples/media/fmi_export_publication_offline.py

Expected wall-time: **~5-10 s on a developer machine** (the FMU build
is ~0.1 s on darwin, the closed-loop simulations are sub-second; the
script is dominated by JIT-compile of two short jaxonomy diagrams).
"""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import textwrap
import time
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
NPZ_OUT = HERE / "fmi_export_publication.npz"

# Top-level constants must match the notebook.
DT = 0.01
T_END = 4.0
KP = 1.5
KI = 0.4
SETPOINT = 1.0
K_PLANT = 1.0
N_STEPS = int(round(T_END / DT))


def _build_controller_diagram():
    import jaxonomy
    from jaxonomy.library import Constant, Adder, Gain
    from jaxonomy.library.dynamics import IntegratorDiscrete

    bld = jaxonomy.DiagramBuilder()
    sp = bld.add(Constant(SETPOINT, name="setpoint"))
    err = bld.add(Adder(2, operators="+-", name="err"))
    kp = bld.add(Gain(KP, name="kp_gain"))
    ki = bld.add(Gain(KI, name="ki_gain"))
    integ = bld.add(IntegratorDiscrete(dt=DT, initial_state=0.0, name="ierr"))
    add_u = bld.add(Adder(2, operators="++", name="add_u"))
    bld.connect(sp.output_ports[0], err.input_ports[0])
    bld.connect(err.output_ports[0], kp.input_ports[0])
    bld.connect(err.output_ports[0], integ.input_ports[0])
    bld.connect(integ.output_ports[0], ki.input_ports[0])
    bld.connect(kp.output_ports[0], add_u.input_ports[0])
    bld.connect(ki.output_ports[0], add_u.input_ports[1])
    bld.export_output(add_u.output_ports[0], name="u")
    bld.export_input(err.input_ports[1], name="measurement")
    return bld.build(name="PIController")


def _make_plant_class():
    from jaxonomy.framework import LeafSystem, parameters

    class DahlquistPlant(LeafSystem):
        """Single-state continuous plant: ``x_dot = -k*x + u``."""

        @parameters(dynamic=["k"])
        def __init__(self, k=1.0, name="dahlquist"):
            super().__init__(name=name)

            def _ode(time, state, *inputs, **params):
                (u,) = inputs
                return -params["k"] * state.continuous_state + u

            self.declare_continuous_state(
                shape=(), default_value=0.0, ode=_ode,
            )
            self.declare_input_port(name="u")
            self.declare_continuous_state_output(name="y")

    return DahlquistPlant


def _simulate_inprocess():
    """Architecture A — fully in-process controller + plant."""
    import jaxonomy

    DahlquistPlant = _make_plant_class()
    bld = jaxonomy.DiagramBuilder()
    ctl = bld.add(_build_controller_diagram())
    plant = bld.add(DahlquistPlant(k=K_PLANT))
    bld.connect(ctl.output_ports[0], plant.input_ports[0])
    bld.connect(plant.output_ports[0], ctl.input_ports[0])
    bld.export_output(plant.output_ports[0], name="x")
    bld.export_output(ctl.output_ports[0], name="u")
    loop = bld.build()
    res = jaxonomy.simulate(
        loop, loop.create_context(), (0.0, T_END),
        options=jaxonomy.SimulatorOptions(max_major_step_length=DT),
        recorded_signals={
            "x": loop.output_ports[0],
            "u": loop.output_ports[1],
        },
    )
    return np.asarray(res.time), np.asarray(res.outputs["x"]), np.asarray(res.outputs["u"])


def _build_fmu_only(tmpdir: Path):
    """Build the controller FMU and return its path + build metrics."""
    from jaxonomy.library.fmu_export import build_fmu

    slave_script = tmpdir / "pi_slave.py"
    slave_script.write_text(_SLAVE_SOURCE)
    fmu_path = tmpdir / "PIController.fmu"

    t0 = time.perf_counter()
    build_fmu(str(slave_script), str(fmu_path))
    build_seconds = time.perf_counter() - t0
    return fmu_path, build_seconds, fmu_path.stat().st_size


def _run_fmu_inside_jaxonomy(fmu_path: Path):
    """Architecture B — controller as a Jaxonomy ``ModelicaFMU`` block;
    plant remains the in-process ``DahlquistPlant`` LeafSystem."""
    import jaxonomy
    from jaxonomy.library import Constant, ModelicaFMU

    DahlquistPlant = _make_plant_class()
    bld = jaxonomy.DiagramBuilder()
    fmu_ctl = bld.add(ModelicaFMU(
        file_name=str(fmu_path), dt=DT,
        input_names=["measurement", "setpoint"],
        output_names=["u"], name="pi_fmu",
    ))
    plant = bld.add(DahlquistPlant(k=K_PLANT, name="plant"))
    sp_const = bld.add(Constant(SETPOINT, name="setpoint"))
    bld.connect(plant.output_ports[0], fmu_ctl.input_ports[0])
    bld.connect(sp_const.output_ports[0], fmu_ctl.input_ports[1])
    bld.connect(fmu_ctl.output_ports[0], plant.input_ports[0])
    bld.export_output(plant.output_ports[0], name="x")
    bld.export_output(fmu_ctl.output_ports[0], name="u")
    loop = bld.build()
    res = jaxonomy.simulate(
        loop, loop.create_context(), (0.0, T_END),
        options=jaxonomy.SimulatorOptions(max_major_step_length=DT),
        recorded_signals={
            "x": loop.output_ports[0],
            "u": loop.output_ports[1],
        },
    )
    return (
        np.asarray(res.time), np.asarray(res.outputs["x"]),
        np.asarray(res.outputs["u"]),
    )


def _manual_fmu_loop_in_subprocess(fmu_path: Path, tmpdir: Path):
    """Run the manual orchestration in a fresh Python process.

    pythonfmu's embedded-Python FMU dylib carries process-global state
    (a singleton ``Py_Initialize``), so the same FMU cannot be
    instantiated twice in one process even after ``freeInstance``.
    Architecture B already instantiated the FMU once via
    ``ModelicaFMU``; running the manual loop here in a fresh
    ``subprocess`` sidesteps that limitation cleanly.
    """
    runner = tmpdir / "manual_loop.py"
    runner.write_text(textwrap.dedent(f"""
        import json, sys, time
        import numpy as np
        import fmpy

        FMU_PATH = {str(fmu_path)!r}
        DT = {DT!r}; T_END = {T_END!r}; SETPOINT = {SETPOINT!r}
        K_PLANT = {K_PLANT!r}; N_STEPS = {N_STEPS!r}

        md = fmpy.read_model_description(FMU_PATH)
        unzipdir = fmpy.extract(FMU_PATH)
        fmu = fmpy.fmi2.FMU2Slave(
            guid=md.guid, unzipDirectory=unzipdir,
            modelIdentifier=md.coSimulation.modelIdentifier,
            instanceName='ctl_manual',
        )
        refs = {{v.name: v.valueReference for v in md.modelVariables}}
        fmu.instantiate()
        fmu.setupExperiment(startTime=0.0)
        fmu.enterInitializationMode(); fmu.exitInitializationMode()
        ts = np.zeros(N_STEPS + 1); xs = np.zeros(N_STEPS + 1); us = np.zeros(N_STEPS + 1)
        x = 0.0; t = 0.0
        t0 = time.perf_counter()
        for k in range(N_STEPS):
            fmu.setReal([refs['setpoint']], [SETPOINT])
            fmu.setReal([refs['measurement']], [x])
            fmu.doStep(currentCommunicationPoint=t, communicationStepSize=DT)
            (u,) = fmu.getReal([refs['u']])
            x = x + DT * (-K_PLANT * x + u)
            t += DT
            ts[k + 1] = t; xs[k + 1] = x; us[k + 1] = u
        wall = time.perf_counter() - t0
        fmu.terminate(); fmu.freeInstance()
        out = {{'t': ts.tolist(), 'x': xs.tolist(), 'u': us.tolist(), 'wall': wall}}
        print('__JSON_BEGIN__')
        print(json.dumps(out))
        print('__JSON_END__')
    """).strip())

    res = subprocess.run(
        [sys.executable, str(runner)],
        capture_output=True, text=True, check=True,
    )
    stdout = res.stdout
    start = stdout.index("__JSON_BEGIN__") + len("__JSON_BEGIN__")
    end = stdout.index("__JSON_END__")
    import json
    payload = json.loads(stdout[start:end].strip())
    return (
        np.asarray(payload["t"]),
        np.asarray(payload["x"]),
        np.asarray(payload["u"]),
        float(payload["wall"]),
    )


def _manual_fmu_loop(fmu_path: Path):
    """Compatibility shim — fall back to in-process if subprocess fails."""
    import fmpy

    md = fmpy.read_model_description(str(fmu_path))
    unzipdir = fmpy.extract(str(fmu_path))
    fmu = fmpy.fmi2.FMU2Slave(
        guid=md.guid, unzipDirectory=unzipdir,
        modelIdentifier=md.coSimulation.modelIdentifier,
        instanceName="ctl_manual",
    )
    refs = {v.name: v.valueReference for v in md.modelVariables}
    fmu.instantiate()
    fmu.setupExperiment(startTime=0.0)
    fmu.enterInitializationMode()
    fmu.exitInitializationMode()

    ts = np.zeros(N_STEPS + 1)
    xs = np.zeros(N_STEPS + 1)
    us = np.zeros(N_STEPS + 1)
    x = 0.0
    t = 0.0

    t0 = time.perf_counter()
    for k in range(N_STEPS):
        fmu.setReal([refs["setpoint"]], [SETPOINT])
        fmu.setReal([refs["measurement"]], [x])
        fmu.doStep(currentCommunicationPoint=t, communicationStepSize=DT)
        (u,) = fmu.getReal([refs["u"]])
        x = x + DT * (-K_PLANT * x + u)
        t += DT
        ts[k + 1] = t
        xs[k + 1] = x
        us[k + 1] = u
    wall_fmu_manual = time.perf_counter() - t0
    fmu.terminate()
    fmu.freeInstance()
    return ts, xs, us, wall_fmu_manual


def _manual_inprocess_loop():
    """Reference pure-Python orchestration matching the slave's protocol."""
    ts = np.zeros(N_STEPS + 1)
    xs = np.zeros(N_STEPS + 1)
    us = np.zeros(N_STEPS + 1)
    x = 0.0
    ie = 0.0
    t = 0.0
    t0 = time.perf_counter()
    for k in range(N_STEPS):
        err = SETPOINT - x
        ie = ie + DT * err  # Forward-Euler integrator update.
        u = KP * err + KI * ie  # Match the FMU's read-output-after-doStep order.
        x = x + DT * (-K_PLANT * x + u)
        t += DT
        ts[k + 1] = t
        xs[k + 1] = x
        us[k + 1] = u
    wall_ref = time.perf_counter() - t0
    return ts, xs, us, wall_ref


_SLAVE_SOURCE = '''
"""PI controller slave for FMI 2.0 co-simulation export."""

import jaxonomy
from jaxonomy.library import Constant, Adder, Gain
from jaxonomy.library.dynamics import IntegratorDiscrete
from jaxonomy.library.fmu_slave import JaxonomyDiagramSlave

DT = 0.01
KP = 1.5
KI = 0.4


def _build():
    bld = jaxonomy.DiagramBuilder()
    sp = bld.add(Constant(0.0, name="setpoint"))
    meas = bld.add(Constant(0.0, name="measurement"))
    err = bld.add(Adder(2, operators="+-", name="err"))
    kp = bld.add(Gain(KP, name="kp_gain"))
    ki = bld.add(Gain(KI, name="ki_gain"))
    integ = bld.add(IntegratorDiscrete(dt=DT, initial_state=0.0, name="ierr"))
    add_u = bld.add(Adder(2, operators="++", name="add_u"))
    bld.connect(sp.output_ports[0], err.input_ports[0])
    bld.connect(meas.output_ports[0], err.input_ports[1])
    bld.connect(err.output_ports[0], kp.input_ports[0])
    bld.connect(err.output_ports[0], integ.input_ports[0])
    bld.connect(integ.output_ports[0], ki.input_ports[0])
    bld.connect(kp.output_ports[0], add_u.input_ports[0])
    bld.connect(ki.output_ports[0], add_u.input_ports[1])
    bld.export_output(add_u.output_ports[0], name="u")
    return bld.build()


class PIController(JaxonomyDiagramSlave):
    DIAGRAM_FACTORY = staticmethod(_build)
    DT = DT
'''


def main(write_placeholder: bool = False):
    if write_placeholder:
        # Structurally-plausible numbers for environments without
        # pythonfmu / fmpy. Real runs will overwrite this NPZ.
        t = np.linspace(0.0, T_END, N_STEPS + 1)
        np.savez_compressed(
            NPZ_OUT,
            placeholder_flag=np.array(True),
            t_A=t, x_A=1.0 - np.exp(-1.5 * t), u_A=np.zeros_like(t),
            t_B=t, x_B=1.0 - np.exp(-1.5 * t), u_B=np.zeros_like(t),
            t_ref=t, x_ref=1.0 - np.exp(-1.5 * t), u_ref=np.zeros_like(t),
            t_fmu_manual=t, x_fmu_manual=1.0 - np.exp(-1.5 * t),
            u_fmu_manual=np.zeros_like(t),
            build_seconds=np.array(0.1),
            fmu_size_bytes=np.array(900000),
            wall_inprocess_seconds=np.array(0.5),
            wall_fmu_seconds=np.array(1.0),
            wall_manual_ref=np.array(0.001),
            wall_manual_fmu=np.array(0.05),
        )
        print(f"placeholder NPZ written to {NPZ_OUT}")
        return

    with tempfile.TemporaryDirectory() as td:
        tmpdir = Path(td)
        print("[1/4] Architecture A (all in-process) running ...")
        t0 = time.perf_counter()
        t_A, x_A, u_A = _simulate_inprocess()
        wall_inprocess = time.perf_counter() - t0
        print(f"      done in {wall_inprocess:.2f} s.")

        print("[2/4] Building binary FMU ...")
        fmu_path, build_seconds, fmu_size = _build_fmu_only(tmpdir)
        print(f"      FMU built in {build_seconds:.3f} s "
              f"({fmu_size / 1024:.0f} KiB on disk).")

        print("      Architecture B (jaxonomy ModelicaFMU + LeafSystem plant) ...")
        t0 = time.perf_counter()
        t_B, x_B, u_B = _run_fmu_inside_jaxonomy(fmu_path)
        wall_fmu = time.perf_counter() - t0
        print(f"      closed loop in {wall_fmu:.2f} s.")

        print("[3/4] Manual fmpy orchestration in subprocess "
              "(controller FMU + Python-loop plant) ...")
        t_fmu_manual, x_fmu_manual, u_fmu_manual, wall_manual_fmu = (
            _manual_fmu_loop_in_subprocess(fmu_path, tmpdir)
        )

        print("[4/4] Manual pure-Python reference ...")
        t_ref, x_ref, u_ref, wall_manual_ref = _manual_inprocess_loop()

    np.savez_compressed(
        NPZ_OUT,
        placeholder_flag=np.array(False),
        t_A=t_A, x_A=x_A, u_A=u_A,
        t_B=t_B, x_B=x_B, u_B=u_B,
        t_ref=t_ref, x_ref=x_ref, u_ref=u_ref,
        t_fmu_manual=t_fmu_manual, x_fmu_manual=x_fmu_manual,
        u_fmu_manual=u_fmu_manual,
        build_seconds=np.array(build_seconds),
        fmu_size_bytes=np.array(fmu_size),
        wall_inprocess_seconds=np.array(wall_inprocess),
        wall_fmu_seconds=np.array(wall_fmu),
        wall_manual_ref=np.array(wall_manual_ref),
        wall_manual_fmu=np.array(wall_manual_fmu),
    )
    print(f"\nNPZ written to {NPZ_OUT}")

    # Headline numbers.
    grid = np.arange(N_STEPS + 1) * DT
    xA_g = np.interp(grid, t_A, x_A)
    xB_g = np.interp(grid, t_B, x_B)
    err_x = np.max(np.abs(xA_g - xB_g))
    err_x_late = np.max(np.abs(xA_g[10:] - xB_g[10:]))
    err_x_manual = float(np.max(np.abs(x_fmu_manual - x_ref)))
    err_u_manual = float(np.max(np.abs(u_fmu_manual - u_ref)))
    print("\nHEADLINE NUMBERS")
    print(f"  In-jaxonomy diagram comparison (A vs B), max |x| diff "
          f"over full {T_END:g} s:               {err_x:.3e}")
    print(f"    (after first 10 steps, post-transient):                "
          f"                            {err_x_late:.3e}")
    print(f"  Manual orchestration (FMU controller + Python plant) vs "
          f"reference, max |x| diff:    {err_x_manual:.3e}")
    print(f"    same comparison on control signal u:                   "
          f"                            {err_u_manual:.3e}")


if __name__ == "__main__":
    write_placeholder = "--write-placeholder" in sys.argv
    main(write_placeholder=write_placeholder)
