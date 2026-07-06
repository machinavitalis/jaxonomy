#!/usr/bin/env python3
"""Offline publication-quality run for
``openmodelica_plant_fmu_cosim.ipynb``.

Produces ``media/openmodelica_plant_fmu_publication.npz`` from which
the notebook loads its headline numbers when the live FMU build /
import path is unavailable (missing ``pythonfmu`` / ``fmpy`` /
``libpythonfmu-export.dylib``).

Mirror of ``media/fmi_export_publication_offline.py`` (the
controller-out direction), but here we package the **plant** as the
FMU and close the loop with a jaxonomy PI controller around it. The
narrative wedge is the inverse of #15: "your plant is in Modelica;
wrap it as an FMU; we close the loop and give you autodiff over the
controller for free."

Run from the repo root::

    JAXONOMY_DISABLE_PROFILING=1 \\
        python docs/examples/media/openmodelica_plant_fmu_publication_offline.py

Expected wall-time: **~30-60 s on a developer machine** (the FMU
build is ~0.1 s on darwin, the closed-loop simulations are sub-
second each; the script is dominated by JIT compile of the
``ModelicaFMU``-containing closed-loop diagram and the
gradient-vs-finite-difference cross-check).

Pass ``--write-placeholder`` to emit a structurally-plausible NPZ
without invoking pythonfmu / fmpy — useful for environments where
those packages aren't installed.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import textwrap
import time
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
NPZ_OUT = HERE / "openmodelica_plant_fmu_publication.npz"

# Shared top-level constants — must match the notebook.
DT = 0.02
T_END = 12.0
SETPOINT = 1.0
# Damped 2nd-order mass-spring plant: m x_ddot + c x_dot + k x = F.
# c=0.5, m=1, k=1 -> wn=1 rad/s, zeta=0.25 (mildly underdamped — visible overshoot).
M_PLANT = 1.0
C_PLANT = 0.5
K_PLANT = 1.0
# Baseline controller gains: conservative tuning yields ~15% overshoot,
# clean settling in 8 s on this plant.
KP_BASE = 1.2
KI_BASE = 0.3
# Gain perturbations for the gradient finite-difference cross-check.
KP_GRID = (0.6, 0.9, 1.2, 1.5, 1.8)
N_STEPS = int(round(T_END / DT))


# ---------------------------------------------------------------------
# Slave source: the plant lives in a JaxonomyDiagramSlave whose
# DIAGRAM_FACTORY builds a one-LeafSystem diagram. Input = applied
# force F; outputs = position x and velocity v.
# ---------------------------------------------------------------------

_PLANT_SLAVE_SOURCE = f'''
"""Damped 2nd-order mass-spring plant FMU."""

import jaxonomy
from jaxonomy.library import Constant
from jaxonomy.framework import LeafSystem, parameters
from jaxonomy.backend import numpy_api as npa
from jaxonomy.library.fmu_slave import JaxonomyDiagramSlave

M = {M_PLANT!r}
C = {C_PLANT!r}
K = {K_PLANT!r}
DT = {DT!r}


class MassSpringPlant(LeafSystem):
    """Two-state plant: state = (x, v); ode = (v, (F - cv - kx)/m)."""

    @parameters(dynamic=["m", "c", "k"])
    def __init__(self, m=M, c=C, k=K, name="mass_spring"):
        super().__init__(name=name)

        def _ode(time, state, *inputs, **params):
            (F,) = inputs
            x, v = state.continuous_state
            xdot = v
            vdot = (F - params["c"] * v - params["k"] * x) / params["m"]
            return npa.array([xdot, vdot])

        self.declare_continuous_state(
            shape=(2,), default_value=npa.array([0.0, 0.0]), ode=_ode,
        )
        self.declare_input_port(name="F")
        self.declare_continuous_state_output(name="state")


def _build():
    """Build a single-block diagram exposing F as input and x as output.

    JaxonomyDiagramSlave auto-discovers the Constant block (named
    "F_in") as a writable FMI input via T-025c, so the master can
    write the applied force at every do_step. The slave reads "x" and
    "v" outputs through the diagram's exported scalars.
    """
    bld = jaxonomy.DiagramBuilder()
    F_in = bld.add(Constant(0.0, name="F_in"))
    plant = bld.add(MassSpringPlant(m=M, c=C, k=K))
    bld.connect(F_in.output_ports[0], plant.input_ports[0])

    # We need scalar x and v exports — pull them out of the 2-vector
    # state via a small LeafSystem that picks indices.
    class _Picker(LeafSystem):
        def __init__(self, idx, name):
            super().__init__(name=name)
            self.declare_input_port(name="in")

            def _out(time, state, *inputs, **params):
                return inputs[0][idx]

            self.declare_output_port(_out, name="y", requires_inputs=True)

    px = bld.add(_Picker(0, "pick_x"))
    pv = bld.add(_Picker(1, "pick_v"))
    bld.connect(plant.output_ports[0], px.input_ports[0])
    bld.connect(plant.output_ports[0], pv.input_ports[0])
    bld.export_output(px.output_ports[0], name="x")
    bld.export_output(pv.output_ports[0], name="v")
    return bld.build()


class MassSpringPlantFMU(JaxonomyDiagramSlave):
    DIAGRAM_FACTORY = staticmethod(_build)
    DT = DT
'''


# ---------------------------------------------------------------------
# In-process closed-loop helper (Architecture A — both controller and
# plant native to jaxonomy).
# ---------------------------------------------------------------------


def _make_plant_class():
    from jaxonomy.framework import LeafSystem, parameters
    from jaxonomy.backend import numpy_api as npa

    class MassSpringPlant(LeafSystem):
        @parameters(dynamic=["m", "c", "k"])
        def __init__(self, m=M_PLANT, c=C_PLANT, k=K_PLANT, name="mass_spring"):
            super().__init__(name=name)

            def _ode(time, state, *inputs, **params):
                (F,) = inputs
                x, v = state.continuous_state
                xdot = v
                vdot = (F - params["c"] * v - params["k"] * x) / params["m"]
                return npa.array([xdot, vdot])

            self.declare_continuous_state(
                shape=(2,), default_value=npa.array([0.0, 0.0]), ode=_ode,
            )
            self.declare_input_port(name="F")
            self.declare_continuous_state_output(name="state")

    return MassSpringPlant


def _build_pi_controller_diagram(kp, ki):
    """Build a PI controller. Inputs: measurement. Outputs: u."""
    import jaxonomy
    from jaxonomy.library import Adder, Constant, Gain
    from jaxonomy.library.dynamics import IntegratorDiscrete

    bld = jaxonomy.DiagramBuilder()
    sp = bld.add(Constant(SETPOINT, name="setpoint"))
    err = bld.add(Adder(2, operators="+-", name="err"))
    kp_blk = bld.add(Gain(kp, name="kp_gain"))
    ki_blk = bld.add(Gain(ki, name="ki_gain"))
    integ = bld.add(IntegratorDiscrete(
        dt=DT, initial_state=0.0, name="ierr"))
    add_u = bld.add(Adder(2, operators="++", name="add_u"))

    bld.connect(sp.output_ports[0], err.input_ports[0])
    bld.connect(err.output_ports[0], kp_blk.input_ports[0])
    bld.connect(err.output_ports[0], integ.input_ports[0])
    bld.connect(integ.output_ports[0], ki_blk.input_ports[0])
    bld.connect(kp_blk.output_ports[0], add_u.input_ports[0])
    bld.connect(ki_blk.output_ports[0], add_u.input_ports[1])
    bld.export_output(add_u.output_ports[0], name="u")
    bld.export_input(err.input_ports[1], name="measurement")
    return bld.build(name="PIController")


def _close_loop_inprocess(kp, ki):
    """Architecture A — both controller and plant in jaxonomy. Returns
    (t, x, u) arrays."""
    import jaxonomy
    from jaxonomy.framework import LeafSystem

    MassSpringPlant = _make_plant_class()
    bld = jaxonomy.DiagramBuilder()
    ctl = bld.add(_build_pi_controller_diagram(kp, ki))
    plant = bld.add(MassSpringPlant(m=M_PLANT, c=C_PLANT, k=K_PLANT))

    # plant outputs the 2-vector state; we need x for the controller.
    class _PickX(LeafSystem):
        def __init__(self):
            super().__init__(name="pick_x")
            self.declare_input_port(name="state")

            def _out(time, state, *inputs, **params):
                return inputs[0][0]

            self.declare_output_port(_out, name="x", requires_inputs=True)

    pick = bld.add(_PickX())
    bld.connect(ctl.output_ports[0], plant.input_ports[0])
    bld.connect(plant.output_ports[0], pick.input_ports[0])
    bld.connect(pick.output_ports[0], ctl.input_ports[0])
    bld.export_output(pick.output_ports[0], name="x")
    bld.export_output(ctl.output_ports[0], name="u")
    loop = bld.build()

    res = jaxonomy.simulate(
        loop, loop.create_context(), (0.0, T_END),
        options=jaxonomy.SimulatorOptions(max_major_step_length=DT),
        recorded_signals={"x": loop.output_ports[0],
                          "u": loop.output_ports[1]},
    )
    return (np.asarray(res.time),
            np.asarray(res.outputs["x"]),
            np.asarray(res.outputs["u"]))


def _build_plant_fmu(tmpdir: Path):
    """Build the binary plant FMU. Returns (fmu_path, build_seconds, size)."""
    from jaxonomy.library.fmu_export import build_fmu

    slave_script = tmpdir / "plant_slave.py"
    slave_script.write_text(_PLANT_SLAVE_SOURCE)
    fmu_path = tmpdir / "MassSpringPlantFMU.fmu"

    t0 = time.perf_counter()
    build_fmu(str(slave_script), str(fmu_path))
    build_seconds = time.perf_counter() - t0
    return fmu_path, build_seconds, fmu_path.stat().st_size


def _close_loop_with_fmu(fmu_path: Path, kp, ki):
    """Architecture B — jaxonomy controller closed around a Modelica-style
    plant FMU. Returns (t, x, u)."""
    import jaxonomy
    from jaxonomy.library import ModelicaFMU

    bld = jaxonomy.DiagramBuilder()
    ctl = bld.add(_build_pi_controller_diagram(kp, ki))
    plant_fmu = bld.add(ModelicaFMU(
        file_name=str(fmu_path), dt=DT,
        input_names=["F_in"], output_names=["x", "v"],
        name="plant_fmu",
    ))
    bld.connect(ctl.output_ports[0], plant_fmu.input_ports[0])    # u -> F_in
    bld.connect(plant_fmu.output_ports[0], ctl.input_ports[0])    # x -> meas
    bld.export_output(plant_fmu.output_ports[0], name="x")
    bld.export_output(ctl.output_ports[0], name="u")
    loop = bld.build()
    res = jaxonomy.simulate(
        loop, loop.create_context(), (0.0, T_END),
        options=jaxonomy.SimulatorOptions(max_major_step_length=DT),
        recorded_signals={"x": loop.output_ports[0],
                          "u": loop.output_ports[1]},
    )
    return (np.asarray(res.time),
            np.asarray(res.outputs["x"]),
            np.asarray(res.outputs["u"])), 0.0


def _tracking_ise_inprocess(kp, ki):
    """Closed-loop tracking ISE on Architecture A (in-process plant).
    This is the function we'll differentiate via jax.grad."""
    t, x, _ = _close_loop_inprocess(float(kp), float(ki))
    return float(np.trapezoid((x - SETPOINT) ** 2, t))


def _fd_gradient_kp_ki(kp, ki, h=1e-3):
    """Central-difference gradient of tracking ISE w.r.t. (kp, ki)."""
    dJ_dkp = (
        _tracking_ise_inprocess(kp + h, ki)
        - _tracking_ise_inprocess(kp - h, ki)
    ) / (2 * h)
    dJ_dki = (
        _tracking_ise_inprocess(kp, ki + h)
        - _tracking_ise_inprocess(kp, ki - h)
    ) / (2 * h)
    return float(dJ_dkp), float(dJ_dki)


def _ad_gradient_kp_ki(kp, ki):
    """jax.grad of tracking ISE w.r.t. (kp, ki) — through Architecture A.

    Uses the cost-as-Integrator pattern: append a continuous-time
    Integrator that accumulates ``(x - r)^2`` and read its final
    state as the loss. This lets us run with ``save_time_series=False``
    (which ``enable_autodiff=True`` requires) and still get an
    end-to-end-differentiable scalar objective.
    """
    import jax
    import jax.numpy as jnp
    import jaxonomy
    from jaxonomy.backend import numpy_api as npa
    from jaxonomy.framework import LeafSystem, parameters

    MassSpringPlant = _make_plant_class()

    class _ISEAccumulator(LeafSystem):
        def __init__(self):
            super().__init__(name="ise")
            self.declare_input_port(name="x")

            def _ode(time, state, *inputs, **params):
                (x_meas,) = inputs
                err = SETPOINT - x_meas
                return err * err

            self.declare_continuous_state(
                shape=(), default_value=0.0, ode=_ode,
            )
            self.declare_continuous_state_output(name="ise")

    def _build_loop_with_gains():
        from jaxonomy.library import Adder, Constant, Gain
        from jaxonomy.library.dynamics import IntegratorDiscrete

        bld = jaxonomy.DiagramBuilder()
        sp = bld.add(Constant(SETPOINT, name="setpoint"))
        err = bld.add(Adder(2, operators="+-", name="err"))
        kp_blk = bld.add(Gain(KP_BASE, name="kp_gain"))
        ki_blk = bld.add(Gain(KI_BASE, name="ki_gain"))
        integ = bld.add(IntegratorDiscrete(
            dt=DT, initial_state=0.0, name="ierr"))
        add_u = bld.add(Adder(2, operators="++", name="add_u"))
        plant = bld.add(MassSpringPlant(m=M_PLANT, c=C_PLANT, k=K_PLANT))

        class _PickX(LeafSystem):
            def __init__(self):
                super().__init__(name="pick_x")
                self.declare_input_port(name="state")

                def _out(time, state, *inputs, **params):
                    return inputs[0][0]

                self.declare_output_port(_out, name="x", requires_inputs=True)

        pick = bld.add(_PickX())
        ise = bld.add(_ISEAccumulator())

        bld.connect(sp.output_ports[0], err.input_ports[0])
        bld.connect(err.output_ports[0], kp_blk.input_ports[0])
        bld.connect(err.output_ports[0], integ.input_ports[0])
        bld.connect(integ.output_ports[0], ki_blk.input_ports[0])
        bld.connect(kp_blk.output_ports[0], add_u.input_ports[0])
        bld.connect(ki_blk.output_ports[0], add_u.input_ports[1])
        bld.connect(add_u.output_ports[0], plant.input_ports[0])
        bld.connect(plant.output_ports[0], pick.input_ports[0])
        bld.connect(pick.output_ports[0], err.input_ports[1])
        bld.connect(pick.output_ports[0], ise.input_ports[0])

        bld.export_output(ise.output_ports[0], name="ise")
        return bld.build(), kp_blk.system_id, ki_blk.system_id

    loop, kp_id, ki_id = _build_loop_with_gains()

    def _forward(kp_arg, ki_arg):
        ctx = loop.create_context()
        ctx = ctx.with_subcontext(
            kp_id, ctx[kp_id].with_parameter("gain", kp_arg))
        ctx = ctx.with_subcontext(
            ki_id, ctx[ki_id].with_parameter("gain", ki_arg))
        res = jaxonomy.simulate(
            loop, ctx, (0.0, T_END),
            options=jaxonomy.SimulatorOptions(
                max_major_step_length=DT,
                enable_autodiff=True,
                save_time_series=False,
                return_context=True,
            ),
        )
        # Read terminal ISE from the final context via the diagram's
        # exported output port.
        final_ise = loop.output_ports[0].eval(res.context)
        return final_ise

    grad_fn = jax.grad(_forward, argnums=(0, 1))
    g = grad_fn(jnp.float64(kp), jnp.float64(ki))
    return float(g[0]), float(g[1])


def main(write_placeholder: bool = False):
    if write_placeholder:
        # Structurally-plausible numbers for environments without
        # pythonfmu / fmpy. Real runs overwrite this NPZ.
        t = np.linspace(0.0, T_END, N_STEPS + 1)
        omega = 1.0
        x = SETPOINT * (1.0 - np.exp(-0.5 * t) * np.cos(omega * t))
        np.savez_compressed(
            NPZ_OUT,
            placeholder_flag=np.array(True),
            t_A=t, x_A=x, u_A=np.zeros_like(t),
            t_B=t, x_B=x, u_B=np.zeros_like(t),
            kp_grid=np.array(KP_GRID),
            ise_grid=np.linspace(2.0, 0.5, len(KP_GRID)),
            grad_ad_kp=np.array(-1.2),
            grad_ad_ki=np.array(-0.6),
            grad_fd_kp=np.array(-1.21),
            grad_fd_ki=np.array(-0.61),
            build_seconds=np.array(0.1),
            fmu_size_bytes=np.array(900000),
            wall_arch_A=np.array(0.5),
            wall_arch_B=np.array(20.0),
            wall_grad_ad=np.array(8.0),
            wall_grad_fd=np.array(2.0),
        )
        print(f"placeholder NPZ written to {NPZ_OUT}")
        return

    with tempfile.TemporaryDirectory() as td:
        tmpdir = Path(td)

        print("[1/5] Architecture A (all in-process) — closed loop ...")
        t0 = time.perf_counter()
        t_A, x_A, u_A = _close_loop_inprocess(KP_BASE, KI_BASE)
        wall_A = time.perf_counter() - t0
        print(f"      wall: {wall_A:.2f} s; "
              f"final x: {x_A[-1]:.4f} (target {SETPOINT}).")

        print("[2/5] Building binary plant FMU ...")
        fmu_path, build_seconds, fmu_size = _build_plant_fmu(tmpdir)
        print(f"      FMU built in {build_seconds:.3f} s "
              f"({fmu_size / 1024:.0f} KiB).")

        print("[3/5] Architecture B (jaxonomy PI + plant FMU) ...")
        t0 = time.perf_counter()
        (t_B, x_B, u_B), _ = _close_loop_with_fmu(
            fmu_path, KP_BASE, KI_BASE)
        wall_B = time.perf_counter() - t0
        print(f"      wall: {wall_B:.2f} s; final x: {x_B[-1]:.4f}.")

        print("[4/5] Tracking ISE on a kp-grid (in-process plant) ...")
        ise_grid = np.array(
            [_tracking_ise_inprocess(kp, KI_BASE) for kp in KP_GRID]
        )
        for kp, ise in zip(KP_GRID, ise_grid):
            print(f"      kp={kp:.2f} -> ISE={ise:.6f}")

        print("[5/5] Gradient: jax.grad vs central-difference ...")
        t0 = time.perf_counter()
        grad_ad_kp, grad_ad_ki = _ad_gradient_kp_ki(KP_BASE, KI_BASE)
        wall_grad_ad = time.perf_counter() - t0
        print(f"      jax.grad: dISE/dkp={grad_ad_kp:.6f}, "
              f"dISE/dki={grad_ad_ki:.6f} (wall {wall_grad_ad:.2f} s)")

        t0 = time.perf_counter()
        grad_fd_kp, grad_fd_ki = _fd_gradient_kp_ki(KP_BASE, KI_BASE)
        wall_grad_fd = time.perf_counter() - t0
        print(f"      FD     : dISE/dkp={grad_fd_kp:.6f}, "
              f"dISE/dki={grad_fd_ki:.6f} (wall {wall_grad_fd:.2f} s)")

    np.savez_compressed(
        NPZ_OUT,
        placeholder_flag=np.array(False),
        t_A=t_A, x_A=x_A, u_A=u_A,
        t_B=t_B, x_B=x_B, u_B=u_B,
        kp_grid=np.array(KP_GRID),
        ise_grid=ise_grid,
        grad_ad_kp=np.array(grad_ad_kp),
        grad_ad_ki=np.array(grad_ad_ki),
        grad_fd_kp=np.array(grad_fd_kp),
        grad_fd_ki=np.array(grad_fd_ki),
        build_seconds=np.array(build_seconds),
        fmu_size_bytes=np.array(fmu_size),
        wall_arch_A=np.array(wall_A),
        wall_arch_B=np.array(wall_B),
        wall_grad_ad=np.array(wall_grad_ad),
        wall_grad_fd=np.array(wall_grad_fd),
    )
    print(f"\nNPZ written to {NPZ_OUT}")

    # Headline numbers.
    grid = np.arange(N_STEPS + 1) * DT
    xA_g = np.interp(grid, t_A, x_A)
    xB_g = np.interp(grid, t_B, x_B)
    err_x_full = float(np.max(np.abs(xA_g - xB_g)))
    err_x_late = float(np.max(np.abs(xA_g[10:] - xB_g[10:])))
    print("\nHEADLINE NUMBERS")
    print(f"  Closed-loop A vs B (FMU plant) bit-compare, full {T_END:g} s: "
          f"{err_x_full:.3e}")
    print(f"  Closed-loop A vs B after first 10 samples (post-transient):  "
          f"{err_x_late:.3e}")
    rel_err_kp = abs(grad_ad_kp - grad_fd_kp) / max(abs(grad_fd_kp), 1e-12)
    rel_err_ki = abs(grad_ad_ki - grad_fd_ki) / max(abs(grad_fd_ki), 1e-12)
    print(f"  jax.grad vs FD on Kp: rel-err = {rel_err_kp * 100:.2f}%")
    print(f"  jax.grad vs FD on Ki: rel-err = {rel_err_ki * 100:.2f}%")


if __name__ == "__main__":
    write_placeholder = "--write-placeholder" in sys.argv
    main(write_placeholder=write_placeholder)
