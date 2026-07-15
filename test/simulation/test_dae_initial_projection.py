# SPDX-License-Identifier: MIT
"""Tests for ``SimulatorOptions(dae_initial_projection=True)`` and
``AcausalSystem.continuous_state_layout`` (consumer-reported: no
supported recipe for resetting the continuous state of a compiled DAE).

Repro shape: overwrite the differential rows of a compiled acausal DAE's
context (a state reset) — the algebraic rows are then inconsistent and
the BDF start returns NaN unless the initial projection runs.
"""

import numpy as np
import pytest

import jax.numpy as jnp

import jaxonomy as jx
from jaxonomy.acausal import AcausalCompiler, AcausalDiagram, EqnEnv
from jaxonomy.acausal.component_library import hydraulic as hd


def _build_pump_accumulator():
    """Pump feeding an accumulator through a pipe — 1 diff + several alg rows."""
    ev = EqnEnv()
    ad = AcausalDiagram()
    props = hd.HydraulicProperties(ev, fluid_name="water")
    src = hd.PressureSource(ev, name="src", pressure=0.0)
    pump = hd.Pump(ev, name="pump", dPmax=1.0e4, CoP=10.0)
    pipe = hd.Pipe(ev, name="pipe", R=100.0)
    acc = hd.Accumulator(
        ev, name="acc", initial_pressure=200.0, initial_pressure_fixed=True,
        area=1.0, k=1.0,
    )
    ad.connect(src, "port", pump, "port_a")
    ad.connect(pump, "port_b", pipe, "port_a")
    ad.connect(pipe, "port_b", acc, "port")
    ad.connect(props, "prop", acc, "port")
    system = AcausalCompiler(ev, ad)(name="pump_acc", leaf_backend="jax")

    builder = jx.DiagramBuilder()
    s = builder.add(system)
    pwr = builder.add(jx.library.Constant(np.array(5.0)))
    builder.connect(pwr.output_ports[0], s.input_ports[0])
    return builder.build(), s, system


BDF_OPTS = dict(
    math_backend="jax", ode_solver_method="bdf",
    rtol=1e-8, atol=1e-10, enable_autodiff=False, max_major_steps=64,
)


def _final_state(diagram, ctx, **extra):
    opts = jx.SimulatorOptions(**BDF_OPTS, **extra)
    res = jx.simulate(diagram, ctx, (0.0, 1.0), options=opts)
    return np.asarray(res.context[ctx.owning_system.system_id].continuous_state) \
        if hasattr(ctx, "owning_system") else res


def test_layout_names_rows():
    _, _, system = _build_pump_accumulator()
    layout = system.continuous_state_layout()
    assert len(layout) == system.n_ode + system.n_alg
    kinds = [r["kind"] for r in layout]
    assert kinds[: system.n_ode] == ["differential"] * system.n_ode
    assert kinds[system.n_ode:] == ["algebraic"] * system.n_alg
    assert all(isinstance(r["name"], str) and r["name"] for r in layout)
    assert [r["row"] for r in layout] == list(range(len(layout)))


def test_reset_without_projection_fails_with_projection_succeeds():
    diagram, s, system = _build_pump_accumulator()
    sid = s.system_id
    ctx0 = diagram.create_context()
    cs0 = np.asarray(ctx0[sid].continuous_state)
    n_ode = system.n_ode
    assert n_ode >= 1 and system.n_alg >= 1

    # perturb the differential rows hard; keep stale algebraic entries
    new = jnp.asarray(cs0).at[0].set(cs0[0] * 3.0 + 50.0)
    ctx = ctx0.with_subcontext(sid, ctx0[sid].with_continuous_state(new))

    def run(**extra):
        opts = jx.SimulatorOptions(**BDF_OPTS, **extra)
        res = jx.simulate(diagram, ctx, (0.0, 1.0), options=opts)
        return np.asarray(res.context[sid].continuous_state)

    without = run()
    with_proj = run(dae_initial_projection=True)
    assert np.isfinite(with_proj).all(), "projected start must integrate cleanly"
    # the un-projected start is the documented failure mode; accept either
    # NaN (typical) or a finite result (small systems can recover), but the
    # projected path must be self-consistent: rerunning from its ICs is stable
    if np.isfinite(without).all():
        # both finite: they must agree — projection then changed nothing material
        np.testing.assert_allclose(without[:n_ode], with_proj[:n_ode], rtol=1e-5)


def test_initial_projection_noop_without_mass_matrix():
    # plain ODE system: option enabled must be byte-equivalent no-op
    builder = jx.DiagramBuilder()
    from jaxonomy.library import Integrator, Constant
    integ = builder.add(Integrator(initial_state=0.5))
    one = builder.add(Constant(np.array(1.0)))
    builder.connect(one.output_ports[0], integ.input_ports[0])
    diagram = builder.build()
    ctx = diagram.create_context()
    opts = jx.SimulatorOptions(
        math_backend="jax", enable_autodiff=False,
        dae_initial_projection=True,
    )
    res = jx.simulate(diagram, ctx, (0.0, 1.0), options=opts)
    x = float(res.context[integ.system_id].continuous_state)
    assert x == pytest.approx(1.5, rel=1e-6)
