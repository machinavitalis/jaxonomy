# SPDX-License-Identifier: MIT
"""
T-002 Phase B — byte-exact reproducibility across the model-category matrix.

One scenario per category, each asserted to be bit-exact across two
back-to-back runs on the CI runner:

  - CT under each selectable solver (rk4, dopri5, bdf)
  - event-driven mode switch
  - acausal DAE (BDF with mass matrix)
  - seeded ``RandomNumber`` block
  - seeded ``WhiteNoise`` block
  - ``simulate_batch`` kernel path
  - ``simulate_batch`` vmap path

All of these run on every PR (no slow marker) and are intentionally
short (<1 s of simulated time apiece) so the total suite stays under a
few seconds.
"""

from __future__ import annotations

from enum import IntEnum

import jax
import jax.numpy as jnp
import pytest

import jaxonomy
from jaxonomy.library import Constant, Gain, Integrator, RandomNumber, WhiteNoise

from ._framework import assert_bitwise_reproducible
from jaxonomy.testing.markers import skip_if_not_jax

skip_if_not_jax()


# ── CT under every solver ───────────────────────────────────────────────────


class _Decay(jaxonomy.LeafSystem):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.declare_dynamic_parameter("a", 1.5)
        self.declare_continuous_state(default_value=jnp.array(2.0), ode=self._ode)

    def _ode(self, time, state, **params):
        return -params["a"] * state.continuous_state


@pytest.mark.parametrize("solver", ["rk4", "dopri5", "bdf"])
def test_ct_every_solver_bit_exact(solver):
    def run():
        sys = _Decay()
        ctx = sys.create_context()
        kwargs = dict(math_backend="jax", ode_solver_method=solver)
        if solver == "rk4":
            kwargs["max_minor_step_size"] = 0.01
        opts = jaxonomy.SimulatorOptions(**kwargs)
        return jaxonomy.simulate(sys, ctx, (0.0, 0.5), options=opts).context.continuous_state

    assert_bitwise_reproducible(run, label=f"CT/Decay/{solver}")


# ── event-driven mode switch ────────────────────────────────────────────────


class _Modes(IntEnum):
    A = 0
    B = 1


class _ModeSwitch(jaxonomy.LeafSystem):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.declare_dynamic_parameter("a", 1.0)
        self.declare_default_mode(_Modes.A)
        self.declare_continuous_state(shape=(), ode=self._ode)
        self.declare_continuous_state_output()
        self.declare_zero_crossing(
            guard=lambda t, s, **p: s.continuous_state,
            direction="crosses_zero",
            name="zc",
            start_mode=_Modes.A,
            end_mode=_Modes.B,
        )

    def _ode(self, time, state, **params):
        a = params["a"]
        return jax.lax.switch(state.mode, [lambda: -a, lambda: a])


def test_event_mode_switch_bit_exact():
    """Zero-crossing localisation uses bisection; bit-exactness proves the
    guard resolution is deterministic (no set/dict iteration leaking in)."""

    def run():
        sys = _ModeSwitch()
        ctx = sys.create_context().with_continuous_state(jnp.array(1.0))
        opts = jaxonomy.SimulatorOptions(
            math_backend="jax", max_major_steps=50,
        )
        sim = jaxonomy.Simulator(sys, options=opts)
        return sim.advance_to(1.4, ctx).context.continuous_state

    assert_bitwise_reproducible(run, label="Event/ModeSwitch")


# ── acausal DAE (BDF + mass matrix) ─────────────────────────────────────────


def test_acausal_dae_bit_exact():
    """RC circuit via AcausalCompiler → BDF on a mass-matrix system."""
    from jaxonomy.acausal import AcausalCompiler, AcausalDiagram, EqnEnv
    from jaxonomy.acausal import electrical as elec

    def run():
        ev = EqnEnv()
        ad = AcausalDiagram()
        vs = elec.VoltageSource(ev, name="vs", v=1.0)
        r1 = elec.Resistor(ev, name="r1", R=1.0)
        c1 = elec.Capacitor(ev, name="c1", C=1.0,
                            initial_voltage=0.5, initial_voltage_fixed=True)
        gnd = elec.Ground(ev, name="gnd")
        ad.connect(vs, "p", r1, "n")
        ad.connect(r1, "p", c1, "p")
        ad.connect(c1, "n", vs, "n")
        ad.connect(vs, "n", gnd, "p")
        rc_sys = AcausalCompiler(ev, ad)()
        bld = jaxonomy.DiagramBuilder()
        s = bld.add(rc_sys)
        diagram = bld.build()
        ctx = diagram.create_context()
        opts = jaxonomy.SimulatorOptions(math_backend="jax", ode_solver_method="bdf")
        return jaxonomy.simulate(diagram, ctx, (0.0, 1.0), options=opts).context[
            s.system_id
        ].continuous_state

    assert_bitwise_reproducible(run, label="Acausal/RC/BDF")


# ── stochastic blocks with explicit seeds ───────────────────────────────────


def test_random_number_seeded_bit_exact():
    """RandomNumber(seed=42) must be reproducible — same seed, same samples."""

    def run():
        bld = jaxonomy.DiagramBuilder()
        rn = bld.add(RandomNumber(dt=0.1, shape=(3,), seed=42, name="rn"))
        integ = bld.add(Integrator(jnp.zeros(3), name="i"))
        bld.connect(rn.output_ports[0], integ.input_ports[0])
        diagram = bld.build()
        ctx = diagram.create_context()
        opts = jaxonomy.SimulatorOptions(math_backend="jax", max_major_steps=50)
        res = jaxonomy.simulate(diagram, ctx, (0.0, 0.5), options=opts)
        return res.context[integ.system_id].continuous_state

    assert_bitwise_reproducible(run, label="Stochastic/RandomNumber(seed=42)")


def test_white_noise_seeded_bit_exact():
    """WhiteNoise(seed=7) must be reproducible."""

    def run():
        bld = jaxonomy.DiagramBuilder()
        wn = bld.add(WhiteNoise(correlation_time=0.01, shape=(1,), seed=7, name="wn"))
        integ = bld.add(Integrator(jnp.zeros(1), name="i"))
        bld.connect(wn.output_ports[0], integ.input_ports[0])
        diagram = bld.build()
        ctx = diagram.create_context()
        opts = jaxonomy.SimulatorOptions(math_backend="jax", max_major_steps=100)
        res = jaxonomy.simulate(diagram, ctx, (0.0, 0.5), options=opts)
        return res.context[integ.system_id].continuous_state

    assert_bitwise_reproducible(run, label="Stochastic/WhiteNoise(seed=7)")


# ── simulate_batch — both execution paths ───────────────────────────────────


def _build_batched_diagram():
    bld = jaxonomy.DiagramBuilder()
    c = bld.add(Constant(jnp.array(1.0), name="c"))
    integ = bld.add(Integrator(jnp.array(0.0), name="i"))
    gain = bld.add(Gain(-0.5, name="g"))
    bld.connect(c.output_ports[0], gain.input_ports[0])
    bld.connect(gain.output_ports[0], integ.input_ports[0])
    diagram = bld.build()
    return diagram, c, integ, gain


def test_simulate_batch_kernel_bit_exact():
    """Kernel path: single JIT kernel with per-element parameter injection."""

    def run():
        diagram, _, integ, gain = _build_batched_diagram()
        opts = jaxonomy.SimulatorOptions(
            math_backend="jax", enable_autodiff=False, max_major_steps=20,
        )
        recorded = {"i": integ.output_ports[0]}
        param_batches = {"g.gain": jnp.linspace(-1.0, -0.1, 4)}
        return jaxonomy.simulate_batch(
            diagram, (0.0, 0.3), param_batches,
            options=opts, recorded_signals=recorded,
        ).outputs["i"]

    assert_bitwise_reproducible(run, label="simulate_batch/kernel")


def test_simulate_batch_vmap_bit_exact():
    """vmap path: all N simulations in one XLA call."""

    def run():
        diagram, _, integ, gain = _build_batched_diagram()
        opts = jaxonomy.SimulatorOptions(
            math_backend="jax", enable_autodiff=False, max_major_steps=20,
        )
        recorded = {"i": integ.output_ports[0]}
        param_batches = {"g.gain": jnp.linspace(-1.0, -0.1, 4)}
        return jaxonomy.simulate_batch(
            diagram, (0.0, 0.3), param_batches,
            options=opts, recorded_signals=recorded, use_vmap=True,
        ).outputs["i"]

    assert_bitwise_reproducible(run, label="simulate_batch/vmap")
