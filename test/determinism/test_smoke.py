# SPDX-License-Identifier: MIT
"""
T-002 Phase A smoke tests.

Minimum viable reproducibility coverage to validate the framework:
  - one pure CT ODE,
  - one pure DT periodic update,
  - one hybrid CT+DT diagram.

Broad coverage (events, DAE, PRNG, simulate_batch, state machines) lives
in ``test_determinism_coverage.py``. Negative controls in
``test_negative_controls.py``. Cross-hardware scaffolding in
``test_cross_hardware.py``.
"""

from __future__ import annotations

import jax.numpy as jnp
import pytest

import jaxonomy
from jaxonomy.library import Adder, Constant, Gain, Integrator

from ._framework import assert_bitwise_reproducible
from jaxonomy.testing.markers import skip_if_not_jax

skip_if_not_jax()


# ── CT ──────────────────────────────────────────────────────────────────────


def test_ct_ode_bit_exact():
    """Pure CT ODE under Dopri5 (default). Two back-to-back calls must match bytes."""

    class Decay(jaxonomy.LeafSystem):
        def __init__(self):
            super().__init__()
            self.declare_dynamic_parameter("a", 1.5)
            self.declare_continuous_state(default_value=jnp.array(2.0), ode=self._ode)

        def _ode(self, time, state, **params):
            return -params["a"] * state.continuous_state

    def run():
        sys = Decay()
        ctx = sys.create_context()
        opts = jaxonomy.SimulatorOptions(math_backend="jax")
        return jaxonomy.simulate(sys, ctx, (0.0, 2.0), options=opts).context.continuous_state

    assert_bitwise_reproducible(run, label="CT/Decay/dopri5")


# ── DT ──────────────────────────────────────────────────────────────────────


def test_dt_periodic_update_bit_exact():
    """Pure discrete: x[n+1] = k·x[n] at 10 Hz for 1 s → 10 updates."""

    class DTScale(jaxonomy.LeafSystem):
        def __init__(self):
            super().__init__()
            self.declare_dynamic_parameter("k", 0.9)
            self.declare_discrete_state(default_value=jnp.array(1.0))
            self.declare_periodic_update(self._upd, period=0.1, offset=0.0)

        def _upd(self, time, state, **params):
            return state.discrete_state * params["k"]

    def run():
        sys = DTScale()
        ctx = sys.create_context()
        opts = jaxonomy.SimulatorOptions(
            math_backend="jax", max_major_steps=30,
        )
        return jaxonomy.simulate(sys, ctx, (0.0, 1.0), options=opts).context.discrete_state

    assert_bitwise_reproducible(run, label="DT/Scale/10Hz")


# ── Hybrid CT+DT ────────────────────────────────────────────────────────────


def test_hybrid_ct_dt_bit_exact():
    """CT integrator driven by DT-updated setpoint; closed-loop decay-to-setpoint."""

    class DTSetpoint(jaxonomy.LeafSystem):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.declare_discrete_state(default_value=jnp.array(1.0))
            self.declare_output_port(
                lambda t, s, **p: s.discrete_state, period=0.1, offset=0.0
            )
            self.declare_periodic_update(
                lambda t, s, **p: s.discrete_state * 0.95, period=0.1, offset=0.0
            )

    def run():
        bld = jaxonomy.DiagramBuilder()
        sp = bld.add(DTSetpoint(name="sp"))
        integ = bld.add(Integrator(jnp.array(0.0), name="int"))
        err = bld.add(Adder(2, operators="+-", name="err"))
        gain = bld.add(Gain(2.0, name="gain"))
        bld.connect(sp.output_ports[0], err.input_ports[0])
        bld.connect(integ.output_ports[0], err.input_ports[1])
        bld.connect(err.output_ports[0], gain.input_ports[0])
        bld.connect(gain.output_ports[0], integ.input_ports[0])
        diagram = bld.build()
        ctx = diagram.create_context()
        opts = jaxonomy.SimulatorOptions(math_backend="jax", max_major_steps=30)
        res = jaxonomy.simulate(diagram, ctx, (0.0, 1.0), options=opts)
        return res.context[integ.system_id].continuous_state

    assert_bitwise_reproducible(run, label="Hybrid/CT+DT")
