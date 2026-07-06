# SPDX-License-Identifier: MIT
"""
T-002 Phase C — negative controls.

For every axis the bit-exact tests claim to pin down (seed, initial
condition, parameter, tolerance, end time), assert that *changing*
that axis *does* change the output. Guards against a trivially-passing
bit-exact test — e.g. a simulation that silently always returns its
initial condition would satisfy bit-exactness but exercise no actual
dynamics.
"""

from __future__ import annotations

import jax.numpy as jnp

import jaxonomy
from jaxonomy.library import Integrator, RandomNumber, WhiteNoise

from ._framework import assert_not_bitwise_equal
from jaxonomy.testing.markers import skip_if_not_jax

skip_if_not_jax()


class _Decay(jaxonomy.LeafSystem):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.declare_dynamic_parameter("a", 1.5)
        self.declare_continuous_state(default_value=jnp.array(2.0), ode=self._ode)

    def _ode(self, time, state, **params):
        return -params["a"] * state.continuous_state


def _decay_run(a, x0, tf, rtol):
    sys = _Decay()
    ctx = sys.create_context().with_parameter("a", a).with_continuous_state(x0)
    opts = jaxonomy.SimulatorOptions(math_backend="jax", rtol=rtol)
    return jaxonomy.simulate(sys, ctx, (0.0, tf), options=opts).context.continuous_state


def test_negative_initial_condition_changes_output():
    """x0=2.0 vs x0=1.0 must produce different outputs."""
    assert_not_bitwise_equal(
        lambda: _decay_run(1.5, jnp.array(2.0), 1.0, 1e-6),
        lambda: _decay_run(1.5, jnp.array(1.0), 1.0, 1e-6),
        label="ic_changes_output",
    )


def test_negative_parameter_changes_output():
    """a=1.5 vs a=0.5 must produce different outputs at matching x0, T."""
    assert_not_bitwise_equal(
        lambda: _decay_run(1.5, jnp.array(2.0), 1.0, 1e-6),
        lambda: _decay_run(0.5, jnp.array(2.0), 1.0, 1e-6),
        label="param_changes_output",
    )


def test_negative_end_time_changes_output():
    """T=1.0 vs T=2.0 must produce different outputs."""
    assert_not_bitwise_equal(
        lambda: _decay_run(1.5, jnp.array(2.0), 1.0, 1e-6),
        lambda: _decay_run(1.5, jnp.array(2.0), 2.0, 1e-6),
        label="tf_changes_output",
    )


def test_negative_tolerance_changes_output():
    """Same inputs but rtol=1e-3 vs rtol=1e-10: adaptive step sizing differs →
    bit-exact mismatch (both are correct solutions within tolerance, but the
    byte representation diverges)."""
    assert_not_bitwise_equal(
        lambda: _decay_run(1.5, jnp.array(2.0), 1.0, 1e-3),
        lambda: _decay_run(1.5, jnp.array(2.0), 1.0, 1e-10),
        label="tol_changes_output",
    )


def test_negative_seed_changes_rng_output():
    """RandomNumber(seed=42) vs seed=7 must produce different samples."""

    def run(seed):
        bld = jaxonomy.DiagramBuilder()
        rn = bld.add(RandomNumber(dt=0.1, shape=(3,), seed=seed, name="rn"))
        integ = bld.add(Integrator(jnp.zeros(3), name="i"))
        bld.connect(rn.output_ports[0], integ.input_ports[0])
        diagram = bld.build()
        ctx = diagram.create_context()
        opts = jaxonomy.SimulatorOptions(math_backend="jax", max_major_steps=50)
        return jaxonomy.simulate(diagram, ctx, (0.0, 0.5), options=opts).context[
            integ.system_id
        ].continuous_state

    assert_not_bitwise_equal(
        lambda: run(42), lambda: run(7), label="seed_changes_rng_output",
    )


def test_negative_white_noise_seed_changes_output():
    """WhiteNoise(seed=7) vs seed=11 must produce different samples."""

    def run(seed):
        bld = jaxonomy.DiagramBuilder()
        wn = bld.add(WhiteNoise(correlation_time=0.01, shape=(1,), seed=seed, name="wn"))
        integ = bld.add(Integrator(jnp.zeros(1), name="i"))
        bld.connect(wn.output_ports[0], integ.input_ports[0])
        diagram = bld.build()
        ctx = diagram.create_context()
        opts = jaxonomy.SimulatorOptions(math_backend="jax", max_major_steps=100)
        return jaxonomy.simulate(diagram, ctx, (0.0, 0.3), options=opts).context[
            integ.system_id
        ].continuous_state

    assert_not_bitwise_equal(
        lambda: run(7), lambda: run(11), label="wn_seed_changes_output",
    )
