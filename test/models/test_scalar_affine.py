# Copyright (C) 2025 Collimator, Inc
# SPDX-License-Identifier: MIT

import pytest

import jax.numpy as jnp

import collimator
from collimator.library import (
    Gain,
    Integrator,
)


pytestmark = pytest.mark.minimal


#
#  dx/dt = -a * x + b
#
# - Test diagram creation with primitives
# - Test diagram simulation
# - Test autodiff of diagram simulation

# With b=0, this is a simple exponential decay
#  x(t) = x(0)*exp(-a*t)
#
# So dx(t)/dx(0) = exp(-a*t)


def test_scalar_linear():
    a = 1.5
    x0 = 4.0
    builder = collimator.DiagramBuilder()
    Gain_0 = builder.add(Gain(-a, name="Gain_0"))
    Integrator_0 = builder.add(Integrator(x0, name="Integrator_0"))

    builder.connect(Gain_0.output_ports[0], Integrator_0.input_ports[0])
    builder.connect(Integrator_0.output_ports[0], Gain_0.input_ports[0])

    diagram = builder.build()
    ctx = diagram.create_context()

    t0, t1 = 0.0, 2.0
    result = collimator.simulate(
        diagram,
        ctx,
        (t0, t1),
    )
    xf = result.context[Integrator_0.system_id].continuous_state

    assert jnp.allclose(xf, x0 * jnp.exp(-a * t1))
