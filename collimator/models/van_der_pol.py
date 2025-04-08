# Copyright (C) 2025 Collimator, Inc
# SPDX-License-Identifier: MIT

import jax.numpy as jnp
from ..framework import LeafSystem


class VanDerPol(LeafSystem):
    def __init__(self, x0=[0.0, 0.0], mu=1.0, input_port=False, name="van_der_pol"):
        super().__init__(name=name)
        self.declare_dynamic_parameter("mu", mu)

        if input_port:
            self.declare_input_port(name="u")

        self.declare_continuous_state(default_value=jnp.array(x0), ode=self.ode)
        self.declare_continuous_state_output()

    def ode(self, time, state, *inputs, **parameters):
        x, y = state.continuous_state
        mu = parameters["mu"]
        dy = mu * (1 - x**2) * y - x

        if inputs:
            (u,) = inputs
            dy += u

        return jnp.array([y, dy])
