# Copyright (C) 2025 Collimator, Inc
# SPDX-License-Identifier: MIT

"""
Cost and loss function structures meant for utilization in optimal control,
optimization, ML tasks, etc.
"""

import jax.numpy as jnp

from ..library import ReduceBlock


class QuadraticCost(ReduceBlock):
    """LQR-type quadratic cost function for a state and input.

    Computes the cost as x'Qx + u'Ru, where Q and R are the cost matrices.
    In order to compute a running cost, combine this with an `Integrator`
    or `IntegratorDiscrete` block.
    """

    def __init__(self, Q, R, name=None):
        super().__init__(2, self._cost, name=name)
        self.Q = Q
        self.R = R

    def _cost(self, inputs):
        x, u = inputs
        J = jnp.dot(x, jnp.dot(self.Q, x)) + jnp.dot(u, jnp.dot(self.R, u))
        return J.squeeze()
