# SPDX-License-Identifier: MIT
"""
T-011 — linearize_to_lti helper tests.

Confirms the helper produces an LTISystem whose (A, B, C, D) matrices
match the analytic linearization, and that the resulting LTISystem
drops into a DiagramBuilder correctly.
"""

from __future__ import annotations

import numpy as np
import pytest
import jax.numpy as jnp

import jaxonomy
from jaxonomy.library import Gain, linearize_to_lti, LTISystem
from jaxonomy.testing.markers import skip_if_not_jax

skip_if_not_jax()


def test_linearize_second_order_spring_damper():
    """Linearize dx/dt = v, dv/dt = -k·x - b·v + F about (x=0, v=0, F=0).

    Expected matrices:
      A = [[0, 1], [-k, -b]]
      B = [[0], [1]]
      C = [[1, 0], [0, 1]]  (full-state output)
      D = [[0], [0]]
    """
    k, b = 4.0, 0.5

    class SecondOrder(jaxonomy.LeafSystem):
        def __init__(self, **kw):
            super().__init__(name="plant", **kw)
            self.declare_dynamic_parameter("k", k)
            self.declare_dynamic_parameter("b", b)
            self.declare_input_port(name="F")
            self.declare_continuous_state(default_value=jnp.zeros(2), ode=self._ode)
            self.declare_continuous_state_output()

        def _ode(self, time, state, F, **params):
            x, v = state.continuous_state
            return jnp.array([v, -params["k"] * x - params["b"] * v + F[0]])

    plant = SecondOrder()
    plant.input_ports[0].fix_value(jnp.zeros(1))
    ctx = plant.create_context()

    lti = linearize_to_lti(plant, ctx)
    assert isinstance(lti, LTISystem)

    # Inspect the resulting LTISystem's parameters.
    A = np.asarray(lti.dynamic_parameters["A"].value)
    B = np.asarray(lti.dynamic_parameters["B"].value)
    C = np.asarray(lti.dynamic_parameters["C"].value)
    D = np.asarray(lti.dynamic_parameters["D"].value)

    np.testing.assert_allclose(A, np.array([[0.0, 1.0], [-k, -b]]), atol=1e-6)
    np.testing.assert_allclose(B, np.array([[0.0], [1.0]]), atol=1e-6)
    np.testing.assert_allclose(C, np.eye(2), atol=1e-6)
    np.testing.assert_allclose(D, np.zeros((2, 1)), atol=1e-6)


def test_lti_block_drops_into_builder():
    """The returned LTISystem composes with DiagramBuilder like any
    other block — add, connect, simulate."""
    k, b = 9.0, 1.0

    class SecondOrder(jaxonomy.LeafSystem):
        def __init__(self, **kw):
            super().__init__(name="plant", **kw)
            self.declare_dynamic_parameter("k", k)
            self.declare_dynamic_parameter("b", b)
            self.declare_input_port(name="F")
            self.declare_continuous_state(default_value=jnp.zeros(2), ode=self._ode)
            self.declare_continuous_state_output()

        def _ode(self, time, state, F, **params):
            x, v = state.continuous_state
            return jnp.array([v, -params["k"] * x - params["b"] * v + F[0]])

    plant = SecondOrder()
    plant.input_ports[0].fix_value(jnp.zeros(1))
    ctx = plant.create_context()
    lti = linearize_to_lti(plant, ctx)

    # Use it in a simulation: feed constant zero input, x(0)=[1, 0].
    bld = jaxonomy.DiagramBuilder()
    from jaxonomy.library import Constant
    src = bld.add(Constant(jnp.zeros(1), name="u"))
    lti_blk = bld.add(lti)
    bld.connect(src.output_ports[0], lti_blk.input_ports[0])
    diagram = bld.build()
    ctx = diagram.create_context()
    ctx = ctx.with_subcontext(
        lti_blk.system_id,
        ctx[lti_blk.system_id].with_continuous_state(jnp.array([1.0, 0.0])),
    )
    opts = jaxonomy.SimulatorOptions(math_backend="jax")
    res = jaxonomy.simulate(diagram, ctx, (0.0, 10.0), options=opts)
    x_final = np.asarray(res.context[lti_blk.system_id].continuous_state)
    # Damped spring → origin over 10 s.
    assert np.all(np.isfinite(x_final))
    assert np.max(np.abs(x_final)) < 0.1, f"did not decay to origin: {x_final}"
