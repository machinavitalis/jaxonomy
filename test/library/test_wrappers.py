# SPDX-License-Identifier: MIT

import jax.numpy as jnp
import numpy as np

import jaxonomy
from jaxonomy.library.wrappers import ode_block, feedthrough_block
from jaxonomy.testing import requires_jax


@requires_jax()
def test_ode_block():
    # Define a simple ODE: dx/dt = -x + u
    @ode_block(state_dim=1, num_inputs=1, name="my_ode")
    def linear_ode(time, state, u, **params):
        x = state.continuous_state
        return -x + u

    assert linear_ode.name == "my_ode"
    assert len(linear_ode.input_ports) == 1
    assert linear_ode.input_ports[0].name == "my_ode:input[0]"
    
    # Test the evaluation of the ODE
    context = linear_ode.create_context()
    context = context.with_continuous_state(jnp.array([2.0]))
    
    linear_ode.input_ports[0].fix_value(jnp.array([5.0]))
    
    derivatives = linear_ode.eval_time_derivatives(context)
    
    # dx/dt = -2.0 + 5.0 = 3.0
    assert jnp.allclose(derivatives, jnp.array([3.0]))


@requires_jax()
def test_feedthrough_block():
    # Define a simple feedthrough block: y = 2*u
    @feedthrough_block
    def gain_block(u):
        return 2.0 * u

    assert gain_block.name == "gain_block"
    assert len(gain_block.input_ports) == 1
    assert len(gain_block.output_ports) == 1
    
    # Test evaluation
    gain_block.input_ports[0].fix_value(jnp.array([3.0]))
    context = gain_block.create_context()
    
    y = gain_block.output_ports[0].eval(context)
    
    assert jnp.allclose(y, jnp.array([6.0]))

if __name__ == "__main__":
    test_ode_block()
    test_feedthrough_block()
