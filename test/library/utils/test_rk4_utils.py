# SPDX-License-Identifier: MIT

import jax.numpy as jnp
import pytest

from jaxonomy.testing import requires_jax
from jaxonomy.library.utils.rk4_utils import _rk4_step_constant_u, rk4_major_step_constant_u


def simple_linear_ode(x, u, t):
    """
    dx/dt = -x + u
    """
    return -x + u

@requires_jax()
def test_rk4_step_constant_u():
    x0 = jnp.array([1.0])
    u = jnp.array([2.0])
    t = 0.0
    dh = 0.1
    
    # Run one RK4 step
    x_next = _rk4_step_constant_u(t, x0, u, dh, simple_linear_ode)
    
    # Analytical solution for dx/dt = -x + u with constant u:
    # x(t) = e^(-t) * x0 + (1 - e^(-t)) * u
    x_analytical = jnp.exp(-dh) * x0 + (1.0 - jnp.exp(-dh)) * u
    
    # RK4 is O(dh^4) accurate, so for dh=0.1 error is very small
    assert jnp.allclose(x_next, x_analytical, atol=1e-5)

@requires_jax()
def test_rk4_major_step_constant_u():
    x0 = jnp.array([1.0])
    u = jnp.array([2.0])
    t0 = 0.0
    dt = 1.0  # Major step size
    nh = 10   # Minor steps
    
    # Run major RK4 step
    x_next = rk4_major_step_constant_u(t0, x0, u, dt, nh, simple_linear_ode)
    
    # Analytical solution at t0 + dt:
    x_analytical = jnp.exp(-dt) * x0 + (1.0 - jnp.exp(-dt)) * u
    
    # Check accuracy
    assert jnp.allclose(x_next, x_analytical, atol=1e-5)

@requires_jax()
def test_rk4_major_step_constant_u_multiple_states():
    # dx1/dt = -x1 + u1
    # dx2/dt = -2*x2 + u2
    def multi_ode(x, u, t):
        A = jnp.array([[-1.0, 0.0], [0.0, -2.0]])
        return A @ x + u
        
    x0 = jnp.array([1.0, 1.0])
    u = jnp.array([2.0, 3.0])
    t0 = 0.0
    dt = 0.5
    nh = 5
    
    x_next = rk4_major_step_constant_u(t0, x0, u, dt, nh, multi_ode)
    
    x1_analytical = jnp.exp(-dt) * x0[0] + (1.0 - jnp.exp(-dt)) * u[0]
    
    # For dx2/dt = -2*x2 + u2  => d/dt(x2) + 2*x2 = u2 
    # x2(t) = e^(-2t)*x2(0) + integral_0^t e^(-2(t-tau)) u2 dtau
    #       = e^(-2t)*x2(0) + (1 - e^(-2t))/2 * u2
    x2_analytical = jnp.exp(-2.0*dt) * x0[1] + (1.0 - jnp.exp(-2.0*dt))/2.0 * u[1]
    
    x_analytical = jnp.array([x1_analytical, x2_analytical])
    
    assert jnp.allclose(x_next, x_analytical, atol=1e-5)
