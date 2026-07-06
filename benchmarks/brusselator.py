# SPDX-License-Identifier: MIT

"""Brusselator 2D model with periodic boundary conditions."""

from jaxonomy import LeafSystem
import jax.numpy as jnp
import numpy as np

class Brusselator2D(LeafSystem):
    def __init__(self, N: int = 8, A: float = 3.4, B: float = 1.0, alpha: float = 10.0, **kwargs):
        super().__init__(**kwargs)
        self.N = N
        self.A = A
        self.B = B
        self.alpha = alpha
        
        # Spatial step size dx (range 0 to 1 with N points)
        self.dx = 1.0 / (N - 1) if N > 1 else 1.0
        
        # Grid coordinates
        self.xyd = np.linspace(0.0, 1.0, N)
        
        # Grid coordinates for JAX meshgrid (JIT compatible)
        self.X, self.Y = jnp.meshgrid(self.xyd, self.xyd, indexing="ij")
        self.dist_sq = (self.X - 0.3) ** 2 + (self.Y - 0.6) ** 2
        self.f_mask = jnp.where(self.dist_sq <= 0.1 ** 2, 5.0, 0.0)
        
        # Initialize state values
        u0 = np.zeros((N, N, 2))
        for i in range(N):
            for j in range(N):
                x = self.xyd[i]
                y = self.xyd[j]
                u0[i, j, 0] = 22.0 * (y * (1.0 - y)) ** 1.5
                u0[i, j, 1] = 27.0 * (x * (1.0 - x)) ** 1.5
                
        self.declare_continuous_state(
            default_value=jnp.array(u0.flatten()),
            ode=self._ode
        )
        self.declare_continuous_state_output(name="state")

    def _ode(self, t, state, **p):
        u = state.continuous_state.reshape((self.N, self.N, 2))
        
        # Compute shifts for periodic boundary conditions using jnp.roll
        u_im1 = jnp.roll(u, shift=1, axis=0)   # u[i-1, j, :]
        u_ip1 = jnp.roll(u, shift=-1, axis=0)  # u[i+1, j, :]
        u_jm1 = jnp.roll(u, shift=1, axis=1)   # u[i, j-1, :]
        u_jp1 = jnp.roll(u, shift=-1, axis=1)  # u[i, j+1, :]
        
        # Laplacian stencil
        laplacian = (u_im1 + u_ip1 + u_jm1 + u_jp1 - 4.0 * u) / (self.dx ** 2)
        
        U = u[:, :, 0]
        V = u[:, :, 1]
        
        # Time-dependent forcing perturbation
        f_val = jnp.where(t >= 1.1, self.f_mask, 0.0)
        
        # Brusselator PDE terms
        dU = self.alpha * laplacian[:, :, 0] + self.B + (U ** 2) * V - (self.A + 1.0) * U + f_val
        dV = self.alpha * laplacian[:, :, 1] + self.A * U - (U ** 2) * V
        
        # Stack and return flat state derivative
        du = jnp.stack([dU, dV], axis=-1)
        return du.flatten()
