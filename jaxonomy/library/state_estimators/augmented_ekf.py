# SPDX-License-Identifier: MIT

"""Augmented-state Extended Kalman Filter for joint state + parameter estimation."""

from typing import NamedTuple

import jax
import jax.numpy as jnp

from ...framework import parameters, DependencyTicket
from ...framework.leaf_system import LeafSystem
from ...backend import numpy_api as npa


class AugmentedStateEKF(LeafSystem):
    """
    Extended Kalman Filter with augmented state for **joint** state and parameter
    estimation.

    The block estimates both the plant state *x* and unknown parameters *θ* online
    by augmenting the state vector:

    .. code-block:: text

        z = [x; θ]   (shape: nx + n_params)

    with augmented dynamics and observation:

    .. code-block:: text

        z[n+1] = f_aug(z[n], u[n]) + noise
               = [f(x[n], u[n], θ[n]); θ[n]]   + [G_x w_x; w_θ]
        y[n]   = h(x[n], u[n], θ[n]) + v[n]

        E(w_x)  = E(w_θ) = E(v) = 0
        Cov(w_x) = Q_x,  Cov(w_θ) = Q_θ,  Cov(v) = R

    Parameters *θ* follow a **random-walk** model (``θ[n+1] = θ[n] + w_θ``).
    Setting ``Q_theta`` small makes parameters quasi-constant; increasing it allows
    tracking of slowly time-varying parameters.

    All Jacobians are computed automatically via ``jax.jacfwd``.

    ```
                    +--------------------+
    --- u[n] ------>|                    |----> x_hat[n]
                    |  AugmentedStateEKF |
    --- y[n] ------>|                    |----> theta_hat[n]
                    +--------------------+
    ```

    Input ports:
        (0) u : control vector at timestep n, shape ``(nu,)`` or scalar
        (1) y : measurement vector at timestep n, shape ``(ny,)`` or scalar

    Output ports:
        (0) x_hat     : state estimate, shape ``(nx,)``
        (1) theta_hat : parameter estimate, shape ``(n_params,)``

    Parameters:
        dt : float
            Sampling period.
        nx : int
            Dimension of the plant state *x*.
        n_params : int
            Dimension of the parameter vector *θ*.
        forward : Callable
            Discrete-time state transition: ``f(x, u, theta) -> x_next``.
            Must be JAX-traceable.
        observation : Callable
            Observation function: ``h(x, u, theta) -> y``.
            Must be JAX-traceable.
        G_x_func : Callable
            Process-noise input matrix for states: ``G_x(t) -> (nx, nw)`` array.
            Pass ``lambda t: jnp.eye(nx)`` for isotropic process noise.
        Q_x_func : Callable
            Process-noise covariance for states: ``Q_x(t, x, u, theta) -> (nw, nw)``.
        Q_theta : array_like
            Constant parameter diffusion covariance matrix ``(n_params, n_params)``.
            Small values → slow/no parameter drift.
        R_func : Callable
            Measurement noise covariance: ``R(t) -> (ny, ny)``.
        x_hat_0 : array_like
            Initial state estimate, shape ``(nx,)``.
        P_hat_0_x : array_like
            Initial state covariance, shape ``(nx, nx)``.
        theta_hat_0 : array_like
            Initial parameter estimate, shape ``(n_params,)``.
        P_hat_0_theta : array_like
            Initial parameter covariance, shape ``(n_params, n_params)``.

    Example::

        import jax.numpy as jnp
        from jaxonomy import library

        # Simple first-order system:  x[n+1] = a*x[n] + b*u[n]
        # where 'a' (decay rate) is unknown and must be estimated.

        def forward(x, u, theta):
            a = theta[0]
            return jnp.array([a * x[0] + u[0]])

        def observation(x, u, theta):
            return jnp.array([x[0]])

        aekf = library.AugmentedStateEKF(
            dt=0.1,
            nx=1,
            n_params=1,
            forward=forward,
            observation=observation,
            G_x_func=lambda t: jnp.eye(1),
            Q_x_func=lambda t, x, u, th: jnp.array([[0.01]]),
            Q_theta=jnp.array([[1e-4]]),
            R_func=lambda t: jnp.array([[0.1]]),
            x_hat_0=jnp.zeros(1),
            P_hat_0_x=jnp.eye(1),
            theta_hat_0=jnp.zeros(1),
            P_hat_0_theta=jnp.eye(1),
        )
    """

    class DiscreteStateType(NamedTuple):
        """Internal filter state (both minus=predicted and plus=corrected estimates)."""

        z_hat_minus: npa.ndarray  # predicted   augmented state [nx+n_params]
        P_hat_minus: npa.ndarray  # predicted   augmented covariance
        z_hat_plus: npa.ndarray   # corrected   augmented state [nx+n_params]
        P_hat_plus: npa.ndarray   # corrected   augmented covariance

    @parameters(
        static=[
            "dt",
            "nx",
            "n_params",
            "forward",
            "observation",
            "G_x_func",
            "Q_x_func",
            "Q_theta",
            "R_func",
            "x_hat_0",
            "P_hat_0_x",
            "theta_hat_0",
            "P_hat_0_theta",
        ],
    )
    def __init__(
        self,
        dt,
        nx,
        n_params,
        forward,
        observation,
        G_x_func,
        Q_x_func,
        Q_theta,
        R_func,
        x_hat_0,
        P_hat_0_x,
        theta_hat_0,
        P_hat_0_theta,
        name=None,
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)

        x_hat_0 = jnp.asarray(x_hat_0, dtype=float)
        P_hat_0_x = jnp.asarray(P_hat_0_x, dtype=float)
        theta_hat_0 = jnp.asarray(theta_hat_0, dtype=float)
        P_hat_0_theta = jnp.asarray(P_hat_0_theta, dtype=float)

        # Augmented initial state and covariance
        z_hat_0 = jnp.concatenate([x_hat_0, theta_hat_0])
        P_z_0 = jax.scipy.linalg.block_diag(P_hat_0_x, P_hat_0_theta)

        # Input ports
        self.u_in_index = self.declare_input_port(name="u")
        self.y_in_index = self.declare_input_port(name="y")

        # Internal discrete state
        self.declare_discrete_state(
            default_value=self.DiscreteStateType(
                z_hat_minus=z_hat_0,
                P_hat_minus=P_z_0,
                z_hat_plus=z_hat_0,
                P_hat_plus=P_z_0,
            ),
            as_array=False,
        )

        # Periodic update at each timestep
        self.declare_periodic_update(
            self._update,
            period=dt,
            offset=0.0,
        )

        # Build dependency tickets for feedthrough outputs
        u_ticket = self.input_ports[self.u_in_index].ticket
        y_ticket = self.input_ports[self.y_in_index].ticket
        prereqs = [DependencyTicket.xd, u_ticket, y_ticket]
        required_inputs = [self.u_in_index, self.y_in_index]

        # Output port 0: x_hat  (state estimate)
        self.declare_output_port(
            self._output_x_hat,
            period=dt,
            offset=0.0,
            default_value=x_hat_0,
            name="x_hat",
            requires_inputs=required_inputs,
            prerequisites_of_calc=prereqs,
        )

        # Output port 1: theta_hat  (parameter estimate)
        self.declare_output_port(
            self._output_theta_hat,
            period=dt,
            offset=0.0,
            default_value=theta_hat_0,
            name="theta_hat",
            requires_inputs=required_inputs,
            prerequisites_of_calc=prereqs,
        )

    def initialize(
        self,
        dt,
        nx,
        n_params,
        forward,
        observation,
        G_x_func,
        Q_x_func,
        Q_theta,
        R_func,
        x_hat_0,
        P_hat_0_x,
        theta_hat_0,
        P_hat_0_theta,
    ):
        """Called at context-creation time to store resolved parameters."""
        self.nx = nx
        self.np = n_params
        self.nz = nx + n_params

        self.forward = forward
        self.observation = observation
        self.G_x_func = G_x_func
        self.Q_x_func = Q_x_func
        self.Q_theta = jnp.asarray(Q_theta, dtype=float)
        self.R_func = R_func

        # Determine ny from R matrix shape
        self.ny = self.R_func(0.0).shape[0]

        # Build augmented dynamics and observation (closures over nx, n_params)
        _nx = nx

        def f_aug(z, u):
            x, theta = z[:_nx], z[_nx:]
            x_next = forward(x, u, theta)
            return jnp.concatenate([x_next, theta])

        def h_aug(z, u):
            x, theta = z[:_nx], z[_nx:]
            return jnp.atleast_1d(observation(x, u, theta))

        self.f_aug = f_aug
        self.h_aug = h_aug

        # Jacobian functions for the augmented system
        self.jac_f_aug = jax.jacfwd(f_aug)   # ∂f_aug/∂z  [nz × nz]
        self.jac_h_aug = jax.jacfwd(h_aug)   # ∂h_aug/∂z  [ny × nz]

        self.eye_z = jnp.eye(self.nz)

    # ──────────────────────────────────────────────────────────────────────────
    # EKF correct step
    # ──────────────────────────────────────────────────────────────────────────

    def _correct(self, time, z_hat_minus, P_hat_minus, u, y):
        """Update estimate using current measurement."""
        y = jnp.atleast_1d(jnp.asarray(y, dtype=float))
        u = jnp.atleast_1d(jnp.asarray(u, dtype=float))

        C = self.jac_h_aug(z_hat_minus, u).reshape((self.ny, self.nz))
        R = self.R_func(time)

        # Kalman gain
        S = C @ P_hat_minus @ C.T + R
        K = P_hat_minus @ C.T @ jnp.linalg.inv(S)

        # State update
        innovation = y - self.h_aug(z_hat_minus, u)
        z_hat_plus = z_hat_minus + K @ innovation

        # Covariance update
        P_hat_plus = (self.eye_z - K @ C) @ P_hat_minus

        return z_hat_plus, P_hat_plus

    # ──────────────────────────────────────────────────────────────────────────
    # EKF propagate step
    # ──────────────────────────────────────────────────────────────────────────

    def _propagate(self, time, z_hat_plus, P_hat_plus, u):
        """Predict next state from corrected estimate."""
        u = jnp.atleast_1d(jnp.asarray(u, dtype=float))

        # Augmented state Jacobian
        A = self.jac_f_aug(z_hat_plus, u).reshape((self.nz, self.nz))

        # Augmented process noise covariance
        #   G_aug = block_diag(G_x, I_theta)
        #   Q_aug = block_diag(G_x Q_x G_x^T, Q_theta)
        x_plus, theta_plus = z_hat_plus[:self.nx], z_hat_plus[self.nx:]
        G_x = self.G_x_func(time)
        Q_x = self.Q_x_func(time, x_plus, u, theta_plus)
        GQG_x = G_x @ Q_x @ G_x.T
        GQG_aug = jax.scipy.linalg.block_diag(GQG_x, self.Q_theta)

        # Propagate
        z_hat_minus = self.f_aug(z_hat_plus, u)
        P_hat_minus = A @ P_hat_plus @ A.T + GQG_aug

        return z_hat_minus, P_hat_minus

    # ──────────────────────────────────────────────────────────────────────────
    # Periodic state update
    # ──────────────────────────────────────────────────────────────────────────

    def _update(self, time, state, *inputs, **params):
        u, y = inputs
        z_hat_minus = state.discrete_state.z_hat_minus
        P_hat_minus = state.discrete_state.P_hat_minus

        z_hat_plus, P_hat_plus = self._correct(time, z_hat_minus, P_hat_minus, u, y)
        z_hat_minus_next, P_hat_minus_next = self._propagate(
            time, z_hat_plus, P_hat_plus, u
        )

        return self.DiscreteStateType(
            z_hat_minus=z_hat_minus_next,
            P_hat_minus=P_hat_minus_next,
            z_hat_plus=z_hat_plus,
            P_hat_plus=P_hat_plus,
        )

    # ──────────────────────────────────────────────────────────────────────────
    # Output callbacks  (feedthrough: recompute correction with current inputs)
    # ──────────────────────────────────────────────────────────────────────────

    def _output_x_hat(self, time, state, *inputs, **params):
        u, y = inputs
        z_hat_minus = state.discrete_state.z_hat_minus
        P_hat_minus = state.discrete_state.P_hat_minus
        z_hat_plus, _ = self._correct(time, z_hat_minus, P_hat_minus, u, y)
        return z_hat_plus[:self.nx]

    def _output_theta_hat(self, time, state, *inputs, **params):
        u, y = inputs
        z_hat_minus = state.discrete_state.z_hat_minus
        P_hat_minus = state.discrete_state.P_hat_minus
        z_hat_plus, _ = self._correct(time, z_hat_minus, P_hat_minus, u, y)
        return z_hat_plus[self.nx:]
