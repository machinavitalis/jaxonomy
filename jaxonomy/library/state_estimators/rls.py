# SPDX-License-Identifier: MIT

"""Recursive Least Squares (RLS) estimator block for online parameter identification."""

from typing import NamedTuple

import jax.numpy as jnp

from ...framework import parameters, DependencyTicket
from ...framework.leaf_system import LeafSystem
from ...backend import numpy_api as npa


class RecursiveLeastSquares(LeafSystem):
    """
    Recursive Least Squares (RLS) estimator for online parameter identification
    in linear-in-parameters models:

        ``y[k] = φ[k]ᵀ θ + noise``

    where:
      - ``y[k]``   is a scalar (or vector) measurement at timestep k,
      - ``φ[k]``   is a regressor vector of size ``n_params``,
      - ``θ``      is the unknown parameter vector to be estimated.

    The RLS update equations with forgetting factor λ are:

    .. code-block:: text

        e[k]    = y[k]  − φ[k]ᵀ θ̂[k−1]               (prediction error)
        K[k]    = P[k−1] φ[k] / (λ + φ[k]ᵀ P[k−1] φ[k])  (Kalman gain)
        θ̂[k]   = θ̂[k−1] + K[k] e[k]                  (parameter update)
        P[k]    = (P[k−1] − K[k] φ[k]ᵀ P[k−1]) / λ    (covariance update)

    A forgetting factor ``λ < 1`` down-weights older measurements, making the
    estimator track slowly time-varying parameters.  ``λ = 1`` (default) is the
    classic batch RLS equivalent.

    The block is fully JAX-traceable and compatible with JIT/autodiff.

    ```
                    +--------------------+
    --- phi[k] ---->|                    |----> theta_hat[k]
                    |  Recursive Least   |----> P[k]
    --- y[k] ------>|      Squares       |----> prediction_error[k]
                    +--------------------+
    ```

    Input ports:
        (0) phi : regressor vector at timestep k, shape ``(n_params,)``
        (1) y   : scalar measurement at timestep k

    Output ports:
        (0) theta_hat        : parameter estimate, shape ``(n_params,)``
        (1) P                : parameter covariance matrix, shape ``(n_params, n_params)``
        (2) prediction_error : scalar prediction residual ``e = y − φᵀ θ̂``

    Parameters:
        dt : float
            Sampling period.
        n_params : int
            Number of parameters to estimate (dimension of θ).
        theta_0 : array_like, optional
            Initial parameter estimate, shape ``(n_params,)``.
            Defaults to the zero vector.
        P_0 : array_like, optional
            Initial covariance matrix, shape ``(n_params, n_params)``.
            Defaults to ``1e4 * I``, which encodes high initial uncertainty.
        forgetting_factor : float, optional
            Forgetting factor λ ∈ (0, 1].  Default ``1.0`` (no forgetting).

    Example::

        import numpy as np
        import jaxonomy
        from jaxonomy import library, DiagramBuilder, SimulatorOptions

        # True parameters: y = 2*phi_0 + 3*phi_1
        TRUE_THETA = np.array([2.0, 3.0])
        DT = 0.1

        rls = library.RecursiveLeastSquares(
            dt=DT, n_params=2,
            forgetting_factor=1.0,
        )
    """

    class DiscreteStateType(NamedTuple):
        """Internal state: current parameter estimate and covariance."""

        theta_hat: npa.ndarray  # shape (n_params,)
        P: npa.ndarray          # shape (n_params, n_params)

    @parameters(
        static=["dt", "n_params", "theta_0", "P_0", "forgetting_factor"],
    )
    def __init__(
        self,
        dt,
        n_params,
        theta_0=None,
        P_0=None,
        forgetting_factor=1.0,
        name=None,
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)

        # Resolve defaults for array arguments so that ports can be declared
        # with appropriate shapes at construction time.
        if theta_0 is None:
            theta_0 = jnp.zeros(n_params)
        if P_0 is None:
            P_0 = jnp.eye(n_params) * 1e4

        theta_0 = jnp.asarray(theta_0, dtype=float)
        P_0 = jnp.asarray(P_0, dtype=float)

        # Input ports
        self.phi_in_index = self.declare_input_port(name="phi")
        self.y_in_index = self.declare_input_port(name="y")

        # Internal discrete state: (theta_hat, P)
        self.declare_discrete_state(
            default_value=self.DiscreteStateType(theta_hat=theta_0, P=P_0),
            as_array=False,
        )

        # Periodic update – runs at each timestep
        self.declare_periodic_update(
            self._update,
            period=dt,
            offset=0.0,
        )

        # Dependency tickets for feedthrough outputs
        phi_ticket = self.input_ports[self.phi_in_index].ticket
        y_ticket = self.input_ports[self.y_in_index].ticket
        prereqs = [DependencyTicket.xd, phi_ticket, y_ticket]
        required_inputs = [self.phi_in_index, self.y_in_index]

        # Output port 0: theta_hat  (feedthrough on phi, y)
        self.declare_output_port(
            self._output_theta_hat,
            period=dt,
            offset=0.0,
            default_value=theta_0,
            name="theta_hat",
            requires_inputs=required_inputs,
            prerequisites_of_calc=prereqs,
        )

        # Output port 1: P  (feedthrough on phi, y)
        self.declare_output_port(
            self._output_P,
            period=dt,
            offset=0.0,
            default_value=P_0,
            name="P",
            requires_inputs=required_inputs,
            prerequisites_of_calc=prereqs,
        )

        # Output port 2: prediction_error  (feedthrough on phi, y)
        self.declare_output_port(
            self._output_prediction_error,
            period=dt,
            offset=0.0,
            default_value=jnp.zeros(()),
            name="prediction_error",
            requires_inputs=required_inputs,
            prerequisites_of_calc=prereqs,
        )

    def initialize(
        self,
        dt,
        n_params,
        theta_0=None,
        P_0=None,
        forgetting_factor=1.0,
    ):
        """Called at context-creation time to store resolved parameters."""
        self.n = n_params
        self.lam = float(forgetting_factor)

    # ──────────────────────────────────────────────────────────────────────────
    # Core RLS computation (shared between update and outputs)
    # ──────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _rls_step(theta_hat, P, phi, y, lam):
        """One RLS correction step.  Returns (theta_new, P_new, error)."""
        phi = jnp.atleast_1d(jnp.asarray(phi, dtype=float)).ravel()
        y_scalar = jnp.asarray(y, dtype=float).reshape(())

        # Prediction error
        e = y_scalar - jnp.dot(phi, theta_hat)

        # Kalman gain
        Pphi = jnp.dot(P, phi)
        denom = lam + jnp.dot(phi, Pphi)
        K = Pphi / denom

        # Parameter update
        theta_new = theta_hat + K * e

        # Covariance update  (Joseph form is more numerically robust but
        # the standard form is sufficient here and cheaper to compute)
        P_new = (P - jnp.outer(K, phi) @ P) / lam

        return theta_new, P_new, e

    # ──────────────────────────────────────────────────────────────────────────
    # Periodic state update
    # ──────────────────────────────────────────────────────────────────────────

    def _update(self, time, state, *inputs, **params):
        phi, y = inputs
        theta_hat = state.discrete_state.theta_hat
        P = state.discrete_state.P

        theta_new, P_new, _ = self._rls_step(theta_hat, P, phi, y, self.lam)

        return self.DiscreteStateType(theta_hat=theta_new, P=P_new)

    # ──────────────────────────────────────────────────────────────────────────
    # Output callbacks  (feedthrough: recompute correction with current inputs)
    # ──────────────────────────────────────────────────────────────────────────

    def _output_theta_hat(self, time, state, *inputs, **params):
        phi, y = inputs
        theta_hat = state.discrete_state.theta_hat
        P = state.discrete_state.P
        theta_new, _, _ = self._rls_step(theta_hat, P, phi, y, self.lam)
        return theta_new

    def _output_P(self, time, state, *inputs, **params):
        phi, y = inputs
        theta_hat = state.discrete_state.theta_hat
        P = state.discrete_state.P
        _, P_new, _ = self._rls_step(theta_hat, P, phi, y, self.lam)
        return P_new

    def _output_prediction_error(self, time, state, *inputs, **params):
        phi, y = inputs
        theta_hat = state.discrete_state.theta_hat
        phi = jnp.atleast_1d(jnp.asarray(phi, dtype=float)).ravel()
        y_scalar = jnp.asarray(y, dtype=float).reshape(())
        return y_scalar - jnp.dot(phi, theta_hat)
