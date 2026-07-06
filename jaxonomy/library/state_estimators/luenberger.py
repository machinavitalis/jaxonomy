# SPDX-License-Identifier: MIT

"""Luenberger observer block (T-109-followup-with-observer).

Counterpart to :class:`jaxonomy.library.KalmanFilter` for callers who
have already computed a steady-state observer gain matrix ``L``
(via :func:`scipy.signal.place`, pole-placement, an LQR/LQG design
flow, or any other method). The Luenberger observer is the simpler
half of the Kalman pair — no Riccati equation, no covariance update,
just the gain-corrected predictor:

.. code-block:: text

    x_hat[k+1] = A·x_hat[k] + B·u[k] + L·(y[k] - C·x_hat[k] - D·u[k])

where ``A, B, C, D`` describe the (discrete-time) plant model, ``L``
is the user-supplied observer gain, ``u[k]`` is the control input,
``y[k]`` is the noisy measurement, and ``x_hat[k]`` is the state
estimate output by the block.

Public API:

* :class:`Luenberger` — the discrete-time observer block.
* :func:`with_observer` — wiring helper that attaches a Luenberger
  observer to a built plant diagram. Returns a new diagram exposing
  ``x_hat`` alongside the original outputs.
"""

from __future__ import annotations

import jax.numpy as jnp

from ...framework import LeafSystem, parameters


__all__ = ["Luenberger"]


class Luenberger(LeafSystem):
    """Discrete-time Luenberger observer with user-supplied gain ``L``.

    State-update equation:

    .. code-block:: text

        x_hat[k+1] = A·x_hat[k] + B·u[k] + L·(y[k] - C·x_hat[k] - D·u[k])

    Args:
        dt: Discrete sample period (seconds). Must match the plant's
            sample period (or, for continuous plants, the discretisation
            period chosen for the design).
        A, B, C, D: Plant state-space matrices (typically the output of
            ``jaxonomy.discretize(linearize(plant, op), dt)``). ``D``
            defaults to a zero matrix of compatible shape.
        L: Observer gain matrix, shape ``(n_states, n_outputs)``. The
            caller computes this — e.g. via ``scipy.signal.place`` for
            pole placement, or via a steady-state Kalman gain.
        x_hat_0: Initial state estimate. Defaults to zeros.

    Input ports:
        (0) ``u``: control input vector, shape ``(n_inputs,)``.
        (1) ``y``: noisy measurement vector, shape ``(n_outputs,)``.

    Output ports:
        (0) ``x_hat``: state estimate vector, shape ``(n_states,)``.

    Notes:
        This block is the simpler half of the Kalman pair — the design
        cost (computing ``L``) is paid offline, leaving only the cheap
        runtime update. If you want online Riccati-based gain updates
        instead, use :class:`KalmanFilter`. If you have a continuous
        plant and want the steady-state infinite-horizon Kalman gain,
        use :class:`InfiniteHorizonKalmanFilter`.
    """

    @parameters(static=["dt", "A", "B", "C", "D", "L", "x_hat_0"])
    def __init__(
        self,
        dt,
        A,
        B,
        C,
        L,
        D=None,
        x_hat_0=None,
        *,
        name=None,
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)

        A_arr = jnp.asarray(A)
        B_arr = jnp.asarray(B)
        C_arr = jnp.asarray(C)
        L_arr = jnp.asarray(L)
        if A_arr.ndim != 2 or A_arr.shape[0] != A_arr.shape[1]:
            raise ValueError(
                f"Luenberger {self.name!r}: A must be square; got shape "
                f"{tuple(A_arr.shape)}."
            )
        n = A_arr.shape[0]
        m = B_arr.shape[1] if B_arr.ndim == 2 else 1
        p = C_arr.shape[0] if C_arr.ndim == 2 else 1
        if L_arr.shape != (n, p):
            raise ValueError(
                f"Luenberger {self.name!r}: L must have shape "
                f"(n_states, n_outputs) = ({n}, {p}); got "
                f"{tuple(L_arr.shape)}."
            )
        if D is None:
            D = jnp.zeros((p, m))

        self._n = n
        self._m = m
        self._p = p

        # Two input ports: u (control), y (measurement).
        self.declare_input_port()  # u
        self.declare_input_port()  # y
        self._update_idx = self.declare_periodic_update()
        self._output_idx = self.declare_output_port()

    def initialize(self, dt, A, B, C, D=None, L=None, x_hat_0=None):
        A = jnp.asarray(A)
        B = jnp.asarray(B)
        C = jnp.asarray(C)
        L = jnp.asarray(L)
        if D is None:
            D = jnp.zeros((self._p, self._m))
        else:
            D = jnp.asarray(D)
        if x_hat_0 is None:
            x_hat_0 = jnp.zeros((self._n,))
        else:
            x_hat_0 = jnp.asarray(x_hat_0)

        self.declare_discrete_state(default_value=x_hat_0, as_array=True)

        dt_f = float(dt)

        def _update(time, state, *inputs, **_params):
            u = jnp.atleast_1d(inputs[0])
            y = jnp.atleast_1d(inputs[1])
            x = state.discrete_state
            y_pred = C @ x + D @ u
            innovation = y - y_pred
            return A @ x + B @ u + L @ innovation

        # ``offset=dt`` so the first update fires at t=dt (matches the
        # UnitDelay convention: x_hat[0] = x_hat_0 visible from t=0 to dt).
        self.configure_periodic_update(
            self._update_idx,
            _update,
            period=dt_f,
            offset=dt_f,
        )

        def _output(time, state, *inputs, **_params):
            return state.discrete_state

        # Output is sampled at the same discrete cadence — period/offset
        # match the canonical discrete-output pattern used elsewhere
        # (UnitDelay, KalmanFilter etc.).
        self.configure_output_port(
            self._output_idx,
            _output,
            period=dt_f,
            offset=0.0,
            default_value=x_hat_0,
        )
