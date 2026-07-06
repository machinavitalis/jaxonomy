# SPDX-License-Identifier: MIT
"""
Linear Quadratic Gaussian (LQG) controller (T-014).

Composes an infinite-horizon Kalman filter (observer) with an
infinite-horizon LQR (regulator) per the separation principle:

  - The observer is synthesised independently from (A, G, C, Qn, Rn):
        L = argmin E[||x − x̂||²]
  - The regulator is synthesised independently from (A, B, Qc, Rc):
        K = argmin J = ∫(x̂ᵀ Qc x̂ + uᵀ Rc u) dt
  - The LQG controller is the cascade: measurement y → x̂ → u = −K x̂.

Input ports
    (0) y  — continuous-time measurement vector.
Output ports
    (0) u  — continuous-time control vector.

Parameters
    A, B, C, D : plant state-space matrices.
    G          : process-noise input matrix.  Defaults to ``I`` if None.
    Qn, Rn     : process- and measurement-noise covariances for the
                 Kalman filter.
    Qc, Rc     : state and control weighting matrices for the LQR.
    x_hat_0    : initial observer estimate.  Defaults to zeros.

The block integrates the observer's continuous-time state internally;
from the user's perspective it behaves as a direct map y → u with
internal memory.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from ..backend import numpy_api as npa
from ..framework import LeafSystem
from ..lazy_loader import LazyLoader


control = LazyLoader("control", globals(), "control")


__all__ = ["LinearQuadraticGaussian"]


class LinearQuadraticGaussian(LeafSystem):
    """Continuous-time infinite-horizon LQG controller (separation principle).

    The observer uses the algebraic Riccati solution for the Kalman gain
    ``L`` given ``(A, G, C, Qn, Rn)``; the regulator uses the algebraic
    Riccati solution for the feedback gain ``K`` given ``(A, B, Qc, Rc)``.
    Both use the ``control`` library's ``lqe`` / ``lqr`` helpers.

    Input / output shapes follow the plant: ``y`` is ``(ny,)``, ``u`` is
    ``(nu,)``, and the observer's internal state is ``(nx,)``.
    """

    def __init__(
        self,
        A: np.ndarray,
        B: np.ndarray,
        C: np.ndarray,
        D: np.ndarray,
        Qn: np.ndarray,
        Rn: np.ndarray,
        Qc: np.ndarray,
        Rc: np.ndarray,
        G: Optional[np.ndarray] = None,
        x_hat_0: Optional[np.ndarray] = None,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        A = np.asarray(A)
        B = np.asarray(B)
        C = np.asarray(C)
        D = np.asarray(D)
        nx, nu = B.shape
        ny = C.shape[0]
        if G is None:
            G = np.eye(nx)
        G = np.asarray(G)

        # Observer gain from LQE (Kalman filter at steady state).
        L, _P_obs, _E_obs = control.lqe(A, G, C, Qn, Rn)

        # Regulator gain from LQR.
        K, _P_reg, _E_reg = control.lqr(A, B, Qc, Rc)

        self.A = A
        self.B = B
        self.C = C
        self.D = D
        self.L = np.asarray(L)
        self.K = np.asarray(K)

        # Pre-compute observer closed-loop matrices for cheaper ODE eval:
        #     dx̂/dt = (A − B·K − L·C) x̂ + L·y     (u = −K·x̂)
        # Rearranging lets us eval u without re-forming it.
        self.A_obs = A - B @ self.K - self.L @ C
        self.B_obs = self.L  # multiplies y

        self.nx = nx
        self.nu = nu
        self.ny = ny

        if x_hat_0 is None:
            x_hat_0 = np.zeros(nx)
        x_hat_0 = np.asarray(x_hat_0)

        # I/O: one input (y), one output (u).  The observer state is the
        # block's continuous state.
        self.declare_input_port(name="y")
        self.declare_continuous_state(
            ode=self._ode, shape=x_hat_0.shape, default_value=x_hat_0, as_array=True,
        )
        self.declare_output_port(
            self._compute_u,
            prerequisites_of_calc=[self.input_ports[0].ticket],
            name="u",
        )

    def _ode(self, time, state, *inputs, **params):
        y = npa.atleast_1d(inputs[0])
        x_hat = state.continuous_state
        return npa.dot(self.A_obs, x_hat) + npa.dot(self.B_obs, y)

    def _compute_u(self, time, state, *inputs, **params):
        x_hat = state.continuous_state
        return -npa.dot(self.K, x_hat)
