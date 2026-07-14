# SPDX-License-Identifier: MIT

"""Data-driven operator identification: DMD, DMDc and ERA (T-146).

This module implements three closely related snapshot-based system-identification
methods that build reduced *linear* operators directly from data:

- **Exact DMD** (Dynamic Mode Decomposition) — Tu, Rowley, Luchtenburg, Brunton &
  Kutz, "On Dynamic Mode Decomposition: Theory and Applications", J. Comput. Dyn.
  1(2), 2014. Fits ``x[k+1] ≈ A x[k]`` from a snapshot pair and returns the
  low-rank spectral decomposition (modes / eigenvalues / amplitudes).
- **DMDc** (DMD with control) — Proctor, Brunton & Kutz, "Dynamic Mode
  Decomposition with Control", SIAM J. Appl. Dyn. Syst. 15(1), 2016. Fits
  ``x[k+1] ≈ A x[k] + B u[k]`` for known- and unknown-``B`` cases.
- **ERA** (Eigensystem Realization Algorithm) — Juang & Pappa, "An Eigensystem
  Realization Algorithm for Modal Parameter Identification and Model Reduction",
  J. Guidance Control Dyn. 8(5), 1985. Builds a minimal state-space realization
  ``(A, B, C, D)`` from impulse-response Markov parameters; the data-driven
  bridge to balanced truncation.

Fitting runs on the host with NumPy. The :class:`DMDForecaster` block wraps a
fitted operator as a jax-traceable discrete-time predictor that simulates inside
:func:`jaxonomy.simulate`.
"""

from dataclasses import dataclass
from typing import Any

import numpy as np
import jax.numpy as jnp

from jaxonomy.framework import LeafSystem, parameters, DependencyTicket
from jaxonomy.backend import numpy_api as npa

__all__ = [
    "DMDResult",
    "DMDcResult",
    "ERAResult",
    "dmd",
    "dmdc",
    "era",
    "DMDForecaster",
]


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------
@dataclass
class DMDResult:
    """Exact-DMD spectral decomposition (Tu et al. 2014).

    Attributes:
        modes: DMD modes ``Φ`` (columns), shape ``(n, r)``, generally complex.
        eigenvalues: Discrete-time DMD eigenvalues ``λ``, shape ``(r,)``. The
            growth/decay and oscillation of the identified linear dynamics; a
            mode is stable iff ``|λ| < 1``.
        amplitudes: Mode amplitudes ``b`` fitting the first snapshot, shape ``(r,)``.
        A_tilde: Reduced ``r×r`` operator in the POD-projected coordinates.
    """

    modes: Any
    eigenvalues: Any
    amplitudes: Any
    A_tilde: Any


@dataclass
class DMDcResult:
    """DMD-with-control operators (Proctor, Brunton & Kutz 2016).

    Attributes:
        A: Full ``n×n`` state operator.
        B: Full ``n×m`` input operator.
        A_tilde: Reduced ``r×r`` state operator (POD-projected).
        B_tilde: Reduced ``r×m`` input operator.
        basis: POD basis ``Û`` (columns), shape ``(n, r)``, mapping reduced ↔ full.
        eigenvalues: Eigenvalues of ``A_tilde``, shape ``(r,)``.
    """

    A: Any
    B: Any
    A_tilde: Any
    B_tilde: Any
    basis: Any
    eigenvalues: Any


@dataclass
class ERAResult:
    """Minimal state-space realization from Markov parameters (Juang & Pappa 1985).

    Attributes:
        A: Realized ``r×r`` state matrix.
        B: Realized ``r×n_inputs`` input matrix.
        C: Realized ``n_outputs×r`` output matrix.
        D: Feedthrough ``n_outputs×n_inputs`` (the zeroth Markov parameter).
        singular_values: Hankel singular values (from the block-Hankel SVD).
    """

    A: Any
    B: Any
    C: Any
    D: Any
    singular_values: Any


# ---------------------------------------------------------------------------
# Exact DMD
# ---------------------------------------------------------------------------
def _svd_truncate(M, rank):
    U, s, Vh = np.linalg.svd(M, full_matrices=False)
    if rank is not None:
        rank = min(int(rank), s.shape[0])
        U, s, Vh = U[:, :rank], s[:rank], Vh[:rank, :]
    return U, s, Vh


def dmd(X, Xp=None, rank=None):
    """Exact Dynamic Mode Decomposition (Tu et al. 2014).

    Fits the best-fit linear operator ``x[k+1] ≈ A x[k]`` and returns its
    low-rank spectral decomposition.

    Args:
        X: Snapshot matrix, columns are states. If ``Xp is None``, the shifted
            pair is formed internally as ``X[:, :-1]`` / ``X[:, 1:]``.
        Xp: Optional one-step-advanced snapshots aligned column-wise with ``X``.
        rank: Optional SVD truncation rank ``r`` (defaults to full rank).

    Returns:
        :class:`DMDResult` with ``modes``, ``eigenvalues``, ``amplitudes`` and
        the reduced ``A_tilde``.
    """
    X = np.asarray(X)
    if Xp is None:
        X1, X2 = X[:, :-1], X[:, 1:]
    else:
        X1, X2 = X, np.asarray(Xp)

    U, s, Vh = _svd_truncate(X1, rank)
    V = Vh.conj().T
    Sinv = np.diag(1.0 / s)

    # Reduced operator: A_tilde = Uᵀ X2 V Σ⁻¹  (POD-projected dynamics).
    A_tilde = U.conj().T @ X2 @ V @ Sinv

    eigenvalues, W = np.linalg.eig(A_tilde)
    # Exact-DMD modes: Φ = X2 V Σ⁻¹ W  (Tu et al. 2014, Eq. 2.5).
    modes = X2 @ V @ Sinv @ W

    # Amplitudes fitting the first snapshot: Φ b = x0.
    amplitudes = np.linalg.lstsq(modes, X1[:, 0], rcond=None)[0]

    return DMDResult(
        modes=modes,
        eigenvalues=eigenvalues,
        amplitudes=amplitudes,
        A_tilde=A_tilde,
    )


# ---------------------------------------------------------------------------
# DMD with control
# ---------------------------------------------------------------------------
def dmdc(X, Xp, U, rank=None, B_known=None):
    """Dynamic Mode Decomposition with control (Proctor, Brunton & Kutz 2016).

    Fits ``x[k+1] ≈ A x[k] + B u[k]`` from snapshot pairs and control inputs.

    Two cases are handled:

    - **Unknown ``B``** (default): regress on the augmented snapshot
      ``Ω = [X; U]`` so ``[A  B] = Xp Ω⁺``.
    - **Known ``B``** (pass ``B_known``): subtract the known control effect first,
      ``A = (Xp − B U) X⁺``.

    Args:
        X: State snapshots ``x[k]``, shape ``(n, k)``.
        Xp: Advanced snapshots ``x[k+1]``, shape ``(n, k)``.
        U: Control inputs ``u[k]``, shape ``(m, k)``.
        rank: Optional POD rank ``r`` for the reduced operators (defaults full).
        B_known: Optional known input matrix ``(n, m)`` for the known-``B`` case.

    Returns:
        :class:`DMDcResult` with full ``A, B`` and reduced ``A_tilde, B_tilde``.
    """
    X = np.asarray(X)
    Xp = np.asarray(Xp)
    U = np.atleast_2d(np.asarray(U))
    if U.shape[1] != X.shape[1]:
        U = U.T  # accept (k, m) as well
    n = X.shape[0]

    if B_known is not None:
        B = np.asarray(B_known)
        if B.ndim == 1:
            B = B.reshape(n, -1)
        A = (Xp - B @ U) @ np.linalg.pinv(X)
    else:
        Omega = np.vstack([X, U])
        G = Xp @ np.linalg.pinv(Omega)
        A, B = G[:, :n], G[:, n:]

    # Reduced operators via the leading POD modes of the advanced snapshots.
    Uhat, _, _ = _svd_truncate(Xp, rank)
    A_tilde = Uhat.conj().T @ A @ Uhat
    B_tilde = Uhat.conj().T @ B
    eigenvalues = np.linalg.eigvals(A_tilde)

    return DMDcResult(
        A=A,
        B=B,
        A_tilde=A_tilde,
        B_tilde=B_tilde,
        basis=Uhat,
        eigenvalues=eigenvalues,
    )


# ---------------------------------------------------------------------------
# Eigensystem Realization Algorithm
# ---------------------------------------------------------------------------
def era(markov, n_inputs, n_outputs, num_rows=None, num_cols=None, rank=None):
    """Eigensystem Realization Algorithm (Juang & Pappa 1985).

    Builds a minimal discrete-time state-space realization ``(A, B, C, D)`` from a
    sequence of impulse-response Markov parameters
    ``Y_0 = D``, ``Y_1 = C B``, ``Y_2 = C A B`` ...

    Args:
        markov: Markov parameters. Either an array of shape
            ``(L+1, n_outputs, n_inputs)`` or a length-``L+1`` sequence of such
            blocks; SISO impulse responses may be passed as a 1-D array.
        n_inputs: Number of inputs ``m``.
        n_outputs: Number of outputs ``p``.
        num_rows: Block rows ``α`` of the Hankel matrix (default ~half the data).
        num_cols: Block cols ``β`` of the Hankel matrix (default ~half the data).
        rank: Optional model order ``r`` (SVD truncation of the Hankel matrix).

    Returns:
        :class:`ERAResult` with the realized ``(A, B, C, D)`` and Hankel
        singular values.
    """
    Y = np.asarray(markov, dtype=float)
    L = Y.shape[0] - 1  # number of pulse-response blocks after D
    Y = Y.reshape(L + 1, n_outputs, n_inputs)

    D = Y[0]
    H = Y[1:]  # pulse response Y_1 .. Y_L (used to build the Hankel matrix)

    if num_rows is None:
        num_rows = L // 2
    if num_cols is None:
        num_cols = L - num_rows
    alpha, beta = int(num_rows), int(num_cols)
    if alpha + beta > L:
        raise ValueError(
            f"era: need num_rows + num_cols <= {L} Markov blocks, "
            f"got {alpha} + {beta}."
        )

    def _hankel(shift):
        blocks = [
            [H[i + j + shift] for j in range(beta)] for i in range(alpha)
        ]
        return np.block(blocks)

    H0 = _hankel(0)  # blocks Y_{i+j+1}
    H1 = _hankel(1)  # blocks Y_{i+j+2}

    Ur, s, Vh = np.linalg.svd(H0, full_matrices=False)
    if rank is not None:
        r = min(int(rank), s.shape[0])
    else:
        tol = max(H0.shape) * np.finfo(float).eps * (s[0] if s.size else 0.0)
        r = int(np.sum(s > tol))
        r = max(r, 1)
    Ur, s, Vh = Ur[:, :r], s[:r], Vh[:r, :]
    Vr = Vh.conj().T

    s_sqrt = np.sqrt(s)
    s_inv_sqrt = 1.0 / s_sqrt
    Obs = Ur * s_sqrt          # observability factor  R Σ^{1/2}
    Cc = (s_sqrt[:, None]) * Vr.conj().T  # controllability factor Σ^{1/2} S*

    A = (s_inv_sqrt[:, None]) * (Ur.conj().T @ H1 @ Vr) * (s_inv_sqrt[None, :])
    B = Cc[:, :n_inputs]
    C = Obs[:n_outputs, :]

    return ERAResult(A=A, B=B, C=C, D=D, singular_values=s)


# ---------------------------------------------------------------------------
# Discrete-time predictor block
# ---------------------------------------------------------------------------
class DMDForecaster(LeafSystem):
    """Discrete-time predictor for a fitted (reduced) linear operator.

    Propagates ``x[k+1] = A x[k] (+ B u[k])`` and outputs ``y[k] = C x[k]``
    (``C`` defaults to the identity, so the state itself is the output). This is
    the online counterpart of :func:`dmd` / :func:`dmdc`: fit ``A`` (and ``B``)
    from snapshots offline, then drop the operator into a Jaxonomy diagram as a
    jax-traceable discrete block that runs inside :func:`jaxonomy.simulate`.

    An input port (and use of ``B``) is created only when ``B`` is provided.

    Input ports:
        (0) u[k]: control input, present iff ``B`` is given.

    Output ports:
        (0) y[k] = C x[k].

    Parameters:
        A: State operator ``(n, n)`` — a ``dynamic`` parameter.
        B: Optional input operator ``(n, m)`` — a ``dynamic`` parameter when given.
        C: Optional output operator ``(p, n)`` — a ``dynamic`` parameter when
            given; defaults to identity.
        dt: Sampling period of the discrete update.
        initial_state: Initial state ``x[0]`` of size ``n`` (default: zeros).
    """

    @parameters(dynamic=["A", "B", "C"], static=["dt", "initial_state"])
    def __init__(self, A, B=None, C=None, dt=1.0, initial_state=None, name=None,
                 **kwargs):
        super().__init__(name=name, **kwargs)

        A = np.asarray(A, dtype=float)
        if A.ndim == 0:
            A = A.reshape(1, 1)
        self.n = A.shape[0]
        self.has_input = B is not None
        if C is not None:
            C = np.asarray(C, dtype=float)
            if C.ndim == 1:
                C = C.reshape(1, -1)
            self.p = C.shape[0]
        else:
            self.p = self.n

        self.dt = dt
        if initial_state is None:
            initial_state = np.zeros(self.n)
        self._x0 = np.asarray(initial_state, dtype=float).reshape(-1)

        if self.has_input:
            self.declare_input_port(name="u")

        self._periodic_update_idx = self.declare_periodic_update()
        self._output_port_idx = self.declare_output_port(name="out_0")

    def initialize(self, A, B=None, C=None, dt=1.0, initial_state=None, **kwargs):
        if initial_state is None:
            x0 = npa.array(self._x0)
        else:
            x0 = npa.reshape(npa.array(initial_state, dtype=npa.float64), (-1,))

        self.declare_discrete_state(default_value=x0)
        self.configure_periodic_update(
            self._periodic_update_idx, self._update, period=self.dt, offset=0.0
        )
        self.configure_output_port(
            self._output_port_idx,
            self._output,
            period=self.dt,
            offset=0.0,
            default_value=npa.zeros(self.p) if self.p > 1 else 0.0,
            requires_inputs=False,
            prerequisites_of_calc=[DependencyTicket.xd],
        )

    def _update(self, _time, state, *inputs, **params):
        x = state.discrete_state
        x_next = params["A"] @ x
        if self.has_input:
            u = jnp.atleast_1d(inputs[0])
            x_next = x_next + params["B"] @ u
        return x_next

    def _output(self, _time, state, *_inputs, **params):
        x = state.discrete_state
        C = params.get("C")
        y = x if C is None else C @ x
        if self.p == 1:
            y = jnp.atleast_1d(y)[0]
        return y
