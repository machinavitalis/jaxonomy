# SPDX-License-Identifier: MIT

"""Koopman operator approximation via extended DMD (eDMD) (T-147).

The Koopman operator advances *observables* of a nonlinear system linearly.
Lifting the state ``x`` through a dictionary of observables ``g(x)`` and fitting a
linear operator on the lifted snapshots yields a finite-dimensional approximation
of the Koopman operator — Williams, Kevrekidis & Rowley, "A Data-Driven
Approximation of the Koopman Operator: Extending Dynamic Mode Decomposition",
J. Nonlinear Sci. 25(6), 2015.

Because the lifted model is **linear** (``z[k+1] = K z[k] (+ B u[k])``), it plugs
directly into the existing linear control stack — Koopman-based linear MPC/LQR
just designs against ``(K, B)`` in the lifted space and de-lifts the result. This
is the Koopman→linear-MPC framing: a nonlinear plant is controlled with linear
tools by working in lifted coordinates.

Convention: every built-in dictionary places the identity observables (the raw
state ``x``) first in the lifted vector, so the physical state is always
recoverable. :func:`edmd` returns the de-lift matrix ``C`` mapping lifted → physical
(via least squares); when identity observables are present this ``C`` reduces to a
selection of those rows.

Fitting runs on the host; the :class:`KoopmanPredictor` block is jax-traceable and
simulates inside :func:`jaxonomy.simulate`.
"""

import itertools
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import jax.numpy as jnp

from jaxonomy.framework import LeafSystem, parameters, DependencyTicket
from jaxonomy.backend import numpy_api as npa

__all__ = [
    "identity_dictionary",
    "polynomial_dictionary",
    "rbf_dictionary",
    "EDMDResult",
    "edmd",
    "KoopmanPredictor",
]


# ---------------------------------------------------------------------------
# Observable dictionaries.  Each returns a callable g(x) -> lifted vector whose
# first n entries are the identity observables (the raw state).
# ---------------------------------------------------------------------------
def identity_dictionary() -> Callable:
    """Trivial dictionary ``g(x) = x``.

    eDMD with this dictionary reduces to plain (linear) DMD — a useful baseline.
    """

    def g(x):
        return jnp.atleast_1d(x)

    return g


def polynomial_dictionary(degree: int, include_constant: bool = True) -> Callable:
    """Monomial dictionary up to ``degree``.

    Layout: ``[x_1..x_n, (1), (degree-2..degree monomials)]`` — identity first so
    the state is recoverable. The constant term is included by default (it makes
    affine dynamics representable in the lifted space).
    """

    def g(x):
        x = jnp.atleast_1d(x)
        n = x.shape[0]
        terms = [x]  # identity observables first
        if include_constant:
            terms.append(jnp.ones((1,), dtype=x.dtype))
        for d in range(2, int(degree) + 1):
            for combo in itertools.combinations_with_replacement(range(n), d):
                term = jnp.prod(jnp.stack([x[i] for i in combo]))
                terms.append(jnp.reshape(term, (1,)))
        return jnp.concatenate(terms)

    return g


def rbf_dictionary(centers, epsilon: float = 1.0) -> Callable:
    """Gaussian radial-basis dictionary.

    Lifts to ``[x, exp(-epsilon ||x - c_j||²) for each center c_j]`` — identity
    observables first, followed by one RBF feature per center.

    Args:
        centers: Array ``(n_centers, n)`` of RBF centers.
        epsilon: Shape parameter of the Gaussian kernel.
    """
    centers = np.asarray(centers, dtype=float)
    if centers.ndim == 1:
        centers = centers.reshape(-1, 1)
    centers_j = jnp.asarray(centers)
    eps = float(epsilon)

    def g(x):
        x = jnp.atleast_1d(x)
        diffs = centers_j - x[None, :]
        rbf = jnp.exp(-eps * jnp.sum(diffs * diffs, axis=1))
        return jnp.concatenate([x, rbf])

    return g


# ---------------------------------------------------------------------------
# eDMD fitting
# ---------------------------------------------------------------------------
@dataclass
class EDMDResult:
    """Fitted Koopman model (Williams, Kevrekidis & Rowley 2015).

    Attributes:
        K: Koopman operator on lifted observables, shape ``(L, L)``.
        B: Input operator on the lifted space, shape ``(L, m)`` (eDMDc) or ``None``.
        C: De-lift matrix mapping lifted → physical state, shape ``(n, L)``.
        dictionary: The observable dictionary ``g`` used for lifting.
    """

    K: Any
    B: Any
    C: Any
    dictionary: Callable


def _lift_columns(X, dictionary):
    """Apply the dictionary to each column of ``X`` -> lifted matrix ``(L, k)``."""
    X = np.asarray(X, dtype=float)
    cols = [np.asarray(dictionary(jnp.asarray(X[:, j]))) for j in range(X.shape[1])]
    return np.column_stack(cols)


def edmd(X, Xp, dictionary, U=None):
    """Extended DMD — approximate the Koopman operator on lifted snapshots.

    Lifts the snapshot pair through ``dictionary`` and least-squares fits the
    lifted linear dynamics ``z[k+1] ≈ K z[k] (+ B u[k])``.

    Args:
        X: State snapshots ``x[k]``, shape ``(n, k)``.
        Xp: Advanced snapshots ``x[k+1]``, shape ``(n, k)``.
        dictionary: Callable ``g(x) -> lifted vector`` (identity observables first).
        U: Optional control inputs ``(m, k)`` for eDMDc.

    Returns:
        :class:`EDMDResult` with the Koopman operator ``K``, the input operator
        ``B`` (or ``None``), and the de-lift matrix ``C``.
    """
    X = np.asarray(X, dtype=float)
    Xp = np.asarray(Xp, dtype=float)

    Z1 = _lift_columns(X, dictionary)
    Z2 = _lift_columns(Xp, dictionary)

    if U is None:
        # K solves Z2 ≈ K Z1  (least squares).
        K = Z2 @ np.linalg.pinv(Z1)
        B = None
    else:
        U = np.atleast_2d(np.asarray(U, dtype=float))
        if U.shape[1] != Z1.shape[1]:
            U = U.T
        n_state = Z1.shape[0]
        Omega = np.vstack([Z1, U])
        G = Z2 @ np.linalg.pinv(Omega)
        K, B = G[:, :n_state], G[:, n_state:]

    # De-lift matrix C: physical state ≈ C · lifted.  Solved by least squares,
    # so it reduces to a row-selection when the identity observables are present.
    C = X @ np.linalg.pinv(Z1)

    return EDMDResult(K=K, B=B, C=C, dictionary=dictionary)


# ---------------------------------------------------------------------------
# Discrete-time Koopman predictor block
# ---------------------------------------------------------------------------
class KoopmanPredictor(LeafSystem):
    """Discrete-time Koopman predictor for a nonlinear system.

    Each step: lift the stored physical state ``z = g(x)``, advance the lifted
    *linear* dynamics ``z[k+1] = K z[k] (+ B u[k])``, then de-lift
    ``x[k+1] = C z[k+1]``. The physical state is the block output.

    Because the lifted model is *linear* in ``z``, ``(K, B)`` is a discrete
    linear model you can use for linear MPC / LQR-style control — design in
    lifted coordinates and de-lift with ``C`` (the Koopman→linear-MPC framing;
    Korda & Mezić 2018). Note a lifted model generally has no state that both
    satisfies ``z = g(x)`` and a hard terminal-equality constraint, so a
    *terminal-cost* MPC is the right fit; the current
    :class:`~jaxonomy.library.mpc.LinearDiscreteTimeMPC` block (hard terminal
    equality, continuous-time model input) does not compose directly — see the
    ``rom_dmdc_koopman_mpc`` example, which uses a compact terminal-cost MPC.

    An input port (and use of ``B``) is created only when ``B`` is provided.

    Input ports:
        (0) u[k]: control input, present iff ``B`` is given.

    Output ports:
        (0) x[k]: de-lifted physical state.

    Parameters:
        K: Koopman operator ``(L, L)`` — a ``dynamic`` parameter.
        C: De-lift matrix ``(n, L)`` — a ``dynamic`` parameter.
        dictionary: Observable dictionary ``g`` used for lifting (identity first).
        B: Optional lifted input operator ``(L, m)`` — a ``dynamic`` parameter when given.
        dt: Sampling period of the discrete update.
        initial_state: Initial physical state ``x[0]`` of size ``n``.
    """

    @parameters(dynamic=["K", "C", "B"], static=["dt", "initial_state"])
    def __init__(self, K, C, dictionary, B=None, dt=1.0, initial_state=None,
                 name=None, **kwargs):
        super().__init__(name=name, **kwargs)

        C = np.asarray(C, dtype=float)
        if C.ndim == 1:
            C = C.reshape(1, -1)
        self.n = C.shape[0]
        self.dictionary = dictionary
        self.has_input = B is not None
        self.dt = dt

        if initial_state is None:
            initial_state = np.zeros(self.n)
        self._x0 = np.asarray(initial_state, dtype=float).reshape(-1)

        if self.has_input:
            self.declare_input_port(name="u")

        self._periodic_update_idx = self.declare_periodic_update()
        self._output_port_idx = self.declare_output_port(name="out_0")

    def initialize(self, K, C, dictionary=None, B=None, dt=1.0, initial_state=None,
                   **kwargs):
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
            default_value=npa.zeros(self.n) if self.n > 1 else 0.0,
            requires_inputs=False,
            prerequisites_of_calc=[DependencyTicket.xd],
        )

    def _update(self, _time, state, *inputs, **params):
        x = state.discrete_state
        z = self.dictionary(x)
        z_next = params["K"] @ z
        if self.has_input:
            u = jnp.atleast_1d(inputs[0])
            z_next = z_next + params["B"] @ u
        x_next = params["C"] @ z_next
        return x_next

    def _output(self, _time, state, *_inputs, **_params):
        x = state.discrete_state
        if self.n == 1:
            return jnp.atleast_1d(x)[0]
        return x
