# SPDX-License-Identifier: MIT

"""POD-Galerkin projection ROMs with DEIM hyper-reduction (T-145).

Given full-order snapshots, :func:`pod_basis` computes a proper-orthogonal-
decomposition (POD) reduced basis, and :func:`galerkin_reduce` projects a
full-order right-hand side onto that basis to yield a jaxonomy ``LeafSystem``
whose continuous state is the ``r`` reduced coordinates. For systems with a
non-affine nonlinearity, :func:`deim` selects interpolation points and
:func:`deim_galerkin_reduce` builds a *hyper-reduced* model whose per-step cost
is independent of the full dimension ``n``.

References:
    Sirovich (1987), "Turbulence and the dynamics of coherent structures",
        Q. Appl. Math. 45:561-590 (snapshot POD).
    Berkooz, Holmes & Lumley (1993), Annu. Rev. Fluid Mech. 25:539-575 (POD).
    Chaturantabut & Sorensen (2010), "Nonlinear model reduction via discrete
        empirical interpolation", SIAM J. Sci. Comput. 32(5):2737-2764 (DEIM).
    Carlberg, Farhat, Cortial & Amsallem (2013), "The GNAT method...", J.
        Comput. Phys. 242:623-647 (least-squares Petrov-Galerkin / LSPG).
"""

from __future__ import annotations

from typing import Callable, Optional, Tuple

import numpy as np
import jax.numpy as jnp

from ...framework import LeafSystem, parameters, DependencyTicket

__all__ = [
    "pod_basis",
    "galerkin_reduce",
    "deim",
    "deim_galerkin_reduce",
]


# ---------------------------------------------------------------------------
# Basis construction
# ---------------------------------------------------------------------------

def _select_rank(
    sigma: np.ndarray,
    rank: Optional[int],
    energy: Optional[float],
) -> int:
    """Resolve the truncation rank from an explicit ``rank`` or a cumulative
    squared-singular-value ``energy`` threshold (e.g. 0.99)."""
    n = int(sigma.shape[0])
    if rank is not None:
        return int(max(1, min(int(rank), n)))
    if energy is not None:
        total = float(np.sum(sigma**2))
        if total == 0.0:
            return 1
        cum = np.cumsum(sigma**2) / total
        # first index whose cumulative energy reaches the threshold
        r = int(np.searchsorted(cum, float(energy)) + 1)
        return int(max(1, min(r, n)))
    # Default: keep all numerically nonzero modes.
    tol = np.finfo(float).eps * n * (sigma[0] if n else 0.0)
    r = int(np.sum(sigma > tol))
    return int(max(1, r))


def pod_basis(
    X,
    rank: Optional[int] = None,
    energy: Optional[float] = None,
) -> Tuple[jnp.ndarray, jnp.ndarray, int]:
    """Proper-orthogonal-decomposition basis of a snapshot matrix.

    Computes the (host-side) thin SVD ``X = U Σ Vᵀ`` and truncates to ``r``
    left singular vectors, which are the energetically optimal orthonormal
    modes (Sirovich 1987).

    Args:
        X: Snapshot matrix, shape ``(n_features, n_samples)``.
        rank: Explicit number of modes to keep.
        energy: Cumulative-energy threshold in ``(0, 1]`` (used when ``rank``
            is ``None``), e.g. ``0.99`` keeps 99% of the snapshot energy.

    Returns:
        ``(Phi, sigma, r)`` where ``Phi`` has shape ``(n_features, r)`` with
        orthonormal columns, ``sigma`` is the full singular-value vector, and
        ``r`` is the retained rank.
    """
    X_np = np.asarray(X, dtype=float)
    U, sigma, _ = np.linalg.svd(X_np, full_matrices=False)
    r = _select_rank(sigma, rank, energy)
    Phi = U[:, :r]
    return jnp.asarray(Phi), jnp.asarray(sigma), int(r)


# ---------------------------------------------------------------------------
# Galerkin / Petrov-Galerkin projection ROM
# ---------------------------------------------------------------------------

class _ProjectionROM(LeafSystem):
    """Reduced-order ``LeafSystem`` with ``r`` continuous reduced coordinates.

    The reduced ODE is ``ẋ_r = W · rhs_fn(t, Φ x_r + x_ref, u)`` where ``W`` is
    ``Φᵀ`` for Galerkin or ``(Ψᵀ Φ)⁻¹ Ψᵀ`` for Petrov-Galerkin/LSPG
    (Carlberg et al. 2013). The output port reconstructs the full state
    ``Φ x_r + x_ref`` (optionally mapped through ``output_fn``).
    """

    @parameters(dynamic=["Phi", "W", "x_ref"])
    def __init__(
        self,
        Phi,
        W,
        x_ref,
        rhs_fn,
        output_fn=None,
        input_size=0,
        name=None,
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)
        self._rhs_fn = rhs_fn
        self._output_fn = output_fn
        self.input_size = int(input_size)
        self.r = int(np.asarray(Phi).shape[1])

        if self.input_size > 0:
            self.declare_input_port(name="u")
        self._output_port_idx = self.declare_output_port(None, name="x_full")
        self._ode_idx = self.declare_continuous_state()

    def initialize(self, Phi, W, x_ref, **kwargs):
        self.configure_continuous_state(
            self._ode_idx,
            ode=self._ode,
            default_value=jnp.zeros(self.r),
        )
        self.configure_output_port(
            self._output_port_idx,
            self._output,
            prerequisites_of_calc=[DependencyTicket.xc],
            requires_inputs=False,
        )

    def _ode(self, time, state, *inputs, **params):
        xr = state.continuous_state
        x_full = params["Phi"] @ xr + params["x_ref"]
        if self.input_size > 0:
            dx_full = self._rhs_fn(time, x_full, inputs[0])
        else:
            dx_full = self._rhs_fn(time, x_full)
        return params["W"] @ dx_full

    def _output(self, time, state, *inputs, **params):
        x_full = params["Phi"] @ state.continuous_state + params["x_ref"]
        if self._output_fn is not None:
            return self._output_fn(x_full)
        return x_full


def galerkin_reduce(
    rhs_fn: Callable,
    basis,
    x_ref=None,
    output_fn: Optional[Callable] = None,
    input_size: int = 0,
    test_basis=None,
    name: Optional[str] = None,
) -> _ProjectionROM:
    """Project a full-order RHS onto a reduced basis (POD-Galerkin / LSPG).

    Args:
        rhs_fn: Full-order dynamics. Called as ``rhs_fn(t, x_full, u)`` when
            ``input_size > 0`` else ``rhs_fn(t, x_full)``; must be
            jax-traceable and return ``dx_full`` of shape ``(n_features,)``.
        basis: Trial basis ``Φ``, shape ``(n_features, r)``.
        x_ref: Reference/offset state added on reconstruction (default zeros).
        output_fn: Optional map applied to the reconstructed full state for the
            output port.
        input_size: Width of the single input port; ``0`` for an autonomous
            block (no input port).
        test_basis: Optional test basis ``Ψ`` (shape ``(n_features, r)``) for a
            Petrov-Galerkin/LSPG projection ``W = (Ψᵀ Φ)⁻¹ Ψᵀ``. When ``None``,
            Galerkin ``W = Φᵀ``.
        name: Optional block name.

    Returns:
        A jaxonomy ``LeafSystem`` with ``r`` reduced continuous states.
    """
    Phi = np.asarray(basis, dtype=float)
    n, r = Phi.shape
    x_ref_arr = np.zeros(n) if x_ref is None else np.asarray(x_ref, dtype=float)

    if test_basis is None:
        W = Phi.T
    else:
        Psi = np.asarray(test_basis, dtype=float)
        W = np.linalg.solve(Psi.T @ Phi, Psi.T)

    return _ProjectionROM(
        Phi=jnp.asarray(Phi),
        W=jnp.asarray(W),
        x_ref=jnp.asarray(x_ref_arr),
        rhs_fn=rhs_fn,
        output_fn=output_fn,
        input_size=input_size,
        name=name,
    )


# ---------------------------------------------------------------------------
# DEIM (Discrete Empirical Interpolation Method)
# ---------------------------------------------------------------------------

def deim(
    nonlinear_snapshots,
    rank: Optional[int] = None,
    energy: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Greedy DEIM point selection (Chaturantabut & Sorensen 2010).

    Takes an SVD basis ``U`` of the nonlinear-term snapshots and greedily
    selects ``m`` interpolation indices, then forms the oblique DEIM projector
    ``U (Pᵀ U)⁻¹`` (``P`` selects the chosen rows).

    Args:
        nonlinear_snapshots: Snapshots of the nonlinear term, shape
            ``(n_features, n_samples)``.
        rank: Number of DEIM modes/points ``m`` to keep.
        energy: Cumulative-energy threshold used when ``rank`` is ``None``.

    Returns:
        ``(indices, projector)`` — ``indices`` are ``m`` distinct row indices
        (``np.ndarray`` of int), ``projector`` has shape ``(n_features, m)``.
    """
    F = np.asarray(nonlinear_snapshots, dtype=float)
    U_full, sigma, _ = np.linalg.svd(F, full_matrices=False)
    m = _select_rank(sigma, rank, energy)
    U = U_full[:, :m]

    indices = np.empty(m, dtype=int)
    indices[0] = int(np.argmax(np.abs(U[:, 0])))
    for j in range(1, m):
        Uj = U[:, :j]                       # (n, j)
        P = indices[:j]                     # selected rows
        # Solve (Pᵀ Uj) c = Pᵀ U[:, j] for interpolation coefficients.
        c = np.linalg.solve(Uj[P, :], U[P, j])
        residual = U[:, j] - Uj @ c
        indices[j] = int(np.argmax(np.abs(residual)))

    projector = U @ np.linalg.inv(U[indices, :])   # U (Pᵀ U)⁻¹, shape (n, m)
    return indices, projector


class _DEIMGalerkinROM(LeafSystem):
    """Hyper-reduced POD-Galerkin ROM.

    The reduced ODE evaluates the nonlinearity ONLY at the ``m`` DEIM points::

        ẋ_r = Ar·x_r + b_off + Br·u + DEIM_reduced · g(Φ_P x_r + x_ref_P)

    where ``Ar = Φᵀ A Φ``, ``Br = Φᵀ B`` and ``DEIM_reduced = Φᵀ U (Pᵀ U)⁻¹``
    are precomputed offline, and ``Φ_P = Φ[P]`` gathers the DEIM rows. Every
    per-step operation is ``O(r² + m·r + m)`` — independent of the full ``n``.
    """

    @parameters(
        dynamic=["Ar", "b_off", "Br", "DEIM_reduced", "Phi_P", "x_ref_P",
                 "Phi", "x_ref"]
    )
    def __init__(
        self,
        Ar,
        b_off,
        Br,
        DEIM_reduced,
        Phi_P,
        x_ref_P,
        Phi,
        x_ref,
        nonlinear_fn,
        input_size=0,
        name=None,
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)
        self._nonlinear_fn = nonlinear_fn
        self.input_size = int(input_size)
        self.r = int(np.asarray(Phi).shape[1])

        if self.input_size > 0:
            self.declare_input_port(name="u")
        self._output_port_idx = self.declare_output_port(None, name="x_full")
        self._ode_idx = self.declare_continuous_state()

    def initialize(self, **kwargs):
        self.configure_continuous_state(
            self._ode_idx,
            ode=self._ode,
            default_value=jnp.zeros(self.r),
        )
        self.configure_output_port(
            self._output_port_idx,
            self._output,
            prerequisites_of_calc=[DependencyTicket.xc],
            requires_inputs=False,
        )

    def _ode(self, time, state, *inputs, **params):
        xr = state.continuous_state
        # State at the DEIM points only — never reconstructs the full n-vector.
        x_pts = params["Phi_P"] @ xr + params["x_ref_P"]
        g = self._nonlinear_fn(x_pts)
        dxr = params["Ar"] @ xr + params["b_off"] + params["DEIM_reduced"] @ g
        if self.input_size > 0:
            dxr = dxr + params["Br"] @ inputs[0]
        return dxr

    def _output(self, time, state, *inputs, **params):
        return params["Phi"] @ state.continuous_state + params["x_ref"]


def deim_galerkin_reduce(
    linear_rhs_fn: Callable,
    nonlinear_fn: Callable,
    basis,
    deim_result: Tuple[np.ndarray, np.ndarray],
    x_ref=None,
    input_size: int = 0,
    name: Optional[str] = None,
) -> _DEIMGalerkinROM:
    """Build a DEIM hyper-reduced POD-Galerkin ROM.

    The full-order dynamics are split as ``ẋ = f_lin(t, x, u) + g(x)`` with a
    (affine-)linear part ``f_lin`` and an elementwise nonlinearity ``g``. The
    linear operator is reduced offline to dense ``r×r`` / ``r×m`` operators, and
    the nonlinearity is approximated by DEIM so it is evaluated only at the
    selected points (Chaturantabut & Sorensen 2010).

    Args:
        linear_rhs_fn: Affine-linear part. Called ``linear_rhs_fn(t, x, u)`` if
            ``input_size > 0`` else ``linear_rhs_fn(t, x)``; jax-traceable,
            returns ``(n_features,)``. Probed offline at ``t=0`` to extract its
            reduced operators, so it must be affine in ``(x, u)``.
        nonlinear_fn: Elementwise nonlinearity ``g``; called with a
            ``(m,)`` vector of states at the DEIM points and returns ``(m,)``.
        basis: POD trial basis ``Φ``, shape ``(n_features, r)``.
        deim_result: The ``(indices, projector)`` pair from :func:`deim`.
        x_ref: Reference/offset state (default zeros).
        input_size: Width of the single input port; ``0`` for autonomous.
        name: Optional block name.

    Returns:
        A jaxonomy ``LeafSystem`` with ``r`` reduced continuous states whose
        per-step cost is independent of the full dimension ``n``.
    """
    Phi = np.asarray(basis, dtype=float)
    n, r = Phi.shape
    x_ref_arr = np.zeros(n) if x_ref is None else np.asarray(x_ref, dtype=float)

    indices, projector = deim_result
    indices = np.asarray(indices, dtype=int)
    projector = np.asarray(projector, dtype=float)   # (n, m)

    # Offline reduction of the affine-linear operator by probing f_lin.
    zeros_n = jnp.zeros(n)
    if input_size > 0:
        zeros_u = jnp.zeros(input_size)
        const = np.asarray(linear_rhs_fn(0.0, zeros_n, zeros_u), dtype=float)
        # A·Φ_k = f_lin(0, Φ_k, 0) − const  (linearity of f_lin in x)
        AtimesPhi = np.stack(
            [np.asarray(linear_rhs_fn(0.0, jnp.asarray(Phi[:, k]), zeros_u),
                        dtype=float) - const
             for k in range(r)],
            axis=1,
        )  # (n, r)
        # B·e_j = f_lin(0, 0, e_j) − const
        B = np.stack(
            [np.asarray(linear_rhs_fn(0.0, zeros_n,
                                      jnp.asarray(np.eye(input_size)[:, j])),
                        dtype=float) - const
             for j in range(input_size)],
            axis=1,
        )  # (n, input_size)
        Br = Phi.T @ B
    else:
        const = np.asarray(linear_rhs_fn(0.0, zeros_n), dtype=float)
        AtimesPhi = np.stack(
            [np.asarray(linear_rhs_fn(0.0, jnp.asarray(Phi[:, k])),
                        dtype=float) - const
             for k in range(r)],
            axis=1,
        )
        Br = np.zeros((r, 0))

    Ar = Phi.T @ AtimesPhi                              # (r, r)
    # Affine offset: Φᵀ (A x_ref + const) = Φᵀ f_lin(0, x_ref, 0).
    if input_size > 0:
        f_at_ref = np.asarray(
            linear_rhs_fn(0.0, jnp.asarray(x_ref_arr), zeros_u), dtype=float)
    else:
        f_at_ref = np.asarray(
            linear_rhs_fn(0.0, jnp.asarray(x_ref_arr)), dtype=float)
    b_off = Phi.T @ f_at_ref                            # (r,)

    DEIM_reduced = Phi.T @ projector                    # (r, m)
    Phi_P = Phi[indices, :]                             # (m, r)
    x_ref_P = x_ref_arr[indices]                        # (m,)

    return _DEIMGalerkinROM(
        Ar=jnp.asarray(Ar),
        b_off=jnp.asarray(b_off),
        Br=jnp.asarray(Br),
        DEIM_reduced=jnp.asarray(DEIM_reduced),
        Phi_P=jnp.asarray(Phi_P),
        x_ref_P=jnp.asarray(x_ref_P),
        Phi=jnp.asarray(Phi),
        x_ref=jnp.asarray(x_ref_arr),
        nonlinear_fn=nonlinear_fn,
        input_size=input_size,
        name=name,
    )
