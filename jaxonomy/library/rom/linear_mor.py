# SPDX-License-Identifier: MIT

"""Linear model-order reduction (MOR) for :class:`LinearizedSystem` models.

Given a state-space realization ``(A, B, C, D)`` — typically produced by
:func:`jaxonomy.library.linear_system.linearize` — these helpers build a
lower-order realization that approximates the input/output behaviour of the
original.  Every reducer returns a :class:`LinearizedSystem` (preserving
``.dt`` for discrete-time models) so the result chains straight back into the
simulation stack via :meth:`LinearizedSystem.to_lti`.

The numerics run host-side (NumPy / SciPy): reduction is an *analysis-time*
operation on a frozen linearization, never something traced inside a running
simulation, so a plain ``numpy``/``scipy.linalg`` implementation is both
simpler and more robust than a JAX one here.

Methods implemented
-------------------
* Balanced truncation / balanced realization — Moore 1981, "Principal
  component analysis in linear systems", *IEEE TAC* 26(1); square-root
  balancing per Laub, Heath, Paige & Ward 1987, "Computation of system
  balancing transformations", *IEEE TAC* 32(2).
* Minimal realization via controllable/observable subspace projection
  (Kalman decomposition).
* Modal truncation and singular-perturbation residualization —
  Kokotović, O'Malley & Sannuti 1976 / Kokotović, Khalil & O'Reilly 1986,
  *Singular Perturbation Methods in Control*.

T-144
"""

from collections import namedtuple

import numpy as np
import scipy.linalg as sla

from jaxonomy.library.linear_system import LinearizedSystem

__all__ = [
    "controllability_gramian",
    "observability_gramian",
    "hankel_singular_values",
    "balanced_realization",
    "balanced_truncation",
    "balred",
    "minimal_realization",
    "minreal",
    "modal_truncation",
    "residualize",
]


# --------------------------------------------------------------------------- #
# small helpers
# --------------------------------------------------------------------------- #
def _np(x):
    """Coerce a (possibly-jax) array-like to a contiguous float64 NumPy array."""
    return np.asarray(x, dtype=float)


def _abcd(sys):
    """Pull ``(A, B, C, D)`` off a :class:`LinearizedSystem` as NumPy arrays."""
    A = _np(sys.A)
    B = _np(sys.B)
    C = _np(sys.C)
    D = _np(sys.D)
    if A.ndim == 0:
        A = A.reshape(1, 1)
    if A.ndim == 1:
        A = A.reshape(1, 1)
    n = A.shape[0]
    B = B.reshape(n, -1) if B.ndim <= 1 else B
    C = C.reshape(-1, n) if C.ndim <= 1 else C
    if D.ndim <= 1:
        D = D.reshape(C.shape[0], B.shape[1])
    return A, B, C, D


def _reduced_operating_point(sys):
    """Operating point for a reduced model.

    The state basis changes under reduction, so the original ``x`` no longer
    applies; the input ``u`` is basis-invariant and is carried through when
    present.
    """
    op = getattr(sys, "operating_point", None) or {}
    return {"u": op["u"]} if "u" in op else {}


def _make_reduced(sys, A, B, C, D):
    """Wrap reduced ``(A, B, C, D)`` in a LinearizedSystem, preserving ``dt``."""
    return LinearizedSystem(
        A=np.asarray(A),
        B=np.asarray(B),
        C=np.asarray(C),
        D=np.asarray(D),
        operating_point=_reduced_operating_point(sys),
        dt=sys.dt,
    )


# --------------------------------------------------------------------------- #
# Gramians
# --------------------------------------------------------------------------- #
def controllability_gramian(A, B, dt=None):
    r"""Controllability Gramian :math:`W_c`.

    Continuous time (``dt is None``) solves the Lyapunov equation

    .. math:: A W_c + W_c A^\mathsf{T} = -B B^\mathsf{T}

    Discrete time (``dt`` given) solves the Stein equation

    .. math:: A W_c A^\mathsf{T} - W_c + B B^\mathsf{T} = 0

    Both require ``A`` stable (continuous: ``Re(eig) < 0``; discrete:
    ``|eig| < 1``) for a positive-semidefinite solution.
    """
    A = _np(A)
    B = _np(B)
    if A.ndim <= 1:
        A = A.reshape(1, 1)
    B = B.reshape(A.shape[0], -1)
    if dt is None:
        return sla.solve_continuous_lyapunov(A, -(B @ B.T))
    return sla.solve_discrete_lyapunov(A, B @ B.T)


def observability_gramian(A, C, dt=None):
    r"""Observability Gramian :math:`W_o`.

    Continuous time (``dt is None``) solves

    .. math:: A^\mathsf{T} W_o + W_o A = -C^\mathsf{T} C

    Discrete time (``dt`` given) solves

    .. math:: A^\mathsf{T} W_o A - W_o + C^\mathsf{T} C = 0
    """
    A = _np(A)
    C = _np(C)
    if A.ndim <= 1:
        A = A.reshape(1, 1)
    C = C.reshape(-1, A.shape[0])
    if dt is None:
        return sla.solve_continuous_lyapunov(A.T, -(C.T @ C))
    return sla.solve_discrete_lyapunov(A.T, C.T @ C)


def hankel_singular_values(sys):
    r"""Hankel singular values, sorted descending.

    :math:`\sigma_i = \sqrt{\lambda_i(W_c W_o)}` for the controllability and
    observability Gramians of ``sys`` (Moore 1981).
    """
    A, B, C, _ = _abcd(sys)
    Wc = controllability_gramian(A, B, dt=sys.dt)
    Wo = observability_gramian(A, C, dt=sys.dt)
    eig = np.linalg.eigvals(Wc @ Wo)
    hsv = np.sqrt(np.clip(np.real(eig), 0.0, None))
    return np.sort(hsv)[::-1]


# --------------------------------------------------------------------------- #
# Balancing (square-root / Laub algorithm)
# --------------------------------------------------------------------------- #
def _psd_sqrt(W):
    """Symmetric square-root factor ``R`` with ``R @ Rᵀ == W`` for a symmetric
    positive-*semi*definite ``W``.

    Uses the symmetric eigendecomposition rather than a Cholesky factor: a
    stiff or non-minimal system has Hankel singular values that fall below
    machine precision, leaving the Gramians numerically semidefinite, on which
    ``scipy.linalg.cholesky`` raises ``LinAlgError``. Any factor with
    ``W = R Rᵀ`` is a valid input to the square-root balancing algorithm — the
    triangularity of a Cholesky factor is not required (Laub et al. 1987).
    """
    W = 0.5 * (W + W.T)
    w, V = np.linalg.eigh(W)
    w = np.clip(w, 0.0, None)
    return V * np.sqrt(w)  # (V √Λ)(V √Λ)ᵀ = V Λ Vᵀ = W


def _balancing_transform(A, B, C, dt):
    """Square-root balanced-realization transform (Laub et al. 1987).

    Returns ``(T, Tinv, hsv)`` such that ``Tinv @ A @ T`` is internally
    balanced, i.e. ``Wc == Wo == diag(hsv)``. Robust to numerically
    semidefinite Gramians (see :func:`_psd_sqrt`); states carrying a
    ~zero Hankel value are kept finite here and removed by
    :func:`balanced_truncation`.
    """
    Wc = controllability_gramian(A, B, dt=dt)
    Wo = observability_gramian(A, C, dt=dt)

    # Symmetric PSD square-root factors: Wc = R Rᵀ, Wo = L Lᵀ.
    R = _psd_sqrt(Wc)
    L = _psd_sqrt(Wo)

    # SVD of Lᵀ R = U Σ Vᵀ  →  balanced gramians both equal Σ.
    U, s, Vt = np.linalg.svd(L.T @ R)
    V = Vt.T

    # Floor near-zero Hankel values so the transform stays finite on stiff /
    # non-minimal systems; the true (unfloored) ``s`` is returned as the HSV
    # spectrum so the error bound and order selection remain exact.
    s_floor = np.maximum(s, np.finfo(s.dtype).eps * (s[0] if s.size else 1.0))
    s_inv_sqrt = np.diag(1.0 / np.sqrt(s_floor))
    T = R @ V @ s_inv_sqrt
    Tinv = s_inv_sqrt @ U.T @ L.T
    return T, Tinv, s


def balanced_realization(sys):
    r"""Internally-balanced realization of ``sys``.

    Returns ``(balanced_system, hsv)`` where ``balanced_system`` is an
    equivalent :class:`LinearizedSystem` whose controllability and
    observability Gramians are equal and diagonal, with the Hankel singular
    values ``hsv`` on the diagonal (Moore 1981; square-root algorithm of
    Laub, Heath, Paige & Ward 1987).

    Requires a stable ``sys``. A non-minimal or stiff system (Gramians only
    numerically semidefinite) is handled — see :func:`_psd_sqrt` — but its
    ~zero Hankel-value states are ill-defined in the *full* balanced form;
    use :func:`balanced_truncation` or :func:`minimal_realization` to remove
    them.
    """
    A, B, C, D = _abcd(sys)
    T, Tinv, hsv = _balancing_transform(A, B, C, sys.dt)
    Ab = Tinv @ A @ T
    Bb = Tinv @ B
    Cb = C @ T
    return _make_reduced(sys, Ab, Bb, Cb, D), hsv


def balanced_truncation(sys, order=None, tol=None):
    r"""Balanced truncation (Moore 1981).

    Balances ``sys`` and keeps the states associated with the largest Hankel
    singular values.

    Order selection:

    * ``order`` given — keep exactly that many states.
    * ``tol`` given (and ``order`` is ``None``) — keep the fewest states whose
      retained "energy" ``Σσ_kept² / Σσ²`` is at least ``1 - tol``; i.e. ``tol``
      is the fraction of Gramian energy allowed to be discarded.
    * neither given — no truncation (returns the balanced realization).

    The returned :class:`LinearizedSystem` additionally exposes:

    * ``.hsv`` — the full Hankel-singular-value spectrum,
    * ``.reduced_order`` — the retained state count ``r``,
    * ``.error_bound`` — the a priori :math:`H_\infty` error bound
      :math:`\lVert G - G_r\rVert_\infty \le 2\sum_{i>r}\sigma_i`
      (Glover 1984 / Enns 1984).
    """
    A, B, C, D = _abcd(sys)
    T, Tinv, hsv = _balancing_transform(A, B, C, sys.dt)
    n = A.shape[0]

    if order is not None:
        r = int(order)
    elif tol is not None:
        energy = np.cumsum(hsv**2)
        total = energy[-1]
        r = int(np.searchsorted(energy, (1.0 - tol) * total) + 1)
    else:
        r = n
    r = max(1, min(r, n))

    Ab = Tinv @ A @ T
    Bb = Tinv @ B
    Cb = C @ T

    reduced = _make_reduced(sys, Ab[:r, :r], Bb[:r, :], Cb[:, :r], D)
    reduced.hsv = hsv
    reduced.reduced_order = r
    reduced.error_bound = float(2.0 * np.sum(hsv[r:]))
    return reduced


#: Alias for :func:`balanced_truncation` (MATLAB / python-control naming).
balred = balanced_truncation


# --------------------------------------------------------------------------- #
# Minimal realization (Kalman decomposition via subspace projection)
# --------------------------------------------------------------------------- #
def _range_basis(M, tol):
    """Orthonormal basis for the column space of ``M`` (relative SVD tol)."""
    if M.size == 0:
        return np.zeros((M.shape[0], 0))
    U, s, _ = np.linalg.svd(M, full_matrices=False)
    if s.size == 0 or s[0] == 0.0:
        return np.zeros((M.shape[0], 0))
    r = int(np.sum(s > tol * s[0]))
    return U[:, :r]


def _controllability_matrix(A, B):
    n = A.shape[0]
    cols = [B]
    Ak = A
    for _ in range(1, n):
        cols.append(Ak @ B)
        Ak = Ak @ A
    return np.hstack(cols)


def minimal_realization(sys, tol=1e-8):
    r"""Minimal realization of ``sys`` (Kalman decomposition).

    Removes uncontrollable and unobservable modes by projecting onto the
    controllable subspace (range of the controllability matrix) and then onto
    the observable subspace (range of the observability matrix transposed).
    Ranks are decided from singular values with the relative threshold ``tol``.
    The input/output transfer function is preserved.
    """
    A, B, C, D = _abcd(sys)

    # 1) restrict to the controllable subspace
    Vc = _range_basis(_controllability_matrix(A, B), tol)
    if Vc.shape[1] == 0:
        # nothing controllable → static system
        return _make_reduced(sys, np.zeros((0, 0)), np.zeros((0, B.shape[1])),
                             np.zeros((C.shape[0], 0)), D)
    Ac = Vc.T @ A @ Vc
    Bc = Vc.T @ B
    Cc = C @ Vc

    # 2) restrict to the observable subspace (dual of step 1)
    Vo = _range_basis(_controllability_matrix(Ac.T, Cc.T), tol)
    if Vo.shape[1] == 0:
        return _make_reduced(sys, np.zeros((0, 0)), np.zeros((0, B.shape[1])),
                             np.zeros((C.shape[0], 0)), D)
    Am = Vo.T @ Ac @ Vo
    Bm = Vo.T @ Bc
    Cm = Cc @ Vo
    return _make_reduced(sys, Am, Bm, Cm, D)


#: Alias for :func:`minimal_realization` (MATLAB / python-control naming).
minreal = minimal_realization


# --------------------------------------------------------------------------- #
# Modal reduction (real modal form) + singular-perturbation residualization
# --------------------------------------------------------------------------- #
_ModalForm = namedtuple("_ModalForm", ["A", "B", "C", "blocks", "eigs"])


def _real_modal_form(A, B, C):
    """Transform ``(A, B, C)`` to a real block-diagonal modal realization.

    Real eigenvalues give 1x1 blocks ``[λ]``; each complex-conjugate pair
    ``σ ± jω`` gives a real 2x2 block ``[[σ, ω], [-ω, σ]]``.  Returns the
    transformed matrices, the list of ``(start, size, eigenvalue)`` blocks,
    and the per-block representative eigenvalues.
    """
    w, V = np.linalg.eig(A)
    n = A.shape[0]

    cols = []          # real basis columns of the transform T
    blocks = []        # (start_index, size, representative eigenvalue)
    used = np.zeros(n, dtype=bool)
    for i in range(n):
        if used[i]:
            continue
        lam = w[i]
        if abs(lam.imag) < 1e-12 * (1.0 + abs(lam.real)):
            used[i] = True
            cols.append(np.real(V[:, i]))
            blocks.append((len(cols) - 1, 1, complex(lam.real, 0.0)))
        else:
            # find the conjugate partner
            j = -1
            for k in range(i + 1, n):
                if not used[k] and abs(w[k] - np.conj(lam)) < 1e-9 * (1.0 + abs(lam)):
                    j = k
                    break
            used[i] = True
            start = len(cols)
            v = V[:, i]
            cols.append(np.real(v))
            cols.append(np.imag(v))
            if j >= 0:
                used[j] = True
            blocks.append((start, 2, lam))

    T = np.column_stack(cols)
    Tinv = np.linalg.inv(T)
    Am = np.real(Tinv @ A @ T)
    Bm = np.real(Tinv @ B)
    Cm = np.real(C @ T)
    return _ModalForm(Am, Bm, Cm, blocks, w)


def _dominance_key(lam, dt):
    """Rank key: larger == more dominant / slower.

    Continuous: real part (closest to the imaginary axis = slowest).
    Discrete: modulus (closest to the unit circle = slowest).
    """
    return abs(lam) if dt is not None else lam.real


def _select_blocks(blocks, dt, order, keep, n):
    """Choose which modal states to retain, keeping conjugate pairs intact.

    Returns a boolean mask over the ``n`` modal states.
    """
    mask = np.zeros(n, dtype=bool)
    if keep is not None:
        keep = set(int(k) for k in keep)
        for (start, size, _lam) in blocks:
            if any((start + off) in keep for off in range(size)):
                mask[start:start + size] = True
        return mask

    ranked = sorted(blocks, key=lambda b: _dominance_key(b[2], dt), reverse=True)
    if order is None:
        mask[:] = True
        return mask

    target = int(order)
    kept = 0
    for (start, size, _lam) in ranked:
        if kept >= target:
            break
        mask[start:start + size] = True
        kept += size
    if kept == 0 and ranked:  # always keep at least the top block
        start, size, _ = ranked[0]
        mask[start:start + size] = True
    return mask


def modal_truncation(sys, order=None, keep=None):
    r"""Modal truncation.

    Transforms ``sys`` to a real block-diagonal modal realization and keeps
    the dominant (slowest) modes, discarding the rest.  Complex-conjugate
    pairs are always kept or dropped together, so the reduced model stays
    real; the retained poles are exactly the retained eigenvalues.

    * ``order`` — target number of retained states (a straddling conjugate
      pair may push the actual count to ``order + 1``).
    * ``keep`` — explicit iterable of modal-state indices to retain (expanded
      to whole blocks).
    * neither — no truncation (returns the modal-form equivalent).

    Because coupling to the discarded modes is dropped outright, the DC gain
    generally shifts; use :func:`residualize` to preserve it.
    """
    A, B, C, D = _abcd(sys)
    mf = _real_modal_form(A, B, C)
    n = A.shape[0]
    mask = _select_blocks(mf.blocks, sys.dt, order, keep, n)

    Ar = mf.A[np.ix_(mask, mask)]
    Br = mf.B[mask, :]
    Cr = mf.C[:, mask]
    return _make_reduced(sys, Ar, Br, Cr, D)


def residualize(sys, order=None, keep=None):
    r"""Singular-perturbation (residualization) reduction.

    Like :func:`modal_truncation`, but instead of deleting the fast modes it
    sets their derivative (continuous) or their increment (discrete) to zero
    and solves for their quasi-steady value, folding it back into the
    retained model.  This matches the DC gain of the discarded modes
    (Kokotović, Khalil & O'Reilly 1986).

    Partitioning the modal realization into retained ``(1)`` and discarded
    ``(2)`` states, continuous time gives

    .. math::

        A_r &= A_{11} - A_{12} A_{22}^{-1} A_{21}, &
        B_r &= B_1 - A_{12} A_{22}^{-1} B_2, \\
        C_r &= C_1 - C_2 A_{22}^{-1} A_{21}, &
        D_r &= D - C_2 A_{22}^{-1} B_2,

    and discrete time replaces ``A_{22}^{-1}`` by ``-(I - A_{22})^{-1}``.

    Selection arguments match :func:`modal_truncation`.
    """
    A, B, C, D = _abcd(sys)
    mf = _real_modal_form(A, B, C)
    n = A.shape[0]
    keep_mask = _select_blocks(mf.blocks, sys.dt, order, keep, n)
    drop_mask = ~keep_mask

    Am, Bm, Cm = mf.A, mf.B, mf.C
    if not drop_mask.any():
        return _make_reduced(sys, Am, Bm, Cm, D)

    A11 = Am[np.ix_(keep_mask, keep_mask)]
    A12 = Am[np.ix_(keep_mask, drop_mask)]
    A21 = Am[np.ix_(drop_mask, keep_mask)]
    A22 = Am[np.ix_(drop_mask, drop_mask)]
    B1 = Bm[keep_mask, :]
    B2 = Bm[drop_mask, :]
    C1 = Cm[:, keep_mask]
    C2 = Cm[:, drop_mask]

    if sys.dt is None:
        M = np.linalg.solve(A22, np.hstack([A21, B2]))  # A22^{-1} [A21  B2]
    else:
        ImA22 = np.eye(A22.shape[0]) - A22
        # discrete steady state uses -(I - A22)^{-1}
        M = -np.linalg.solve(ImA22, np.hstack([A21, B2]))
    k = A21.shape[1]
    MA = M[:, :k]
    MB = M[:, k:]

    Ar = A11 - A12 @ MA
    Br = B1 - A12 @ MB
    Cr = C1 - C2 @ MA
    Dr = D - C2 @ MB
    return _make_reduced(sys, Ar, Br, Cr, Dr)
