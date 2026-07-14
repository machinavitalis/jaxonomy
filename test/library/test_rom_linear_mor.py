# SPDX-License-Identifier: MIT

"""Tests for linear model-order reduction (``jaxonomy.library.rom.linear_mor``).

Covers Gramians, Hankel singular values, balanced realization/truncation,
minimal realization, modal truncation and residualization.  Cross-validated
against ``python-control`` (``control.balred`` / ``control.minreal`` /
``control.hsvd``, which require ``slycot``) where available, and against a
direct NumPy evaluation of the transfer function ``G(s) = C (sI - A)^{-1}B + D``
otherwise.
"""

import numpy as np
import pytest

import jaxonomy
from jaxonomy.library.linear_system import LinearizedSystem
from jaxonomy.library.rom.linear_mor import (
    controllability_gramian,
    observability_gramian,
    hankel_singular_values,
    balanced_realization,
    balanced_truncation,
    balred,
    minimal_realization,
    minreal,
    modal_truncation,
    residualize,
)

pytestmark = pytest.mark.minimal


# --------------------------------------------------------------------------- #
# helpers / fixtures
# --------------------------------------------------------------------------- #
def _linsys(A, B, C, D, dt=None):
    return LinearizedSystem(
        A=np.asarray(A, float),
        B=np.asarray(B, float),
        C=np.asarray(C, float),
        D=np.asarray(D, float),
        operating_point={"u": np.zeros(np.asarray(B, float).reshape(np.asarray(A).shape[0], -1).shape[1])},
        dt=dt,
    )


def siso_sys():
    """Stable 3-state SISO plant with a clear Hankel-value gap."""
    A = np.array([[-1.0, 0.0, 0.0], [0.0, -2.0, 0.0], [0.0, 0.0, -8.0]])
    B = np.array([[1.0], [1.0], [1.0]])
    C = np.array([[1.0, 1.0, 0.5]])
    D = np.array([[0.0]])
    return _linsys(A, B, C, D)


def mimo_sys():
    """Stable 4-state, 2-input, 2-output plant."""
    rng = np.random.default_rng(1)
    A = np.diag([-1.0, -3.0, -6.0, -10.0])
    B = rng.standard_normal((4, 2))
    C = rng.standard_normal((2, 4))
    D = np.zeros((2, 2))
    return _linsys(A, B, C, D)


def _tf(sys, s):
    """Evaluate the transfer-function matrix G(s) of ``sys`` at complex ``s``."""
    A, B, C, D = (np.asarray(sys.A), np.asarray(sys.B),
                  np.asarray(sys.C), np.asarray(sys.D))
    n = A.shape[0]
    return C @ np.linalg.solve(s * np.eye(n) - A, B) + D


def _hinf(sysA, sysB, w=None):
    """Peak ‖G_A(jw) − G_B(jw)‖₂ over a log frequency grid."""
    if w is None:
        w = np.logspace(-3, 3, 400)
    peak = 0.0
    for wi in w:
        diff = _tf(sysA, 1j * wi) - _tf(sysB, 1j * wi)
        peak = max(peak, float(np.linalg.svd(diff, compute_uv=False)[0]))
    return peak


# --------------------------------------------------------------------------- #
# Gramians
# --------------------------------------------------------------------------- #
class TestGramians:
    def test_continuous_lyapunov_residuals(self):
        sys = siso_sys()
        A, B, C = np.asarray(sys.A), np.asarray(sys.B), np.asarray(sys.C)
        Wc = controllability_gramian(A, B)
        Wo = observability_gramian(A, C)
        assert np.allclose(A @ Wc + Wc @ A.T, -(B @ B.T), atol=1e-10)
        assert np.allclose(A.T @ Wo + Wo @ A, -(C.T @ C), atol=1e-10)
        # Gramians of a stable system are symmetric PSD.
        assert np.allclose(Wc, Wc.T, atol=1e-12)
        assert np.all(np.linalg.eigvalsh(Wc) > -1e-12)

    def test_discrete_lyapunov_residuals(self):
        Ad = np.diag([0.5, 0.2, 0.05])
        Bd = np.array([[1.0], [1.0], [1.0]])
        Cd = np.array([[1.0, 1.0, 0.5]])
        Wc = controllability_gramian(Ad, Bd, dt=0.1)
        Wo = observability_gramian(Ad, Cd, dt=0.1)
        assert np.allclose(Ad @ Wc @ Ad.T - Wc, -(Bd @ Bd.T), atol=1e-10)
        assert np.allclose(Ad.T @ Wo @ Ad - Wo, -(Cd.T @ Cd), atol=1e-10)


# --------------------------------------------------------------------------- #
# Hankel singular values
# --------------------------------------------------------------------------- #
class TestHankelSingularValues:
    def test_sorted_descending_and_positive(self):
        hsv = hankel_singular_values(siso_sys())
        assert hsv.shape == (3,)
        assert np.all(np.diff(hsv) <= 1e-12)
        assert np.all(hsv >= 0.0)

    def test_matches_control_hsvd(self):
        control = pytest.importorskip("control")
        pytest.importorskip("slycot")
        sys = siso_sys()
        ss = control.ss(sys.A, sys.B, sys.C, sys.D)
        ref = np.sort(np.asarray(control.hsvd(ss), float))[::-1]
        assert np.allclose(hankel_singular_values(sys), ref, rtol=1e-6, atol=1e-9)


# --------------------------------------------------------------------------- #
# Balanced realization
# --------------------------------------------------------------------------- #
class TestBalancedRealization:
    def test_gramians_equal_and_diagonal(self):
        sys = siso_sys()
        bal, hsv = balanced_realization(sys)
        Wc = controllability_gramian(np.asarray(bal.A), np.asarray(bal.B))
        Wo = observability_gramian(np.asarray(bal.A), np.asarray(bal.C))
        # Both Gramians equal diag(hsv).
        assert np.allclose(Wc, np.diag(hsv), atol=1e-8)
        assert np.allclose(Wo, np.diag(hsv), atol=1e-8)

    def test_preserves_transfer_function(self):
        sys = siso_sys()
        bal, _ = balanced_realization(sys)
        assert _hinf(sys, bal) < 1e-8


# --------------------------------------------------------------------------- #
# Balanced truncation / balred
# --------------------------------------------------------------------------- #
class TestBalancedTruncation:
    def test_order_selection(self):
        red = balred(siso_sys(), order=2)
        assert np.asarray(red.A).shape == (2, 2)
        assert red.reduced_order == 2
        assert red.dt is None

    def test_tol_energy_selection(self):
        # 99.9% energy retained → drops the tiny third HSV.
        red = balanced_truncation(siso_sys(), tol=1e-3)
        assert red.reduced_order == 2

    def test_hinf_error_bound_siso(self):
        sys = siso_sys()
        red = balred(sys, order=2)
        err = _hinf(sys, red)
        # a priori bound: ‖G - Gr‖∞ ≤ 2 Σ σ_truncated
        assert err <= red.error_bound + 1e-9
        # bound should be non-trivial and reasonably tight here
        assert red.error_bound > 0.0

    def test_hinf_error_bound_mimo(self):
        sys = mimo_sys()
        red = balred(sys, order=2)
        assert np.asarray(red.B).shape == (2, 2)
        assert np.asarray(red.C).shape == (2, 2)
        assert _hinf(sys, red) <= red.error_bound + 1e-9

    def test_matches_control_balred_siso(self):
        control = pytest.importorskip("control")
        pytest.importorskip("slycot")
        sys = siso_sys()
        ss = control.ss(sys.A, sys.B, sys.C, sys.D)
        ref = control.balred(ss, 2)
        ref_sys = _linsys(ref.A, ref.B, ref.C, ref.D)
        red = balred(sys, order=2)
        # Balanced truncation of a system with distinct HSVs is unique up to
        # state coordinates → transfer functions must agree.
        assert _hinf(red, ref_sys) < 1e-6

    def test_matches_control_balred_mimo(self):
        control = pytest.importorskip("control")
        pytest.importorskip("slycot")
        sys = mimo_sys()
        ss = control.ss(sys.A, sys.B, sys.C, sys.D)
        ref = control.balred(ss, 2)
        ref_sys = _linsys(ref.A, ref.B, ref.C, ref.D)
        red = balred(sys, order=2)
        assert _hinf(red, ref_sys) < 1e-6

    def test_discrete_preserves_dt(self):
        sd = _linsys(np.diag([0.5, 0.2, 0.05]),
                     [[1.0], [1.0], [1.0]], [[1.0, 1.0, 0.5]], [[0.0]], dt=0.1)
        red = balred(sd, order=2)
        assert red.dt == 0.1
        assert np.asarray(red.A).shape == (2, 2)


# --------------------------------------------------------------------------- #
# Minimal realization / minreal
# --------------------------------------------------------------------------- #
class TestMinimalRealization:
    def _augmented(self):
        """A minimal 3-state plant padded with 1 uncontrollable + 1 unobservable
        mode → a 5-state non-minimal realization with the same TF."""
        # controllable+observable core (diagonal, 3 states)
        Ac = np.diag([-1.0, -2.0, -8.0])
        Bc = np.array([[1.0], [1.0], [1.0]])
        Cc = np.array([[1.0, 1.0, 0.5]])
        # add uncontrollable mode (B row = 0) and unobservable mode (C col = 0)
        A = np.diag([-1.0, -2.0, -8.0, -5.0, -4.0])
        B = np.array([[1.0], [1.0], [1.0], [0.0], [1.0]])   # state 4 uncontrollable
        C = np.array([[1.0, 1.0, 0.5, 1.0, 0.0]])           # state 5 unobservable
        core = _linsys(Ac, Bc, Cc, [[0.0]])
        full = _linsys(A, B, C, [[0.0]])
        return core, full

    def test_removes_hidden_modes(self):
        core, full = self._augmented()
        red = minreal(full)
        assert np.asarray(red.A).shape == (3, 3)
        # transfer function is preserved
        assert _hinf(full, red) < 1e-8
        assert _hinf(core, red) < 1e-8

    def test_minimal_system_unchanged_order(self):
        red = minimal_realization(siso_sys())
        assert np.asarray(red.A).shape == (3, 3)

    def test_matches_control_minreal(self):
        control = pytest.importorskip("control")
        pytest.importorskip("slycot")
        _core, full = self._augmented()
        ss = control.ss(full.A, full.B, full.C, full.D)
        ref = ss.minreal()
        red = minreal(full)
        assert np.asarray(red.A).shape[0] == ref.nstates == 3
        ref_sys = _linsys(ref.A, ref.B, ref.C, ref.D)
        assert _hinf(red, ref_sys) < 1e-6


# --------------------------------------------------------------------------- #
# Modal truncation & residualization
# --------------------------------------------------------------------------- #
class TestModalReduction:
    def test_modal_keeps_slowest_poles(self):
        red = modal_truncation(siso_sys(), order=2)
        poles = np.sort(np.linalg.eigvals(np.asarray(red.A)).real)
        assert np.allclose(poles, [-2.0, -1.0], atol=1e-8)

    def test_modal_real_for_complex_pair(self):
        # underdamped 2nd-order → complex conjugate pole pair
        A = np.array([[-0.5, 3.0], [-3.0, -0.5]])
        sys = _linsys(A, [[1.0], [0.0]], [[1.0, 0.0]], [[0.0]])
        red = modal_truncation(sys)  # no truncation, just modal form
        assert not np.iscomplexobj(np.asarray(red.A))
        assert _hinf(sys, red) < 1e-8

    def test_residualize_matches_dc_gain(self):
        sys = siso_sys()
        res = residualize(sys, order=2)
        trunc = modal_truncation(sys, order=2)
        dc_full = _tf(sys, 0.0).real
        # residualization matches the DC gain exactly...
        assert np.allclose(_tf(res, 0.0).real, dc_full, atol=1e-8)
        # ...while plain truncation does not.
        assert not np.allclose(_tf(trunc, 0.0).real, dc_full, atol=1e-3)

    def test_residualize_discrete_dc_gain(self):
        sd = _linsys(np.diag([0.5, 0.2, 0.05]),
                     [[1.0], [1.0], [1.0]], [[1.0, 1.0, 0.5]], [[0.0]], dt=0.1)
        res = residualize(sd, order=2)
        assert res.dt == 0.1
        # discrete DC gain is G(z=1)
        assert np.allclose(_tf(res, 1.0).real, _tf(sd, 1.0).real, atol=1e-8)


# --------------------------------------------------------------------------- #
# End-to-end: a reduced LinearizedSystem simulates in jaxonomy
# --------------------------------------------------------------------------- #
class TestSimulateReduced:
    def _step(self, sys, tf=6.0):
        block = sys.to_lti()
        block.input_ports[0].fix_value(1.0)
        ctx = block.create_context()
        results = jaxonomy.simulate(
            block, ctx, (0.0, tf),
            recorded_signals={"y": block.output_ports[0]},
        )
        return np.asarray(results.time), np.asarray(results.outputs["y"])

    def test_reduced_step_matches_full(self):
        sys = siso_sys()
        red = balred(sys, order=2)

        t_full, y_full = self._step(sys)
        t_red, y_red = self._step(red)

        # interpolate the reduced response onto the full time grid, compare
        y_red_i = np.interp(t_full, t_red, y_red)
        max_err = float(np.max(np.abs(y_full - y_red_i)))
        # order-2 reduction of a system whose 3rd HSV is ~4e-3 → tight match
        assert max_err < 0.05
        # final (settled) value should agree with the DC step response
        dc = float(np.asarray(_tf(sys, 0.0).real).reshape(-1)[0])
        assert abs(y_full[-1] - dc) < 1e-2
        assert abs(y_red[-1] - dc) < 5e-2


# --------------------------------------------------------------------------- #
# Regression: balancing must survive numerically semidefinite Gramians
# (stiff / large / non-minimal systems). Found by the balanced-truncation
# thermal-rod tutorial: the previous Cholesky-based square root raised
# LinAlgError once the Hankel spectrum fell below machine precision.
# --------------------------------------------------------------------------- #
def _heat_rod(n=100, alpha=1.0):
    """1-D finite-difference heat-conduction rod: a stiff, effectively
    low-rank (numerically semidefinite Gramian) SISO LTI of order n."""
    dx = 1.0 / (n + 1)
    A = np.zeros((n, n))
    for i in range(n):
        A[i, i] = -2.0 * alpha / dx**2
        if i > 0:
            A[i, i - 1] = alpha / dx**2
        if i < n - 1:
            A[i, i + 1] = alpha / dx**2
    B = np.zeros((n, 1))
    B[0, 0] = alpha / dx**2
    C = np.zeros((1, n))
    C[0, n // 2] = 1.0
    D = np.zeros((1, 1))
    return LinearizedSystem(A, B, C, D, {})


class TestSemidefiniteGramians:
    def test_hsv_spectrum_collapses(self):
        # The rod's Hankel values plunge below machine precision → the
        # Gramians are only numerically semidefinite (Cholesky would fail).
        hsv = hankel_singular_values(_heat_rod(100))
        assert hsv[0] > 1e-3
        assert hsv[-1] < 1e-12

    def test_balred_survives_and_is_finite(self):
        sys = _heat_rod(100)
        red = balred(sys, order=5)          # previously raised LinAlgError
        for M in (red.A, red.B, red.C, red.D):
            assert np.all(np.isfinite(np.asarray(M)))
        assert np.asarray(red.A).shape == (5, 5)
        # reduced model is stable
        assert np.all(np.linalg.eigvals(np.asarray(red.A)).real < 0)

    def test_balred_bound_holds_on_stiff_system(self):
        sys = _heat_rod(80)
        red = balred(sys, order=6)
        # sup over a frequency sweep of |G(jw) - Gr(jw)| ≤ 2*Σ tail HSV
        ws = np.logspace(-3, 4, 200)
        err = max(
            float(np.abs(np.asarray(_tf(sys, 1j * w) - _tf(red, 1j * w)).reshape(-1)[0]))
            for w in ws
        )
        assert err <= red.error_bound + 1e-9

    def test_balanced_realization_no_cholesky_crash(self):
        # The bare balanced_realization entry point must also not crash.
        _, hsv = balanced_realization(_heat_rod(60))
        assert np.all(np.isfinite(hsv))
        assert hsv[0] > 0.0
