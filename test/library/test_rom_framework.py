# SPDX-License-Identifier: MIT

"""Integration tests for the ``reduce`` front door and ``ReducedOrderModel``
wrapper (T-143). The per-method numerics are covered in test_rom_linear_mor /
_pod / _dmd / _koopman / _surrogates; here we check the dispatcher wires each
family to a simulatable ``.system`` and reports sane provenance.
"""

import numpy as np
import pytest

import jaxonomy
from jaxonomy.library import reduce, ReducedOrderModel
from jaxonomy.library.rom import polynomial_dictionary

pytestmark = pytest.mark.minimal


def _static_gain(A, B, C, D):
    """Continuous-time DC gain y/u at steady state: D - C A^{-1} B."""
    A, B, C, D = map(np.asarray, (A, B, C, D))
    return (D - C @ np.linalg.solve(A, B)).ravel()


class TestLinearMOR:
    def _full(self):
        # Three well-separated real modes; the fast one (-40) barely reaches
        # the output, so a 2-state balanced truncation should be near-exact.
        A = np.diag([-1.0, -3.0, -40.0])
        B = np.array([[1.0], [1.0], [1.0]])
        C = np.array([[1.0, 1.0, 0.02]])
        D = np.zeros((1, 1))
        return A, B, C, D

    def test_balred_wraps_lti_and_reports_bound(self):
        A, B, C, D = self._full()
        rm = reduce((A, B, C, D), method="balred", order=2)
        assert isinstance(rm, ReducedOrderModel)
        assert rm.full_order == 3 and rm.reduced_order == 2
        assert rm.info["error_bound"] > 0.0
        # DC gain preserved by balanced truncation to within the H-inf bound.
        g_full = _static_gain(A, B, C, D)
        g_red = _static_gain(rm.system.A, rm.system.B, rm.system.C, rm.system.D)
        assert abs(g_full - g_red) <= rm.info["error_bound"] + 1e-6

    def test_balred_system_simulates(self):
        A, B, C, D = self._full()
        rm = reduce((A, B, C, D), method="balred", order=2)
        sys = rm.system
        sys.input_ports[0].fix_value(1.0)
        ctx = sys.create_context()
        res = jaxonomy.simulate(
            sys, ctx, (0.0, 20.0),
            recorded_signals={"y": sys.output_ports[0]},
        )
        # Settled step response of the reduced block ≈ its own DC gain.
        y_final = np.asarray(res.outputs["y"])[-1]
        g_red = _static_gain(sys.A, sys.B, sys.C, sys.D)
        assert np.allclose(y_final, g_red, atol=1e-3)

    def test_minreal_drops_unobservable_mode(self):
        # Mode at -5 is unobservable (C row is zero there) → minreal removes it.
        A = np.diag([-2.0, -5.0])
        B = np.array([[1.0], [1.0]])
        C = np.array([[1.0, 0.0]])
        D = np.zeros((1, 1))
        rm = reduce((A, B, C, D), method="minreal")
        assert rm.reduced_order < rm.full_order

    def test_modal_and_residualize_run(self):
        A, B, C, D = self._full()
        for method in ("modal", "residualize"):
            rm = reduce((A, B, C, D), method=method, order=2)
            assert rm.reduced_order <= 3
            assert rm.method == ("modal" if method == "modal" else "residualize")


class TestDataDriven:
    def test_dmd_forecaster_matches_propagation(self):
        # Stable spiral: snapshots of a known discrete linear system.
        theta = 0.3
        Ad = 0.97 * np.array([[np.cos(theta), -np.sin(theta)],
                              [np.sin(theta), np.cos(theta)]])
        x = np.array([1.0, 0.0])
        cols = [x]
        for _ in range(60):
            x = Ad @ x
            cols.append(x)
        X = np.array(cols).T
        rm = reduce(X, method="dmd")
        assert rm.method == "dmd" and rm.reduced_order == 2
        # Recovered eigenvalues match the true spectrum.
        true_eigs = np.sort(np.linalg.eigvals(Ad))
        got = np.sort(rm.info["eigenvalues"])
        assert np.allclose(np.sort(np.abs(true_eigs)), np.sort(np.abs(got)), atol=1e-6)
        # The reconstructed full operator predicts a *fresh* IC (the DMD modes
        # are genuine eigenvectors of A, so Φ diag(λ) Φ⁺ recovers A exactly).
        res = rm.info["result"]
        A_full = np.real(res.modes @ np.diag(res.eigenvalues) @ np.linalg.pinv(res.modes))
        assert np.allclose(A_full, Ad, atol=1e-9)

    def test_dmdc_requires_inputs(self):
        X = np.random.default_rng(0).standard_normal((3, 20))
        with pytest.raises(ValueError, match="control inputs"):
            reduce(X, method="dmdc")

    def test_edmd_builds_koopman_predictor(self):
        rng = np.random.default_rng(1)
        # Mildly nonlinear map lifted with a polynomial dictionary; X[k] and
        # Xp=x[k+1] are aligned column-wise.
        X = rng.uniform(-1, 1, size=(2, 200))
        Xp = np.vstack([0.9 * X[0] + 0.1 * X[1] ** 2, 0.8 * X[1]])
        rm = reduce(
            X, method="edmd",
            dictionary=polynomial_dictionary(2), Xp=Xp, initial_state=X[:, 0],
        )
        assert rm.method == "edmd"
        # Predictor advances and de-lifts to a finite physical state.
        sys = rm.system
        ctx = sys.create_context()
        res = jaxonomy.simulate(sys, ctx, (0.0, 5.0))
        assert np.isfinite(np.asarray(res.context.discrete_state)).all()


class TestErrors:
    def test_projection_method_points_to_galerkin(self):
        with pytest.raises(ValueError, match="galerkin_reduce"):
            reduce(np.eye(3), method="pod")

    def test_unknown_method(self):
        with pytest.raises(ValueError, match="Unknown reduction method"):
            reduce(np.eye(3), method="nope")

    def test_bad_linear_target(self):
        with pytest.raises(TypeError):
            reduce(12345, method="balred")
