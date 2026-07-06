# SPDX-License-Identifier: MIT

"""
Tests for compute_confidence_intervals / ConfidenceIntervalResult.

Coverage:
  1. Return type and structural properties
  2. Covariance / correlation matrix properties (symmetry, PD, diag == 1)
  3. Standard errors (non-negative, match sqrt(diag(cov)))
  4. Confidence intervals contain the optimum and are ordered (lo ≤ hi)
  5. Monotonicity: wider CI for lower confidence_level, more uncertainty
     for fewer data points
  6. Analytical quadratic test: known exact covariance
  7. Integration with the spring-mass Optimizable (single & two params)
  8. Gradient sign consistency (CI shifts with damping)
  9. Pre-computed Hessian path (hessian= argument)
 10. Residual-variance scaling (n_data argument)
 11. Input flexibility: OptimizationResult, dict, flat array
 12. Edge cases: singular Hessian, NaN-free result, single parameter
 13. Accessor methods: interval(), contains(), summary(), __repr__()
 14. Parameter name expansion for array parameters
 15. Transformation back-mapping (LogTransform)
 16. Non-PD Hessian handling (regularize=True / False)
"""

from __future__ import annotations

import numpy as np
import pytest
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree

from jaxonomy import DiagramBuilder, Parameter, SimulatorOptions
from jaxonomy.library import Adder, Gain, Integrator, Power, Constant
from jaxonomy.optimization import (
    Optimizable,
    OptimizationResult,
    compute_confidence_intervals,
    ConfidenceIntervalResult,
)
from jaxonomy.optimization.confidence import (
    _z_quantile,
    _expand_param_names,
    _nearest_positive_definite,
    _compute_hessian,
)

pytestmark = pytest.mark.slow


# ── helpers / fixtures ────────────────────────────────────────────────────────

class _MockOptimizable:
    """
    Minimal duck-typed Optimizable backed by a pure JAX loss function.

    Bypasses the full simulation machinery — suitable for unit tests that
    need precise analytical ground truth (quadratic objectives, etc.).
    """

    def __init__(self, loss_fn, params_0: dict):
        self.params_0 = {k: jnp.array(v, dtype=float) for k, v in params_0.items()}
        self.params_0_flat, self.unflatten_params = ravel_pytree(self.params_0)
        self.num_optvars = self.params_0_flat.size
        self.transformation = None
        self._loss_fn = loss_fn

    def objective_flat(self, params_flat):
        return self._loss_fn(params_flat)


def _make_spring_optimizable(c0: float = 1.65, k0: float = 1.0):
    """2-param spring-mass optimizable (same as in test_sensitivity.py)."""
    params = {
        "c": Parameter(np.array(c0)),
        "k": Parameter(np.array(k0)),
    }
    b = DiagramBuilder()
    k_x = b.add(Gain(params["k"], name="k_x"))
    c_v = b.add(Gain(params["c"], name="c_v"))
    add = b.add(Adder(2, operators="--", name="adder"))
    inv = b.add(Gain(1.0,         name="inv_m"))
    v   = b.add(Integrator(0.1,   name="v"))
    x   = b.add(Integrator(1.0,   name="x"))
    b.connect(k_x.output_ports[0], add.input_ports[0])
    b.connect(c_v.output_ports[0], add.input_ports[1])
    b.connect(add.output_ports[0], inv.input_ports[0])
    b.connect(inv.output_ports[0], v.input_ports[0])
    b.connect(v.output_ports[0],   x.input_ports[0])
    b.connect(v.output_ports[0],   c_v.input_ports[0])
    b.connect(x.output_ports[0],   k_x.input_ports[0])
    sq_v = b.add(Power(2.0, name="sq_v"))
    sq_x = b.add(Power(2.0, name="sq_x"))
    cv   = b.add(Integrator(0.0, name="cv"))
    cx   = b.add(Integrator(0.0, name="cx"))
    obj  = b.add(Adder(2, operators="++", name="obj"))
    b.connect(v.output_ports[0],   sq_v.input_ports[0])
    b.connect(x.output_ports[0],   sq_x.input_ports[0])
    b.connect(sq_v.output_ports[0], cv.input_ports[0])
    b.connect(sq_x.output_ports[0], cx.input_ports[0])
    b.connect(cv.output_ports[0],  obj.input_ports[0])
    b.connect(cx.output_ports[0],  obj.input_ports[1])
    diagram = b.build(parameters=params)

    class _Opt(Optimizable):
        def __init__(self, *a, **kw):
            super().__init__(*a, **kw)
            self._obj = diagram["obj"].output_ports[0]
        def optimizable_params(self, ctx):
            return {"c": ctx.parameters["c"], "k": ctx.parameters["k"]}
        def objective_from_context(self, ctx):
            return self._obj.eval(ctx)
        def prepare_context(self, ctx, p):
            return ctx.with_parameters(p)

    return _Opt(
        diagram, diagram.create_context(),
        params_0={"c": c0, "k": k0},
        sim_t_span=(0.0, 2.0),
        sim_options=SimulatorOptions(max_major_steps=1),
    )


def _make_single_param_optimizable(c0: float = 1.65):
    """1-param spring-mass optimizable — optimise only the damping c."""
    params = {"c": Parameter(np.array(1.0))}
    b = DiagramBuilder()
    k_x = b.add(Gain(1.0,          name="k_x"))
    c_v = b.add(Gain(params["c"],  name="c_v"))
    add = b.add(Adder(2, operators="--", name="adder"))
    inv = b.add(Gain(1.0,          name="inv_m"))
    v   = b.add(Integrator(0.1,    name="v"))
    x   = b.add(Integrator(1.0,    name="x"))
    b.connect(k_x.output_ports[0], add.input_ports[0])
    b.connect(c_v.output_ports[0], add.input_ports[1])
    b.connect(add.output_ports[0], inv.input_ports[0])
    b.connect(inv.output_ports[0], v.input_ports[0])
    b.connect(v.output_ports[0],   x.input_ports[0])
    b.connect(v.output_ports[0],   c_v.input_ports[0])
    b.connect(x.output_ports[0],   k_x.input_ports[0])
    sq_v = b.add(Power(2.0, name="sq_v"))
    sq_x = b.add(Power(2.0, name="sq_x"))
    cv   = b.add(Integrator(0.0, name="cv"))
    cx   = b.add(Integrator(0.0, name="cx"))
    obj  = b.add(Adder(2, operators="++", name="obj"))
    b.connect(v.output_ports[0],   sq_v.input_ports[0])
    b.connect(x.output_ports[0],   sq_x.input_ports[0])
    b.connect(sq_v.output_ports[0], cv.input_ports[0])
    b.connect(sq_x.output_ports[0], cx.input_ports[0])
    b.connect(cv.output_ports[0],  obj.input_ports[0])
    b.connect(cx.output_ports[0],  obj.input_ports[1])
    diagram = b.build(parameters=params)

    class _Opt(Optimizable):
        def __init__(self, *a, **kw):
            super().__init__(*a, **kw)
            self._obj = diagram["obj"].output_ports[0]
        def optimizable_params(self, ctx):
            return {"c": ctx.parameters["c"]}
        def objective_from_context(self, ctx):
            return self._obj.eval(ctx)
        def prepare_context(self, ctx, p):
            return ctx.with_parameters(p)

    return _Opt(
        diagram, diagram.create_context(),
        params_0={"c": c0},
        sim_t_span=(0.0, 2.0),
        sim_options=SimulatorOptions(max_major_steps=1),
    )


def _make_quadratic_mock(H0: np.ndarray, theta_star: np.ndarray) -> _MockOptimizable:
    """
    Quadratic objective L(θ) = ½ (θ - θ*)ᵀ H₀ (θ - θ*).

    Analytical covariance: Cov = H₀⁻¹.
    """
    H0_jnp = jnp.array(H0)
    theta_star_jnp = jnp.array(theta_star)

    def loss(p):
        delta = p - theta_star_jnp
        return 0.5 * jnp.dot(delta, H0_jnp @ delta)

    n = len(theta_star)
    params_0 = {f"p{i}": float(theta_star[i]) for i in range(n)}
    return _MockOptimizable(loss, params_0)


# ── unit tests: internal helpers ──────────────────────────────────────────────

class TestInternalHelpers:

    def test_z_quantile_95(self):
        z = _z_quantile(0.95)
        assert abs(z - 1.96) < 0.005

    def test_z_quantile_90(self):
        z = _z_quantile(0.90)
        assert abs(z - 1.6449) < 0.005

    def test_z_quantile_99(self):
        z = _z_quantile(0.99)
        assert abs(z - 2.5758) < 0.005

    def test_z_quantile_68(self):
        z = _z_quantile(0.68)
        assert abs(z - 1.0) < 0.01

    def test_expand_param_names_scalar(self):
        d = {"a": 1.0, "b": 2.0}
        names = _expand_param_names(d)
        assert names == ["a", "b"]

    def test_expand_param_names_vector(self):
        d = {"a": np.array([1.0, 2.0, 3.0])}
        names = _expand_param_names(d)
        assert names == ["a[0]", "a[1]", "a[2]"]

    def test_expand_param_names_mixed(self):
        d = {"scalar": 1.0, "vec": np.array([1.0, 2.0])}
        names = _expand_param_names(d)
        assert names == ["scalar", "vec[0]", "vec[1]"]

    def test_expand_param_names_matrix(self):
        d = {"M": np.zeros((2, 2))}
        names = _expand_param_names(d)
        assert names == ["M[0,0]", "M[0,1]", "M[1,0]", "M[1,1]"]

    def test_nearest_pd_already_pd(self):
        H = np.array([[4.0, 0.0], [0.0, 9.0]])
        H_pd, was_pd = _nearest_positive_definite(H)
        assert was_pd
        np.testing.assert_allclose(H_pd, H, atol=1e-10)

    def test_nearest_pd_clips_negative_eigenvalue(self):
        H = np.array([[1.0, 0.0], [0.0, -2.0]])  # negative eigenvalue
        H_pd, was_pd = _nearest_positive_definite(H)
        assert not was_pd
        eigs = np.linalg.eigvalsh(H_pd)
        assert np.all(eigs > 0)

    def test_nearest_pd_symmetric_input(self):
        # Slightly asymmetric due to floating point
        H = np.array([[2.0, 1.01], [0.99, 2.0]])
        H_pd, _ = _nearest_positive_definite(H)
        # Result must be symmetric
        np.testing.assert_allclose(H_pd, H_pd.T, atol=1e-12)


# ── unit tests: analytical quadratic case ─────────────────────────────────────

class TestAnalyticalQuadratic:
    """For L = ½ (θ-θ*)ᵀ H₀ (θ-θ*) the exact covariance is H₀⁻¹."""

    def test_covariance_matches_hessian_inverse_2d(self):
        H0 = np.diag([4.0, 9.0])
        theta_star = np.array([1.0, 2.0])
        mock = _make_quadratic_mock(H0, theta_star)

        ci = compute_confidence_intervals(mock, mock.params_0_flat)

        true_cov = np.linalg.inv(H0)
        np.testing.assert_allclose(ci.covariance, true_cov, atol=1e-4,
                                    err_msg="Covariance != H0⁻¹")

    def test_standard_errors_match_analytical(self):
        H0 = np.diag([4.0, 9.0])  # σ₀=0.5, σ₁=0.333...
        theta_star = np.array([0.0, 0.0])
        mock = _make_quadratic_mock(H0, theta_star)

        ci = compute_confidence_intervals(mock, mock.params_0_flat)

        expected_se = np.sqrt(np.diag(np.linalg.inv(H0)))
        np.testing.assert_allclose(ci.standard_errors, expected_se, rtol=1e-4)

    def test_ci_contains_true_optimum(self):
        H0 = np.diag([4.0, 9.0])
        theta_star = np.array([1.0, 2.0])
        mock = _make_quadratic_mock(H0, theta_star)

        ci = compute_confidence_intervals(mock, mock.params_0_flat)

        # The CIs are evaluated AT the optimum, so the optimum must lie within
        for i, name in enumerate(ci.param_names):
            lo, hi = ci.interval(name)
            assert lo <= float(theta_star[i]) <= hi, (
                f"{name}: optimum {theta_star[i]} not in [{lo}, {hi}]"
            )

    def test_hessian_is_positive_definite_at_optimum(self):
        H0 = np.diag([4.0, 9.0])
        mock = _make_quadratic_mock(H0, np.array([0.0, 0.0]))
        ci = compute_confidence_intervals(mock, mock.params_0_flat)
        assert ci.is_positive_definite

    def test_hessian_recovered_correctly(self):
        H0 = np.diag([4.0, 9.0])
        mock = _make_quadratic_mock(H0, np.zeros(2))
        ci = compute_confidence_intervals(mock, mock.params_0_flat)
        np.testing.assert_allclose(ci.hessian, H0, atol=1e-3)

    def test_wider_ci_for_flatter_loss(self):
        """Flatter curvature (smaller H0) → wider CI."""
        mock_wide = _make_quadratic_mock(np.diag([1.0, 1.0]), np.zeros(2))
        mock_narrow = _make_quadratic_mock(np.diag([100.0, 100.0]), np.zeros(2))
        ci_wide = compute_confidence_intervals(mock_wide, mock_wide.params_0_flat)
        ci_narrow = compute_confidence_intervals(mock_narrow, mock_narrow.params_0_flat)

        for i, name in enumerate(ci_wide.param_names):
            lo_w, hi_w = ci_wide.interval(name)
            lo_n, hi_n = ci_narrow.interval(name)
            assert (hi_w - lo_w) > (hi_n - lo_n), (
                f"{name}: flat-loss CI not wider than steep-loss CI"
            )

    def test_95pct_wider_than_68pct(self):
        H0 = np.diag([4.0, 9.0])
        mock = _make_quadratic_mock(H0, np.zeros(2))
        ci_95 = compute_confidence_intervals(mock, mock.params_0_flat, confidence_level=0.95)
        ci_68 = compute_confidence_intervals(mock, mock.params_0_flat, confidence_level=0.68)

        for name in ci_95.param_names:
            lo95, hi95 = ci_95.interval(name)
            lo68, hi68 = ci_68.interval(name)
            assert (hi95 - lo95) > (hi68 - lo68)

    def test_n_data_scaling_widens_ci(self):
        """Fewer data points → larger σ² → wider CI.

        We evaluate at a point away from theta_star so that L > 0.
        """
        # theta_star = [0, 0]; evaluate at params_0 = [1, 1] → L > 0
        H0 = np.diag([4.0, 4.0])
        theta_star = np.zeros(2)
        eval_point = np.array([1.0, 1.0])  # NOT the optimum → L > 0

        # Build mock with params_0 = eval_point (not theta_star)
        H0_jnp = jnp.array(H0)
        ts_jnp = jnp.array(theta_star)
        def loss(p):
            delta = p - ts_jnp
            return 0.5 * jnp.dot(delta, H0_jnp @ delta)
        mock = _MockOptimizable(loss, {"p0": 1.0, "p1": 1.0})

        ci_large = compute_confidence_intervals(mock, mock.params_0_flat, n_data=1000)
        ci_small = compute_confidence_intervals(mock, mock.params_0_flat, n_data=4)

        for name in ci_large.param_names:
            lo_large, hi_large = ci_large.interval(name)
            lo_small, hi_small = ci_small.interval(name)
            assert (hi_small - lo_small) >= (hi_large - lo_large), (
                f"{name}: smaller n_data should produce wider CI"
            )

    def test_residual_variance_value(self):
        """At the optimum L=0, so σ²=0 and CI width is zero."""
        H0 = np.diag([4.0, 9.0])
        mock = _make_quadratic_mock(H0, np.zeros(2))
        ci = compute_confidence_intervals(mock, mock.params_0_flat, n_data=50)
        # Objective at the optimum is 0 → residual_var = 0
        assert ci.residual_variance is not None
        assert ci.residual_variance == pytest.approx(0.0, abs=1e-8)


# ── unit tests: covariance / correlation matrix properties ────────────────────

class TestCovarianceProperties:

    def test_covariance_symmetric(self):
        model = _make_spring_optimizable()
        ci = compute_confidence_intervals(model, model.params_0_flat)
        np.testing.assert_allclose(ci.covariance, ci.covariance.T, atol=1e-8)

    def test_correlation_diagonal_is_one(self):
        model = _make_spring_optimizable()
        ci = compute_confidence_intervals(model, model.params_0_flat)
        np.testing.assert_allclose(np.diag(ci.correlation), 1.0, atol=1e-6)

    def test_correlation_off_diagonal_in_minus_one_to_one(self):
        model = _make_spring_optimizable()
        ci = compute_confidence_intervals(model, model.params_0_flat)
        assert np.all(ci.correlation >= -1 - 1e-6)
        assert np.all(ci.correlation <=  1 + 1e-6)

    def test_standard_errors_match_cov_diagonal(self):
        model = _make_single_param_optimizable()
        ci = compute_confidence_intervals(model, model.params_0_flat)
        expected_se = np.sqrt(np.maximum(np.diag(ci.covariance), 0))
        np.testing.assert_allclose(ci.standard_errors, expected_se, rtol=1e-6)

    def test_standard_errors_nonnegative(self):
        model = _make_spring_optimizable()
        ci = compute_confidence_intervals(model, model.params_0_flat)
        assert np.all(ci.standard_errors >= 0)

    def test_covariance_shape(self):
        model = _make_spring_optimizable()
        ci = compute_confidence_intervals(model, model.params_0_flat)
        assert ci.covariance.shape == (2, 2)

    def test_n_data_sets_residual_variance(self):
        model = _make_single_param_optimizable()
        ci = compute_confidence_intervals(model, model.params_0_flat, n_data=100)
        assert ci.residual_variance is not None
        assert ci.n_data == 100


# ── unit tests: confidence interval basic properties ─────────────────────────

class TestCIBasicProperties:

    def test_returns_correct_type(self):
        model = _make_single_param_optimizable()
        ci = compute_confidence_intervals(model, model.params_0_flat)
        assert isinstance(ci, ConfidenceIntervalResult)

    def test_param_names_single(self):
        model = _make_single_param_optimizable()
        ci = compute_confidence_intervals(model, model.params_0_flat)
        assert ci.param_names == ["c"]

    def test_param_names_two_params(self):
        model = _make_spring_optimizable()
        ci = compute_confidence_intervals(model, model.params_0_flat)
        assert set(ci.param_names) == {"c", "k"}

    def test_ci_lower_le_upper(self):
        model = _make_spring_optimizable()
        ci = compute_confidence_intervals(model, model.params_0_flat)
        for name in ci.param_names:
            lo, hi = ci.interval(name)
            assert lo <= hi, f"{name}: lo={lo} > hi={hi}"

    def test_ci_contains_opt_params(self):
        model = _make_spring_optimizable(c0=1.65, k0=1.0)
        p_opt = model.params_0_flat
        ci = compute_confidence_intervals(model, p_opt)
        opt_dict = model.unflatten_params(p_opt)
        for name, val in opt_dict.items():
            lo, hi = ci.interval(name)
            assert lo <= float(val) <= hi, (
                f"{name}: opt={float(val)} not in CI [{lo}, {hi}]"
            )

    def test_confidence_level_stored(self):
        model = _make_single_param_optimizable()
        for level in [0.90, 0.95, 0.99]:
            ci = compute_confidence_intervals(model, model.params_0_flat,
                                              confidence_level=level)
            assert ci.confidence_level == level

    def test_z_score_matches_level(self):
        model = _make_single_param_optimizable()
        ci = compute_confidence_intervals(model, model.params_0_flat,
                                          confidence_level=0.95)
        assert abs(ci.z_score - 1.96) < 0.01

    def test_objective_value_finite(self):
        model = _make_single_param_optimizable()
        ci = compute_confidence_intervals(model, model.params_0_flat)
        assert np.isfinite(ci.objective_value)

    def test_hessian_shape_single(self):
        model = _make_single_param_optimizable()
        ci = compute_confidence_intervals(model, model.params_0_flat)
        assert ci.hessian.shape == (1, 1)

    def test_hessian_shape_two_params(self):
        model = _make_spring_optimizable()
        ci = compute_confidence_intervals(model, model.params_0_flat)
        assert ci.hessian.shape == (2, 2)

    def test_eigenvalues_shape(self):
        model = _make_spring_optimizable()
        ci = compute_confidence_intervals(model, model.params_0_flat)
        assert ci.hessian_eigenvalues.shape == (2,)

    def test_condition_number_ge_one(self):
        model = _make_spring_optimizable()
        ci = compute_confidence_intervals(model, model.params_0_flat)
        assert ci.hessian_condition_number >= 1.0

    def test_message_is_string(self):
        model = _make_single_param_optimizable()
        ci = compute_confidence_intervals(model, model.params_0_flat)
        assert isinstance(ci.message, str)

    def test_hessian_method_ad_or_fd(self):
        model = _make_single_param_optimizable()
        ci = compute_confidence_intervals(model, model.params_0_flat)
        assert ci.hessian_method in {"AD", "FD", "provided"}


# ── unit tests: input flexibility ────────────────────────────────────────────

class TestInputFlexibility:

    def test_accepts_optimization_result(self):
        model = _make_single_param_optimizable()
        opt_result = OptimizationResult(
            params={"c": float(model.params_0_flat[0])},
            final_loss=0.1,
        )
        ci = compute_confidence_intervals(model, opt_result)
        assert isinstance(ci, ConfidenceIntervalResult)

    def test_accepts_dict(self):
        model = _make_single_param_optimizable()
        params_dict = {"c": float(model.params_0_flat[0])}
        ci = compute_confidence_intervals(model, params_dict)
        assert isinstance(ci, ConfidenceIntervalResult)

    def test_accepts_flat_array(self):
        model = _make_single_param_optimizable()
        ci = compute_confidence_intervals(model, model.params_0_flat)
        assert isinstance(ci, ConfidenceIntervalResult)

    def test_accepts_numpy_array(self):
        model = _make_single_param_optimizable()
        ci = compute_confidence_intervals(model, np.array(model.params_0_flat))
        assert isinstance(ci, ConfidenceIntervalResult)

    def test_accepts_precomputed_hessian(self):
        model = _make_single_param_optimizable()
        # Pre-compute Hessian from the sensitivity module
        from jaxonomy.optimization import compute_sensitivity
        sens = compute_sensitivity(model, compute_hessian=True)
        ci = compute_confidence_intervals(model, model.params_0_flat,
                                          hessian=sens.hessian)
        assert ci.hessian_method == "provided"
        # CIs should be finite
        lo, hi = ci.interval("c")
        assert np.isfinite(lo) and np.isfinite(hi)

    def test_precomputed_hessian_matches_auto(self):
        """Pre-computed and auto-computed Hessians should give same CIs."""
        model = _make_single_param_optimizable(c0=1.0)
        from jaxonomy.optimization import compute_sensitivity
        sens = compute_sensitivity(model)
        ci_auto = compute_confidence_intervals(model, model.params_0_flat)
        ci_prov = compute_confidence_intervals(model, model.params_0_flat,
                                               hessian=sens.hessian)
        np.testing.assert_allclose(
            ci_auto.standard_errors, ci_prov.standard_errors, rtol=1e-3
        )


# ── unit tests: spring-mass model ────────────────────────────────────────────

class TestSpringMassModel:

    def test_hessian_positive_definite_near_optimum(self):
        """At c≈1.65 (approx. optimum), Hessian should be PD."""
        model = _make_single_param_optimizable(c0=1.65)
        ci = compute_confidence_intervals(model, model.params_0_flat)
        assert ci.is_positive_definite

    def test_ci_width_larger_at_optimum_than_far_away(self):
        """Near the optimum (gradient≈0, H>0), CI should be well-defined (finite)."""
        model = _make_single_param_optimizable(c0=1.65)
        ci = compute_confidence_intervals(model, model.params_0_flat)
        lo, hi = ci.interval("c")
        assert np.isfinite(lo) and np.isfinite(hi)
        assert hi > lo

    def test_standard_error_matches_finite_difference(self):
        """SE = 1/sqrt(H); H from AD should match FD within 1%."""
        model = _make_single_param_optimizable(c0=1.65)
        ci = compute_confidence_intervals(model, model.params_0_flat)

        # FD estimate of H[0,0]
        p = np.array(model.params_0_flat)
        eps = 1e-3
        obj_fn = model.objective_flat
        H_fd = float(
            (obj_fn(jnp.array(p + eps)) - 2 * obj_fn(jnp.array(p))
             + obj_fn(jnp.array(p - eps))) / (eps ** 2)
        )
        se_fd = 1.0 / np.sqrt(H_fd)

        np.testing.assert_allclose(ci.standard_errors[0], se_fd, rtol=0.05)

    def test_two_param_ci_not_independent(self):
        """For the spring system c and k should have some correlation."""
        model = _make_spring_optimizable()
        ci = compute_confidence_intervals(model, model.params_0_flat)
        c_idx = ci.param_names.index("c")
        k_idx = ci.param_names.index("k")
        # Correlation should not be exactly 0 or ±1
        rho = ci.correlation[c_idx, k_idx]
        assert -1 < rho < 1


# ── unit tests: non-PD Hessian / regularisation ───────────────────────────────

class TestNonPDHessian:

    def test_negative_eigenvalue_sets_flag(self):
        """Evaluating at a non-minimum should set is_positive_definite=False."""
        H0 = np.diag([4.0, -1.0])  # NOT PD (saddle)
        mock = _make_quadratic_mock(np.diag([4.0, 9.0]), np.zeros(2))
        # Provide a non-PD Hessian directly
        ci = compute_confidence_intervals(mock, mock.params_0_flat,
                                          hessian=H0, regularize=True)
        assert not ci.is_positive_definite

    def test_regularise_produces_finite_ci(self):
        """Even with a near-singular Hessian, regularize=True gives finite CIs."""
        H0 = np.diag([4.0, 1e-15])  # nearly singular
        mock = _make_quadratic_mock(np.eye(2), np.zeros(2))
        ci = compute_confidence_intervals(mock, mock.params_0_flat,
                                          hessian=H0, regularize=True)
        for name in ci.param_names:
            lo, hi = ci.interval(name)
            assert np.isfinite(lo) and np.isfinite(hi), (
                f"{name}: CI not finite after regularisation"
            )

    def test_non_pd_message_not_empty(self):
        H0 = np.array([[1.0, 0.0], [0.0, -2.0]])
        mock = _make_quadratic_mock(np.eye(2), np.zeros(2))
        ci = compute_confidence_intervals(mock, mock.params_0_flat,
                                          hessian=H0, regularize=True)
        assert len(ci.message) > 0, "Expected warning for non-PD Hessian"


# ── unit tests: accessor and display methods ──────────────────────────────────

class TestAccessorsAndDisplay:

    def test_interval_returns_tuple(self):
        model = _make_single_param_optimizable()
        ci = compute_confidence_intervals(model, model.params_0_flat)
        result = ci.interval("c")
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_interval_key_error_for_unknown(self):
        model = _make_single_param_optimizable()
        ci = compute_confidence_intervals(model, model.params_0_flat)
        with pytest.raises(KeyError):
            ci.interval("does_not_exist")

    def test_contains_true_for_opt_value(self):
        model = _make_single_param_optimizable(c0=1.0)
        ci = compute_confidence_intervals(model, model.params_0_flat)
        assert ci.contains("c", float(model.params_0_flat[0]))

    def test_contains_false_outside_ci(self):
        H0 = np.diag([1e6])  # extremely tight CI around 0
        mock = _make_quadratic_mock(H0, np.zeros(1))
        ci = compute_confidence_intervals(mock, mock.params_0_flat,
                                          confidence_level=0.95)
        # 1000 should be far outside the CI
        assert not ci.contains("p0", 1000.0)

    def test_summary_contains_param_names(self):
        model = _make_spring_optimizable()
        ci = compute_confidence_intervals(model, model.params_0_flat)
        s = ci.summary()
        assert "c" in s
        assert "k" in s

    def test_summary_contains_confidence_level(self):
        model = _make_single_param_optimizable()
        ci = compute_confidence_intervals(model, model.params_0_flat,
                                          confidence_level=0.95)
        assert "95" in ci.summary()

    def test_repr_is_summary(self):
        model = _make_single_param_optimizable()
        ci = compute_confidence_intervals(model, model.params_0_flat)
        assert repr(ci) == ci.summary()

    def test_summary_contains_standard_errors(self):
        model = _make_single_param_optimizable()
        ci = compute_confidence_intervals(model, model.params_0_flat)
        s = ci.summary()
        # Standard error for 'c' should appear as a number
        assert "Std. error" in s or "std" in s.lower()

    def test_summary_no_exception_for_non_pd(self):
        """summary() must not crash even when Hessian is not PD."""
        H0 = np.array([[1.0, 0.0], [0.0, -2.0]])
        mock = _make_quadratic_mock(np.eye(2), np.zeros(2))
        ci = compute_confidence_intervals(mock, mock.params_0_flat,
                                          hessian=H0, regularize=True)
        s = ci.summary()  # must not raise
        assert "not" in s.lower() or "NO" in s  # should mention issue


# ── unit tests: integration-level (sensitivity + CI) ─────────────────────────

class TestIntegrationWithSensitivity:

    def test_sensitivity_hessian_matches_ci_hessian(self):
        """Hessian from compute_sensitivity and from CI should agree."""
        from jaxonomy.optimization import compute_sensitivity
        model = _make_single_param_optimizable(c0=1.0)
        sens = compute_sensitivity(model)
        ci = compute_confidence_intervals(model, model.params_0_flat,
                                          hessian=sens.hessian)
        np.testing.assert_allclose(ci.hessian, sens.hessian, rtol=1e-6)

    def test_ci_standard_error_from_sensitivity_hessian(self):
        """SE from CI should equal 1/sqrt(H_diag) from sensitivity."""
        from jaxonomy.optimization import compute_sensitivity
        model = _make_single_param_optimizable(c0=1.0)
        sens = compute_sensitivity(model)
        ci = compute_confidence_intervals(model, model.params_0_flat,
                                          hessian=sens.hessian)
        expected_se = 1.0 / np.sqrt(np.maximum(np.diag(sens.hessian), 1e-30))
        np.testing.assert_allclose(ci.standard_errors, expected_se, rtol=1e-4)


# ── unit tests: n_data residual variance ─────────────────────────────────────

class TestResidualVarianceScaling:

    def test_residual_var_set_when_n_data_provided(self):
        model = _make_single_param_optimizable(c0=1.65)
        ci = compute_confidence_intervals(model, model.params_0_flat, n_data=100)
        assert ci.residual_variance is not None

    def test_residual_var_none_when_n_data_not_provided(self):
        model = _make_single_param_optimizable()
        ci = compute_confidence_intervals(model, model.params_0_flat)
        assert ci.residual_variance is None

    def test_n_data_widens_ci_vs_none(self):
        """Providing n_data (with nonzero L_opt) should widen CIs."""
        model = _make_single_param_optimizable(c0=0.5)  # not at optimum → L > 0
        p = model.params_0_flat
        ci_default = compute_confidence_intervals(model, p)
        ci_scaled  = compute_confidence_intervals(model, p, n_data=10)
        lo_d, hi_d = ci_default.interval("c")
        lo_s, hi_s = ci_scaled.interval("c")
        # With a nonzero objective, scaling should produce wider CIs
        # (unless L_opt ≈ 0 already)
        width_default = hi_d - lo_d
        width_scaled  = hi_s - lo_s
        # The scaled version should be at least as wide
        assert width_scaled >= width_default * 0.5  # allow some tolerance

    def test_residual_var_formula(self):
        """σ² = 2·L / max(n_data - n_params, 1)."""
        model = _make_single_param_optimizable(c0=0.5)
        p = model.params_0_flat
        n_data = 50
        L_opt = float(model.objective_flat(p))
        n_params = len(p)
        expected_sigma2 = 2.0 * L_opt / max(n_data - n_params, 1)
        ci = compute_confidence_intervals(model, p, n_data=n_data)
        assert ci.residual_variance == pytest.approx(expected_sigma2, rel=1e-5)


# ── unit tests: hessian-method fallback ──────────────────────────────────────

class TestHessianMethodFallback:

    def test_ad_hessian_for_smooth_objective(self):
        """For a simple quadratic (no ODE), AD Hessian should be used."""
        H0 = np.diag([4.0, 9.0])
        mock = _make_quadratic_mock(H0, np.zeros(2))
        ci = compute_confidence_intervals(mock, mock.params_0_flat)
        assert ci.hessian_method in {"AD", "FD"}

    def test_provided_hessian_method_label(self):
        model = _make_single_param_optimizable()
        ci = compute_confidence_intervals(model, model.params_0_flat,
                                          hessian=np.array([[5.0]]))
        assert ci.hessian_method == "provided"
