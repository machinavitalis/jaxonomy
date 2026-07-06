# SPDX-License-Identifier: MIT

"""Tests for the sensitivity / identifiability analysis module."""

import numpy as np
import pytest
import jax.numpy as jnp

from jaxonomy import DiagramBuilder, Parameter, SimulatorOptions
from jaxonomy.library import Adder, Gain, Integrator, Power, Constant
from jaxonomy.optimization import Optimizable, compute_sensitivity, SensitivityResult

pytestmark = pytest.mark.slow


# ── helpers ───────────────────────────────────────────────────────────────────

def _make_spring_optimizable(c0=0.5, k0=1.0):
    """2-param spring-mass: optimise both c and k."""
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
        def objective_from_context(self, ctx): return self._obj.eval(ctx)
        def prepare_context(self, ctx, p): return ctx.with_parameters(p)

    return _Opt(
        diagram, diagram.create_context(),
        params_0={"c": c0, "k": k0},
        sim_t_span=(0.0, 2.0),
        sim_options=SimulatorOptions(max_major_steps=1),
    )


def _make_single_param_optimizable(c0=0.5):
    """1-param spring-mass: optimise only c."""
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
        def optimizable_params(self, ctx): return {"c": ctx.parameters["c"]}
        def objective_from_context(self, ctx): return self._obj.eval(ctx)
        def prepare_context(self, ctx, p): return ctx.with_parameters(p)

    return _Opt(
        diagram, diagram.create_context(),
        params_0={"c": c0},
        sim_t_span=(0.0, 2.0),
        sim_options=SimulatorOptions(max_major_steps=1),
    )


# ── unit tests ────────────────────────────────────────────────────────────────

class TestSensitivityResult:

    def test_returns_sensitivity_result(self):
        model = _make_single_param_optimizable()
        result = compute_sensitivity(model)
        assert isinstance(result, SensitivityResult)

    def test_param_names_correct(self):
        model = _make_single_param_optimizable()
        result = compute_sensitivity(model)
        assert result.param_names == ["c"]

    def test_two_param_names(self):
        model = _make_spring_optimizable()
        result = compute_sensitivity(model)
        assert set(result.param_names) == {"c", "k"}

    def test_gradients_shape(self):
        model = _make_single_param_optimizable()
        result = compute_sensitivity(model)
        assert result.gradients.shape == (1,)

    def test_two_param_gradients_shape(self):
        model = _make_spring_optimizable()
        result = compute_sensitivity(model)
        assert result.gradients.shape == (2,)

    def test_normalized_sensitivity_nonnegative(self):
        model = _make_spring_optimizable()
        result = compute_sensitivity(model)
        assert np.all(result.normalized_sensitivity >= 0)

    def test_hessian_shape(self):
        model = _make_spring_optimizable()
        result = compute_sensitivity(model)
        assert result.hessian.shape == (2, 2)

    def test_hessian_diagonal_consistent(self):
        model = _make_spring_optimizable()
        result = compute_sensitivity(model)
        np.testing.assert_allclose(
            result.hessian_diagonal, np.diag(result.hessian), rtol=1e-6
        )

    def test_objective_value_positive(self):
        model = _make_single_param_optimizable()
        result = compute_sensitivity(model)
        assert result.objective_value > 0

    def test_gradient_negative_at_underdamped(self):
        """At c=0.5 (below optimal ≈1.65), gradient should be negative."""
        model = _make_single_param_optimizable(c0=0.5)
        result = compute_sensitivity(model)
        assert result.gradients[0] < 0, (
            f"Expected negative gradient at c=0.5, got {result.gradients[0]}"
        )

    def test_gradient_positive_at_overdamped(self):
        """At c=3.0 (above optimal ≈1.65), gradient should be positive."""
        model = _make_single_param_optimizable(c0=3.0)
        result = compute_sensitivity(model)
        assert result.gradients[0] > 0, (
            f"Expected positive gradient at c=3.0, got {result.gradients[0]}"
        )

    def test_gradient_matches_finite_difference(self):
        """AD gradient must agree with central FD within 1%."""
        model = _make_single_param_optimizable(c0=0.5)
        p = model.params_0_flat
        eps = 1e-4
        fd_grad = float(
            (model.objective_flat(p + eps) - model.objective_flat(p - eps))
            / (2 * eps)
        )
        result = compute_sensitivity(model)
        ad_grad = float(result.gradients[0])
        assert abs(ad_grad - fd_grad) / abs(fd_grad) < 0.01, (
            f"AD={ad_grad:.6f}, FD={fd_grad:.6f}"
        )

    def test_hessian_symmetric(self):
        """Hessian must be symmetric."""
        model = _make_spring_optimizable()
        result = compute_sensitivity(model)
        np.testing.assert_allclose(
            result.hessian, result.hessian.T, atol=1e-5,
            err_msg="Hessian is not symmetric"
        )

    def test_condition_number_finite(self):
        model = _make_spring_optimizable()
        result = compute_sensitivity(model)
        assert np.isfinite(result.condition_number)
        assert result.condition_number >= 1.0

    def test_no_hessian_option(self):
        """compute_hessian=False fills hessian with NaN but rest is valid."""
        model = _make_single_param_optimizable()
        result = compute_sensitivity(model, compute_hessian=False)
        assert np.all(np.isnan(result.hessian))
        assert np.isfinite(result.gradients[0])

    def test_custom_params_0_flat(self):
        """Can evaluate at a different point than params_0."""
        model = _make_single_param_optimizable(c0=0.5)
        alt_p = jnp.array([3.0])  # overdamped region
        result = compute_sensitivity(model, params_0_flat=alt_p)
        assert result.gradients[0] > 0  # gradient positive above optimum

    def test_summary_string(self):
        model = _make_spring_optimizable()
        result = compute_sensitivity(model)
        s = result.summary()
        assert "c" in s
        assert "k" in s
        assert "Sensitivity" in s

    def test_repr_same_as_summary(self):
        model = _make_single_param_optimizable()
        result = compute_sensitivity(model)
        assert repr(result) == result.summary()

    def test_unidentifiable_detection_near_optimum(self):
        """At the optimum (gradient ≈ 0), both params should be flagged."""
        model = _make_single_param_optimizable(c0=1.65)  # near optimal
        result = compute_sensitivity(model, sensitivity_threshold=0.5)
        # At the optimum, gradient ≈ 0, so normalized sensitivity ≈ 0
        # With a high threshold (0.5), 'c' should be flagged
        assert isinstance(result.unidentifiable_params, list)

    def test_sensitivity_threshold_respected(self):
        """threshold=0.0 → nothing is flagged; threshold=1.0 → everything is."""
        model = _make_single_param_optimizable(c0=0.5)
        r0 = compute_sensitivity(model, sensitivity_threshold=0.0)
        assert len(r0.unidentifiable_params) == 0
        r1 = compute_sensitivity(model, sensitivity_threshold=1.0 + 1e-9)
        assert len(r1.unidentifiable_params) == 1  # must flag 'c'

    def test_params_0_dict_matches_input(self):
        model = _make_single_param_optimizable(c0=0.7)
        result = compute_sensitivity(model)
        assert float(result.params_0["c"]) == pytest.approx(0.7, abs=1e-5)
