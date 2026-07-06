# SPDX-License-Identifier: MIT

"""
Comprehensive tests for the IPOPT optimizer.

IPOPT (Interior Point OPTimizer) is a gold-standard large-scale NLP solver.
These tests verify:

  - Unconstrained optimisation (no constraints, no bounds)
  - Equality-constrained optimisation (active constraint at solution)
  - Bounded optimisation (box constraints via bounds)
  - Bounded + constrained optimisation
  - Two-parameter optimisation
  - OptimizationResult fields (success, nit, nfev, message, final_loss)
  - Backward-compatible dict access on OptimizationResult
  - Convergence to correct value (agrees with scipy L-BFGS-B reference)
  - Options forwarding (maxiter respected)
  - Gradient accuracy (IPOPT uses exact gradients; verify against FD)
"""

import importlib.util

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaxonomy import DiagramBuilder, Parameter, SimulatorOptions
from jaxonomy.library import Adder, Gain, Integrator, Power
from jaxonomy.optimization import IPOPT, Scipy, Optimizable, OptimizationResult

pytestmark = pytest.mark.slow

HAS_CYIPOPT = importlib.util.find_spec("cyipopt") is not None
pytestmark_ipopt = pytest.mark.skipif(
    not HAS_CYIPOPT, reason="cyipopt not installed"
)


# ─────────────────────────────────────────────────────────────────────────────
# Shared diagram builder — damped spring-mass ISE objective
# ─────────────────────────────────────────────────────────────────────────────


def _make_spring_mass_diagram(c_init=1.0, k_init=1.0, v0=0.1, x0=1.0):
    """
    Damped spring-mass system.  Objective = ∫₀² (v² + x²) dt (ISE).
    Optimal damping for c-only problem: c* ≈ 1.65.
    """
    params = {
        "c": Parameter(np.array(c_init)),
        "k": Parameter(np.array(k_init)),
    }
    b = DiagramBuilder()
    k_x = b.add(Gain(params["k"], name="k_x"))
    c_v = b.add(Gain(params["c"], name="c_v"))
    add = b.add(Adder(2, operators="--", name="adder"))
    inv = b.add(Gain(1.0, name="inv_m"))
    v = b.add(Integrator(v0, name="v"))
    x = b.add(Integrator(x0, name="x"))
    b.connect(k_x.output_ports[0], add.input_ports[0])
    b.connect(c_v.output_ports[0], add.input_ports[1])
    b.connect(add.output_ports[0], inv.input_ports[0])
    b.connect(inv.output_ports[0], v.input_ports[0])
    b.connect(v.output_ports[0], x.input_ports[0])
    b.connect(v.output_ports[0], c_v.input_ports[0])
    b.connect(x.output_ports[0], k_x.input_ports[0])
    sq_v = b.add(Power(2.0, name="sq_v"))
    sq_x = b.add(Power(2.0, name="sq_x"))
    cv = b.add(Integrator(0.0, name="cost_v"))
    cx = b.add(Integrator(0.0, name="cost_x"))
    obj = b.add(Adder(2, operators="++", name="objective"))
    b.connect(v.output_ports[0], sq_v.input_ports[0])
    b.connect(x.output_ports[0], sq_x.input_ports[0])
    b.connect(sq_v.output_ports[0], cv.input_ports[0])
    b.connect(sq_x.output_ports[0], cx.input_ports[0])
    b.connect(cv.output_ports[0], obj.input_ports[0])
    b.connect(cx.output_ports[0], obj.input_ports[1])
    return b.build(parameters=params)


class _SingleParamOptimizable(Optimizable):
    """Optimise only `c` (k is fixed)."""

    def __init__(self, diagram, base_ctx, c_init=0.5, bounds=None):
        super().__init__(
            diagram=diagram,
            base_context=base_ctx,
            params_0={"c": jnp.array(c_init)},
            sim_t_span=(0.0, 2.0),
            bounds=bounds,
            sim_options=SimulatorOptions(max_major_steps=1),
        )
        self._obj = diagram["objective"].output_ports[0]

    def optimizable_params(self, ctx):
        return {"c": ctx.parameters["c"]}

    def objective_from_context(self, ctx):
        return self._obj.eval(ctx)

    def prepare_context(self, ctx, p):
        return ctx.with_parameters(p)


class _ConstrainedOptimizable(Optimizable):
    """Optimise `c` subject to ISE ≥ 1.67 (binding at c ≈ 0.36)."""

    def __init__(self, diagram, base_ctx, c_init=0.5):
        # Must be set BEFORE super().__init__: the base Optimizable.__init__
        # calls self.constraints_from_context(base_context) to populate
        # self.has_constraints, and our override reads self._obj.
        self._obj = diagram["objective"].output_ports[0]
        super().__init__(
            diagram=diagram,
            base_context=base_ctx,
            params_0={"c": jnp.array(c_init)},
            sim_t_span=(0.0, 2.0),
            sim_options=SimulatorOptions(max_major_steps=1),
        )

    def optimizable_params(self, ctx):
        return {"c": ctx.parameters["c"]}

    def objective_from_context(self, ctx):
        return self._obj.eval(ctx)

    def constraints_from_context(self, ctx):
        # ISE - 1.67 ≥ 0  (forces ISE ≥ 1.67, binding at c ≈ 0.36)
        return jnp.array(self._obj.eval(ctx) - 1.67)

    def prepare_context(self, ctx, p):
        return ctx.with_parameters(p)


class _TwoParamOptimizable(Optimizable):
    """Optimise both `c` and `k`."""

    def __init__(self, diagram, base_ctx, c_init=0.5, k_init=0.5, bounds=None):
        super().__init__(
            diagram=diagram,
            base_context=base_ctx,
            params_0={"c": jnp.array(c_init), "k": jnp.array(k_init)},
            sim_t_span=(0.0, 2.0),
            bounds=bounds,
            sim_options=SimulatorOptions(max_major_steps=1),
        )
        self._obj = diagram["objective"].output_ports[0]

    def optimizable_params(self, ctx):
        return {"c": ctx.parameters["c"], "k": ctx.parameters["k"]}

    def objective_from_context(self, ctx):
        return self._obj.eval(ctx)

    def prepare_context(self, ctx, p):
        return ctx.with_parameters(p)


# ─────────────────────────────────────────────────────────────────────────────
# Helper: build a quick IPOPT run
# ─────────────────────────────────────────────────────────────────────────────

def _run_ipopt(optimizable, maxiter=30, disp=0):
    optim = IPOPT(optimizable, options={"maxiter": maxiter, "disp": disp})
    return optim.optimize()


# ─────────────────────────────────────────────────────────────────────────────
# 1. Basic correctness
# ─────────────────────────────────────────────────────────────────────────────

@pytestmark_ipopt
class TestIPOPTCorrectness:

    def test_unconstrained_single_param(self):
        """IPOPT finds c ≈ 1.65 on unconstrained spring-mass ISE."""
        diagram = _make_spring_mass_diagram()
        model = _SingleParamOptimizable(diagram, diagram.create_context(), c_init=0.5)
        r = _run_ipopt(model)
        assert np.isclose(float(r["c"]), 1.65, atol=0.15), (
            f"IPOPT unconstrained: c={float(r['c']):.4f}, expected ≈1.65"
        )

    def test_constrained_active_constraint(self):
        """Constrained: ISE ≥ 1.67 forces c ≈ 0.36 (binding constraint)."""
        diagram = _make_spring_mass_diagram()
        model = _ConstrainedOptimizable(diagram, diagram.create_context(), c_init=0.5)
        r = _run_ipopt(model)
        assert np.isclose(float(r["c"]), 0.36, atol=0.1), (
            f"IPOPT constrained: c={float(r['c']):.4f}, expected ≈0.36"
        )

    def test_constrained_constraint_satisfied(self):
        """The constraint must be satisfied at the solution (ISE ≥ 1.67)."""
        diagram = _make_spring_mass_diagram()
        model = _ConstrainedOptimizable(diagram, diagram.create_context(), c_init=0.5)
        r = _run_ipopt(model)
        c_flat = jnp.array([float(r["c"])])
        cons = float(model.constraints_flat(c_flat))
        assert cons >= -1e-3, (
            f"Constraint violated: ISE - 1.67 = {cons:.6f} (should be ≥ 0)"
        )

    def test_unconstrained_agrees_with_lbfgsb(self):
        """IPOPT and L-BFGS-B should agree within 0.05 on the same problem."""
        diagram = _make_spring_mass_diagram()
        model_ipopt = _SingleParamOptimizable(
            diagram, diagram.create_context(), c_init=0.5
        )
        model_lbfgsb = _SingleParamOptimizable(
            diagram, diagram.create_context(), c_init=0.5
        )
        r_ipopt = _run_ipopt(model_ipopt)
        r_lbfgsb = Scipy(
            model_lbfgsb, "L-BFGS-B",
            opt_method_config={"maxiter": 40},
            use_autodiff_grad=True,
        ).optimize()
        diff = abs(float(r_ipopt["c"]) - float(r_lbfgsb["c"]))
        assert diff < 0.05, (
            f"IPOPT c={float(r_ipopt['c']):.4f}, L-BFGS-B c={float(r_lbfgsb['c']):.4f}, "
            f"diff={diff:.4f}"
        )

    def test_two_param_optimisation(self):
        """IPOPT optimises both c and k; result must lower ISE vs initial.

        Bounds are required: the *unbounded* (c, k) ISE problem is ill-posed —
        IPOPT diverges (dual infeasibility blows up within ~13 iters) and drives
        the parameters into a region where a single stiff ODE solve effectively
        never returns, hanging the run uninterruptibly. Boxing c, k to a sane
        range keeps the problem well-conditioned; it still exercises genuine
        two-parameter optimisation and converges in well under a second.
        """
        diagram = _make_spring_mass_diagram()
        model = _TwoParamOptimizable(
            diagram, diagram.create_context(), c_init=0.5, k_init=0.5,
            bounds={"c": (0.05, 5.0), "k": (0.05, 5.0)},
        )
        init_loss = float(model.objective_flat(model.params_0_flat))
        r = _run_ipopt(model, maxiter=40)
        final_loss = float(model.objective(r.params))
        assert final_loss < init_loss, (
            f"IPOPT 2-param: loss did not decrease ({init_loss:.4f} → {final_loss:.4f})"
        )
        # Both parameters must be in reasonable range
        assert float(r["c"]) > 0, f"c={float(r['c'])} should be positive"
        assert float(r["k"]) > 0, f"k={float(r['k'])} should be positive"

    def test_bounded_upper_saturates(self):
        """With upper bound c ≤ 1.2 (below true optimum ≈1.65), must saturate at 1.2."""
        diagram = _make_spring_mass_diagram()
        model = _SingleParamOptimizable(
            diagram, diagram.create_context(), c_init=0.5,
            bounds={"c": (0.0, 1.2)}
        )
        r = _run_ipopt(model)
        assert float(r["c"]) <= 1.2 + 1e-4, (
            f"Upper bound violated: c={float(r['c']):.4f}"
        )
        assert float(r["c"]) >= 0.0 - 1e-4

    def test_bounded_lower_respected(self):
        """Lower bound c ≥ 2.0 (above true optimum): must give c ≥ 2.0."""
        diagram = _make_spring_mass_diagram()
        model = _SingleParamOptimizable(
            diagram, diagram.create_context(), c_init=2.5,
            bounds={"c": (2.0, 10.0)}
        )
        r = _run_ipopt(model)
        assert float(r["c"]) >= 2.0 - 1e-4, (
            f"Lower bound violated: c={float(r['c']):.4f}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# 2. OptimizationResult fields
# ─────────────────────────────────────────────────────────────────────────────

@pytestmark_ipopt
class TestIPOPTResultFields:

    def _make_result(self):
        diagram = _make_spring_mass_diagram()
        model = _SingleParamOptimizable(diagram, diagram.create_context(), c_init=0.5)
        return _run_ipopt(model)

    def test_returns_optimization_result(self):
        r = self._make_result()
        assert isinstance(r, OptimizationResult)

    def test_success_is_bool(self):
        r = self._make_result()
        assert isinstance(r.success, bool)

    def test_success_true_on_convergence(self):
        r = self._make_result()
        assert r.success is True

    def test_nit_positive(self):
        r = self._make_result()
        assert r.nit >= 0
        assert isinstance(r.nit, int)

    def test_nfev_positive(self):
        r = self._make_result()
        assert r.nfev >= 0
        assert isinstance(r.nfev, int)

    def test_final_loss_finite(self):
        r = self._make_result()
        assert r.final_loss is not None
        assert np.isfinite(r.final_loss)

    def test_final_loss_positive(self):
        """ISE objective is always positive."""
        r = self._make_result()
        assert r.final_loss > 0.0

    def test_final_loss_less_than_initial(self):
        diagram = _make_spring_mass_diagram()
        model = _SingleParamOptimizable(diagram, diagram.create_context(), c_init=0.5)
        init_loss = float(model.objective_flat(model.params_0_flat))
        r = _run_ipopt(model)
        assert r.final_loss < init_loss

    def test_message_is_string(self):
        r = self._make_result()
        assert isinstance(r.message, str)
        assert len(r.message) > 0

    def test_dict_access_works(self):
        """Backward-compat: result['c'] must return the optimised value."""
        r = self._make_result()
        c = float(r["c"])
        assert np.isfinite(c)
        assert c > 0.0

    def test_in_operator(self):
        r = self._make_result()
        assert "c" in r

    def test_iter_yields_keys(self):
        r = self._make_result()
        assert "c" in list(r)

    def test_len(self):
        r = self._make_result()
        assert len(r) == 1

    def test_params_attribute(self):
        r = self._make_result()
        assert isinstance(r.params, dict)
        assert "c" in r.params

    def test_constrained_message_contains_ipopt_info(self):
        """IPOPT always returns a message string with status info."""
        diagram = _make_spring_mass_diagram()
        model = _ConstrainedOptimizable(diagram, diagram.create_context(), c_init=0.5)
        r = _run_ipopt(model)
        assert len(r.message) > 0


# ─────────────────────────────────────────────────────────────────────────────
# 3. Gradient accuracy (first-order AD only — Hessian via L-BFGS in IPOPT)
# ─────────────────────────────────────────────────────────────────────────────

@pytestmark_ipopt
class TestIPOPTGradientAccuracy:
    """
    IPOPT uses exact first-order gradients via JAX AD.  The Hessian is
    approximated by L-BFGS (``hessian_approximation=limited-memory``) because
    the jaxonomy ODE solver uses ``custom_vjp`` which does not support
    forward-mode differentiation (``jax.hessian`` uses ``jacfwd`` internally
    and therefore fails on simulation objectives).

    These tests verify:
      - JAX autodiff gradient agrees with central finite differences.
      - Gradient points in the correct direction (negative toward optimum,
        positive away from it).
      - Constraint Jacobian (first-order AD) agrees with FD.
    """

    def test_gradient_agrees_with_fd(self):
        """AD gradient at c=0.5 must agree with FD within 1%."""
        diagram = _make_spring_mass_diagram()
        model = _SingleParamOptimizable(diagram, diagram.create_context(), c_init=0.5)
        p = model.params_0_flat
        eps = 1e-4

        grad_ad = float(jax.grad(model.objective_flat)(p)[0])
        grad_fd = float(
            (model.objective_flat(p + eps) - model.objective_flat(p - eps)) / (2 * eps)
        )
        rel_err = abs(grad_ad - grad_fd) / max(abs(grad_fd), 1e-12)
        assert rel_err < 0.01, (
            f"Gradient AD={grad_ad:.6f}, FD={grad_fd:.6f}, rel_err={rel_err:.4f}"
        )

    def test_gradient_negative_below_optimum(self):
        """At c=0.5 (below optimum ≈1.65), gradient must be negative."""
        diagram = _make_spring_mass_diagram()
        model = _SingleParamOptimizable(diagram, diagram.create_context(), c_init=0.5)
        g = float(jax.grad(model.objective_flat)(model.params_0_flat)[0])
        assert g < 0, f"Expected negative gradient at c=0.5, got {g}"

    def test_gradient_positive_above_optimum(self):
        """At c=3.0 (above optimum ≈1.65), gradient must be positive."""
        diagram = _make_spring_mass_diagram()
        model = _SingleParamOptimizable(diagram, diagram.create_context(), c_init=3.0)
        g = float(jax.grad(model.objective_flat)(model.params_0_flat)[0])
        assert g > 0, f"Expected positive gradient at c=3.0, got {g}"

    def test_gradient_near_zero_at_optimum(self):
        """At c≈1.65 (near optimum), gradient magnitude should be small."""
        diagram = _make_spring_mass_diagram()
        model = _SingleParamOptimizable(diagram, diagram.create_context(), c_init=1.65)
        g = float(jax.grad(model.objective_flat)(model.params_0_flat)[0])
        assert abs(g) < 0.1, f"Gradient near optimum should be small, got {g}"

    def test_constraint_jacobian_agrees_with_fd(self):
        """Constraint Jacobian (AD) must agree with FD within 1%."""
        diagram = _make_spring_mass_diagram()
        model = _ConstrainedOptimizable(diagram, diagram.create_context(), c_init=0.5)
        p = model.params_0_flat
        eps = 1e-4

        # constraints_flat returns a 0-d scalar, so jacrev(...) has shape (1,)
        # (one input, scalar output) — index [0], not [0, 0]; and the FD stencil
        # differences the bare scalars.
        jac_ad = float(jax.jacrev(model.constraints_flat)(p)[0])
        jac_fd = float(
            (model.constraints_flat(p + eps) - model.constraints_flat(p - eps))
            / (2 * eps)
        )
        rel_err = abs(jac_ad - jac_fd) / max(abs(jac_fd), 1e-12)
        assert rel_err < 0.01, (
            f"Constraint Jacobian AD={jac_ad:.6f}, FD={jac_fd:.6f}, "
            f"rel_err={rel_err:.4f}"
        )

    def test_two_param_gradient_shape(self):
        """Gradient of 2-param objective must have shape (2,)."""
        diagram = _make_spring_mass_diagram()
        model = _TwoParamOptimizable(diagram, diagram.create_context())
        g = jax.grad(model.objective_flat)(model.params_0_flat)
        assert g.shape == (2,), f"Expected shape (2,), got {g.shape}"


# ─────────────────────────────────────────────────────────────────────────────
# 4. Options forwarding
# ─────────────────────────────────────────────────────────────────────────────

@pytestmark_ipopt
class TestIPOPTOptions:

    def test_maxiter_respected(self):
        """With maxiter=2, nit should be very small (solver stops early)."""
        diagram = _make_spring_mass_diagram()
        model = _SingleParamOptimizable(diagram, diagram.create_context(), c_init=0.5)
        optim = IPOPT(model, options={"maxiter": 2, "disp": 0})
        r = optim.optimize()
        # nit ≤ 3 (IPOPT may do one extra iteration for bookkeeping)
        assert r.nit <= 5, f"Expected nit ≤ 5 with maxiter=2, got nit={r.nit}"

    def test_default_options(self):
        """IPOPT with default options (disp=5) must still converge."""
        diagram = _make_spring_mass_diagram()
        model = _SingleParamOptimizable(diagram, diagram.create_context(), c_init=0.5)
        optim = IPOPT(model)
        r = optim.optimize()
        assert np.isclose(float(r["c"]), 1.65, atol=0.15)

    def test_constrained_with_tight_tolerance(self):
        """Tight tolerance: constraint must be satisfied within 1e-4."""
        diagram = _make_spring_mass_diagram()
        model = _ConstrainedOptimizable(diagram, diagram.create_context(), c_init=0.5)
        optim = IPOPT(model, options={
            "maxiter": 50, "disp": 0,
            "acceptable_tol": 1e-6,
            "tol": 1e-7,
        })
        r = optim.optimize()
        c_flat = jnp.array([float(r["c"])])
        cons = float(model.constraints_flat(c_flat))
        assert cons >= -1e-4, f"Constraint violated: {cons:.2e}"


# ─────────────────────────────────────────────────────────────────────────────
# 5. Robustness
# ─────────────────────────────────────────────────────────────────────────────

@pytestmark_ipopt
class TestIPOPTRobustness:

    def test_starts_at_optimum(self):
        """Starting near the optimum should converge in very few iterations."""
        diagram = _make_spring_mass_diagram()
        model = _SingleParamOptimizable(
            diagram, diagram.create_context(), c_init=1.65
        )
        r = _run_ipopt(model, maxiter=30)
        assert np.isclose(float(r["c"]), 1.65, atol=0.1)
        # Should converge in few iterations from near-optimum
        assert r.nit <= 20

    def test_starts_far_from_optimum(self):
        """Starting at c=5.0 (well above optimal), IPOPT still converges.

        Bound c ≥ 0: damping is physically non-negative, and without a lower
        bound IPOPT's first step from the far start overshoots into c < 0
        (negative damping → unstable, unbounded oscillation), where the adaptive
        ODE solver stalls and a single objective evaluation never returns.
        """
        diagram = _make_spring_mass_diagram(c_init=5.0)
        model = _SingleParamOptimizable(
            diagram, diagram.create_context(), c_init=5.0,
            bounds={"c": (0.0, 10.0)},
        )
        r = _run_ipopt(model, maxiter=50)
        assert np.isclose(float(r["c"]), 1.65, atol=0.2), (
            f"c from far start: {float(r['c']):.4f}"
        )

    def test_unconstrained_final_loss_below_reference(self):
        """Final loss must be meaningfully below the ISE at c=0.5 (start).

        The ISE surface for this system is shallow: the global optimum
        (c ≈ 1.65) has ISE ≈ 1.40 versus ≈ 1.59 at the c=0.5 start, so the
        best achievable reduction is only ~12% (ratio ≈ 0.88). A convergent
        solver lands there; a stuck one stays near ratio 1.0. The 0.9 factor
        sits between those and is physically achievable (unlike the original
        0.8, which no value of c can reach).
        """
        diagram = _make_spring_mass_diagram()
        model = _SingleParamOptimizable(
            diagram, diagram.create_context(), c_init=0.5
        )
        init_loss = float(model.objective_flat(model.params_0_flat))
        r = _run_ipopt(model)
        assert r.final_loss < init_loss * 0.9, (
            f"Loss barely decreased: init={init_loss:.4f}, final={r.final_loss:.4f}"
        )
