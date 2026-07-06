# SPDX-License-Identifier: MIT
"""
Parameter confidence intervals after system identification.

This example shows how to quantify parameter uncertainty using the Laplace
approximation after fitting a spring-mass-damper model to simulated data.

Steps:
  1. Define a spring-mass system and an Optimizable.
  2. Run L-BFGS-B to find the best-fit damping coefficient *c*.
  3. Call ``compute_confidence_intervals`` to get the 95 % CI.
  4. Print the result table and examine the covariance.

Theory:
  At the optimum θ* the loss L(θ) is locally quadratic:
      L(θ) ≈ L(θ*) + ½ (θ - θ*)ᵀ H (θ - θ*)
  Inverting H gives the parameter covariance Cov ≈ H⁻¹.
  The 95 % CI for parameter i is  θ*_i ± 1.96 × sqrt(Cov_ii).
"""

import numpy as np
import jax.numpy as jnp

import jaxonomy
from jaxonomy import DiagramBuilder, Parameter, SimulatorOptions
from jaxonomy.library import Adder, Gain, Integrator, Power
from jaxonomy.optimization import (
    Optimizable,
    Scipy,
    compute_confidence_intervals,
    compute_sensitivity,
)


# ── 1. Build the spring-mass-damper diagram ───────────────────────────────────
# Equation:  ẍ + c·ẋ + k·x = 0,  x(0)=1, ẋ(0)=0
# Objective: ∫₀ᵀ [x(t)² + ẋ(t)²] dt   (minimise oscillation energy)

def make_diagram(c_init: float, k: float = 1.0):
    params = {"c": Parameter(np.array(c_init))}
    b = DiagramBuilder()

    k_x  = b.add(Gain(k,            name="k_x"))
    c_v  = b.add(Gain(params["c"],  name="c_v"))
    add  = b.add(Adder(2, operators="--", name="add"))
    inv  = b.add(Gain(1.0,          name="inv_m"))
    v    = b.add(Integrator(0.0,    name="v"))      # ẋ(0) = 0
    x    = b.add(Integrator(1.0,    name="x"))      # x(0) = 1

    b.connect(k_x.output_ports[0],  add.input_ports[0])
    b.connect(c_v.output_ports[0],  add.input_ports[1])
    b.connect(add.output_ports[0],  inv.input_ports[0])
    b.connect(inv.output_ports[0],  v.input_ports[0])
    b.connect(v.output_ports[0],    x.input_ports[0])
    b.connect(v.output_ports[0],    c_v.input_ports[0])
    b.connect(x.output_ports[0],    k_x.input_ports[0])

    sq_v = b.add(Power(2.0, name="sq_v"))
    sq_x = b.add(Power(2.0, name="sq_x"))
    cv   = b.add(Integrator(0.0, name="cv"))
    cx   = b.add(Integrator(0.0, name="cx"))
    obj  = b.add(Adder(2, operators="++", name="obj"))

    b.connect(v.output_ports[0],    sq_v.input_ports[0])
    b.connect(x.output_ports[0],    sq_x.input_ports[0])
    b.connect(sq_v.output_ports[0], cv.input_ports[0])
    b.connect(sq_x.output_ports[0], cx.input_ports[0])
    b.connect(cv.output_ports[0],   obj.input_ports[0])
    b.connect(cx.output_ports[0],   obj.input_ports[1])

    return b.build(parameters=params), params


# ── 2. Define the Optimizable ─────────────────────────────────────────────────

class SpringMassOpt(Optimizable):
    def __init__(self, diagram, params, c_init: float):
        self._obj_port = diagram["obj"].output_ports[0]
        super().__init__(
            diagram, diagram.create_context(),
            params_0={"c": c_init},
            sim_t_span=(0.0, 5.0),
            sim_options=SimulatorOptions(max_major_steps=1),
        )

    def optimizable_params(self, ctx):
        return {"c": ctx.parameters["c"]}

    def objective_from_context(self, ctx):
        return self._obj_port.eval(ctx)

    def prepare_context(self, ctx, p):
        return ctx.with_parameters(p)


# ── 3. Optimise ───────────────────────────────────────────────────────────────

print("=" * 60)
print("Setting up spring-mass-damper optimisation …")
diagram, params = make_diagram(c_init=0.5)
opt = SpringMassOpt(diagram, params, c_init=0.5)

scipy_opt = Scipy(opt, "L-BFGS-B", use_autodiff_grad=True,
                  opt_method_config={"maxiter": 100, "ftol": 1e-10})
result = scipy_opt.optimize()

print(f"Optimisation finished: success={result.success}, nit={result.nit}")
print(f"  c* = {result['c']:.6f}  (optimal damping)")
print(f"  L* = {result.final_loss:.6g}")

# ── 4. Sensitivity check (pre-requisite: are we actually at the minimum?) ─────

print("\nSensitivity check …")
sens = compute_sensitivity(opt, params_0_flat=jnp.array([result["c"]]))
print(sens.summary())

# ── 5. Confidence intervals ───────────────────────────────────────────────────

print("\nComputing 95 % confidence intervals …")
ci = compute_confidence_intervals(opt, result, confidence_level=0.95)
print(ci.summary())

# ── 6. Also show 90 % and 99 % ────────────────────────────────────────────────

for level in [0.90, 0.99]:
    ci_l = compute_confidence_intervals(opt, result, confidence_level=level)
    lo, hi = ci_l.interval("c")
    print(f"  {int(level*100):3d}% CI for c:  [{lo:.4f},  {hi:.4f}]   "
          f"width={hi-lo:.4f}")

# ── 7. With n_data for least-squares objective ────────────────────────────────

print("\nLeast-squares scaling with n_data=50 observations …")
ci_ls = compute_confidence_intervals(opt, result, confidence_level=0.95, n_data=50)
lo_ls, hi_ls = ci_ls.interval("c")
print(f"  Residual variance σ²: {ci_ls.residual_variance:.4g}")
print(f"  95% CI for c (LS):   [{lo_ls:.4f},  {hi_ls:.4f}]")

# ── 8. Covariance matrix ──────────────────────────────────────────────────────

print("\nCovariance matrix (1×1 for single parameter):")
print(f"  Var(c) = {ci.covariance[0,0]:.4g}")
print(f"  SE(c)  = {ci.standard_errors[0]:.4g}")

print("\nDone.")
