# SPDX-License-Identifier: MIT
"""
Multi-start optimization example
=================================

Demonstrates how to use ``MultiStart`` to escape local minima when fitting
a non-convex objective.

Problem
-------
Damped spring-mass system.  ISE objective in c (damping coefficient).
The true optimum is c ≈ 1.65.  Starting far from it (e.g. c = 3.0) causes
gradient-based methods to converge slowly — multi-start with random restarts
reliably finds the global minimum.

Usage::

    python docs/examples/optimization/multi_start_example.py
"""

import numpy as np

from jaxonomy import DiagramBuilder, Parameter, SimulatorOptions
from jaxonomy.library import Adder, Gain, Integrator, Power
from jaxonomy.optimization import Optimizable, Scipy, MultiStart

# ── build diagram ─────────────────────────────────────────────────────────────

params = {"c": Parameter(np.array(1.0))}
b = DiagramBuilder()
k_x = b.add(Gain(1.0,         name="k_x"))
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


# ── define optimizable ────────────────────────────────────────────────────────

class SpringOptimizable(Optimizable):
    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self._obj = diagram["obj"].output_ports[0]

    def optimizable_params(self, ctx):
        return {"c": ctx.parameters["c"]}

    def objective_from_context(self, ctx):
        return self._obj.eval(ctx)

    def prepare_context(self, ctx, p):
        return ctx.with_parameters(p)


model = SpringOptimizable(
    diagram, diagram.create_context(),
    params_0={"c": 0.5},
    sim_t_span=(0.0, 2.0),
    sim_options=SimulatorOptions(max_major_steps=1),
)


# ── single start (for comparison) ────────────────────────────────────────────

single_optim = Scipy(model, "L-BFGS-B",
                     opt_method_config={"maxiter": 40},
                     use_autodiff_grad=True)
single_result = single_optim.optimize()
print("=== Single-start result ===")
print(f"  c = {float(single_result['c']):.4f}  (expected ≈ 1.65)")
print(f"  success={single_result.success}, nit={single_result.nit}, "
      f"final_loss={single_result.final_loss:.6g}")


# ── multi-start ───────────────────────────────────────────────────────────────

def factory(opt):
    return Scipy(opt, "L-BFGS-B",
                 opt_method_config={"maxiter": 40},
                 use_autodiff_grad=True)


ms = MultiStart(
    model,
    factory,
    n_starts=6,
    sample_scale=1.5,   # search window: ±1.5×|c0|
    seed=42,
    include_initial=True,
)

ms_result = ms.run()
print("\n=== Multi-start result ===")
print(ms_result.summary())
print(f"\nBest c = {float(ms_result.best_result['c']):.4f}  (expected ≈ 1.65)")
