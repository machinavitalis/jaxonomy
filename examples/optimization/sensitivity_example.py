# SPDX-License-Identifier: MIT
"""
Parameter sensitivity / identifiability analysis example
=========================================================

Before fitting parameters, check whether the objective is actually *sensitive*
to each parameter.  Near-zero gradient → the simulation output barely changes
with that parameter → the parameter is unidentifiable from this data.

This example uses ``compute_sensitivity`` to:

1. Report the gradient and normalised sensitivity for each parameter.
2. Compute the Hessian (Fisher Information Matrix approximation).
3. Flag parameters with near-zero sensitivity.

Usage::

    python examples/optimization/sensitivity_example.py
"""

import numpy as np

from jaxonomy import DiagramBuilder, Parameter, SimulatorOptions
from jaxonomy.library import Adder, Gain, Integrator, Power
from jaxonomy.optimization import Optimizable
from jaxonomy.optimization.sensitivity import compute_sensitivity


# ── build diagram ─────────────────────────────────────────────────────────────

params = {
    "c": Parameter(np.array(0.5)),   # damping — identifiable
    "k": Parameter(np.array(1.0)),   # stiffness — identifiable
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


# ── define optimizable ────────────────────────────────────────────────────────

class SpringOptimizable(Optimizable):
    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self._obj = diagram["obj"].output_ports[0]

    def optimizable_params(self, ctx):
        return {"c": ctx.parameters["c"], "k": ctx.parameters["k"]}

    def objective_from_context(self, ctx):
        return self._obj.eval(ctx)

    def prepare_context(self, ctx, p):
        return ctx.with_parameters(p)


model = SpringOptimizable(
    diagram, diagram.create_context(),
    params_0={"c": 0.5, "k": 1.0},
    sim_t_span=(0.0, 2.0),
    sim_options=SimulatorOptions(max_major_steps=1),
)


# ── sensitivity analysis ──────────────────────────────────────────────────────

print("Computing sensitivity at initial parameters (c=0.5, k=1.0) ...")
sens = compute_sensitivity(model, sensitivity_threshold=1e-3)
print(sens.summary())

print("\n--- Raw values ---")
print(f"Gradients:             {sens.gradients}")
print(f"Normalised sensitivity:{sens.normalized_sensitivity}")
print(f"Hessian diagonal:      {sens.hessian_diagonal}")
print(f"Eigenvalues:           {sens.eigenvalues}")
print(f"Condition number:      {sens.condition_number:.3g}")

if sens.unidentifiable_params:
    print(f"\n⚠  The following parameters have low sensitivity and may be "
          f"unidentifiable:\n  {sens.unidentifiable_params}")
else:
    print("\n✓  All parameters appear identifiable at this operating point.")
