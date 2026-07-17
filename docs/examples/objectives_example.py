# SPDX-License-Identifier: MIT
"""
Objective function helpers — before / after comparison.

Shows how ``ise_objective``, ``lqr_objective``, ``tracking_mse``, and
``weighted_sum`` eliminate boilerplate in parameter-estimation diagrams.

Each section has a *before* (manual wiring) block and an *after* (helper)
block that produce numerically identical results.
"""

import numpy as np
import jax.numpy as jnp

import jaxonomy
from jaxonomy import DiagramBuilder, Parameter, SimulatorOptions
from jaxonomy.library import (
    Adder, Constant, Gain, Integrator, Power, SumOfElements,
)
from jaxonomy.optimization import (
    ise_objective, lqr_objective, tracking_mse, weighted_sum,
    Optimizable, Scipy,
)


T_END = 5.0
DT = 0.1

def _opts(t):
    from math import ceil
    n = ceil(t / DT)
    return SimulatorOptions(max_major_steps=20 * n, max_major_step_length=DT)


# ─────────────────────────────────────────────────────────────────────────────
# 1.  ISE objective  (before vs. after)
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 60)
print("1.  ISE objective — before vs after")
print("=" * 60)

# ── BEFORE: 10 lines of boilerplate ────────────────────────────────────────
b_before = DiagramBuilder()
v_src = b_before.add(Constant(1.0, name="v"))
x_src = b_before.add(Constant(2.0, name="x"))

ref_v = b_before.add(Constant(0.0, name="ref_v"))
ref_x = b_before.add(Constant(0.0, name="ref_x"))
err_v = b_before.add(Adder(2, operators="+-", name="err_v"))
err_x = b_before.add(Adder(2, operators="+-", name="err_x"))
sq_v  = b_before.add(Power(2.0, name="sq_v"));  sv = b_before.add(SumOfElements(name="sv"))
sq_x  = b_before.add(Power(2.0, name="sq_x"));  sx = b_before.add(SumOfElements(name="sx"))
cv    = b_before.add(Integrator(0.0, name="cv"))
cx    = b_before.add(Integrator(0.0, name="cx"))
obj_b = b_before.add(Adder(2, operators="++", name="obj"))
b_before.connect(ref_v.output_ports[0], err_v.input_ports[0])
b_before.connect(v_src.output_ports[0], err_v.input_ports[1])
b_before.connect(ref_x.output_ports[0], err_x.input_ports[0])
b_before.connect(x_src.output_ports[0], err_x.input_ports[1])
b_before.connect(err_v.output_ports[0], sq_v.input_ports[0]);  b_before.connect(sq_v.output_ports[0], sv.input_ports[0])
b_before.connect(err_x.output_ports[0], sq_x.input_ports[0]);  b_before.connect(sq_x.output_ports[0], sx.input_ports[0])
b_before.connect(sv.output_ports[0], cv.input_ports[0])
b_before.connect(sx.output_ports[0], cx.input_ports[0])
b_before.connect(cv.output_ports[0], obj_b.input_ports[0])
b_before.connect(cx.output_ports[0], obj_b.input_ports[1])
diag_before = b_before.build()

sol_before = jaxonomy.simulate(diag_before, diag_before.create_context(),
                               (0.0, T_END), options=_opts(T_END),
                               recorded_signals={"J": obj_b.output_ports[0]})

# ── AFTER: 3 lines ──────────────────────────────────────────────────────────
b_after = DiagramBuilder()
v_a = b_after.add(Constant(1.0, name="v"))
x_a = b_after.add(Constant(2.0, name="x"))
cost_v = ise_objective(b_after, v_a.output_ports[0], name="ise_v")
cost_x = ise_objective(b_after, x_a.output_ports[0], name="ise_x")
obj_port = weighted_sum(b_after, [cost_v, cost_x])
diag_after = b_after.build()

sol_after = jaxonomy.simulate(diag_after, diag_after.create_context(),
                              (0.0, T_END), options=_opts(T_END),
                              recorded_signals={"J": obj_port})

J_before = float(sol_before.outputs["J"][-1])
J_after  = float(sol_after.outputs["J"][-1])
print(f"  J (before, manual):  {J_before:.6f}")
print(f"  J (after,  helpers): {J_after:.6f}")
print(f"  Max absolute difference: {abs(J_before - J_after):.2e}")


# ─────────────────────────────────────────────────────────────────────────────
# 2.  LQR objective
# ─────────────────────────────────────────────────────────────────────────────
print()
print("=" * 60)
print("2.  LQR quadratic objective")
print("=" * 60)

Q = jnp.diag(jnp.array([10.0, 1.0]))  # state weight
R = jnp.array([[0.1]])                  # control weight

b = DiagramBuilder()
x_src = b.add(Constant(jnp.array([1.0, 0.5]), name="x"))  # state = [1, 0.5]
u_src = b.add(Constant(jnp.array([2.0]),       name="u"))  # control = [2]
cost  = lqr_objective(b, x_src.output_ports[0], Q,
                      control_port=u_src.output_ports[0], R=R,
                      name="lqr_cost")
diag = b.build()
sol  = jaxonomy.simulate(diag, diag.create_context(), (0.0, T_END),
                         options=_opts(T_END),
                         recorded_signals={"J": cost})

J_lqr = float(sol.outputs["J"][-1])
# Analytical: x^T Q x = 10*1 + 1*0.25 = 10.25;  u^T R u = 0.1*4 = 0.4
# Total integrand = 10.65; J = 10.65 * T_END
analytic = (10.0 * 1.0 + 1.0 * 0.25 + 0.1 * 4.0) * T_END
print(f"  Q = diag([10, 1]),  R = [[0.1]]")
print(f"  x = [1, 0.5],  u = [2]")
print(f"  x^T Q x + u^T R u = {10.0*1.0 + 1.0*0.25 + 0.1*4.0:.4f}")
print(f"  J (simulated): {J_lqr:.4f}")
print(f"  J (analytic):  {analytic:.4f}")


# ─────────────────────────────────────────────────────────────────────────────
# 3.  tracking_mse against a reference dataset
# ─────────────────────────────────────────────────────────────────────────────
print()
print("=" * 60)
print("3.  Dataset tracking MSE")
print("=" * 60)

# Reference dataset: constant y_ref = 1.0 (e.g. steady-state target)
t_data = np.linspace(0, T_END, 100)
y_data = np.ones(100)  # reference = 1

# Two signals: "wrong" (signal = 2) and "right" (signal = 1)
b_wrong = DiagramBuilder()
sig_wrong = b_wrong.add(Constant(2.0, name="sig"))   # error = 1
cost_wrong = tracking_mse(b_wrong, sig_wrong.output_ports[0], t_data, y_data, name="track")
d_wrong = b_wrong.build()

b_right = DiagramBuilder()
sig_right = b_right.add(Constant(1.0, name="sig"))   # perfect match, error = 0
cost_right = tracking_mse(b_right, sig_right.output_ports[0], t_data, y_data, name="track")
d_right = b_right.build()

sol_wrong = jaxonomy.simulate(d_wrong, d_wrong.create_context(), (0.0, T_END),
                              options=_opts(T_END), recorded_signals={"J": cost_wrong})
sol_right = jaxonomy.simulate(d_right, d_right.create_context(), (0.0, T_END),
                              options=_opts(T_END), recorded_signals={"J": cost_right})

J_wrong = float(sol_wrong.outputs["J"][-1])
J_right = float(sol_right.outputs["J"][-1])
print(f"  Reference: y_ref(t) = 1.0  (constant dataset of 100 points)")
print(f"  J (signal=2, error=1):  {J_wrong:.4f}   ≈ T = {T_END}")
print(f"  J (signal=1, error=0):  {J_right:.2e}   ≈ 0")
print(f"  J_wrong > J_right: {J_wrong > J_right}")


# ─────────────────────────────────────────────────────────────────────────────
# 4.  weighted_sum combining multiple objectives
# ─────────────────────────────────────────────────────────────────────────────
print()
print("=" * 60)
print("4.  weighted_sum of multiple objectives")
print("=" * 60)

b = DiagramBuilder()
s_a = b.add(Constant(1.0, name="a"))
s_b = b.add(Constant(2.0, name="b"))
s_c = b.add(Constant(3.0, name="c"))

j_a = ise_objective(b, s_a.output_ports[0], name="j_a")  # ∫ 1 dt = T
j_b = ise_objective(b, s_b.output_ports[0], name="j_b")  # ∫ 4 dt = 4T
j_c = ise_objective(b, s_c.output_ports[0], name="j_c")  # ∫ 9 dt = 9T

total = weighted_sum(b, [j_a, j_b, j_c], weights=[1.0, 2.0, 0.5],
                     name="total")
# Expected: ∫ (1·1 + 2·4 + 0.5·9) dt = ∫ 13.5 dt = 13.5·T

diag = b.build()
sol = jaxonomy.simulate(diag, diag.create_context(), (0.0, T_END),
                        options=_opts(T_END),
                        recorded_signals={"J": total})
J_total = float(sol.outputs["J"][-1])
expected = (1*1 + 2*4 + 0.5*9) * T_END
print(f"  weights = [1.0, 2.0, 0.5]")
print(f"  signals = [1², 2², 3²] = [1, 4, 9]")
print(f"  J (simulated): {J_total:.4f}")
print(f"  J (analytic):  {expected:.4f}")

print("\nDone.")
