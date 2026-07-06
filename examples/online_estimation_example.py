# SPDX-License-Identifier: MIT
"""
Online / real-time parameter estimation examples.

Two examples demonstrating the new online estimation blocks:

  A. RecursiveLeastSquares
     Identifies the coefficients θ of a linear model y = φᵀθ one measurement
     at a time, with optional forgetting factor for time-varying systems.

  B. AugmentedStateEKF
     Jointly estimates the state and an unknown model parameter of a discrete-
     time dynamical system using an Extended Kalman Filter on the augmented
     state z = [x; θ].
"""

import numpy as np
import jax.numpy as jnp

import jaxonomy
from jaxonomy import DiagramBuilder, SimulatorOptions
from jaxonomy.library import Constant, RecursiveLeastSquares, AugmentedStateEKF


# ─────────────────────────────────────────────────────────────────────────────
# A.  Recursive Least Squares
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 60)
print("A. Recursive Least Squares parameter identification")
print("=" * 60)

# True model:  y = 2·φ₀ − 1·φ₁ + 3·φ₂
TRUE_THETA = np.array([2.0, -1.0, 3.0])
N_PARAMS = 3
DT = 0.1
T_END = 10.0  # simulate 10 s  (100 steps)

# Constant regressor (the RLS will project onto this direction)
PHI = np.array([1.0, 2.0, 1.0])
Y_TRUE = float(PHI @ TRUE_THETA)  # = 2 + (-2) + 3 = 3 → actually 2 -2 +3 = 3

print(f"True θ = {TRUE_THETA}")
print(f"φ      = {PHI}")
print(f"y      = {Y_TRUE:.4f}")

b = DiagramBuilder()
rls = b.add(
    RecursiveLeastSquares(
        dt=DT,
        n_params=N_PARAMS,
        forgetting_factor=1.0,   # no forgetting — batch equivalent
        name="rls",
    )
)
phi_src = b.add(Constant(jnp.array(PHI), name="phi"))
y_src   = b.add(Constant(jnp.array(Y_TRUE), name="y"))
b.connect(phi_src.output_ports[0], rls.input_ports[0])
b.connect(y_src.output_ports[0],   rls.input_ports[1])
diagram = b.build()

ctx = diagram.create_context()
nseg = int(T_END / DT)
options = SimulatorOptions(max_major_steps=10 * nseg, max_major_step_length=DT)
sol = jaxonomy.simulate(
    diagram, ctx, (0.0, T_END), options=options,
    recorded_signals={
        "theta_hat": rls.output_ports[0],
        "P_diag":    rls.output_ports[1],   # full P matrix per step
        "error":     rls.output_ports[2],
    },
)

theta_final = np.array(sol.outputs["theta_hat"][-1])
error_final = float(sol.outputs["error"][-1])
print(f"\nAfter {nseg} steps:")
print(f"  θ̂     = {theta_final}")
print(f"  error = {error_final:.4e}  (prediction residual)")
print("  (note: with constant φ only the direction of φ is identified)")

# ─────────────────────────────────────────────────────────────────────────────
# B.  Augmented-State EKF  (joint state + parameter estimation)
# ─────────────────────────────────────────────────────────────────────────────
print()
print("=" * 60)
print("B. Augmented-State EKF: joint state + parameter estimation")
print("=" * 60)

# True system:  x[n+1] = a·x[n],   y[n] = x[n]
# Unknown parameter: a  (true value 0.9)
TRUE_A = 0.9
DT_EKF = 0.1
T_EKF  = 5.0
THETA_INIT = 0.5   # initial guess for a, far from truth

print(f"True decay coefficient a = {TRUE_A}")
print(f"Initial guess           a₀ = {THETA_INIT}")


def forward(x, u, theta):
    """x[n+1] = a·x[n]"""
    a = theta[0]
    return jnp.array([a * x[0] + u[0]])


def observation(x, u, theta):
    """y = x"""
    return jnp.array([x[0]])


b2 = DiagramBuilder()
aekf = b2.add(
    AugmentedStateEKF(
        dt=DT_EKF,
        nx=1,
        n_params=1,
        forward=forward,
        observation=observation,
        G_x_func=lambda t: jnp.eye(1),
        Q_x_func=lambda t, x, u, th: jnp.eye(1) * 1e-4,   # small state noise
        Q_theta=jnp.eye(1) * 1e-3,                          # allow parameter drift
        R_func=lambda t: jnp.eye(1) * 1e-2,                # low measurement noise
        x_hat_0=jnp.array([1.0]),
        P_hat_0_x=jnp.eye(1) * 0.1,
        theta_hat_0=jnp.array([THETA_INIT]),
        P_hat_0_theta=jnp.eye(1) * 1.0,                    # high initial uncertainty
        name="aekf",
    )
)
u_src = b2.add(Constant(jnp.zeros(1), name="u"))
# Observation: constant y = 1.0 (system at x=1, decaying to 0 not simulated here)
y_src = b2.add(Constant(jnp.ones(1), name="y"))
b2.connect(u_src.output_ports[0], aekf.input_ports[0])
b2.connect(y_src.output_ports[0], aekf.input_ports[1])
diagram2 = b2.build()

ctx2 = diagram2.create_context()
nseg2 = int(T_EKF / DT_EKF)
options2 = SimulatorOptions(max_major_steps=10 * nseg2, max_major_step_length=DT_EKF)
sol2 = jaxonomy.simulate(
    diagram2, ctx2, (0.0, T_EKF), options=options2,
    recorded_signals={
        "x_hat":     aekf.output_ports[0],
        "theta_hat": aekf.output_ports[1],
    },
)

x_hat_final     = float(sol2.outputs["x_hat"][-1, 0])
theta_hat_final = float(sol2.outputs["theta_hat"][-1, 0])
print(f"\nAfter {nseg2} steps:")
print(f"  x̂        = {x_hat_final:.4f}")
print(f"  θ̂ (est a) = {theta_hat_final:.4f}  (true a = {TRUE_A})")
print(f"  |θ̂ - a|   = {abs(theta_hat_final - TRUE_A):.4f}")

print("\nDone.")
