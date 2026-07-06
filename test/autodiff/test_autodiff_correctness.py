# SPDX-License-Identifier: MIT
"""
Correctness tests for jaxonomy autodiff across all major system types.

Each test runs a simulation and verifies that automatic differentiation
(via JAX adjoint / custom VJP) matches an analytic formula or a finite-
difference ground truth.  System taxonomy covered:

  CT    – causal continuous-time ODE
  DT    – causal discrete-time  (hybrid CT+DT; pure-DT is noted as unsupported)
  AC    – acausal (Modelica-style DAE compiled by AcausalCompiler)
  SM    – state-machine / mode-switching with zero-crossing events
  NN    – neural-network blocks (equinox MLP in a feedback ODE)
  HYB   – hybrid combinations of the above

Known limitations documented as skipped/xfail markers:
  - Pure discrete-time systems (no CT state): autodiff VJP pytree mismatch
  - Acausal DAE + separate causal CT integrator: adjoint init not implemented
  - Adjoint accuracy note: rtol=1e-8 is non-monotone for Dopri5 adjoint;
    use rtol=1e-6 (default) for reliable gradients.
"""

from enum import IntEnum
from functools import partial

import numpy as np
import pytest
import jax
import jax.numpy as jnp
import equinox as eqx

import jaxonomy
from jaxonomy.library import (
    Integrator,
    Gain,
    Adder,
    Demultiplexer,
)
from jaxonomy.testing import fd_grad
from jaxonomy.testing.markers import skip_if_not_jax

skip_if_not_jax()
pytestmark = pytest.mark.slow

# ── helpers ──────────────────────────────────────────────────────────────────

# NOTE: rtol=1e-6 (not 1e-8) is intentional.  The Dopri5 adjoint ODE solver
# has a non-monotone accuracy response; rtol=1e-8 can produce ~3% gradient
# error while rtol=1e-6 and rtol=1e-10 both converge correctly.
_OPTS = jaxonomy.SimulatorOptions(
    math_backend="jax",
    enable_autodiff=True,
)
_OPTS_BDF = jaxonomy.SimulatorOptions(
    math_backend="jax",
    enable_autodiff=True,
    ode_solver_method="bdf",
)


# ═══════════════════════════════════════════════════════════════════════════
# CT  – causal continuous-time ODE
# ═══════════════════════════════════════════════════════════════════════════


def test_ct_scalar_param_sensitivity():
    """CT – scalar exponential decay: dx/dt = -a·x.

    Exact:  x(T) = x0·exp(-a·T)

    Verified gradients:
        ∂x(T)/∂x0  = exp(-a·T)
        ∂x(T)/∂a   = -T·x0·exp(-a·T)
    """
    a_val, x0_val, T = 1.5, 4.0, 2.0

    class ScalarDecay(jaxonomy.LeafSystem):
        def __init__(self):
            super().__init__()
            self.declare_dynamic_parameter("a", a_val)
            self.declare_continuous_state(default_value=jnp.array(x0_val), ode=self._ode)

        def _ode(self, time, state, **params):
            return -params["a"] * state.continuous_state

    sys = ScalarDecay()
    ctx = sys.create_context()

    @jax.jit
    def fwd(x0, a, context):
        context = context.with_continuous_state(x0)
        context.parameters["a"] = a
        return jaxonomy.simulate(sys, context, (0.0, T), options=_OPTS).context.continuous_state

    x0 = jnp.array(x0_val)
    a = jnp.array(a_val)
    xf = fwd(x0, a, ctx)
    assert jnp.allclose(xf, x0 * jnp.exp(-a * T), rtol=1e-5), f"forward: {xf}"

    dx0, da = jax.jit(jax.grad(fwd, argnums=(0, 1)))(x0, a, ctx)
    assert jnp.allclose(dx0, jnp.exp(-a * T), rtol=1e-4), f"∂x/∂x0: {dx0}"
    assert jnp.allclose(da, -T * x0 * jnp.exp(-a * T), rtol=1e-4), f"∂x/∂a: {da}"


def test_ct_harmonic_oscillator_jacobian():
    """CT – 2-D harmonic oscillator: [x', v'] = [v, -ω²x].

    Analytic state-transition Jacobian at time T:
        Φ(T) = [[cos(ωT),      sin(ωT)/ω],
                [-ω·sin(ωT),   cos(ωT)  ]]

    Tests that jax.jacobian through simulate returns this matrix.
    """

    class HarmonicOscillator(jaxonomy.LeafSystem):
        def __init__(self, omega=2.0):
            super().__init__()
            self.declare_dynamic_parameter("omega", omega)
            self.declare_continuous_state(default_value=jnp.zeros(2), ode=self._ode)

        def _ode(self, time, state, **params):
            x, v = state.continuous_state
            w = params["omega"]
            return jnp.array([v, -(w**2) * x])

    omega, T = 2.0, 0.75
    sys = HarmonicOscillator(omega=omega)
    ctx = sys.create_context()

    @jax.jit
    def fwd(x0, context):
        context = context.with_continuous_state(x0)
        return jaxonomy.simulate(sys, context, (0.0, T), options=_OPTS).context.continuous_state

    x0 = jnp.array([1.5, -0.5])
    Phi_ad = jax.jit(jax.jacobian(fwd))(x0, ctx)

    Phi_ana = jnp.array(
        [
            [np.cos(omega * T), np.sin(omega * T) / omega],
            [-omega * np.sin(omega * T), np.cos(omega * T)],
        ]
    )
    assert jnp.allclose(Phi_ad, Phi_ana, atol=1e-4), f"Φ error:\n{Phi_ad - Phi_ana}"


def test_ct_linearize_second_order_system():
    """CT – linearize a 2nd-order ODE about (x=0, v=0).

    Plant:  dx/dt = v,  dv/dt = -k·x - b·v + F
    Input:  F (scalar force)
    Output: full state [x, v] (identity C matrix)

    Exact state-space:
        A = [[0, 1], [-k, -b]]
        B = [[0], [1]]
        C = [[1, 0], [0, 1]]   (full state output)
        D = [[0], [0]]
    """
    k, b = 4.0, 0.5

    class SecondOrder(jaxonomy.LeafSystem):
        def __init__(self):
            super().__init__(name="plant")
            self.declare_dynamic_parameter("k", k)
            self.declare_dynamic_parameter("b", b)
            self.declare_input_port(name="F")
            self.declare_continuous_state(default_value=jnp.zeros(2), ode=self._ode)
            self.declare_continuous_state_output()  # outputs full state → C = I

        def _ode(self, time, state, F, **params):
            x, v = state.continuous_state
            return jnp.array([v, -params["k"] * x - params["b"] * v + F[0]])

    plant = SecondOrder()
    plant.input_ports[0].fix_value(jnp.zeros(1))
    ctx = plant.create_context()

    lin = jaxonomy.linearize(plant, ctx).to_lti()
    A = lin.dynamic_parameters["A"].value
    B = lin.dynamic_parameters["B"].value
    C = lin.dynamic_parameters["C"].value
    D = lin.dynamic_parameters["D"].value

    A_ana = jnp.array([[0.0, 1.0], [-k, -b]])
    B_ana = jnp.array([[0.0], [1.0]])
    C_ana = jnp.eye(2)           # full state output
    D_ana = jnp.zeros((2, 1))

    assert jnp.allclose(A, A_ana, atol=1e-6), f"A={A}, expected\n{A_ana}"
    assert jnp.allclose(B, B_ana, atol=1e-6), f"B={B}"
    assert jnp.allclose(C, C_ana, atol=1e-6), f"C={C}"
    assert jnp.allclose(D, D_ana, atol=1e-6), f"D={D}"

    # Eigenvalues of a stable spring-damper should have negative real parts
    # (call eigenvalues on the LinearizedSystem before converting to LTISystem)
    lin_sys = jaxonomy.linearize(plant, ctx)
    eigs = lin_sys.eigenvalues()
    assert all(jnp.real(e) < 0 for e in eigs), f"eigenvalues not stable: {eigs}"


def test_ct_nonlinear_vector_param_grad():
    """CT – nonlinear vector ODE: falling body with quadratic drag.

    dx/dt = v,   dv/dt = -g - a·v²·sign(v)

    Tests ∂x_final/∂x0, ∂x_final/∂g, ∂x_final/∂a against finite differences.
    """

    class FallingBody(jaxonomy.LeafSystem):
        def __init__(self):
            super().__init__()
            self.declare_dynamic_parameter("g", 9.81)
            self.declare_dynamic_parameter("a", 0.1)
            self.declare_continuous_state(default_value=jnp.zeros(2), ode=self._ode)

        def _ode(self, time, state, **params):
            x, v = state.continuous_state
            g, a = params["g"], params["a"]
            return jnp.array([v, -g - a * v**2 * jnp.sign(v)])

    g0, a0 = 9.81, 0.1
    x0 = jnp.array([0.0, 10.0])
    sys = FallingBody()
    ctx = sys.create_context()

    @jax.jit
    def fwd(x0_, g_, a_, context):
        context = context.with_continuous_state(x0_)
        context.parameters["g"] = g_
        context.parameters["a"] = a_
        return jaxonomy.simulate(sys, context, (0.0, 0.5), options=_OPTS).context.continuous_state[0]

    # Autodiff gradients
    dx0_ad, dg_ad, da_ad = jax.jit(jax.grad(fwd, argnums=(0, 1, 2)))(x0, g0, a0, ctx)

    # Finite-difference ground truth (scalar parameters)
    eps = 1e-4
    dg_fd = (fwd(x0, g0 + eps, a0, ctx) - fwd(x0, g0 - eps, a0, ctx)) / (2 * eps)
    da_fd = (fwd(x0, g0, a0 + eps, ctx) - fwd(x0, g0, a0 - eps, ctx)) / (2 * eps)

    # FD for vector x0
    dx0_fd = jnp.array([
        (fwd(x0.at[i].add(eps), g0, a0, ctx) - fwd(x0.at[i].add(-eps), g0, a0, ctx)) / (2 * eps)
        for i in range(len(x0))
    ])

    assert jnp.allclose(dx0_ad, dx0_fd, rtol=1e-3), f"dx/dx0: ad={dx0_ad}, fd={dx0_fd}"
    assert jnp.allclose(dg_ad, dg_fd, rtol=1e-3), f"dx/dg: ad={dg_ad:.6f}, fd={dg_fd:.6f}"
    assert jnp.allclose(da_ad, da_fd, rtol=1e-3), f"dx/da: ad={da_ad:.6f}, fd={da_fd:.6f}"


# ═══════════════════════════════════════════════════════════════════════════
# DT  – causal discrete-time
# ═══════════════════════════════════════════════════════════════════════════


def test_dt_pure_discrete_autodiff():
    """DT – pure discrete-time system: gradient works after pytree-unravel fix.

    A LeafSystem with only discrete state (no CT state) previously caused a
    ``TypeError: pytree mismatch`` in JAX's VJP machinery because the ODE solver
    initializer was wrapped with a nested custom VJP that re-called ravel_pytree
    on the empty continuous state, producing two non-equal unravel callables
    stored as pytree aux_data.  The fix skips the custom VJP wrapping when there
    is no continuous state.

    Model: x[n+1] = a * x[n],  period=1 s, simulated for T=3 s.
    Three updates fire (at t=0, 1, 2), so x[3] = a³·x₀.

    Verified gradients:
        ∂x[3]/∂x₀ = a³ = 8.0   (a=2, x₀=1)
    """

    class PureLinearMap(jaxonomy.LeafSystem):
        """x[n+1] = a * x[n]"""

        def __init__(self, period=1.0):
            super().__init__()
            self.declare_dynamic_parameter("a", 2.0)
            self.declare_discrete_state(default_value=jnp.array(1.0))
            self.declare_periodic_update(self._update, period=period, offset=0.0)

        def _update(self, time, state, **params):
            return state.discrete_state * params["a"]

    a_val, x0_val, N = 2.0, 1.0, 3
    sys = PureLinearMap(period=1.0)
    ctx = sys.create_context()
    opts = jaxonomy.SimulatorOptions(
        math_backend="jax", enable_autodiff=True, max_major_steps=20
    )

    @jax.jit
    def fwd(x0):
        ctx2 = ctx.with_discrete_state(x0)
        res = jaxonomy.simulate(sys, ctx2, (0.0, float(N)), options=opts)
        return res.context.discrete_state

    dx0 = float(jax.grad(fwd)(jnp.array(x0_val)))
    expected = a_val ** N  # 8.0
    assert abs(dx0 - expected) < 1e-4, f"∂x[N]/∂x₀={dx0:.4f}, expected {expected:.4f}"


class _Staircase(jaxonomy.LeafSystem):
    """Periodic DT block: x[n+1] = x[n] + 1,  y[n] = x[n]."""

    def __init__(self, period=1.0, name="staircase"):
        super().__init__(name=name)
        self.declare_discrete_state(default_value=jnp.array(0.0))
        self.declare_output_port(
            self._output, period=period, offset=0.0, name="y"
        )
        self.declare_periodic_update(self._update, period=period, offset=0.0)

    def _update(self, time, state, *inputs):
        return state.discrete_state + 1.0

    def _output(self, time, state, *inputs):
        return state.discrete_state


def test_dt_hybrid_staircase_grad():
    """DT – hybrid CT+DT: staircase-driven linear ODE.

    dx/dt = -a*(x - u[n]),  u[n] = n  (staircase, n = step count)

    Recursive exact solution over N steps of length τ:
        x[k] = x[k-1]*exp(-a*τ) + (k-1)*(1 - exp(-a*τ))

    Exact gradient w.r.t. x[0]:
        ∂x[N]/∂x[0] = exp(-a*N*τ)
    """
    a, tau, N = 1.5, 1.0, 3

    builder = jaxonomy.DiagramBuilder()
    gain = builder.add(Gain(-a))
    integ = builder.add(Integrator(0.0))
    adder = builder.add(Adder(2, operators="+-"))
    stair = builder.add(_Staircase(period=tau))

    builder.connect(integ.output_ports[0], adder.input_ports[0])
    builder.connect(stair.output_ports[0], adder.input_ports[1])
    builder.connect(adder.output_ports[0], gain.input_ports[0])
    builder.connect(gain.output_ports[0], integ.input_ports[0])

    diagram = builder.build()
    ctx = diagram.create_context()

    opts = jaxonomy.SimulatorOptions(
        math_backend="jax",
        enable_autodiff=True,
        max_major_steps=100,
    )

    @jax.jit
    def fwd(x0, context):
        int_ctx = context[integ.system_id].with_continuous_state(x0)
        context = context.with_subcontext(integ.system_id, int_ctx)
        return jaxonomy.simulate(diagram, context, (0.0, float(N * tau)), options=opts).context[
            integ.system_id
        ].continuous_state

    @partial(jax.jit, static_argnums=(1,))
    def exact_fwd(x0, n):
        x = x0
        for k in range(n):
            x = x * jnp.exp(-a * tau) + k * (1 - jnp.exp(-a * tau))
        return x

    x0 = jnp.array(4.0)
    assert jnp.allclose(fwd(x0, ctx), exact_fwd(x0, N), rtol=1e-3)

    dxf = jax.jit(jax.grad(fwd))(x0, ctx)
    assert jnp.allclose(dxf, jnp.exp(-a * N * tau), rtol=1e-3), f"∂xf/∂x0={dxf}"


def test_dt_hybrid_param_grad():
    """DT – hybrid CT+DT: gradient w.r.t. initial state in a CT+DT diagram.

    Architecture: CT integrator (dx/dt = k*x) driven by a Product block whose
    second input comes from a DT block that outputs a constant gain -b.
    Effective ODE: dx/dt = -b*x.

    Exact: x(T) = x0*exp(-b*T)
    ∂x(T)/∂x0 = exp(-b*T)

    Note: The gradient ∂x(T)/∂b through a DT-updated parameter is not
    propagated correctly by the adjoint (the discrete update path is treated
    as constant during adjoint), so we only verify ∂x(T)/∂x0 here.
    """
    from jaxonomy.library import Product

    b_val, T = 2.0, 1.5

    class ConstDT(jaxonomy.LeafSystem):
        """Outputs a constant -b, updated each period."""
        def __init__(self, period=0.5):
            super().__init__()
            self.declare_dynamic_parameter("b", b_val)
            self.declare_discrete_state(default_value=jnp.array(-b_val))
            self.declare_output_port(
                lambda t, s, **p: s.discrete_state, period=period, offset=0.0
            )
            self.declare_periodic_update(
                lambda t, s, **p: jnp.array(-p["b"]), period=period, offset=0.0
            )

    builder = jaxonomy.DiagramBuilder()
    integ = builder.add(Integrator(jnp.array(1.0)))
    dt_blk = builder.add(ConstDT())
    mul = builder.add(Product(2))
    builder.connect(integ.output_ports[0], mul.input_ports[0])
    builder.connect(dt_blk.output_ports[0], mul.input_ports[1])
    builder.connect(mul.output_ports[0], integ.input_ports[0])
    diagram = builder.build()
    ctx = diagram.create_context()

    opts = jaxonomy.SimulatorOptions(
        math_backend="jax",
        enable_autodiff=True,
        max_major_steps=200,
    )

    @jax.jit
    def fwd(x0, context):
        int_ctx = context[integ.system_id].with_continuous_state(x0)
        context = context.with_subcontext(integ.system_id, int_ctx)
        res = jaxonomy.simulate(diagram, context, (0.0, T), options=opts)
        return res.context[integ.system_id].continuous_state

    x0 = jnp.array(1.0)
    xf = fwd(x0, ctx)
    assert jnp.allclose(xf, x0 * jnp.exp(-b_val * T), rtol=1e-3), f"fwd={xf}"

    # Gradient w.r.t. initial state is well-defined
    dx0 = jax.jit(jax.grad(fwd))(x0, ctx)
    assert jnp.allclose(dx0, jnp.exp(-b_val * T), rtol=1e-3), f"∂x/∂x0={dx0}"


# ═══════════════════════════════════════════════════════════════════════════
# AC  – acausal DAE systems
# ═══════════════════════════════════════════════════════════════════════════


def test_acausal_rc_ic_grad():
    """AC – RC circuit: ∂Vc(T)/∂Vc(0) = exp(-T/(R·C)).

    The acausal DAE state vector has Vc as the first (differential) state.
    Gradient via adjoint through BDF solver on DAE with mass matrix.
    """
    from jaxonomy.acausal import AcausalCompiler, AcausalDiagram, EqnEnv
    from jaxonomy.acausal import electrical as elec

    R, C, V = 1.0, 1.0, 1.0
    Vc0_val = 0.5
    T = 2.0

    ev = EqnEnv()
    ad = AcausalDiagram()
    vs = elec.VoltageSource(ev, name="vs", v=V)
    r1 = elec.Resistor(ev, name="r1", R=R)
    c1 = elec.Capacitor(ev, name="c1", C=C,
                         initial_voltage=Vc0_val, initial_voltage_fixed=True)
    gnd = elec.Ground(ev, name="gnd")
    ad.connect(vs, "p", r1, "n")
    ad.connect(r1, "p", c1, "p")
    ad.connect(c1, "n", vs, "n")
    ad.connect(vs, "n", gnd, "p")
    ac = AcausalCompiler(ev, ad)
    rc_sys = ac()

    b = jaxonomy.DiagramBuilder()
    s = b.add(rc_sys)
    diagram = b.build()
    ctx0 = diagram.create_context()

    # Vc is the first differential state (index 0 in the DAE state vector)
    vc_idx = 0
    x0_full = jnp.array(ctx0[s.system_id].continuous_state)

    @jax.jit
    def fwd(vc0_scalar):
        # Vary only the differential state; algebraic vars handled by BDF
        x_new = x0_full.at[vc_idx].set(vc0_scalar)
        rc_ctx = ctx0[s.system_id].with_continuous_state(x_new)
        ctx = ctx0.with_subcontext(s.system_id, rc_ctx)
        res = jaxonomy.simulate(diagram, ctx, (0.0, T), options=_OPTS_BDF)
        return res.context[s.system_id].continuous_state[vc_idx]

    dVc_dVc0 = float(jax.jit(jax.grad(fwd))(jnp.array(Vc0_val)))
    analytic = np.exp(-T / (R * C))
    assert abs(dVc_dVc0 - analytic) < 1e-3, (
        f"∂Vc(T)/∂Vc(0): AD={dVc_dVc0:.6f}, analytic={analytic:.6f}"
    )


def test_acausal_spring_mass_ic_grad():
    """AC – spring-mass: ∂x(T)/∂x(0) = cos(ω·T), ∂x(T)/∂v(0) = sin(ω·T)/ω.

    State layout: [v(t), x(t), F1, F2].
    We vary the differential states (v at idx 0, x at idx 1) independently,
    holding algebraic force variables fixed.  The BDF solver re-satisfies
    algebraic constraints at each step; varying only the differential IC
    gives the physically correct sensitivity.

    Note: BDF adjoint on DAE has ~1% error vs analytic at default tolerance.
    """
    from jaxonomy.acausal import AcausalCompiler, AcausalDiagram, EqnEnv
    from jaxonomy.acausal import translational as trans

    K, M = 4.0, 1.0
    omega = (K / M) ** 0.5
    T = 0.5

    ev = EqnEnv()
    ad = AcausalDiagram()
    fp = trans.FixedPosition(ev, name="fp")
    sp1 = trans.Spring(ev, name="sp", K=K)
    m1 = trans.Mass(
        ev, name="m1", M=M,
        initial_velocity=0.0, initial_velocity_fixed=True,
        initial_position=1.0, initial_position_fixed=True,
    )
    ad.connect(fp, "flange", sp1, "flange_a")
    ad.connect(sp1, "flange_b", m1, "flange")
    ac = AcausalCompiler(ev, ad)
    system = ac()
    b = jaxonomy.DiagramBuilder()
    s = b.add(system)
    diagram = b.build()
    ctx0 = diagram.create_context()

    x0_full = jnp.array(ctx0[s.system_id].continuous_state)
    # Determine v/x indices empirically: run briefly with v0=1.5, x0=0
    # → state[0] evolves as velocity, state[1] as position.
    v_idx, x_idx = 0, 1

    @jax.jit
    def fwd_x(x0_scalar):
        x_new = x0_full.at[x_idx].set(x0_scalar)
        ctx = ctx0.with_subcontext(s.system_id, ctx0[s.system_id].with_continuous_state(x_new))
        res = jaxonomy.simulate(diagram, ctx, (0.0, T), options=_OPTS_BDF)
        return res.context[s.system_id].continuous_state[x_idx]

    @jax.jit
    def fwd_v(v0_scalar):
        x_new = x0_full.at[v_idx].set(v0_scalar)
        ctx = ctx0.with_subcontext(s.system_id, ctx0[s.system_id].with_continuous_state(x_new))
        res = jaxonomy.simulate(diagram, ctx, (0.0, T), options=_OPTS_BDF)
        return res.context[s.system_id].continuous_state[x_idx]

    dx_dx0 = float(jax.grad(fwd_x)(x0_full[x_idx]))
    dx_dv0 = float(jax.grad(fwd_v)(x0_full[v_idx]))

    cos_wT = np.cos(omega * T)
    sin_wT_over_w = np.sin(omega * T) / omega

    # 1% tolerance accounts for BDF adjoint accuracy on the acausal DAE
    assert abs(dx_dx0 - cos_wT) < 0.01, f"∂x/∂x0: AD={dx_dx0:.4f}, ana={cos_wT:.4f}"
    assert abs(dx_dv0 - sin_wT_over_w) < 0.01, (
        f"∂x/∂v0: AD={dx_dv0:.4f}, ana={sin_wT_over_w:.4f}"
    )


def test_acausal_thermal_two_cap_ic_grad():
    """AC – two thermal capacitors: ∂T1(T)/∂T1(0) = (1 + exp(-T/τ)) / 2.

    C=15 J/K each, R=0.1 K/W, T1(0)=373.15 K, T2(0)=273.15 K.
    T_eq = (T1+T2)/2 = 323.15 K,  τ = R·C/2 = 0.75 s.

    Exact: T1(t) = T_eq + (T1(0)−T_eq)·exp(−t/τ)
    ∂T1(T)/∂T1(0) = 1/2 + (1/2)·exp(−T/τ) = (1 + exp(−T/τ)) / 2
    """
    from jaxonomy.acausal import AcausalCompiler, AcausalDiagram, EqnEnv
    from jaxonomy.acausal import thermal as ht

    C_val, R_val = 15.0, 0.1
    T1_0, T2_0 = 373.15, 273.15
    tau = R_val * C_val / 2  # = 0.75 s
    T_sim = 2.0

    ev = EqnEnv()
    ad = AcausalDiagram()
    c1 = ht.HeatCapacitor(ev, name="c1", C=C_val,
                           initial_temperature=T1_0, initial_temperature_fixed=True)
    r1 = ht.Insulator(ev, name="r1", R=R_val)
    c2 = ht.HeatCapacitor(ev, name="c2", C=C_val,
                           initial_temperature=T2_0, initial_temperature_fixed=True)
    ad.connect(c1, "port", r1, "port_a")
    ad.connect(r1, "port_b", c2, "port")
    ac = AcausalCompiler(ev, ad)
    system = ac()
    b = jaxonomy.DiagramBuilder()
    s = b.add(system)
    diagram = b.build()
    ctx0 = diagram.create_context()

    x_init = np.array(ctx0[s.system_id].continuous_state)
    # Identify T1 and T2 by their initial values
    t1_idx = int(np.argmin(np.abs(x_init - T1_0)))
    t2_idx = int(np.argmin(np.abs(x_init - T2_0)))
    assert abs(x_init[t1_idx] - T1_0) < 1.0, "T1 index not found"
    assert abs(x_init[t2_idx] - T2_0) < 1.0, "T2 index not found"

    x0_full = jnp.array(x_init)

    @jax.jit
    def fwd(T1_init):
        x_new = x0_full.at[t1_idx].set(T1_init)
        ctx = ctx0.with_subcontext(s.system_id, ctx0[s.system_id].with_continuous_state(x_new))
        res = jaxonomy.simulate(diagram, ctx, (0.0, T_sim), options=_OPTS_BDF)
        return res.context[s.system_id].continuous_state[t1_idx]

    dT1_dT10 = float(jax.jit(jax.grad(fwd))(jnp.array(T1_0)))
    analytic = (1.0 + np.exp(-T_sim / tau)) / 2.0
    assert abs(dT1_dT10 - analytic) < 1e-3, (
        f"∂T1/∂T1(0): AD={dT1_dT10:.6f}, ana={analytic:.6f}"
    )


# ═══════════════════════════════════════════════════════════════════════════
# SM  – state machine / mode-switching
# ═══════════════════════════════════════════════════════════════════════════


class _ModeSwitchingIntegrator(jaxonomy.LeafSystem):
    """Piecewise-constant dynamics with a zero-crossing reset.

    Mode 0: xdot = -a   (decreasing)
    Mode 1: xdot = +a   (increasing)
    Transition 0→1 when x = 0.

    Analytic solution from x(0) = x0 > 0:
        x(t) = x0 - a·t,        t ≤ x0/a
        x(t) = a·(t - x0/a),    t > x0/a
    """

    class Stage(IntEnum):
        MODE_0 = 0
        MODE_1 = 1

    def __init__(self, a=1.0):
        super().__init__()
        self.declare_dynamic_parameter("a", a)
        self.declare_default_mode(self.Stage.MODE_0)
        self.declare_continuous_state(shape=(), ode=self._ode)
        self.declare_continuous_state_output()
        self.declare_zero_crossing(
            guard=self._guard,
            name="cross_zero",
            start_mode=self.Stage.MODE_0,
            end_mode=self.Stage.MODE_1,
        )

    def _ode(self, time, state, **params):
        a = params["a"]
        return jax.lax.switch(state.mode, [lambda: -a, lambda: a])

    def _guard(self, time, state, **params):
        return state.continuous_state


def test_sm_mode_switch_grad_before_crossing():
    """SM – gradient before mode transition: tf < x0/a.

    x(tf) = x0 - a·tf
    ∂x(tf)/∂x0 = 1,   ∂x(tf)/∂tf = -a,   ∂x(tf)/∂a = -tf
    """
    jaxonomy.set_backend("jax")
    a, x0, tf = 1.0, 1.5, 0.8  # 0.8 < 1.5/1 = 1.5

    model = _ModeSwitchingIntegrator(a=a)
    ctx = model.create_context()
    opts = jaxonomy.SimulatorOptions(enable_autodiff=True, max_major_steps=100)
    sim = jaxonomy.Simulator(model, options=opts)

    def forward(context, x0_, tf_, a_):
        context = context.with_continuous_state(x0_).with_parameter("a", a_)
        return sim.advance_to(tf_, context).context.continuous_state

    if sim.enable_tracing:
        forward = jax.jit(forward)
        grad_fwd = jax.jit(jax.grad(forward, argnums=(1, 2, 3)))

    xf = forward(ctx, jnp.array(x0), jnp.array(tf), jnp.array(a))
    assert jnp.allclose(xf, x0 - a * tf, atol=1e-5), f"fwd: {xf}"

    dx0, dtf, da = grad_fwd(ctx, jnp.array(x0), jnp.array(tf), jnp.array(a))
    assert jnp.allclose(dx0, 1.0, atol=1e-5), f"∂x/∂x0={dx0}"
    assert jnp.allclose(dtf, -a, atol=1e-5), f"∂x/∂tf={dtf}"
    assert jnp.allclose(da, -tf, atol=1e-5), f"∂x/∂a={da}"


def test_sm_mode_switch_grad_after_crossing():
    """SM – gradient after mode transition: tf > x0/a.

    x(tf) = a·tf - x0
    ∂x(tf)/∂x0 = -1,   ∂x(tf)/∂tf = +a,   ∂x(tf)/∂a = tf
    """
    jaxonomy.set_backend("jax")
    a, x0, tf = 1.0, 1.0, 1.5  # 1.5 > 1.0/1 = 1.0

    model = _ModeSwitchingIntegrator(a=a)
    ctx = model.create_context()
    opts = jaxonomy.SimulatorOptions(enable_autodiff=True, max_major_steps=100)
    sim = jaxonomy.Simulator(model, options=opts)

    def forward(context, x0_, tf_, a_):
        context = context.with_continuous_state(x0_).with_parameter("a", a_)
        return sim.advance_to(tf_, context).context.continuous_state

    if sim.enable_tracing:
        forward = jax.jit(forward)
        grad_fwd = jax.jit(jax.grad(forward, argnums=(1, 2, 3)))

    xf = forward(ctx, jnp.array(x0), jnp.array(tf), jnp.array(a))
    assert jnp.allclose(xf, a * tf - x0, atol=1e-5), f"fwd: {xf}"

    dx0, dtf, da = grad_fwd(ctx, jnp.array(x0), jnp.array(tf), jnp.array(a))
    assert jnp.allclose(dx0, -1.0, atol=1e-4), f"∂x/∂x0={dx0}"
    assert jnp.allclose(dtf, a, atol=1e-4), f"∂x/∂tf={dtf}"
    assert jnp.allclose(da, tf, atol=1e-4), f"∂x/∂a={da}"   # ∂(a·tf−x0)/∂a = tf


def test_sm_mode_switch_in_diagram_grad():
    """SM – mode-switching system inside a DiagramBuilder with a causal Gain.

    After mode transition: x_sm(T) = a·T − x0
    ∂x_sm(T)/∂x0 = −1  →  ∂(g·x_sm)/∂x0 = g·(−1) = −g
    """
    jaxonomy.set_backend("jax")
    a, g_val, x0, tf = 1.0, 3.0, 1.0, 1.5  # tf > x0/a → after switch

    bld = jaxonomy.DiagramBuilder()
    sm = bld.add(_ModeSwitchingIntegrator(a=a))
    gain = bld.add(Gain(g_val))
    bld.connect(sm.output_ports[0], gain.input_ports[0])
    diagram = bld.build()
    ctx = diagram.create_context()

    opts = jaxonomy.SimulatorOptions(
        math_backend="jax",
        enable_autodiff=True,
        max_major_steps=200,
    )

    @jax.jit
    def fwd(x0_, context):
        sm_ctx = context[sm.system_id].with_continuous_state(x0_)
        context = context.with_subcontext(sm.system_id, sm_ctx)
        res = jaxonomy.simulate(diagram, context, (0.0, tf), options=opts)
        return res.context[sm.system_id].continuous_state

    dx0 = g_val * float(jax.grad(fwd)(jnp.array(x0), ctx))
    assert abs(dx0 - g_val * (-1.0)) < 1e-3, f"∂y/∂x0={dx0}, expected {-g_val}"


# ═══════════════════════════════════════════════════════════════════════════
# NN  – neural-network blocks (equinox MLP)
# ═══════════════════════════════════════════════════════════════════════════


def test_nn_mlp_output_grad_matches_eqx():
    """NN – MLP feedforward block: simulation gradient matches eqx.filter_grad.

    We embed a jaxonomy MLP block in a feedback ODE: dx/dt = MLP(x; θ).
    For a very short T, the integral is nearly linear in MLP(x0):
        x(T) ≈ x0 + T·MLP(x0; θ)
    so  ∂x(T)/∂θ ≈ T·∂MLP(x0)/∂θ.

    The ratio of simulation gradient to T·eqx-gradient should be ≈1 (±10%).
    We also verify the sign and relative magnitude of FD vs AD for one weight.
    """
    from jaxonomy.library.nn import MLP as JaxonomyMLP

    jaxonomy.set_backend("jax")
    T_tiny = 0.02
    x0_val = 0.5

    bld = jaxonomy.DiagramBuilder()
    integ = bld.add(Integrator(jnp.array([x0_val]), name="x"))
    mlp_blk = bld.add(
        JaxonomyMLP(in_size=1, out_size=1, width_size=4, depth=2, seed=42,
                    activation_str="tanh")
    )
    bld.connect(integ.output_ports[0], mlp_blk.input_ports[0])
    bld.connect(mlp_blk.output_ports[0], integ.input_ports[0])
    diagram = bld.build()
    ctx0 = diagram.create_context()

    # mlp_params is the eqx-partitioned array-only part of the model
    mlp_params0 = ctx0[mlp_blk.system_id].parameters["mlp_params"]
    mlp_static = mlp_blk.mlp_static
    leaves0, treedef = jax.tree.flatten(mlp_params0)

    opts = jaxonomy.SimulatorOptions(math_backend="jax", enable_autodiff=True)

    def fwd_flat(flat_leaves):
        mlp_params = treedef.unflatten(flat_leaves)
        ctx2 = ctx0.with_subcontext(
            mlp_blk.system_id,
            ctx0[mlp_blk.system_id].with_parameters({"mlp_params": mlp_params}),
        )
        res = jaxonomy.simulate(diagram, ctx2, (0.0, T_tiny), options=opts)
        return res.context[integ.system_id].continuous_state[0]

    ad_grads = jax.grad(fwd_flat)(leaves0)

    # FD for one element
    eps = 1e-4
    lp = list(leaves0)
    lm = list(leaves0)
    lp[0] = leaves0[0].at[0, 0].add(eps)
    lm[0] = leaves0[0].at[0, 0].add(-eps)
    fd_w00 = (fwd_flat(lp) - fwd_flat(lm)) / (2 * eps)

    assert abs(float(ad_grads[0][0, 0]) - float(fd_w00)) < 5e-4, (
        f"AD={ad_grads[0][0,0]:.6f}, FD={fd_w00:.6f}"
    )

    # Direct eqx gradient: d/dtheta [MLP(x0; theta)]
    x0_vec = jnp.array([x0_val])

    def direct_mlp_flat(flat_leaves):
        params = treedef.unflatten(flat_leaves)
        mlp_full = eqx.combine(params, mlp_static)
        return mlp_full(x0_vec)[0]

    eqx_grads = jax.grad(direct_mlp_flat)(leaves0)

    # Verify sim_grad ≈ T_tiny * eqx_grad  (linear approximation)
    for i, (g_sim, g_eqx) in enumerate(zip(ad_grads, eqx_grads)):
        denom = T_tiny * np.asarray(g_eqx) + 1e-12
        ratio = np.asarray(g_sim) / denom
        valid = np.abs(denom) > 1e-6
        if np.any(valid):
            assert np.allclose(ratio[valid], 1.0, atol=0.15), (
                f"leaf {i}: sim/T/eqx deviates from 1: {ratio[valid]}"
            )


def test_nn_neural_ode_weight_grad_vs_fd():
    """NN – NeuralODE: MLP as ODE right-hand side, gradient w.r.t. weights.

    System: dx/dt = MLP(x; θ) − x,  x(0)=0
    The −x term ensures stability regardless of θ.
    Loss: L(θ) = x(T)[0]

    AD gradient matches FD for each weight leaf.
    Uses eqx.partition/combine to correctly handle the eqx model as a
    jaxonomy dynamic parameter.
    """
    jaxonomy.set_backend("jax")
    T = 0.3

    seed_key = jax.random.PRNGKey(7)
    mlp_model = eqx.nn.MLP(1, 1, width_size=4, depth=2, key=seed_key)
    mlp_params0, mlp_static = eqx.partition(mlp_model, eqx.is_array)

    class NeuralODE(jaxonomy.LeafSystem):
        def __init__(self):
            super().__init__()
            # as_array=False: store eqx pytree (arrays only) without casting
            self.declare_dynamic_parameter("mlp_params", mlp_params0, as_array=False)
            self._mlp_static = mlp_static
            self.declare_continuous_state(default_value=jnp.zeros(1), ode=self._ode)

        def _ode(self, time, state, **params):
            x = state.continuous_state
            mlp_full = eqx.combine(params["mlp_params"], self._mlp_static)
            return mlp_full(x) - x

    sys = NeuralODE()
    ctx = sys.create_context()
    leaves0, treedef = jax.tree.flatten(mlp_params0)

    def loss_flat(flat_leaves):
        params = treedef.unflatten(flat_leaves)
        ctx2 = ctx.with_parameter("mlp_params", params)
        res = jaxonomy.simulate(sys, ctx2, (0.0, T), options=_OPTS)
        return res.context.continuous_state[0]

    ad_grads = jax.grad(loss_flat)(leaves0)

    eps = 1e-4
    # Check first leaf, element [0,0] (or [0] for 1D)
    leaf0 = leaves0[0]
    for ri in range(min(2, leaf0.shape[0])):
        ci = 0  # first column
        idx = (ri, ci) if leaf0.ndim == 2 else (ri,)
        lp, lm = list(leaves0), list(leaves0)
        lp[0] = leaf0.at[idx].add(eps)
        lm[0] = leaf0.at[idx].add(-eps)
        g_fd = float((loss_flat(lp) - loss_flat(lm)) / (2 * eps))
        g_ad = float(ad_grads[0][idx])
        assert abs(g_ad - g_fd) < 1e-3, (
            f"leaf[0]{idx}: AD={g_ad:.6f}, FD={g_fd:.6f}"
        )


# ═══════════════════════════════════════════════════════════════════════════
# HYB  – hybrid combinations
# ═══════════════════════════════════════════════════════════════════════════


def test_hyb_acausal_causal_stateless_grad():
    """HYB (AC + stateless causal) – RC circuit with causal Gain in diagram.

    Diagram: AcausalRC → Demultiplexer → Gain(G)
    The Gain is stateless; the only CT state is the RC DAE.

    Gradient ∂Vc(T)/∂Vc(0) = exp(−T/(R·C)) is unchanged by the Gain.
    Demonstrates that acausal systems can be embedded in causal diagrams
    and remain differentiable end-to-end.
    """
    from jaxonomy.acausal import AcausalCompiler, AcausalDiagram, EqnEnv
    from jaxonomy.acausal import electrical as elec

    R, C, V = 1.0, 1.0, 1.0
    Vc0_val = 0.5
    T = 1.5

    ev = EqnEnv()
    ad = AcausalDiagram()
    vs = elec.VoltageSource(ev, name="vs", v=V)
    r1 = elec.Resistor(ev, name="r1", R=R)
    c1 = elec.Capacitor(ev, name="c1", C=C,
                         initial_voltage=Vc0_val, initial_voltage_fixed=True)
    gnd = elec.Ground(ev, name="gnd")
    ad.connect(vs, "p", r1, "n")
    ad.connect(r1, "p", c1, "p")
    ad.connect(c1, "n", vs, "n")
    ad.connect(vs, "n", gnd, "p")
    ac = AcausalCompiler(ev, ad)
    rc_sys = ac()

    # Determine number of states
    tmp_b = jaxonomy.DiagramBuilder()
    tmp_s = tmp_b.add(rc_sys)
    n_states = len(tmp_b.build().create_context()[tmp_s.system_id].continuous_state)

    bld = jaxonomy.DiagramBuilder()
    rc = bld.add(rc_sys)
    demux = bld.add(Demultiplexer(n_states, name="dmx"))
    gain = bld.add(Gain(2.0, name="gain"))
    bld.connect(rc.output_ports[0], demux.input_ports[0])
    bld.connect(demux.output_ports[0], gain.input_ports[0])
    diagram = bld.build()
    ctx0 = diagram.create_context()
    x0_full = jnp.array(ctx0[rc.system_id].continuous_state)

    @jax.jit
    def fwd(vc0_scalar):
        x_new = x0_full.at[0].set(vc0_scalar)
        rc_ctx = ctx0[rc.system_id].with_continuous_state(x_new)
        ctx = ctx0.with_subcontext(rc.system_id, rc_ctx)
        res = jaxonomy.simulate(diagram, ctx, (0.0, T), options=_OPTS_BDF)
        return res.context[rc.system_id].continuous_state[0]

    dVc = float(jax.jit(jax.grad(fwd))(jnp.array(Vc0_val)))
    analytic = np.exp(-T / (R * C))
    assert abs(dVc - analytic) < 1e-3, f"AD={dVc:.6f}, ana={analytic:.6f}"


def test_hyb_acausal_ct_integrator_grad():
    """HYB (AC + CT integrator) – gradient works after mass-matrix permutation fix.

    Combining an acausal BDF/DAE RC circuit with a separate causal CT integrator
    in a Diagram previously raised ``NotImplementedError`` because the combined
    block-diagonal mass matrix [[1,0,0],[0,0,0],[0,0,1]] has an algebraic row
    sandwiched between two differential rows, violating the assumption of the
    adjoint initialization routine.

    The fix computes a permutation that brings the mass matrix into canonical
    semi-explicit form before applying the Cao et al. (2003) adjoint IC formula,
    then un-permutes back to the original state ordering.

    AD gradient ∂(integrator state)/∂Vc₀ is verified against finite differences.
    """
    from jaxonomy.acausal import AcausalCompiler, AcausalDiagram, EqnEnv
    from jaxonomy.acausal import electrical as elec

    ev = EqnEnv()
    ad = AcausalDiagram()
    vs = elec.VoltageSource(ev, name="vs", v=1.0)
    r1 = elec.Resistor(ev, name="r1", R=1.0)
    c1 = elec.Capacitor(ev, name="c1", C=1.0,
                         initial_voltage=0.5, initial_voltage_fixed=True)
    gnd = elec.Ground(ev, name="gnd")
    ad.connect(vs, "p", r1, "n")
    ad.connect(r1, "p", c1, "p")
    ad.connect(c1, "n", vs, "n")
    ad.connect(vs, "n", gnd, "p")
    ac = AcausalCompiler(ev, ad)
    rc_sys = ac()

    tmp_b = jaxonomy.DiagramBuilder()
    tmp_s = tmp_b.add(rc_sys)
    n_st = len(tmp_b.build().create_context()[tmp_s.system_id].continuous_state)

    bld = jaxonomy.DiagramBuilder()
    rc = bld.add(rc_sys)
    demux = bld.add(Demultiplexer(n_st, name="dmx"))
    integ = bld.add(Integrator(0.0, name="integ"))
    bld.connect(rc.output_ports[0], demux.input_ports[0])
    bld.connect(demux.output_ports[0], integ.input_ports[0])
    diagram = bld.build()
    ctx0 = diagram.create_context()
    x0_full = jnp.array(ctx0[rc.system_id].continuous_state)

    @jax.jit
    def fwd(vc0_scalar):
        x_new = x0_full.at[0].set(vc0_scalar)
        rc_ctx = ctx0[rc.system_id].with_continuous_state(x_new)
        ctx = ctx0.with_subcontext(rc.system_id, rc_ctx)
        res = jaxonomy.simulate(diagram, ctx, (0.0, 1.0), options=_OPTS_BDF)
        return res.context[integ.system_id].continuous_state

    # AD gradient
    dvc0_ad = float(jax.grad(fwd)(jnp.array(0.5)))

    # FD ground truth
    eps = 1e-4
    dvc0_fd = float((fwd(jnp.array(0.5 + eps)) - fwd(jnp.array(0.5 - eps))) / (2 * eps))

    assert abs(dvc0_ad - dvc0_fd) < 5e-3, (
        f"AD={dvc0_ad:.6f}, FD={dvc0_fd:.6f}"
    )


def test_hyb_ct_mode_switch_diagram_param_grad():
    """HYB (SM + CT) – mode-switching ODE embedded in a causal diagram.

    After crossing: x_sm(T) = a·T − x0
    ∂x_sm(T)/∂x0 = −1

    Wrapping in a DiagramBuilder should not corrupt the adjoint computation.
    """
    jaxonomy.set_backend("jax")
    a, x0_val, tf = 1.0, 1.0, 1.3  # tf > x0/a → after crossing

    bld = jaxonomy.DiagramBuilder()
    sm = bld.add(_ModeSwitchingIntegrator(a=a))
    gain_blk = bld.add(Gain(1.0, name="g"))
    bld.connect(sm.output_ports[0], gain_blk.input_ports[0])
    diagram = bld.build()
    ctx = diagram.create_context()

    opts = jaxonomy.SimulatorOptions(
        math_backend="jax",
        enable_autodiff=True,
        max_major_steps=200,
    )

    @jax.jit
    def fwd(x0_, context):
        sm_ctx = context[sm.system_id].with_continuous_state(x0_)
        context = context.with_subcontext(sm.system_id, sm_ctx)
        res = jaxonomy.simulate(diagram, context, (0.0, tf), options=opts)
        return res.context[sm.system_id].continuous_state

    dx0 = float(jax.grad(fwd)(jnp.array(x0_val), ctx))
    assert abs(dx0 - (-1.0)) < 1e-3, f"∂x(T)/∂x0={dx0}, expected -1"


def test_hyb_nn_stiff_ode_bdf_grad():
    """HYB (NN + CT BDF) – MLP inside a stiff ODE solved by BDF.

    System: dx/dt = −λ·x + ε·MLP(x; θ),  λ=50, ε=0.01, x(0)=1
    The dominant mode is exp(−λt); MLP is a small perturbation.

    Verifies that the checkpoint-based adjoint correctly propagates through
    MLP computations inside a stiff BDF solver.
    AD gradient matches FD for the first weight leaf.
    """
    jaxonomy.set_backend("jax")
    lam, eps_nn, T = 50.0, 0.01, 0.05

    seed_key = jax.random.PRNGKey(13)
    mlp_model = eqx.nn.MLP(1, 1, width_size=4, depth=1, key=seed_key)
    mlp_params0, mlp_static = eqx.partition(mlp_model, eqx.is_array)

    class StiffNeuralODE(jaxonomy.LeafSystem):
        def __init__(self):
            super().__init__()
            self.declare_dynamic_parameter("mlp_params", mlp_params0, as_array=False)
            self._mlp_static = mlp_static
            self.declare_continuous_state(default_value=jnp.array([1.0]), ode=self._ode)

        def _ode(self, time, state, **params):
            x = state.continuous_state
            mlp_full = eqx.combine(params["mlp_params"], self._mlp_static)
            return -lam * x + eps_nn * mlp_full(x)

    sys = StiffNeuralODE()
    ctx = sys.create_context()
    leaves0, treedef = jax.tree.flatten(mlp_params0)

    def loss_flat(flat_leaves):
        params = treedef.unflatten(flat_leaves)
        ctx2 = ctx.with_parameter("mlp_params", params)
        res = jaxonomy.simulate(sys, ctx2, (0.0, T), options=_OPTS_BDF)
        return res.context.continuous_state[0]

    ad_grads = jax.grad(loss_flat)(leaves0)

    _eps = 1e-4
    leaf0 = leaves0[0]
    for ri in range(min(2, leaf0.shape[0])):
        ci = 0
        idx = (ri, ci) if leaf0.ndim == 2 else (ri,)
        lp, lm = list(leaves0), list(leaves0)
        lp[0] = leaf0.at[idx].add(_eps)
        lm[0] = leaf0.at[idx].add(-_eps)
        g_fd = float((loss_flat(lp) - loss_flat(lm)) / (2 * _eps))
        g_ad = float(ad_grads[0][idx])
        assert abs(g_ad - g_fd) < 1e-3, (
            f"leaf[0]{idx}: AD={g_ad:.6f}, FD={g_fd:.6f}"
        )


def test_hyb_ac_dt_coexistence_grad():
    """HYB (AC + DT) – acausal RC + independent DT counter coexisting in a diagram.

    A purely-autonomous DT step counter ticks every 0.5 s but has NO signal
    connection to the RC circuit, forcing the simulator into hybrid (CT+DT) mode
    without introducing CT→DT signal coupling (which would cause gradient
    accumulation across major steps — a known limitation).

    ∂Vc(T)/∂Vc(0) = exp(−T/(R·C)) must be unaffected by the presence of the
    autonomous DT block.
    """
    from jaxonomy.acausal import AcausalCompiler, AcausalDiagram, EqnEnv
    from jaxonomy.acausal import electrical as elec

    R, C, V = 1.0, 1.0, 1.0
    Vc0_val = 0.3
    T = 1.0

    ev = EqnEnv()
    ad_diag = AcausalDiagram()
    vs = elec.VoltageSource(ev, name="vs", v=V)
    r1 = elec.Resistor(ev, name="r1", R=R)
    c1 = elec.Capacitor(ev, name="c1", C=C,
                         initial_voltage=Vc0_val, initial_voltage_fixed=True)
    gnd = elec.Ground(ev, name="gnd")
    ad_diag.connect(vs, "p", r1, "n")
    ad_diag.connect(r1, "p", c1, "p")
    ad_diag.connect(c1, "n", vs, "n")
    ad_diag.connect(vs, "n", gnd, "p")
    ac = AcausalCompiler(ev, ad_diag)
    rc_sys = ac()

    # Autonomous DT step counter — no input ports, no connection to RC.
    # Its sole purpose is to force the simulator into hybrid (CT+DT) mode.
    class DTCounter(jaxonomy.LeafSystem):
        def __init__(self, period=0.5):
            super().__init__()
            self.declare_discrete_state(default_value=jnp.array(0.0))
            self.declare_periodic_update(self._update, period=period, offset=0.0)

        def _update(self, time, state, *inputs):
            return state.discrete_state + 1.0

    bld = jaxonomy.DiagramBuilder()
    rc = bld.add(rc_sys)
    _counter = bld.add(DTCounter(period=0.5))
    # Intentionally NO signal connection between rc and counter.
    diagram = bld.build()
    ctx0 = diagram.create_context()
    x0_full = jnp.array(ctx0[rc.system_id].continuous_state)

    @jax.jit
    def fwd(vc0_scalar):
        x_new = x0_full.at[0].set(vc0_scalar)
        rc_ctx = ctx0[rc.system_id].with_continuous_state(x_new)
        ctx = ctx0.with_subcontext(rc.system_id, rc_ctx)
        res = jaxonomy.simulate(diagram, ctx, (0.0, T), options=_OPTS_BDF)
        return res.context[rc.system_id].continuous_state[0]

    dVc = float(jax.grad(fwd)(jnp.array(Vc0_val)))
    analytic = np.exp(-T / (R * C))
    assert abs(dVc - analytic) < 1e-3, f"AD={dVc:.6f}, ana={analytic:.6f}"


def test_hyb_sm_nn_weight_grad():
    """HYB (SM + NN) – neural ODE with mode-based dynamics, gradient vs FD.

    A custom LeafSystem embeds an equinox MLP as the ODE RHS in Mode 0 (x > 0)
    and switches to a simple linear decay −2x in Mode 1 (x < 0).

    Starting from x(0) = 1.0 and choosing T short enough that x stays in Mode 0,
    the system effectively runs dx/dt = MLP(x; θ) − x.

    We verify that ∂x(T)/∂θ (AD) matches finite differences for the first weight
    leaf, checking that the adjoint correctly propagates through mode-switching
    infrastructure even when no switch occurs.
    """

    jaxonomy.set_backend("jax")
    T = 0.1

    seed_key = jax.random.PRNGKey(99)
    mlp_model = eqx.nn.MLP(1, 1, width_size=4, depth=1, key=seed_key)
    mlp_params0, mlp_static = eqx.partition(mlp_model, eqx.is_array)

    class SMNeuralODE(jaxonomy.LeafSystem):
        def __init__(self):
            super().__init__()
            self.declare_dynamic_parameter("mlp_params", mlp_params0, as_array=False)
            self._mlp_static = mlp_static
            self.declare_default_mode(0)
            self.declare_continuous_state(default_value=jnp.array([1.0]), ode=self._ode)
            self.declare_zero_crossing(
                guard=lambda t, s, **p: s.continuous_state[0],
                name="cross_zero",
                start_mode=0,
                end_mode=1,
            )

        def _ode(self, time, state, **params):
            x = state.continuous_state
            mlp = eqx.combine(params["mlp_params"], self._mlp_static)
            return jax.lax.switch(
                state.mode,
                [lambda: mlp(x) - x,   # Mode 0 (x > 0): NN-damped
                 lambda: -2.0 * x],    # Mode 1 (x < 0): fast linear decay
            )

    sys = SMNeuralODE()
    ctx = sys.create_context()
    leaves0, treedef = jax.tree.flatten(mlp_params0)

    opts = jaxonomy.SimulatorOptions(
        math_backend="jax", enable_autodiff=True, max_major_steps=50
    )

    def loss_flat(flat_leaves):
        params = treedef.unflatten(flat_leaves)
        ctx2 = ctx.with_parameter("mlp_params", params)
        res = jaxonomy.simulate(sys, ctx2, (0.0, T), options=opts)
        return res.context.continuous_state[0]

    ad_grads = jax.grad(loss_flat)(leaves0)

    eps = 1e-4
    leaf0 = leaves0[0]
    for ri in range(min(2, leaf0.shape[0])):
        ci = 0
        idx = (ri, ci) if leaf0.ndim == 2 else (ri,)
        lp, lm = list(leaves0), list(leaves0)
        lp[0] = leaf0.at[idx].add(eps)
        lm[0] = leaf0.at[idx].add(-eps)
        g_fd = float((loss_flat(lp) - loss_flat(lm)) / (2 * eps))
        g_ad = float(ad_grads[0][idx])
        assert abs(g_ad - g_fd) < 1e-3, (
            f"SM+NN leaf[0]{idx}: AD={g_ad:.6f}, FD={g_fd:.6f}"
        )


def test_hyb_ac_ct_dt_three_way_grad():
    """HYB (AC + CT + DT) – three-component hybrid: gradient via FD.

    Diagram:
      - Acausal RC circuit (DAE, BDF solver)
      - Causal CT integrator (ODE, same BDF solver, Bug 2 fix needed)
      - DT sampler (discrete, reads RC output at fixed intervals)

    This combines all three elements that were previously problematic:
    AC+CT (Bug 2 fix) and the presence of DT state.

    AD gradient ∂(CT integrator state)/∂Vc₀ is verified against FD.
    """
    from jaxonomy.acausal import AcausalCompiler, AcausalDiagram, EqnEnv
    from jaxonomy.acausal import electrical as elec

    ev = EqnEnv()
    ad_diag = AcausalDiagram()
    vs = elec.VoltageSource(ev, name="vs", v=1.0)
    r1 = elec.Resistor(ev, name="r1", R=1.0)
    c1 = elec.Capacitor(ev, name="c1", C=1.0,
                         initial_voltage=0.3, initial_voltage_fixed=True)
    gnd = elec.Ground(ev, name="gnd")
    ad_diag.connect(vs, "p", r1, "n")
    ad_diag.connect(r1, "p", c1, "p")
    ad_diag.connect(c1, "n", vs, "n")
    ad_diag.connect(vs, "n", gnd, "p")
    ac = AcausalCompiler(ev, ad_diag)
    rc_sys = ac()

    tmp_b = jaxonomy.DiagramBuilder()
    tmp_s = tmp_b.add(rc_sys)
    n_st = len(tmp_b.build().create_context()[tmp_s.system_id].continuous_state)

    class DTSampler(jaxonomy.LeafSystem):
        def __init__(self, period=0.5):
            super().__init__()
            self.declare_discrete_state(default_value=jnp.array(0.0))
            self.declare_input_port(name="u")
            self.declare_periodic_update(self._update, period=period, offset=0.0)

        def _update(self, time, state, *inputs):
            return state.discrete_state + inputs[0][0]

    bld = jaxonomy.DiagramBuilder()
    rc = bld.add(rc_sys)
    demux = bld.add(Demultiplexer(n_st, name="dmx"))
    integ = bld.add(Integrator(0.0, name="integ"))
    sampler = bld.add(DTSampler(period=0.5))
    bld.connect(rc.output_ports[0], demux.input_ports[0])
    bld.connect(demux.output_ports[0], integ.input_ports[0])
    bld.connect(rc.output_ports[0], sampler.input_ports[0])
    diagram = bld.build()
    ctx0 = diagram.create_context()
    x0_full = jnp.array(ctx0[rc.system_id].continuous_state)

    @jax.jit
    def fwd(vc0_scalar):
        x_new = x0_full.at[0].set(vc0_scalar)
        rc_ctx = ctx0[rc.system_id].with_continuous_state(x_new)
        ctx = ctx0.with_subcontext(rc.system_id, rc_ctx)
        res = jaxonomy.simulate(diagram, ctx, (0.0, 1.0), options=_OPTS_BDF)
        return res.context[integ.system_id].continuous_state

    # AD gradient
    dvc0_ad = float(jax.grad(fwd)(jnp.array(0.3)))

    # FD ground truth
    eps = 1e-4
    dvc0_fd = float((fwd(jnp.array(0.3 + eps)) - fwd(jnp.array(0.3 - eps))) / (2 * eps))

    assert abs(dvc0_ad - dvc0_fd) < 5e-3, (
        f"AC+CT+DT: AD={dvc0_ad:.6f}, FD={dvc0_fd:.6f}"
    )


if __name__ == "__main__":
    test_ct_scalar_param_sensitivity()
    test_ct_harmonic_oscillator_jacobian()
    test_ct_linearize_second_order_system()
    test_ct_nonlinear_vector_param_grad()
    test_dt_pure_discrete_autodiff()
    test_dt_hybrid_staircase_grad()
    test_dt_hybrid_param_grad()
    test_acausal_rc_ic_grad()
    test_acausal_spring_mass_ic_grad()
    test_acausal_thermal_two_cap_ic_grad()
    test_sm_mode_switch_grad_before_crossing()
    test_sm_mode_switch_grad_after_crossing()
    test_sm_mode_switch_in_diagram_grad()
    test_nn_mlp_output_grad_matches_eqx()
    test_nn_neural_ode_weight_grad_vs_fd()
    test_hyb_acausal_causal_stateless_grad()
    test_hyb_acausal_ct_integrator_grad()
    test_hyb_ct_mode_switch_diagram_param_grad()
    test_hyb_nn_stiff_ode_bdf_grad()
    test_hyb_ac_dt_coexistence_grad()
    test_hyb_sm_nn_weight_grad()
    test_hyb_ac_ct_dt_three_way_grad()
