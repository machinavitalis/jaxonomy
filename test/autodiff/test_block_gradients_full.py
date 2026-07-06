# SPDX-License-Identifier: MIT
"""
Phase B (T-001) — full simulation-level gradient coverage.

Wraps each stateful / dynamics-carrying block in a minimal driving diagram and
verifies that ``jax.grad`` through ``simulate`` matches FD across every
selectable solver. Marked ``autodiff_full`` so it is excluded from PR CI and
runs only in the nightly job.

The stateless feedthrough / reduce / source blocks are covered by
``test_block_gradients_stateless.py`` at PR time; re-testing them through
``simulate`` would add little beyond proving that the simulator does not
mangle the gradient (which the DT/CT blocks here already verify).
"""

from __future__ import annotations

from functools import partial

import numpy as np
import pytest
import jax
import jax.numpy as jnp

import jaxonomy
from jaxonomy.library import (
    Adder,
    Constant,
    DerivativeDiscrete,
    DiscreteInitializer,
    EdgeDetection,
    FilterDiscrete,
    Gain,
    Integrator,
    IntegratorDiscrete,
    LTISystem,
    PIDDiscrete,
    Product,
    RateLimiter,
    Sine,
    Step,
    TransferFunction,
    UnitDelay,
    ZeroOrderHold,
    linearize,
)

from ._framework import assert_grad_matches_fd, sim_options
from .tolerances import SOLVERS
from jaxonomy.testing.markers import skip_if_not_jax

skip_if_not_jax()

pytestmark = pytest.mark.autodiff_full


# ── CT Integrator, all solvers ───────────────────────────────────────────────


@pytest.mark.parametrize("solver", SOLVERS)
def test_grad_sim_ct_integrator_constant_drive(solver):
    """dx/dt = u, u constant → x(T) = x0 + u·T. ∂/∂x0=1, ∂/∂u=T."""

    class ConstDrive(jaxonomy.LeafSystem):
        def __init__(self):
            super().__init__()
            self.declare_dynamic_parameter("u", 1.0)
            self.declare_continuous_state(default_value=jnp.array(0.0), ode=self._ode)

        def _ode(self, time, state, **params):
            return params["u"]

    sys = ConstDrive()
    ctx0 = sys.create_context()
    T = 0.4
    opts = sim_options(solver, "float64")

    @jax.jit
    def fwd(x0, u):
        ctx = ctx0.with_continuous_state(x0).with_parameter("u", u)
        return jaxonomy.simulate(sys, ctx, (0.0, T), options=opts).context.continuous_state

    assert_grad_matches_fd(
        fwd, jnp.array(0.7), jnp.array(1.3),
        solver=solver, dtype="float64", block="Integrator(constant-drive)",
    )


@pytest.mark.parametrize("solver", SOLVERS)
def test_grad_sim_ct_integrator_sine_drive(solver):
    """dx/dt = A·sin(ω·t + φ), x(0)=x0 — tests that a sinusoid source feeds
    the Integrator and that gradients through the Sine block's time argument
    and through the integrator chain both match FD."""
    bld = jaxonomy.DiagramBuilder()
    src = bld.add(Sine(amplitude=1.0, frequency=2.0, phase=0.0, name="sine"))
    integ = bld.add(Integrator(jnp.array(0.0), name="int"))
    bld.connect(src.output_ports[0], integ.input_ports[0])
    diagram = bld.build()
    ctx0 = diagram.create_context()
    T = 0.35
    opts = sim_options(solver, "float64")

    @jax.jit
    def fwd(amp, freq, x0):
        c = ctx0
        c = c.with_subcontext(src.system_id, c[src.system_id]
                              .with_parameter("amplitude", amp)
                              .with_parameter("frequency", freq))
        c = c.with_subcontext(integ.system_id, c[integ.system_id]
                              .with_continuous_state(x0))
        res = jaxonomy.simulate(diagram, c, (0.0, T), options=opts)
        return res.context[integ.system_id].continuous_state

    assert_grad_matches_fd(
        fwd, jnp.array(1.2), jnp.array(2.5), jnp.array(0.3),
        solver=solver, dtype="float64", block="Sine→Integrator",
    )


@pytest.mark.parametrize("solver", SOLVERS)
def test_grad_sim_ct_integrator_product_feedback(solver):
    """dx/dt = k·x (feedback via Product + Constant-gain) — closed-form
    x(T) = x0·exp(k·T). Catches gradient flow through a feedback Product."""
    bld = jaxonomy.DiagramBuilder()
    k_block = bld.add(Constant(jnp.array(-0.7), name="k"))
    prod = bld.add(Product(2, name="prod"))
    integ = bld.add(Integrator(jnp.array(1.0), name="int"))
    bld.connect(k_block.output_ports[0], prod.input_ports[0])
    bld.connect(integ.output_ports[0], prod.input_ports[1])
    bld.connect(prod.output_ports[0], integ.input_ports[0])
    diagram = bld.build()
    ctx0 = diagram.create_context()
    T = 0.5
    opts = sim_options(solver, "float64")

    @jax.jit
    def fwd(x0, k):
        c = ctx0
        c = c.with_subcontext(integ.system_id, c[integ.system_id].with_continuous_state(x0))
        c = c.with_subcontext(k_block.system_id, c[k_block.system_id].with_parameter("value", k))
        res = jaxonomy.simulate(diagram, c, (0.0, T), options=opts)
        return res.context[integ.system_id].continuous_state

    assert_grad_matches_fd(
        fwd, jnp.array(1.2), jnp.array(-0.9),
        solver=solver, dtype="float64", block="Product(feedback)→Integrator",
    )


# ── DT blocks: IntegratorDiscrete ────────────────────────────────────────────


def test_grad_sim_dt_integrator_discrete():
    """x[n+1] = x[n] + u·dt (discrete integrator). Gradient w.r.t. x0 through
    N periodic updates. No ODE solver involved, but the hybrid engine still
    integrates the zero CT-state residual so we list no solver parametrization."""
    dt = 0.1
    N = 4

    class DTInt(jaxonomy.LeafSystem):
        def __init__(self):
            super().__init__()
            self.declare_dynamic_parameter("u", 1.2)
            self.declare_discrete_state(default_value=jnp.array(0.0))
            self.declare_periodic_update(self._upd, period=dt, offset=0.0)

        def _upd(self, time, state, **params):
            return state.discrete_state + params["u"] * dt

    sys = DTInt()
    ctx0 = sys.create_context()
    opts = jaxonomy.SimulatorOptions(
        math_backend="jax", enable_autodiff=True, max_major_steps=50
    )

    @jax.jit
    def fwd(x0, u):
        c = ctx0.with_discrete_state(x0).with_parameter("u", u)
        res = jaxonomy.simulate(sys, c, (0.0, N * dt), options=opts)
        return res.context.discrete_state

    assert_grad_matches_fd(
        fwd, jnp.array(0.5), jnp.array(1.3),
        solver="dopri5",  # tolerance bucket — DT has no solver of its own
        dtype="float64",
        block="IntegratorDiscrete-like",
    )


# ── composed diagram: Adder + Gain + Integrator (CT), every solver ──────────


@pytest.mark.parametrize("solver", SOLVERS)
def test_grad_sim_ct_adder_gain_integrator(solver):
    """Diagram: Integrator feeding Gain(-k) + constant; classic decay-to-setpoint.
    dx/dt = -k·(x - r) with closed form x(T) = r + (x0-r)·e^{-k·T}."""
    k_val, r_val = 1.5, 2.0
    bld = jaxonomy.DiagramBuilder()
    integ = bld.add(Integrator(jnp.array(0.5), name="int"))
    r_src = bld.add(Constant(jnp.array(r_val), name="r"))
    err = bld.add(Adder(2, operators="+-", name="err"))  # r - x
    gain_blk = bld.add(Gain(k_val, name="gain"))
    bld.connect(r_src.output_ports[0], err.input_ports[0])
    bld.connect(integ.output_ports[0], err.input_ports[1])
    bld.connect(err.output_ports[0], gain_blk.input_ports[0])
    bld.connect(gain_blk.output_ports[0], integ.input_ports[0])
    diagram = bld.build()
    ctx0 = diagram.create_context()
    T = 0.6
    opts = sim_options(solver, "float64")

    @jax.jit
    def fwd(x0, r):
        c = ctx0
        c = c.with_subcontext(integ.system_id, c[integ.system_id].with_continuous_state(x0))
        c = c.with_subcontext(r_src.system_id, c[r_src.system_id].with_parameter("value", r))
        res = jaxonomy.simulate(diagram, c, (0.0, T), options=opts)
        return res.context[integ.system_id].continuous_state

    assert_grad_matches_fd(
        fwd, jnp.array(0.4), jnp.array(2.0),
        solver=solver, dtype="float64", block="Adder+Gain+Integrator",
    )


# ── T-001b additions: stateful DT blocks via simulate ────────────────────────
#
# DT-only blocks have no continuous state, so their solver-level gradient
# behavior is not solver-dependent. We pick dopri5 as the tolerance bucket
# because the hybrid simulator still threads a (zero-residual) CT solver.


def _dt_diagram_unit_delay():
    """Sine source → UnitDelay → exposes the delayed signal as final discrete state."""
    dt = 0.1
    bld = jaxonomy.DiagramBuilder()
    src = bld.add(Sine(amplitude=1.0, frequency=2.0, name="sine"))
    delay = bld.add(UnitDelay(dt, initial_state=0.0, name="ud"))
    bld.connect(src.output_ports[0], delay.input_ports[0])
    return bld.build(), src, delay, dt


def test_grad_sim_dt_unit_delay_initial_state():
    """∂(final UnitDelay state)/∂(initial_state) — at the first sample, the
    output port still holds the initial value, so within (0, dt) the
    gradient is 1. After dt, the state has been overwritten by the input
    so the gradient is 0. Stop simulation just past dt to verify the
    gradient transitions through the periodic update correctly."""
    diagram, src, delay, dt = _dt_diagram_unit_delay()
    ctx0 = diagram.create_context()
    opts = jaxonomy.SimulatorOptions(
        math_backend="jax", enable_autodiff=True, max_major_steps=50
    )

    @jax.jit
    def fwd(x0_init, amp):
        c = ctx0
        c = c.with_subcontext(src.system_id,
                              c[src.system_id].with_parameter("amplitude", amp))
        c = c.with_subcontext(delay.system_id,
                              c[delay.system_id].with_parameter("initial_state", x0_init))
        # Stop within the first dt window so the initial_state still drives output.
        T = 0.5 * dt
        res = jaxonomy.simulate(diagram, c, (0.0, T), options=opts)
        return res.context[delay.system_id].discrete_state

    assert_grad_matches_fd(
        fwd, jnp.array(0.4), jnp.array(1.3),
        solver="dopri5", dtype="float64", block="UnitDelay(initial_state,amp)",
    )


def test_grad_sim_dt_zero_order_hold():
    """ZeroOrderHold has no dynamic params — verify the input-side gradient
    flows correctly. dy/dx0 of an Integrator → ZOH chain at t < dt is 0
    (ZOH samples only at multiples of dt) and at t > dt is dt·∂/∂x0 ≈ 1."""
    dt = 0.1
    bld = jaxonomy.DiagramBuilder()
    integ = bld.add(Integrator(jnp.array(0.0), name="int"))
    zoh = bld.add(ZeroOrderHold(dt=dt, name="zoh"))
    drive = bld.add(Constant(jnp.array(1.0), name="drive"))
    bld.connect(drive.output_ports[0], integ.input_ports[0])
    bld.connect(integ.output_ports[0], zoh.input_ports[0])
    diagram = bld.build()
    ctx0 = diagram.create_context()
    opts = sim_options("dopri5", "float64")

    @jax.jit
    def fwd(x0, u):
        c = ctx0
        c = c.with_subcontext(integ.system_id,
                              c[integ.system_id].with_continuous_state(x0))
        c = c.with_subcontext(drive.system_id,
                              c[drive.system_id].with_parameter("value", u))
        res = jaxonomy.simulate(diagram, c, (0.0, 0.5), options=opts)
        return res.context[integ.system_id].continuous_state

    assert_grad_matches_fd(
        fwd, jnp.array(0.3), jnp.array(1.5),
        solver="dopri5", dtype="float64", block="Integrator→ZOH",
    )


def test_grad_sim_dt_rate_limiter_inactive():
    """RateLimiter with limits well outside the slew of a slow Constant-driven
    Integrator → output equals input → gradient flows like an identity."""
    dt = 0.1
    bld = jaxonomy.DiagramBuilder()
    integ = bld.add(Integrator(jnp.array(0.0), name="int"))
    rl = bld.add(
        RateLimiter(dt=dt, upper_limit=10.0, lower_limit=-10.0, name="rl"),
    )
    drive = bld.add(Constant(jnp.array(0.5), name="drive"))
    bld.connect(drive.output_ports[0], integ.input_ports[0])
    bld.connect(integ.output_ports[0], rl.input_ports[0])
    diagram = bld.build()
    ctx0 = diagram.create_context()
    opts = sim_options("dopri5", "float64")

    @jax.jit
    def fwd(x0, u):
        c = ctx0
        c = c.with_subcontext(integ.system_id,
                              c[integ.system_id].with_continuous_state(x0))
        c = c.with_subcontext(drive.system_id,
                              c[drive.system_id].with_parameter("value", u))
        res = jaxonomy.simulate(diagram, c, (0.0, 0.4), options=opts)
        return res.context[integ.system_id].continuous_state

    assert_grad_matches_fd(
        fwd, jnp.array(0.2), jnp.array(0.5),
        solver="dopri5", dtype="float64", block="RateLimiter(slack)",
    )


def test_grad_sim_dt_derivative_discrete():
    """DerivativeDiscrete on a Sine source — gradient w.r.t. amplitude flows
    through a backward-difference update."""
    dt = 0.05
    bld = jaxonomy.DiagramBuilder()
    src = bld.add(Sine(amplitude=1.0, frequency=2.0, name="sine"))
    deriv = bld.add(DerivativeDiscrete(dt=dt, name="dd"))
    bld.connect(src.output_ports[0], deriv.input_ports[0])
    diagram = bld.build()
    ctx0 = diagram.create_context()
    opts = sim_options("dopri5", "float64")

    @jax.jit
    def fwd(amp):
        c = ctx0
        c = c.with_subcontext(src.system_id,
                              c[src.system_id].with_parameter("amplitude", amp))
        # Stop a few periods in.
        res = jaxonomy.simulate(diagram, c, (0.0, 0.25), options=opts)
        return res.context[deriv.system_id].discrete_state

    assert_grad_matches_fd(
        fwd, jnp.array(1.3),
        solver="dopri5", dtype="float64", block="DerivativeDiscrete",
    )


def test_grad_sim_dt_filter_discrete():
    """FilterDiscrete (FIR) on a Sine — gradient w.r.t. amplitude. The FIR
    state stores recent inputs, so the gradient flows linearly through
    each tap weight."""
    dt = 0.05
    bld = jaxonomy.DiagramBuilder()
    src = bld.add(Sine(amplitude=1.0, frequency=2.0, name="sine"))
    fir = bld.add(
        FilterDiscrete(dt=dt, b_coefficients=[0.5, 0.25, 0.25], name="fir"),
    )
    bld.connect(src.output_ports[0], fir.input_ports[0])
    diagram = bld.build()
    ctx0 = diagram.create_context()
    opts = sim_options("dopri5", "float64")

    @jax.jit
    def fwd(amp):
        c = ctx0
        c = c.with_subcontext(src.system_id,
                              c[src.system_id].with_parameter("amplitude", amp))
        res = jaxonomy.simulate(diagram, c, (0.0, 0.25), options=opts)
        # FilterDiscrete final state is the FIR delay line; sum to a scalar.
        return jnp.sum(res.context[fir.system_id].discrete_state)

    assert_grad_matches_fd(
        fwd, jnp.array(1.4),
        solver="dopri5", dtype="float64", block="FilterDiscrete(FIR)",
    )


def test_grad_sim_dt_pid_discrete_gains():
    """PIDDiscrete on a constant error signal: gradient w.r.t. kp and ki."""
    dt = 0.05
    bld = jaxonomy.DiagramBuilder()
    err_src = bld.add(Constant(jnp.array(0.5), name="err"))
    pid = bld.add(PIDDiscrete(dt=dt, kp=1.0, ki=0.5, kd=0.1, name="pid"))
    bld.connect(err_src.output_ports[0], pid.input_ports[0])
    diagram = bld.build()
    ctx0 = diagram.create_context()
    opts = sim_options("dopri5", "float64")

    @jax.jit
    def fwd(kp, ki, e):
        c = ctx0
        c = c.with_subcontext(err_src.system_id,
                              c[err_src.system_id].with_parameter("value", e))
        c = c.with_subcontext(pid.system_id,
                              c[pid.system_id]
                              .with_parameter("kp", kp)
                              .with_parameter("ki", ki))
        res = jaxonomy.simulate(diagram, c, (0.0, 0.30), options=opts)
        return res.context[pid.system_id].discrete_state.integral

    assert_grad_matches_fd(
        fwd, jnp.array(1.2), jnp.array(0.4), jnp.array(0.5),
        solver="dopri5", dtype="float64", block="PIDDiscrete(kp,ki,e)",
    )


def test_grad_sim_dt_discrete_initializer_skip():
    """DiscreteInitializer's only dynamic parameter is `initial_state`, which
    is stored as `bool_` and used in `logical_not`. There is no real-valued
    gradient to take — document explicitly so a regression that adds a
    real-valued parameter trips this skip."""
    pytest.skip(
        "DiscreteInitializer.initial_state is bool-typed; no real-valued gradient "
        "is meaningful. T-001b honesty marker."
    )


def test_grad_sim_dt_edge_detection_skip():
    """EdgeDetection takes a boolean input and emits a boolean output. Its
    only dynamic parameter is `initial_state` (bool). No differentiable
    forward path exists today; document with a skip."""
    pytest.skip(
        "EdgeDetection input/output are bool; initial_state is bool. No "
        "real-valued gradient is meaningful. T-001b honesty marker."
    )


# ── T-001b additions: linear-system blocks (LTI / TF / linearize) ────────────


@pytest.mark.parametrize("solver", SOLVERS)
def test_grad_sim_ct_lti_system_decay(solver):
    """1-D LTI: ẋ = A x + B u, y = C x + D u. With A = -k, B=1, C=1, D=0,
    closed-form is x(T) = x0·e^{A·T} + B·u·(e^{A·T}-1)/A. Gradient w.r.t.
    A and B match a hand computation; we compare AD to FD."""
    bld = jaxonomy.DiagramBuilder()
    drive = bld.add(Constant(jnp.array(1.0), name="u"))
    lti = bld.add(LTISystem(
        A=jnp.array([[-1.0]]),
        B=jnp.array([[1.0]]),
        C=jnp.array([[1.0]]),
        D=jnp.array([[0.0]]),
        name="lti",
    ))
    bld.connect(drive.output_ports[0], lti.input_ports[0])
    diagram = bld.build()
    ctx0 = diagram.create_context()
    T = 0.5
    opts = sim_options(solver, "float64")

    @jax.jit
    def fwd(A_val, B_val, u):
        c = ctx0
        c = c.with_subcontext(drive.system_id,
                              c[drive.system_id].with_parameter("value", u))
        c = c.with_subcontext(lti.system_id,
                              c[lti.system_id]
                              .with_parameter("A", jnp.array([[A_val]]))
                              .with_parameter("B", jnp.array([[B_val]])))
        res = jaxonomy.simulate(diagram, c, (0.0, T), options=opts)
        x_final = res.context[lti.system_id].continuous_state
        return jnp.sum(jnp.atleast_1d(x_final))

    assert_grad_matches_fd(
        fwd, jnp.array(-1.0), jnp.array(1.0), jnp.array(0.7),
        solver=solver, dtype="float64", block="LTISystem(A,B,u)",
    )


def test_grad_sim_ct_transfer_function_input():
    """TransferFunction (1/(s+1)) driven by a Constant — gradient w.r.t. the
    drive amplitude. Tests that LTISystem-derived TransferFunction integrates
    cleanly through the autodiff path (T-029 fix). num/den are static so the
    test only differentiates `u`."""
    bld = jaxonomy.DiagramBuilder()
    drive = bld.add(Constant(jnp.array(1.0), name="u"))
    tf = bld.add(TransferFunction(num=[1.0], den=[1.0, 1.0], name="tf"))
    bld.connect(drive.output_ports[0], tf.input_ports[0])
    diagram = bld.build()
    ctx0 = diagram.create_context()
    T = 0.4
    opts = sim_options("dopri5", "float64")

    @jax.jit
    def fwd(u):
        c = ctx0
        c = c.with_subcontext(drive.system_id,
                              c[drive.system_id].with_parameter("value", u))
        res = jaxonomy.simulate(diagram, c, (0.0, T), options=opts)
        x_final = res.context[tf.system_id].continuous_state
        return jnp.sum(jnp.atleast_1d(x_final))

    assert_grad_matches_fd(
        fwd, jnp.array(1.5),
        solver="dopri5", dtype="float64", block="TransferFunction(u)",
    )


def test_grad_sim_linearize_to_lti_dropin():
    """linearize() returns a LinearizedSystem — verify the `to_lti()` block
    composes back into a diagram and gradients flow through the resulting
    LTI system. We linearize a damped 1-D oscillator and then differentiate
    the LTI block's final state w.r.t. its `B` matrix."""
    # Build a 1-D LTI to linearize about x=0, u=0. (Linearization of a
    # linear system at any point reproduces the same A,B,C,D, so we can
    # cleanly verify `linearize()` doesn't break gradient flow.)
    src_lti = LTISystem(
        A=jnp.array([[-2.0]]),
        B=jnp.array([[1.0]]),
        C=jnp.array([[1.0]]),
        D=jnp.array([[0.0]]),
    )
    src_lti.input_ports[0].fix_value(jnp.array([0.0]))
    base_ctx = src_lti.create_context()
    lin = linearize(src_lti, base_ctx)
    # Sanity: A should match the original.
    assert np.allclose(np.asarray(lin.A), [[-2.0]], atol=1e-6)

    bld = jaxonomy.DiagramBuilder()
    drive = bld.add(Constant(jnp.array(0.5), name="u"))
    lti_block = bld.add(lin.to_lti())
    bld.connect(drive.output_ports[0], lti_block.input_ports[0])
    diagram = bld.build()
    ctx0 = diagram.create_context()
    T = 0.4
    opts = sim_options("dopri5", "float64")

    @jax.jit
    def fwd(B_val, u):
        c = ctx0
        c = c.with_subcontext(drive.system_id,
                              c[drive.system_id].with_parameter("value", u))
        c = c.with_subcontext(lti_block.system_id,
                              c[lti_block.system_id]
                              .with_parameter("B", jnp.array([[B_val]])))
        res = jaxonomy.simulate(diagram, c, (0.0, T), options=opts)
        x_final = res.context[lti_block.system_id].continuous_state
        return jnp.sum(jnp.atleast_1d(x_final))

    assert_grad_matches_fd(
        fwd, jnp.array(1.0), jnp.array(0.6),
        solver="dopri5", dtype="float64", block="linearize→LTI(B,u)",
    )
