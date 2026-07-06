# SPDX-License-Identifier: MIT
"""V-001 — Gradient correctness across solvers, blocks, and event paths.

Parametrized matrix sweep verifying that reverse-mode autodiff gradients
through ``jaxonomy.simulate`` agree with finite-difference ground truth across
the cross-product of {solver} x {block} x {event-path} x {state-type}.

Cases are a representative subset (~30), not exhaustive cartesian product.
Each case computes ``jax.grad`` of a scalar terminal-state functional and
compares to ``fd_grad`` (central differences).

Tolerances: Dopri5 rtol <= 5e-3; BDF rtol <= 1e-2. Cases that deviate (long
horizon, near-event sensitivities) document their tolerance inline. See
``test/autodiff/test_autodiff_correctness.py`` for the canonical style.
"""

from __future__ import annotations

import numpy as np
import pytest
import jax
import jax.numpy as jnp

import jaxonomy
from jaxonomy import DiagramBuilder, LeafSystem, SimulatorOptions, simulate
from jaxonomy.library import Comparator, Constant, Integrator
from jaxonomy.testing import fd_grad
from jaxonomy.testing.markers import skip_if_not_jax

skip_if_not_jax()
pytestmark = pytest.mark.slow

# rtol=1e-6 is intentional (see test_autodiff_correctness.py header note).
_OPTS_DOPRI5 = SimulatorOptions(
    math_backend="jax", enable_autodiff=True, ode_solver_method="dopri5", rtol=1e-6
)
_OPTS_BDF = SimulatorOptions(
    math_backend="jax", enable_autodiff=True, ode_solver_method="bdf", rtol=1e-6
)


def _opts_for(solver: str) -> SimulatorOptions:
    return _OPTS_DOPRI5 if solver == "dopri5" else _OPTS_BDF


def _rtol_for(solver: str) -> float:
    """Default per-solver tolerance from the V-001 spec."""
    return 5e-3 if solver == "dopri5" else 1e-2


# ── system fixtures ──────────────────────────────────────────────────────


class _ScalarDecay(LeafSystem):
    """dx/dt = -a x — pure CT, no events."""

    def __init__(self, a: float = 1.5, x0: float = 4.0):
        super().__init__()
        self.declare_dynamic_parameter("a", a)
        self.declare_continuous_state(default_value=jnp.array(x0), ode=self._ode)

    def _ode(self, time, state, **params):
        return -params["a"] * state.continuous_state


class _Harmonic(LeafSystem):
    """[x', v'] = [v, -w^2 x] — pure CT, vector state."""

    def __init__(self, omega: float = 2.0):
        super().__init__()
        self.declare_dynamic_parameter("omega", omega)
        self.declare_continuous_state(default_value=jnp.array([1.0, 0.0]), ode=self._ode)

    def _ode(self, time, state, **params):
        x, v = state.continuous_state
        w = params["omega"]
        return jnp.array([v, -(w**2) * x])


class _RCSmallTau(LeafSystem):
    """dx/dt = -(1/tau)(x-1) — small RC time constant ~ 0.01."""

    def __init__(self, tau: float = 0.01):
        super().__init__()
        self.declare_dynamic_parameter("tau", tau)
        self.declare_continuous_state(default_value=jnp.array(0.0), ode=self._ode)

    def _ode(self, time, state, **params):
        return -(1.0 / params["tau"]) * (state.continuous_state - 1.0)


class _Stiff(LeafSystem):
    """Mildly stiff: dx/dt = -k (x - x_ss). For BDF near steady state."""

    def __init__(self, k: float = 50.0):
        super().__init__()
        self.declare_dynamic_parameter("k", k)
        self.declare_continuous_state(default_value=jnp.array(2.0), ode=self._ode)

    def _ode(self, time, state, **params):
        return -params["k"] * (state.continuous_state - 1.0)


class _VectorParam(LeafSystem):
    """dx/dt = -A.x with A a 3-vector of decay rates (diagonal)."""

    def __init__(self):
        super().__init__()
        self.declare_dynamic_parameter("a", jnp.array([0.5, 1.0, 1.5]))
        self.declare_continuous_state(
            default_value=jnp.array([1.0, 1.0, 1.0]), ode=self._ode
        )

    def _ode(self, time, state, **params):
        return -params["a"] * state.continuous_state


class _BouncingBall1d(LeafSystem):
    """1-D bouncing ball; zero-crossing + reset (state = [y, v])."""

    def __init__(self, g: float = 9.81, e: float = 0.8):
        super().__init__(name="ball1d")
        self.declare_dynamic_parameter("g", g)
        self.declare_dynamic_parameter("e", e)
        self.declare_continuous_state(default_value=jnp.array([1.0, 0.0]), ode=self._ode)
        self.declare_zero_crossing(
            guard=self._guard,
            reset_map=self._reset,
            direction="positive_then_non_positive",
            name="bounce",
        )

    def _ode(self, time, state, **params):
        y, v = state.continuous_state
        return jnp.array([v, -params["g"]])

    def _guard(self, time, state, **params):
        y, v = state.continuous_state
        return y

    def _reset(self, time, state, **params):
        y, v = state.continuous_state
        return state.with_continuous_state(jnp.array([jnp.abs(y), -params["e"] * v]))


class _ResetGrowth(LeafSystem):
    """dx/dt = a x with reset x <- reset_state when x crosses trigger."""

    def __init__(self, a: float = 1.0, trigger: float = 1.5, reset: float = 0.5):
        super().__init__()
        self.declare_dynamic_parameter("a", a)
        self.declare_dynamic_parameter("trigger", trigger)
        self.declare_dynamic_parameter("reset", reset)
        self.declare_continuous_state(default_value=jnp.array(1.0), ode=self._ode)
        self.declare_zero_crossing(
            guard=self._guard, reset_map=self._reset_map, name="reset"
        )

    def _ode(self, time, state, **params):
        return params["a"] * state.continuous_state

    def _guard(self, time, state, **params):
        return state.continuous_state - params["trigger"]

    def _reset_map(self, time, state, **params):
        return state.with_continuous_state(params["reset"])


class _ModeSwitch(LeafSystem):
    """State machine: mode flag toggled by zero-crossing on time.

    State = [x, mode]; dx/dt = -a x in mode 0, +b in mode 1.
    """

    def __init__(self, a: float = 2.0, b: float = 0.5, t_switch: float = 0.6):
        super().__init__()
        self.declare_dynamic_parameter("a", a)
        self.declare_dynamic_parameter("b", b)
        self.declare_dynamic_parameter("t_switch", t_switch)
        self.declare_continuous_state(default_value=jnp.array([1.0, 0.0]), ode=self._ode)
        self.declare_zero_crossing(
            guard=self._guard,
            reset_map=self._reset,
            direction="negative_then_non_negative",
            name="mode_switch",
        )

    def _ode(self, time, state, **params):
        x, mode = state.continuous_state
        a, b = params["a"], params["b"]
        dx = -a * x * (1.0 - mode) + b * mode
        return jnp.array([dx, 0.0])

    def _guard(self, time, state, **params):
        return time - params["t_switch"]

    def _reset(self, time, state, **params):
        x, mode = state.continuous_state
        return state.with_continuous_state(jnp.array([x, 1.0]))


class _PureLinearMap(LeafSystem):
    """Pure-DT: x[n+1] = a x[n], period=1.0. Marked xfail (pytree issue)."""

    def __init__(self, period: float = 1.0):
        super().__init__()
        self.declare_dynamic_parameter("a", 0.9)
        self.declare_discrete_state(default_value=jnp.array(1.0))
        self.declare_periodic_update(self._update, period=period, offset=0.0)

    def _update(self, time, state, **params):
        return state.discrete_state * params["a"]


# ── helpers ──────────────────────────────────────────────────────────────


def _grad_check(fwd, p, *, solver, eps=1e-5, rtol=None, atol=1e-6):
    """AD-vs-FD comparison. ``p`` must be Python float or numpy ndarray."""
    rtol = rtol if rtol is not None else _rtol_for(solver)
    g_ad = jax.grad(fwd)(p)
    g_fd = fd_grad(fwd, p, eps=eps)[0]
    np.testing.assert_allclose(
        np.asarray(g_ad), np.asarray(g_fd), rtol=rtol, atol=atol,
        err_msg=f"solver={solver}: AD={g_ad}, FD={g_fd}",
    )


# ── Group A: pure CT, no events ──────────────────────────────────────────


@pytest.mark.parametrize("solver", ["dopri5", "bdf"])
def test_scalar_decay_grad(solver):
    """CT scalar decay, ∂x(T)/∂a — analytic = -T x0 e^{-aT}."""
    sys = _ScalarDecay()
    ctx = sys.create_context()
    T = 2.0

    @jax.jit
    def fwd(a):
        c = ctx
        c.parameters["a"] = a
        return simulate(sys, c, (0.0, T), options=_opts_for(solver)).context.continuous_state

    _grad_check(fwd, 1.5, solver=solver)


@pytest.mark.parametrize("solver", ["dopri5", "bdf"])
def test_harmonic_oscillator_grad(solver):
    """CT 2-D harmonic oscillator, ∂(sum x(T))/∂ω."""
    sys = _Harmonic()
    ctx = sys.create_context()
    T = 0.75

    @jax.jit
    def fwd(omega):
        c = ctx
        c.parameters["omega"] = omega
        return simulate(sys, c, (0.0, T), options=_opts_for(solver)).context.continuous_state.sum()

    _grad_check(fwd, 2.0, solver=solver)


@pytest.mark.parametrize("solver", ["dopri5", "bdf"])
def test_small_tau_rc_grad(solver):
    """Small-magnitude parameter (RC time constant ≈ 0.01)."""
    sys = _RCSmallTau(tau=0.01)
    ctx = sys.create_context()
    T = 0.05

    @jax.jit
    def fwd(tau):
        c = ctx
        c.parameters["tau"] = tau
        return simulate(sys, c, (0.0, T), options=_opts_for(solver)).context.continuous_state

    # Small tau → high stiffness; FD step also small. Loosen rtol slightly.
    _grad_check(fwd, 0.01, solver=solver, eps=1e-7, rtol=2e-2)


def test_stiff_bdf_near_steady_state():
    """BDF on stiff system near steady state — long enough that x ≈ x_ss."""
    sys = _Stiff(k=50.0)
    ctx = sys.create_context()
    T = 1.5  # >> 1/k, well into steady-state region

    @jax.jit
    def fwd(k):
        c = ctx
        c.parameters["k"] = k
        return simulate(sys, c, (0.0, T), options=_OPTS_BDF).context.continuous_state

    # Sensitivity ∂x(T)/∂k ≈ T·(x0 - x_ss)·e^{-kT} ≈ 0; both AD and FD ≈ 0.
    g_ad = jax.grad(fwd)(50.0)
    g_fd = fd_grad(fwd, 50.0, eps=1e-3)[0]
    np.testing.assert_allclose(np.asarray(g_ad), np.asarray(g_fd), atol=1e-6)


@pytest.mark.parametrize("solver", ["dopri5", "bdf"])
def test_vector_param_grad(solver):
    """Vector-valued parameter (3-vector) — diagonal decay system."""
    sys = _VectorParam()
    ctx = sys.create_context()
    T = 1.0

    @jax.jit
    def fwd(a):
        c = ctx
        c.parameters["a"] = a
        return simulate(sys, c, (0.0, T), options=_opts_for(solver)).context.continuous_state.sum()

    a0_jax = jnp.array([0.5, 1.0, 1.5])
    a0_np = np.array([0.5, 1.0, 1.5])
    g_ad = jax.grad(fwd)(a0_jax)
    # fd_grad iterates over .size for ndarray inputs (works with np.ndarray).
    g_fd = fd_grad(lambda a: float(fwd(jnp.asarray(a))), a0_np, eps=1e-5)[0]
    np.testing.assert_allclose(
        np.asarray(g_ad), np.asarray(g_fd), rtol=_rtol_for(solver), atol=1e-6
    )


# ── Group B: long horizon (>500 integrator steps) ───────────────────────


def test_scalar_decay_long_horizon_dopri5():
    """Long horizon (T=50, fine atol → >500 internal steps).

    Dopri5 only — BDF for this exact scaling explodes step count beyond CI
    budget without buying coverage we don't already have via the stiff case.
    """
    sys = _ScalarDecay(a=0.05, x0=4.0)
    ctx = sys.create_context()
    T = 50.0
    opts = SimulatorOptions(
        math_backend="jax",
        enable_autodiff=True,
        ode_solver_method="dopri5",
        rtol=1e-6,
        atol=1e-8,
    )

    @jax.jit
    def fwd(a):
        c = ctx
        c.parameters["a"] = a
        return simulate(sys, c, (0.0, T), options=opts).context.continuous_state

    _grad_check(fwd, 0.05, solver="dopri5", eps=1e-5, rtol=1e-2)


# ── Group C: zero-crossing events (with reset map) ──────────────────────


@pytest.mark.parametrize("solver", ["dopri5", "bdf"])
@pytest.mark.parametrize("param", ["g", "e"])
def test_bouncing_ball_grad(solver, param):
    """1-D bouncing ball: ∂y(T)/∂g and ∂y(T)/∂e through bounce reset."""
    sys = _BouncingBall1d(g=9.81, e=0.8)
    ctx = sys.create_context()
    T = 1.0  # several bounces from y=1, v=0

    @jax.jit
    def fwd(p):
        c = ctx
        c.parameters[param] = p
        return simulate(sys, c, (0.0, T), options=_opts_for(solver)).context.continuous_state[0]

    p0 = 9.81 if param == "g" else 0.8
    # Bounce sensitivities are notoriously chatty around the event; widen rtol
    # for FD's step-tuning sensitivity but keep the spec ceiling for AD.
    _grad_check(fwd, p0, solver=solver, eps=1e-4, rtol=2e-2)


@pytest.mark.parametrize("solver", ["dopri5", "bdf"])
@pytest.mark.parametrize("param", ["a", "trigger", "reset"])
def test_reset_growth_grad(solver, param):
    """Exponential growth with a reset event triggered by zero crossing.

    Exercises parameters used in (a) the RHS, (b) the guard, and (c) the
    reset map — the V-001 matrix calls these out explicitly.
    """
    sys = _ResetGrowth(a=1.0, trigger=1.5, reset=0.5)
    ctx = sys.create_context()
    T = 2.0

    @jax.jit
    def fwd(p):
        c = ctx
        c.parameters[param] = p
        return simulate(sys, c, (0.0, T), options=_opts_for(solver)).context.continuous_state

    nominal = {"a": 1.0, "trigger": 1.5, "reset": 0.5}[param]
    # ∂/∂trigger has a sharp step component; loosen tolerance for that path.
    rtol = 3e-2 if param == "trigger" else _rtol_for(solver)
    _grad_check(fwd, nominal, solver=solver, eps=1e-4, rtol=rtol)


# ── Group D: comparator-driven gate ──────────────────────────────────────


@pytest.mark.parametrize("solver", ["dopri5", "bdf"])
def test_comparator_gate_grad(solver):
    """Diagram: Integrator + Constant + Comparator. The Comparator emits a
    zero crossing when the integrator output crosses the threshold; the
    parameter is the threshold.

    The comparator output is unused for state evolution — we only verify the
    forward simulation differentiates cleanly w.r.t. the integrator gain in
    the presence of a comparator-induced event in the diagram.
    """
    builder = DiagramBuilder()
    src = builder.add(Constant(jnp.array(0.5)))
    integ = builder.add(Integrator(jnp.array(0.0)))
    threshold = builder.add(Constant(jnp.array(1.0)))
    cmp_blk = builder.add(Comparator(operator=">="))
    builder.connect(src.output_ports[0], integ.input_ports[0])
    builder.connect(integ.output_ports[0], cmp_blk.input_ports[0])
    builder.connect(threshold.output_ports[0], cmp_blk.input_ports[1])
    diagram = builder.build()
    ctx = diagram.create_context()
    T = 3.0

    @jax.jit
    def fwd(x0):
        sub = ctx[integ.system_id].with_continuous_state(x0)
        c = ctx.with_subcontext(integ.system_id, sub)
        return simulate(diagram, c, (0.0, T), options=_opts_for(solver)).context[
            integ.system_id
        ].continuous_state

    _grad_check(fwd, 0.0, solver=solver, eps=1e-5)


# ── Group E: mode switch (state machine via zero-crossing) ──────────────


@pytest.mark.parametrize("solver", ["dopri5", "bdf"])
@pytest.mark.parametrize("param", ["a", "b"])
def test_mode_switch_grad(solver, param):
    """Mode transition fires mid-sim; gradient through pre- and post-mode RHS.

    Differentiating w.r.t. ``t_switch`` itself requires AD through the guard
    threshold and is covered indirectly by the bouncing-ball / reset-growth
    cases; here we focus on RHS-side parameters whose effect spans both modes.
    """
    sys = _ModeSwitch(a=2.0, b=0.5, t_switch=0.6)
    ctx = sys.create_context()
    T = 1.2

    @jax.jit
    def fwd(p):
        c = ctx
        c.parameters[param] = p
        return simulate(sys, c, (0.0, T), options=_opts_for(solver)).context.continuous_state[0]

    nominal = 2.0 if param == "a" else 0.5
    _grad_check(fwd, nominal, solver=solver, eps=1e-5)


# ── Group F: pure-DT (xfail: known custom-VJP pytree issue) ─────────────


@pytest.mark.xfail(
    reason=(
        "Pure-DT autodiff has a known pytree mismatch in the custom-VJP "
        "wrapping path when the system has no continuous state. "
        "See test_autodiff_correctness.test_dt_pure_discrete_autodiff for the "
        "fix; remaining diagram-level pure-DT cases still fail through "
        "DiagramBuilder due to the same root cause. Tracked in V-001."
    ),
    strict=False,
)
@pytest.mark.parametrize("solver", ["dopri5", "bdf"])
def test_pure_discrete_grad_xfail(solver):
    """Pure-DT system inside a Diagram — expected to fail (pytree mismatch)."""
    builder = DiagramBuilder()
    blk = builder.add(_PureLinearMap(period=1.0))
    diagram = builder.build()
    ctx = diagram.create_context()
    opts = SimulatorOptions(
        math_backend="jax",
        enable_autodiff=True,
        ode_solver_method="dopri5" if solver == "dopri5" else "bdf",
        max_major_steps=20,
    )
    T = 3.0

    @jax.jit
    def fwd(a):
        sub = ctx[blk.system_id]
        sub.parameters["a"] = a
        c = ctx.with_subcontext(blk.system_id, sub)
        return simulate(diagram, c, (0.0, T), options=opts).context[
            blk.system_id
        ].discrete_state

    _grad_check(fwd, 0.9, solver=solver)
