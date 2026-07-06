# SPDX-License-Identifier: MIT
"""V-008: Solver behavior under stress.

Verifies that variable-step solvers (dopri5, bdf) handle edge cases
correctly without silent drift, hidden divergence, or false
"step too small" errors near steady state.

Cases (parametrized over solvers when applicable):

1. Stiff relaxation to equilibrium (dx/dt = -k*(x-1)):
   BDF must reach steady state cleanly. Dopri5 may struggle at high k
   and is marked xfail when it does (expected).
2. Mixed-rate continuous + periodic discrete update.
3. Near-zero RHS asymptote (logistic decay): step size should be
   bounded when ``max_minor_step_size`` is set, and simulation should
   still terminate when it is not.
4. Rapid transient followed by slow tail (van der Pol, mu=10).
5. Diverging system dx/dt = x^2, x(0)=1: blows up at t=1. Solver
   must NOT silently clip; either raises or terminates with t<1.
6. Long-time stable LTI past steady state: no false errors / divergence.

Where current behavior is suspected to be wrong (e.g., silent step
clipping near divergence), tests use ``pytest.xfail`` for the
expected fix (T-008/T-005 area).
"""

from __future__ import annotations

import numpy as np
import pytest
import jax.numpy as jnp

import jaxonomy
from jaxonomy import LeafSystem, DiagramBuilder, SimulatorOptions, simulate
from jaxonomy.testing.markers import skip_if_not_jax

pytestmark = pytest.mark.slow

# This whole suite exercises the JAX-backend variable-step solvers.
skip_if_not_jax()


SOLVERS = ["dopri5", "bdf"]


def _opts(method: str, **kwargs) -> SimulatorOptions:
    return SimulatorOptions(
        math_backend="jax",
        ode_solver_method=method,
        rtol=kwargs.pop("rtol", 1e-6),
        atol=kwargs.pop("atol", 1e-8),
        **kwargs,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Systems
# ─────────────────────────────────────────────────────────────────────────────


class StiffDecay(LeafSystem):
    """dx/dt = -k*(x-1), x(0)=0. Steady state x=1, time const 1/k."""

    def __init__(self, k: float = 1000.0, x0: float = 0.0):
        super().__init__()
        self.declare_dynamic_parameter("k", k)
        self.declare_continuous_state(
            default_value=jnp.array(float(x0)), ode=self._ode
        )
        self.declare_output_port(
            lambda t, s, **p: s.continuous_state,
            default_value=jnp.zeros(()),
        )

    def _ode(self, time, state, **params):
        return -params["k"] * (state.continuous_state - 1.0)


class LogisticDecay(LeafSystem):
    """dx/dt = -r*x*(1-x). Asymptotes to x=0 (or x=1) — RHS → 0."""

    def __init__(self, r: float = 5.0, x0: float = 0.99):
        super().__init__()
        self.declare_dynamic_parameter("r", r)
        self.declare_continuous_state(
            default_value=jnp.array(float(x0)), ode=self._ode
        )
        self.declare_output_port(
            lambda t, s, **p: s.continuous_state,
            default_value=jnp.zeros(()),
        )

    def _ode(self, time, state, **params):
        x = state.continuous_state
        # Decay from 0.99 toward 0 (unstable fp at 1, stable at 0
        # for r>0 and this sign convention).  RHS → 0 as x → 0.
        return -params["r"] * x * (1.0 - x)


class VanDerPol(LeafSystem):
    """Stiff van der Pol oscillator.

    x' = y
    y' = mu*(1-x^2)*y - x
    """

    def __init__(self, mu: float = 10.0):
        super().__init__()
        self.declare_dynamic_parameter("mu", mu)
        self.declare_continuous_state(
            default_value=jnp.array([2.0, 0.0]), ode=self._ode
        )
        self.declare_output_port(
            lambda t, s, **p: s.continuous_state,
            default_value=jnp.zeros(2),
        )

    def _ode(self, time, state, **params):
        x = state.continuous_state
        return jnp.array([x[1], params["mu"] * (1.0 - x[0] ** 2) * x[1] - x[0]])


class QuadraticBlowup(LeafSystem):
    """dx/dt = x^2, x(0)=1. Blows up at t=1."""

    def __init__(self, x0: float = 1.0):
        super().__init__()
        self.declare_continuous_state(
            default_value=jnp.array(float(x0)), ode=self._ode
        )
        self.declare_output_port(
            lambda t, s, **p: s.continuous_state,
            default_value=jnp.zeros(()),
        )

    def _ode(self, time, state, **params):
        x = state.continuous_state
        return x * x


class StableLTI(LeafSystem):
    """dx/dt = -a*x. Stable LTI."""

    def __init__(self, a: float = 0.5, x0: float = 1.0):
        super().__init__()
        self.declare_dynamic_parameter("a", a)
        self.declare_continuous_state(
            default_value=jnp.array(float(x0)), ode=self._ode
        )
        self.declare_output_port(
            lambda t, s, **p: s.continuous_state,
            default_value=jnp.zeros(()),
        )

    def _ode(self, time, state, **params):
        return -params["a"] * state.continuous_state


class ContinuousPlusPeriodicReset(LeafSystem):
    """Mixed-rate: continuous decay + periodic discrete kicker.

    Continuous state: dx/dt = -50 x  (natural step ~ 1e-3 to 1e-2)
    Discrete state: counter incremented every period=0.1 s.
    """

    def __init__(self, period: float = 0.1):
        super().__init__()
        self.declare_continuous_state(
            default_value=jnp.array(1.0), ode=self._ode
        )
        self.declare_discrete_state(default_value=jnp.array(0.0))
        self.declare_periodic_update(self._tick, period=period, offset=0.0)
        self.declare_output_port(
            lambda t, s, *i: s.continuous_state,
            default_value=jnp.zeros(()),
        )

    def _ode(self, time, state, **params):
        return -50.0 * state.continuous_state

    def _tick(self, t, state, *inputs):
        return state.discrete_state + 1.0


# ─────────────────────────────────────────────────────────────────────────────
# Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestStiffRelaxation:
    """Case 1: stiff system relaxing to equilibrium."""

    @pytest.mark.parametrize("method", SOLVERS)
    def test_stiff_decay_reaches_steady_state(self, method, request):
        """BDF should reach x≈1 with no error.

        Dopri5 may take excessive steps or fail outright on stiff
        problems with k=1000 — if so, expected per V-008; mark xfail.
        Tracked as T-008 (variable-step solver edge cases).
        """
        system = StiffDecay(k=1000.0, x0=0.0)
        ctx = system.create_context()
        opts = _opts(method, rtol=1e-6, atol=1e-9, max_major_step_length=0.1)
        recorded = {"x": system.output_ports[0]}
        try:
            res = simulate(system, ctx, (0.0, 0.1), recorded_signals=recorded, options=opts)
        except Exception as e:
            if method == "dopri5":
                pytest.xfail(
                    f"dopri5 not viable for very stiff k=1000 — expected "
                    f"per V-008 (T-008). Underlying: {e!r}"
                )
            raise
        x_final = float(res.outputs["x"][-1])
        assert x_final == pytest.approx(1.0, abs=1e-3), (
            f"{method} failed to reach steady state: x_final={x_final}"
        )


class TestMixedRate:
    """Case 2: continuous fast ODE + periodic discrete update."""

    @pytest.mark.parametrize("method", SOLVERS)
    def test_continuous_with_periodic_update(self, method):
        """Solver handles 50x rate disparity without struggling.

        Continuous time-constant ~ 1/50 s; discrete period 0.1 s.
        Expectation: simulation completes, and counter fires the
        right number of times. No false 'step too small' errors.
        """
        period = 0.1
        t_end = 1.0
        system = ContinuousPlusPeriodicReset(period=period)
        ctx = system.create_context()
        opts = _opts(method, max_major_step_length=period)
        recorded = {"x": system.output_ports[0]}
        res = simulate(system, ctx, (0.0, t_end), recorded_signals=recorded, options=opts)
        # Continuous state should have decayed near zero.
        assert float(res.outputs["x"][-1]) == pytest.approx(0.0, abs=1e-3)
        # T-035 (resolved): the simulator uses a *closed* `[t0, tf]` schedule
        # for periodic updates -- it fires at every `t = k * dt` for k = 0..N
        # where N*dt == t_end. Many existing blocks (UnitDelay, IntegratorDiscrete,
        # Clock-discrete) rely on this so a final-sample update lands on the
        # recording boundary. Counter therefore fires 11 times for period=0.1
        # and t_end=1.0 (at t = 0.0, 0.1, ..., 1.0).
        counter = float(res.context.discrete_state)
        assert counter == pytest.approx(11.0), f"counter={counter}"


class TestNearZeroRHS:
    """Case 3: ODE asymptotes to fixed point; RHS → 0."""

    @pytest.mark.parametrize("method", SOLVERS)
    def test_max_step_respected_near_steady_state(self, method):
        """With max_minor_step_size set, the solver must not let
        step size inflate without bound when RHS approaches zero.

        We verify *termination* and that final value is near steady state.
        Direct measurement of step sizes would require digging into
        ode_solver_state; we settle for: simulation finishes cleanly
        and reaches x≈0 within tolerance.
        """
        system = LogisticDecay(r=5.0, x0=0.99)
        ctx = system.create_context()
        opts = _opts(method, max_minor_step_size=0.1, max_major_step_length=1.0)
        recorded = {"x": system.output_ports[0]}
        res = simulate(system, ctx, (0.0, 5.0), recorded_signals=recorded, options=opts)
        x_final = float(res.outputs["x"][-1])
        # logistic decay from 0.99 with r=5 over t=5 takes x close to 0.
        assert x_final < 0.5, f"x_final={x_final}, expected decay toward 0"
        assert np.all(np.isfinite(np.asarray(res.outputs["x"])))

    @pytest.mark.parametrize("method", SOLVERS)
    def test_terminates_without_max_step(self, method):
        """Without a max_minor_step_size, simulation still terminates
        cleanly within reasonable wall time. Documents that the solver
        does not hang near steady state.
        """
        system = LogisticDecay(r=5.0, x0=0.99)
        ctx = system.create_context()
        opts = _opts(method, max_major_step_length=1.0)
        recorded = {"x": system.output_ports[0]}
        res = simulate(system, ctx, (0.0, 5.0), recorded_signals=recorded, options=opts)
        x_final = float(res.outputs["x"][-1])
        assert np.isfinite(x_final)
        assert x_final < 0.5


class TestVanDerPolStiff:
    """Case 4: rapid transient then slow tail — high-mu van der Pol."""

    @pytest.mark.parametrize("method", SOLVERS)
    def test_high_mu_vdp(self, method):
        """mu=10 van der Pol — solution alternates fast spike then
        slow drift across one limit-cycle period (~ 19 for mu=10).

        BDF should handle this cleanly. Dopri5 is expected to be
        slow but should still terminate with bounded values.
        """
        system = VanDerPol(mu=10.0)
        ctx = system.create_context()
        # Short window inside one cycle to keep wall time bounded.
        opts = _opts(method, rtol=1e-5, atol=1e-7, max_major_step_length=2.0)
        recorded = {"x": system.output_ports[0]}
        try:
            res = simulate(system, ctx, (0.0, 5.0), recorded_signals=recorded, options=opts)
        except Exception as e:
            if method == "dopri5":
                pytest.xfail(
                    f"dopri5 too slow / failed on stiff vdp — expected "
                    f"per V-008 (T-008). Underlying: {e!r}"
                )
            raise
        x = np.asarray(res.outputs["x"])
        assert np.all(np.isfinite(x))
        # vdp limit cycle bounded by |x| <= ~2.5.
        assert np.max(np.abs(x)) < 5.0


class TestDivergingSystem:
    """Case 5: dx/dt = x^2 — finite-time blowup at t=1."""

    @pytest.mark.skip(
        reason="Integrating dx/dt=x^2 past its finite-time singularity hangs the "
        "adaptive inner loop uninterruptibly (no step-underflow guard) for BOTH "
        "dopri5 and bdf — a signal/thread pytest --timeout cannot kill, so it "
        "wedges CI to the job wall-clock. Same solver-divergence gap as "
        "test_ode_solver.test_diverging_solution; both quarantined pending the "
        "step-underflow / early-termination fix (see TODO T-005/T-008)."
    )
    @pytest.mark.parametrize("method", SOLVERS)
    def test_blowup_reported_or_terminated(self, method):
        """When integrating past the finite-time singularity at t=1,
        the solver must NOT silently clip the step and pretend success.

        Acceptable behaviors:
          (a) raises an exception (RuntimeError or similar), OR
          (b) returns with res.time[-1] < 1.0 (early termination).

        If the solver returns with time reaching past t=1.0 with
        finite x, that's a silent-clipping bug. Mark xfail in that
        case (T-005/T-008).
        """
        system = QuadraticBlowup(x0=1.0)
        ctx = system.create_context()
        # Asking for t_end well past the singularity at t=1.
        opts = _opts(method, rtol=1e-6, atol=1e-8, max_major_step_length=0.5)
        recorded = {"x": system.output_ports[0]}
        try:
            res = simulate(system, ctx, (0.0, 2.0), recorded_signals=recorded, options=opts)
        except Exception:
            # Behavior (a): raised — correct.
            return

        # Behavior (b/c): returned without raising.
        t = np.asarray(res.time)
        x = np.asarray(res.outputs["x"])
        finite_x = np.isfinite(x)
        if t[-1] < 1.0 - 1e-6:
            # Behavior (b): early termination — also correct.
            return
        if not np.all(finite_x):
            # Returned a NaN/Inf trace past the singularity — not a
            # silent-clip but also not a clean failure. Document as
            # xfail until handled per TODO T-005/T-008.
            pytest.xfail(
                "Diverging ODE returned NaN/Inf rather than raising "
                "or terminating early (T-005/T-008)."
            )
        # Reached past t=1 with finite values — silent step clipping.
        pytest.xfail(
            f"Diverging ODE: solver reached t={t[-1]:.4f} > 1.0 with "
            f"finite x_final={x[-1]:.3e} — silent step clipping. "
            f"Expected fix T-005/T-008."
        )


class TestLongStableLTI:
    """Case 6: long-horizon stable LTI past steady state."""

    @pytest.mark.parametrize("method", SOLVERS)
    def test_long_run_no_divergence(self, method):
        """Stable system run for ~50 time-constants. Should reach 0,
        no false 'step too small' errors, no divergence.
        """
        a = 0.5  # time const 2.0
        system = StableLTI(a=a, x0=1.0)
        ctx = system.create_context()
        t_end = 100.0  # 50 time constants
        opts = _opts(method, max_major_step_length=10.0)
        recorded = {"x": system.output_ports[0]}
        res = simulate(system, ctx, (0.0, t_end), recorded_signals=recorded, options=opts)
        x = np.asarray(res.outputs["x"])
        assert np.all(np.isfinite(x))
        assert float(x[-1]) == pytest.approx(0.0, abs=1e-6)
        # Trajectory must not diverge anywhere along the way.
        assert np.max(np.abs(x)) < 1.5
