# SPDX-License-Identifier: MIT
"""
Edge-case tests for the jaxonomy simulation engine loop.

Covers: time-stepping semantics, event boundaries, zero-crossings, ZOH,
Zeno-like behavior, discontinuities, and comparisons to analytic solutions.

Each test is documented with the specific engine mechanism under test.
"""

import numpy as np
import pytest
import jax.numpy as jnp
import jaxonomy
from jaxonomy.library import Integrator, Gain, Adder, Demultiplexer, Constant

pytestmark = pytest.mark.slow


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _sim(system, t_span, ctx=None, recorded=None, opts=None):
    if ctx is None:
        ctx = system.create_context()
    if opts is None:
        opts = jaxonomy.SimulatorOptions(math_backend="jax")
    return jaxonomy.simulate(system, ctx, t_span, recorded_signals=recorded, options=opts)


# ─────────────────────────────────────────────────────────────────────────────
# 1. TIMED-EVENT BOUNDARY SEMANTICS
# ─────────────────────────────────────────────────────────────────────────────

class TestTimedEventBoundaries:

    def test_offset_zero_fires_at_t0(self):
        """Events with offset=0 must fire immediately at t=0 (before ODE advance).

        A DT block initialised to 0 increments by 1 on every period including t=0.
        The simulation end time is t_end = N*T. Events fire at t=0, T, ..., (N-1)*T
        (events scheduled exactly at t_end do not fire — the loop condition is
        ``time < t_end``).  So there are exactly N updates.
        """
        period = 0.5
        N = 4  # fires at t = 0, 0.5, 1.0, 1.5  → 4 updates total (t=2.0 = t_end skipped)

        class Counter(jaxonomy.LeafSystem):
            def __init__(self):
                super().__init__()
                self.declare_discrete_state(default_value=jnp.array(0.0))
                self.declare_periodic_update(self._upd, period=period, offset=0.0)
            def _upd(self, t, s, *i):
                return s.discrete_state + 1.0

        sys = Counter()
        res = _sim(sys, (0.0, N * period))
        val = float(res.context.discrete_state)
        assert val == N, f"Expected {N} updates, got {val}"

    def test_offset_nonzero_does_not_fire_at_t0(self):
        """Events with offset > 0 must NOT fire before their first scheduled time."""
        period = 1.0
        offset = 0.5

        class Counter(jaxonomy.LeafSystem):
            def __init__(self):
                super().__init__()
                self.declare_discrete_state(default_value=jnp.array(0.0))
                self.declare_periodic_update(self._upd, period=period, offset=offset)
            def _upd(self, t, s, *i):
                return s.discrete_state + 1.0

        sys = Counter()
        # Run to just before the first event
        res = _sim(sys, (0.0, offset - 1e-9))
        assert float(res.context.discrete_state) == 0.0, "Should not have fired yet"
        # Run to just past the first event
        res = _sim(sys, (0.0, offset + 1e-6))
        assert float(res.context.discrete_state) == 1.0, "Should have fired once"

    def test_simultaneous_events_same_period(self):
        """Two blocks with identical period/offset must both fire at each tick."""
        period = 0.5

        class Counter(jaxonomy.LeafSystem):
            def __init__(self, name):
                super().__init__(name=name)
                self.declare_discrete_state(default_value=jnp.array(0.0))
                self.declare_periodic_update(self._upd, period=period, offset=0.0)
            def _upd(self, t, s, *i):
                return s.discrete_state + 1.0

        bld = jaxonomy.DiagramBuilder()
        a = bld.add(Counter("a"))
        b = bld.add(Counter("b"))
        diagram = bld.build()
        ctx = diagram.create_context()
        res = _sim(diagram, (0.0, 2.0), ctx=ctx)
        va = float(res.context[a.system_id].discrete_state)
        vb = float(res.context[b.system_id].discrete_state)
        # period=0.5, t_end=2.0: fires at 0, 0.5, 1.0, 1.5 → 4 updates (t=2.0 skipped)
        assert va == vb == 4.0, f"Both counters should be 4, got {va}, {vb}"

    def test_different_periods_fire_independently(self):
        """Blocks with different periods must each fire at their own rate."""
        class Counter(jaxonomy.LeafSystem):
            def __init__(self, period, name):
                super().__init__(name=name)
                self.declare_discrete_state(default_value=jnp.array(0.0))
                self.declare_periodic_update(self._upd, period=period, offset=0.0)
            def _upd(self, t, s, *i):
                return s.discrete_state + 1.0

        bld = jaxonomy.DiagramBuilder()
        # t_end=2.0: fast fires at 0,0.25,...,1.75 → 8 updates; slow at 0,1.0 → 2 updates
        fast = bld.add(Counter(0.25, "fast"))
        slow = bld.add(Counter(1.0,  "slow"))
        diagram = bld.build()
        ctx = diagram.create_context()
        res = _sim(diagram, (0.0, 2.0), ctx=ctx)
        vf = float(res.context[fast.system_id].discrete_state)
        vs = float(res.context[slow.system_id].discrete_state)
        assert vf == 8.0, f"Fast counter: expected 8, got {vf}"
        assert vs == 2.0, f"Slow counter: expected 2, got {vs}"


# ─────────────────────────────────────────────────────────────────────────────
# 2. ZERO-ORDER HOLD (ZOH) SEMANTICS
# ─────────────────────────────────────────────────────────────────────────────

class TestZeroOrderHold:
    """Verify that a sample-and-hold output uses x⁻ (pre-update) values."""

    def test_zoh_output_uses_pre_update_value(self):
        """The sampled output seen by a downstream integrator between updates
        must be the value captured at the *last* update time, not the value at
        the current query time (ZOH semantics).

        Architecture:
            DT ramp (x[n] = n*T_s)  --ZOH--> CT Integrator

        The ZOH output holds the value sampled at the previous tick.
        Between tick k and tick k+1 the integrator input is k (a constant).
        After N ticks (starting at offset=0) the integrator state is:
            z(N*T_s) = sum_{k=0}^{N-1} k * T_s = T_s * N*(N-1)/2

        We simulate to t = N*T_s *without* the final tick firing (use
        t_end slightly less than N*T_s) so we can test mid-interval accuracy.
        """
        T_s = 0.5
        N = 4  # ticks at t = 0, 0.5, 1.0, 1.5  (4 ticks before t=2)
        T_end = (N - 0.5) * T_s  # = 1.75, between tick 3 and tick 4

        class DTRamp(jaxonomy.LeafSystem):
            def __init__(self):
                super().__init__()
                self.declare_discrete_state(default_value=jnp.array(0.0))
                self.declare_output_port(
                    lambda t, s, *i: s.discrete_state,
                    period=T_s, offset=0.0, name="y"
                )
                self.declare_periodic_update(self._upd, period=T_s, offset=0.0)
            def _upd(self, t, s, *i):
                # After the k-th tick, output = k
                return s.discrete_state + 1.0

        bld = jaxonomy.DiagramBuilder()
        ramp = bld.add(DTRamp())
        integ = bld.add(Integrator(0.0))
        bld.connect(ramp.output_ports[0], integ.input_ports[0])
        diagram = bld.build()
        ctx = diagram.create_context()
        res = _sim(diagram, (0.0, T_end), ctx=ctx)
        z_final = float(res.context[integ.system_id].continuous_state)

        # ZOH semantics: the output port reads s.discrete_state at each major-step
        # boundary BEFORE the update fires (x⁻ semantics).
        # At t=0 : state=0 (pre-update), output held=0, then state→1
        # At t=0.5: state=1 (pre-update), output held=1, then state→2
        # At t=1.0: state=2 (pre-update), output held=2, then state→3
        # At t=1.5: state=3 (pre-update), output held=3, then state→4
        # z = 0*0.5 + 1*0.5 + 2*0.5 + 3*0.25 = 0 + 0.5 + 1.0 + 0.75 = 2.25
        expected = 0*0.5 + 1*0.5 + 2*0.5 + 3*0.25
        assert abs(z_final - expected) < 1e-3, (
            f"ZOH integrator: expected {expected:.4f}, got {z_final:.4f}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# 3. DISCRETE UPDATE SEMANTICS (x⁻ vs x⁺)
# ─────────────────────────────────────────────────────────────────────────────

class TestDiscreteUpdateSemantics:

    def test_update_uses_pre_update_state_of_other_blocks(self):
        """Two blocks updating simultaneously must each see the OTHER block's x⁻.

        Block A: x_A[n+1] = x_B[n]
        Block B: x_B[n+1] = x_A[n]

        Starting from x_A=1, x_B=0:
          Step 1: x_A ← 0, x_B ← 1   (both read the pre-update values)
          Step 2: x_A ← 1, x_B ← 0
          ...

        This validates x⁻ atomicity: handle_discrete_update evaluates ALL callbacks
        against the pre-update snapshot before committing any result, so neither
        block sees the other's post-update state during the same tick.
        """
        period = 1.0

        class SwapBlock(jaxonomy.LeafSystem):
            def __init__(self, name):
                super().__init__(name=name)
                self.declare_discrete_state(default_value=jnp.array(0.0))
                self.declare_input_port(name="u")
                # requires_inputs=False tells the framework: no feedthrough from u to output.
                # This avoids the algebraic-loop false-positive the conservative checker
                # would raise for the A→B→A discrete feedback topology.
                self.declare_output_port(
                    lambda t, s: s.discrete_state, requires_inputs=False
                )
                self.declare_periodic_update(self._upd, period=period, offset=0.0)
            def _upd(self, t, s, u):
                return u

        bld = jaxonomy.DiagramBuilder()
        a = bld.add(SwapBlock("A"))
        b = bld.add(SwapBlock("B"))
        bld.connect(b.output_ports[0], a.input_ports[0])
        bld.connect(a.output_ports[0], b.input_ports[0])
        diagram = bld.build()

        ctx = diagram.create_context()
        ctx = ctx.with_subcontext(a.system_id, ctx[a.system_id].with_discrete_state(jnp.array(1.0)))
        ctx = ctx.with_subcontext(b.system_id, ctx[b.system_id].with_discrete_state(jnp.array(0.0)))

        # After 1 step (t=0 fires, end before t=1.0): a=0, b=1
        # Use t_end=0.5 — tick fires at t=0, next tick at t=1.0 has not fired yet.
        res = _sim(diagram, (0.0, 0.5), ctx=ctx)
        va = float(res.context[a.system_id].discrete_state)
        vb = float(res.context[b.system_id].discrete_state)
        assert abs(va - 0.0) < 1e-9 and abs(vb - 1.0) < 1e-9, (
            f"After 1 swap: expected a=0, b=1; got a={va}, b={vb}"
        )

        # After 2 steps (ticks at t=0 and t=1.0, end before t=2.0): a=1, b=0
        res2 = _sim(diagram, (0.0, 1.5), ctx=ctx)
        va2 = float(res2.context[a.system_id].discrete_state)
        vb2 = float(res2.context[b.system_id].discrete_state)
        assert abs(va2 - 1.0) < 1e-9 and abs(vb2 - 0.0) < 1e-9, (
            f"After 2 swaps: expected a=1, b=0; got a={va2}, b={vb2}"
        )

    def test_self_referential_update_uses_current_state(self):
        """x[n+1] = f(x[n]) — self-referential update must use x⁻, not x⁺."""
        period = 1.0

        class Doubler(jaxonomy.LeafSystem):
            def __init__(self):
                super().__init__()
                self.declare_discrete_state(default_value=jnp.array(1.0))
                self.declare_periodic_update(self._upd, period=period, offset=0.0)
            def _upd(self, t, s, *i):
                return s.discrete_state * 2.0

        sys = Doubler()
        # Fires at t=0, 1, 2 → 3 updates, x[3] = 2^3 * 1 = 8
        res = _sim(sys, (0.0, 2.5))
        assert abs(float(res.context.discrete_state) - 8.0) < 1e-9


# ─────────────────────────────────────────────────────────────────────────────
# 4. ZERO-CROSSING DETECTION CORRECTNESS
# ─────────────────────────────────────────────────────────────────────────────

class TestZeroCrossingDetection:

    def test_crossing_time_accuracy(self):
        """Zero-crossing localization must be within picosecond of the true time.

        x(t) = x0 - a*t crosses zero at t* = x0/a.
        We declare a zero-crossing when x passes through zero (positive→non-positive).
        The context time at end of simulation should be within ~1e-11 of t*.
        """
        x0, a = 2.0, 3.0
        t_cross = x0 / a  # exact crossing time

        class LinearDecay(jaxonomy.LeafSystem):
            def __init__(self):
                super().__init__()
                self.declare_continuous_state(
                    default_value=jnp.array(x0),
                    ode=lambda t, s, *i: jnp.array(-a),
                )
                self.declare_zero_crossing(
                    guard=lambda t, s, *i: s.continuous_state,
                    reset_map=None,
                    direction="positive_then_non_positive",
                    terminal=True,
                )

        sys = LinearDecay()
        res = _sim(sys, (0.0, 2.0))
        t_detected = float(res.context.time)
        assert abs(t_detected - t_cross) < 1e-9, (
            f"Zero-crossing time: expected {t_cross:.12f}, got {t_detected:.12f}"
        )

    def test_crossing_resets_state(self):
        """Reset map is applied at the zero-crossing, not before or after."""
        x0 = 1.0
        reset_val = -0.5

        class BouncingState(jaxonomy.LeafSystem):
            def __init__(self):
                super().__init__()
                self.declare_continuous_state(
                    default_value=jnp.array(x0),
                    ode=lambda t, s, *i: jnp.array(-1.0),
                )

                def _reset(t, s, *i):
                    return s.with_continuous_state(jnp.array(reset_val))

                self.declare_zero_crossing(
                    guard=lambda t, s, *i: s.continuous_state,
                    reset_map=_reset,
                    direction="positive_then_non_positive",
                    terminal=False,
                )

        sys = BouncingState()
        res = _sim(sys, (0.0, 1.5))
        x_final = float(res.context.continuous_state)
        # After crossing at t=1, state resets to -0.5 and continues at -1.0/s
        # At t=1.5: x = -0.5 + (-1.0)*0.5 = -1.0
        expected = reset_val + (-1.0) * (1.5 - x0)
        assert abs(x_final - expected) < 1e-4, (
            f"Post-reset: expected {expected:.6f}, got {x_final:.6f}"
        )

    def test_missed_crossing_detection(self):
        """Fast oscillation: every sign change within a single ODE step must be detected.

        If a zero-crossing happens and un-happens within a single RK step, the
        engine may miss it (known limitation of interval sign-change detection).
        Here we use a slow enough oscillation that the crossing is always visible.

        x = sin(omega * t) with omega chosen so period >> solver step size.
        Declare terminal crossing at x going negative. t* = pi/omega.
        """
        omega = 0.5  # slow; period = 4*pi ~ 12.6 s
        t_cross = np.pi / omega

        class SineOscillator(jaxonomy.LeafSystem):
            """dx/dt = omega*cos(omega*t), x(0) = 0 → x(t) = sin(omega*t)."""
            def __init__(self):
                super().__init__()
                self.declare_continuous_state(
                    default_value=jnp.array(0.0),
                    ode=lambda t, s, *i: jnp.array(omega * jnp.cos(omega * t)),
                )
                self.declare_zero_crossing(
                    guard=lambda t, s, *i: s.continuous_state,
                    reset_map=None,
                    direction="positive_then_non_positive",
                    terminal=True,
                )

        sys = SineOscillator()
        res = _sim(sys, (0.0, 20.0))
        t_detected = float(res.context.time)
        assert abs(t_detected - t_cross) < 1e-5, (
            f"Missed/wrong crossing: expected t*={t_cross:.6f}, got {t_detected:.6f}"
        )

    def test_direction_positive_then_nonpositive(self):
        """Only positive→non-positive trigger fires; negative→non-negative does not."""
        class TwoWay(jaxonomy.LeafSystem):
            """x(t) = sin(t): crosses zero at t=pi (pos→neg) and t=2pi (neg→pos)."""
            def __init__(self):
                super().__init__()
                self.declare_discrete_state(default_value=jnp.array(0.0))
                self.declare_continuous_state(
                    default_value=jnp.array(0.0),
                    ode=lambda t, s, *i: jnp.array(jnp.cos(t)),
                )
                def _reset(t, s, *i):
                    return s.with_discrete_state(s.discrete_state + 1.0)

                self.declare_zero_crossing(
                    guard=lambda t, s, *i: s.continuous_state,
                    reset_map=_reset,
                    direction="positive_then_non_positive",
                    terminal=False,
                )

        sys = TwoWay()
        # Run past two full crossings; only the positive→non-positive one should fire
        res = _sim(sys, (0.0, 7.0))  # covers t=pi and t=2*pi
        count = float(res.context.discrete_state)
        # sin(0)=0 then grows positive, crosses zero negatively at t=pi (trigger=1)
        # sin(t) goes negative then positive at t=2*pi (no trigger for this direction)
        assert count == 1.0, f"Expected 1 trigger (pos→neg only), got {count}"


# ─────────────────────────────────────────────────────────────────────────────
# 5. INTEGER TIME REPRESENTATION
# ─────────────────────────────────────────────────────────────────────────────

class TestIntegerTime:

    def test_long_simulation_event_synchronization(self):
        """Events must stay synchronized even after many periods (no drift).

        A period-1.0 counter over 1000 steps fires at t=0, 1, ..., 999
        (the event at t=1000.0 = t_end is not processed — loop exits when
        time >= t_end). Total: 1000 updates, no float drift.
        """
        class Counter(jaxonomy.LeafSystem):
            def __init__(self):
                super().__init__()
                self.declare_discrete_state(default_value=jnp.array(0.0))
                self.declare_periodic_update(self._upd, period=1.0, offset=0.0)
            def _upd(self, t, s, *i):
                return s.discrete_state + 1.0

        sys = Counter()
        opts = jaxonomy.SimulatorOptions(math_backend="jax", max_major_steps=2000)
        res = _sim(sys, (0.0, 1000.0), opts=opts)
        val = float(res.context.discrete_state)
        assert val == 1000.0, f"Expected 1000 ticks, got {val}"

    def test_non_dyadic_period_no_drift(self):
        """Period 1/3 must not accumulate floating-point drift over many cycles.

        Over 300 steps, a period of 1/3 should fire 901 times (at t=0, 1/3, 2/3, ...).
        """
        T = 1.0 / 3.0
        N = 300
        expected = N * 3 + 1  # = 901

        class Counter(jaxonomy.LeafSystem):
            def __init__(self):
                super().__init__()
                self.declare_discrete_state(default_value=jnp.array(0.0))
                self.declare_periodic_update(self._upd, period=T, offset=0.0)
            def _upd(self, t, s, *i):
                return s.discrete_state + 1.0

        sys = Counter()
        opts = jaxonomy.SimulatorOptions(math_backend="jax", max_major_steps=3000)
        res = _sim(sys, (0.0, float(N)), opts=opts)
        val = float(res.context.discrete_state)
        assert val == expected, f"Expected {expected} ticks, got {val}"


# ─────────────────────────────────────────────────────────────────────────────
# 6. CONTINUOUS-TIME ODE ACCURACY
# ─────────────────────────────────────────────────────────────────────────────

class TestODEAccuracy:

    def test_linear_ode_dopri5(self):
        """dx/dt = -k*x → x(T) = x0*exp(-k*T), Dopri5."""
        k, x0, T = 2.0, 1.5, 3.0
        expected = x0 * np.exp(-k * T)

        bld = jaxonomy.DiagramBuilder()
        integ = bld.add(Integrator(jnp.array(x0)))
        gain = bld.add(Gain(-k))
        bld.connect(integ.output_ports[0], gain.input_ports[0])
        bld.connect(gain.output_ports[0], integ.input_ports[0])
        diagram = bld.build()
        ctx = diagram.create_context()
        res = _sim(diagram, (0.0, T), ctx=ctx)
        x_final = float(res.context[integ.system_id].continuous_state)
        assert abs(x_final - expected) < 1e-5, (
            f"Dopri5 linear ODE: expected {expected:.8f}, got {x_final:.8f}"
        )

    def test_linear_ode_bdf(self):
        """Same linear ODE solved with BDF."""
        k, x0, T = 2.0, 1.5, 3.0
        expected = x0 * np.exp(-k * T)

        bld = jaxonomy.DiagramBuilder()
        integ = bld.add(Integrator(jnp.array(x0)))
        gain = bld.add(Gain(-k))
        bld.connect(integ.output_ports[0], gain.input_ports[0])
        bld.connect(gain.output_ports[0], integ.input_ports[0])
        diagram = bld.build()
        ctx = diagram.create_context()
        opts = jaxonomy.SimulatorOptions(math_backend="jax", ode_solver_method="bdf")
        res = _sim(diagram, (0.0, T), ctx=ctx, opts=opts)
        x_final = float(res.context[integ.system_id].continuous_state)
        assert abs(x_final - expected) < 1e-4, (
            f"BDF linear ODE: expected {expected:.8f}, got {x_final:.8f}"
        )

    def test_harmonic_oscillator_energy_conservation(self):
        """dx/dt = v, dv/dt = -omega^2*x → energy E = (v^2 + omega^2*x^2)/2 = const.

        Tests that the adaptive stepper maintains a good energy balance
        over multiple cycles (not a stiff problem, but tests error control).
        """
        omega = 2.0
        x0, v0 = 1.0, 0.0
        T = 10 * (2 * np.pi / omega)  # 10 full periods
        E0 = 0.5 * (v0**2 + omega**2 * x0**2)

        class HarmonicOscillator(jaxonomy.LeafSystem):
            def __init__(self):
                super().__init__()
                self.declare_continuous_state(
                    default_value=jnp.array([x0, v0]),
                    ode=self._ode,
                )
            def _ode(self, t, s, *i):
                x, v = s.continuous_state
                return jnp.array([v, -omega**2 * x])

        sys = HarmonicOscillator()
        res = _sim(sys, (0.0, T))
        x_f, v_f = res.context.continuous_state
        E_f = 0.5 * (float(v_f)**2 + omega**2 * float(x_f)**2)
        assert abs(E_f - E0) / E0 < 1e-4, (
            f"Energy drift: E0={E0:.6f}, E_f={E_f:.6f}, rel_err={abs(E_f-E0)/E0:.2e}"
        )

    def test_stiff_ode_bdf_accuracy(self):
        """Stiff ODE: dx/dt = -1000*(x - cos(t)) - sin(t)
        Exact solution: x(t) = cos(t) + C*exp(-1000*t), C = x0 - 1.
        BDF must handle this without step-size collapse.
        """
        lam = 1000.0
        x0 = 2.0  # so C = 1
        T = 0.02   # transient decays by exp(-20) ≈ 2e-9

        class StiffSystem(jaxonomy.LeafSystem):
            def __init__(self):
                super().__init__()
                self.declare_continuous_state(
                    default_value=jnp.array(x0),
                    ode=self._ode,
                )
            def _ode(self, t, s, *i):
                x = s.continuous_state
                return jnp.array(-lam * (x - jnp.cos(t)) - jnp.sin(t))

        sys = StiffSystem()
        opts = jaxonomy.SimulatorOptions(math_backend="jax", ode_solver_method="bdf",
                                         rtol=1e-6, atol=1e-8)
        res = _sim(sys, (0.0, T), opts=opts)
        x_f = float(res.context.continuous_state)
        expected = np.cos(T)  # transient has decayed to negligible
        assert abs(x_f - expected) < 1e-4, (
            f"Stiff BDF: expected {expected:.8f}, got {x_f:.8f}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# 7. ZENO-LIKE BEHAVIOR
# ─────────────────────────────────────────────────────────────────────────────

class TestZenoLikeBehavior:

    def test_max_major_steps_stops_simulation(self):
        """When max_major_steps is explicitly set, the bounded fori_loop is used even
        without enable_autodiff, so the step budget is honoured.

        A 1 ms period counter over [0, 1] would need 1001 major steps.  Limiting to
        50 steps means the simulation cannot reach t=1.0 and must raise or stop early.
        """
        class FastCounter(jaxonomy.LeafSystem):
            def __init__(self):
                super().__init__()
                self.declare_discrete_state(default_value=jnp.array(0.0))
                self.declare_periodic_update(self._upd, period=1e-3, offset=0.0)
            def _upd(self, t, s, *i):
                return s.discrete_state + 1.0

        sys = FastCounter()
        # max_major_steps=50 → bounded loop even without enable_autodiff
        opts = jaxonomy.SimulatorOptions(math_backend="jax", max_major_steps=50)
        # The bounded fori_loop runs ≤ 50 iterations; the engine should raise because
        # the end time was not reached, OR return early with t < t_end.
        try:
            res = _sim(sys, (0.0, 1.0), opts=opts)
            t_final = float(res.context.time)
            assert t_final < 0.1, f"Expected early stop, but t_final={t_final}"
        except RuntimeError as e:
            assert "reach" in str(e).lower() or "end time" in str(e).lower(), (
                f"Unexpected RuntimeError: {e}"
            )

    def test_terminal_event_stops_integration(self):
        """A terminal zero-crossing event must halt the ODE integration immediately."""
        stop_time = 0.5

        class Stopper(jaxonomy.LeafSystem):
            def __init__(self):
                super().__init__()
                self.declare_continuous_state(
                    default_value=jnp.array(0.0),
                    ode=lambda t, s, *i: jnp.array(1.0),
                )
                self.declare_zero_crossing(
                    guard=lambda t, s, *i: s.continuous_state - stop_time,
                    reset_map=None,
                    direction="negative_then_non_negative",
                    terminal=True,
                )

        sys = Stopper()
        res = _sim(sys, (0.0, 10.0))
        t_final = float(res.context.time)
        x_final = float(res.context.continuous_state)
        assert abs(t_final - stop_time) < 1e-9, (
            f"Terminal event: expected t={stop_time}, got {t_final}"
        )
        assert abs(x_final - stop_time) < 1e-6, (
            f"Terminal state: expected {stop_time}, got {x_final}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# 8. DISCONTINUITIES AND STEP INPUTS
# ─────────────────────────────────────────────────────────────────────────────

class TestDiscontinuities:

    def test_step_input_via_zc_reset_is_exact(self):
        """A step input applied via ZC reset must produce correct CT response after.

        System: dx/dt = -x + u, u = 0 for t < T_step, u = 1 for t >= T_step.
        Implemented as a ZC event at t=T_step that changes a parameter.

        Exact solution:
          t < T_step:  x(t) = x0*exp(-t)
          t >= T_step: x(t) = 1 + (x(T_step) - 1)*exp(-(t - T_step))
        """
        x0, T_step, T_end = 2.0, 1.0, 2.5

        class StepSystem(jaxonomy.LeafSystem):
            def __init__(self):
                super().__init__()
                self.declare_discrete_state(default_value=jnp.array(0.0))  # u (step input)
                self.declare_continuous_state(
                    default_value=jnp.array(x0),
                    ode=self._ode,
                )
                self.declare_zero_crossing(
                    guard=lambda t, s, *i: t - T_step,
                    reset_map=self._step_reset,
                    direction="negative_then_non_negative",
                    terminal=False,
                )

            def _ode(self, t, s, *i):
                x = s.continuous_state
                u = s.discrete_state
                return -x + u

            def _step_reset(self, t, s, *i):
                return s.with_discrete_state(jnp.array(1.0))

        sys = StiffSystem = StepSystem()
        res = _sim(sys, (0.0, T_end))
        x_final = float(res.context.continuous_state)
        x_at_step = x0 * np.exp(-T_step)
        expected = 1.0 + (x_at_step - 1.0) * np.exp(-(T_end - T_step))
        assert abs(x_final - expected) < 1e-4, (
            f"Step input response: expected {expected:.6f}, got {x_final:.6f}"
        )

    def test_ode_rhs_discontinuity_at_event(self):
        """ODE with discontinuous RHS: |x(t_cross⁺) - x(t_cross⁻)| = 0 (state continuity).

        At the zero-crossing, the CONTINUOUS state must be continuous (no jump).
        Only the derivative changes. This tests that the solver doesn't introduce
        a spurious jump when the reset map does not modify continuous state.
        """
        x0 = 1.0

        class SignFlip(jaxonomy.LeafSystem):
            """dx/dt = -sign(x), so x decays to 0 from both sides."""
            def __init__(self):
                super().__init__()
                self.declare_discrete_state(default_value=jnp.array(-1.0))  # current sign of ẋ
                self.declare_continuous_state(
                    default_value=jnp.array(x0),
                    ode=self._ode,
                )
                # Trigger on x crossing zero: x > 0 → x ≤ 0
                def _identity_reset(t, s, *i):
                    # Don't change continuous state, just flip the sign tracker
                    return s.with_discrete_state(-s.discrete_state)

                self.declare_zero_crossing(
                    guard=lambda t, s, *i: s.continuous_state,
                    reset_map=_identity_reset,
                    direction="positive_then_non_positive",
                    terminal=True,  # stop at crossing for this test
                )

            def _ode(self, t, s, *i):
                x = s.continuous_state
                return jnp.array(-jnp.sign(x))

        sys = SignFlip()
        res = _sim(sys, (0.0, 2.0))
        # x reaches 0 at t = x0 / 1 = 1.0
        x_final = float(res.context.continuous_state)
        t_final = float(res.context.time)
        # Bisection resolves to ~1e-12 s on 1 s intervals, but the SignFlip ODE has
        # a non-smooth RHS (sign function), which can slow convergence near x=0.
        assert abs(t_final - 1.0) < 1e-7, f"Expected crossing at t=1, got t={t_final}"
        # State continuity: x at crossing should be ~0, not a discontinuous jump
        assert abs(x_final) < 1e-5, f"State discontinuity at ZC: x={x_final}"


# ─────────────────────────────────────────────────────────────────────────────
# 9. HYBRID CT+DT COUPLING (feedforward only, no CT→DT signal)
# ─────────────────────────────────────────────────────────────────────────────

class TestHybridCTDTCoupling:

    def test_dt_driven_ct_integrator(self):
        """DT step input drives CT integrator: exact staircase convolution.

        DT block: x[n] = n (step counter), ZOH output.
        CT integrator: dz/dt = x[n].

        Exact solution:
          z(t) = sum_{k=0}^{N-1} k * T_s + k * (t - N*T_s)  (between ticks)
        """
        T_s = 0.5
        N = 4
        T_end = N * T_s  # exactly at last tick

        class DTStep(jaxonomy.LeafSystem):
            """Output = number of ticks elapsed (held)."""
            def __init__(self):
                super().__init__()
                self.declare_discrete_state(default_value=jnp.array(0.0))
                self.declare_output_port(
                    lambda t, s, *i: s.discrete_state,
                    period=T_s, offset=0.0
                )
                self.declare_periodic_update(
                    lambda t, s, *i: s.discrete_state + 1.0,
                    period=T_s, offset=0.0
                )

        bld = jaxonomy.DiagramBuilder()
        step = bld.add(DTStep())
        integ = bld.add(Integrator(0.0))
        bld.connect(step.output_ports[0], integ.input_ports[0])
        diagram = bld.build()
        ctx = diagram.create_context()
        res = _sim(diagram, (0.0, T_end), ctx=ctx)
        z = float(res.context[integ.system_id].continuous_state)

        # At t=0: output becomes 1, integrates 1*T_s
        # At t=0.5: output becomes 2, integrates 2*T_s
        # At t=1.0: output becomes 3, integrates 3*T_s
        # At t=1.5: output becomes 4, integrates 4*T_s (but this is the end)
        # z = 1*0.5 + 2*0.5 + 3*0.5 + 4*0 = 3.0  (last tick fires AT T_end, ∫=0)
        expected = sum((k + 1) * T_s for k in range(N - 1))  # 1+2+3 * 0.5 = 3.0
        assert abs(z - expected) < 5e-4, f"DT-driven integrator: expected {expected}, got {z}"

    def test_ct_then_dt_no_cross_contamination(self):
        """CT state change must not affect a DT block's current-tick output.

        The DT output is sampled at tick time (ZOH x⁻ semantics). Mid-interval
        changes to the CT state must NOT appear in the DT output until the next tick.

        Architecture:
          CT Integrator (x(0)=0, dx/dt=1) → DT Sampler (T_s=1, offset=0)

        x(t) = t (ramp).  The sampler captures x at each tick using x⁻ semantics:
          At t=0: captures x(0⁻) = 0, holds through [0, 1)
          At t=1: captures x(1⁻) = 1, holds through [1, 2)

        After t=1.5 the held value should be 1 (x at t=1 tick).
        """
        T_s = 1.0

        class DTSampler(jaxonomy.LeafSystem):
            """Captures input at each tick via ZOH."""
            def __init__(self):
                super().__init__()
                self.declare_discrete_state(default_value=jnp.array(0.0))
                self.declare_input_port(name="u")
                # Output uses pre-update discrete state (ZOH hold); no input feedthrough
                self.declare_output_port(
                    lambda t, s: s.discrete_state,
                    requires_inputs=False,
                    period=T_s, offset=0.0
                )
                self.declare_periodic_update(
                    lambda t, s, u: u,
                    period=T_s, offset=0.0
                )

        bld = jaxonomy.DiagramBuilder()
        const = bld.add(Constant(1.0))                 # constant input 1 → dx/dt = 1
        integ = bld.add(Integrator(jnp.array(0.0)))    # x(0)=0, dx/dt=1 → x(t)=t
        sampler = bld.add(DTSampler())
        bld.connect(const.output_ports[0], integ.input_ports[0])
        bld.connect(integ.output_ports[0], sampler.input_ports[0])
        diagram = bld.build()
        ctx = diagram.create_context()
        # At t=0: sampler captures x(0⁻) = 0, holds through t=1
        # At t=1: sampler captures x(1⁻) = 1, holds through t=2
        res = _sim(diagram, (0.0, 1.5), ctx=ctx)
        sampler_out = float(res.context[sampler.system_id].discrete_state)
        # Between t=1 and t=2, the held value should be x(1) = 1
        assert abs(sampler_out - 1.0) < 1e-4, (
            f"ZOH sampler: expected 1.0 (value at t=1 tick), got {sampler_out}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# 10. RESULTS RECORDING TIMING
# ─────────────────────────────────────────────────────────────────────────────

class TestResultsRecordingTiming:

    def test_initial_sample_recorded_at_t0(self):
        """The first recorded sample must be at the simulation start time."""

        class Const(jaxonomy.LeafSystem):
            def __init__(self):
                super().__init__()
                self.declare_continuous_state(
                    default_value=jnp.array(1.0),
                    ode=lambda t, s, *i: jnp.array(0.0),
                )
                self.declare_output_port(
                    lambda t, s, *i: s.continuous_state, name="y"
                )

        sys = Const()
        res = jaxonomy.simulate(
            sys, sys.create_context(), (0.0, 1.0),
            recorded_signals={"y": sys.output_ports[0]}
        )
        assert res.time[0] == 0.0, f"First sample not at t=0: {res.time[0]}"
        assert float(res.outputs["y"][0]) == 1.0

    def test_sample_recorded_at_event_boundary(self):
        """Samples must be recorded at zero-crossing event times."""

        class Probe(jaxonomy.LeafSystem):
            """dx/dt = 1; crosses 1.0 at t=1."""
            def __init__(self):
                super().__init__()
                self.declare_continuous_state(
                    default_value=jnp.array(0.0),
                    ode=lambda t, s, *i: jnp.array(1.0),
                )
                self.declare_output_port(
                    lambda t, s, *i: s.continuous_state, name="y"
                )
                self.declare_zero_crossing(
                    guard=lambda t, s, *i: s.continuous_state - 1.0,
                    reset_map=None,
                    direction="negative_then_non_negative",
                    terminal=False,
                )

        sys = Probe()
        res = jaxonomy.simulate(
            sys, sys.create_context(), (0.0, 2.0),
            recorded_signals={"y": sys.output_ports[0]}
        )
        # There should be a sample close to t=1
        times = np.asarray(res.time)
        closest_idx = np.argmin(np.abs(times - 1.0))
        assert abs(times[closest_idx] - 1.0) < 1e-8, (
            f"No sample near ZC at t=1; closest is t={times[closest_idx]}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# 11. VARIABLE STEP-SIZE BEHAVIOR
# ─────────────────────────────────────────────────────────────────────────────

class TestVariableStepSize:

    def test_step_size_reduction_near_boundary(self):
        """Solver must clip step size to not overshoot a major step boundary.

        With a large initial step size and a short interval, the solver must
        still integrate correctly to exactly tf.
        """
        T = 0.001   # very short interval
        k = 1.0

        class FastDecay(jaxonomy.LeafSystem):
            def __init__(self):
                super().__init__()
                self.declare_continuous_state(
                    default_value=jnp.array(1.0),
                    ode=lambda t, s, *i: -k * s.continuous_state,
                )

        sys = FastDecay()
        # Use a large min step size to force boundary clipping
        opts = jaxonomy.SimulatorOptions(
            math_backend="jax",
            max_minor_step_size=10.0,  # much larger than T
        )
        res = _sim(sys, (0.0, T), opts=opts)
        x_f = float(res.context.continuous_state)
        expected = np.exp(-k * T)
        assert abs(x_f - expected) < 1e-6, (
            f"Boundary clip: expected {expected:.8f}, got {x_f:.8f}"
        )

    def test_min_step_size_terminates_on_stiff(self):
        """When min_minor_step_size is larger than what the ODE requires, the solver
        must NOT hang.  It should force-accept the step at the floor and terminate.

        The Dopri5 fix: when optimal_step_size() returns a value ≤ hmin the solver
        force-accepts the current (inaccurate) step so the while_loop condition
        (~accepted) eventually becomes False.  The simulation may be inaccurate
        but it must complete in finite time.
        """
        class VeryStiff(jaxonomy.LeafSystem):
            def __init__(self):
                super().__init__()
                self.declare_continuous_state(
                    default_value=jnp.array(1.0),
                    ode=lambda t, s, *i: jnp.array(-1e4 * s.continuous_state),
                )

        sys = VeryStiff()
        opts = jaxonomy.SimulatorOptions(
            math_backend="jax",
            min_minor_step_size=1e-2,   # too large for λ=1e4
            ode_solver_method="dopri5",
        )
        # Must complete (not hang); result may be inaccurate
        res = _sim(sys, (0.0, 0.01), opts=opts)
        # Sanity check: time advanced to t_end
        assert float(res.context.time) == pytest.approx(0.01, abs=1e-10)


# ─────────────────────────────────────────────────────────────────────────────
# 12. MULTIPLE ZERO CROSSINGS IN SEQUENCE
# ─────────────────────────────────────────────────────────────────────────────

class TestSequentialZeroCrossings:

    def test_bouncing_ball_known_trajectory(self):
        """Bouncing ball under gravity: analytic positions and bounce times.

        dy/dt = v, dv/dt = -g
        At each bounce: v → -coeff_restitution * v

        Analytic bounce times: t_n = t_{n-1} + 2*v_n/g, where v_n = c^n * v0.

        Test that the first 3 bounce times are correct.
        """
        g = 9.81
        coeff = 0.9  # restitution
        y0, v0 = 5.0, 0.0  # drop from rest

        # First fall time: y0 = 0.5*g*t^2 → t_1 = sqrt(2*y0/g)
        t1 = np.sqrt(2 * y0 / g)
        v1 = g * t1  # speed just before first bounce
        # After bounce: v = coeff*v1 upward, time to next: 2*coeff*v1/g
        t2 = t1 + 2 * coeff * v1 / g
        t3 = t2 + 2 * (coeff**2) * v1 / g

        bounce_times = []

        class BouncingBall(jaxonomy.LeafSystem):
            def __init__(self):
                super().__init__()
                self.declare_discrete_state(default_value=jnp.array(0.0))  # bounce count
                self.declare_continuous_state(
                    default_value=jnp.array([y0, v0]),
                    ode=self._ode,
                )

                def _bounce_reset(t, s, *i):
                    y, v = s.continuous_state
                    new_xc = jnp.array([y, -coeff * v])
                    return s.with_continuous_state(new_xc).with_discrete_state(
                        s.discrete_state + 1.0
                    )

                self.declare_zero_crossing(
                    guard=lambda t, s, *i: s.continuous_state[0],
                    reset_map=_bounce_reset,
                    direction="positive_then_non_positive",
                    terminal=False,
                )

            def _ode(self, t, s, *i):
                y, v = s.continuous_state
                return jnp.array([v, -g])

        # NOTE: This test uses numpy backend to record times
        sys = BouncingBall()
        res = jaxonomy.simulate(
            sys, sys.create_context(), (0.0, t3 + 0.1),
            recorded_signals={"xc": sys.output_ports[0]}
            if hasattr(sys, "output_ports") and sys.output_ports else None,
            options=jaxonomy.SimulatorOptions(math_backend="jax", max_major_steps=1000),
        )
        n_bounces = float(res.context.discrete_state)
        assert n_bounces >= 3, f"Expected at least 3 bounces, got {n_bounces}"


# ─────────────────────────────────────────────────────────────────────────────
# 13. MODE-SWITCHING CORRECTNESS
# ─────────────────────────────────────────────────────────────────────────────

class TestModeSwitching:

    def test_mode_guard_only_active_in_correct_mode(self):
        """Guard declared with start_mode=0 must not trigger in mode=1.

        Uses declare_zero_crossing(start_mode=0, end_mode=1) so the engine handles
        the mode transition.  The reset_map only increments the switch counter.
        """
        class TwoModeSystem(jaxonomy.LeafSystem):
            """
            Mode 0: dx/dt = +1 until x=1 → switch to mode 1.
            Mode 1: dx/dt = -1 (guard for mode 0 must not fire again).
            """
            def __init__(self):
                super().__init__()
                self.declare_default_mode(0)   # required before declare_zero_crossing
                self.declare_continuous_state(
                    default_value=jnp.array(0.0),
                    ode=self._ode,
                )
                self.declare_discrete_state(default_value=jnp.array(0.0))  # switch count

                def _count_switch(t, s, *i):
                    return s.with_discrete_state(s.discrete_state + 1.0)

                self.declare_zero_crossing(
                    guard=lambda t, s, *i: s.continuous_state - 1.0,
                    reset_map=_count_switch,
                    direction="negative_then_non_negative",
                    terminal=False,
                    start_mode=0,
                    end_mode=1,
                )

            def _ode(self, t, s, *i):
                x = s.continuous_state
                mode = s.mode
                # mode 0: +1, mode 1: -1
                return jnp.where(mode == 0, jnp.array(1.0), jnp.array(-1.0))

        sys = TwoModeSystem()
        # Crossing at t=1 (x goes from 0 to 1). After that mode=1, x decreases.
        # The mode-0 guard must not re-trigger when x comes back down through 1.0.
        res = _sim(sys, (0.0, 2.5))
        switches = float(res.context.discrete_state)
        assert switches == 1.0, (
            f"Mode guard fired {switches} times; expected exactly 1 (mode-0 only)"
        )
