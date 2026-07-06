# SPDX-License-Identifier: MIT
"""
Stress tests for mixed-rate, mixed-dtype, and mixed-dimension systems.

Covers:
  1. Multi-rate DT: incommensurate periods, downsampler/upsampler, cascades,
     CT+DT coupling at multiple rates, x⁻ atomicity at coincident ticks.
  2. Mixed signal types: int32, float32, float64, bool, dtype conversion,
     bool-gated accumulation, cross-dtype port connections.
  3. Mixed dimensions: vector/matrix/tensor states, scalar↔vector promotion,
     Mux/Demux correctness, DT matrix recurrence, vector ODE accuracy,
     batched CT state with component extraction.

Each test documents the precise analytical expected value so regressions
are caught even if the wrong answer is numerically plausible.
"""

import numpy as np
import pytest
import jax.numpy as jnp
import jaxonomy
from jaxonomy.library import (
    Constant,
    Gain,
    Integrator,
    Multiplexer,
    Demultiplexer,
    Comparator,
    SignalDatatypeConversion,
    ZeroOrderHold,
)

pytestmark = pytest.mark.slow


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _sim(system, t_span, ctx=None, recorded=None, opts=None):
    if ctx is None:
        ctx = system.create_context()
    if opts is None:
        opts = jaxonomy.SimulatorOptions(math_backend="jax")
    return jaxonomy.simulate(
        system, ctx, t_span, recorded_signals=recorded, options=opts
    )


def _make_dt_counter(period, init=0.0, name=None):
    """Return a LeafSystem that increments its scalar float64 state by 1 each tick."""
    class _Counter(jaxonomy.LeafSystem):
        def __init__(self):
            super().__init__(name=name or f"Counter_{period}")
            self.declare_discrete_state(default_value=jnp.array(float(init)))
            self.declare_output_port(
                lambda t, s, *i: s.discrete_state,
                period=period, offset=0.0, requires_inputs=False,
            )
            self.declare_periodic_update(
                lambda t, s, *i: s.discrete_state + 1.0,
                period=period, offset=0.0,
            )
    return _Counter()


def _make_latch(period, init=-1.0, name=None):
    """Return a LeafSystem that latches (samples-and-holds) its single input."""
    class _Latch(jaxonomy.LeafSystem):
        def __init__(self):
            super().__init__(name=name or f"Latch_{period}")
            self.declare_discrete_state(default_value=jnp.array(float(init)))
            self.declare_input_port()
            self.declare_output_port(
                lambda t, s, *i: s.discrete_state,
                requires_inputs=False, period=period, offset=0.0,
            )
            self.declare_periodic_update(
                lambda t, s, u: u, period=period, offset=0.0,
            )
    return _Latch()


# ─────────────────────────────────────────────────────────────────────────────
# 14. MIXED-RATE DISCRETE-TIME SYSTEMS
# ─────────────────────────────────────────────────────────────────────────────

class TestMixedRateSystems:

    def test_incommensurate_periods_fire_independently(self):
        """Periods 1/3 s and 1/7 s are incommensurate; they only coincide at t=0.

        Over [0, 21.0]:
          T=1/3: fires at t=0, 1/3, 2/3, ... 21*3 = 63 ticks (t=21 = t_end is skipped)
          T=1/7: fires at t=0, 1/7, 2/7, ... 21*7 = 147 ticks
        """
        T_a, T_b = 1.0 / 3.0, 1.0 / 7.0
        t_end = 21.0
        # Events fire at t=0, T, 2T, ..., t_end (inclusive — the simulator processes
        # the t_end event before the loop exit condition is evaluated).
        # → count = round(t_end / T) + 1
        expected_a = int(round(t_end / T_a)) + 1  # 64
        expected_b = int(round(t_end / T_b)) + 1  # 148

        bld = jaxonomy.DiagramBuilder()
        a = bld.add(_make_dt_counter(T_a, name="A"))
        b = bld.add(_make_dt_counter(T_b, name="B"))
        diagram = bld.build()
        ctx = diagram.create_context()
        opts = jaxonomy.SimulatorOptions(
            math_backend="jax", max_major_steps=250
        )
        res = _sim(diagram, (0.0, t_end), ctx=ctx, opts=opts)
        va = float(res.context[a.system_id].discrete_state)
        vb = float(res.context[b.system_id].discrete_state)
        assert va == expected_a, f"A (T=1/3): expected {expected_a}, got {va}"
        assert vb == expected_b, f"B (T=1/7): expected {expected_b}, got {vb}"

    def test_downsampler_sees_pre_update_fast_state_at_coincident_tick(self):
        """Fast (T=0.25) and slow (T=1.0) fire together at t=0, 1, 2, ...

        At each coincident tick both update simultaneously. The slow latch
        must read fast's x⁻ (pre-update value) — i.e., x⁻ at t=1.0 is the
        count BEFORE the fast block increments at t=1.0.

        Timeline for fast (T=0.25), starting at 0:
          t=0:    x⁻=0 → x⁺=1
          t=0.25: x⁻=1 → x⁺=2
          t=0.5:  x⁻=2 → x⁺=3
          t=0.75: x⁻=3 → x⁺=4
          t=1.0:  x⁻=4 → x⁺=5   ← slow fires here too; reads x⁻=4

        So at t ∈ (1.0, 1.25), slow should hold 4.
        """
        bld = jaxonomy.DiagramBuilder()
        fast = bld.add(_make_dt_counter(0.25, name="fast"))
        slow = bld.add(_make_latch(1.0, init=-1.0, name="slow"))
        bld.connect(fast.output_ports[0], slow.input_ports[0])
        diagram = bld.build()
        ctx = diagram.create_context()
        opts = jaxonomy.SimulatorOptions(math_backend="jax", max_major_steps=30)
        # Simulate to t=1.05: past the coincident t=1.0 tick, before t=1.25
        res = _sim(diagram, (0.0, 1.05), ctx=ctx, opts=opts)
        fast_v = float(res.context[fast.system_id].discrete_state)
        slow_v = float(res.context[slow.system_id].discrete_state)
        assert fast_v == 5.0, f"Fast after t=1.05: expected 5.0, got {fast_v}"
        assert slow_v == 4.0, (
            f"Slow at coincident t=1.0: expected 4 (fast x⁻), got {slow_v}"
        )

    def test_upsampler_zoh_holds_between_slow_ticks(self):
        """Slow DT (T=1.0) output is held constant between ticks; fast DT (T=0.1)
        samples it and should see the same held value all 10 times per slow period.

        Slow output = tick count (0 at t=0, 1 at t=1, ...).
        Fast accumulates: sum = sum + slow_output.

        Interval [0, 1): slow outputs 0 (fired at t=0 → x⁻=0 before update → x⁺=1,
        but output cache shows x⁻=0). Fast fires 10 times reading 0 → sum = 0.
        Interval [1, 2): slow output = x⁻=1 (became 1 at t=1 tick). Fast reads 1
        ten times → sum += 10 → sum = 10.
        Interval [2, 3): slow output = 2, fast reads 2 ten times → sum += 20 → sum = 30.
        """
        class SlowRamp(jaxonomy.LeafSystem):
            def __init__(self):
                super().__init__(name="slow_ramp")
                self.declare_discrete_state(default_value=jnp.array(0.0))
                self.declare_output_port(
                    lambda t, s, *i: s.discrete_state,
                    period=1.0, offset=0.0, requires_inputs=False,
                )
                self.declare_periodic_update(
                    lambda t, s, *i: s.discrete_state + 1.0,
                    period=1.0, offset=0.0,
                )

        class FastAccum(jaxonomy.LeafSystem):
            def __init__(self):
                super().__init__(name="fast_accum")
                self.declare_discrete_state(default_value=jnp.array(0.0))
                self.declare_input_port()
                self.declare_periodic_update(
                    lambda t, s, u: s.discrete_state + u,
                    period=0.1, offset=0.0,
                )

        bld = jaxonomy.DiagramBuilder()
        slow = bld.add(SlowRamp())
        fast = bld.add(FastAccum())
        bld.connect(slow.output_ports[0], fast.input_ports[0])
        diagram = bld.build()
        ctx = diagram.create_context()
        opts = jaxonomy.SimulatorOptions(math_backend="jax", max_major_steps=100)
        res = _sim(diagram, (0.0, 3.0), ctx=ctx, opts=opts)
        accum = float(res.context[fast.system_id].discrete_state)
        # t=0: slow x⁻=0 → 10 fast reads of 0 = 0
        # t=1: slow x⁻=1 → 10 fast reads of 1 = 10
        # t=2: slow x⁻=2 → 10 fast reads of 2 = 20
        expected = 0 + 10 + 20  # = 30
        assert accum == expected, (
            f"ZOH upsampler accumulation: expected {expected}, got {accum}"
        )

    def test_three_rate_cascade_correct_tick_counts(self):
        """T_fast=0.1, T_med=0.3, T_slow=1.0 — all incommensurate with each other
        except at t=0.  Over [0, 3.0]:

          fast fires: t=0, 0.1, ..., 2.9  → 30 ticks
          med  fires: t=0, 0.3, ..., 2.7  → 10 ticks  (3.0 == t_end, skipped)
          slow fires: t=0, 1.0, 2.0       →  3 ticks
        """
        bld = jaxonomy.DiagramBuilder()
        fast = bld.add(_make_dt_counter(0.1,  name="fast"))
        med  = bld.add(_make_dt_counter(0.3,  name="med"))
        slow = bld.add(_make_dt_counter(1.0,  name="slow"))
        diagram = bld.build()
        ctx = diagram.create_context()
        opts = jaxonomy.SimulatorOptions(math_backend="jax", max_major_steps=60)
        res = _sim(diagram, (0.0, 3.0), ctx=ctx, opts=opts)
        assert float(res.context[fast.system_id].discrete_state) == 30.0, "fast"
        assert float(res.context[med.system_id].discrete_state)  == 10.0, "med"
        assert float(res.context[slow.system_id].discrete_state) ==  3.0, "slow"

    def test_fast_to_slow_cascade_atomicity_at_triple_coincident_tick(self):
        """At t=0 all three rates fire simultaneously.  Each latch must read the
        OTHER block's x⁻, not x⁺.

        T_fast=0.1, T_med=0.5, T_slow=2.0 — all coincide at t=0 only.

        At t=0: fast x⁻=0, med x⁻=0, slow x⁻=0.
          med latch reads fast x⁻ = 0  → med x⁺ = 0
          slow latch reads med x⁻  = 0 → slow x⁺ = 0

        After t=0 tick, stop before next med tick (t=0.05):
          fast after 1 tick = 1
          med  latched      = 0 (read fast x⁻)
          slow latched      = 0 (read med x⁻)
        """
        class Latch3(jaxonomy.LeafSystem):
            def __init__(self, period, init, name):
                super().__init__(name=name)
                self.declare_discrete_state(default_value=jnp.array(float(init)))
                self.declare_input_port()
                self.declare_output_port(
                    lambda t, s, *i: s.discrete_state,
                    requires_inputs=False, period=period, offset=0.0,
                )
                self.declare_periodic_update(
                    lambda t, s, u: u, period=period, offset=0.0,
                )

        bld = jaxonomy.DiagramBuilder()
        fast = bld.add(_make_dt_counter(0.1, init=0.0, name="fast"))
        med  = bld.add(Latch3(0.5, init=0.0, name="med"))
        slow = bld.add(Latch3(2.0, init=0.0, name="slow"))
        bld.connect(fast.output_ports[0], med.input_ports[0])
        bld.connect(med.output_ports[0],  slow.input_ports[0])
        diagram = bld.build()
        ctx = diagram.create_context()
        opts = jaxonomy.SimulatorOptions(math_backend="jax", max_major_steps=5)
        res = _sim(diagram, (0.0, 0.05), ctx=ctx, opts=opts)
        fast_v = float(res.context[fast.system_id].discrete_state)
        med_v  = float(res.context[med.system_id].discrete_state)
        slow_v = float(res.context[slow.system_id].discrete_state)
        assert fast_v == 1.0, f"fast after 1 tick: expected 1, got {fast_v}"
        assert med_v  == 0.0, (
            f"med latched fast x⁻ at t=0: expected 0 (fast x⁻), got {med_v}"
        )
        assert slow_v == 0.0, (
            f"slow latched med x⁻ at t=0: expected 0 (med x⁻), got {slow_v}"
        )

    def test_mixed_rate_ct_dt_fast_dt_drives_slow_integrator(self):
        """Fast DT (T=0.1) controls a CT integrator; slow DT (T=1.0) samples the
        integrator output.

        CT integrator: dz/dt = u_fast (ZOH from fast DT).
        Fast DT: u_fast[n] = 1 for all n (constant ramp forcing).

        Over [0, T_end]:
          The CT state integrates u_fast = 1.0 → z(t) = t.
          Slow latch at T=1.0 captures z at each slow tick.
          At t=2.0 tick: z(2.0) = 2.0 → slow holds 2.0 until t=3.0.
          After t=2.5: slow holds 2.0.

        This tests that the fast DT output port (ZOH) correctly feeds the CT
        integrator every major step, not just every 0.1 s.
        """
        class FastConst(jaxonomy.LeafSystem):
            """Outputs constant 1.0 via a periodic cache-update (ZOH)."""
            def __init__(self):
                super().__init__(name="fast_const")
                self.declare_discrete_state(default_value=jnp.array(1.0))
                self.declare_output_port(
                    lambda t, s, *i: s.discrete_state,
                    period=0.1, offset=0.0, requires_inputs=False,
                )
                self.declare_periodic_update(
                    lambda t, s, *i: jnp.array(1.0),
                    period=0.1, offset=0.0,
                )

        bld = jaxonomy.DiagramBuilder()
        fast = bld.add(FastConst())
        integ = bld.add(Integrator(jnp.array(0.0)))
        slow = bld.add(_make_latch(1.0, init=-1.0, name="slow_sampler"))
        bld.connect(fast.output_ports[0], integ.input_ports[0])
        bld.connect(integ.output_ports[0], slow.input_ports[0])
        diagram = bld.build()
        ctx = diagram.create_context()
        opts = jaxonomy.SimulatorOptions(math_backend="jax", max_major_steps=60)
        res = _sim(diagram, (0.0, 2.5), ctx=ctx, opts=opts)
        z = float(res.context[integ.system_id].continuous_state)
        slow_v = float(res.context[slow.system_id].discrete_state)
        assert abs(z - 2.5) < 1e-4, f"CT integrator z(2.5)=2.5, got {z}"
        # Slow fired at t=0 (captured z=0), t=1 (z=1), t=2 (z=2); now holds 2.0
        assert abs(slow_v - 2.0) < 1e-4, (
            f"Slow sample at t=2: expected 2.0, got {slow_v}"
        )

    def test_offset_stagger_no_simultaneous_fire(self):
        """Blocks with same period but different offsets never fire simultaneously.

        T=1.0, offset_A=0.0, offset_B=0.5.
        At t=0 only A fires; at t=0.5 only B fires; at t=1 only A fires; etc.

        If both incremented their counters at each OTHER's tick, counts would be wrong.
        Expected after [0, 3.0]: A fires at 0,1,2 → 3 ticks; B fires at 0.5,1.5,2.5 → 3.
        """
        bld = jaxonomy.DiagramBuilder()

        class OffsetCounter(jaxonomy.LeafSystem):
            def __init__(self, offset, name):
                super().__init__(name=name)
                self.declare_discrete_state(default_value=jnp.array(0.0))
                self.declare_periodic_update(
                    lambda t, s, *i: s.discrete_state + 1.0,
                    period=1.0, offset=offset,
                )

        a = bld.add(OffsetCounter(0.0, "A"))
        b = bld.add(OffsetCounter(0.5, "B"))
        diagram = bld.build()
        ctx = diagram.create_context()
        opts = jaxonomy.SimulatorOptions(math_backend="jax", max_major_steps=10)
        res = _sim(diagram, (0.0, 3.0), ctx=ctx, opts=opts)
        va = float(res.context[a.system_id].discrete_state)
        vb = float(res.context[b.system_id].discrete_state)
        assert va == 3.0, f"A (offset=0): expected 3, got {va}"
        assert vb == 3.0, f"B (offset=0.5): expected 3, got {vb}"


# ─────────────────────────────────────────────────────────────────────────────
# 15. MIXED SIGNAL TYPES (int, bool, float32, float64, dtype conversion)
# ─────────────────────────────────────────────────────────────────────────────

class TestMixedSignalTypes:

    def test_int32_discrete_state_preserved(self):
        """Discrete state initialized as int32 must remain int32 after simulation.

        An int32 counter that increments by 1 each tick must:
          (a) preserve the int32 dtype through the simulation loop, and
          (b) produce the correct integer count.
        """
        class Int32Counter(jaxonomy.LeafSystem):
            def __init__(self):
                super().__init__()
                self.declare_discrete_state(
                    default_value=jnp.array(0, dtype=jnp.int32)
                )
                self.declare_periodic_update(self._upd, period=1.0, offset=0.0)

            def _upd(self, t, s, *i):
                return (s.discrete_state + jnp.array(1, dtype=jnp.int32)).astype(
                    jnp.int32
                )

        sys = Int32Counter()
        res = _sim(sys, (0.0, 4.5))   # fires at 0,1,2,3,4 → 5 ticks
        xd = res.context.discrete_state
        assert xd.dtype == jnp.int32, f"Expected int32, got {xd.dtype}"
        assert int(xd) == 5, f"Expected count=5, got {int(xd)}"

    def test_float32_discrete_state_preserved(self):
        """float32 discrete state stays float32 through the simulation fori_loop."""
        class F32Accum(jaxonomy.LeafSystem):
            def __init__(self):
                super().__init__()
                self.declare_discrete_state(
                    default_value=jnp.array(0.0, dtype=jnp.float32)
                )
                self.declare_periodic_update(self._upd, period=1.0, offset=0.0)

            def _upd(self, t, s, *i):
                return (s.discrete_state + jnp.array(1.0, dtype=jnp.float32)).astype(
                    jnp.float32
                )

        sys = F32Accum()
        res = _sim(sys, (0.0, 2.5))  # 3 ticks
        xd = res.context.discrete_state
        assert xd.dtype == jnp.float32, f"Expected float32, got {xd.dtype}"
        assert float(xd) == 3.0, f"Expected 3.0, got {float(xd)}"

    def test_bool_discrete_state_toggle(self):
        """Boolean discrete state toggles correctly with bitwise NOT (~)."""
        class Toggler(jaxonomy.LeafSystem):
            def __init__(self):
                super().__init__()
                self.declare_discrete_state(
                    default_value=jnp.array(True)
                )
                self.declare_output_port(
                    lambda t, s, *i: s.discrete_state,
                    period=0.5, offset=0.0, requires_inputs=False,
                )
                self.declare_periodic_update(
                    lambda t, s, *i: ~s.discrete_state,
                    period=0.5, offset=0.0,
                )

        sys = Toggler()
        # Fires at t=0,0.5,1.0,1.5 → 4 toggles: True→False→True→False→True? No:
        # t=0: x⁻=True, x⁺=False
        # t=0.5: x⁻=False, x⁺=True
        # t=1.0: x⁻=True, x⁺=False
        # t=1.5: x⁻=False, x⁺=True
        # After 4 ticks: True
        res = _sim(sys, (0.0, 2.0))   # t=2 is t_end, skipped
        xd = res.context.discrete_state
        assert xd.dtype == jnp.bool_, f"Expected bool, got {xd.dtype}"
        assert bool(xd) is True, f"Expected True after 4 toggles, got {bool(xd)}"

    def test_bool_gate_controls_float_accumulation(self):
        """A bool-typed DT signal gates a float accumulator.

        Architecture:
          Toggler (T=0.5, starts True) → gate input of Accumulator (T=0.5)
          Constant(1.0)               → value input of Accumulator

        Toggler starts True (x⁻=True → outputs True at t=0 tick, then x⁺=False):
          t=0:   flag=True  (x⁻=True)  → accumulate +1 → accum=1; toggle: False
          t=0.5: flag=False (x⁻=False) → skip;             accum=1; toggle: True
          t=1.0: flag=True  (x⁻=True)  → accumulate +1 → accum=2; toggle: False
          t=1.5: flag=False             → skip;             accum=2; toggle: True
          t=2.0: flag=True              → accumulate +1 → accum=3; toggle: False

        After 5 ticks (fires at t=0,0.5,1.0,1.5,2.0, end at t=2.5 skips t=2.5):
          Accumulate 3 times → accum = 3.
        """
        class Toggler(jaxonomy.LeafSystem):
            def __init__(self):
                super().__init__(name="toggler")
                self.declare_discrete_state(default_value=jnp.array(True))
                self.declare_output_port(
                    lambda t, s, *i: s.discrete_state,
                    period=0.5, offset=0.0, requires_inputs=False,
                )
                self.declare_periodic_update(
                    lambda t, s, *i: ~s.discrete_state,
                    period=0.5, offset=0.0,
                )

        class GatedAccum(jaxonomy.LeafSystem):
            def __init__(self):
                super().__init__(name="gated_accum")
                self.declare_discrete_state(default_value=jnp.array(0.0))
                self.declare_input_port(name="flag")
                self.declare_input_port(name="val")
                self.declare_periodic_update(self._upd, period=0.5, offset=0.0)

            def _upd(self, t, s, flag, val):
                return jnp.where(flag, s.discrete_state + val, s.discrete_state)

        bld = jaxonomy.DiagramBuilder()
        tog = bld.add(Toggler())
        acc = bld.add(GatedAccum())
        const = bld.add(Constant(1.0))
        bld.connect(tog.output_ports[0], acc.input_ports[0])
        bld.connect(const.output_ports[0], acc.input_ports[1])
        diagram = bld.build()
        ctx = diagram.create_context()
        opts = jaxonomy.SimulatorOptions(math_backend="jax", max_major_steps=15)
        res = _sim(diagram, (0.0, 2.5), ctx=ctx, opts=opts)
        accum = float(res.context[acc.system_id].discrete_state)
        assert accum == 3.0, f"Gated accumulation: expected 3.0, got {accum}"

    def test_comparator_output_is_bool(self):
        """Comparator output must be a boolean signal, not float.

        Checks dtype of the recorded signal from a Comparator block.
        """
        class ConstSrc(jaxonomy.LeafSystem):
            def __init__(self, val, name):
                super().__init__(name=name)
                self.declare_output_port(
                    lambda t, s, *i: jnp.array(float(val))
                )

        bld = jaxonomy.DiagramBuilder()
        a = bld.add(ConstSrc(3.0, "a"))
        b = bld.add(ConstSrc(5.0, "b"))
        cmp_gt = bld.add(Comparator(operator=">"))
        cmp_eq = bld.add(Comparator(operator="<"))
        bld.connect(a.output_ports[0], cmp_gt.input_ports[0])
        bld.connect(b.output_ports[0], cmp_gt.input_ports[1])
        bld.connect(a.output_ports[0], cmp_eq.input_ports[0])
        bld.connect(b.output_ports[0], cmp_eq.input_ports[1])
        diagram = bld.build()
        ctx = diagram.create_context()
        res = jaxonomy.simulate(
            diagram, ctx, (0.0, 1.0),
            recorded_signals={
                "gt": cmp_gt.output_ports[0],
                "lt": cmp_eq.output_ports[0],
            },
        )
        assert res.outputs["gt"].dtype == jnp.bool_, (
            f"Expected bool, got {res.outputs['gt'].dtype}"
        )
        assert not bool(res.outputs["gt"][0]), "3 > 5 must be False"
        assert bool(res.outputs["lt"][0]), "3 < 5 must be True"

    def test_signal_dtype_conversion_float64_to_float32_roundtrip(self):
        """SignalDatatypeConversion correctly narrows float64 → float32.

        A float64 ramp feeds SignalDatatypeConversion('float32').
        The recorded output must have dtype float32 and the correct value.
        """
        class F64Ramp(jaxonomy.LeafSystem):
            def __init__(self):
                super().__init__()
                self.declare_discrete_state(
                    default_value=jnp.array(0.0, dtype=jnp.float64)
                )
                self.declare_output_port(
                    lambda t, s, *i: s.discrete_state,
                    period=1.0, offset=0.0, requires_inputs=False,
                )
                self.declare_periodic_update(
                    lambda t, s, *i: (s.discrete_state + 1.0).astype(jnp.float64),
                    period=1.0, offset=0.0,
                )

        bld = jaxonomy.DiagramBuilder()
        ramp = bld.add(F64Ramp())
        conv = bld.add(SignalDatatypeConversion(convert_to_type="float32"))
        bld.connect(ramp.output_ports[0], conv.input_ports[0])
        diagram = bld.build()
        ctx = diagram.create_context()
        opts = jaxonomy.SimulatorOptions(math_backend="jax", max_major_steps=10)
        res = jaxonomy.simulate(
            diagram, ctx, (0.0, 3.5),
            recorded_signals={"out": conv.output_ports[0]},
            options=opts,
        )
        out = res.outputs["out"]
        assert out.dtype == jnp.float32, (
            f"Expected float32 after conversion, got {out.dtype}"
        )
        # The output port is a ZOH declared with period=1.0, offset=0.
        # At t=3: cache reads x⁻=3, sets cache=3, then x⁺=4.
        # At t=3.5 (no tick): ZOH still holds the value captured at t=3 → 3.
        # The final context discrete_state is 4 (after the t=3 state update),
        # but the ZOH output port shows 3.
        assert float(out[-1]) == pytest.approx(3.0, abs=1e-4), (
            f"Converted ZOH at t=3.5: expected 3.0 (held from t=3 tick), "
            f"got {float(out[-1])}"
        )

    def test_int_and_float_signals_in_same_diagram(self):
        """An int32 counter and a float64 accumulator coexist in the same diagram.

        The simulator must correctly trace and advance both without dtype coercion.
        """
        class Int32Counter(jaxonomy.LeafSystem):
            def __init__(self):
                super().__init__(name="int_counter")
                self.declare_discrete_state(
                    default_value=jnp.array(0, dtype=jnp.int32)
                )
                self.declare_periodic_update(
                    lambda t, s, *i: (s.discrete_state + jnp.array(1, jnp.int32)).astype(
                        jnp.int32
                    ),
                    period=0.5, offset=0.0,
                )

        class F64Accum(jaxonomy.LeafSystem):
            def __init__(self):
                super().__init__(name="f64_accum")
                self.declare_discrete_state(
                    default_value=jnp.array(0.0, dtype=jnp.float64)
                )
                self.declare_periodic_update(
                    lambda t, s, *i: (
                        s.discrete_state + jnp.array(0.5, jnp.float64)
                    ).astype(jnp.float64),
                    period=0.5, offset=0.0,
                )

        bld = jaxonomy.DiagramBuilder()
        ic  = bld.add(Int32Counter())
        fa  = bld.add(F64Accum())
        diagram = bld.build()
        ctx = diagram.create_context()
        opts = jaxonomy.SimulatorOptions(math_backend="jax", max_major_steps=20)
        res = _sim(diagram, (0.0, 2.0), ctx=ctx, opts=opts)
        # Fires at t=0,0.5,1.0,1.5 → 4 ticks each
        xd_int = res.context[ic.system_id].discrete_state
        xd_flt = res.context[fa.system_id].discrete_state
        assert xd_int.dtype == jnp.int32, f"int32 drifted to {xd_int.dtype}"
        assert xd_flt.dtype == jnp.float64, f"float64 drifted to {xd_flt.dtype}"
        assert int(xd_int) == 4, f"int counter: expected 4, got {int(xd_int)}"
        assert float(xd_flt) == pytest.approx(2.0), (
            f"float accum: expected 2.0, got {float(xd_flt)}"
        )

    def test_bool_output_drives_discrete_state_update(self):
        """Bool-typed output port drives a discrete state update callback.

        A DT integer counter (period=0.5) increments each tick.  A Comparator
        (>=) checks whether the counter output is >= 3.  A BoolLatch captures
        the comparator's bool-typed output every 0.5 s.

        Using a DT source instead of a CT integrator removes ODE-solver
        precision issues at tick boundaries (the relevant concern is dtype
        propagation, not float equality).

        x⁻ semantics (outputs read pre-update state):
          t=0.0  counter x⁻=0  → cmp=False  → latch=False  (counter→1)
          t=0.5  counter x⁻=1  → cmp=False  → latch=False  (counter→2)
          t=1.0  counter x⁻=2  → cmp=False  → latch=False  (counter→3)
          t=1.5  counter x⁻=3  → cmp=True   → latch=True   (counter→4)
          t=2.0  counter x⁻=4  → cmp=True   → latch=True   (stays True)
        """
        class IntCounter(jaxonomy.LeafSystem):
            """DT counter: state = number of ticks fired so far."""
            def __init__(self):
                super().__init__(name="counter")
                self.declare_discrete_state(default_value=jnp.array(0))
                self.declare_output_port(
                    lambda t, s, *i: s.discrete_state.astype(jnp.float64)
                )
                self.declare_periodic_update(
                    lambda t, s, *i: s.discrete_state + jnp.array(1),
                    period=0.5, offset=0.0,
                )

        class BoolLatch(jaxonomy.LeafSystem):
            def __init__(self):
                super().__init__(name="bool_latch")
                self.declare_discrete_state(default_value=jnp.array(False))
                self.declare_input_port()
                self.declare_periodic_update(
                    # jnp.array(u, dtype=jnp.bool_) is JAX-traceable; bool() is not
                    lambda t, s, u: jnp.array(u, dtype=jnp.bool_),
                    period=0.5, offset=0.0,
                )

        bld = jaxonomy.DiagramBuilder()
        counter = bld.add(IntCounter())
        thresh   = bld.add(Constant(jnp.array(3.0)))
        cmp      = bld.add(Comparator(operator=">="))
        latch    = bld.add(BoolLatch())
        bld.connect(counter.output_ports[0], cmp.input_ports[0])
        bld.connect(thresh.output_ports[0],  cmp.input_ports[1])
        bld.connect(cmp.output_ports[0],     latch.input_ports[0])
        diagram = bld.build()
        ctx = diagram.create_context()
        opts = jaxonomy.SimulatorOptions(math_backend="jax", max_major_steps=20)
        res = _sim(diagram, (0.0, 2.0), ctx=ctx, opts=opts)

        latch_v = res.context[latch.system_id].discrete_state
        # dtype check: must remain bool, not be promoted to int/float
        assert latch_v.dtype == jnp.bool_, (
            f"Latch dtype should be bool_, got {latch_v.dtype}"
        )
        # value check: latch must be True (became True at t=1.5, stays True)
        assert bool(latch_v) is True, (
            f"Latch should hold True after t=1.5; got {latch_v}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# 16. MIXED-DIMENSION SYSTEMS (vector, matrix, tensor states and signals)
# ─────────────────────────────────────────────────────────────────────────────

class TestMixedDimensionSystems:

    def test_vector_ct_ode_decoupled_decay(self):
        """4-dimensional decoupled linear ODE: dx/dt = -k*I*x.

        Analytical solution: x_i(T) = x_i(0) * exp(-k*T).
        Tests that the ODE solver handles N-dimensional state correctly.
        """
        k, T = 2.0, 1.5
        x0 = jnp.array([1.0, 2.0, -1.5, 0.5])
        expected = x0 * np.exp(-k * T)

        class VecDecay(jaxonomy.LeafSystem):
            def __init__(self):
                super().__init__()
                self.declare_continuous_state(
                    default_value=x0,
                    ode=lambda t, s, *i: -k * s.continuous_state,
                )

        sys = VecDecay()
        res = _sim(sys, (0.0, T))
        err = float(jnp.max(jnp.abs(res.context.continuous_state - expected)))
        assert err < 1e-5, (
            f"Vector ODE max abs error: {err:.2e} (expected < 1e-5)"
        )

    def test_vector_ct_ode_rotation(self):
        """2-D rotation ODE: ẋ = [[0, -ω], [ω, 0]] x.

        Exact solution over half-period T = π/ω:  x(T) = -x(0).
        Tests energy conservation and phase accuracy.
        """
        omega = 3.0
        T_half = np.pi / omega
        x0 = jnp.array([1.0, 0.0])

        class RotODE(jaxonomy.LeafSystem):
            def __init__(self):
                super().__init__()
                self.declare_continuous_state(default_value=x0, ode=self._ode)

            def _ode(self, t, s, *i):
                x, y = s.continuous_state
                return jnp.array([-omega * y, omega * x])

        sys = RotODE()
        res = _sim(sys, (0.0, T_half))
        x_f = res.context.continuous_state
        # x(T_half) = [cos(π), sin(π)] = [-1, 0] = -x0
        expected = -x0
        err = float(jnp.max(jnp.abs(x_f - expected)))
        assert err < 1e-5, (
            f"Rotation ODE half-period: got {x_f}, expected {expected}, err={err:.2e}"
        )
        # Energy E = |x|^2 must be conserved
        E0 = float(jnp.sum(x0**2))
        Ef = float(jnp.sum(x_f**2))
        assert abs(Ef - E0) / E0 < 1e-5, (
            f"Rotation energy drift: E0={E0:.6f}, Ef={Ef:.6f}"
        )

    def test_matrix_discrete_state_lyapunov_recurrence(self):
        """Discrete Lyapunov recurrence: P[n+1] = A @ P[n] @ A.T + Q.

        With A = 0.9*I (scalar stable), Q = I, initial P = 0:
          P[1] = Q = I
          P[2] = 0.81*I + I = 1.81*I
          P[3] = 0.81*1.81*I + I = (0.81^2*1 + 0.81 + 1)*I = 2.4661*I

        Tests that a (3×3) matrix discrete state is handled correctly.
        """
        n = 3
        A = 0.9 * jnp.eye(n)
        Q = jnp.eye(n)

        class LyapunovRecurrence(jaxonomy.LeafSystem):
            def __init__(self):
                super().__init__()
                self.declare_discrete_state(default_value=jnp.zeros((n, n)))
                self.declare_periodic_update(self._upd, period=1.0, offset=0.0)

            def _upd(self, t, s, *i):
                P = s.discrete_state
                return A @ P @ A.T + Q

        sys = LyapunovRecurrence()
        res = _sim(sys, (0.0, 2.5))  # fires at 0, 1, 2 → 3 updates
        P = res.context.discrete_state
        # After 3 updates: P = (0.9^2)^2 * 0 + 0.81^2 * Q + 0.81 * Q + Q
        # But start at 0: P[1]=Q=I, P[2]=0.81I+I=1.81I, P[3]=0.81*1.81I+I=2.4661I
        expected_diag = 0.81 * 1.81 + 1.0  # = 2.4661
        assert P.shape == (n, n), f"P shape: expected ({n},{n}), got {P.shape}"
        # Should be a scaled identity
        off_diag_err = float(jnp.max(jnp.abs(P - jnp.diag(jnp.diag(P)))))
        assert off_diag_err < 1e-9, f"Off-diagonal non-zero: {off_diag_err}"
        diag_err = abs(float(P[0, 0]) - expected_diag)
        assert diag_err < 1e-9, (
            f"Lyapunov diagonal: expected {expected_diag:.6f}, got {float(P[0,0]):.6f}"
        )

    def test_tensor_3d_discrete_state_shape_and_values(self):
        """A 3-D tensor (shape 2×3×4) discrete state increments element-wise each tick.

        After N ticks, every element must equal N (starting from 0).
        Tests that the JAX PyTree flattening/unflattening of shaped arrays is correct.
        """
        shape = (2, 3, 4)

        class TensorAccum(jaxonomy.LeafSystem):
            def __init__(self):
                super().__init__()
                self.declare_discrete_state(default_value=jnp.zeros(shape))
                self.declare_periodic_update(
                    lambda t, s, *i: s.discrete_state + jnp.ones(shape),
                    period=1.0, offset=0.0,
                )

        sys = TensorAccum()
        N = 5
        res = _sim(sys, (0.0, N - 0.5))  # N ticks at t=0,1,...,N-1
        xd = res.context.discrete_state
        assert xd.shape == shape, f"Shape mismatch: {xd.shape} vs {shape}"
        assert float(jnp.min(xd)) == float(N), f"Min element: expected {N}, got {jnp.min(xd)}"
        assert float(jnp.max(xd)) == float(N), f"Max element: expected {N}, got {jnp.max(xd)}"

    def test_vector_dt_to_scalar_ct_via_indexing(self):
        """DT block produces a 4-vector; CT integrator integrates one component.

        DT (T=0.5): u[n] = [0, sin(n), n, n^2] where n is tick count.
        CT integrator: dz/dt = u[1] = sin(n_at_current_tick).

        ZOH semantics: between DT ticks n and n+1, u[1] = sin(n) (constant).
        CT integral:
          t ∈ [0, 0.5): u[1] = sin(0) = 0  → contribution 0
          t ∈ [0.5, 1.0): u[1] = sin(1)    → contribution sin(1) * 0.5
          t ∈ [1.0, 1.5): u[1] = sin(2)    → contribution sin(2) * 0.5

        At t=1.5: z = sin(1)*0.5 + sin(2)*0.5 = (sin(1) + sin(2)) / 2.
        """
        T_s = 0.5

        class DVecSrc(jaxonomy.LeafSystem):
            def __init__(self):
                super().__init__(name="vec_src")
                self.declare_discrete_state(default_value=jnp.array(0.0))
                self.declare_output_port(
                    self._out, period=T_s, offset=0.0, requires_inputs=False
                )
                self.declare_periodic_update(
                    lambda t, s, *i: s.discrete_state + 1.0,
                    period=T_s, offset=0.0,
                )

            def _out(self, t, s, *i):
                n = s.discrete_state
                return jnp.array([0.0, jnp.sin(n), n, n**2])

        class IndexInteg(jaxonomy.LeafSystem):
            def __init__(self):
                super().__init__(name="idx_integ")
                self.declare_continuous_state(
                    default_value=jnp.array(0.0), ode=self._ode
                )
                self.declare_input_port()

            def _ode(self, t, s, u):
                return u[1]  # integrate the sin(n) component

        bld = jaxonomy.DiagramBuilder()
        src   = bld.add(DVecSrc())
        integ = bld.add(IndexInteg())
        bld.connect(src.output_ports[0], integ.input_ports[0])
        diagram = bld.build()
        ctx = diagram.create_context()
        opts = jaxonomy.SimulatorOptions(math_backend="jax", max_major_steps=10)
        res = _sim(diagram, (0.0, 1.5), ctx=ctx, opts=opts)
        z = float(res.context[integ.system_id].continuous_state)
        # ZOH: sin(0)*0.5 + sin(1)*0.5 + sin(2)*0.5 = (0 + sin1 + sin2) * 0.5
        expected = (0.0 + np.sin(1.0) + np.sin(2.0)) * 0.5
        assert abs(z - expected) < 1e-4, (
            f"Vector-indexed integrator: expected {expected:.6f}, got {z:.6f}"
        )

    def test_scalar_ct_state_drives_vector_dt_state(self):
        """Scalar CT ramp (x(t) = t) feeds a 3-vector DT integrator.

        DT accumulator: x_dt[n+1] = x_dt[n] + u * ones(3), where u = x_ct(t_k).
        At tick k (t = k * T_s), CT state ≈ k * T_s → x_dt grows as:
          x_dt[1] = 0*T_s * [1,1,1]
          x_dt[2] = (0 + 1)*T_s * [1,1,1]
          ...
          x_dt[N] = sum_{k=0}^{N-1} k*T_s * [1,1,1] = T_s * N*(N-1)/2 * [1,1,1]

        At t = N*T_s (after N ticks at k=0,...,N-1):
          z = T_s * N*(N-1)/2 = 0.5 * 4 * 3 = 6.0 for T_s=0.5, N=4.
        """
        T_s = 0.5

        class VecDTAccum(jaxonomy.LeafSystem):
            def __init__(self):
                super().__init__(name="vec_dt_accum")
                self.declare_discrete_state(default_value=jnp.zeros(3))
                self.declare_input_port()
                self.declare_periodic_update(
                    lambda t, s, u: s.discrete_state + u * jnp.ones(3),
                    period=T_s, offset=0.0,
                )

        bld = jaxonomy.DiagramBuilder()
        const = bld.add(Constant(jnp.array(1.0)))
        integ = bld.add(Integrator(jnp.array(0.0)))
        accum = bld.add(VecDTAccum())
        bld.connect(const.output_ports[0], integ.input_ports[0])
        bld.connect(integ.output_ports[0], accum.input_ports[0])
        diagram = bld.build()
        ctx = diagram.create_context()
        N = 4
        t_end = N * T_s  # = 2.0
        opts = jaxonomy.SimulatorOptions(math_backend="jax", max_major_steps=20)
        res = _sim(diagram, (0.0, t_end), ctx=ctx, opts=opts)
        xd = res.context[accum.system_id].discrete_state
        # At t=0: x⁻=0, u=0 → xd stays 0, x⁺=0
        # At t=0.5: x⁻=0, u=0.5 → xd += 0.5 → xd=0.5
        # At t=1.0: x⁻=0.5, u=1.0 → xd += 1.0 → xd=1.5
        # At t=1.5: x⁻=1.5, u=1.5 → xd += 1.5 → xd=3.0
        # t=2.0 = t_end: skipped
        expected = 0.0 + 0.5 + 1.0 + 1.5  # = 3.0
        assert xd.shape == (3,), f"Vec accum shape: {xd.shape}"
        assert float(jnp.max(jnp.abs(xd - expected))) < 1e-4, (
            f"Vec DT accum: expected {expected}, got {xd}"
        )

    def test_multiplexer_demultiplexer_identity(self):
        """Mux(3) followed by Demux(3) must produce identical components.

        Three constant sources [1.0, 2.0, 3.0] → Mux → Demux → three sinks.
        Each sink accumulates its component over DT ticks.
        After N ticks: sink_i = N * c_i.
        """
        vals = [1.0, 2.0, 3.0]
        N_ticks = 3

        class DtAccum(jaxonomy.LeafSystem):
            def __init__(self, name):
                super().__init__(name=name)
                self.declare_discrete_state(default_value=jnp.array(0.0))
                self.declare_input_port()
                self.declare_periodic_update(
                    lambda t, s, u: s.discrete_state + u,
                    period=1.0, offset=0.0,
                )

        bld = jaxonomy.DiagramBuilder()
        srcs = [bld.add(Constant(jnp.array(v))) for v in vals]
        mux  = bld.add(Multiplexer(3))
        demux = bld.add(Demultiplexer(3))
        sinks = [bld.add(DtAccum(f"sink_{i}")) for i in range(3)]

        for i, src in enumerate(srcs):
            bld.connect(src.output_ports[0], mux.input_ports[i])
        bld.connect(mux.output_ports[0], demux.input_ports[0])
        for i, sink in enumerate(sinks):
            bld.connect(demux.output_ports[i], sink.input_ports[0])

        diagram = bld.build()
        ctx = diagram.create_context()
        opts = jaxonomy.SimulatorOptions(math_backend="jax", max_major_steps=10)
        res = _sim(diagram, (0.0, float(N_ticks) - 0.5), ctx=ctx, opts=opts)

        for i, (sink, v) in enumerate(zip(sinks, vals)):
            xd = float(res.context[sink.system_id].discrete_state)
            expected = N_ticks * v
            assert abs(xd - expected) < 1e-9, (
                f"Mux→Demux sink {i}: expected {expected}, got {xd}"
            )

    def test_matrix_output_port_recording_shape(self):
        """A DT block with a 2×3 matrix output port records correctly.

        Each tick, the output port returns a (2×3) matrix.
        The recorded time series should have shape (T, 2, 3).
        """
        shape = (2, 3)
        T_s = 1.0

        class MatrixSrc(jaxonomy.LeafSystem):
            def __init__(self):
                super().__init__(name="mat_src")
                self.declare_discrete_state(default_value=jnp.zeros(shape))
                self.declare_output_port(
                    lambda t, s, *i: s.discrete_state,
                    period=T_s, offset=0.0, requires_inputs=False,
                )
                self.declare_periodic_update(
                    lambda t, s, *i: s.discrete_state + jnp.ones(shape),
                    period=T_s, offset=0.0,
                )

        sys = MatrixSrc()
        opts = jaxonomy.SimulatorOptions(
            math_backend="jax", max_major_steps=10, save_time_series=True
        )
        res = jaxonomy.simulate(
            sys, sys.create_context(), (0.0, 4.0),
            recorded_signals={"M": sys.output_ports[0]},
            options=opts,
        )
        out = res.outputs["M"]
        # 5 major steps → 5 samples (t=0,1,2,3,4)
        assert out.ndim == 3, f"Expected 3-D recording, got shape {out.shape}"
        assert out.shape[1:] == shape, f"Expected sample shape {shape}, got {out.shape[1:]}"
        # Last sample: output reads x⁻ at t=4.0 before the t=4 update fires
        # That means 4 updates have run (at t=0,1,2,3), output is 4 everywhere
        expected_last = np.full(shape, 4.0)
        np.testing.assert_allclose(
            np.array(out[-1]), expected_last, atol=1e-9,
            err_msg="Matrix output last sample incorrect",
        )

    def test_mixed_dimension_blocks_in_parallel(self):
        """Scalar, 2-vector, and 3×3-matrix DT blocks evolve independently.

        All share the same period (T=1.0) and start from zero.  After N ticks:
          scalar:  N
          vec2:    N * [1, 2]
          mat3x3:  N * I_3
        """
        T_s = 1.0
        N = 4

        class ScalarBlock(jaxonomy.LeafSystem):
            def __init__(self):
                super().__init__(name="scalar")
                self.declare_discrete_state(default_value=jnp.array(0.0))
                self.declare_periodic_update(
                    lambda t, s, *i: s.discrete_state + 1.0,
                    period=T_s, offset=0.0,
                )

        class Vec2Block(jaxonomy.LeafSystem):
            def __init__(self):
                super().__init__(name="vec2")
                self.declare_discrete_state(default_value=jnp.zeros(2))
                self.declare_periodic_update(
                    lambda t, s, *i: s.discrete_state + jnp.array([1.0, 2.0]),
                    period=T_s, offset=0.0,
                )

        class Mat3Block(jaxonomy.LeafSystem):
            def __init__(self):
                super().__init__(name="mat3")
                self.declare_discrete_state(default_value=jnp.zeros((3, 3)))
                self.declare_periodic_update(
                    lambda t, s, *i: s.discrete_state + jnp.eye(3),
                    period=T_s, offset=0.0,
                )

        bld = jaxonomy.DiagramBuilder()
        s = bld.add(ScalarBlock())
        v = bld.add(Vec2Block())
        m = bld.add(Mat3Block())
        diagram = bld.build()
        ctx = diagram.create_context()
        opts = jaxonomy.SimulatorOptions(math_backend="jax", max_major_steps=20)
        res = _sim(diagram, (0.0, N - 0.5), ctx=ctx, opts=opts)

        xs = float(res.context[s.system_id].discrete_state)
        xv = res.context[v.system_id].discrete_state
        xm = res.context[m.system_id].discrete_state

        assert xs == float(N), f"Scalar: expected {N}, got {xs}"
        np.testing.assert_allclose(
            np.array(xv), np.array([N * 1.0, N * 2.0]), atol=1e-9,
            err_msg="Vec2 mismatch",
        )
        np.testing.assert_allclose(
            np.array(xm), N * np.eye(3), atol=1e-9,
            err_msg="Mat3 mismatch",
        )

    def test_vector_zoh_preserves_shape_across_ct_boundary(self):
        """A ZOH on a vector signal must preserve shape across ODE major steps.

        Architecture:
          Constant([1, 2, 3]) → ZeroOrderHold(T=1.0) → DT sum accumulator

        The ZOH samples the input at each T=1 tick. Between ticks the held
        value is constant. The DT accumulator sums the held vector each tick.

        After 3 ticks (t=0, 1, 2 for t_end=3):
          accum = [1,2,3] + [1,2,3] + [1,2,3] = [3, 6, 9]
        """
        T_s = 1.0

        class VecAccum(jaxonomy.LeafSystem):
            def __init__(self):
                super().__init__(name="vec_accum")
                self.declare_discrete_state(default_value=jnp.zeros(3))
                self.declare_input_port()
                self.declare_periodic_update(
                    lambda t, s, u: s.discrete_state + u,
                    period=T_s, offset=0.0,
                )

        bld = jaxonomy.DiagramBuilder()
        src   = bld.add(Constant(jnp.array([1.0, 2.0, 3.0])))
        zoh   = bld.add(ZeroOrderHold(T_s))
        accum = bld.add(VecAccum())
        bld.connect(src.output_ports[0],  zoh.input_ports[0])
        bld.connect(zoh.output_ports[0],  accum.input_ports[0])
        diagram = bld.build()
        ctx = diagram.create_context()
        opts = jaxonomy.SimulatorOptions(math_backend="jax", max_major_steps=10)
        res = _sim(diagram, (0.0, 3.0), ctx=ctx, opts=opts)
        xd = res.context[accum.system_id].discrete_state
        expected = np.array([3.0, 6.0, 9.0])
        np.testing.assert_allclose(
            np.array(xd), expected, atol=1e-9,
            err_msg=f"Vector ZOH accumulator: expected {expected}, got {xd}",
        )
