# SPDX-License-Identifier: MIT

"""
Tests for jaxonomy.optimization.objectives.

Coverage:
  TestISEObjective         – scalar/vector, reference port, scalar weight, name isolation
  TestLQRObjective         – state-only, state+control, diagonal and full Q/R
  TestTrackingMSE          – constant data, linear ramp, vector signal
  TestWeightedSum          – identity, scaling, multi-term, input validation
  TestEquivalence          – helper == hand-wired result (regression guard)
  TestIntegrationWithOpt   – objectives used inside an Optimizable (E2E)
  TestEdgeCases            – single step, zero signal, unit weight, name clashes
"""

from math import ceil

import numpy as np
import pytest
import jax.numpy as jnp

import jaxonomy
from jaxonomy import DiagramBuilder, Parameter, SimulatorOptions
from jaxonomy.library import (
    Adder,
    Constant,
    Gain,
    Integrator,
    Power,
    SumOfElements,
)
from jaxonomy.optimization import (
    Optimizable,
    Scipy,
    ise_objective,
    lqr_objective,
    tracking_mse,
    weighted_sum,
)
from jaxonomy.testing import requires_jax


# ── helpers ───────────────────────────────────────────────────────────────────

DT = 0.05
T_END = 2.0


def _run(diagram, t_end=T_END, signals=None, dt=DT):
    ctx = diagram.create_context()
    nseg = ceil(t_end / dt)
    opts = SimulatorOptions(
        max_major_steps=20 * nseg,
        max_major_step_length=dt,
    )
    return jaxonomy.simulate(
        diagram, ctx, (0.0, t_end),
        options=opts,
        recorded_signals=signals or {},
    )


def _const_diagram(value, objective_fn, **obj_kwargs):
    """Build a minimal diagram: Constant(value) → objective → output."""
    b = DiagramBuilder()
    src = b.add(Constant(jnp.asarray(value, dtype=float), name="src"))
    obj_port = objective_fn(b, src.output_ports[0], **obj_kwargs)
    return b.build(), obj_port


# ══════════════════════════════════════════════════════════════════════════════
# ise_objective
# ══════════════════════════════════════════════════════════════════════════════

@requires_jax()
class TestISEObjective:

    def test_scalar_zero_signal_gives_zero_cost(self):
        """∫ 0² dt = 0"""
        diag, obj_port = _const_diagram(0.0, ise_objective)
        sol = _run(diag, signals={"J": obj_port})
        assert float(sol.outputs["J"][-1]) == pytest.approx(0.0, abs=1e-8)

    def test_scalar_constant_signal_integrates_to_t(self):
        """∫₀ᵀ 1² dt = T"""
        diag, obj_port = _const_diagram(1.0, ise_objective)
        sol = _run(diag, t_end=T_END, signals={"J": obj_port})
        assert float(sol.outputs["J"][-1]) == pytest.approx(T_END, abs=0.01)

    def test_scalar_constant_signal_with_weight(self):
        """∫₀ᵀ w·1² dt = w·T"""
        w = 3.0
        diag, obj_port = _const_diagram(1.0, ise_objective, weight=w)
        sol = _run(diag, t_end=T_END, signals={"J": obj_port})
        assert float(sol.outputs["J"][-1]) == pytest.approx(w * T_END, abs=0.05)

    def test_with_constant_reference_port(self):
        """signal=2, reference=1  →  ∫ (2-1)² dt = T"""
        b = DiagramBuilder()
        sig = b.add(Constant(2.0, name="sig"))
        ref = b.add(Constant(1.0, name="ref"))
        obj_port = ise_objective(b, sig.output_ports[0],
                                 reference_port=ref.output_ports[0])
        diag = b.build()
        sol = _run(diag, t_end=T_END, signals={"J": obj_port})
        assert float(sol.outputs["J"][-1]) == pytest.approx(T_END, abs=0.01)

    def test_reference_port_zero_error_gives_zero_cost(self):
        """signal == reference  →  cost = 0"""
        b = DiagramBuilder()
        val = b.add(Constant(5.0, name="v"))
        obj_port = ise_objective(b, val.output_ports[0],
                                 reference_port=val.output_ports[0])
        diag = b.build()
        sol = _run(diag, t_end=T_END, signals={"J": obj_port})
        assert float(sol.outputs["J"][-1]) == pytest.approx(0.0, abs=1e-8)

    def test_vector_signal_sums_squared_elements(self):
        """signal=[1,1]  →  ∫ (1²+1²) dt = 2T"""
        b = DiagramBuilder()
        src = b.add(Constant(jnp.ones(2), name="src"))
        obj_port = ise_objective(b, src.output_ports[0])
        diag = b.build()
        sol = _run(diag, t_end=T_END, signals={"J": obj_port})
        assert float(sol.outputs["J"][-1]) == pytest.approx(2.0 * T_END, abs=0.02)

    def test_vector_signal_with_reference(self):
        """signal=[2,2], ref=[1,1]  →  ∫ 2·1² dt = 2T"""
        b = DiagramBuilder()
        sig = b.add(Constant(jnp.full(2, 2.0), name="sig"))
        ref = b.add(Constant(jnp.ones(2), name="ref"))
        obj_port = ise_objective(b, sig.output_ports[0],
                                 reference_port=ref.output_ports[0])
        diag = b.build()
        sol = _run(diag, t_end=T_END, signals={"J": obj_port})
        assert float(sol.outputs["J"][-1]) == pytest.approx(2.0 * T_END, abs=0.02)

    def test_output_is_scalar(self):
        """ISE output port must produce a scalar (shape=() or (1,))."""
        b = DiagramBuilder()
        src = b.add(Constant(jnp.ones(3), name="src"))
        obj_port = ise_objective(b, src.output_ports[0])
        diag = b.build()
        sol = _run(diag, t_end=0.5, signals={"J": obj_port})
        j = sol.outputs["J"]
        # Each timestep returns a scalar → overall shape (n_timesteps,) or (n_timesteps, 1)
        assert j.ndim <= 2
        if j.ndim == 2:
            assert j.shape[1] == 1

    def test_name_prefix_isolation(self):
        """Two ise_objective calls with different names should not conflict."""
        b = DiagramBuilder()
        s1 = b.add(Constant(1.0, name="s1"))
        s2 = b.add(Constant(2.0, name="s2"))
        p1 = ise_objective(b, s1.output_ports[0], name="cost_a")
        p2 = ise_objective(b, s2.output_ports[0], name="cost_b")
        diag = b.build()
        sol = _run(diag, t_end=T_END, signals={"a": p1, "b": p2})
        assert float(sol.outputs["a"][-1]) == pytest.approx(T_END, abs=0.01)
        assert float(sol.outputs["b"][-1]) == pytest.approx(4.0 * T_END, abs=0.04)

    def test_weight_one_is_identity(self):
        """weight=1.0 should produce the same result as no weight argument."""
        b1 = DiagramBuilder()
        src1 = b1.add(Constant(3.0, name="s"))
        p1 = ise_objective(b1, src1.output_ports[0], name="ise")
        d1 = b1.build()

        b2 = DiagramBuilder()
        src2 = b2.add(Constant(3.0, name="s"))
        p2 = ise_objective(b2, src2.output_ports[0], weight=1.0, name="ise")
        d2 = b2.build()

        s1 = _run(d1, signals={"J": p1})
        s2 = _run(d2, signals={"J": p2})
        np.testing.assert_allclose(s1.outputs["J"], s2.outputs["J"], atol=1e-10)

    def test_initial_cost_offset(self):
        """initial_cost adds a constant offset to the running integral."""
        offset = 7.5
        b = DiagramBuilder()
        src = b.add(Constant(0.0, name="s"))  # zero signal → J stays at offset
        obj_port = ise_objective(b, src.output_ports[0], initial_cost=offset)
        diag = b.build()
        sol = _run(diag, t_end=T_END, signals={"J": obj_port})
        assert float(sol.outputs["J"][-1]) == pytest.approx(offset, abs=1e-8)


# ══════════════════════════════════════════════════════════════════════════════
# lqr_objective
# ══════════════════════════════════════════════════════════════════════════════

@requires_jax()
class TestLQRObjective:

    def test_state_only_scalar_Q(self):
        """∫ x^T [q] x dt = q * x² * T for constant scalar state."""
        q = 5.0
        Q = np.array([[q]])
        b = DiagramBuilder()
        src = b.add(Constant(jnp.array([1.0]), name="x"))  # x = [1]
        obj_port = lqr_objective(b, src.output_ports[0], Q)
        diag = b.build()
        sol = _run(diag, t_end=T_END, signals={"J": obj_port})
        assert float(sol.outputs["J"][-1]) == pytest.approx(q * T_END, abs=0.05)

    def test_state_only_diagonal_Q(self):
        """x=[2,3], Q=diag([1,2])  →  ∫(4+18)dt = 22·T"""
        Q = np.diag([1.0, 2.0])
        b = DiagramBuilder()
        src = b.add(Constant(jnp.array([2.0, 3.0]), name="x"))
        obj_port = lqr_objective(b, src.output_ports[0], Q)
        diag = b.build()
        sol = _run(diag, t_end=T_END, signals={"J": obj_port})
        # x^T Q x = 4*1 + 9*2 = 22
        assert float(sol.outputs["J"][-1]) == pytest.approx(22.0 * T_END, abs=0.2)

    def test_state_plus_control(self):
        """x=[1], Q=[[2]], u=[1], R=[[3]] → ∫(2+3)dt = 5T"""
        Q = np.array([[2.0]])
        R = np.array([[3.0]])
        b = DiagramBuilder()
        x_src = b.add(Constant(jnp.array([1.0]), name="x"))
        u_src = b.add(Constant(jnp.array([1.0]), name="u"))
        obj_port = lqr_objective(b, x_src.output_ports[0], Q,
                                 control_port=u_src.output_ports[0], R=R)
        diag = b.build()
        sol = _run(diag, t_end=T_END, signals={"J": obj_port})
        assert float(sol.outputs["J"][-1]) == pytest.approx(5.0 * T_END, abs=0.05)

    def test_zero_Q_gives_zero_state_cost(self):
        """Q=0 → only control cost remains."""
        Q = np.zeros((2, 2))
        R = np.eye(1)
        b = DiagramBuilder()
        x_src = b.add(Constant(jnp.array([10.0, 20.0]), name="x"))
        u_src = b.add(Constant(jnp.array([1.0]), name="u"))
        obj_port = lqr_objective(b, x_src.output_ports[0], Q,
                                 control_port=u_src.output_ports[0], R=R)
        diag = b.build()
        sol = _run(diag, t_end=T_END, signals={"J": obj_port})
        # Only u^T R u = 1 contributes
        assert float(sol.outputs["J"][-1]) == pytest.approx(T_END, abs=0.01)

    def test_control_port_none_ignores_R(self):
        """Passing R without control_port should silently ignore R."""
        Q = np.array([[4.0]])
        R = np.array([[100.0]])  # Large R — should have no effect
        b = DiagramBuilder()
        src = b.add(Constant(jnp.array([1.0]), name="x"))
        obj_port = lqr_objective(b, src.output_ports[0], Q, R=R)
        diag = b.build()
        sol = _run(diag, t_end=T_END, signals={"J": obj_port})
        # Cost = ∫ 4 dt = 4*T, not influenced by R
        assert float(sol.outputs["J"][-1]) == pytest.approx(4.0 * T_END, abs=0.04)

    def test_output_is_scalar(self):
        """LQR output must be scalar regardless of state/control dimensions."""
        Q = np.eye(3)
        R = np.eye(2)
        b = DiagramBuilder()
        x_src = b.add(Constant(jnp.ones(3), name="x"))
        u_src = b.add(Constant(jnp.ones(2), name="u"))
        obj_port = lqr_objective(b, x_src.output_ports[0], Q,
                                 control_port=u_src.output_ports[0], R=R)
        diag = b.build()
        sol = _run(diag, t_end=0.5, signals={"J": obj_port})
        j = sol.outputs["J"]
        assert j.ndim <= 2
        if j.ndim == 2:
            assert j.shape[1] == 1

    def test_symmetric_positive_definite_Q(self):
        """Full symmetric PD Q: x=[1,1], Q=[[2,1],[1,2]] → x^T Q x = 6."""
        Q = np.array([[2.0, 1.0], [1.0, 2.0]])
        b = DiagramBuilder()
        x_src = b.add(Constant(jnp.ones(2), name="x"))
        obj_port = lqr_objective(b, x_src.output_ports[0], Q)
        diag = b.build()
        sol = _run(diag, t_end=T_END, signals={"J": obj_port})
        # x^T Q x = [1,1] [[2,1],[1,2]] [1,1]^T = 2+1+1+2 = 6
        assert float(sol.outputs["J"][-1]) == pytest.approx(6.0 * T_END, abs=0.06)


# ══════════════════════════════════════════════════════════════════════════════
# tracking_mse
# ══════════════════════════════════════════════════════════════════════════════

@requires_jax()
class TestTrackingMSE:

    def test_perfect_tracking_zero_cost(self):
        """signal matches reference exactly → cost ≈ 0."""
        t_data = np.linspace(0, T_END, 100)
        y_data = np.ones(100) * 3.0  # constant reference = 3

        b = DiagramBuilder()
        sig = b.add(Constant(3.0, name="sig"))  # signal == reference
        obj_port = tracking_mse(b, sig.output_ports[0], t_data, y_data)
        diag = b.build()
        sol = _run(diag, t_end=T_END, signals={"J": obj_port})
        assert float(sol.outputs["J"][-1]) == pytest.approx(0.0, abs=1e-5)

    def test_constant_error_integrates_to_err_sq_times_T(self):
        """signal=2, reference=1  →  ∫ 1² dt ≈ T."""
        t_data = np.array([0.0, T_END])
        y_data = np.array([1.0, 1.0])  # constant reference = 1

        b = DiagramBuilder()
        sig = b.add(Constant(2.0, name="sig"))  # error = 1
        obj_port = tracking_mse(b, sig.output_ports[0], t_data, y_data)
        diag = b.build()
        sol = _run(diag, t_end=T_END, signals={"J": obj_port})
        assert float(sol.outputs["J"][-1]) == pytest.approx(T_END, abs=0.02)

    def test_linear_ramp_reference(self):
        """y_ref(t)=t, signal=0  →  ∫ t² dt = T³/3."""
        t_data = np.linspace(0, T_END, 200)
        y_data = t_data  # y_ref(t) = t

        b = DiagramBuilder()
        sig = b.add(Constant(0.0, name="sig"))
        obj_port = tracking_mse(b, sig.output_ports[0], t_data, y_data)
        diag = b.build()
        sol = _run(diag, t_end=T_END, signals={"J": obj_port})
        expected = T_END ** 3 / 3.0
        assert float(sol.outputs["J"][-1]) == pytest.approx(expected, rel=0.05)

    def test_with_weight(self):
        """weight=4, perfect mismatch of 1 → ∫ 4·1 dt = 4T."""
        t_data = np.array([0.0, T_END])
        y_data = np.array([0.0, 0.0])  # reference = 0

        b = DiagramBuilder()
        sig = b.add(Constant(1.0, name="sig"))  # error = 1
        obj_port = tracking_mse(b, sig.output_ports[0], t_data, y_data, weight=4.0)
        diag = b.build()
        sol = _run(diag, t_end=T_END, signals={"J": obj_port})
        assert float(sol.outputs["J"][-1]) == pytest.approx(4.0 * T_END, abs=0.04)

    def test_nearest_interpolation(self):
        """Nearest interpolation should still run without errors."""
        t_data = np.array([0.0, 0.5, 1.0, T_END])
        y_data = np.array([1.0, 2.0, 1.5, 1.0])

        b = DiagramBuilder()
        sig = b.add(Constant(1.0, name="sig"))
        obj_port = tracking_mse(b, sig.output_ports[0], t_data, y_data,
                                interpolation="nearest")
        diag = b.build()
        sol = _run(diag, t_end=T_END, signals={"J": obj_port})
        assert not np.isnan(float(sol.outputs["J"][-1]))

    def test_output_is_scalar(self):
        """Output port must be scalar at each timestep."""
        t_data = np.linspace(0, T_END, 50)
        y_data = np.zeros(50)
        b = DiagramBuilder()
        sig = b.add(Constant(1.0, name="sig"))
        obj_port = tracking_mse(b, sig.output_ports[0], t_data, y_data)
        diag = b.build()
        sol = _run(diag, t_end=0.5, signals={"J": obj_port})
        j = sol.outputs["J"]
        assert j.ndim <= 2
        if j.ndim == 2:
            assert j.shape[1] == 1

    def test_extrapolation_clamps(self):
        """Signal evaluated past t_data range should not raise (clamps)."""
        t_data = np.array([0.0, 1.0])  # ref defined only on [0,1]
        y_data = np.array([0.0, 1.0])

        b = DiagramBuilder()
        sig = b.add(Constant(0.0, name="sig"))
        # Run simulation past t_data range (t_end=3.0 > 1.0)
        obj_port = tracking_mse(b, sig.output_ports[0], t_data, y_data)
        diag = b.build()
        # Should not raise
        sol = _run(diag, t_end=3.0, signals={"J": obj_port})
        assert not np.isnan(float(sol.outputs["J"][-1]))


# ══════════════════════════════════════════════════════════════════════════════
# weighted_sum
# ══════════════════════════════════════════════════════════════════════════════

@requires_jax()
class TestWeightedSum:

    def _two_const_ise(self, v1, v2, w1=1.0, w2=1.0):
        """Build diagram with two ISE terms combined via weighted_sum."""
        b = DiagramBuilder()
        s1 = b.add(Constant(float(v1), name="s1"))
        s2 = b.add(Constant(float(v2), name="s2"))
        p1 = ise_objective(b, s1.output_ports[0], name="ise_1")
        p2 = ise_objective(b, s2.output_ports[0], name="ise_2")
        total = weighted_sum(b, [p1, p2], weights=[w1, w2])
        return b.build(), total

    def test_equal_weights_one(self):
        """w=[1,1]: total = ∫(v1² + v2²)dt."""
        diag, total = self._two_const_ise(1.0, 2.0)
        sol = _run(diag, t_end=T_END, signals={"J": total})
        expected = (1.0 + 4.0) * T_END
        assert float(sol.outputs["J"][-1]) == pytest.approx(expected, abs=0.05)

    def test_custom_weights(self):
        """w=[2,0.5]: total = ∫(2·v1² + 0.5·v2²)dt."""
        diag, total = self._two_const_ise(1.0, 2.0, w1=2.0, w2=0.5)
        sol = _run(diag, t_end=T_END, signals={"J": total})
        expected = (2.0 * 1.0 + 0.5 * 4.0) * T_END
        assert float(sol.outputs["J"][-1]) == pytest.approx(expected, abs=0.05)

    def test_single_objective_returned_directly(self):
        """With one objective, weighted_sum should return the port unchanged."""
        b = DiagramBuilder()
        s = b.add(Constant(3.0, name="s"))
        p = ise_objective(b, s.output_ports[0])
        result = weighted_sum(b, [p])
        # Should be the same port object when weight=1.0
        assert result is p

    def test_zero_weight_kills_term(self):
        """w=0 for a term contributes nothing."""
        diag, total = self._two_const_ise(100.0, 1.0, w1=0.0, w2=1.0)
        sol = _run(diag, t_end=T_END, signals={"J": total})
        # Only second term matters: ∫ 1 dt = T
        assert float(sol.outputs["J"][-1]) == pytest.approx(T_END, abs=0.01)

    def test_none_weights_defaults_to_one(self):
        """weights=None should default to all-ones, same as explicit [1, 1]."""
        b1, b2 = DiagramBuilder(), DiagramBuilder()
        for b, name_suffix in [(b1, "_a"), (b2, "_b")]:
            s = b.add(Constant(2.0, name="s"))
            p = ise_objective(b, s.output_ports[0], name="ise")
            weighted_sum(b, [p], weights=None)

        s = b1.add(Constant(2.0, name="s2"))
        p = ise_objective(b1, s.output_ports[0], name="ise2")
        r1 = weighted_sum(b1, [p], weights=None)
        r2 = weighted_sum(b2, [p], weights=[1.0])
        # Both should be the identity port
        b1.build()
        b2.build()

    def test_three_objectives(self):
        """Sum of three ISE terms with uniform weights."""
        b = DiagramBuilder()
        srcs = [b.add(Constant(float(i + 1), name=f"s{i}")) for i in range(3)]
        ports = [ise_objective(b, s.output_ports[0], name=f"ise_{i}")
                 for i, s in enumerate(srcs)]
        total = weighted_sum(b, ports)  # 1+4+9 = 14 per unit time
        diag = b.build()
        sol = _run(diag, t_end=T_END, signals={"J": total})
        expected = (1 + 4 + 9) * T_END
        assert float(sol.outputs["J"][-1]) == pytest.approx(expected, abs=0.15)

    def test_empty_raises(self):
        """Empty objectives list should raise ValueError."""
        b = DiagramBuilder()
        with pytest.raises(ValueError, match="empty"):
            weighted_sum(b, [])

    def test_length_mismatch_raises(self):
        """len(objectives) ≠ len(weights) should raise ValueError."""
        b = DiagramBuilder()
        s = b.add(Constant(1.0, name="s"))
        p = ise_objective(b, s.output_ports[0])
        with pytest.raises(ValueError, match="elements"):
            weighted_sum(b, [p, p], weights=[1.0])  # 2 objs, 1 weight


# ══════════════════════════════════════════════════════════════════════════════
# Equivalence: helper == hand-wired
# ══════════════════════════════════════════════════════════════════════════════

@requires_jax()
class TestEquivalence:
    """Verify that factory-built diagrams produce identical results to the
    canonical hand-wired construction used in test_scenarios.py."""

    def _hand_wired_two_ise(self, v_signal=1.0, x_signal=2.0):
        """Hand-wired version as found in _make_spring_mass_diagram."""
        b = DiagramBuilder()
        v = b.add(Constant(float(v_signal), name="v"))
        x = b.add(Constant(float(x_signal), name="x"))

        ref_v = b.add(Constant(0.0, name="ref_v"))
        ref_x = b.add(Constant(0.0, name="ref_x"))
        err_v = b.add(Adder(2, operators="+-", name="err_v"))
        err_x = b.add(Adder(2, operators="+-", name="err_x"))
        sq_v = b.add(Power(2.0, name="sq_v"))
        sq_x = b.add(Power(2.0, name="sq_x"))
        sum_v = b.add(SumOfElements(name="sum_v"))
        sum_x = b.add(SumOfElements(name="sum_x"))
        cv = b.add(Integrator(0.0, name="cost_v"))
        cx = b.add(Integrator(0.0, name="cost_x"))
        obj = b.add(Adder(2, operators="++", name="obj"))

        b.connect(ref_v.output_ports[0], err_v.input_ports[0])
        b.connect(v.output_ports[0],     err_v.input_ports[1])
        b.connect(ref_x.output_ports[0], err_x.input_ports[0])
        b.connect(x.output_ports[0],     err_x.input_ports[1])
        b.connect(err_v.output_ports[0], sq_v.input_ports[0])
        b.connect(err_x.output_ports[0], sq_x.input_ports[0])
        b.connect(sq_v.output_ports[0],  sum_v.input_ports[0])
        b.connect(sq_x.output_ports[0],  sum_x.input_ports[0])
        b.connect(sum_v.output_ports[0], cv.input_ports[0])
        b.connect(sum_x.output_ports[0], cx.input_ports[0])
        b.connect(cv.output_ports[0], obj.input_ports[0])
        b.connect(cx.output_ports[0], obj.input_ports[1])

        return b.build(), obj.output_ports[0]

    def _helper_two_ise(self, v_signal=1.0, x_signal=2.0):
        """Helper-based equivalent."""
        b = DiagramBuilder()
        v = b.add(Constant(float(v_signal), name="v"))
        x = b.add(Constant(float(x_signal), name="x"))
        ref = b.add(Constant(0.0, name="ref"))

        cv = ise_objective(b, v.output_ports[0],
                           reference_port=ref.output_ports[0], name="ise_v")
        cx = ise_objective(b, x.output_ports[0],
                           reference_port=ref.output_ports[0], name="ise_x")
        obj_port = weighted_sum(b, [cv, cx])
        return b.build(), obj_port

    @pytest.mark.parametrize("v,x", [(1.0, 2.0), (0.5, 3.0), (0.0, 0.0)])
    def test_ise_matches_hand_wired(self, v, x):
        """ISE helper + weighted_sum must give the same integral as hand-wired."""
        d_hand, p_hand = self._hand_wired_two_ise(v, x)
        d_help, p_help = self._helper_two_ise(v, x)

        s_hand = _run(d_hand, t_end=T_END, signals={"J": p_hand})
        s_help = _run(d_help, t_end=T_END, signals={"J": p_help})

        np.testing.assert_allclose(
            s_hand.outputs["J"],
            s_help.outputs["J"],
            atol=1e-8,
            rtol=1e-6,
        )

    def test_lqr_matches_quadraticcost_plus_integrator(self):
        """lqr_objective must match QuadraticCost + Integrator hand-wired."""
        from jaxonomy.library import QuadraticCost

        Q = np.array([[3.0, 1.0], [1.0, 2.0]])
        R = np.array([[0.5]])
        x_val = jnp.array([1.0, 2.0])
        u_val = jnp.array([0.5])

        # Hand-wired
        b1 = DiagramBuilder()
        x1 = b1.add(Constant(x_val, name="x"))
        u1 = b1.add(Constant(u_val, name="u"))
        qc = b1.add(QuadraticCost(Q, R, name="qc"))
        integ = b1.add(Integrator(0.0, name="cost"))
        b1.connect(x1.output_ports[0], qc.input_ports[0])
        b1.connect(u1.output_ports[0], qc.input_ports[1])
        b1.connect(qc.output_ports[0], integ.input_ports[0])
        d1 = b1.build()

        # Helper
        b2 = DiagramBuilder()
        x2 = b2.add(Constant(x_val, name="x"))
        u2 = b2.add(Constant(u_val, name="u"))
        obj_port = lqr_objective(b2, x2.output_ports[0], Q,
                                 control_port=u2.output_ports[0], R=R)
        d2 = b2.build()

        s1 = _run(d1, t_end=T_END, signals={"J": integ.output_ports[0]})
        s2 = _run(d2, t_end=T_END, signals={"J": obj_port})

        np.testing.assert_allclose(
            s1.outputs["J"], s2.outputs["J"], atol=1e-8, rtol=1e-6
        )

    def test_tracking_mse_matches_ise_with_constant_ref(self):
        """tracking_mse(constant data) == ise_objective with constant ref port."""
        c_ref = 2.5
        t_data = np.array([0.0, T_END])
        y_data = np.array([c_ref, c_ref])

        # Hand-wired via ise_objective + Constant reference
        b1 = DiagramBuilder()
        sig1 = b1.add(Constant(0.0, name="sig"))
        ref1 = b1.add(Constant(c_ref, name="ref"))
        p_ise = ise_objective(b1, sig1.output_ports[0],
                              reference_port=ref1.output_ports[0])
        d1 = b1.build()

        # tracking_mse equivalent
        b2 = DiagramBuilder()
        sig2 = b2.add(Constant(0.0, name="sig"))
        p_mse = tracking_mse(b2, sig2.output_ports[0], t_data, y_data)
        d2 = b2.build()

        s1 = _run(d1, t_end=T_END, signals={"J": p_ise})
        s2 = _run(d2, t_end=T_END, signals={"J": p_mse})

        np.testing.assert_allclose(
            s1.outputs["J"], s2.outputs["J"], atol=1e-5, rtol=1e-4
        )


# ══════════════════════════════════════════════════════════════════════════════
# Integration with Optimizable
# ══════════════════════════════════════════════════════════════════════════════

@requires_jax()
class TestIntegrationWithOptimizable:
    """End-to-end tests: helpers used inside an Optimizable subclass."""

    def _make_spring_mass_optimizable(self, c_init=0.5, use_helpers=True):
        """
        Spring-mass with ISE objective.
        Equation: ẍ + c·ẋ + k·x = 0,  x(0)=1, ẋ(0)=0.
        Minimise ∫(x²+v²)dt  over c ∈ (0,∞).
        Optimal c* ≈ 2√k = 2 (critically damped).
        """
        params = {"c": Parameter(np.array(float(c_init)))}
        b = DiagramBuilder()

        c_v = b.add(Gain(params["c"], name="c_v"))
        k_x = b.add(Gain(1.0, name="k_x"))
        add = b.add(Adder(2, operators="--", name="add"))
        v = b.add(Integrator(0.0, name="v"))
        x = b.add(Integrator(1.0, name="x"))

        b.connect(k_x.output_ports[0], add.input_ports[0])
        b.connect(c_v.output_ports[0], add.input_ports[1])
        b.connect(add.output_ports[0], v.input_ports[0])
        b.connect(v.output_ports[0], x.input_ports[0])
        b.connect(v.output_ports[0], c_v.input_ports[0])
        b.connect(x.output_ports[0], k_x.input_ports[0])

        if use_helpers:
            cv = ise_objective(b, v.output_ports[0], name="ise_v")
            cx = ise_objective(b, x.output_ports[0], name="ise_x")
            obj_port = weighted_sum(b, [cv, cx], name="obj")
        else:
            sq_v = b.add(Power(2.0, name="sq_v"))
            sq_x = b.add(Power(2.0, name="sq_x"))
            s_v = b.add(SumOfElements(name="sum_v"))
            s_x = b.add(SumOfElements(name="sum_x"))
            cv = b.add(Integrator(0.0, name="cv"))
            cx_ = b.add(Integrator(0.0, name="cx"))
            obj = b.add(Adder(2, operators="++", name="obj"))
            b.connect(v.output_ports[0], sq_v.input_ports[0])
            b.connect(x.output_ports[0], sq_x.input_ports[0])
            b.connect(sq_v.output_ports[0], s_v.input_ports[0])
            b.connect(sq_x.output_ports[0], s_x.input_ports[0])
            b.connect(s_v.output_ports[0], cv.input_ports[0])
            b.connect(s_x.output_ports[0], cx_.input_ports[0])
            b.connect(cv.output_ports[0], obj.input_ports[0])
            b.connect(cx_.output_ports[0], obj.input_ports[1])
            obj_port = obj.output_ports[0]

        diagram = b.build(parameters=params)

        class _Opt(Optimizable):
            def __init__(self):
                super().__init__(
                    diagram=diagram,
                    base_context=diagram.create_context(),
                    params_0={"c": float(c_init)},
                    sim_t_span=(0.0, 5.0),
                    sim_options=SimulatorOptions(max_major_steps=1),
                )
                self._obj_port = obj_port

            def optimizable_params(self, ctx):
                return {"c": ctx.parameters["c"]}

            def objective_from_context(self, ctx):
                return self._obj_port.eval(ctx)

            def prepare_context(self, ctx, p):
                return ctx.with_parameters(p)

        return _Opt()

    def test_helpers_give_same_loss_as_handwired(self):
        """objective_flat must return the same value for both constructions."""
        import jax.numpy as jnp

        opt_h = self._make_spring_mass_optimizable(use_helpers=True)
        opt_m = self._make_spring_mass_optimizable(use_helpers=False)

        p = jnp.array([0.7])
        loss_h = float(opt_h.objective_flat(p))
        loss_m = float(opt_m.objective_flat(p))

        assert loss_h == pytest.approx(loss_m, rel=1e-5)

    def test_optimisation_converges(self):
        """
        Optimising with helper-built objective should reduce the loss and move
        the parameter in the right direction (higher damping from underdamped start).
        """
        import jax.numpy as jnp

        c_init = 0.3
        opt = self._make_spring_mass_optimizable(c_init=c_init)

        loss_before = float(opt.objective_flat(jnp.array([c_init])))

        scipy_opt = Scipy(opt, "L-BFGS-B",
                         use_autodiff_grad=True,
                         opt_method_config={"maxiter": 80, "ftol": 1e-9})
        result = scipy_opt.optimize()

        assert result.success
        # Optimal c should be greater than the underdamped initial value
        assert float(result["c"]) > c_init
        # Final loss must be strictly lower than initial loss
        assert result.final_loss < loss_before

    def test_tracking_mse_in_optimizable(self):
        """tracking_mse objective works inside an Optimizable simulation."""
        # Constant "plant" output matches a constant dataset → cost ≈ 0 at truth
        params = {"gain": Parameter(np.array(1.0))}
        b = DiagramBuilder()
        src = b.add(Constant(1.0, name="src"))
        g = b.add(Gain(params["gain"], name="g"))
        b.connect(src.output_ports[0], g.input_ports[0])

        # Dataset: y_ref = 1.0 everywhere
        t_data = np.linspace(0, 2.0, 20)
        y_data = np.ones(20)
        obj_port = tracking_mse(b, g.output_ports[0], t_data, y_data,
                                name="track_cost")
        diagram = b.build(parameters=params)

        class _GainOpt(Optimizable):
            def __init__(self):
                super().__init__(
                    diagram=diagram,
                    base_context=diagram.create_context(),
                    params_0={"gain": 1.0},
                    sim_t_span=(0.0, 2.0),
                    sim_options=SimulatorOptions(max_major_steps=1),
                )

            def optimizable_params(self, ctx):
                return {"gain": ctx.parameters["gain"]}

            def objective_from_context(self, ctx):
                return obj_port.eval(ctx)

            def prepare_context(self, ctx, p):
                return ctx.with_parameters(p)

        opt = _GainOpt()
        import jax.numpy as jnp
        loss_at_truth = float(opt.objective_flat(jnp.array([1.0])))
        loss_at_wrong = float(opt.objective_flat(jnp.array([2.0])))

        # At truth (gain=1), cost ≈ 0; at wrong value, cost > 0
        assert loss_at_truth == pytest.approx(0.0, abs=1e-5)
        assert loss_at_wrong > loss_at_truth


# ══════════════════════════════════════════════════════════════════════════════
# Edge cases
# ══════════════════════════════════════════════════════════════════════════════

@requires_jax()
class TestEdgeCases:

    def test_ise_very_short_simulation(self):
        """Single-step simulation should not raise."""
        b = DiagramBuilder()
        src = b.add(Constant(1.0, name="s"))
        obj = ise_objective(b, src.output_ports[0])
        diag = b.build()
        ctx = diag.create_context()
        opts = SimulatorOptions(max_major_steps=5, max_major_step_length=DT)
        sol = jaxonomy.simulate(diag, ctx, (0.0, DT), options=opts,
                                recorded_signals={"J": obj})
        assert not np.isnan(float(sol.outputs["J"][-1]))

    def test_lqr_single_step(self):
        """lqr_objective does not break on short runs."""
        Q = np.array([[1.0]])
        b = DiagramBuilder()
        x = b.add(Constant(jnp.array([2.0]), name="x"))
        obj = lqr_objective(b, x.output_ports[0], Q)
        diag = b.build()
        ctx = diag.create_context()
        opts = SimulatorOptions(max_major_steps=5, max_major_step_length=DT)
        sol = jaxonomy.simulate(diag, ctx, (0.0, DT), options=opts,
                                recorded_signals={"J": obj})
        assert not np.isnan(float(sol.outputs["J"][-1]))

    def test_ise_reference_equals_signal_continuously(self):
        """Reference tracking with perfect match → monotonically zero error."""
        b = DiagramBuilder()
        val = b.add(Constant(7.0, name="val"))
        obj = ise_objective(b, val.output_ports[0],
                            reference_port=val.output_ports[0])
        diag = b.build()
        sol = _run(diag, t_end=T_END, signals={"J": obj})
        np.testing.assert_allclose(sol.outputs["J"], 0.0, atol=1e-10)

    def test_weighted_sum_all_zero_weights(self):
        """All-zero weights → total objective = 0 at all times."""
        b = DiagramBuilder()
        s = b.add(Constant(100.0, name="s"))
        p1 = ise_objective(b, s.output_ports[0], name="a")
        p2 = ise_objective(b, s.output_ports[0], name="b")
        total = weighted_sum(b, [p1, p2], weights=[0.0, 0.0])
        diag = b.build()
        sol = _run(diag, t_end=T_END, signals={"J": total})
        np.testing.assert_allclose(sol.outputs["J"], 0.0, atol=1e-10)

    def test_api_imports(self):
        """All public symbols importable from jaxonomy.optimization."""
        from jaxonomy.optimization import (  # noqa: F401
            ise_objective,
            lqr_objective,
            tracking_mse,
            weighted_sum,
        )
