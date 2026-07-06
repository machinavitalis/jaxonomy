# SPDX-License-Identifier: MIT

"""Regression tests for the second batch of DX follow-ups (Tier A/B/C).

Covers:
  A1  scalar_cost_simulate helper
  A2  Simulator.advance_to JIT cache reuse
  A3  max_major_step_size alias for max_major_step_length
  A4  requires_inputs inference + clear error on inputs[i] when trimmed
  A5  jax.grad-through-FMU clear error (mechanism tested in isolation)
  B1  empty output_ports actionable IndexError
  B2  context[port] -> suggest port.eval
  B4  MLP.mlp pre-context error
  B5  PyTorch/TensorFlow *Predictor aliases
  B6  int_time_scale="auto" default
  C2  LeafSystem.continuous_state_default property
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import jaxonomy
from jaxonomy.framework.dependency_graph import DependencyTicket
from jaxonomy.library import Integrator


# ---------------------------------------------------------------------------
# A3 — max_major_step_size alias
# ---------------------------------------------------------------------------


def test_max_major_step_size_alias():
    o = jaxonomy.SimulatorOptions(max_major_step_size=0.1)
    assert o.max_major_step_length == 0.1
    assert o.max_major_step_size == 0.1
    o2 = jaxonomy.SimulatorOptions(max_major_step_length=0.25)
    assert o2.max_major_step_size == 0.25


def test_max_major_step_size_alias_survives_replace():
    import dataclasses
    o = jaxonomy.SimulatorOptions(max_major_step_length=0.2)
    o3 = dataclasses.replace(o, max_major_steps=50)
    assert o3.max_major_step_length == 0.2
    assert o3.max_major_step_size == 0.2


# ---------------------------------------------------------------------------
# A4 — requires_inputs inference + clear error
# ---------------------------------------------------------------------------


def _block_with_input():
    b = jaxonomy.LeafSystem()
    b.declare_input_port(name="u")
    return b


def test_requires_inputs_inferred_false_only_for_nothing():
    b = _block_with_input()
    assert b._resolve_requires_inputs(None, [DependencyTicket.nothing]) is False
    # Non-input prereqs default to True (legacy-safe: prereqs can be transitive).
    assert b._resolve_requires_inputs(None, [DependencyTicket.xd]) is True
    assert b._resolve_requires_inputs(None, [DependencyTicket.xcdot]) is True
    assert b._resolve_requires_inputs(None, None) is True
    # Explicit value always wins.
    assert b._resolve_requires_inputs(False, [DependencyTicket.u]) is False


def test_requires_inputs_mismatch_gives_clear_error():
    class Bad(jaxonomy.LeafSystem):
        def __init__(self):
            super().__init__()
            self.declare_input_port(name="u")
            self.declare_output_port(
                lambda t, s, *u, **p: 2.0 * u[0],
                name="y",
                requires_inputs=False,  # but the callback reads u[0]
            )

    b = jaxonomy.DiagramBuilder()
    src = b.add(jaxonomy.library.Constant(1.0))
    bad = b.add(Bad())
    b.connect(src.output_ports[0], bad.input_ports[0])
    diag = b.build()
    with pytest.raises(Exception) as exc:
        diag.create_context()
    assert "requires_inputs" in str(exc.value)


# ---------------------------------------------------------------------------
# B1 — empty output_ports actionable error
# ---------------------------------------------------------------------------


def test_empty_output_ports_actionable_error():
    class Plant(jaxonomy.LeafSystem):
        def __init__(self):
            super().__init__()
            self.declare_continuous_state(2, ode=lambda t, s, *u, **p: s.continuous_state)

    p = Plant()
    with pytest.raises(IndexError, match="no output ports"):
        p.output_ports[0]


def test_normal_output_ports_unaffected():
    g = jaxonomy.library.Gain(2.0)
    assert g.output_ports[0].name == "out_0"


# ---------------------------------------------------------------------------
# B2 — context[port] error
# ---------------------------------------------------------------------------


def test_context_indexed_by_output_port_suggests_eval():
    b = jaxonomy.DiagramBuilder()
    src = b.add(jaxonomy.library.Constant(1.0))
    g = b.add(jaxonomy.library.Gain(2.0, name="g"))
    b.connect(src.output_ports[0], g.input_ports[0])
    b.export_output(g.output_ports[0])
    diag = b.build()
    ctx = diag.create_context()
    with pytest.raises(TypeError, match="port.eval"):
        _ = ctx[g.output_ports[0]]


# ---------------------------------------------------------------------------
# B4 — MLP.mlp pre-context error
# ---------------------------------------------------------------------------


def test_mlp_attribute_pre_context_error():
    pytest.importorskip("equinox")
    from jaxonomy.library import MLP

    m = MLP(in_size=2, out_size=1, width_size=4, depth=2, seed=0)
    with pytest.raises(AttributeError, match="create_context"):
        _ = m.mlp


# ---------------------------------------------------------------------------
# B5 — predictor aliases
# ---------------------------------------------------------------------------


def test_predictor_aliases():
    from jaxonomy.library import (
        PyTorch,
        TensorFlow,
        PyTorchPredictor,
        TensorFlowPredictor,
    )

    assert PyTorchPredictor is PyTorch
    assert TensorFlowPredictor is TensorFlow


# ---------------------------------------------------------------------------
# B6 — int_time_scale="auto" default
# ---------------------------------------------------------------------------


class _DiscreteClock(jaxonomy.LeafSystem):
    def __init__(self, dt, **kwargs):
        super().__init__(**kwargs)
        self.declare_discrete_state(default_value=jnp.array(0.0))
        self.declare_periodic_update(
            lambda t, s, **p: s.discrete_state + 1.0, period=dt, offset=0.0
        )
        self.declare_output_port(
            lambda t, s, **p: s.discrete_state,
            prerequisites_of_calc=[DependencyTicket.xd],
        )


def test_int_time_scale_auto_default_handles_long_sim():
    assert jaxonomy.SimulatorOptions().int_time_scale == "auto"
    tf = 3.156e8  # 10 years — overflows the picosecond default scale
    sys = _DiscreteClock(0.01 * tf)
    ctx = sys.create_context()
    res = jaxonomy.simulate(sys, ctx, (0.0, tf))
    assert float(res.context.time) == pytest.approx(tf, rel=1e-6)


def test_int_time_scale_explicit_fine_still_raises_on_long_sim():
    tf = 3.156e8
    sys = _DiscreteClock(0.01 * tf)
    ctx = sys.create_context()
    with pytest.raises(RuntimeError):
        jaxonomy.simulate(
            sys, ctx, (0.0, tf),
            options=jaxonomy.SimulatorOptions(int_time_scale=1e-12),
        )


def test_int_time_scale_auto_is_trace_safe_under_grad():
    """T-B6-followup-int-time-scale-trace-safety: the "auto" default calls
    IntegerTime.set_scale inside the autodiff-through-simulate trace; the
    integer-time bound (max_float_time) must stay a host-side concrete value
    so the representability check's float() does not hit a tracer. Regression
    for a ConcretizationTypeError introduced by the auto default.
    """
    class _P(jaxonomy.LeafSystem):
        def __init__(self, **k):
            super().__init__(**k)
            self.declare_continuous_state(1, ode=self._o)
            self.declare_dynamic_parameter("k", jnp.asarray(1.0))
            self.declare_continuous_state_output()

        def _o(self, t, s, *u, **p):
            return -p["k"] * s.continuous_state

    sys = _P()
    ctx = sys.create_context().with_continuous_state(jnp.array([1.0]))
    opts = jaxonomy.SimulatorOptions(
        math_backend="jax", enable_autodiff=True, max_major_steps=100,
    )  # int_time_scale defaults to "auto"

    def f(k):
        c = ctx.with_parameter("k", k)
        return jnp.sum(
            jaxonomy.simulate(sys, c, (0.0, 2.0), options=opts).context.continuous_state
        )

    g = float(jax.grad(f)(jnp.array(1.0)))
    # d/dk sum(e^{-2k}) = -2 e^{-2} ≈ -0.27067
    assert g == pytest.approx(-2.0 * math.exp(-2.0), rel=1e-2)


# ---------------------------------------------------------------------------
# C2 — continuous_state_default property
# ---------------------------------------------------------------------------


def test_continuous_state_default_property():
    class Plant(jaxonomy.LeafSystem):
        def __init__(self):
            super().__init__()
            self.declare_continuous_state(
                default_value=jnp.array([1.0, 2.0, 3.0]),
                ode=lambda t, s, *u, **p: s.continuous_state,
            )

    p = Plant()
    np.testing.assert_array_equal(
        np.asarray(p.continuous_state_default), np.array([1.0, 2.0, 3.0])
    )
    # No continuous state -> None.
    g = jaxonomy.library.Gain(2.0)
    assert g.continuous_state_default is None


# ---------------------------------------------------------------------------
# A1 / A2 — scalar_cost_simulate + advance_to caching
# ---------------------------------------------------------------------------


class _DecayPlant(jaxonomy.LeafSystem):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.declare_continuous_state(1, ode=self._ode)
        self.declare_dynamic_parameter("k", jnp.asarray(1.0))
        self.declare_continuous_state_output()

    def _ode(self, t, s, *u, **p):
        return -p["k"] * s.continuous_state


def _build_cost_diagram():
    b = jaxonomy.DiagramBuilder()
    plant = b.add(_DecayPlant(name="plant"))
    acc = b.add(Integrator(jnp.array([0.0]), name="acc"))
    b.connect(plant.output_ports[0], acc.input_ports[0])
    diag = b.build()
    base = diag.create_context()
    base = base.with_subcontext(
        plant.system_id, base[plant.system_id].with_continuous_state(jnp.array([1.0]))
    )
    return diag, plant, acc, base


def test_scalar_cost_simulate_value_and_grad():
    diag, plant, acc, base = _build_cost_diagram()

    def make_ctx(k):
        leaf = base[plant.system_id].with_parameter("k", k)
        return base.with_subcontext(plant.system_id, leaf)

    cost = lambda ctx: ctx[acc.system_id].continuous_state[0]
    opts = jaxonomy.SimulatorOptions(math_backend="jax", max_major_steps=200)
    f = lambda k: jaxonomy.scalar_cost_simulate(
        diag, make_ctx, (0.0, 3.0), k, cost, options=opts
    )
    val = float(f(jnp.array(1.0)))
    grad = float(jax.grad(f)(jnp.array(1.0)))

    k = 1.0
    analytic_val = (1 - math.exp(-3 * k)) / k
    analytic_grad = ((3 * k * math.exp(-3 * k)) - (1 - math.exp(-3 * k))) / k**2
    assert val == pytest.approx(analytic_val, rel=1e-3)
    assert grad == pytest.approx(analytic_grad, rel=1e-3)

    v2, g2 = jaxonomy.scalar_cost_simulate(
        diag, make_ctx, (0.0, 3.0), jnp.array(1.0), cost,
        options=opts, return_grad=True,
    )
    assert float(v2) == pytest.approx(analytic_val, rel=1e-3)
    assert float(g2) == pytest.approx(analytic_grad, rel=1e-3)


def test_advance_to_reuses_compiled_kernel():
    from jaxonomy.simulation import Simulator

    sys = _DecayPlant()
    opts = jaxonomy.SimulatorOptions(math_backend="jax", max_major_steps=50)
    sim = Simulator(sys, options=opts)

    ctx0 = sys.create_context().with_continuous_state(jnp.array([1.0]))
    s1 = sim.advance_to(1.0, ctx0)
    s1.context.continuous_state.block_until_ready()

    # The jitted advance_to is a stable instance attribute, so a second call
    # with a same-shaped context reuses the compiled kernel.
    import time
    ctx1 = sys.create_context().with_continuous_state(jnp.array([2.0]))
    t0 = time.perf_counter()
    s2 = sim.advance_to(1.0, ctx1)
    s2.context.continuous_state.block_until_ready()
    second = time.perf_counter() - t0

    assert float(s1.context.continuous_state[0]) == pytest.approx(math.exp(-1.0), rel=1e-3)
    assert float(s2.context.continuous_state[0]) == pytest.approx(2 * math.exp(-1.0), rel=1e-3)
    # Cached call should be fast (well under the cold-compile time). Generous
    # bound to avoid CI flakiness.
    assert second < 0.05
