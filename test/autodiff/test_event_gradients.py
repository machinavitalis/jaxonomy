# SPDX-License-Identifier: MIT
"""
Phase C (T-001) — gradient correctness across every event-handling path.

Three axes of parametrization:
  - ``direction`` ∈ {crosses_zero, positive_then_non_positive,
                     negative_then_non_negative}
  - ``solver``   ∈ {dopri5, bdf}
  - ``phase``    ∈ {before_crossing, after_crossing}

The ``edge_detection`` direction is for DT-state changes only and is exercised
by the state-machine file.  ``none`` is a no-op path and has no gradient
semantics to test.

For each (direction, solver, phase) tuple the test:
  1. builds a scalar ODE that will trigger exactly one mode transition for
     after-crossing phase, or zero for before-crossing phase,
  2. differentiates the final state w.r.t. (x0, tf, slope_parameter),
  3. checks AD vs FD within the solver/dtype tolerance bucket.

Marked ``autodiff_full`` — exercised in the nightly job only.
"""

from __future__ import annotations

from enum import IntEnum

import pytest
import jax
import jax.numpy as jnp

import jaxonomy
from jaxonomy.library import Gain

from ._framework import assert_grad_matches_fd, sim_options
from .tolerances import SOLVERS
from jaxonomy.testing.markers import skip_if_not_jax

skip_if_not_jax()

pytestmark = pytest.mark.autodiff_full


# ── shared fixture: single-crossing ODE parameterised by direction ──────────


class _Mode(IntEnum):
    A = 0
    B = 1


def _build_model(direction: str):
    """Build a LeafSystem whose state crosses zero from a direction-appropriate
    sign, with an ``a`` parameter scaling the ramp.

    - "crosses_zero"              : x(0) = +x0, dx/dt = -a in mode A, +a in B
                                    → guard = x, crosses zero going down.
    - "positive_then_non_positive": same dynamics; guard fires only on P→N.
    - "negative_then_non_negative": x(0) = -x0, dx/dt = +a in mode A, -a in B
                                    → guard = x, crosses zero going up.
    """

    positive_start = direction != "negative_then_non_negative"

    class _M(jaxonomy.LeafSystem):
        def __init__(self):
            super().__init__()
            self.declare_dynamic_parameter("a", 1.0)
            self.declare_default_mode(_Mode.A)
            self.declare_continuous_state(shape=(), ode=self._ode)
            self.declare_continuous_state_output()
            self.declare_zero_crossing(
                guard=lambda t, s, **p: s.continuous_state,
                direction=direction,
                name=f"zc_{direction}",
                start_mode=_Mode.A,
                end_mode=_Mode.B,
            )

        def _ode(self, time, state, **params):
            a = params["a"]
            if positive_start:
                return jax.lax.switch(state.mode, [lambda: -a, lambda: a])
            return jax.lax.switch(state.mode, [lambda: a, lambda: -a])

    return _M(), positive_start


_DIRECTIONS = (
    "crosses_zero",
    "positive_then_non_positive",
    "negative_then_non_negative",
)


@pytest.mark.parametrize("direction", _DIRECTIONS)
@pytest.mark.parametrize("solver", SOLVERS)
@pytest.mark.parametrize("phase", ["before_crossing", "after_crossing"])
def test_grad_zero_crossing(direction, solver, phase):
    model, positive_start = _build_model(direction)
    ctx0 = model.create_context()

    # Start 1.0 away from zero with slope magnitude 1.0.  For "before" choose
    # tf < 1.0; for "after" choose tf > 1.0.
    x0_val = 1.0 if positive_start else -1.0
    tf_val = 0.6 if phase == "before_crossing" else 1.4
    a_val = 1.0

    # BDF on a trivially non-stiff 1-D ODE is slow and redundant with Dopri5.
    if solver == "bdf" and phase == "before_crossing":
        pytest.skip("bdf+before_crossing adds no coverage beyond dopri5")

    opts = sim_options(solver, "float64", max_major_steps=50)
    sim = jaxonomy.Simulator(model, options=opts)

    @jax.jit
    def fwd(x0, tf, a):
        ctx = ctx0.with_continuous_state(x0).with_parameter("a", a)
        return sim.advance_to(tf, ctx).context.continuous_state

    assert_grad_matches_fd(
        fwd,
        jnp.array(x0_val),
        jnp.array(tf_val),
        jnp.array(a_val),
        solver=solver,
        dtype="float64",
        block=f"ZeroCrossing[{direction}]",
        extra={"phase": phase},
    )


# ── guard-tied-to-input: guard reads external port, not internal state ─────


@pytest.mark.parametrize("solver", SOLVERS)
def test_grad_zero_crossing_guard_tied_to_input(solver):
    """Guard reads an *input* signal (not internal state). The adjoint must
    still resolve the crossing time as a function of (x0, tf, a)."""

    class Driven(jaxonomy.LeafSystem):
        def __init__(self):
            super().__init__()
            self.declare_dynamic_parameter("a", 1.0)
            self.declare_default_mode(_Mode.A)
            self.declare_input_port(name="u")
            self.declare_continuous_state(shape=(), ode=self._ode)
            self.declare_continuous_state_output()
            self.declare_zero_crossing(
                guard=lambda t, s, u, **p: u - 0.3,
                direction="crosses_zero",
                name="zc_input",
                start_mode=_Mode.A,
                end_mode=_Mode.B,
            )

        def _ode(self, time, state, u, **params):
            a = params["a"]
            return jax.lax.switch(state.mode, [lambda: -a, lambda: a])

    bld = jaxonomy.DiagramBuilder()
    driven = bld.add(Driven())
    # Feed the driven block a time-varying signal so the guard crosses.
    # A Gain on the driven block's output gives u = k·x; pick k so u crosses 0.3.
    gain = bld.add(Gain(0.5, name="half"))
    bld.connect(driven.output_ports[0], gain.input_ports[0])
    bld.connect(gain.output_ports[0], driven.input_ports[0])
    diagram = bld.build()
    ctx0 = diagram.create_context()

    # Starting from x0=1, u=x/2=0.5. Mode A: dx/dt=-a=−1. u(t)=(1−t)/2.
    # Crosses 0.3 at t = 0.4.  Choose tf=0.7 (after).
    opts = sim_options(solver, "float64", max_major_steps=60)

    @jax.jit
    def fwd(x0, tf, a):
        c = ctx0
        c = c.with_subcontext(
            driven.system_id,
            c[driven.system_id].with_continuous_state(x0).with_parameter("a", a),
        )
        res = jaxonomy.simulate(diagram, c, (0.0, tf), options=opts)
        return res.context[driven.system_id].continuous_state

    # tf is NOT differentiable here because `simulate` takes the span
    # statically — only (x0, a) are.
    assert_grad_matches_fd(
        fwd,
        jnp.array(1.0),
        jnp.array(0.7),
        jnp.array(1.0),
        solver=solver,
        dtype="float64",
        block="ZeroCrossing[guard-on-input]",
        argnums=(0, 2),
    )


# ── two-crossing / mode-walk: A → B → A within one simulation ───────────────


@pytest.mark.parametrize("solver", SOLVERS)
def test_grad_two_sequential_mode_transitions(solver):
    """Triangle-wave generator: x descends, hits 0, flips to ascending, hits 1,
    flips to descending.  Two crossings in one simulation; gradient must flow
    through both transitions."""

    class _Tri(jaxonomy.LeafSystem):
        def __init__(self):
            super().__init__()
            self.declare_dynamic_parameter("a", 1.0)
            self.declare_default_mode(_Mode.A)
            self.declare_continuous_state(shape=(), ode=self._ode)
            self.declare_continuous_state_output()
            self.declare_zero_crossing(
                guard=lambda t, s, **p: s.continuous_state,
                direction="positive_then_non_positive",
                name="zc_a_to_b",
                start_mode=_Mode.A,
                end_mode=_Mode.B,
            )
            self.declare_zero_crossing(
                guard=lambda t, s, **p: s.continuous_state - 1.0,
                direction="negative_then_non_negative",
                name="zc_b_to_a",
                start_mode=_Mode.B,
                end_mode=_Mode.A,
            )

        def _ode(self, time, state, **params):
            a = params["a"]
            return jax.lax.switch(state.mode, [lambda: -a, lambda: a])

    model = _Tri()
    ctx0 = model.create_context()

    # x0=1, a=1: descend 1→0 by t=1, ascend 0→1 by t=2.  Pick tf=2.5 to land
    # on the descending branch after the second transition.
    opts = sim_options(solver, "float64", max_major_steps=100)
    sim = jaxonomy.Simulator(model, options=opts)

    @jax.jit
    def fwd(x0, tf, a):
        ctx = ctx0.with_continuous_state(x0).with_parameter("a", a)
        return sim.advance_to(tf, ctx).context.continuous_state

    assert_grad_matches_fd(
        fwd,
        jnp.array(1.0),
        jnp.array(2.5),
        jnp.array(1.0),
        solver=solver,
        dtype="float64",
        block="TwoTransitions",
    )
