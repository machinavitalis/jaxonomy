# SPDX-License-Identifier: MIT

"""T-001c: end-to-end gradient through a StateMachineBuilder-built block.

The underlying ``declare_zero_crossing`` autodiff path is covered by
``test_state_transition_autodiff.py`` and ``test_event_gradients.py``. This
file exercises the *composed* path: a SM block constructed via
:class:`StateMachineBuilder` is embedded in a Diagram driven by a continuous
upstream block, and ``jax.grad`` is taken w.r.t. parameters that propagate
through the SM via its input port.

Bugs surfaced while authoring this test (filed as T-001c-followup):

1. **Saltation crash on blocks with no continuous state.**
   ``jaxonomy/framework/leaf_system.py::_reset_map_adj`` performed
   ``jnp.dot(dg_dx=None, xdot_minus=None)`` for SM blocks during the adjoint
   pass, raising ``TypeError``. Fixed earlier by short-circuiting the
   saltation rank-1 correction when the local block has no continuous state.

1b. **JAX-internal float0 cotangent crash (FIXED 2026-05-20).**
   The saltation fix unmasked a JAX ``_cond_transpose`` failure
   (``ValueError: too many values to unpack (expected 1)`` at
   ``conditionals.py``'s ``out_tree, = set(out_trees)``): the
   ``cond(active, callback, passthrough)`` in ``Event.handle`` /
   ``ZeroCrossingEvent.handle`` produced cotangent pytrees that differed at
   one leaf — a ``Zero`` of ``float0[]`` (strong) vs ``~float0[]`` (weak)
   for the state-machine ``mode`` integer. Fixed by severing the gradient of
   integer / boolean leaves of the context in *both* branches via
   ``stop_gradient`` (``jaxonomy.framework.event._sever_discrete_cotangents``),
   so both branches emit a consistently-typed ``Zero``. The same fix resolves
   the gradient w.r.t. the upstream integrator initial condition (#1c), which
   hit the identical crash — see ``test_sm_gradient_initial_state``.

2. ~~**Discrete-mode SM autodiff returns silent zero.**~~
   Resolved 2026-04-27 — see ``test_sm_discrete_mode_gradient_matches_fd``
   below.

The one remaining strict-xfail (``test_sm_gradient_amplitude_timing_matches_fd``)
is NOT the float0 crash — it pins a *separate, deeper* gap: the pure-timing
saltation (event-time) gradient through a piecewise-constant SM *output*
feeding a continuous integrator is not propagated by the internal custom-VJP
path (the deferred ``T-125-followup-custom-vjp`` integration). With a constant
action (``y = 1.0``) the only ``amplitude`` dependence is the *timing* of the
switch, and AD reports 0 where FD reports ~0.471. Value-path gradients
(constant initial state, input-dependent actions) are correct.
"""

from __future__ import annotations

import numpy as np
import jax
import jax.numpy as jnp
import pytest

import jaxonomy
from jaxonomy import DiagramBuilder, SimulatorOptions, StateMachineBuilder, simulate
from jaxonomy.library import Sine, Integrator, Constant, StateMachine
from jaxonomy.testing import fd_grad
from jaxonomy.testing.markers import skip_if_not_jax

skip_if_not_jax()


SOLVERS = ["dopri5", "bdf"]  # rk4 currently has unrelated issues; covered by smoke test


def _opts(method: str, **kw) -> SimulatorOptions:
    base = dict(
        math_backend="jax",
        enable_autodiff=True,
        ode_solver_method=method,
        rtol=1e-6,
        atol=1e-8,
        max_major_steps=400,
    )
    base.update(kw)
    return SimulatorOptions(**base)


def _build_three_state_sm(time_mode: str = "agnostic", dt: float | None = None):
    """3-state SM driven by an input ``x`` crossing thresholds 0.5 and 1.5.

    Outputs ``y`` set by entry actions on each transition.
    """
    smb = StateMachineBuilder()
    a = smb.add_state("idle")
    b = smb.add_state("running")
    c = smb.add_state("done")
    smb.set_initial_state(a)
    smb.add_transition(a, b, guard="x > 0.5", actions=["y = 1.0"])
    smb.add_transition(b, c, guard="x > 1.5", actions=["y = 2.0"])
    sm = smb.build(name="three_state_sm")
    if time_mode == "agnostic":
        return sm
    return StateMachine(
        sm_data=sm._sm,
        inputs=list(sm._input_names),
        outputs=list(sm._output_names),
        dt=dt,
        time_mode=time_mode,
        name="three_state_sm_disc",
        accelerate_with_jax=False,
    )


def _build_sine_to_sm_diagram(amplitude=1.5, frequency=0.5):
    """Sine -> SM -> Integrator. SM transitions when sin output crosses 0.5/1.5."""
    builder = DiagramBuilder()
    src = builder.add(Sine(amplitude=amplitude, frequency=frequency, name="sine"))
    sm = builder.add(_build_three_state_sm(time_mode="agnostic"))
    integ = builder.add(Integrator(initial_state=0.0, name="integ"))
    builder.connect(src.output_ports[0], sm.input_ports[0])
    builder.connect(sm.output_ports[0], integ.input_ports[0])
    diagram = builder.build(name="sine_to_sm")
    return diagram, src, sm, integ


# ── Forward smoke test (passes) ─────────────────────────────────────────────


@pytest.mark.parametrize("method", SOLVERS + ["rk4"])
def test_sm_forward_smoke_three_solvers(method):
    """Forward pass of a SM-bearing diagram works on every shipped solver."""
    diagram, src, sm, integ = _build_sine_to_sm_diagram()
    ctx = diagram.create_context()
    res = simulate(diagram, ctx, (0.0, 4.0), options=_opts(method))
    final = float(res.context[integ.system_id].continuous_state)
    assert np.isfinite(final), f"{method} produced non-finite final state: {final}"


# ── Static-threshold non-differentiable regression guard (passes) ───────────


def test_sm_static_threshold_not_differentiable():
    """A guard threshold passed as a Python literal at build time has no
    differentiable surface today (T-018: SM has no parameter mechanism in
    guards). This test pins that behaviour: any future gradient path through
    a static threshold must intentionally flip this assertion."""
    smb = StateMachineBuilder()
    a = smb.add_state("a")
    b = smb.add_state("b")
    smb.set_initial_state(a)
    # The literal `0.5` is baked into the guard expression; not a Parameter.
    smb.add_transition(a, b, guard="x > 0.5", actions=["y = 1.0"])
    sm = smb.build(name="static_thresh_sm")
    # We can't even *reach* a threshold parameter — the SM's public surface
    # exposes only inputs/outputs, no per-transition Parameter. Asserting
    # that fact:
    assert not hasattr(sm, "transitions") or not any(
        hasattr(t, "guard_threshold") for t in getattr(sm, "transitions", [])
    ), (
        "SM transitions appear to have a `guard_threshold` Parameter — "
        "T-018 may have landed; flip this test into a real grad-vs-FD check."
    )


# ── Gradient through Sine amplitude (crash fixed; timing-value pinned) ───────


def _amplitude_fwd(method, amp):
    """Terminal Integrator state of Sine(amp) -> 3-state SM -> Integrator."""
    builder = DiagramBuilder()
    # Pass `amp` directly so the dynamic-parameter machinery picks up the
    # tracer; using `float(amp)` at construction trips
    # ConcretizationTypeError under jax.grad.
    src = builder.add(Sine(amplitude=amp, frequency=0.5, name="sine"))
    sm = builder.add(_build_three_state_sm(time_mode="agnostic"))
    integ = builder.add(Integrator(initial_state=0.0, name="integ"))
    builder.connect(src.output_ports[0], sm.input_ports[0])
    builder.connect(sm.output_ports[0], integ.input_ports[0])
    diagram = builder.build(name="sine_to_sm")
    ctx = diagram.create_context()
    res = simulate(diagram, ctx, (0.0, 4.0), options=_opts(method))
    return res.context[integ.system_id].continuous_state.sum()


@pytest.mark.slow
@pytest.mark.parametrize("method", SOLVERS)
def test_sm_gradient_amplitude_no_cond_transpose_crash(method):
    """``jax.grad`` through the SM no longer hits the JAX ``_cond_transpose``
    ``set(out_trees)`` crash (T-001c-followup #1b, fixed 2026-05-20).

    Pre-fix this raised ``ValueError: too many values to unpack (expected
    1)`` from the weak-vs-strong float0 cotangent disagreement on the SM
    ``mode`` integer. The fix (``_sever_discrete_cotangents``) lets the
    reverse pass complete and return a finite gradient.
    """
    g_ad = jax.grad(lambda amp: _amplitude_fwd(method, amp))(jnp.array(1.5))
    assert np.isfinite(float(g_ad)), f"gradient must be finite; got {g_ad}"


@pytest.mark.slow
@pytest.mark.parametrize("method", SOLVERS)
def test_sm_gradient_amplitude_timing_matches_fd(method):
    """Pure-timing saltation gradient w.r.t. amplitude matches FD (FIXED).

    With a constant action (``y = 1.0``) the only ``amplitude`` dependence is
    the *timing* of the SM transition, so this is a pure event-time (saltation)
    gradient through a guard that depends on a parameter entering via an
    upstream input signal.

    History: long strict-xfailed. Two pieces had to land. (1) The saltation
    *propagation* fix (T-001c #1d): differentiate the guard against a refreshed
    port cache and apply the rank-1 correction across the full root context
    (`leaf_system._wrap_reset_map`) plus, for the SM's stateless block whose
    dynamics jump lives in the downstream Integrator, the cross-block correction
    (`autodiff_rules._cross_block_saltation_correction`). (2) A *smooth guard
    surface* (T-NEW-sm-smooth-guard): the SM compiled ``x > 0.5`` to a boolean
    ``±1`` guard with ``∇g ≡ 0``; it now also derives a smooth residual
    ``x - 0.5`` (`grad_guard`) used only by the saltation paths, while the
    boolean still drives triggering / localization. AD now matches central FD.
    """
    a_val = 1.5
    g_ad = jax.grad(lambda amp: _amplitude_fwd(method, amp))(jnp.array(a_val))
    g_fd = fd_grad(lambda amp: _amplitude_fwd(method, amp), a_val, eps=1e-4)[0]
    np.testing.assert_allclose(g_ad, g_fd, rtol=5e-3)


def test_smooth_guard_residual_parser():
    """``_smooth_guard_residual_src`` derives a smooth residual only for a
    single ``< > <= >=`` comparison; everything else yields ``None`` so the SM
    falls back to its (non-differentiable) boolean guard (T-NEW-sm-smooth-guard).
    """
    from jaxonomy.dashboard.serialization.block_interface import (
        _smooth_guard_residual_src as r,
    )

    # Single comparisons -> residual (overall sign is irrelevant to dt_e/dp).
    assert r("x > 0.5") == "(x) - (0.5)"
    assert r("x < 1.5") == "(x) - (1.5)"
    assert r("speed >= v_max") == "(speed) - (v_max)"
    assert r("0.5 > x") == "(0.5) - (x)"
    # Not a single threshold crossing -> None (keep legacy behaviour).
    assert r("x == 2") is None
    assert r("x != 2") is None
    assert r("x > 0.5 and y < 1.0") is None
    assert r("0 < x < 1") is None  # chained comparison
    assert r("flag") is None
    assert r("not done") is None


@pytest.mark.slow
def test_sm_gradient_initial_state():
    """Gradient w.r.t. upstream Integrator initial condition (T-001c #1c).

    History: this gradient hit the same JAX-internal ``_cond_transpose``
    ``set(out_trees)`` crash as #1b (a weak-vs-strong float0 cotangent
    disagreement on the SM ``mode`` integer). It is now FIXED by
    ``_sever_discrete_cotangents`` and AD matches central FD — a value-path
    gradient (``x0`` shifts the integral by a constant), so unlike the
    pure-timing amplitude case it does not depend on the deferred event-time
    custom-VJP work.

    (Note: the historical incarnation also had a test bug — it passed a
    *dict* to ``ctx.with_continuous_state``, which iterates positionally and
    so set the integrator state to a ``system_id`` int. The correct call is
    the positional list ``ctx.with_continuous_state([x0])`` used below.)
    """
    diagram, src, sm, integ = _build_sine_to_sm_diagram()

    @jax.jit
    def fwd(x0, ctx):
        # Pass the per-subcontext list (positional), not a dict — see the
        # xfail reason above for why a dict was historically broken.
        ctx = ctx.with_continuous_state([x0])
        res = simulate(diagram, ctx, (0.0, 4.0), options=_opts("dopri5"))
        return res.context[integ.system_id].continuous_state.sum()

    ctx = diagram.create_context()
    g_ad = jax.grad(fwd)(jnp.array(0.0), ctx)
    g_fd = fd_grad(lambda x0: float(fwd(jnp.array(x0), ctx)), 0.0, eps=1e-4)[0]
    np.testing.assert_allclose(g_ad, g_fd, rtol=5e-3)


# ── Discrete-mode SM gradient through a non-constant action (passes) ────────


@pytest.mark.slow
def test_sm_discrete_mode_gradient_matches_fd():
    """Gradient through a discrete-mode SM with ``accelerate_with_jax=True``
    matches FD when the transition action depends on the SM input.

    History (T-001c-followup #2, 2026-04-27): the original incarnation of
    this test asserted "AD silently returns 0 while FD is non-zero" and was
    strict-xfailed.  Investigation showed the original setup had two
    independent bugs that each independently produced a zero AD reading, and
    the xfail rationale conflated them:

    1. The forward function called ``Sine(amplitude=float(amp), ...)``,
       which raises ``ConcretizationTypeError`` under ``jax.grad`` — the
       error was being swallowed by the bare ``xfail`` (no ``raises=``).
    2. The action body was ``y = 1.0`` — a literal constant.  With a
       constant action, the SM output ``y`` does not depend on ``amp`` at
       all; the only ``amp`` dependence is through the *timing* of the
       state transition, which on a fixed dt grid is locally flat (FD is
       genuinely 0 at ``amp=1.5``, ``eps=1e-4``).  So the "AD=0 vs FD!=0"
       claim was wrong: both were 0.

    With ``amp`` passed through to ``Sine`` directly (no ``float(...)``)
    AND an action that genuinely depends on the input (``y = x``), AD and
    FD agree to ~1e-7 — the discrete-mode SM autodiff path was correct
    all along.  No source change required.
    """
    smb = StateMachineBuilder()
    a = smb.add_state("a")
    b = smb.add_state("b")
    smb.set_initial_state(a)
    smb.add_transition(a, b, guard="x > 0.5", actions=["y = x"])
    sm_template = smb.build(name="disc_sm_template")
    sm = StateMachine(
        sm_data=sm_template._sm,
        inputs=list(sm_template._input_names),
        outputs=list(sm_template._output_names),
        dt=0.05,
        time_mode="discrete",
        name="disc_sm",
        accelerate_with_jax=True,
    )

    def fwd(amp):
        builder = DiagramBuilder()
        src = builder.add(Sine(amplitude=amp, frequency=0.5, name="sine"))
        smb_ = builder.add(sm)
        integ = builder.add(Integrator(initial_state=0.0, name="integ"))
        builder.connect(src.output_ports[0], smb_.input_ports[0])
        builder.connect(smb_.output_ports[0], integ.input_ports[0])
        diagram = builder.build(name="disc_sm_root")
        ctx = diagram.create_context()
        res = simulate(diagram, ctx, (0.0, 4.0), options=_opts("dopri5"))
        return res.context[integ.system_id].continuous_state.sum()

    g_ad = jax.grad(fwd)(jnp.array(1.5))
    g_fd = fd_grad(fwd, 1.5, eps=1e-4)[0]
    np.testing.assert_allclose(g_ad, g_fd, rtol=5e-3)


# ── #1d FIX: event-time gradient through a guard on an upstream source ───────


def _build_switcher_diagram(amp, method="dopri5"):
    """Sine(amp) -> Switcher.  The Switcher's guard reads the Sine output and
    the *dynamics jump* (dx/dt: 0 -> 1) is local to the Switcher, so the whole
    saltation is self-contained in one block.

    J = x(T) = T - t_e, and t_e (where ``amp·sin(ω t) = 0.5``) moves with
    ``amp``; the gradient w.r.t. ``amp`` is therefore purely an event-time
    (saltation) gradient flowing through a guard that depends on a parameter
    entering via an *upstream input signal*.
    """

    class Switcher(jaxonomy.LeafSystem):
        def __init__(self):
            super().__init__(name="switcher")
            self.declare_input_port(name="u")
            self.declare_default_mode(0)
            self.declare_continuous_state(
                default_value=jnp.array(0.0), ode=self._ode
            )

            def _guard(time, state, *inputs, **params):
                return jnp.asarray(inputs[0]).reshape(()) - 0.5

            self.declare_zero_crossing(
                guard=_guard,
                reset_map=None,
                start_mode=0,
                end_mode=1,
                direction="negative_then_non_negative",
                name="switch_0_1",
            )

        def _ode(self, time, state, *inputs, **params):
            # dx/dt = mode (0 in mode 0, 1 in mode 1)
            return jnp.asarray(state.mode, dtype=jnp.float64).reshape(())

    builder = DiagramBuilder()
    src = builder.add(Sine(amplitude=amp, frequency=0.5, name="sine"))
    sw = builder.add(Switcher())
    builder.connect(src.output_ports[0], sw.input_ports[0])
    diagram = builder.build(name="switcher_diag")
    ctx = diagram.create_context()
    res = simulate(diagram, ctx, (0.0, 4.0), options=_opts(method))
    return res.context[sw.system_id].continuous_state.sum()


@pytest.mark.slow
@pytest.mark.parametrize("method", SOLVERS)
def test_event_time_gradient_guard_on_upstream_source(method):
    """T-001c-followup #1d (FIXED 2026-05-20): the event-time (saltation)
    gradient propagates when the zero-crossing guard depends on a parameter
    that enters through an *upstream input signal* (here a ``Sine`` amplitude).

    Pre-fix the saltation rank-1 correction in
    ``leaf_system._wrap_reset_map`` only saw the guard's *local* arguments, so
    the denominator (``dg/dt`` along the trajectory) collapsed to zero — the
    guard's time/parameter dependence through the cached input port was
    invisible — and AD reported 0 where FD reports ~0.471.  The fix
    differentiates the guard against a *refreshed* port cache so ``∇g``
    carries the dependence on the upstream source, and applies the rank-1
    correction across the full root context.  This is the single-block
    discriminator the #1d spec used to isolate the gap.
    """
    a_val = 1.5
    g_ad = jax.grad(lambda amp: _build_switcher_diagram(amp, method))(jnp.array(a_val))
    g_fd = fd_grad(lambda amp: _build_switcher_diagram(amp, method), a_val, eps=1e-4)[0]
    np.testing.assert_allclose(g_ad, g_fd, rtol=5e-3)


def _build_xblock_diagram(amp, method="dopri5"):
    """Sine(amp) -> SmoothSwitcher (STATELESS) -> Integrator.

    The switching block has *no continuous state* (mode only) and a *smooth*
    guard ``u - 0.5``; its output ``y = mode`` (0 -> 1) drives the downstream
    Integrator, so the dynamics jump that the crossing-time moves lives in a
    DIFFERENT block than the event.  This is the cross-block saltation case the
    per-block reset adjoint cannot reach.
    """

    class SmoothSwitcher(jaxonomy.LeafSystem):
        def __init__(self):
            super().__init__(name="smooth_switcher")
            self.declare_input_port(name="u")
            self.declare_default_mode(0)

            def _out(time, state, *inputs, **params):
                return jnp.asarray(state.mode, dtype=jnp.float64).reshape(())

            self.declare_output_port(_out, name="y", requires_inputs=False)

            def _guard(time, state, *inputs, **params):
                return jnp.asarray(inputs[0]).reshape(()) - 0.5

            self.declare_zero_crossing(
                guard=_guard,
                reset_map=None,
                start_mode=0,
                end_mode=1,
                direction="negative_then_non_negative",
                name="sw01",
            )

    builder = DiagramBuilder()
    src = builder.add(Sine(amplitude=amp, frequency=0.5, name="sine"))
    sw = builder.add(SmoothSwitcher())
    integ = builder.add(Integrator(initial_state=0.0, name="integ"))
    builder.connect(src.output_ports[0], sw.input_ports[0])
    builder.connect(sw.output_ports[0], integ.input_ports[0])
    diagram = builder.build(name="xblock_diag")
    ctx = diagram.create_context()
    res = simulate(diagram, ctx, (0.0, 4.0), options=_opts(method))
    return res.context[integ.system_id].continuous_state.sum()


@pytest.mark.slow
@pytest.mark.parametrize("method", SOLVERS)
def test_event_time_gradient_cross_block_stateless(method):
    """T-001c-followup #1d cross-block (FIXED 2026-05-20): event-time gradient
    when the event fires in a block with NO continuous state and the dynamics
    jump it induces lives in a DOWNSTREAM block.

    The per-block saltation adjoint short-circuits for stateless blocks, so the
    correction is supplied at the simulator level
    (``autodiff_rules._cross_block_saltation_correction``), where the full
    continuous costate ``λ⁺`` is available: it forms the global event-time
    cotangent ``(f⁻ − f⁺)·λ⁺`` and redistributes it via the (smooth) guard's
    implicit-function theorem.  Requires a *smooth* guard — a boolean guard
    (e.g. the one ``StateMachine`` emits) has ∇g ≡ 0 and no recoverable
    event-time gradient.
    """
    a_val = 1.5
    g_ad = jax.grad(lambda amp: _build_xblock_diagram(amp, method))(jnp.array(a_val))
    g_fd = fd_grad(lambda amp: _build_xblock_diagram(amp, method), a_val, eps=1e-4)[0]
    np.testing.assert_allclose(g_ad, g_fd, rtol=5e-3)
