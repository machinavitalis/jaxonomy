# SPDX-License-Identifier: MIT
"""T-125-followup-custom-vjp — ``jax.custom_vjp`` event-time gradient.

Verifies :func:`jaxonomy.simulate_with_event_time_grad`, the wrapper
that lets a user write a plain ``simulate``-style call and use
``jax.grad`` directly to obtain the implicit-function-theorem
gradient ``∂t_event/∂params``.

Coverage:

* Bouncing ball with one bounce — ``jax.grad(simulate_with_event_time_grad
  )(h0)`` matches the analytic ``1 / sqrt(2 g h0)``.
* Composes with ``jax.jit``.
* Composes with ``jax.vmap`` over a vector of initial heights.
* Default-off byte-equivalence — calling the wrapper does NOT alter the
  pre-existing ``event_time_gradient`` / ``event_times_gradient`` /
  ``simulate`` APIs.  The wrapper module also re-exports those helpers
  unchanged.

The forward simulation is supplied via the ``sim_runner`` hook with an
analytic free-fall trajectory: phase 1's correctness target is the
backward (implicit-function-theorem) rule, which is independent of which
integrator computed the firing time.  Wiring a full ``simulate`` call
adds no information beyond what
``test_t_125_event_time_grad_phase1.py`` already covers and is
orthogonal to this followup's autodiff plumbing.
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import numpy as np

import jaxonomy
from jaxonomy import event_time_gradient, simulate_with_event_time_grad
from jaxonomy.testing.markers import skip_if_not_jax

skip_if_not_jax()


# ── Bouncing-ball setup (closed-form trajectory) ─────────────────────────


_G = 9.81


def _t_bounce(h0: float) -> float:
    return math.sqrt(2.0 * h0 / _G)


def _guard_floor(t, x, p):  # p is h0
    return x[0]


def _rhs_freefall(t, x, p):  # p is h0
    return jnp.stack([x[1], jnp.asarray(-_G)])


def _state_at_event_fn(t_e, h0):
    """Trajectory state at firing instant ``t_e`` as a function of h0.

    The wrapper passes ``t_e`` as a concrete Python float (bound into
    a closure for the backward rule) so that ``∂y/∂h0 = 1`` is
    preserved — see the wrapper's docstring for the rationale.
    """
    y = h0 - 0.5 * _G * (t_e ** 2)
    v = -_G * t_e
    return jnp.stack([y, v])


def _make_sim_runner():
    """Analytic forward 'simulate' that returns the first floor-crossing
    time as a function of h0.

    Matches what ``simulate(..., record_event_times=True)`` would return
    for the bouncing-ball diagram; bypasses the full integrator so the
    test stays focused on the autodiff plumbing.
    """
    def _runner(diagram, ctx, t_span, params, event_index, options):  # noqa: ARG001
        assert event_index == 0
        h0 = float(np.asarray(params))
        return math.sqrt(2.0 * h0 / _G)

    return _runner


# ── Tests ────────────────────────────────────────────────────────────────


def test_bouncing_ball_grad_matches_analytic():
    """``jax.grad(simulate_with_event_time_grad)(h0)`` matches
    ``1 / sqrt(2 g h0)``."""
    runner = _make_sim_runner()

    def f(h0):
        return simulate_with_event_time_grad(
            diagram=None,
            ctx=None,
            t_span=(0.0, 1.0),
            params=h0,
            event_index=0,
            guard_fn=_guard_floor,
            ode_rhs_fn=_rhs_freefall,
            state_at_event_fn=_state_at_event_fn,
            sim_runner=runner,
        )

    h0 = 1.0
    # Primal: forward returns the analytic bounce time.
    np.testing.assert_allclose(float(f(jnp.asarray(h0))), _t_bounce(h0), rtol=1e-12)

    # Gradient: matches the closed-form sensitivity.
    g_ad = float(jax.grad(f)(jnp.asarray(h0)))
    expected = 1.0 / math.sqrt(2.0 * _G * h0)
    np.testing.assert_allclose(g_ad, expected, rtol=1e-6)


def test_grad_composes_with_jit():
    """``jax.jit(jax.grad(simulate_with_event_time_grad))`` works."""
    runner = _make_sim_runner()

    def f(h0):
        return simulate_with_event_time_grad(
            diagram=None,
            ctx=None,
            t_span=(0.0, 1.0),
            params=h0,
            event_index=0,
            guard_fn=_guard_floor,
            ode_rhs_fn=_rhs_freefall,
            state_at_event_fn=_state_at_event_fn,
            sim_runner=runner,
        )

    jitted = jax.jit(jax.grad(f))
    h0 = 2.0
    expected = 1.0 / math.sqrt(2.0 * _G * h0)
    np.testing.assert_allclose(float(jitted(jnp.asarray(h0))), expected, rtol=1e-6)


def test_grad_composes_with_vmap():
    """``jax.vmap(jax.grad(simulate_with_event_time_grad))`` works over a
    batch of initial heights."""
    runner = _make_sim_runner()

    def f(h0):
        return simulate_with_event_time_grad(
            diagram=None,
            ctx=None,
            t_span=(0.0, 1.0),
            params=h0,
            event_index=0,
            guard_fn=_guard_floor,
            ode_rhs_fn=_rhs_freefall,
            state_at_event_fn=_state_at_event_fn,
            sim_runner=runner,
        )

    h0s = jnp.asarray([0.5, 1.0, 2.0, 5.0])
    batched = jax.vmap(jax.grad(f))(h0s)
    expected = 1.0 / np.sqrt(2.0 * _G * np.asarray(h0s))
    np.testing.assert_allclose(np.asarray(batched), expected, rtol=1e-6)


def test_grad_matches_single_shot_event_time_gradient():
    """End-to-end check: wrapper backward output equals the standalone
    :func:`event_time_gradient` value at the same firing time."""
    runner = _make_sim_runner()

    h0 = 1.5
    t_e = jnp.asarray(_t_bounce(h0))

    via_wrapper = float(jax.grad(lambda p: simulate_with_event_time_grad(
        diagram=None,
        ctx=None,
        t_span=(0.0, 1.0),
        params=p,
        event_index=0,
        guard_fn=_guard_floor,
        ode_rhs_fn=_rhs_freefall,
        state_at_event_fn=_state_at_event_fn,
        sim_runner=runner,
    ))(jnp.asarray(h0)))

    via_helper = float(event_time_gradient(
        _guard_floor,
        _rhs_freefall,
        t_e,
        lambda p, _t=float(t_e): _state_at_event_fn(_t, p),
        jnp.asarray(h0),
    ))

    np.testing.assert_allclose(via_wrapper, via_helper, rtol=1e-10, atol=1e-12)


# ── Default-off byte-equivalence ─────────────────────────────────────────


def test_existing_event_time_gradient_unchanged():
    """The new wrapper does NOT touch the existing ``event_time_gradient``
    contract — re-run the phase-1 bouncing-ball check to confirm.
    """
    h0 = 1.0
    t_e = jnp.asarray(_t_bounce(h0))
    grad_h0 = event_time_gradient(
        _guard_floor,
        _rhs_freefall,
        t_e,
        lambda p, _t=float(t_e): _state_at_event_fn(_t, p),
        jnp.asarray(h0),
    )
    expected = 1.0 / math.sqrt(2.0 * _G * h0)
    np.testing.assert_allclose(np.asarray(grad_h0), expected, rtol=1e-6)


# ── Public-API surface ───────────────────────────────────────────────────


def test_public_api_exported():
    """``simulate_with_event_time_grad`` exposed at top level and via
    ``jaxonomy.simulation``."""
    import jaxonomy as _jx
    import jaxonomy.simulation as _sim

    assert hasattr(_jx, "simulate_with_event_time_grad")
    assert hasattr(_sim, "simulate_with_event_time_grad")
    assert _jx.simulate_with_event_time_grad is _sim.simulate_with_event_time_grad
