# SPDX-License-Identifier: MIT

"""Regression tests for T-125-followup-multi-event-saltation (FIXED 2026-05-20).

The IFT-based event-time gradient
(:func:`jaxonomy.event_time_gradient`) reuses a user-supplied
``state_at_event_fn`` to compute ``∂R/∂p`` where
``R(p) = guard(t_e, state_fn(p), p)``.  The chain ``state_fn(p)`` is
supposed to give JAX the ``∂x_e/∂p`` term so the saltation gradient is
exact through the trajectory.

For a *single* event this works because users can write a closed-form
arc that captures the state-vs-parameter dependence analytically.  For
*multiple* events on the same arc — where each firing re-initialises the
trajectory from the previous reset map (e.g. a bouncing ball:
``v_after = -e * v_before``) — the closed-form-arc trick stops working:
``state_fn`` would need to compose every prior reset map analytically,
which the user typically cannot do in JAX-traceable form.  Pre-fix the
AD path reported ~zero where finite differences reported the true
cumulative sensitivity (``+0.903 s / grade`` on bounce #2 of the
restitution sweep below).

The fix (:func:`jaxonomy.multi_event_time_gradient` in
``jaxonomy/simulation/event_gradient.py``) tracks the forward
sensitivity ``S(t) = ∂x(t;p)/∂p`` along the recorded trajectory: it
integrates the variational equation ``Ṡ = f_x S + f_p`` within each arc
and applies the saltation jump
``S⁺ = R_x S⁻ + R_p + (R_t + R_x f⁻ − f⁺)(dt_e/dp)`` at every event, so
the implicit-function-theorem formula sees the *correctly propagated*
``S⁻(t_e)`` for firings past the first.

The repro is the simplest multi-bounce ball with a parameter (``e``, the
restitution coefficient) whose perturbation materially shifts every
bounce after the first.  ``e`` enters via the reset map
(``v_after = -e * v_before``), so the first-bounce time is insensitive to
it but every subsequent bounce time shifts linearly in ``e``.
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaxonomy import event_time_gradient, multi_event_time_gradient
from jaxonomy.testing.markers import skip_if_not_jax


pytestmark = pytest.mark.minimal
skip_if_not_jax()


# ---------------------------------------------------------------------
# Closed-form multi-bounce ball
# ---------------------------------------------------------------------


_G = 9.81
_H0 = 1.0


def _bounce_times(e: float, n: int) -> np.ndarray:
    """Analytic bounce times for a vertical drop from height ``_H0``
    with restitution ``e``.
    """
    g = _G
    t1 = math.sqrt(2.0 * _H0 / g)  # first impact
    out = [t1]
    v_post = math.sqrt(2.0 * g * _H0) * e
    for _ in range(n - 1):
        flight = 2.0 * v_post / g
        out.append(out[-1] + flight)
        v_post = v_post * e
    return np.asarray(out)


def _bounce_time_fd_dt_de(e: float, idx: int, h: float = 1e-4) -> float:
    """Central-difference dt_e/de for bounce ``idx``."""
    tp = _bounce_times(e + h, idx + 1)[idx]
    tm = _bounce_times(e - h, idx + 1)[idx]
    return (tp - tm) / (2 * h)


# ---------------------------------------------------------------------
# Hybrid-trajectory descriptors fed to multi_event_time_gradient
# ---------------------------------------------------------------------


def _guard_floor(_t, x, _p):
    return x[0]


def _rhs_freefall(_t, x, _p):
    return jnp.stack([x[1], jnp.asarray(-_G)])


def _reset_bounce(_t_e, x, e_param):
    """Restitution reset: height unchanged, velocity flips and scales by e."""
    return jnp.stack([x[0], -e_param * x[1]])


def _x0(_p):
    """Drop from rest at height _H0 (independent of the parameter)."""
    return jnp.stack([jnp.asarray(_H0), jnp.asarray(0.0)])


# ---------------------------------------------------------------------
# Single-shot wrapper used only by the (unbroken) first-bounce control
# ---------------------------------------------------------------------


def _state_at_event_fn_single(t_e):
    """First free-fall arc, curried at a fixed firing time."""

    def _f(_e_param):
        g = _G
        y = _H0 - 0.5 * g * (t_e ** 2)
        v = -g * t_e
        return jnp.stack([y, v])

    return _f


# ---------------------------------------------------------------------
# Fixed: dt_e/de for the *second* bounce now matches FD
# ---------------------------------------------------------------------


def test_event_time_gradient_second_bounce_de_matches_fd():
    """dt_e/de on bounce #2 matches FD via the forward-sensitivity helper.

    Pre-fix the single-shot IFT helper returned ~0 here because the
    user-supplied state callable could not carry ``∂x_e/∂p`` past the
    first reset map.  ``multi_event_time_gradient`` propagates the
    sensitivity through the bounce and recovers the true value.
    """
    e = 0.8
    bounces = _bounce_times(e, 3)

    grads = multi_event_time_gradient(
        _guard_floor,
        _rhs_freefall,
        _reset_bounce,
        _x0,
        jnp.asarray(bounces),
        jnp.asarray(e),
    )
    g_ad = float(np.asarray(grads)[1])  # second bounce
    g_fd = _bounce_time_fd_dt_de(e, idx=1)

    rel_err = abs(g_ad - g_fd) / (abs(g_fd) + 1e-12)
    assert rel_err < 0.05, (
        f"dt_e/de on bounce 2: AD={g_ad:.6f}, FD={g_fd:.6f}, "
        f"rel_err={rel_err:.4f}"
    )


# ---------------------------------------------------------------------
# Fixed: every recorded firing's dt_e/de matches FD
# ---------------------------------------------------------------------


def test_event_times_gradient_batch_de_matches_fd():
    """All recorded firings' dt_e/de match FD; pre-fix every firing past
    the first collapsed to ~0."""
    e = 0.8
    n_bounces = 3
    bounces = _bounce_times(e, n_bounces)

    grads = multi_event_time_gradient(
        _guard_floor,
        _rhs_freefall,
        _reset_bounce,
        _x0,
        jnp.asarray(bounces),
        jnp.asarray(e),
    )
    g_ad = np.asarray(grads)
    g_fd = np.asarray([_bounce_time_fd_dt_de(e, k) for k in range(n_bounces)])

    # The first bounce is identically zero (insensitive to e); compare it
    # on an absolute basis and the rest on a relative basis.
    assert abs(g_ad[0]) < 1e-6 and abs(g_fd[0]) < 1e-6
    rel_err = np.abs(g_ad[1:] - g_fd[1:]) / (np.abs(g_fd[1:]) + 1e-12)
    assert np.all(rel_err < 0.05), (
        f"dt_e/de across {n_bounces} bounces: AD={g_ad}, FD={g_fd}, "
        f"rel_err={rel_err}"
    )


# ---------------------------------------------------------------------
# PyTree-parameter coverage + jax.grad composability
# ---------------------------------------------------------------------


def test_multi_event_gradient_pytree_params_and_composability():
    """The helper handles a dict-of-params PyTree and is itself
    differentiable (``jax.grad`` of a scalar cost over the firings)."""
    e = 0.75
    bounces = _bounce_times(e, 3)

    def reset_dict(_t_e, x, p):
        return jnp.stack([x[0], -p["e"] * x[1]])

    grads = multi_event_time_gradient(
        _guard_floor,
        _rhs_freefall,
        reset_dict,
        _x0,
        jnp.asarray(bounces),
        {"e": jnp.asarray(e)},
    )
    assert set(grads.keys()) == {"e"}
    g_ad = np.asarray(grads["e"])
    g_fd = np.asarray([_bounce_time_fd_dt_de(e, k) for k in range(3)])
    assert np.all(
        np.abs(g_ad[1:] - g_fd[1:]) / (np.abs(g_fd[1:]) + 1e-12) < 0.05
    )

    def cost(ee):
        g = multi_event_time_gradient(
            _guard_floor, _rhs_freefall, _reset_bounce, _x0,
            jnp.asarray(bounces), ee,
        )
        return jnp.sum(g)

    assert np.isfinite(float(jax.grad(cost)(jnp.asarray(e))))


# ---------------------------------------------------------------------
# Control: single-event path still works (sanity check the fix scope)
# ---------------------------------------------------------------------


def test_single_event_dt_de_first_bounce_is_zero_correctly():
    """First-bounce dt_e/de must be exactly 0 (the first bounce is
    insensitive to ``e``) and both helpers must agree.

    This is the *control* — the single-event path was never broken and
    the fix must not regress it.
    """
    e = 0.8
    bounces = _bounce_times(e, 1)
    t_e_1 = jnp.asarray(bounces[0])

    g_ad_single = float(event_time_gradient(
        _guard_floor, _rhs_freefall, t_e_1,
        _state_at_event_fn_single(t_e_1), jnp.asarray(e),
    ))
    g_ad_multi = float(np.asarray(multi_event_time_gradient(
        _guard_floor, _rhs_freefall, _reset_bounce, _x0,
        bounces, jnp.asarray(e),
    ))[0])
    g_fd = _bounce_time_fd_dt_de(e, idx=0)

    assert abs(g_ad_single) < 1e-6, f"single-shot must be ~0; got {g_ad_single}"
    assert abs(g_ad_multi) < 1e-6, f"multi-event must be ~0; got {g_ad_multi}"
    assert abs(g_fd) < 1e-6, f"FD also confirms ~0; got {g_fd}"
