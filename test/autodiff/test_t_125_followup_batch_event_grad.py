# SPDX-License-Identifier: MIT
"""T-125-followup-batch-event-grad — vmap-batch event-time gradient.

Verifies that:

* ``jax.vmap`` composes cleanly with the existing
  :func:`jaxonomy.event_time_gradient` for batched parameter sweeps.
* :func:`jaxonomy.vmap_event_time_gradient` (the convenience wrapper)
  matches both the per-sample analytic gradient AND the
  Python-loop fallback variant byte-for-byte (within float64 tolerance).
* PyTree ``params_batch`` (dict of batched leaves) is preserved.
* The wrapper composes with downstream ``jax.grad`` of a scalar
  reduction over the batch axis.

Worked example: 1-D bouncing ball over a batch of ``N=8`` different
initial heights ``h0``.

    State:    (y, v)
    Dynamics: dy/dt = v, dv/dt = -g
    Initial:  y(0) = h0, v(0) = 0
    Guard:    g(t, x, p) = y      (fires when ball hits floor)
    First crossing: t_e = sqrt(2 h0 / g)
    Analytic:       ∂t_e/∂h0 = 1 / sqrt(2 g h0)
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import numpy as np

import jaxonomy
from jaxonomy import (
    event_time_gradient,
    vmap_event_time_gradient,
)
from jaxonomy.testing.markers import skip_if_not_jax

skip_if_not_jax()


_G = 9.81
_H0_BATCH = jnp.asarray([0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0])


# ── Shared helpers ───────────────────────────────────────────────────────


def _guard_fn(t, x, p):
    # Guard does not depend on the differentiation parameter.
    return x[0]


def _rhs_fn(t, x, p):
    # rhs uses the constant g — independent of the differentiation
    # parameter h0.
    return jnp.stack([x[1], jnp.asarray(-_G)])


def _state_at_event_fn(t_e, h0):
    """Trajectory state (y, v) at fixed time t_e parametrised by h0.

    y(t; h0) = h0 - g t² / 2 ; v(t; h0) = -g t.
    The implicit-function chain rule needs ``∂y/∂h0 |_{t fixed} = 1``,
    which is preserved when ``t_e`` is treated as a constant w.r.t. ``h0``.
    The wrapper handles the ``stop_gradient`` itself.
    """
    y = h0 - 0.5 * _G * (t_e ** 2)
    v = -_G * t_e
    return jnp.stack([y, jnp.asarray(v)])


def _analytic_dt_dh0(h0_arr):
    return 1.0 / np.sqrt(2.0 * _G * np.asarray(h0_arr))


def _t_event_array(h0_arr):
    return jnp.sqrt(2.0 * jnp.asarray(h0_arr) / _G)


# ── Tests: vmap_event_time_gradient ─────────────────────────────────────


def test_vmap_wrapper_bouncing_ball_batch():
    """``vmap_event_time_gradient`` matches the analytic per-sample
    ``1 / sqrt(2 g h0)`` for every element of an N=8 batch.
    """
    h0_batch = _H0_BATCH
    t_event_array = _t_event_array(h0_batch)

    grads = vmap_event_time_gradient(
        _guard_fn,
        _rhs_fn,
        t_event_array,
        _state_at_event_fn,
        h0_batch,
    )

    expected = _analytic_dt_dh0(h0_batch)
    np.testing.assert_allclose(np.asarray(grads), expected, rtol=1e-6)
    assert grads.shape == h0_batch.shape


def test_vmap_wrapper_python_loop_matches_vmap_path():
    """Python-loop fallback agrees with the default ``jax.vmap`` path."""
    h0_batch = _H0_BATCH
    t_event_array = _t_event_array(h0_batch)

    grads_vmap = vmap_event_time_gradient(
        _guard_fn,
        _rhs_fn,
        t_event_array,
        _state_at_event_fn,
        h0_batch,
    )
    grads_loop = vmap_event_time_gradient(
        _guard_fn,
        _rhs_fn,
        t_event_array,
        _state_at_event_fn,
        h0_batch,
        use_python_loop=True,
    )

    np.testing.assert_allclose(
        np.asarray(grads_vmap), np.asarray(grads_loop), rtol=1e-12, atol=0.0,
    )


def test_vmap_wrapper_per_sample_composability():
    """Each batch element is gradient-correct independent of the others.

    Compute the gradient at ``h0_i`` two ways:
      * via :func:`vmap_event_time_gradient` over a batch containing
        ``h0_i`` together with other samples.
      * via :func:`event_time_gradient` called independently on ``h0_i``
        alone.

    The two must agree element-wise — i.e. the wrapper introduces no
    cross-talk between batch members.
    """
    h0_batch = _H0_BATCH
    t_event_array = _t_event_array(h0_batch)

    batched = vmap_event_time_gradient(
        _guard_fn,
        _rhs_fn,
        t_event_array,
        _state_at_event_fn,
        h0_batch,
    )

    # Per-sample single-shot gradient.  ``state_fn`` must close over a
    # constant ``t_e`` (stop_gradient — same convention as the wrapper).
    per_sample = []
    for i in range(int(h0_batch.shape[0])):
        h0_i = h0_batch[i]
        t_e_i = t_event_array[i]
        t_e_const = jax.lax.stop_gradient(t_e_i)

        def state_fn(p, _t=t_e_const):
            return _state_at_event_fn(_t, p)

        g_i = event_time_gradient(
            _guard_fn, _rhs_fn, t_e_i, state_fn, h0_i,
        )
        per_sample.append(float(g_i))

    np.testing.assert_allclose(
        np.asarray(batched), np.asarray(per_sample), rtol=1e-12, atol=0.0,
    )


# ── Tests: jax.vmap of event_time_gradient directly ─────────────────────


def test_jax_vmap_of_event_time_gradient_directly():
    """``jax.vmap`` of :func:`event_time_gradient` works given the right
    closure discipline (no Python ``float(t_e)`` cast; ``stop_gradient``
    on the closed-over ``t_e``).  Same numerical result as the wrapper.
    """

    def single(h0):
        t_e = jnp.sqrt(2.0 * h0 / _G)
        t_e_const = jax.lax.stop_gradient(t_e)

        def state_fn(h0_):
            return _state_at_event_fn(t_e_const, h0_)

        return event_time_gradient(
            _guard_fn, _rhs_fn, t_e, state_fn, h0,
        )

    h0_batch = _H0_BATCH
    grads = jax.vmap(single)(h0_batch)

    expected = _analytic_dt_dh0(h0_batch)
    np.testing.assert_allclose(np.asarray(grads), expected, rtol=1e-6)


# ── Tests: PyTree parameter batch ───────────────────────────────────────


def test_vmap_wrapper_pytree_params_batch():
    """``params_batch`` as a dict of batched leaves — output is a dict
    with the same structure, leaves of shape ``(N,)``.
    """
    h0_batch = _H0_BATCH
    t_event_array = _t_event_array(h0_batch)
    params_batch = {"h0": h0_batch}

    def state_fn_dict(t_e, p):
        return _state_at_event_fn(t_e, p["h0"])

    def guard_fn_dict(t, x, p):
        return x[0]

    def rhs_fn_dict(t, x, p):
        return jnp.stack([x[1], jnp.asarray(-_G)])

    grads = vmap_event_time_gradient(
        guard_fn_dict,
        rhs_fn_dict,
        t_event_array,
        state_fn_dict,
        params_batch,
    )

    assert isinstance(grads, dict) and "h0" in grads
    assert grads["h0"].shape == h0_batch.shape
    np.testing.assert_allclose(
        np.asarray(grads["h0"]), _analytic_dt_dh0(h0_batch), rtol=1e-6,
    )


# ── Tests: downstream jax.grad composability ────────────────────────────


def test_vmap_wrapper_under_jax_grad():
    """A scalar reduction over the batched gradients composes with
    ``jax.grad`` without erroring.

    Cost ``C(h0_batch) = Σ (∂t_e/∂h0)_i² + ‖h0_batch‖²``.  Both
    ``t_event_array`` and the wrapper's internal closure on ``t_e`` are
    ``stop_gradient``'d, so JAX sees the first term as a constant w.r.t.
    ``h0_batch`` — the analytic gradient is exactly ``2 h0_i``.  This
    test verifies (a) the wrapper traces under ``jax.grad`` without
    erroring and (b) the well-defined component agrees with its
    closed-form analytic value.
    """

    def cost(h0_batch):
        t_event_array = _t_event_array(h0_batch)
        # Detach t_event_array from the batch parameter — the recorded
        # firing instants are inputs to the implicit-function rule, not
        # quantities to differentiate through.  Same convention as
        # :func:`simulate_with_event_time_grad` (T-125 phase 1).
        t_event_array = jax.lax.stop_gradient(t_event_array)

        grads = vmap_event_time_gradient(
            _guard_fn,
            _rhs_fn,
            t_event_array,
            _state_at_event_fn,
            h0_batch,
        )
        return jnp.sum(grads * grads) + jnp.sum(h0_batch * h0_batch)

    h0_batch = _H0_BATCH
    g_ad = jax.grad(cost)(h0_batch)
    assert jnp.all(jnp.isfinite(g_ad))
    # Analytic: ∂cost/∂h0_i = 2 h0_i (gradient term is stop_gradient'd
    # both in t_event_array and inside the wrapper).
    np.testing.assert_allclose(
        np.asarray(g_ad), 2.0 * np.asarray(h0_batch), rtol=1e-6,
    )


# ── Tests: shape contract & public API ──────────────────────────────────


def test_vmap_wrapper_n_equals_1_degenerate():
    """N=1 — the wrapper still produces a leading axis of length 1."""
    h0_batch = jnp.asarray([1.0])
    t_event_array = _t_event_array(h0_batch)

    grads = vmap_event_time_gradient(
        _guard_fn,
        _rhs_fn,
        t_event_array,
        _state_at_event_fn,
        h0_batch,
    )
    assert grads.shape == (1,)
    expected = _analytic_dt_dh0(h0_batch)
    np.testing.assert_allclose(np.asarray(grads), expected, rtol=1e-6)


def test_public_api_exported():
    """``vmap_event_time_gradient`` is exposed at top level and via
    ``jaxonomy.simulation``.
    """
    import jaxonomy
    import jaxonomy.simulation

    assert hasattr(jaxonomy, "vmap_event_time_gradient")
    assert hasattr(jaxonomy.simulation, "vmap_event_time_gradient")
    assert jaxonomy.vmap_event_time_gradient is jaxonomy.simulation.vmap_event_time_gradient
