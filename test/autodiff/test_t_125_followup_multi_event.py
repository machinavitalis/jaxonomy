# SPDX-License-Identifier: MIT
"""T-125-followup-multi-event — batch event-time gradient API.

Verifies :func:`jaxonomy.event_times_gradient`, the one-shot wrapper
that takes a :class:`SimulationResults` populated with
``record_event_times=True`` and returns the per-firing implicit-
function-theorem gradient ``∂t_event/∂params`` for every recorded
firing.

Coverage:

* Bouncing ball with multiple bounces — wrapper output matches
  :func:`jaxonomy.event_time_gradient` called individually at each
  recorded firing instant.
* Multi-event diagram (two distinct guards) — each event index is
  reported under its own dict key with the correct per-firing
  shape.
* Default-off byte-equivalence — calling the wrapper on a
  ``results`` whose ``event_times is None`` raises a clear
  ``ValueError`` (the helper itself is opt-in: callers who don't
  use it pay zero cost).
* ``event_indices`` selection — the wrapper restricts to the
  requested subset.
* Empty firing list — events that never fired return a
  structurally-correct empty gradient.
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import jaxonomy
from jaxonomy import event_time_gradient, event_times_gradient
from jaxonomy.simulation import SimulationResults
from jaxonomy.testing.markers import skip_if_not_jax

skip_if_not_jax()


# ── Bouncing-ball helpers (closed-form trajectory) ───────────────────────
#
# Same closed-form model as ``test_t_125_event_time_grad_phase1.py``: a
# vertical drop with restitution ``e``.  Bounce times follow the
# geometric schedule ``t_k = t_1 * (1 + 2 e (1 - e^k) / (1 - e))`` for
# ``k >= 1``, where ``t_1 = sqrt(2 h0 / g)``.


_G = 9.81


def _bounce_times(h0: float, g: float, e: float, n: int) -> list[float]:
    t1 = math.sqrt(2.0 * h0 / g)
    out = [t1]
    v = math.sqrt(2.0 * g * h0) * e  # post-impact upward speed
    for _ in range(n - 1):
        flight = 2.0 * v / g
        out.append(out[-1] + flight)
        v = v * e
    return out


def _state_at_event_fn_freefall(t_e, params):
    """Trajectory state ``(y, v)`` at ``t_e`` for the free-fall arc
    starting at the immediately-prior bounce.

    For the ``∂t_e/∂h0`` test we only need a state callable that
    carries the explicit ``h0`` dependence at the recorded firing
    instant.  We use the *first-bounce* arc — exact for the first
    bounce, and for the multi-bounce check we still call the wrapper
    against the recorded ``t_e`` array; the per-firing match is
    confirmed by feeding identical state callables into the
    single-shot helper.
    """
    h0 = params
    g = _G
    y = h0 - 0.5 * g * (t_e ** 2)
    v = -g * t_e
    return jnp.stack([y, v])


def _guard_floor(t, x, p):
    return x[0]


def _rhs_freefall(t, x, p):
    return jnp.stack([x[1], jnp.asarray(-_G)])


# ── Test: bouncing-ball — multi-firing batch matches single-shot ─────────


def test_bouncing_ball_multi_firing_matches_single_shot():
    """Three bounces over a 5 s sim — each batch entry equals the
    single-shot ``event_time_gradient`` output for that firing time.
    """
    h0 = 1.0
    n_bounces = 3
    bounces = _bounce_times(h0, _G, e=0.6, n=n_bounces)

    fake_results = SimulationResults(
        context=None,
        event_times={0: np.asarray(bounces, dtype=np.float64)},
    )

    params = jnp.asarray(h0)
    batch = event_times_gradient(
        fake_results,
        params,
        guards=_guard_floor,
        ode_rhs_fn=_rhs_freefall,
        state_at_event_fn=_state_at_event_fn_freefall,
    )

    assert isinstance(batch, dict) and 0 in batch
    grads = np.asarray(batch[0])
    assert grads.shape == (n_bounces,), grads.shape

    # Per-firing equivalence with the single-shot helper.
    for k, t_e in enumerate(bounces):
        single = event_time_gradient(
            _guard_floor,
            _rhs_freefall,
            jnp.asarray(t_e),
            lambda p, _t=t_e: _state_at_event_fn_freefall(jnp.asarray(_t), p),
            params,
        )
        np.testing.assert_allclose(
            float(grads[k]), float(single), rtol=1e-10, atol=1e-12,
            err_msg=f"firing {k} (t_e={t_e}) batch ≠ single-shot",
        )


# ── Test: multi-event diagram — distinct guards under distinct keys ──────


def test_multi_event_diagram_distinct_guards():
    """Two events with distinct guards.

    Event 0 is the bouncing-ball floor crossing (guard ``g_0(x) =
    x[0]``); event 1 is a comparator-style crossing on a second
    state coordinate (guard ``g_1(x) = x[1] - threshold(p)``).
    Each event's gradient is reported under its own index key.
    """
    h0 = 1.0
    threshold = 0.5

    def state_fn(t_e, params):
        # 4-D synthetic state: two free-fall coordinates + two
        # placeholder slots so the comparator guard sees a state that
        # actually depends on params.
        h, thr = params["h0"], params["thr"]
        g = _G
        y = h - 0.5 * g * (t_e ** 2)
        v = -g * t_e
        # Coordinate 1 is a linear ramp through ``threshold`` at t = 1.0.
        u = thr + (t_e - 1.0)
        return jnp.stack([y, u, v, jnp.asarray(0.0)])

    def rhs_fn(t, x, p):
        return jnp.stack([
            x[2],
            jnp.asarray(1.0),
            jnp.asarray(-_G),
            jnp.asarray(0.0),
        ])

    def guard_floor(t, x, p):
        return x[0]

    def guard_threshold(t, x, p):
        return x[1] - p["thr"]

    # Synthetic firing times: event 0 fires twice, event 1 fires once.
    t1_floor = math.sqrt(2.0 * h0 / _G)
    fake_results = SimulationResults(
        context=None,
        event_times={
            0: np.asarray([t1_floor, t1_floor + 0.3], dtype=np.float64),
            1: np.asarray([1.0], dtype=np.float64),
        },
    )

    params = {"h0": jnp.asarray(h0), "thr": jnp.asarray(threshold)}
    batch = event_times_gradient(
        fake_results,
        params,
        guards={0: guard_floor, 1: guard_threshold},
        ode_rhs_fn=rhs_fn,
        state_at_event_fn=state_fn,
    )

    assert set(batch.keys()) == {0, 1}
    # Event 0: dict-of-arrays, each leaf has leading axis 2.
    for leaf in jax.tree_util.tree_leaves(batch[0]):
        assert leaf.shape[0] == 2
    # Event 1: leading axis 1.
    for leaf in jax.tree_util.tree_leaves(batch[1]):
        assert leaf.shape[0] == 1

    # Cross-check event 1's derivative w.r.t. ``thr`` equals the
    # single-shot helper.
    single = event_time_gradient(
        guard_threshold,
        rhs_fn,
        jnp.asarray(1.0),
        lambda p: state_fn(jnp.asarray(1.0), p),
        params,
    )
    np.testing.assert_allclose(
        np.asarray(batch[1]["thr"])[0], np.asarray(single["thr"]),
        rtol=1e-10, atol=1e-12,
    )


# ── Test: default-off byte-equivalence — clear error on missing capture ──


def test_missing_event_times_raises_clear_error():
    """Calling ``event_times_gradient`` on a ``results`` whose
    ``event_times is None`` raises ``ValueError`` with the
    remediation hint."""
    fake_results = SimulationResults(context=None, event_times=None)
    with pytest.raises(ValueError, match="record_event_times=True"):
        event_times_gradient(
            fake_results,
            jnp.asarray(1.0),
            guards=_guard_floor,
            ode_rhs_fn=_rhs_freefall,
            state_at_event_fn=_state_at_event_fn_freefall,
        )


# ── Test: ``event_indices`` selection ────────────────────────────────────


def test_event_indices_selects_subset():
    """When ``event_indices`` is given, only the requested keys appear
    in the output dict."""
    h0 = 1.0
    bounces = _bounce_times(h0, _G, e=0.6, n=2)
    fake_results = SimulationResults(
        context=None,
        event_times={
            0: np.asarray(bounces, dtype=np.float64),
            1: np.asarray([0.7], dtype=np.float64),
        },
    )

    out = event_times_gradient(
        fake_results,
        jnp.asarray(h0),
        guards=_guard_floor,
        ode_rhs_fn=_rhs_freefall,
        state_at_event_fn=_state_at_event_fn_freefall,
        event_indices=[1],
    )
    assert set(out.keys()) == {1}
    assert np.asarray(out[1]).shape == (1,)


def test_event_indices_unknown_raises():
    """Requesting an event index not in ``results.event_times`` raises."""
    fake_results = SimulationResults(
        context=None,
        event_times={0: np.asarray([0.5], dtype=np.float64)},
    )
    with pytest.raises(KeyError, match="event index 7"):
        event_times_gradient(
            fake_results,
            jnp.asarray(1.0),
            guards=_guard_floor,
            ode_rhs_fn=_rhs_freefall,
            state_at_event_fn=_state_at_event_fn_freefall,
            event_indices=[7],
        )


# ── Test: empty firing list — structurally-correct empty axis ────────────


def test_empty_firing_list_yields_empty_axis():
    """Events that never fired produce empty arrays, not errors."""
    fake_results = SimulationResults(
        context=None,
        event_times={
            0: np.asarray([], dtype=np.float64),
            1: np.asarray([0.5], dtype=np.float64),
        },
    )
    out = event_times_gradient(
        fake_results,
        jnp.asarray(1.0),
        guards=_guard_floor,
        ode_rhs_fn=_rhs_freefall,
        state_at_event_fn=_state_at_event_fn_freefall,
    )
    assert np.asarray(out[0]).shape == (0,), np.asarray(out[0]).shape
    assert np.asarray(out[1]).shape == (1,)


# ── Test: dict-mapping ``guards`` requires every used index ──────────────


def test_per_event_guards_missing_index_raises():
    """When ``guards`` is a dict, a missing key raises a clear error."""
    fake_results = SimulationResults(
        context=None,
        event_times={
            0: np.asarray([0.5], dtype=np.float64),
            1: np.asarray([0.7], dtype=np.float64),
        },
    )
    with pytest.raises(KeyError, match="no guard supplied for event index 1"):
        event_times_gradient(
            fake_results,
            jnp.asarray(1.0),
            guards={0: _guard_floor},  # missing 1
            ode_rhs_fn=_rhs_freefall,
            state_at_event_fn=_state_at_event_fn_freefall,
        )


# ── Test: public-API surface ─────────────────────────────────────────────


def test_public_api_exported():
    """``event_times_gradient`` exposed at top level + via .simulation."""
    import jaxonomy as _jx
    import jaxonomy.simulation as _sim

    assert hasattr(_jx, "event_times_gradient")
    assert hasattr(_sim, "event_times_gradient")
    assert _jx.event_times_gradient is _sim.event_times_gradient
