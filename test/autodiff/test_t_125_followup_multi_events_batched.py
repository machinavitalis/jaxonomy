# SPDX-License-Identifier: MIT
"""T-125-followup-multi-events-batched-vmap — cross product of multi-event
+ batched parameter sweep.

Verifies :func:`jaxonomy.vmap_event_times_gradient`, the helper that
combines :func:`event_times_gradient` (multi-firing, single sample) with
:func:`vmap_event_time_gradient` (single firing, multi-sample) into one
``{event_index: (N, n_firings, ...)}`` table.

Coverage:

* Bouncing ball with batched ``h0`` values ``[1, 2, 3]`` —
  ``vmap_event_times_gradient`` returns a ``(3, n_firings)`` gradient
  array per event index, matching the per-sample analytic and the
  per-(sample, firing) single-shot helper output.
* Multi-event diagram (two distinct guards) — each event index reports
  a ``(N, n_firings_idx)`` slab keyed under its own dict key.
* Composes with ``jax.jit`` — trace-time correctness.
* Python-loop fallback agrees with the default ``jax.vmap`` path.
* PyTree ``params_batch`` support (dict of batched leaves).
* ``event_indices`` selection — wrapper restricts to the requested
  subset.
* Empty firing list — events that never fired produce a structurally
  correct ``(N, 0)`` slab.
* Default-off byte-equivalence: existing T-125 followups still pass
  (verified by the surrounding test files); the wrapper raises a clear
  ``ValueError`` when ``results.event_times is None``.
* Public API export.
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import jaxonomy
from jaxonomy import (
    event_time_gradient,
    vmap_event_times_gradient,
)
from jaxonomy.simulation import SimulationResults
from jaxonomy.testing.markers import skip_if_not_jax

skip_if_not_jax()


# ── Shared helpers ──────────────────────────────────────────────────────

_G = 9.81
_H0_BATCH = jnp.asarray([1.0, 2.0, 3.0])


def _guard_floor(t, x, p):
    # Guard does not depend on the differentiation parameter.
    return x[0]


def _rhs_freefall(t, x, p):
    return jnp.stack([x[1], jnp.asarray(-_G)])


def _state_at_event_freefall(t_e, h0):
    """Trajectory state ``(y, v)`` at fixed time ``t_e`` parametrised by
    ``h0``.

    ``y(t; h0) = h0 - g t² / 2``, ``v(t; h0) = -g t``.  At fixed
    ``t_e`` the implicit-function chain rule sees ``∂y/∂h0 = 1``.
    """
    return jnp.stack([h0 - 0.5 * _G * (t_e ** 2), -_G * t_e])


def _state_at_event_h0_squared(t_e, h0):
    """Variant of the free-fall state where ``y(t; h0) = h0² - g t²/2``.

    Picked so the per-sample analytic gradient ``2 h0 / (g t_e)``
    actually depends on the sample axis (otherwise the implicit-function
    answer at fixed ``t_e`` collapses to the sample-independent value
    ``1 / (g t_e)`` and the per-sample contract is trivial).
    """
    return jnp.stack([h0 ** 2 - 0.5 * _G * (t_e ** 2), -_G * t_e])


# ── Test: shape contract + per-sample analytic match ────────────────────


def test_bouncing_ball_batched_shape_and_analytic():
    """``vmap_event_times_gradient`` returns ``(N, n_firings)`` per event
    index and matches the per-sample analytic ``2 h0 / (g t_e)``.
    """
    h0_batch = _H0_BATCH
    firings = np.asarray([0.3, 0.4], dtype=np.float64)
    results = SimulationResults(
        context=None,
        event_times={0: firings},
    )

    out = vmap_event_times_gradient(
        results,
        h0_batch,
        guards=_guard_floor,
        ode_rhs_fn=_rhs_freefall,
        state_at_event_fn=_state_at_event_h0_squared,
    )

    assert isinstance(out, dict) and set(out.keys()) == {0}
    grads = np.asarray(out[0])
    assert grads.shape == (3, 2), grads.shape

    # Analytic: dt_e/dh0 = 2 h0 / (g t_e).
    for i, h0 in enumerate([1.0, 2.0, 3.0]):
        for k, t_e in enumerate(firings):
            analytic = 2.0 * h0 / (_G * float(t_e))
            np.testing.assert_allclose(
                grads[i, k], analytic, rtol=1e-10, atol=1e-12,
                err_msg=f"sample {i}, firing {k} (t_e={t_e}, h0={h0})",
            )


def test_per_sample_per_firing_matches_single_shot():
    """Each ``(sample, firing)`` cell equals the single-shot
    :func:`event_time_gradient` invocation with that sample's parameter
    and that firing's recorded instant.
    """
    h0_batch = _H0_BATCH
    firings = np.asarray([0.3, 0.4], dtype=np.float64)
    results = SimulationResults(
        context=None,
        event_times={0: firings},
    )

    batch = vmap_event_times_gradient(
        results,
        h0_batch,
        guards=_guard_floor,
        ode_rhs_fn=_rhs_freefall,
        state_at_event_fn=_state_at_event_h0_squared,
    )
    grads = np.asarray(batch[0])

    for i in range(int(h0_batch.shape[0])):
        h0_i = h0_batch[i]
        for k, t_e in enumerate(firings):
            t_e_const = jax.lax.stop_gradient(jnp.asarray(t_e))

            def state_fn(p, _t=t_e_const):
                return _state_at_event_h0_squared(_t, p)

            single = event_time_gradient(
                _guard_floor, _rhs_freefall, jnp.asarray(t_e),
                state_fn, h0_i,
            )
            np.testing.assert_allclose(
                grads[i, k], float(single), rtol=1e-12, atol=0.0,
                err_msg=f"sample {i}, firing {k}: batch ≠ single-shot",
            )


# ── Test: multi-event diagram — distinct guards under distinct keys ─────


def test_multi_event_diagram_batched():
    """Two events with distinct guards across an ``N=3`` parameter
    batch.  Each event index gets its own ``(N, n_firings_idx)`` slab.
    """
    h0_batch = _H0_BATCH
    firings_e0 = np.asarray([0.3, 0.4], dtype=np.float64)  # n_firings = 2
    firings_e1 = np.asarray([0.5], dtype=np.float64)        # n_firings = 1

    results = SimulationResults(
        context=None,
        event_times={0: firings_e0, 1: firings_e1},
    )

    def guard_alt(t, x, p):
        # Distinct guard for event 1 — fires on the velocity coordinate.
        return x[1] + _G * 0.5  # zero at t = 0.5 in the model

    out = vmap_event_times_gradient(
        results,
        h0_batch,
        guards={0: _guard_floor, 1: guard_alt},
        ode_rhs_fn=_rhs_freefall,
        state_at_event_fn=_state_at_event_h0_squared,
    )

    assert set(out.keys()) == {0, 1}
    assert np.asarray(out[0]).shape == (3, 2)
    assert np.asarray(out[1]).shape == (3, 1)

    # Spot-check event 0's first cell against the analytic.
    np.testing.assert_allclose(
        np.asarray(out[0])[0, 0],
        2.0 * 1.0 / (_G * 0.3),
        rtol=1e-10,
    )


# ── Test: composes with jax.jit ─────────────────────────────────────────


def test_composes_with_jit():
    """The wrapper traces cleanly under ``jax.jit`` — same numbers."""
    h0_batch = _H0_BATCH
    firings = np.asarray([0.3, 0.4], dtype=np.float64)
    # Capture firings as a JAX value bound into the closure so the
    # results object is not retraced as a leaf.
    firings_j = jnp.asarray(firings)

    @jax.jit
    def go(h0_batch_):
        results_local = SimulationResults(
            context=None,
            event_times={0: firings_j},
        )
        out = vmap_event_times_gradient(
            results_local,
            h0_batch_,
            guards=_guard_floor,
            ode_rhs_fn=_rhs_freefall,
            state_at_event_fn=_state_at_event_h0_squared,
        )
        return out[0]

    eager = vmap_event_times_gradient(
        SimulationResults(context=None, event_times={0: firings}),
        h0_batch,
        guards=_guard_floor,
        ode_rhs_fn=_rhs_freefall,
        state_at_event_fn=_state_at_event_h0_squared,
    )[0]
    jitted = go(h0_batch)

    np.testing.assert_allclose(
        np.asarray(jitted), np.asarray(eager), rtol=1e-12, atol=0.0,
    )


# ── Test: Python-loop fallback agrees with default vmap path ────────────


def test_python_loop_fallback_matches_vmap_path():
    """``use_python_loop=True`` produces the same numbers as the default
    ``jax.vmap``-backed path (byte-identical within float64 tolerance).
    """
    h0_batch = _H0_BATCH
    firings = np.asarray([0.3, 0.4, 0.5], dtype=np.float64)
    results = SimulationResults(
        context=None,
        event_times={0: firings},
    )

    out_vmap = vmap_event_times_gradient(
        results, h0_batch,
        guards=_guard_floor, ode_rhs_fn=_rhs_freefall,
        state_at_event_fn=_state_at_event_h0_squared,
    )
    out_loop = vmap_event_times_gradient(
        results, h0_batch,
        guards=_guard_floor, ode_rhs_fn=_rhs_freefall,
        state_at_event_fn=_state_at_event_h0_squared,
        use_python_loop=True,
    )
    np.testing.assert_allclose(
        np.asarray(out_vmap[0]), np.asarray(out_loop[0]),
        rtol=1e-12, atol=0.0,
    )


# ── Test: PyTree params_batch ───────────────────────────────────────────


def test_pytree_params_batch():
    """``params_batch`` as a dict of batched leaves — output mirrors the
    input PyTree per event index.
    """
    h0_batch = _H0_BATCH
    firings = np.asarray([0.3, 0.4], dtype=np.float64)
    params_batch = {"h0": h0_batch}
    results = SimulationResults(
        context=None,
        event_times={0: firings},
    )

    def state_fn_dict(t_e, p):
        return _state_at_event_h0_squared(t_e, p["h0"])

    def guard_fn_dict(t, x, p):
        return x[0]

    def rhs_fn_dict(t, x, p):
        return jnp.stack([x[1], jnp.asarray(-_G)])

    out = vmap_event_times_gradient(
        results,
        params_batch,
        guards=guard_fn_dict,
        ode_rhs_fn=rhs_fn_dict,
        state_at_event_fn=state_fn_dict,
    )

    assert isinstance(out[0], dict) and set(out[0].keys()) == {"h0"}
    assert out[0]["h0"].shape == (3, 2)
    # Spot-check via analytic.
    for i, h0 in enumerate([1.0, 2.0, 3.0]):
        for k, t_e in enumerate(firings):
            np.testing.assert_allclose(
                float(out[0]["h0"][i, k]),
                2.0 * h0 / (_G * float(t_e)),
                rtol=1e-10,
            )


# ── Test: event_indices selection ───────────────────────────────────────


def test_event_indices_selects_subset():
    """When ``event_indices`` is given, only the requested keys appear in
    the output dict.
    """
    h0_batch = _H0_BATCH
    results = SimulationResults(
        context=None,
        event_times={
            0: np.asarray([0.3, 0.4], dtype=np.float64),
            1: np.asarray([0.5], dtype=np.float64),
        },
    )
    out = vmap_event_times_gradient(
        results,
        h0_batch,
        guards=_guard_floor,
        ode_rhs_fn=_rhs_freefall,
        state_at_event_fn=_state_at_event_h0_squared,
        event_indices=[1],
    )
    assert set(out.keys()) == {1}
    assert np.asarray(out[1]).shape == (3, 1)


def test_event_indices_unknown_raises():
    """Requesting an event index not in ``results.event_times`` raises."""
    h0_batch = _H0_BATCH
    results = SimulationResults(
        context=None,
        event_times={0: np.asarray([0.3], dtype=np.float64)},
    )
    with pytest.raises(KeyError, match="event index 7"):
        vmap_event_times_gradient(
            results,
            h0_batch,
            guards=_guard_floor,
            ode_rhs_fn=_rhs_freefall,
            state_at_event_fn=_state_at_event_h0_squared,
            event_indices=[7],
        )


# ── Test: empty firing list — structurally correct (N, 0) slab ──────────


def test_empty_firing_list_yields_empty_axis():
    """Events that never fired return a ``(N, 0)`` slab, not an error."""
    h0_batch = _H0_BATCH
    results = SimulationResults(
        context=None,
        event_times={
            0: np.asarray([], dtype=np.float64),
            1: np.asarray([0.3], dtype=np.float64),
        },
    )
    out = vmap_event_times_gradient(
        results,
        h0_batch,
        guards=_guard_floor,
        ode_rhs_fn=_rhs_freefall,
        state_at_event_fn=_state_at_event_h0_squared,
    )
    assert np.asarray(out[0]).shape == (3, 0), np.asarray(out[0]).shape
    assert np.asarray(out[1]).shape == (3, 1)


# ── Test: default-off byte-equivalence — clear error on missing capture ─


def test_missing_event_times_raises_clear_error():
    """Calling ``vmap_event_times_gradient`` on a ``results`` whose
    ``event_times is None`` raises ``ValueError`` with the remediation
    hint.
    """
    fake_results = SimulationResults(context=None, event_times=None)
    with pytest.raises(ValueError, match="record_event_times=True"):
        vmap_event_times_gradient(
            fake_results,
            _H0_BATCH,
            guards=_guard_floor,
            ode_rhs_fn=_rhs_freefall,
            state_at_event_fn=_state_at_event_h0_squared,
        )


# ── Test: per-event guards — missing index raises a clear error ─────────


def test_per_event_guards_missing_index_raises():
    """When ``guards`` is a dict, a missing key raises a clear error."""
    h0_batch = _H0_BATCH
    results = SimulationResults(
        context=None,
        event_times={
            0: np.asarray([0.3], dtype=np.float64),
            1: np.asarray([0.4], dtype=np.float64),
        },
    )
    with pytest.raises(KeyError, match="no guard supplied for event index 1"):
        vmap_event_times_gradient(
            results,
            h0_batch,
            guards={0: _guard_floor},  # missing 1
            ode_rhs_fn=_rhs_freefall,
            state_at_event_fn=_state_at_event_h0_squared,
        )


# ── Test: public-API surface ────────────────────────────────────────────


def test_public_api_exported():
    """``vmap_event_times_gradient`` is exposed at top level + via
    ``jaxonomy.simulation``.
    """
    import jaxonomy as _jx
    import jaxonomy.simulation as _sim

    assert hasattr(_jx, "vmap_event_times_gradient")
    assert hasattr(_sim, "vmap_event_times_gradient")
    assert _jx.vmap_event_times_gradient is _sim.vmap_event_times_gradient
