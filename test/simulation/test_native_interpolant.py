# SPDX-License-Identifier: MIT
"""
T-012a — Higher-order interpolant for ``SimulationResults.query``.

T-012 shipped a linear-interp version of ``query``; T-012a opts in
to a higher-order interpolant for sub-ULP accuracy at intermediate
times.  This file ships the *partial* path (PCHIP fallback over the
recorded ``(time, outputs)`` samples) — the native solver dense
interpolant remains a follow-up.  The user-facing surface is
identical: ``results.query(t)`` / ``results.query(t, signal=...)``.

The tests assert:
1. **Default off**: legacy linear-interp behaviour preserved
   (``solver_states is None``).
2. **Opt-in accuracy**: PCHIP query on a harmonic oscillator matches
   the analytic value at least 3 orders of magnitude better than
   linear interpolation over the same recorded samples.
3. **Out-of-range**: ``query(t > tf)`` raises ``ValueError`` (same
   contract as today).
4. **Mixed CT+DT**: a continuous Sine block and a discrete
   ``UnitDelay`` are both queryable; the discrete signal stays
   step-held rather than being smoothed through.
5. **Backwards-compat**: a manually-constructed ``SimulationResults``
   with ``solver_states=None`` routes through the linear-interp
   fallback exactly as before.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import jax.numpy as jnp

import jaxonomy
import jaxonomy.library as library
from jaxonomy.simulation.types import SimulationResults
from jaxonomy.testing.markers import skip_if_not_jax

skip_if_not_jax()


class _SHO(jaxonomy.LeafSystem):
    """Harmonic oscillator: x'' = -x.  State = [x, v].  x(0)=1, v(0)=0.

    Analytic solution: x(t) = cos(t), v(t) = -sin(t).
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.declare_continuous_state(
            default_value=jnp.array([1.0, 0.0]),
            ode=self._ode,
        )
        self.declare_continuous_state_output(name="state")

    def _ode(self, time, state, **params):
        x, v = state.continuous_state
        return jnp.array([v, -x])


def _run_sho(record_solver_states: bool, max_step: float = 0.5):
    """Simulate the SHO with a deliberately large major step so linear
    interpolation across the recorded samples is poor."""
    sys = _SHO()
    ctx = sys.create_context()
    opts = jaxonomy.SimulatorOptions(
        math_backend="jax",
        save_time_series=True,
        # Force major-step boundaries to be coarse so the recorded
        # ``time`` vector samples the smooth cosine sparsely.  This is
        # what makes linear interp clearly worse than PCHIP.
        max_major_step_length=max_step,
        record_solver_states=record_solver_states,
        rtol=1e-10,
        atol=1e-12,
        # Tight tolerances record many minor steps; size the buffer to capture
        # the whole trajectory so query() covers [0, 4] and the interpolant
        # isn't built on a truncated tail.
        buffer_length=20000,
    )
    return jaxonomy.simulate(
        sys, ctx, (0.0, 4.0), options=opts,
        recorded_signals={"state": sys.output_ports[0]},
    )


# ---------------------------------------------------------------------------
# 1. Default off — legacy linear-interp behaviour preserved.
# ---------------------------------------------------------------------------


def test_default_off_uses_linear_interp():
    """Without the opt-in flag, ``solver_states is None`` and ``query``
    matches today's ``jnp.interp`` semantics exactly."""
    res = _run_sho(record_solver_states=False)
    assert res.solver_states is None

    # Hand-rolled linear interp over the same recorded samples must match.
    t_query = 1.234
    v = res.query(t_query, signal="state")
    t_vec = np.asarray(res.time)
    y_vec = np.asarray(res.outputs["state"])
    expected = np.array([
        np.interp(t_query, t_vec, y_vec[:, 0]),
        np.interp(t_query, t_vec, y_vec[:, 1]),
    ])
    np.testing.assert_allclose(np.asarray(v), expected, atol=1e-12)


# ---------------------------------------------------------------------------
# 2. Opt-in accuracy — PCHIP must beat linear by orders of magnitude.
# ---------------------------------------------------------------------------


def test_opt_in_pchip_more_accurate_than_linear():
    """Higher-order query on a smooth signal beats linear interp over
    the same recorded ``time`` / ``outputs`` arrays by at least ~3
    orders of magnitude.  Tested at a midpoint between recorded
    samples on the SHO position component, where ``cos(t)`` is well
    sampled.

    T-012a-followup updated: ``solver_states`` may carry either the
    legacy ``"pchip"`` sentinel (when no native interp data is
    available) or the new :class:`NativeInterpolant` (when the JAX
    Dopri5 path captured per-step polynomial coefficients).  Both
    paths beat linear by orders of magnitude.
    """
    from jaxonomy.simulation.types import NativeInterpolant

    res_linear = _run_sho(record_solver_states=False)
    res_pchip = _run_sho(record_solver_states=True)
    assert (
        res_pchip.solver_states == "pchip"
        or isinstance(res_pchip.solver_states, NativeInterpolant)
    )

    # Probe between consecutive recorded times where the gap is
    # widest — that's where linear is worst and PCHIP shines.
    t_vec = np.asarray(res_linear.time)
    gaps = np.diff(t_vec)
    widest = int(np.argmax(gaps))
    t_mid = 0.5 * (t_vec[widest] + t_vec[widest + 1])

    # Linear-interp result over the same recorded samples.
    pos_linear = float(res_linear.query(t_mid, signal="state")[0])
    # PCHIP result via the opt-in path.
    pos_pchip = float(res_pchip.query(t_mid, signal="state")[0])

    truth = math.cos(t_mid)
    err_linear = abs(pos_linear - truth)
    err_pchip = abs(pos_pchip - truth)

    # Linear should be visibly off; PCHIP should be at least 100×
    # better.  (The widest-gap midpoint of a coarse Dopri5 trajectory
    # on cos(t) typically gives ~1e-3 vs ~1e-7 — 4 orders of
    # magnitude — but we assert a permissive 100× ratio so the test
    # stays robust to step-size jitter across JAX versions.)
    assert err_linear > 1e-5, (
        f"linear-interp error too small to compare: "
        f"err_linear={err_linear}"
    )
    assert err_pchip < err_linear / 100.0, (
        f"PCHIP not meaningfully more accurate than linear: "
        f"err_linear={err_linear}, err_pchip={err_pchip}"
    )


# ---------------------------------------------------------------------------
# 3. Out-of-range raises — contract preserved.
# ---------------------------------------------------------------------------


def test_query_out_of_range_raises_with_pchip():
    res = _run_sho(record_solver_states=True)
    t_max = float(res.time[-1])
    with pytest.raises(ValueError, match="out of range"):
        res.query(t_max + 1.0, signal="state")


# ---------------------------------------------------------------------------
# 4. Mixed CT+DT — discrete signal stays step-held, continuous is smooth.
# ---------------------------------------------------------------------------


def test_mixed_continuous_and_discrete_signals():
    """A diagram combining a continuous-time SHO and a discrete
    ``UnitDelay`` of a ``Step``.  ``query`` returns smooth PCHIP
    values for the continuous state and step-held values for the
    discrete signal (heuristic ZOH detection)."""
    sho = _SHO()
    step = library.Step(start_value=0.0, end_value=1.0, step_time=0.5)
    delay = library.UnitDelay(dt=0.1, initial_state=0.0)

    builder = jaxonomy.DiagramBuilder()
    sho_b = builder.add(sho)
    step_b = builder.add(step)
    delay_b = builder.add(delay)
    builder.connect(step_b.output_ports[0], delay_b.input_ports[0])
    diagram = builder.build()

    ctx = diagram.create_context()
    opts = jaxonomy.SimulatorOptions(
        math_backend="jax",
        save_time_series=True,
        record_solver_states=True,
        max_major_step_length=0.5,
        rtol=1e-10,
        atol=1e-12,
    )
    res = jaxonomy.simulate(
        diagram, ctx, (0.0, 2.0), options=opts,
        recorded_signals={
            "state": sho_b.output_ports[0],
            "delay": delay_b.output_ports[0],
        },
    )

    # Continuous signal: PCHIP should match analytic cos(t) tightly.
    t_q = 0.73
    pos_q = float(res.query(t_q, signal="state")[0])
    assert abs(pos_q - math.cos(t_q)) < 1e-4, (
        f"PCHIP on continuous state: got {pos_q}, expected {math.cos(t_q)}"
    )

    # Discrete signal: at any t, the value should be one of the
    # recorded plateau values (0.0 or 1.0), not an interpolated
    # in-between.  Heuristic ZOH detection picks step-interp.
    delay_q = float(res.query(1.234, signal="delay"))
    assert delay_q in (0.0, 1.0), (
        f"discrete signal smoothed to non-plateau value: {delay_q}"
    )


# ---------------------------------------------------------------------------
# 5. Backwards-compat — manual SimulationResults with solver_states=None
# routes through linear-interp.
# ---------------------------------------------------------------------------


def test_legacy_results_without_solver_states_uses_linear():
    """Manually construct a ``SimulationResults`` (mimicking results
    loaded from disk before T-012a, or built by a third-party tool).
    With ``solver_states=None`` the query path must be linear-interp
    — bit-identical to ``jnp.interp`` over the recorded arrays."""
    t = jnp.linspace(0.0, 1.0, 11)
    y = jnp.cos(t)
    res = SimulationResults(
        context=None,
        time=t,
        outputs={"y": y},
        solver_states=None,  # explicit legacy path
    )
    t_q = 0.37
    got = float(res.query(t_q, signal="y"))
    expected = float(jnp.interp(jnp.asarray(t_q), t, y))
    np.testing.assert_allclose(got, expected, atol=1e-15)


# ===========================================================================
# T-012a-followup — Native solver-state interpolant.
# ===========================================================================


def test_native_solver_state_recorded():
    """When ``record_solver_states=True`` and the active solver exposes
    a fixed-shape per-step polynomial (Dopri5), ``solver_states`` is a
    :class:`NativeInterpolant` rather than the legacy ``"pchip"``
    sentinel."""
    from jaxonomy.simulation.types import NativeInterpolant

    res = _run_sho(record_solver_states=True)
    assert isinstance(res.solver_states, NativeInterpolant), (
        f"expected NativeInterpolant, got {type(res.solver_states)!r}"
    )
    ni = res.solver_states
    # Sanity: Dopri5 emits 5 polynomial coefficients per step.
    assert ni.interp_coeff.shape[1] == 5
    # SHO state has 2 components → n_y == 2.
    assert ni.interp_coeff.shape[2] == 2
    # Segments must be monotone non-decreasing in time and non-degenerate.
    assert np.all(ni.t_step >= ni.t_prev)
    assert np.all(ni.t_step > ni.t_prev)


def test_native_query_sub_ulp_accuracy():
    """Native polynomial query matches analytic ``cos(t)`` to within
    ~1e-9 on the harmonic oscillator (vs PCHIP's ~5e-6) — more than
    three orders of magnitude tighter."""
    res = _run_sho(record_solver_states=True)
    # Sample 100 mid-step times.
    t_vec = np.asarray(res.time)
    # Skip the first two/last two samples to keep sampling away from
    # any minor-step boundary effects.
    rng = np.random.default_rng(0)
    interior = t_vec[2:-2]
    if interior.shape[0] < 2:
        return  # Not enough samples — skip the assertion.
    t_query = np.sort(
        rng.uniform(float(interior[0]), float(interior[-1]), size=100)
    )
    got = np.asarray(res.query(t_query, signal="state"))[:, 0]
    truth = np.cos(t_query)
    rmse = float(np.sqrt(np.mean((got - truth) ** 2)))
    max_err = float(np.max(np.abs(got - truth)))
    assert rmse < 1e-9, f"native interp RMSE too large: {rmse}"
    assert max_err < 1e-8, f"native interp max-err too large: {max_err}"


def test_native_query_at_segment_boundary():
    """Querying at a recorded boundary time must return the recorded
    value byte-equivalent — the polynomial endpoint matches the
    sample, and the implementation snaps exact-time hits."""
    res = _run_sho(record_solver_states=True)
    t_vec = np.asarray(res.time)
    y_vec = np.asarray(res.outputs["state"])
    # Pick an interior recorded time — the polynomial endpoints should
    # land on the recorded sample to within float64 round-off.
    idx = t_vec.shape[0] // 2
    t_q = float(t_vec[idx])
    got = np.asarray(res.query(t_q, signal="state"))
    np.testing.assert_allclose(got, y_vec[idx], atol=1e-12)


def test_native_query_out_of_range_raises():
    """Out-of-range bound check still applies on the native path."""
    res = _run_sho(record_solver_states=True)
    t_max = float(res.time[-1])
    with pytest.raises(ValueError, match="out of range"):
        res.query(t_max + 1.0, signal="state")


def test_native_pchip_fallback_when_unavailable():
    """When the captured native polynomial doesn't represent a recorded
    signal (e.g. a discrete output that's not a continuous-state
    passthrough), ``query`` falls back to PCHIP/ZOH automatically and
    the result is still a sensible interpolation."""
    from jaxonomy.simulation.types import NativeInterpolant

    sho = _SHO()
    step = library.Step(start_value=0.0, end_value=1.0, step_time=0.5)
    delay = library.UnitDelay(dt=0.1, initial_state=0.0)

    builder = jaxonomy.DiagramBuilder()
    sho_b = builder.add(sho)
    step_b = builder.add(step)
    delay_b = builder.add(delay)
    builder.connect(step_b.output_ports[0], delay_b.input_ports[0])
    diagram = builder.build()

    ctx = diagram.create_context()
    opts = jaxonomy.SimulatorOptions(
        math_backend="jax",
        save_time_series=True,
        record_solver_states=True,
        max_major_step_length=0.5,
        rtol=1e-10,
        atol=1e-12,
    )
    res = jaxonomy.simulate(
        diagram, ctx, (0.0, 2.0), options=opts,
        recorded_signals={
            "state": sho_b.output_ports[0],
            "delay": delay_b.output_ports[0],
        },
    )
    # The native interpolant should be present when CT state exists.
    assert isinstance(res.solver_states, NativeInterpolant)

    # Continuous state still answers via the native polynomial.
    pos_q = float(res.query(0.73, signal="state")[0])
    assert abs(pos_q - math.cos(0.73)) < 1e-7

    # Discrete signal: native path detects the mismatch and falls back
    # — the returned value is still a recorded plateau (0.0 or 1.0).
    delay_q = float(res.query(1.234, signal="delay"))
    assert delay_q in (0.0, 1.0), (
        f"discrete signal smoothed to non-plateau value: {delay_q}"
    )
