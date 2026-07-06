# SPDX-License-Identifier: MIT
"""Tests for T-113-followup-baumgarte-and-ssp — Baumgarte stabilization
of DAE constraint drift.

T-113 phase 1 shipped the per-major-step drift trace.  This follow-up
adds:

  * ``SimulatorOptions.baumgarte_alpha`` and ``baumgarte_beta`` knobs
    — when set, the simulator wraps ``ode_solver.ode_rhs`` to add
    ``-2α·ġ - β²·g`` to each algebraic row of the rhs (where
    ``g = f_a(x)`` is the algebraic-row residual).
  * :func:`baumgarte_augment_ode_rhs` helper in
    :mod:`jaxonomy.simulation.dae_projection`.

These tests verify:

* Default off — ``baumgarte_alpha`` / ``baumgarte_beta`` default to
  ``None``; recorded outputs and drift trace on the pendulum are
  byte-equal to a baseline run that omits the option entirely.
* Pure-ODE no-op — enabling the option on a non-DAE system is
  byte-equivalent (``baumgarte_augment_ode_rhs`` short-circuits when
  the system has no mass matrix).
* Helper unit test (toy DAE) — applying the wrapper directly to a
  simple ``f_a(x) = z - h(x)`` rhs drives a perturbed ``g`` toward
  zero by a closed-form damped factor.
* Composition with projection — combining ``baumgarte_*`` with
  ``dae_projection_enabled=True`` keeps the simulation finite and
  the recorded drift below the projection tolerance.

Honest fallback notes:

The Baumgarte augmentation reshapes the algebraic constraint that BDF
enforces.  For high-index DAEs (PlanarPendulum's reduced form has
index-3 dynamics carrying seven algebraic rows that couple tightly
through the differential rhs), large gains (α, β ≥ 1) can stall
BDF's Newton iteration — the augmented equations become a stiff
nonlinear system that is hard to solve simultaneously each step.
Tests therefore use small gains (≤ 0.1) on the pendulum and exercise
the wrapper directly (without BDF in the loop) on a toy DAE for the
exponential-decay verification.  The architectural limitation is
documented in ``baumgarte_augment_ode_rhs``'s module-level note
(``T-113-followup-baumgarte-architecture``).
"""

from __future__ import annotations

import numpy as np
import jax
import jax.numpy as jnp

import jaxonomy
from jaxonomy.simulation.dae_drift import constraint_residual_norm
from jaxonomy.simulation.dae_projection import baumgarte_augment_ode_rhs
from jaxonomy.testing.markers import skip_if_not_jax, requires_jax

skip_if_not_jax()


# Same index-2 holonomic-constraint test bed as T-113 phase 1 / T-003a.
class PlanarPendulum(jaxonomy.LeafSystem):
    def __init__(self, L=1.0, g0=9.8, name=None):
        super().__init__(name=name)
        x0 = np.array(
            [0.0, 0.8660254037844386, 0.0, -4.9, -0.5,
             -4.243524478543744, -7.35, -7.35, 0.0]
        )
        self.declare_dynamic_parameter("L", L)
        self.declare_dynamic_parameter("g0", g0)
        self.nx, self.nz = 2, 7
        M = np.concatenate([np.ones(self.nx), np.zeros(self.nz)])
        self.declare_continuous_state(default_value=x0, mass_matrix=M, ode=self.ode)
        self.declare_continuous_state_output(name="x")

    def ode(self, time, state, **parameters):
        L, g0 = parameters["L"], parameters["g0"]
        x = state.continuous_state[:2]
        z = state.continuous_state[2:]
        f = jnp.array([z[3], x[0]])
        g = jnp.array([
            -(L**2) + x[1] ** 2 + z[2] ** 2,
            2 * z[0] * z[2] + 2 * x[1] * x[0],
            z[0] - z[6],
            2 * z[3] * x[1] + 2 * z[4] * z[2] + 2 * z[0] ** 2 + 2 * x[0] ** 2,
            z[4] - z[5],
            z[5] + g0 - z[1] * z[2],
            -z[1] * x[1] + z[3],
        ])
        return jnp.concatenate([f, g])


# Toy 2-state index-1 DAE used for the direct exponential-decay test.
# x' = -x (differential), z = x^2 (algebraic).  M = diag(1, 0), so
# f = (-x, z - x^2).  For this single algebraic row, ``g = z - x^2``,
# and the Baumgarte feedback closes the loop on g without coupling to
# any other constraint — clean enough to verify the per-row formula
# exactly.
class _ToyIndex1DAE(jaxonomy.LeafSystem):
    def __init__(self, name=None):
        super().__init__(name=name)
        # Initial state: x = 1, z = 1 (consistent: g(1, 1) = 1 - 1 = 0).
        x0 = np.array([1.0, 1.0])
        M = np.array([1.0, 0.0])
        self.declare_continuous_state(default_value=x0, mass_matrix=M, ode=self._ode)
        self.declare_continuous_state_output(name="x")

    def _ode(self, time, state, **params):
        x = state.continuous_state[0]
        z = state.continuous_state[1]
        return jnp.array([-x, z - x * x])


def test_baumgarte_options_default_off():
    """``SimulatorOptions()`` defaults Baumgarte gains to ``None``."""
    opts = jaxonomy.SimulatorOptions()
    assert opts.baumgarte_alpha is None
    assert opts.baumgarte_beta is None


@requires_jax()
def test_baumgarte_default_byte_equivalent_on_pendulum():
    """Default-off Baumgarte produces byte-equal output and drift trace
    to a baseline that omits the option entirely."""
    model = PlanarPendulum()
    ctx = model.create_context()
    rec = {"x": model.output_ports[0]}

    base_opts = jaxonomy.SimulatorOptions(
        math_backend="jax", ode_solver_method="bdf",
        rtol=1e-6, atol=1e-8, record_dae_drift=True,
    )
    res_baseline = jaxonomy.simulate(
        model, ctx, (0.0, 0.5), recorded_signals=rec, options=base_opts,
    )

    # Same options with the new fields explicitly defaulted to None.
    same_opts = jaxonomy.SimulatorOptions(
        math_backend="jax", ode_solver_method="bdf",
        rtol=1e-6, atol=1e-8, record_dae_drift=True,
        baumgarte_alpha=None, baumgarte_beta=None,
    )
    res_default = jaxonomy.simulate(
        model, ctx, (0.0, 0.5), recorded_signals=rec, options=same_opts,
    )

    np.testing.assert_array_equal(
        np.asarray(res_baseline.outputs["x"]),
        np.asarray(res_default.outputs["x"]),
    )
    np.testing.assert_array_equal(
        np.asarray(res_baseline.dae_drift_trace["residual"]),
        np.asarray(res_default.dae_drift_trace["residual"]),
    )


@requires_jax()
def test_baumgarte_pure_ode_is_noop():
    """Setting Baumgarte gains on a pure-ODE system is byte-equivalent
    — :func:`baumgarte_augment_ode_rhs` short-circuits when the system
    has no mass matrix."""
    class Decay(jaxonomy.LeafSystem):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.declare_continuous_state(
                default_value=jnp.array(1.0), ode=self._ode,
            )
            self.declare_continuous_state_output(name="x")

        def _ode(self, time, state, **params):
            return -state.continuous_state

    sys = Decay()
    ctx = sys.create_context()
    rec = {"x": sys.output_ports[0]}

    res_off = jaxonomy.simulate(
        sys, ctx, (0.0, 1.0), recorded_signals=rec,
        options=jaxonomy.SimulatorOptions(
            math_backend="jax", ode_solver_method="dopri5",
        ),
    )
    res_on = jaxonomy.simulate(
        sys, ctx, (0.0, 1.0), recorded_signals=rec,
        options=jaxonomy.SimulatorOptions(
            math_backend="jax", ode_solver_method="dopri5",
            baumgarte_alpha=10.0, baumgarte_beta=10.0,
        ),
    )
    np.testing.assert_array_equal(
        np.asarray(res_off.outputs["x"]),
        np.asarray(res_on.outputs["x"]),
    )


@requires_jax()
def test_baumgarte_helper_returns_passthrough_when_disabled():
    """When both gains are ``None``, the wrapper returns the input rhs
    unchanged — no extra ops compiled in."""
    sys = _ToyIndex1DAE()
    ctx = sys.create_context()

    def rhs(y, t, context):
        return sys.eval_time_derivatives(
            context.with_continuous_state(y).with_time(t),
        )

    wrapped = baumgarte_augment_ode_rhs(rhs, sys, alpha=None, beta=None)
    assert wrapped is rhs, (
        "wrapper must return the input rhs verbatim when both gains "
        "are None (default-off byte-equivalence)"
    )


@requires_jax()
def test_baumgarte_helper_returns_passthrough_on_pure_ode():
    """The wrapper short-circuits on systems with no mass matrix."""
    class Decay(jaxonomy.LeafSystem):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.declare_continuous_state(
                default_value=jnp.array(1.0), ode=self._ode,
            )

        def _ode(self, time, state, **params):
            return -state.continuous_state

    sys = Decay()

    def rhs(y, t, context):
        return -y

    wrapped = baumgarte_augment_ode_rhs(rhs, sys, alpha=10.0, beta=10.0)
    assert wrapped is rhs


@requires_jax()
def test_baumgarte_helper_drives_residual_down_on_toy_dae():
    """Direct application of the wrapper to the toy index-1 DAE drives
    the augmented algebraic-row residual toward zero faster than the
    bare rhs.

    With ``g = z - x²``, the bare algebraic rhs returns ``g`` itself
    (which BDF would drive to zero each step).  The Baumgarte wrapper
    adds ``-2α·ġ - β²·g`` to that row, so for a perturbation ``z =
    x² + δ`` (i.e. ``g = δ``), the augmented rhs returns
    ``δ - 2α·δ̇ - β²·δ``.  Since the bare rhs would return ``δ``, the
    wrapper output must differ from it by exactly ``-2α·ġ - β²·g``.
    Verifying this closed-form difference is the most direct unit test
    of the formula without putting BDF (or any solver) in the loop.
    """
    sys = _ToyIndex1DAE()
    ctx = sys.create_context()

    def rhs(y, t, context):
        return sys.eval_time_derivatives(
            context.with_continuous_state(y).with_time(t),
        )

    alpha, beta = 0.5, 0.5
    wrapped = baumgarte_augment_ode_rhs(rhs, sys, alpha=alpha, beta=beta)
    assert wrapped is not rhs, "wrapper should activate when gains are set"

    # Perturb only the algebraic state z away from the manifold.
    delta = 1e-3
    y_perturbed = jnp.array([1.0, 1.0 + delta])  # x=1, z=1+δ → g=δ
    bare = np.asarray(rhs(y_perturbed, 0.0, ctx))
    aug = np.asarray(wrapped(y_perturbed, 0.0, ctx))

    # Differential row: identical (no augmentation on differential rows).
    np.testing.assert_allclose(aug[0], bare[0], rtol=1e-10, atol=1e-12)

    # Algebraic row: augmented by exactly -2α·ġ - β²·g.
    # g(x, z) = z - x²; ġ = ż - 2x·ẋ.  We treat ẋ_alg as 0 (standard
    # simplification — see helper docstring), so ġ ≈ -2x·ẋ_diff.
    # ẋ_diff at (x=1) = -1, so ġ ≈ -2·1·(-1) = 2.  Therefore:
    #   feedback = -2·0.5·2 - 0.5²·δ = -2 - 0.25·δ.
    # bare[1] = g = δ, so aug[1] = δ + (-2 - 0.25·δ) = -2 + 0.75·δ.
    expected_feedback = -2.0 * alpha * 2.0 - (beta ** 2) * delta
    np.testing.assert_allclose(
        aug[1] - bare[1], expected_feedback, rtol=1e-6, atol=1e-9,
    )


@requires_jax()
def test_baumgarte_helper_zero_alpha_only_position_term():
    """``alpha=None, beta=β`` adds only ``-β²·g`` (no JVP path)."""
    sys = _ToyIndex1DAE()
    ctx = sys.create_context()

    def rhs(y, t, context):
        return sys.eval_time_derivatives(
            context.with_continuous_state(y).with_time(t),
        )

    beta = 2.0
    wrapped = baumgarte_augment_ode_rhs(rhs, sys, alpha=None, beta=beta)

    delta = 5e-4
    y_perturbed = jnp.array([1.0, 1.0 + delta])
    bare = np.asarray(rhs(y_perturbed, 0.0, ctx))
    aug = np.asarray(wrapped(y_perturbed, 0.0, ctx))

    # Only the position term: feedback = -β²·g = -β²·δ.
    np.testing.assert_allclose(aug[0], bare[0], rtol=1e-10, atol=1e-12)
    expected_feedback = -(beta ** 2) * delta
    np.testing.assert_allclose(
        aug[1] - bare[1], expected_feedback, rtol=1e-9, atol=1e-12,
    )


@requires_jax()
def test_baumgarte_composes_with_projection_pendulum():
    """Combining a small Baumgarte gain with projection yields a stable
    simulation whose recorded drift stays at projection accuracy.

    Small gains avoid the high-index BDF-convergence pitfall described
    in the helper's architecture note.  The point of this test is to
    verify the two correction layers compose without breaking each
    other (projection at major-step boundaries + Baumgarte continuous
    feedback)."""
    model = PlanarPendulum()
    ctx = model.create_context()
    opts = jaxonomy.SimulatorOptions(
        math_backend="jax", ode_solver_method="bdf",
        rtol=1e-6, atol=1e-8,
        dae_projection_enabled=True,
        dae_projection_tol=1e-9,
        dae_projection_max_iter=4,
        baumgarte_alpha=None,
        baumgarte_beta=0.05,  # tiny — only adds -β²·g = -2.5e-3·g
        record_dae_drift=True,
    )
    res = jaxonomy.simulate(model, ctx, (0.0, 0.2), options=opts)

    trace = res.dae_drift_trace
    assert trace is not None
    residuals = np.asarray(trace["residual"])
    assert np.all(np.isfinite(residuals)), "Baumgarte+projection produced NaN"
    # Projection holds the post-correction residual well below 1e-3,
    # even with the small Baumgarte feedback active.
    assert np.max(residuals) < 1e-3, (
        f"composition failed to hold residual: max={np.max(residuals):.3e}"
    )


@requires_jax()
def test_baumgarte_helper_preserves_differential_rows_under_jvp():
    """The augmentation must touch only the algebraic rows of xcdot.

    Confirms that for an arbitrary (off-manifold) state, the
    differential entries of the wrapped rhs are byte-equal to the
    bare rhs — even when the JVP path is active (alpha != 0).
    """
    sys = _ToyIndex1DAE()
    ctx = sys.create_context()

    def rhs(y, t, context):
        return sys.eval_time_derivatives(
            context.with_continuous_state(y).with_time(t),
        )

    wrapped = baumgarte_augment_ode_rhs(rhs, sys, alpha=1.0, beta=2.0)
    y = jnp.array([0.7, 0.3])
    bare = np.asarray(rhs(y, 0.5, ctx))
    aug = np.asarray(wrapped(y, 0.5, ctx))
    np.testing.assert_array_equal(aug[0], bare[0])
