# SPDX-License-Identifier: MIT

"""T-109 phase 1: linearization workflow primitives.

Covers the new helpers in ``jaxonomy.library.linearization_workflow``:

* :func:`frequency_response` — analytic LTI Bode points.
* :func:`bode_data` — matplotlib-ready arrays (no plotting dependency).
* :func:`findop` — Newton-iteration operating-point solver.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaxonomy.library import (
    LTISystem,
    bode_data,
    findop,
    frequency_response,
)


# ---------------------------------------------------------------------------
# frequency_response — pure-LTI primitives
# ---------------------------------------------------------------------------


def _make_integrator():
    """Return a LinearizedSystem-shaped namespace for ``G(s) = 1/s``."""
    from jaxonomy.library import LinearizedSystem

    return LinearizedSystem(
        A=jnp.array([[0.0]]),
        B=jnp.array([[1.0]]),
        C=jnp.array([[1.0]]),
        D=jnp.array([[0.0]]),
        operating_point={"x": jnp.zeros(1), "u": jnp.zeros(1)},
    )


def test_frequency_response_integrator_at_unit_omega():
    """``G(s) = 1/s`` → ``|G(j1)| = 1`` and ``arg G(j1) = -π/2``."""
    linsys = _make_integrator()
    fr = frequency_response(linsys, jnp.array([1.0]))

    assert fr.response.shape == (1, 1, 1)
    assert np.isclose(float(fr.magnitudes[0, 0, 0]), 1.0, atol=1e-6)
    assert np.isclose(float(fr.phases[0, 0, 0]), -np.pi / 2.0, atol=1e-6)


def test_frequency_response_integrator_sweep():
    """Magnitude ``= 1/ω`` and phase ``= -π/2`` across a sweep."""
    linsys = _make_integrator()
    omegas = jnp.array([0.1, 1.0, 10.0, 100.0])
    fr = frequency_response(linsys, omegas)

    # |G(jω)| = 1/ω
    expected_mag = 1.0 / np.asarray(omegas)
    np.testing.assert_allclose(
        np.asarray(fr.magnitudes[:, 0, 0]), expected_mag, rtol=1e-5
    )
    # arg G(jω) = -π/2 everywhere
    np.testing.assert_allclose(
        np.asarray(fr.phases[:, 0, 0]), -np.pi / 2.0 * np.ones_like(expected_mag),
        atol=1e-6,
    )


def test_frequency_response_first_order_lowpass():
    """``G(s) = 1/(s+1)`` corner frequency: ``|G(j1)| = 1/√2``."""
    from jaxonomy.library import LinearizedSystem

    linsys = LinearizedSystem(
        A=jnp.array([[-1.0]]),
        B=jnp.array([[1.0]]),
        C=jnp.array([[1.0]]),
        D=jnp.array([[0.0]]),
        operating_point={"x": jnp.zeros(1), "u": jnp.zeros(1)},
    )
    fr = frequency_response(linsys, jnp.array([1.0]))
    assert np.isclose(
        float(fr.magnitudes[0, 0, 0]), 1.0 / np.sqrt(2.0), atol=1e-5
    )
    # Phase at corner is exactly -π/4
    assert np.isclose(float(fr.phases[0, 0, 0]), -np.pi / 4.0, atol=1e-5)


def test_bode_data_shapes_and_units():
    """``bode_data`` returns dB magnitudes and degree phases over the sweep."""
    linsys = _make_integrator()
    omegas = jnp.array([0.1, 1.0, 10.0])
    bd = bode_data(linsys, omegas)

    assert set(bd.keys()) == {"omega", "freq_hz", "magnitude_db", "phase_deg"}
    assert bd["omega"].shape == (3,)
    assert bd["freq_hz"].shape == (3,)
    # SISO arrays are squeezed to 1-D
    assert bd["magnitude_db"].shape == (3,)
    assert bd["phase_deg"].shape == (3,)
    # 1/ω → magnitude_db = -20*log10(ω)
    expected_db = -20.0 * np.log10(np.asarray(omegas))
    np.testing.assert_allclose(
        np.asarray(bd["magnitude_db"]), expected_db, atol=1e-5
    )
    # phase_deg ~ -90 (modulo unwrap)
    np.testing.assert_allclose(
        np.asarray(bd["phase_deg"]), -90.0 * np.ones(3), atol=1e-3
    )


def test_frequency_response_is_jit_traceable():
    """``frequency_response`` composes inside ``jax.jit``."""
    linsys = _make_integrator()

    @jax.jit
    def mag_at(omega):
        fr = frequency_response(linsys, jnp.atleast_1d(omega))
        return fr.magnitudes[0, 0, 0]

    assert np.isclose(float(mag_at(1.0)), 1.0, atol=1e-6)
    assert np.isclose(float(mag_at(10.0)), 0.1, atol=1e-6)


# ---------------------------------------------------------------------------
# findop — Newton operating-point solver on a damped harmonic oscillator
# ---------------------------------------------------------------------------


def _make_damped_oscillator(m=1.0, c=0.5, k=2.0):
    """``mẍ + cẋ + kx = u`` as a 2-state continuous LTI system.

    State: ``[x, ẋ]``.  Equilibrium under fixed ``u``: ``x* = u/k``, ``ẋ* = 0``.
    """
    A = jnp.array([[0.0, 1.0], [-k / m, -c / m]])
    B = jnp.array([[0.0], [1.0 / m]])
    C = jnp.array([[1.0, 0.0]])
    D = jnp.array([[0.0]])
    return LTISystem(A=A, B=B, C=C, D=D)


def test_findop_damped_oscillator_matches_analytic_equilibrium():
    """For ``u = 3``, ``k = 2``: equilibrium state is ``[1.5, 0]``."""
    k = 2.0
    sys = _make_damped_oscillator(m=1.0, c=0.5, k=k)

    u_val = 3.0
    sys.input_ports[0].fix_value(jnp.array([u_val]))
    base_ctx = sys.create_context()
    # Start far from the equilibrium
    base_ctx = base_ctx.with_continuous_state(jnp.array([0.0, 0.0]))

    op = findop(sys, base_ctx, tol=1e-10, max_iter=50)

    assert op.converged, f"findop did not converge: residual={op.residual_norm}"
    assert np.allclose(np.asarray(op.x), [u_val / k, 0.0], atol=1e-7)
    assert op.residual_norm < 1e-9
    # Linear residual should be solved in a single Newton step.
    assert op.iterations <= 2


def test_findop_returns_input_used():
    """The reported ``u`` matches the input held fixed during the search."""
    sys = _make_damped_oscillator()
    u_val = jnp.array([1.25])
    sys.input_ports[0].fix_value(u_val)
    base_ctx = sys.create_context()

    op = findop(sys, base_ctx)
    np.testing.assert_allclose(np.asarray(op.u), np.asarray(u_val))


def test_findop_residual_is_differentiable():
    """``jax.grad`` of ``‖ẋ(x, u₀)‖²`` w.r.t. the initial guess is non-zero
    away from equilibrium — the residual built inside ``findop`` is fully
    JAX-traceable.
    """
    sys = _make_damped_oscillator(k=2.0)
    sys.input_ports[0].fix_value(jnp.array([1.0]))
    base_ctx = sys.create_context()

    # Re-build the same residual ``findop`` uses so we can grad through it.
    from jaxonomy.library.linearization_workflow import _residual_fn

    residual, _ = _residual_fn(sys, base_ctx, sys.input_ports[0])

    def loss(x):
        r = residual(x)
        return jnp.sum(r * r)

    g = jax.grad(loss)(jnp.array([0.1, 0.2]))
    # Away from equilibrium the gradient must be finite and non-zero.
    assert jnp.all(jnp.isfinite(g))
    assert float(jnp.linalg.norm(g)) > 0.0


def test_findop_converged_flag_with_zero_iterations_when_at_equilibrium():
    """Starting from the equilibrium short-circuits the Newton loop."""
    k = 4.0
    sys = _make_damped_oscillator(m=1.0, c=0.1, k=k)
    u_val = 2.0
    sys.input_ports[0].fix_value(jnp.array([u_val]))
    base_ctx = sys.create_context()
    base_ctx = base_ctx.with_continuous_state(jnp.array([u_val / k, 0.0]))

    op = findop(sys, base_ctx, tol=1e-10)
    assert op.converged
    assert op.iterations == 0
    assert op.residual_norm < 1e-10


def _make_passive_state_system():
    """2-state LTI ``[v, psi]`` with one *passive* state (T-128).

    ``dv/dt = -v + u`` settles to ``v* = u``; ``dpsi/dt = u`` has an
    intrinsically nonzero equilibrium derivative (a free integrator, like a
    cornering vehicle's heading ``ψ̇ = r ≠ 0``).  The full-state Newton can't
    drive ``dpsi/dt`` to zero — ``psi`` has no influence on any residual — so
    it must be excluded via ``axis_mask``.
    """
    A = jnp.array([[-1.0, 0.0], [0.0, 0.0]])
    B = jnp.array([[1.0], [1.0]])
    C = jnp.eye(2)
    D = jnp.zeros((2, 1))
    return LTISystem(A=A, B=B, C=C, D=D)


def _passive_ctx():
    """Fresh passive-state system + context (one per findop call — findop
    currently leaves a fixed input port unfixed, so each call needs a fresh
    one; that re-entrancy quirk is tracked separately)."""
    sys = _make_passive_state_system()
    sys.input_ports[0].fix_value(jnp.array([1.0]))
    return sys, sys.create_context().with_continuous_state(jnp.array([0.0, 5.0]))


def test_findop_axis_mask_skips_passive_state():
    """``axis_mask`` solves the active state and holds the passive one."""
    # Full-state solve cannot zero the passive ``dpsi/dt = u = 1`` component.
    sys, ctx = _passive_ctx()
    op_full = findop(sys, ctx, tol=1e-8, max_iter=50)
    assert not op_full.converged
    assert op_full.residual_norm >= 1.0 - 1e-6  # dominated by dpsi/dt = 1

    # Masking out the passive 2nd state converges: v* = u = 1, psi held at 5.0.
    sys, ctx = _passive_ctx()
    op = findop(sys, ctx, axis_mask=[True, False], tol=1e-10, max_iter=50)
    assert op.converged, f"masked findop did not converge: {op.residual_norm}"
    assert np.isclose(float(op.x[0]), 1.0, atol=1e-7)
    assert np.isclose(float(op.x[1]), 5.0)  # passive state held at initial guess
    assert op.residual_norm < 1e-9

    # Integer-index form is equivalent to the boolean mask.
    sys, ctx = _passive_ctx()
    op_idx = findop(sys, ctx, axis_mask=[0], tol=1e-10, max_iter=50)
    assert op_idx.converged
    assert np.allclose(np.asarray(op_idx.x), np.asarray(op.x), atol=1e-9)


def test_findop_residual_scaling_auto_matches_unscaled_equilibrium():
    """``residual_scaling`` changes conditioning, not the equilibrium (T-128).

    An ill-scaled system (residuals spanning ~6 orders of magnitude) converges
    to the same operating point with ``"auto"`` scaling, and the reported
    ``residual_norm`` is in scaled units (order 1, not dominated by the
    large-magnitude component).
    """
    # dx1/dt = -1e3 (x1 - u),  dx2/dt = -1e-3 (x2 - u);  equilibrium = [u, u].
    def _ill_scaled():
        A = jnp.array([[-1e3, 0.0], [0.0, -1e-3]])
        B = jnp.array([[1e3], [1e-3]])
        sys = LTISystem(A=A, B=B, C=jnp.eye(2), D=jnp.zeros((2, 1)))
        sys.input_ports[0].fix_value(jnp.array([2.0]))
        return sys, sys.create_context().with_continuous_state(jnp.array([0.0, 0.0]))

    sys, base_ctx = _ill_scaled()
    op = findop(sys, base_ctx, residual_scaling="auto", tol=1e-6, max_iter=50)
    assert op.converged
    assert np.allclose(np.asarray(op.x), [2.0, 2.0], atol=1e-5)

    # Explicit per-component scaling array is also accepted.
    sys, base_ctx = _ill_scaled()
    op_arr = findop(
        sys, base_ctx, residual_scaling=jnp.array([1e-3, 1e3]), tol=1e-6
    )
    assert op_arr.converged
    assert np.allclose(np.asarray(op_arr.x), [2.0, 2.0], atol=1e-5)


def test_findop_residual_fn_override():
    """A caller-supplied ``residual_fn`` replaces the default ``ẋ`` residual."""
    sys = _make_damped_oscillator(m=1.0, c=0.5, k=2.0)
    sys.input_ports[0].fix_value(jnp.array([0.0]))
    base_ctx = sys.create_context().with_continuous_state(jnp.array([1.0, 1.0]))

    # Custom residual: drive the state to [3, -1] regardless of dynamics.
    target = jnp.array([3.0, -1.0])
    op = findop(
        sys, base_ctx, residual_fn=lambda x: x - target, tol=1e-12, max_iter=20
    )
    assert op.converged
    assert np.allclose(np.asarray(op.x), np.asarray(target), atol=1e-9)
