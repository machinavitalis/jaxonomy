# SPDX-License-Identifier: MIT

"""T-109 phase 4 (LTI sub-piece) — discretize(linsys, dt, method=...).

Ships the matrix-level discretization helper promised by T-109 phase 4
at the LinearizedSystem level. The full diagram-level lift (walking a
Diagram tree and converting every continuous block to its discrete
equivalent) is deferred to a follow-up; this slice handles the
controller-design path where ``linearize → discretize → design`` is
the common downstream flow.

Tested:

* Validation (positive dt, method ∈ {zoh, euler}, refuses already-discrete input).
* ZOH on a scalar integrator (``dx/dt = u``) recovers the analytical
  ``x[k+1] = x[k] + dt·u[k]``.
* ZOH on a first-order plant matches the closed-form ``A_d = exp(-dt/τ)``.
* Euler matches its closed-form ``A_d = I + A·dt``, ``B_d = B·dt``.
* ZOH is exact for a singular-but-nonzero A (any plant carrying an
  integrator state): B_d stays finite and matches the augmented matrix
  exponential, and gradients through it stay finite.
* dt and is_discrete bookkeeping: continuous-in stays None, discrete-out
  carries the supplied dt, is_discrete flips, is_stable uses
  ``|eig(A)| < 1`` for the discrete case.
* C and D are forwarded untouched.
* Differentiable through dt + A via jax.grad on the resulting A_d.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaxonomy import discretize
from jaxonomy.library import LinearizedSystem


def _integrator() -> LinearizedSystem:
    """G(s) = 1/s — scalar integrator: A=0, B=1, C=1, D=0."""
    return LinearizedSystem(
        A=jnp.array([[0.0]]),
        B=jnp.array([[1.0]]),
        C=jnp.array([[1.0]]),
        D=jnp.array([[0.0]]),
        operating_point={"x": jnp.zeros(1), "u": jnp.zeros(1)},
    )


def _first_order(tau: float = 1.0, gain: float = 1.0) -> LinearizedSystem:
    """G(s) = gain / (τ s + 1)."""
    return LinearizedSystem(
        A=jnp.array([[-1.0 / tau]]),
        B=jnp.array([[1.0 / tau]]),
        C=jnp.array([[gain]]),
        D=jnp.array([[0.0]]),
        operating_point={"x": jnp.zeros(1), "u": jnp.zeros(1)},
    )


# ---------------------------------------------------------------------------
# Input validation.
# ---------------------------------------------------------------------------


def test_dt_must_be_positive():
    linsys = _first_order()
    with pytest.raises(ValueError, match="dt must be positive"):
        discretize(linsys, dt=0.0)
    with pytest.raises(ValueError, match="dt must be positive"):
        discretize(linsys, dt=-0.1)


def test_method_must_be_zoh_or_euler():
    linsys = _first_order()
    with pytest.raises(ValueError, match="unknown method"):
        discretize(linsys, dt=0.01, method="rk4")


def test_refuses_already_discrete_linsys():
    """Re-discretizing a discrete LinearizedSystem is not well-defined
    (you'd need to re-continuize first). The function must raise."""
    linsys = _first_order()
    discrete = discretize(linsys, dt=0.01, method="zoh")
    with pytest.raises(ValueError, match="already carries dt"):
        discretize(discrete, dt=0.01)


# ---------------------------------------------------------------------------
# Numerical correctness — ZOH.
# ---------------------------------------------------------------------------


def test_zoh_integrator_recovers_analytical_form():
    """For A=0, B=1: closed-form ZOH gives A_d = 1, B_d = dt."""
    linsys = _integrator()
    dt = 0.05
    d = discretize(linsys, dt=dt, method="zoh")

    np.testing.assert_allclose(np.asarray(d.A), [[1.0]], atol=1e-12)
    np.testing.assert_allclose(np.asarray(d.B), [[dt]], atol=1e-12)


def test_zoh_first_order_matches_closed_form():
    """For A = -1/τ, B = 1/τ: A_d = exp(-dt/τ), B_d = 1 - exp(-dt/τ)."""
    tau = 0.5
    dt = 0.1
    linsys = _first_order(tau=tau)
    d = discretize(linsys, dt=dt, method="zoh")

    expected_Ad = float(np.exp(-dt / tau))
    expected_Bd = 1.0 - expected_Ad  # B_d = (1/τ) · τ · (1 - exp(-dt/τ))
    np.testing.assert_allclose(float(d.A[0, 0]), expected_Ad, rtol=1e-12)
    np.testing.assert_allclose(float(d.B[0, 0]), expected_Bd, rtol=1e-10)


def _augmented_expm_reference(A, B, dt):
    """A_d, B_d from expm([[A, B], [0, 0]]·dt), computed in float64 NumPy.

    Independent of the implementation under test only in arithmetic, not
    in formula — the analytical assertions below are the formula check;
    this is the multi-state numerical reference.
    """
    from scipy.linalg import expm

    A = np.asarray(A, dtype=np.float64)
    B = np.asarray(B, dtype=np.float64)
    n, m = A.shape[0], B.shape[1]
    M = expm(np.block([[A, B], [np.zeros((m, n + m))]]) * dt)
    return M[:n, :n], M[:n, n:]


def _double_integrator() -> LinearizedSystem:
    """G(s) = 1/s² — A is singular but nonzero (norm(A) == 1)."""
    return LinearizedSystem(
        A=jnp.array([[0.0, 1.0], [0.0, 0.0]]),
        B=jnp.array([[0.0], [1.0]]),
        C=jnp.array([[1.0, 0.0]]),
        D=jnp.array([[0.0]]),
        operating_point={"x": jnp.zeros(2), "u": jnp.zeros(1)},
    )


def test_zoh_singular_nonzero_A_matches_closed_form():
    """Double integrator: A_d = [[1, dt], [0, 1]], B_d = [dt²/2, dt].

    A is singular yet norm(A) = 1, so the old norm-based guard picked the
    ``A⁻¹(A_d − I)B`` branch and B_d came back infinite.
    """
    dt = 0.05
    d = discretize(_double_integrator(), dt=dt, method="zoh")

    assert np.all(np.isfinite(np.asarray(d.B)))
    np.testing.assert_allclose(np.asarray(d.A), [[1.0, dt], [0.0, 1.0]], atol=1e-12)
    np.testing.assert_allclose(
        np.asarray(d.B), [[dt ** 2 / 2.0], [dt]], atol=1e-12
    )


def test_zoh_singular_A_gradients_stay_finite():
    """``jnp.where`` evaluates both branches, so the old fallback still
    poisoned gradients with NaN even when the finite branch was selected."""

    def loss(dt_val):
        d = discretize(_double_integrator(), dt=dt_val, method="zoh")
        return jnp.sum(d.B ** 2)

    dt = 0.05
    grad = float(jax.grad(loss)(jnp.asarray(dt)))
    # B_d = [dt²/2, dt] ⇒ ‖B_d‖² = dt⁴/4 + dt², d/dt = dt³ + 2dt.
    np.testing.assert_allclose(grad, dt ** 3 + 2.0 * dt, rtol=1e-6)


def test_zoh_matches_augmented_expm_on_a_real_singular_plant():
    """Regression for the reported failure: linearizing QubeServoModel at
    the upright equilibrium yields a singular A (its first column is all
    zeros), which used to produce ``B_d[0] == inf``."""
    from jaxonomy import library

    plant = library.QubeServoModel(full_state_output=False, name="qube")
    plant.input_ports[0].fix_value(np.zeros(1))
    ctx = plant.create_context().with_continuous_state(
        np.array([0.0, np.pi, 0.0, 0.0])
    )
    linsys = library.linearize(plant, ctx)

    # Precondition: this really is the singular-but-nonzero case.
    A = np.asarray(linsys.A, dtype=np.float64)
    assert np.linalg.matrix_rank(A) < A.shape[0]
    assert np.linalg.norm(A) > 1.0

    dt = 1e-3
    d = discretize(linsys, dt=dt, method="zoh")
    Ad_ref, Bd_ref = _augmented_expm_reference(A, np.asarray(linsys.B).reshape(4, 1), dt)

    assert np.all(np.isfinite(np.asarray(d.B)))
    np.testing.assert_allclose(np.asarray(d.A), Ad_ref, rtol=1e-6, atol=1e-9)
    np.testing.assert_allclose(
        np.asarray(d.B).reshape(4, 1), Bd_ref, rtol=1e-6, atol=1e-9
    )


def test_zoh_nonsingular_A_still_matches_the_inverse_formula():
    """Guard against a regression in the well-conditioned case the old
    solve branch handled correctly: B_d = A⁻¹(A_d − I)B."""
    A = np.array([[-2.0, 1.0], [0.5, -3.0]])
    B = np.array([[1.0], [0.0]])
    linsys = LinearizedSystem(
        A=jnp.asarray(A),
        B=jnp.asarray(B),
        C=jnp.array([[1.0, 0.0]]),
        D=jnp.array([[0.0]]),
        operating_point={"x": jnp.zeros(2), "u": jnp.zeros(1)},
    )
    dt = 0.1
    d = discretize(linsys, dt=dt, method="zoh")

    from scipy.linalg import expm

    Ad_ref = expm(A * dt)
    Bd_ref = np.linalg.solve(A, (Ad_ref - np.eye(2)) @ B)
    np.testing.assert_allclose(np.asarray(d.A), Ad_ref, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(np.asarray(d.B), Bd_ref, rtol=1e-8, atol=1e-12)


# ---------------------------------------------------------------------------
# Numerical correctness — Euler.
# ---------------------------------------------------------------------------


def test_euler_matches_closed_form():
    """Forward Euler: A_d = I + A·dt, B_d = B·dt — exact, no approximation."""
    linsys = _first_order(tau=0.5)
    dt = 0.1
    d = discretize(linsys, dt=dt, method="euler")

    expected_Ad = 1.0 + (-1.0 / 0.5) * dt    # A·dt = -0.2
    expected_Bd = (1.0 / 0.5) * dt           # B·dt =  0.2
    np.testing.assert_allclose(float(d.A[0, 0]), expected_Ad, rtol=1e-15)
    np.testing.assert_allclose(float(d.B[0, 0]), expected_Bd, rtol=1e-15)


# ---------------------------------------------------------------------------
# Bookkeeping: dt + is_discrete + is_stable.
# ---------------------------------------------------------------------------


def test_continuous_input_has_dt_none_discrete_output_carries_dt():
    linsys = _first_order()
    assert linsys.dt is None
    assert not linsys.is_discrete

    d = discretize(linsys, dt=0.03, method="zoh")
    assert d.dt == pytest.approx(0.03)
    assert d.is_discrete


def test_c_and_d_are_forwarded_untouched():
    linsys = _first_order(tau=2.0, gain=3.0)
    d = discretize(linsys, dt=0.05, method="zoh")
    np.testing.assert_array_equal(np.asarray(d.C), np.asarray(linsys.C))
    np.testing.assert_array_equal(np.asarray(d.D), np.asarray(linsys.D))


def test_is_stable_uses_unit_disk_for_discrete_linsys():
    """A stable continuous A=-1/τ becomes a stable discrete A=exp(-dt/τ),
    which lives inside the unit disk."""
    linsys = _first_order(tau=1.0)
    assert linsys.is_stable  # Re(eig) < 0 in continuous-time

    d = discretize(linsys, dt=0.1, method="zoh")
    assert d.is_stable  # |eig| = exp(-0.1) < 1 in discrete-time


def test_is_stable_detects_marginal_discrete_instability():
    """A continuous integrator (A=0) becomes a marginally-stable
    discrete A=1, which is NOT strictly inside the unit disk → is_stable False."""
    linsys = _integrator()
    d = discretize(linsys, dt=0.05, method="zoh")
    # |eig| = 1, not < 1 → not strictly stable.
    assert not d.is_stable


# ---------------------------------------------------------------------------
# Differentiability — composes with jax.grad.
# ---------------------------------------------------------------------------


def test_discretize_is_differentiable_through_dt():
    """For first-order plant, A_d = exp(-dt/τ). ∂A_d/∂dt = (-1/τ) exp(-dt/τ)."""
    tau = 0.5

    def loss(dt_val):
        linsys = LinearizedSystem(
            A=jnp.array([[-1.0 / tau]]),
            B=jnp.array([[1.0 / tau]]),
            C=jnp.array([[1.0]]),
            D=jnp.array([[0.0]]),
            operating_point={"x": jnp.zeros(1), "u": jnp.zeros(1)},
        )
        d = discretize(linsys, dt=dt_val, method="zoh")
        return jnp.sum(d.A ** 2)

    dt = 0.1
    grad = float(jax.grad(loss)(jnp.asarray(dt)))
    # d/dt of A_d^2 = 2 A_d * (-1/τ) A_d = -2/τ * A_d^2
    analytical = -2.0 / tau * np.exp(-dt / tau) ** 2
    np.testing.assert_allclose(grad, analytical, rtol=1e-6)


def test_discretize_top_level_export_resolves():
    """Sanity-check that `from jaxonomy import discretize` works."""
    import jaxonomy

    assert callable(jaxonomy.discretize)
    assert "discretize" in jaxonomy.__all__
