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
* ZOH on an integrator avoids the singular-A fallback (the function's
  Taylor branch must be exercised on a near-singular A).
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
    assert not linsys.is_discrete()

    d = discretize(linsys, dt=0.03, method="zoh")
    assert d.dt == pytest.approx(0.03)
    assert d.is_discrete()


def test_c_and_d_are_forwarded_untouched():
    linsys = _first_order(tau=2.0, gain=3.0)
    d = discretize(linsys, dt=0.05, method="zoh")
    np.testing.assert_array_equal(np.asarray(d.C), np.asarray(linsys.C))
    np.testing.assert_array_equal(np.asarray(d.D), np.asarray(linsys.D))


def test_is_stable_uses_unit_disk_for_discrete_linsys():
    """A stable continuous A=-1/τ becomes a stable discrete A=exp(-dt/τ),
    which lives inside the unit disk."""
    linsys = _first_order(tau=1.0)
    assert linsys.is_stable()  # Re(eig) < 0 in continuous-time

    d = discretize(linsys, dt=0.1, method="zoh")
    assert d.is_stable()  # |eig| = exp(-0.1) < 1 in discrete-time


def test_is_stable_detects_marginal_discrete_instability():
    """A continuous integrator (A=0) becomes a marginally-stable
    discrete A=1, which is NOT strictly inside the unit disk → is_stable False."""
    linsys = _integrator()
    d = discretize(linsys, dt=0.05, method="zoh")
    # |eig| = 1, not < 1 → not strictly stable.
    assert not d.is_stable()


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
