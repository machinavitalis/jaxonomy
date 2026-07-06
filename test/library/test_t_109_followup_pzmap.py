# SPDX-License-Identifier: MIT

"""T-109 followup-pzmap-step-impulse — pole-zero map + step/impulse helpers.

Covers the three classical-control analysis helpers added to
``jaxonomy.library.linearization_workflow``:

* :func:`pole_zero_map` — eigenvalues of ``A`` (poles) and SISO
  transmission zeros from the Rosenbrock pencil.
* :func:`step_response` — exact ``y(t) = C·∫₀ᵗ expm(A·s) ds·B + D``
  via the augmented-matrix ``expm`` trick.
* :func:`impulse_response` — exact ``y(t) = C·expm(A·t)·B`` (finite
  part; ``D·δ(t)`` is omitted from the returned samples).
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaxonomy.library import (
    LinearizedSystem,
    impulse_response,
    pole_zero_map,
    step_response,
)


# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------


def _integrator():
    """``G(s) = 1/s`` — A=0, B=1, C=1, D=0."""
    return LinearizedSystem(
        A=jnp.array([[0.0]]),
        B=jnp.array([[1.0]]),
        C=jnp.array([[1.0]]),
        D=jnp.array([[0.0]]),
        operating_point={"x": jnp.zeros(1), "u": jnp.zeros(1)},
    )


def _first_order():
    """``G(s) = 1/(s+1)`` — A=-1, B=1, C=1, D=0."""
    return LinearizedSystem(
        A=jnp.array([[-1.0]]),
        B=jnp.array([[1.0]]),
        C=jnp.array([[1.0]]),
        D=jnp.array([[0.0]]),
        operating_point={"x": jnp.zeros(1), "u": jnp.zeros(1)},
    )


def _oscillator(omega, zeta):
    """2-state mass-spring-damper: ``A = [[0, 1], [-ω², -2ζω]]``."""
    A = jnp.array([[0.0, 1.0], [-omega ** 2, -2.0 * zeta * omega]])
    B = jnp.array([[0.0], [1.0]])
    C = jnp.array([[1.0, 0.0]])
    D = jnp.array([[0.0]])
    return LinearizedSystem(
        A=A, B=B, C=C, D=D,
        operating_point={"x": jnp.zeros(2), "u": jnp.zeros(1)},
    )


# ---------------------------------------------------------------------------
# pole_zero_map
# ---------------------------------------------------------------------------


def test_pole_zero_map_underdamped_oscillator():
    """Poles of underdamped oscillator: ``-ζω ± jω√(1-ζ²)``."""
    omega, zeta = 2.0, 0.1
    linsys = _oscillator(omega, zeta)
    pz = pole_zero_map(linsys)
    poles = np.sort_complex(np.asarray(pz["poles"]))

    expected_real = -zeta * omega
    expected_imag = omega * np.sqrt(1.0 - zeta ** 2)
    expected = np.array([
        expected_real - 1j * expected_imag,
        expected_real + 1j * expected_imag,
    ])
    expected = np.sort_complex(expected)
    np.testing.assert_allclose(poles, expected, atol=1e-8)


def test_pole_zero_map_overdamped_oscillator():
    """Overdamped (``ζ > 1``) → two real poles ``-ζω ± ω√(ζ²-1)``."""
    omega, zeta = 2.0, 2.0
    linsys = _oscillator(omega, zeta)
    pz = pole_zero_map(linsys)
    poles = np.sort(np.asarray(pz["poles"]).real)

    discriminant = omega * np.sqrt(zeta ** 2 - 1.0)
    expected = np.sort(np.array([
        -zeta * omega - discriminant,
        -zeta * omega + discriminant,
    ]))
    np.testing.assert_allclose(poles, expected, atol=1e-8)


def test_pole_zero_map_integrator_pole():
    """Integrator: single pole at 0, gain 0 (strictly proper)."""
    pz = pole_zero_map(_integrator())
    np.testing.assert_allclose(np.asarray(pz["poles"]), np.array([0.0]), atol=1e-12)
    assert abs(pz["gain"]) < 1e-12


def test_pole_zero_map_first_order_pole():
    """First-order ``1/(s+1)``: single pole at ``-1``."""
    pz = pole_zero_map(_first_order())
    np.testing.assert_allclose(
        np.asarray(pz["poles"]).real, np.array([-1.0]), atol=1e-12
    )


def test_pole_zero_map_returns_dict_keys():
    """API contract: returns dict with keys ``poles``, ``zeros``, ``gain``."""
    pz = pole_zero_map(_first_order())
    assert set(pz.keys()) == {"poles", "zeros", "gain"}
    assert isinstance(pz["gain"], float)


def test_pole_zero_map_feedthrough_gain():
    """Pure feedthrough ``D = 2.5`` should be reported as ``gain``."""
    linsys = LinearizedSystem(
        A=jnp.array([[-1.0]]),
        B=jnp.array([[1.0]]),
        C=jnp.array([[1.0]]),
        D=jnp.array([[2.5]]),
        operating_point={"x": jnp.zeros(1), "u": jnp.zeros(1)},
    )
    pz = pole_zero_map(linsys)
    np.testing.assert_allclose(pz["gain"], 2.5, atol=1e-12)


# ---------------------------------------------------------------------------
# step_response
# ---------------------------------------------------------------------------


def test_step_response_integrator_equals_t():
    """Integrator step response ``y(t) = t``."""
    linsys = _integrator()
    t = jnp.linspace(0.0, 1.0, 11)
    y = step_response(linsys, t)
    # Shape (K, p=1, m=1).
    assert y.shape == (11, 1, 1)
    np.testing.assert_allclose(np.asarray(y[:, 0, 0]), np.asarray(t), atol=1e-8)


def test_step_response_first_order_closed_form():
    """``G(s) = 1/(s+1)`` step response: ``y(t) = 1 − exp(−t)``."""
    linsys = _first_order()
    t = jnp.linspace(0.0, 3.0, 13)
    y = step_response(linsys, t)
    expected = 1.0 - np.exp(-np.asarray(t))
    np.testing.assert_allclose(np.asarray(y[:, 0, 0]), expected, atol=1e-8)


def test_step_response_scalar_input_returns_pm_shape():
    """Scalar ``t_grid`` → returned shape is ``(p, m)``."""
    linsys = _first_order()
    y = step_response(linsys, 1.0)
    assert y.shape == (1, 1)
    np.testing.assert_allclose(float(y[0, 0]), 1.0 - np.exp(-1.0), atol=1e-8)


def test_step_response_at_zero_is_D():
    """At ``t = 0`` the step response equals ``D`` (here ``D = 0``)."""
    linsys = _first_order()
    y = step_response(linsys, jnp.array([0.0]))
    np.testing.assert_allclose(float(y[0, 0, 0]), 0.0, atol=1e-12)


def test_step_response_includes_feedthrough_D():
    """With ``D ≠ 0`` the step response should include the ``D`` constant."""
    linsys = LinearizedSystem(
        A=jnp.array([[-1.0]]),
        B=jnp.array([[1.0]]),
        C=jnp.array([[1.0]]),
        D=jnp.array([[0.5]]),
        operating_point={"x": jnp.zeros(1), "u": jnp.zeros(1)},
    )
    t = jnp.linspace(0.0, 2.0, 5)
    y = step_response(linsys, t)
    expected = 0.5 + (1.0 - np.exp(-np.asarray(t)))
    np.testing.assert_allclose(np.asarray(y[:, 0, 0]), expected, atol=1e-8)


# ---------------------------------------------------------------------------
# impulse_response
# ---------------------------------------------------------------------------


def test_impulse_response_integrator_is_constant():
    """Integrator impulse response = constant 1."""
    linsys = _integrator()
    t = jnp.linspace(0.0, 2.0, 7)
    h = impulse_response(linsys, t)
    assert h.shape == (7, 1, 1)
    np.testing.assert_allclose(np.asarray(h[:, 0, 0]), np.ones(7), atol=1e-8)


def test_impulse_response_first_order_closed_form():
    """``G(s) = 1/(s+1)`` impulse response: ``h(t) = exp(−t)``."""
    linsys = _first_order()
    t = jnp.linspace(0.0, 3.0, 13)
    h = impulse_response(linsys, t)
    expected = np.exp(-np.asarray(t))
    np.testing.assert_allclose(np.asarray(h[:, 0, 0]), expected, atol=1e-8)


def test_impulse_response_scalar_input_returns_pm_shape():
    """Scalar ``t`` → shape ``(p, m)``."""
    linsys = _first_order()
    h = impulse_response(linsys, 0.5)
    assert h.shape == (1, 1)
    np.testing.assert_allclose(float(h[0, 0]), np.exp(-0.5), atol=1e-8)


def test_impulse_response_oscillator_underdamped():
    """Underdamped 2nd-order: ``h(t) = (1/ωd) exp(-ζω t) sin(ωd t)``."""
    omega, zeta = 3.0, 0.2
    linsys = _oscillator(omega, zeta)
    t = jnp.linspace(0.0, 2.0, 21)
    h = impulse_response(linsys, t)
    omega_d = omega * np.sqrt(1.0 - zeta ** 2)
    t_np = np.asarray(t)
    expected = (1.0 / omega_d) * np.exp(-zeta * omega * t_np) * np.sin(omega_d * t_np)
    np.testing.assert_allclose(np.asarray(h[:, 0, 0]), expected, atol=1e-8)


# ---------------------------------------------------------------------------
# Differentiability
# ---------------------------------------------------------------------------


def test_step_response_grad_through_A_matrix_finite():
    """``jax.grad`` of an integrated step response through ``A`` is finite."""
    A0 = jnp.array([[-1.0]])
    B = jnp.array([[1.0]])
    C = jnp.array([[1.0]])
    D = jnp.array([[0.0]])
    t = jnp.linspace(0.0, 2.0, 21)

    def integrated_step(A):
        linsys = LinearizedSystem(
            A=A, B=B, C=C, D=D,
            operating_point={"x": jnp.zeros(1), "u": jnp.zeros(1)},
        )
        y = step_response(linsys, t)
        return jnp.sum(y)

    g = jax.grad(integrated_step)(A0)
    assert g.shape == A0.shape
    assert np.all(np.isfinite(np.asarray(g)))


def test_impulse_response_grad_through_B_matrix_finite():
    """``jax.grad`` of an integrated impulse response through ``B`` is finite."""
    A = jnp.array([[-1.0]])
    B0 = jnp.array([[1.0]])
    C = jnp.array([[1.0]])
    D = jnp.array([[0.0]])
    t = jnp.linspace(0.0, 2.0, 21)

    def integrated_impulse(B):
        linsys = LinearizedSystem(
            A=A, B=B, C=C, D=D,
            operating_point={"x": jnp.zeros(1), "u": jnp.zeros(1)},
        )
        h = impulse_response(linsys, t)
        return jnp.sum(h)

    g = jax.grad(integrated_impulse)(B0)
    assert g.shape == B0.shape
    assert np.all(np.isfinite(np.asarray(g)))


def test_default_float64_preserved_in_outputs():
    """T-005 default-float64 policy: returned arrays should be float64."""
    linsys = _first_order()
    t = jnp.linspace(0.0, 1.0, 5)
    y = step_response(linsys, t)
    h = impulse_response(linsys, t)
    # Both should be at least float64 (or higher) under the default policy.
    assert y.dtype in (jnp.float64, jnp.float32)
    assert h.dtype in (jnp.float64, jnp.float32)
    # If float64 is enabled (default), confirm it is honoured.
    import jax
    if jax.config.read("jax_enable_x64"):
        assert y.dtype == jnp.float64
        assert h.dtype == jnp.float64
