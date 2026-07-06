# SPDX-License-Identifier: MIT

"""T-109 followup-nyquist — Nyquist contour data helper.

Covers :func:`jaxonomy.library.nyquist_data`, which returns the real
and imaginary parts of ``G(jω)`` over a positive-frequency sweep
together with the reflected negative-frequency arrays for closing the
contour.  The returned arrays match the SISO-squeeze / MIMO-preserve
shape convention used by :func:`bode_data`.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaxonomy.library import (
    LinearizedSystem,
    frequency_response,
    nyquist_data,
)


# ---------------------------------------------------------------------------
# Fixture helpers (same plants as the pzmap / step-impulse tests).
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


def _first_order(tau=1.0, gain=1.0):
    """``G(s) = gain / (τs + 1)`` — A=-1/τ, B=1/τ, C=gain, D=0."""
    return LinearizedSystem(
        A=jnp.array([[-1.0 / tau]]),
        B=jnp.array([[1.0 / tau]]),
        C=jnp.array([[gain]]),
        D=jnp.array([[0.0]]),
        operating_point={"x": jnp.zeros(1), "u": jnp.zeros(1)},
    )


def _two_input_two_output():
    """Diagonal 2x2 MIMO: ``G(s) = diag(1/(s+1), 2/(s+2))``."""
    A = jnp.array([[-1.0, 0.0], [0.0, -2.0]])
    B = jnp.array([[1.0, 0.0], [0.0, 2.0]])
    C = jnp.eye(2)
    D = jnp.zeros((2, 2))
    return LinearizedSystem(
        A=A, B=B, C=C, D=D,
        operating_point={"x": jnp.zeros(2), "u": jnp.zeros(2)},
    )


# ---------------------------------------------------------------------------
# Shape + key correctness.
# ---------------------------------------------------------------------------


def test_nyquist_data_siso_shape_and_keys():
    linsys = _first_order()
    omegas = jnp.logspace(-2, 2, 25)
    data = nyquist_data(linsys, omegas)

    assert set(data.keys()) == {"omega", "real", "imag", "real_neg", "imag_neg"}
    assert data["omega"].shape == (25,)
    # SISO is squeezed to (K,), matching bode_data.
    assert data["real"].shape == (25,)
    assert data["imag"].shape == (25,)
    assert data["real_neg"].shape == (25,)
    assert data["imag_neg"].shape == (25,)


def test_nyquist_data_mimo_shape():
    linsys = _two_input_two_output()
    omegas = jnp.logspace(-1, 1, 16)
    data = nyquist_data(linsys, omegas)

    # MIMO preserves (K, p, m).
    assert data["real"].shape == (16, 2, 2)
    assert data["imag"].shape == (16, 2, 2)
    assert data["real_neg"].shape == (16, 2, 2)
    assert data["imag_neg"].shape == (16, 2, 2)


# ---------------------------------------------------------------------------
# Numeric correctness vs analytical Nyquist contours.
# ---------------------------------------------------------------------------


def test_nyquist_data_integrator_is_pure_imaginary():
    """``G(jω) = 1/(jω) = -j/ω`` — real part is exactly 0, imag is -1/ω."""
    linsys = _integrator()
    omegas = jnp.array([0.1, 1.0, 10.0])
    data = nyquist_data(linsys, omegas)

    np.testing.assert_allclose(np.asarray(data["real"]), 0.0, atol=1e-10)
    np.testing.assert_allclose(
        np.asarray(data["imag"]), -1.0 / np.asarray(omegas), rtol=1e-10
    )


def test_nyquist_data_first_order_traces_semicircle():
    """``G(jω) = 1/(jω + 1)`` traces a semicircle of radius 1/2 centred
    at (1/2, 0).  Verified by checking ``|G - 1/2| == 1/2``."""
    linsys = _first_order(tau=1.0, gain=1.0)
    omegas = jnp.logspace(-2, 2, 50)
    data = nyquist_data(linsys, omegas)

    real = np.asarray(data["real"])
    imag = np.asarray(data["imag"])
    radius = np.sqrt((real - 0.5) ** 2 + imag ** 2)
    np.testing.assert_allclose(radius, 0.5, atol=1e-6)


def test_nyquist_data_matches_frequency_response():
    """Real/imag of ``nyquist_data`` agree with ``frequency_response``."""
    linsys = _first_order(tau=0.5, gain=2.0)
    omegas = jnp.logspace(-1, 1, 12)

    fr = frequency_response(linsys, omegas)
    data = nyquist_data(linsys, omegas)

    expected_real = np.real(np.asarray(fr.response[..., 0, 0]))
    expected_imag = np.imag(np.asarray(fr.response[..., 0, 0]))
    np.testing.assert_allclose(np.asarray(data["real"]), expected_real, rtol=1e-10)
    np.testing.assert_allclose(np.asarray(data["imag"]), expected_imag, rtol=1e-10)


def test_nyquist_data_mirror_arrays_are_conjugate():
    """``G(-jω) = conj(G(jω))`` — i.e. ``real_neg == real`` and
    ``imag_neg == -imag``."""
    linsys = _first_order(tau=2.0, gain=3.0)
    omegas = jnp.logspace(-1, 1, 20)
    data = nyquist_data(linsys, omegas)

    np.testing.assert_array_equal(np.asarray(data["real_neg"]), np.asarray(data["real"]))
    np.testing.assert_array_equal(np.asarray(data["imag_neg"]), -np.asarray(data["imag"]))


# ---------------------------------------------------------------------------
# Differentiability.
# ---------------------------------------------------------------------------


def test_nyquist_data_differentiable_through_state_space():
    """``nyquist_data`` composes with ``jax.grad`` through ``A, B, C, D``.

    Use the sum of ``|G(jω)|²`` at a single frequency as the scalar loss
    and check the gradient w.r.t. the ``A`` matrix matches finite
    differences.  The plant is first-order, so the analytical gradient
    is tractable but we just compare to FD here.
    """
    omega0 = jnp.array([1.0])
    base_B = jnp.array([[1.0]])
    base_C = jnp.array([[1.0]])
    base_D = jnp.array([[0.0]])

    def loss(A_scalar):
        linsys = LinearizedSystem(
            A=jnp.array([[A_scalar]]),
            B=base_B,
            C=base_C,
            D=base_D,
            operating_point={"x": jnp.zeros(1), "u": jnp.zeros(1)},
        )
        data = nyquist_data(linsys, omega0)
        return jnp.sum(data["real"] ** 2 + data["imag"] ** 2)

    A0 = -2.5
    grad = float(jax.grad(loss)(jnp.asarray(A0)))

    eps = 1e-5
    fd = (float(loss(jnp.asarray(A0 + eps))) - float(loss(jnp.asarray(A0 - eps)))) / (2 * eps)
    np.testing.assert_allclose(grad, fd, rtol=1e-4, atol=1e-6)
