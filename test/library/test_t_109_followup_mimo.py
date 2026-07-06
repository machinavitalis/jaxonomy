# SPDX-License-Identifier: MIT

"""T-109 followup-mimo-frequency-response — MIMO Bode / frequency-response.

Covers the MIMO (vector → vector) behaviour of the linearization-workflow
helpers in ``jaxonomy.library.linearization_workflow``:

* :func:`frequency_response` evaluates the full ``(p, m)`` transfer-function
  matrix ``G(jω) = C (jωI − A)⁻¹ B + D`` and returns shape ``(K, p, m)``.
* :func:`bode_data` preserves the ``(K, p, m)`` shape for MIMO systems
  (one Bode pair per ``(output, input)`` channel) and unwraps phase along
  the frequency axis independently per channel.

The phase-1 SISO contract is preserved: ``frequency_response`` still
returns ``(K, 1, 1)`` for SISO and :func:`bode_data` still squeezes
to ``(K,)`` for SISO ergonomics (matplotlib-friendly 1-D arrays).
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaxonomy.library import (
    LinearizedSystem,
    bode_data,
    frequency_response,
)


# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------


def _decoupled_2x2(a1=-1.0, a2=-2.0):
    """Two independent first-order channels.

    ``A = diag(a1, a2)``, ``B = C = I₂``, ``D = 0``.  Each channel is
    a SISO low-pass ``G_ii(s) = 1 / (s − aᵢ)`` and ``G_ij(s) = 0`` for
    ``i ≠ j``.
    """
    return LinearizedSystem(
        A=jnp.array([[a1, 0.0], [0.0, a2]]),
        B=jnp.eye(2),
        C=jnp.eye(2),
        D=jnp.zeros((2, 2)),
        operating_point={"x": jnp.zeros(2), "u": jnp.zeros(2)},
    )


def _coupled_2x2():
    """Small off-diagonal coupling.  All four channels have non-zero G."""
    A = jnp.array([[-1.0, 0.1], [0.05, -2.0]])
    B = jnp.eye(2)
    C = jnp.eye(2)
    D = jnp.zeros((2, 2))
    return LinearizedSystem(
        A=A, B=B, C=C, D=D,
        operating_point={"x": jnp.zeros(2), "u": jnp.zeros(2)},
    )


def _siso_lowpass(a=-1.0):
    """``G(s) = 1/(s−a)`` as a 1-state SISO system."""
    return LinearizedSystem(
        A=jnp.array([[a]]),
        B=jnp.array([[1.0]]),
        C=jnp.array([[1.0]]),
        D=jnp.array([[0.0]]),
        operating_point={"x": jnp.zeros(1), "u": jnp.zeros(1)},
    )


# ---------------------------------------------------------------------------
# frequency_response — MIMO shape and value checks
# ---------------------------------------------------------------------------


def test_frequency_response_decoupled_2x2_shape_and_diagonal():
    """Decoupled diag(-1, -2): shape (K, 2, 2), off-diagonals zero, diags
    match per-axis SISO low-pass."""
    linsys = _decoupled_2x2(a1=-1.0, a2=-2.0)
    omegas = jnp.array([0.1, 1.0, 2.0, 10.0])
    fr = frequency_response(linsys, omegas)

    # MIMO shape contract.
    assert fr.response.shape == (4, 2, 2)
    assert fr.magnitudes.shape == (4, 2, 2)
    assert fr.phases.shape == (4, 2, 2)

    # Off-diagonals are exactly zero.
    np.testing.assert_allclose(np.asarray(fr.response[:, 0, 1]), 0.0, atol=1e-12)
    np.testing.assert_allclose(np.asarray(fr.response[:, 1, 0]), 0.0, atol=1e-12)

    # Diagonal channels match per-axis SISO low-pass ``1/(jω − a)``.
    om_np = np.asarray(omegas)
    expected_00 = 1.0 / (1j * om_np - (-1.0))
    expected_11 = 1.0 / (1j * om_np - (-2.0))
    np.testing.assert_allclose(
        np.asarray(fr.response[:, 0, 0]), expected_00, rtol=1e-6
    )
    np.testing.assert_allclose(
        np.asarray(fr.response[:, 1, 1]), expected_11, rtol=1e-6
    )


def test_frequency_response_decoupled_matches_siso_per_axis():
    """Diagonal of MIMO frequency_response equals SISO frequency_response of
    each isolated channel."""
    linsys_mimo = _decoupled_2x2(a1=-1.0, a2=-2.0)
    linsys_siso_0 = _siso_lowpass(a=-1.0)
    linsys_siso_1 = _siso_lowpass(a=-2.0)
    omegas = jnp.array([0.1, 1.0, 5.0])

    fr_m = frequency_response(linsys_mimo, omegas)
    fr_0 = frequency_response(linsys_siso_0, omegas)
    fr_1 = frequency_response(linsys_siso_1, omegas)

    np.testing.assert_allclose(
        np.asarray(fr_m.response[:, 0, 0]),
        np.asarray(fr_0.response[:, 0, 0]),
        rtol=1e-7,
    )
    np.testing.assert_allclose(
        np.asarray(fr_m.response[:, 1, 1]),
        np.asarray(fr_1.response[:, 0, 0]),
        rtol=1e-7,
    )


def test_frequency_response_coupled_2x2_structure():
    """Coupled system: all four channels non-zero, and the analytic matrix
    inverse ``(jωI − A)⁻¹`` agrees with our solver-based implementation."""
    linsys = _coupled_2x2()
    omegas = jnp.array([0.5, 2.0, 4.0])
    fr = frequency_response(linsys, omegas)

    assert fr.response.shape == (3, 2, 2)

    # Recompute G(jω) = (jωI − A)⁻¹ analytically (B = C = I, D = 0).
    A = np.asarray(linsys.A)
    for k, om in enumerate(np.asarray(omegas)):
        M = 1j * om * np.eye(2) - A
        expected = np.linalg.inv(M)
        np.testing.assert_allclose(
            np.asarray(fr.response[k]), expected, rtol=1e-6, atol=1e-10
        )

    # And the off-diagonals must be small (proportional to coupling) but
    # non-zero — distinguishes coupled from decoupled.
    assert float(jnp.max(jnp.abs(fr.response[:, 0, 1]))) > 0.0
    assert float(jnp.max(jnp.abs(fr.response[:, 1, 0]))) > 0.0


def test_frequency_response_2_input_1_output():
    """Rectangular MIMO: 1 output, 2 inputs → shape (K, 1, 2)."""
    linsys = LinearizedSystem(
        A=jnp.array([[-1.0, 0.0], [0.0, -2.0]]),
        B=jnp.eye(2),                 # (2, 2)
        C=jnp.array([[1.0, 1.0]]),    # (1, 2): y = x1 + x2
        D=jnp.zeros((1, 2)),
        operating_point={"x": jnp.zeros(2), "u": jnp.zeros(2)},
    )
    omegas = jnp.array([0.5, 1.0, 3.0])
    fr = frequency_response(linsys, omegas)
    assert fr.response.shape == (3, 1, 2)

    om_np = np.asarray(omegas)
    # y/u1 = 1/(jω+1), y/u2 = 1/(jω+2)
    expected_in1 = 1.0 / (1j * om_np + 1.0)
    expected_in2 = 1.0 / (1j * om_np + 2.0)
    np.testing.assert_allclose(
        np.asarray(fr.response[:, 0, 0]), expected_in1, rtol=1e-6
    )
    np.testing.assert_allclose(
        np.asarray(fr.response[:, 0, 1]), expected_in2, rtol=1e-6
    )


def test_frequency_response_siso_backward_compatible_shape():
    """SISO case unchanged: shape ``(K, 1, 1)`` from phase 1."""
    linsys = _siso_lowpass(a=-1.0)
    fr = frequency_response(linsys, jnp.array([1.0]))
    assert fr.response.shape == (1, 1, 1)
    assert np.isclose(
        float(fr.magnitudes[0, 0, 0]), 1.0 / np.sqrt(2.0), atol=1e-5
    )


# ---------------------------------------------------------------------------
# bode_data — MIMO shape and per-channel agreement
# ---------------------------------------------------------------------------


def test_bode_data_mimo_returns_K_p_m_shape():
    """For a 2×2 MIMO system :func:`bode_data` keeps ``(K, p, m)`` shape."""
    linsys = _decoupled_2x2()
    omegas = jnp.array([0.1, 1.0, 10.0])
    bd = bode_data(linsys, omegas)

    assert bd["omega"].shape == (3,)
    assert bd["freq_hz"].shape == (3,)
    assert bd["magnitude_db"].shape == (3, 2, 2)
    assert bd["phase_deg"].shape == (3, 2, 2)


def test_bode_data_mimo_diagonal_matches_siso():
    """Each diagonal Bode channel of the MIMO system equals the
    corresponding isolated SISO :func:`bode_data` curve."""
    linsys_mimo = _decoupled_2x2(a1=-1.0, a2=-2.0)
    linsys_siso_0 = _siso_lowpass(a=-1.0)
    linsys_siso_1 = _siso_lowpass(a=-2.0)
    omegas = jnp.array([0.1, 1.0, 5.0, 20.0])

    bd_m = bode_data(linsys_mimo, omegas)
    bd_0 = bode_data(linsys_siso_0, omegas)
    bd_1 = bode_data(linsys_siso_1, omegas)

    np.testing.assert_allclose(
        np.asarray(bd_m["magnitude_db"][:, 0, 0]),
        np.asarray(bd_0["magnitude_db"]),
        rtol=1e-6,
    )
    np.testing.assert_allclose(
        np.asarray(bd_m["magnitude_db"][:, 1, 1]),
        np.asarray(bd_1["magnitude_db"]),
        rtol=1e-6,
    )
    np.testing.assert_allclose(
        np.asarray(bd_m["phase_deg"][:, 0, 0]),
        np.asarray(bd_0["phase_deg"]),
        atol=1e-4,
    )
    np.testing.assert_allclose(
        np.asarray(bd_m["phase_deg"][:, 1, 1]),
        np.asarray(bd_1["phase_deg"]),
        atol=1e-4,
    )


def test_bode_data_mimo_off_diagonal_is_neg_inf_db():
    """For the decoupled system the off-diagonal magnitudes hit the
    log floor (``-300·20`` dB) — i.e. truly zero."""
    linsys = _decoupled_2x2()
    omegas = jnp.array([0.1, 1.0, 10.0])
    bd = bode_data(linsys, omegas)
    mag_db = np.asarray(bd["magnitude_db"])
    # Off-diagonals: |G| = 0 → clipped to 1e-300 → -6000 dB.
    assert np.all(mag_db[:, 0, 1] < -1000.0)
    assert np.all(mag_db[:, 1, 0] < -1000.0)


def test_bode_data_siso_squeezes_to_1d_unchanged():
    """SISO case stays squeezed (backward-compat with phase 1)."""
    linsys = _siso_lowpass(a=-1.0)
    omegas = jnp.array([0.1, 1.0, 10.0])
    bd = bode_data(linsys, omegas)
    # Phase-1 contract: 1-D arrays for SISO.
    assert bd["magnitude_db"].shape == (3,)
    assert bd["phase_deg"].shape == (3,)


def test_bode_data_mimo_phase_unwrap_per_channel():
    """Phase unwrap runs along the frequency axis independently per
    ``(output, input)`` channel — a wrap on one channel must not
    pollute another channel."""
    # Construct a system whose two channels have very different phase
    # behaviour over the frequency band so a single global unwrap would
    # mix them.  Diagonal poles at -1 and -10 → channel 0 hits -90° much
    # earlier than channel 1.
    linsys = _decoupled_2x2(a1=-1.0, a2=-10.0)
    # Dense omega sweep across both corners to exercise unwrap.
    omegas = jnp.logspace(-2.0, 2.5, 40)
    bd = bode_data(linsys, omegas)
    phase_deg = np.asarray(bd["phase_deg"])

    # Channel-(0, 0) phase is monotonically decreasing from 0 toward -90°.
    diffs_00 = np.diff(phase_deg[:, 0, 0])
    assert np.all(diffs_00 <= 1e-3)  # non-increasing (within fp noise)
    assert phase_deg[0, 0, 0] > -10.0
    assert phase_deg[-1, 0, 0] < -85.0

    # Channel-(1, 1) phase is similarly monotone.
    diffs_11 = np.diff(phase_deg[:, 1, 1])
    assert np.all(diffs_11 <= 1e-3)
    assert phase_deg[0, 1, 1] > -10.0
    assert phase_deg[-1, 1, 1] < -85.0


# ---------------------------------------------------------------------------
# Differentiability / jit on MIMO path
# ---------------------------------------------------------------------------


def test_frequency_response_mimo_is_jit_traceable():
    """``frequency_response`` composes inside ``jax.jit`` for MIMO."""
    linsys = _decoupled_2x2()

    @jax.jit
    def mag_at(omega):
        fr = frequency_response(linsys, jnp.atleast_1d(omega))
        return fr.magnitudes[0]

    out = mag_at(1.0)
    assert out.shape == (2, 2)
    # Channel (0,0): |1/(j·1+1)| = 1/sqrt(2)
    assert np.isclose(float(out[0, 0]), 1.0 / np.sqrt(2.0), atol=1e-6)
    # Off-diagonal stays zero.
    assert np.isclose(float(out[0, 1]), 0.0, atol=1e-12)


def test_frequency_response_mimo_grad_through_A():
    """Gradient of a MIMO magnitude w.r.t. an entry of ``A`` is finite."""
    omegas = jnp.array([1.0])

    def channel00_mag(a11):
        A = jnp.array([[a11, 0.0], [0.0, -2.0]])
        linsys = LinearizedSystem(
            A=A,
            B=jnp.eye(2),
            C=jnp.eye(2),
            D=jnp.zeros((2, 2)),
            operating_point={"x": jnp.zeros(2), "u": jnp.zeros(2)},
        )
        fr = frequency_response(linsys, omegas)
        return fr.magnitudes[0, 0, 0]

    g = jax.grad(channel00_mag)(-1.0)
    assert jnp.isfinite(g)
    # Sanity: derivative of 1/sqrt(ω²+a²) at ω=1, a=1 w.r.t. a is
    # -a/(ω²+a²)^{3/2} = -1/2^{3/2} ≈ -0.3536.  The state-space param
    # has ``a = -a11`` so d|G|/da11 = +1/(2√2) ≈ 0.3536.
    assert np.isclose(float(g), 1.0 / (2.0 * np.sqrt(2.0)), atol=1e-5)
