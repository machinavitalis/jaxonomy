# SPDX-License-Identifier: MIT

"""T-109 followup-control-cross-validate — cross-tool validation against
``python-control`` (SLICOT-backed) on the canonical controls fixtures
called out in T-109 Verification: Mass-Spring-Damper, DC motor, and a
linearized pendulum.

Why this file exists: the T-109 phase 1/2/3 helpers (``bode_data``,
``nyquist_data``, ``step_response``, ``impulse_response``,
``pole_zero_map``) were each unit-tested against analytical fixtures
(integrator, first-order plant, oscillator) inside their own followup
files.  This file adds the cross-tool sanity check — same input,
different implementation — so that any future refactor that breaks
agreement with the reference shows up here rather than going
unnoticed.

If ``python-control`` is not importable, the whole module is skipped
(non-blocking dependency for the jaxonomy fast suite).
"""

import jax.numpy as jnp
import numpy as np
import pytest

control = pytest.importorskip("control")

from jaxonomy.library import (
    LinearizedSystem,
    bode_data,
    impulse_response,
    nyquist_data,
    pole_zero_map,
    step_response,
)


# ---------------------------------------------------------------------------
# Fixture plants.
# ---------------------------------------------------------------------------


def _mass_spring_damper(m=1.0, c=0.4, k=2.0):
    """Mass-spring-damper: m·ẍ + c·ẋ + k·x = u, output x.

    State [x, v]; A = [[0, 1], [-k/m, -c/m]]; B = [[0], [1/m]];
    C = [[1, 0]]; D = [[0]].  Transfer function 1/(m s² + c s + k).
    """
    A = jnp.array([[0.0, 1.0], [-k / m, -c / m]])
    B = jnp.array([[0.0], [1.0 / m]])
    C = jnp.array([[1.0, 0.0]])
    D = jnp.array([[0.0]])
    return (
        LinearizedSystem(A=A, B=B, C=C, D=D,
                         operating_point={"x": jnp.zeros(2), "u": jnp.zeros(1)}),
        control.StateSpace(np.asarray(A), np.asarray(B), np.asarray(C), np.asarray(D)),
    )


def _dc_motor(R=2.0, L=0.5, J=0.02, b=0.1, K=0.1):
    """Armature-controlled DC motor.

    State [omega, i]; angular velocity and armature current.
    Inputs: voltage V.  Output: angular velocity omega.

    .. code-block:: text

        J·dω/dt = K·i − b·ω
        L·di/dt = V − R·i − K·ω
    """
    A = jnp.array([[-b / J, K / J],
                   [-K / L, -R / L]])
    B = jnp.array([[0.0], [1.0 / L]])
    C = jnp.array([[1.0, 0.0]])
    D = jnp.array([[0.0]])
    return (
        LinearizedSystem(A=A, B=B, C=C, D=D,
                         operating_point={"x": jnp.zeros(2), "u": jnp.zeros(1)}),
        control.StateSpace(np.asarray(A), np.asarray(B), np.asarray(C), np.asarray(D)),
    )


def _pendulum_linearized_around_downward(g=9.81, l=1.0, m=1.0, c=0.1):
    """Simple pendulum linearized around the downward equilibrium.

    .. code-block:: text

        ml²·θ̈ + c·θ̇ + mgl·sin(θ) = u
        ≈ ml²·θ̈ + c·θ̇ + mgl·θ        for small θ

    State [θ, θ̇]; A = [[0, 1], [-g/l, -c/(ml²)]]; B = [[0], [1/(ml²)]];
    C = [[1, 0]]; D = [[0]].  Returns a *stable* second-order system
    so its Bode/Nyquist/step/impulse responses are well behaved.
    """
    A = jnp.array([[0.0, 1.0],
                   [-g / l, -c / (m * l ** 2)]])
    B = jnp.array([[0.0], [1.0 / (m * l ** 2)]])
    C = jnp.array([[1.0, 0.0]])
    D = jnp.array([[0.0]])
    return (
        LinearizedSystem(A=A, B=B, C=C, D=D,
                         operating_point={"x": jnp.zeros(2), "u": jnp.zeros(1)}),
        control.StateSpace(np.asarray(A), np.asarray(B), np.asarray(C), np.asarray(D)),
    )


_PLANTS = [
    pytest.param(_mass_spring_damper, id="mass_spring_damper"),
    pytest.param(_dc_motor, id="dc_motor"),
    pytest.param(_pendulum_linearized_around_downward, id="pendulum_linearized"),
]


# ---------------------------------------------------------------------------
# Bode magnitude/phase.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("plant_fn", _PLANTS)
def test_bode_data_matches_python_control(plant_fn):
    linsys, ref = plant_fn()
    omegas = np.logspace(-2, 2, 40)
    data = bode_data(linsys, jnp.asarray(omegas))

    mag, phase, _ = control.frequency_response(ref, omegas)
    mag = np.atleast_1d(np.squeeze(np.asarray(mag)))
    phase = np.atleast_1d(np.squeeze(np.asarray(phase)))

    mag_db_ref = 20.0 * np.log10(np.maximum(mag, 1e-300))
    phase_deg_ref = np.unwrap(phase) * (180.0 / np.pi)

    # Magnitude is logarithmic — tolerate 1e-6 dB; phase tolerance 1e-4°.
    np.testing.assert_allclose(np.asarray(data["magnitude_db"]), mag_db_ref, atol=1e-6)
    np.testing.assert_allclose(np.asarray(data["phase_deg"]), phase_deg_ref, atol=1e-4)


# ---------------------------------------------------------------------------
# Nyquist real/imag.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("plant_fn", _PLANTS)
def test_nyquist_data_matches_python_control(plant_fn):
    linsys, ref = plant_fn()
    omegas = np.logspace(-2, 2, 40)
    data = nyquist_data(linsys, jnp.asarray(omegas))

    # python-control's frequency_response returns complex G(jω).  The
    # real/imag parts are exactly what nyquist_data exposes.  ``frdata``
    # replaced the deprecated ``fresp`` attribute in python-control 0.10.
    fr = control.frequency_response(ref, omegas)
    resp = np.atleast_1d(np.squeeze(np.asarray(
        getattr(fr, "frdata", None) if getattr(fr, "frdata", None) is not None
        else fr.fresp
    )))
    np.testing.assert_allclose(np.asarray(data["real"]), np.real(resp), atol=1e-9)
    np.testing.assert_allclose(np.asarray(data["imag"]), np.imag(resp), atol=1e-9)


# ---------------------------------------------------------------------------
# Poles + zeros.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("plant_fn", _PLANTS)
def test_pole_zero_map_matches_python_control(plant_fn):
    linsys, ref = plant_fn()
    pzm = pole_zero_map(linsys)

    ref_poles = np.sort_complex(np.asarray(ref.poles()))
    poles = np.sort_complex(np.asarray(pzm["poles"]))
    np.testing.assert_allclose(poles, ref_poles, atol=1e-8)

    ref_zeros = np.asarray(ref.zeros())
    zeros = np.asarray(pzm["zeros"])
    # All three fixtures are minimum-phase + strictly proper, so they
    # have zero (or one, depending on the SISO/MIMO interpretation) finite
    # transmission zeros.  Compare *sets* after sorting; either both
    # empty or both equal.
    if ref_zeros.size == 0:
        # python-control reports no finite zeros; our Rosenbrock pencil
        # may still flag near-infinity zeros.  Filter ours to a sensible
        # band ([1e3 magnitude] is "essentially infinity" for these
        # plants) before comparing.
        finite_zeros = zeros[np.abs(zeros) < 1e3]
        assert finite_zeros.size == 0, (
            f"Expected no finite zeros, got {finite_zeros}"
        )
    else:
        np.testing.assert_allclose(
            np.sort_complex(zeros), np.sort_complex(ref_zeros), atol=1e-6
        )


# ---------------------------------------------------------------------------
# Step + impulse response.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("plant_fn", _PLANTS)
def test_step_response_matches_python_control(plant_fn):
    linsys, ref = plant_fn()
    t = np.linspace(0.0, 5.0, 51)

    y_ours = np.asarray(step_response(linsys, jnp.asarray(t)))  # (K, p, m)
    # SISO: extract the (output 0, input 0) channel.
    y_ours = y_ours[..., 0, 0]

    _, y_ref = control.step_response(ref, T=t)
    y_ref = np.atleast_1d(np.squeeze(np.asarray(y_ref)))

    np.testing.assert_allclose(y_ours, y_ref, atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize("plant_fn", _PLANTS)
def test_impulse_response_matches_python_control(plant_fn):
    linsys, ref = plant_fn()
    t = np.linspace(0.0, 5.0, 51)

    y_ours = np.asarray(impulse_response(linsys, jnp.asarray(t)))  # (K, p, m)
    y_ours = y_ours[..., 0, 0]

    # python-control's ``impulse_response`` uses a discrete-time
    # simulation convention that introduces a one-sample lag relative
    # to the closed-form ``y(t) = C·expm(A·t)·B`` formula.  The
    # mathematically equivalent reference is ``initial_response`` with
    # ``X0 = B``: for an LTI ``ẋ = Ax, y = Cx`` started from
    # ``x(0) = B``, ``y(t) = C·expm(A·t)·B``, which is exactly the
    # finite part of the continuous-time impulse response that
    # :func:`impulse_response` returns.
    X0 = np.asarray(linsys.B)[:, 0]
    _, y_ref = control.initial_response(ref, T=t, X0=X0)
    y_ref = np.atleast_1d(np.squeeze(np.asarray(y_ref)))

    np.testing.assert_allclose(y_ours, y_ref, atol=1e-6, rtol=1e-6)
