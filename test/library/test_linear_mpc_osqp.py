# SPDX-License-Identifier: MIT

"""Regression tests for ``LinearDiscreteTimeMPC``'s OSQP setup.

The block passed the OSQP **1.x** ``warm_starting`` setting to ``setup`` while
the package pinned ``osqp ~= 0.6.5`` (whose setting is ``warm_start``), so on
the documented ``jaxonomy[nmpc]`` install ``setup`` raised
``TypeError: 'warm_starting' is an invalid keyword argument``. The setup is now
version-adaptive (:func:`jaxonomy.library.mpc._osqp_warm_start_kwarg`). This
path previously had no unit test — which is how the mismatch shipped.
"""

import types

import numpy as np
import pytest

from jaxonomy.library import (
    LTISystem,
    LTISystemDiscrete,
    LinearDiscreteTimeMPC,
    LinearizedSystem,
)
from jaxonomy.library import mpc as mpc_mod

pytestmark = pytest.mark.minimal

pytest.importorskip("osqp")


@pytest.mark.parametrize(
    "version,expected",
    [
        ("0.6.5", "warm_start"),
        ("0.6.7", "warm_start"),
        ("1.0.0", "warm_starting"),
        ("1.1.3", "warm_starting"),
        ("", "warm_start"),
        ("not-a-version", "warm_start"),
    ],
)
def test_warm_start_kwarg_by_version(monkeypatch, version, expected):
    # The setting was renamed warm_start -> warm_starting in OSQP 1.0; the
    # helper must pick the right keyword for whichever version is installed.
    monkeypatch.setattr(mpc_mod, "osqp", types.SimpleNamespace(__version__=version))
    assert mpc_mod._osqp_warm_start_kwarg() == expected


def _double_integrator_mpc(warm_start):
    # Controllable double integrator; unconstrained input so the QP (with its
    # hard terminal equality) is always feasible.
    A = np.array([[0.0, 1.0], [0.0, 0.0]])
    B = np.array([[0.0], [1.0]])
    C = np.eye(2)
    D = np.zeros((2, 1))
    plant = LTISystem(A, B, C, D)
    Q = np.eye(2)
    R = np.array([[0.1]])
    return LinearDiscreteTimeMPC(
        plant, Q, R, N=5, dt=0.1, x_ref=np.zeros(2), warm_start=warm_start
    )


@pytest.mark.parametrize("warm_start", [False, True])
def test_setup_and_solve(warm_start):
    # Construction runs OSQP `setup` (the crash site pre-fix); `_np_solve`
    # exercises `update` + `solve`. Both must work on the installed OSQP.
    mpc = _double_integrator_mpc(warm_start)
    u = mpc._np_solve(0.0, None, np.array([1.0, 0.0]))
    assert np.asarray(u).shape == ((2 + 1) * 5,)
    assert np.all(np.isfinite(u))


# ---------------------------------------------------------------------------
# Already-discrete prediction models (dmdc/edmd fits, discretize() output).
# Previously the block always applied its internal Euler step, so a discrete
# model had to be hand-"un-discretized" (A_c=(A-I)/dt, B_c=B/dt) — a silent
# wrong-answer trap for anyone who skipped that step.
# ---------------------------------------------------------------------------

_A_C = np.array([[0.0, 1.0], [0.0, 0.0]])
_B_C = np.array([[0.0], [1.0]])
_DT = 0.1
# Forward-Euler discretization — exactly what the continuous path constructs
# internally, so the two paths must produce identical QPs.
_A_D = np.eye(2) + _DT * _A_C
_B_D = _DT * _B_C


def _mpc_args():
    return dict(Q=np.eye(2), R=np.array([[0.1]]), N=5, dt=_DT, x_ref=np.zeros(2))


def test_discrete_linearized_system_used_as_is():
    # A LinearizedSystem carrying dt must skip the internal Euler step: feeding
    # the Euler-discretized (A_d, B_d) directly must reproduce the continuous
    # path's control exactly.
    cont = LinearDiscreteTimeMPC(
        LTISystem(_A_C, _B_C, np.eye(2), np.zeros((2, 1))), **_mpc_args()
    )
    disc = LinearDiscreteTimeMPC(
        LinearizedSystem(_A_D, _B_D, np.eye(2), np.zeros((2, 1)), {}, dt=_DT),
        **_mpc_args(),
    )
    x0 = np.array([1.0, 0.0])
    u_cont = np.asarray(cont._np_solve(0.0, None, x0))
    u_disc = np.asarray(disc._np_solve(0.0, None, x0))
    assert np.all(np.isfinite(u_disc))
    np.testing.assert_allclose(u_disc, u_cont, rtol=1e-6, atol=1e-8)


def test_discrete_lti_block_accepted():
    plant = LTISystemDiscrete(_A_D, _B_D, np.eye(2), np.zeros((2, 1)), dt=_DT)
    mpc = LinearDiscreteTimeMPC(plant, **_mpc_args())
    u = np.asarray(mpc._np_solve(0.0, None, np.array([1.0, 0.0])))
    assert np.all(np.isfinite(u))


def test_discrete_model_dt_mismatch_raises():
    model = LinearizedSystem(_A_D, _B_D, np.eye(2), np.zeros((2, 1)), {}, dt=0.05)
    with pytest.raises(ValueError, match="discrete-time with dt"):
        LinearDiscreteTimeMPC(model, **_mpc_args())
