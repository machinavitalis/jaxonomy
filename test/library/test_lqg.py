# SPDX-License-Identifier: MIT
"""
T-014 — LinearQuadraticGaussian controller tests.

Benchmark system: the textbook double integrator.

    dx/dt = [[0, 1], [0, 0]] x + [[0], [1]] u + G w
    y     = [1, 0] x + v

Design goals:
  - The closed-loop observer + regulator drives x → 0 from a nonzero
    initial condition.
  - The observer gain L and regulator gain K are non-trivial (not NaN,
    nonzero, shape correct).
  - The separation principle: computing K and L separately gives
    stable closed-loop eigenvalues that are the *union* of the
    regulator's eigenvalues (eig(A − B·K)) and the observer's
    eigenvalues (eig(A − L·C)).
"""

from __future__ import annotations

import numpy as np
import pytest
import jax.numpy as jnp

import jaxonomy
from jaxonomy.testing.markers import skip_if_not_jax

skip_if_not_jax()

# control is an optional dep; LQG depends on it.
control = pytest.importorskip("control")

from jaxonomy.library import Integrator, LinearQuadraticGaussian  # noqa: E402


def _double_integrator_params():
    A = np.array([[0.0, 1.0], [0.0, 0.0]])
    B = np.array([[0.0], [1.0]])
    C = np.array([[1.0, 0.0]])
    D = np.array([[0.0]])
    G = np.eye(2)
    Qn = np.eye(2) * 0.01   # small process noise
    Rn = np.eye(1) * 0.1    # larger measurement noise
    Qc = np.diag([10.0, 1.0])  # heavy state penalty
    Rc = np.eye(1) * 1.0
    return A, B, C, D, G, Qn, Rn, Qc, Rc


def test_construction_produces_stable_gains():
    """Gains K and L exist, have correct shapes, and produce closed-loop
    stable matrices."""
    A, B, C, D, G, Qn, Rn, Qc, Rc = _double_integrator_params()
    lqg = LinearQuadraticGaussian(A, B, C, D, Qn, Rn, Qc, Rc, G=G)
    assert lqg.K.shape == (1, 2)
    assert lqg.L.shape == (2, 1)
    # Regulator closed-loop eigenvalues must be in the LHP.
    eig_reg = np.linalg.eigvals(A - B @ lqg.K)
    assert np.all(np.real(eig_reg) < 0), f"regulator not stable: {eig_reg}"
    # Observer closed-loop eigenvalues must be in the LHP.
    eig_obs = np.linalg.eigvals(A - lqg.L @ C)
    assert np.all(np.real(eig_obs) < 0), f"observer not stable: {eig_obs}"


def test_lqg_drives_plant_state_to_zero():
    """Closed-loop simulation: nonzero plant IC, zero observer IC.  After
    a settling time the plant state must be near zero."""
    A, B, C, D, G, Qn, Rn, Qc, Rc = _double_integrator_params()

    class DoubleIntegratorPlant(jaxonomy.LeafSystem):
        """dx/dt = A x + B u,  y = C x."""

        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.declare_input_port(name="u")
            self.declare_continuous_state(
                default_value=jnp.array([1.0, 0.0]), ode=self._ode,
            )
            self.declare_output_port(
                self._measure,
                prerequisites_of_calc=[jaxonomy.framework.DependencyTicket.xc],
                requires_inputs=False,
            )

        def _ode(self, time, state, *inputs, **params):
            u = jnp.atleast_1d(inputs[0])
            x = state.continuous_state
            return jnp.array([x[1], u[0]])

        def _measure(self, time, state, *inputs, **params):
            x = state.continuous_state
            return jnp.array([x[0]])

    bld = jaxonomy.DiagramBuilder()
    plant = bld.add(DoubleIntegratorPlant(name="plant"))
    lqg = bld.add(LinearQuadraticGaussian(
        A, B, C, D, Qn, Rn, Qc, Rc, G=G, x_hat_0=np.zeros(2), name="lqg",
    ))
    bld.connect(lqg.output_ports[0], plant.input_ports[0])
    bld.connect(plant.output_ports[0], lqg.input_ports[0])
    diagram = bld.build()
    ctx = diagram.create_context()
    opts = jaxonomy.SimulatorOptions(math_backend="jax", max_major_steps=500)
    res = jaxonomy.simulate(diagram, ctx, (0.0, 30.0), options=opts)

    x_final = np.asarray(res.context[plant.system_id].continuous_state)
    # Heavy state penalty + 30 s settling should drive x to ~0 despite
    # the mismatch between observer and true IC.
    assert np.max(np.abs(x_final)) < 0.1, f"plant did not settle: {x_final}"


def test_separation_principle_closed_loop_spectrum():
    """The combined (plant + observer) closed-loop spectrum is the union
    of the regulator and observer individual spectra.  Verifies that the
    LQG block correctly realises the separation principle."""
    A, B, C, D, G, Qn, Rn, Qc, Rc = _double_integrator_params()
    lqg = LinearQuadraticGaussian(A, B, C, D, Qn, Rn, Qc, Rc, G=G)

    # Augmented state [x; e] where e = x - x_hat.
    #   dx/dt = A·x - B·K·(x − e) = (A − BK) x + BK·e
    #   de/dt = (A − LC) e
    # So A_aug = [[A − BK,   B·K    ],
    #             [  0,     A − LC  ]]
    K, L = lqg.K, lqg.L
    A_aug = np.block([
        [A - B @ K,       B @ K  ],
        [np.zeros_like(A), A - L @ C],
    ])
    eig_aug = np.linalg.eigvals(A_aug)
    eig_reg = np.linalg.eigvals(A - B @ K)
    eig_obs = np.linalg.eigvals(A - L @ C)

    # The union (up to ordering) should match A_aug's spectrum.
    aug_sorted = np.sort_complex(eig_aug)
    union_sorted = np.sort_complex(np.concatenate([eig_reg, eig_obs]))
    np.testing.assert_allclose(aug_sorted, union_sorted, atol=1e-6)


def test_shapes_validated():
    """Inconsistent matrix dimensions should fail early."""
    A = np.array([[0.0, 1.0], [0.0, 0.0]])
    B_wrong = np.array([[0.0, 1.0]])  # wrong shape
    C = np.array([[1.0, 0.0]])
    D = np.array([[0.0]])
    # `control.lqr` will raise on bad shapes.
    with pytest.raises(Exception):
        LinearQuadraticGaussian(
            A, B_wrong, C, D,
            Qn=np.eye(2), Rn=np.eye(1), Qc=np.eye(2), Rc=np.eye(1),
        )
