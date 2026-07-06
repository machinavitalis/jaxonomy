# SPDX-License-Identifier: MIT

"""T-044 NeuralDAEBlock — phase 1 (post-hoc RHS correction).

``add_neural_correction`` adds a learned term ``f_NN(t, x; θ)`` to the
*differential* rows of a compiled acausal DAE's RHS, with ``θ`` a dynamic
parameter that ``jax.grad`` flows into through ``simulate``.  Acceptance subset
for phase 1:

* zero-NN byte-equivalence (the wrapper is inert when ``f_NN ≡ 0``),
* the algebraic constraint rows are untouched (constraint structure preserved),
* gradients flow into ``θ`` end-to-end through the BDF-DAE adjoint,
* fitting recovers an unmodeled term (a velocity-proportional drag a damper
  would have produced).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import jaxonomy
from jaxonomy import SimulatorOptions
from jaxonomy.acausal import AcausalCompiler, AcausalDiagram, EqnEnv
from jaxonomy.acausal import translational as trans
from jaxonomy.library.neural_dae import add_neural_correction
from jaxonomy.testing.markers import skip_if_not_jax

skip_if_not_jax()


def _build_msd(D=None):
    """Compiled mass-spring (+ optional damper ``D``) acausal DAE.

    Mass ``M=1`` from ``x0=1, v0=0`` on a spring ``K=1`` to a fixed wall; an
    optional parallel damper ``D`` supplies the "true" unmodeled drag.
    """
    ev = EqnEnv()
    ad = AcausalDiagram()
    m1 = trans.Mass(
        ev, name="m1", M=1.0,
        initial_position=1.0, initial_position_fixed=True,
        initial_velocity=0.0, initial_velocity_fixed=True,
    )
    sp1 = trans.Spring(ev, name="sp1", K=1.0)
    r1 = trans.FixedPosition(ev, name="r1", initial_position=0.0)
    ad.connect(m1, "flange", sp1, "flange_a")
    ad.connect(sp1, "flange_b", r1, "flange")
    if D is not None:
        d1 = trans.Damper(ev, name="d1", D=D)
        ad.connect(m1, "flange", d1, "flange_a")
        ad.connect(r1, "flange", d1, "flange_b")
    return AcausalCompiler(ev, ad)(leaf_backend="jax")


def _opts(autodiff=True):
    return SimulatorOptions(
        math_backend="jax", ode_solver_method="bdf", enable_autodiff=autodiff,
        rtol=1e-8, atol=1e-10, max_major_steps=400,
    )


def _final_state(sys, theta=None, autodiff=True):
    builder = jaxonomy.DiagramBuilder()
    s = builder.add(sys)
    diagram = builder.build()
    ctx = diagram.create_context()
    if theta is not None:
        sub = ctx[s.system_id].with_parameter("nn_theta", theta)
        ctx = ctx.with_subcontext(s.system_id, sub)
    res = jaxonomy.simulate(diagram, ctx, (0.0, 3.0), options=_opts(autodiff))
    return res.context[s.system_id].continuous_state


def test_zero_nn_byte_equivalence():
    """With ``f_NN ≡ 0`` the corrected system matches the bare system exactly."""
    base = np.asarray(_final_state(_build_msd(), autodiff=False))
    sysz = _build_msd()
    add_neural_correction(sysz, lambda t, x, th: jnp.zeros(2), jnp.zeros(1))
    zero = np.asarray(_final_state(sysz, theta=jnp.zeros(1), autodiff=False))
    np.testing.assert_array_equal(base, zero)


def test_algebraic_constraint_rows_untouched():
    """The neural term enters only the differential rows; the algebraic
    constraint rows of the RHS are byte-identical to the bare system (T-003)."""
    x_test = jnp.array([0.4, -0.7, 0.3, 0.1])  # arbitrary [x_diff(2), x_alg(2)]

    def rhs_at(sys, theta=None):
        builder = jaxonomy.DiagramBuilder()
        s = builder.add(sys)
        diagram = builder.build()
        ctx = diagram.create_context()
        sub = ctx[s.system_id].with_continuous_state(x_test)
        if theta is not None:
            sub = sub.with_parameter("nn_theta", theta)
        ctx = ctx.with_subcontext(s.system_id, sub)
        return np.asarray(s.eval_time_derivatives(ctx))

    base = rhs_at(_build_msd())
    corr_sys = _build_msd()
    add_neural_correction(corr_sys, lambda t, x, th: th * x, jnp.array([0.5, -0.3]))
    corr = rhs_at(corr_sys, theta=jnp.array([0.5, -0.3]))

    n_ode = corr_sys.n_ode
    # Algebraic constraint rows identical; differential rows differ by f_NN.
    np.testing.assert_allclose(base[n_ode:], corr[n_ode:], atol=1e-12)
    np.testing.assert_allclose(
        corr[:n_ode] - base[:n_ode],
        np.asarray(jnp.array([0.5, -0.3]) * x_test[:n_ode]),
        atol=1e-10,
    )


@pytest.mark.slow
def test_gradient_flows_into_theta():
    """``jax.grad`` of a terminal cost flows through ``simulate`` into ``θ``."""
    truth = _final_state(_build_msd(D=0.3), autodiff=False)[:2]
    model = _build_msd()
    add_neural_correction(model, lambda t, x, th: th * x, jnp.zeros(2))

    def loss(theta):
        return jnp.sum((_final_state(model, theta)[:2] - truth) ** 2)

    g = jax.grad(loss)(jnp.zeros(2))
    assert np.all(np.isfinite(np.asarray(g)))
    assert float(jnp.linalg.norm(g)) > 1e-3  # away from the optimum -> nonzero


@pytest.mark.slow
def test_drag_recovery_reduces_loss():
    """Fitting the correction recovers the unmodeled damping a damper produced:
    gradient descent drives the model+correction toward the damped truth."""
    truth = _final_state(_build_msd(D=0.3), autodiff=False)[:2]
    model = _build_msd()
    add_neural_correction(model, lambda t, x, th: th * x, jnp.zeros(2))

    def loss(theta):
        return jnp.sum((_final_state(model, theta)[:2] - truth) ** 2)

    # NB: ``loss`` rebuilds the Diagram + Context each call (the acausal
    # consistent-IC solve is not jit-safe), so differentiate it directly rather
    # than ``jax.jit``-ing the whole loss.
    grad = jax.grad(loss)
    theta = jnp.zeros(2)
    l0 = float(loss(theta))
    for _ in range(40):
        theta = theta - 0.5 * grad(theta)
    lf = float(loss(theta))
    assert lf < 0.05 * l0, f"fit did not reduce loss enough: {l0:.5f} -> {lf:.5f}"


def test_rejects_non_compiled_system():
    """A non-compiled object (no ``_cs_base_ode``) is rejected with a clear error."""
    with pytest.raises(TypeError, match="compiled AcausalSystem"):
        add_neural_correction(object(), lambda t, x, th: th, jnp.zeros(1))


def test_rejects_algebraic_row():
    """``state_rows`` may only target differential rows."""
    sys = _build_msd()
    n_ode = sys.n_ode
    with pytest.raises(ValueError, match="differential row"):
        add_neural_correction(
            sys, lambda t, x, th: th, jnp.zeros(1), state_rows=[n_ode],
        )
