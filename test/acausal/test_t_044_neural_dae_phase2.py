# SPDX-License-Identifier: MIT

"""T-044 NeuralDAEBlock — phase 2 (first-class, diagram-authored).

Phase 1 (``add_neural_correction``) bolts a learned term onto an already
*compiled* system.  Phase 2 lets the author declare the term *in the diagram*
via :class:`NeuralDAEBlock` + ``AcausalDiagram.add_neural_correction_block``;
the compiler resolves the block's physical-state targets to differential rows
and injects ``f_NN`` at the same post-index-reduction RHS site (never touching
the symbolic / Pantelides path).  Acceptance subset for phase 2:

* zero-NN byte-equivalence (a diagram-authored inert block changes nothing),
* the algebraic constraint rows stay untouched (T-003),
* ``targets=[(comp, "v")]`` lands the correction on the right differential row,
* the block's ``θ`` is exposed as ``f"{name}_theta"`` and ``jax.grad`` flows
  into it end-to-end, recovering an unmodeled drag,
* multiple blocks compose; bad targets / dup names fail with clear errors.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import jaxonomy
from jaxonomy import SimulatorOptions
from jaxonomy.acausal import AcausalCompiler, AcausalDiagram, EqnEnv, NeuralDAEBlock
from jaxonomy.acausal import translational as trans
from jaxonomy.acausal.component_library.planar import PlanarPendulum
from jaxonomy.testing.markers import skip_if_not_jax

skip_if_not_jax()


def _msd(D=None):
    """Compiled mass-spring (+ optional parallel damper ``D`` for ground truth)."""
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
    return AcausalCompiler(ev, ad)(leaf_backend="jax"), m1


def _opts(autodiff=True):
    return SimulatorOptions(
        math_backend="jax", ode_solver_method="bdf", enable_autodiff=autodiff,
        rtol=1e-8, atol=1e-10, max_major_steps=400,
    )


def _final_state(sys, params=None, autodiff=True):
    builder = jaxonomy.DiagramBuilder()
    s = builder.add(sys)
    diagram = builder.build()
    ctx = diagram.create_context()
    if params:
        sub = ctx[s.system_id]
        for k, v in params.items():
            sub = sub.with_parameter(k, v)
        ctx = ctx.with_subcontext(s.system_id, sub)
    res = jaxonomy.simulate(diagram, ctx, (0.0, 3.0), options=_opts(autodiff))
    return res.context[s.system_id].continuous_state


def test_zero_nn_byte_equivalence():
    """An inert diagram-authored block matches the bare system exactly."""
    base = np.asarray(_final_state(_msd()[0], autodiff=False))
    sysz, _ = _build_with_drag(theta=jnp.zeros(1))
    zero = np.asarray(
        _final_state(sysz, params={"drag_theta": jnp.zeros(1)}, autodiff=False)
    )
    np.testing.assert_array_equal(base, zero)


def _build_with_drag(theta):
    """MSD with a velocity-drag NeuralDAEBlock targeting m1's velocity."""
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
    ad.add_neural_correction_block(
        NeuralDAEBlock(lambda t, v, th: th[0] * v, theta,
                       targets=[(m1, "v")], name="drag")
    )
    return AcausalCompiler(ev, ad)(leaf_backend="jax"), m1


def test_param_name_exposed():
    """The block's θ is exposed as ``f"{name}_theta"`` in the context."""
    sys, _ = _build_with_drag(theta=jnp.zeros(1))
    builder = jaxonomy.DiagramBuilder()
    s = builder.add(sys)
    diagram = builder.build()
    ctx = diagram.create_context()
    assert "drag_theta" in ctx[s.system_id].parameters


def test_target_lands_on_velocity_row_and_constraints_untouched():
    """``targets=[(m1, "v")]`` puts the correction on the differential velocity
    row only; algebraic constraint rows are byte-identical to the bare system."""
    x_test = jnp.array([0.4, -0.7, 0.3, 0.1])  # [x_diff(2), x_alg(2)]

    def rhs_at(sys, params=None):
        builder = jaxonomy.DiagramBuilder()
        s = builder.add(sys)
        diagram = builder.build()
        ctx = diagram.create_context()
        sub = ctx[s.system_id].with_continuous_state(x_test)
        if params:
            for k, v in params.items():
                sub = sub.with_parameter(k, v)
        ctx = ctx.with_subcontext(s.system_id, sub)
        return np.asarray(s.eval_time_derivatives(ctx))

    base = rhs_at(_msd()[0])
    sys, _ = _build_with_drag(theta=jnp.array([0.5]))
    corr = rhs_at(sys, params={"drag_theta": jnp.array([0.5])})

    n_ode = sys.n_ode
    # algebraic rows identical
    np.testing.assert_allclose(base[n_ode:], corr[n_ode:], atol=1e-12)
    # exactly one differential row changed (the velocity row), by θ·v
    delta = corr[:n_ode] - base[:n_ode]
    changed = np.flatnonzero(np.abs(delta) > 1e-12)
    assert changed.size == 1, f"expected one corrected row, got {changed}"
    row = int(changed[0])
    np.testing.assert_allclose(delta[row], 0.5 * float(x_test[row]), atol=1e-10)


def test_multiple_blocks_compose():
    """Two blocks with distinct names sum, each on its own target row."""
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
    # one block on velocity, one on position (both differential rows of the MSD)
    ad.add_neural_correction_block(
        NeuralDAEBlock(lambda t, v, th: th[0] * v, jnp.array([0.5]),
                       targets=[(m1, "v")], name="a"))
    ad.add_neural_correction_block(
        NeuralDAEBlock(lambda t, x, th: th[0] * x, jnp.array([0.25]),
                       targets=[(m1, "x")], name="b"))
    sys = AcausalCompiler(ev, ad)(leaf_backend="jax")

    base, _ = _msd()
    x_test = jnp.array([0.4, -0.7, 0.3, 0.1])

    def rhs_at(s, params=None):
        builder = jaxonomy.DiagramBuilder()
        sw = builder.add(s)
        diagram = builder.build()
        ctx = diagram.create_context()
        sub = ctx[sw.system_id].with_continuous_state(x_test)
        for k, v in (params or {}).items():
            sub = sub.with_parameter(k, v)
        ctx = ctx.with_subcontext(sw.system_id, sub)
        return np.asarray(sw.eval_time_derivatives(ctx))

    d_base = rhs_at(base)
    d_two = rhs_at(sys, params={"a_theta": jnp.array([0.5]), "b_theta": jnp.array([0.25])})
    n_ode = sys.n_ode
    # constraints untouched; exactly the two differential rows changed
    np.testing.assert_allclose(d_base[n_ode:], d_two[n_ode:], atol=1e-12)
    changed = np.flatnonzero(np.abs(d_two[:n_ode] - d_base[:n_ode]) > 1e-12)
    assert changed.size == 2, f"expected two corrected rows, got {changed}"


def test_rejects_targets_and_state_rows_both():
    with pytest.raises(ValueError, match="either `targets`"):
        NeuralDAEBlock(lambda t, x, th: th, jnp.zeros(1),
                       targets=[(object(), "v")], state_rows=[0], name="x")


def test_rejects_empty_name():
    with pytest.raises(ValueError, match="non-empty"):
        NeuralDAEBlock(lambda t, x, th: th, jnp.zeros(1), name="")


def test_rejects_missing_state_name():
    """A target naming a state the component does not have fails at compile."""
    ev = EqnEnv()
    ad = AcausalDiagram()
    m1 = trans.Mass(ev, name="m1", M=1.0,
                    initial_position=1.0, initial_position_fixed=True,
                    initial_velocity=0.0, initial_velocity_fixed=True)
    sp1 = trans.Spring(ev, name="sp1", K=1.0)
    r1 = trans.FixedPosition(ev, name="r1", initial_position=0.0)
    ad.connect(m1, "flange", sp1, "flange_a")
    ad.connect(sp1, "flange_b", r1, "flange")
    ad.add_neural_correction_block(
        NeuralDAEBlock(lambda t, z, th: th[0] * z, jnp.zeros(1),
                       targets=[(m1, "nonexistent")], name="bad"))
    with pytest.raises(ValueError, match="no .*state symbol named"):
        AcausalCompiler(ev, ad)(leaf_backend="jax")


def test_rejects_algebraic_target():
    """Targeting a state that index reduction made *algebraic* fails clearly.

    Uses the pendulum's Lagrange multiplier ``lam``, which is a structural
    algebraic unknown (no derivative chain) and is therefore algebraic in every
    index-reduction partition. (Which Cartesian velocity ``vx``/``vy`` stays
    differential is PYTHONHASHSEED-dependent, so it can't be relied on here.)"""
    ev = EqnEnv()
    ad = AcausalDiagram()
    p = PlanarPendulum(ev, name="pend", m=1.0, L=1.0)
    ad.comps[p] = None  # self-contained component, no connect
    ad.add_neural_correction_block(
        NeuralDAEBlock(lambda t, v, th: th[0] * v, jnp.zeros(1),
                       targets=[(p, "lam")], name="drag"))
    with pytest.raises(ValueError, match="algebraic"):
        AcausalCompiler(ev, ad)(leaf_backend="jax")


def test_rejects_duplicate_block_names():
    ev = EqnEnv()
    ad = AcausalDiagram()
    m1 = trans.Mass(ev, name="m1", M=1.0,
                    initial_position=1.0, initial_position_fixed=True,
                    initial_velocity=0.0, initial_velocity_fixed=True)
    sp1 = trans.Spring(ev, name="sp1", K=1.0)
    r1 = trans.FixedPosition(ev, name="r1", initial_position=0.0)
    ad.connect(m1, "flange", sp1, "flange_a")
    ad.connect(sp1, "flange_b", r1, "flange")
    ad.add_neural_correction_block(
        NeuralDAEBlock(lambda t, v, th: th[0] * v, jnp.zeros(1),
                       targets=[(m1, "v")], name="dup"))
    ad.add_neural_correction_block(
        NeuralDAEBlock(lambda t, x, th: th[0] * x, jnp.zeros(1),
                       targets=[(m1, "x")], name="dup"))
    with pytest.raises(ValueError, match="[Dd]uplicate"):
        AcausalCompiler(ev, ad)(leaf_backend="jax")


@pytest.mark.slow
def test_gradient_flows_into_theta():
    """``jax.grad`` of a terminal cost flows through ``simulate`` into the
    diagram-authored block's θ."""
    truth = _final_state(_msd(D=0.3)[0], autodiff=False)[:2]
    model, _ = _build_with_drag(theta=jnp.zeros(1))

    def loss(theta):
        fs = _final_state(model, params={"drag_theta": theta})[:2]
        return jnp.sum((fs - truth) ** 2)

    g = jax.grad(loss)(jnp.zeros(1))
    assert np.all(np.isfinite(np.asarray(g)))
    assert float(jnp.linalg.norm(g)) > 1e-3


@pytest.mark.slow
def test_drag_recovery_reduces_loss():
    """Fitting the diagram-authored correction recovers the damping a real
    Damper component produced."""
    truth = _final_state(_msd(D=0.3)[0], autodiff=False)[:2]
    model, _ = _build_with_drag(theta=jnp.zeros(1))

    def loss(theta):
        fs = _final_state(model, params={"drag_theta": theta})[:2]
        return jnp.sum((fs - truth) ** 2)

    grad = jax.grad(loss)
    theta = jnp.zeros(1)
    l0 = float(loss(theta))
    for _ in range(40):
        theta = theta - 0.5 * grad(theta)
    lf = float(loss(theta))
    assert lf < 0.05 * l0, f"fit did not reduce loss enough: {l0:.5f} -> {lf:.5f}"
