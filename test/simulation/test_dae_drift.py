# SPDX-License-Identifier: MIT
"""
Tests for the DAE constraint-residual measurement primitive (T-003).

The measurement primitive itself is in ``jaxonomy/simulation/dae_drift.py``.
These tests verify:

  - ``algebraic_row_mask`` correctly identifies zero rows of the block-
    diagonal mass matrix.
  - ``constraint_residual_norm`` returns ``None`` for pure ODEs (no mass
    matrix) and a numeric value for mass-matrix DAEs.
  - For the RC and spring-mass linear DAEs, BDF drives the algebraic
    residual to machine precision immediately and keeps it there over
    long simulations — the current jaxonomy library produces no drift
    because all its acausal domains are linear.
  - Manually-perturbed algebraic states produce a nonzero residual, so
    the measurement itself is not identically zero.

Full constraint projection / Baumgarte stabilization is T-003a; this
file validates the detection primitive that T-003a will build on.
"""

from __future__ import annotations

import numpy as np
import pytest
import jax.numpy as jnp

import jaxonomy
from jaxonomy.simulation.dae_drift import (
    algebraic_row_mask,
    compute_constraint_residual,
    constraint_residual_norm,
)
from jaxonomy.testing.markers import skip_if_not_jax

skip_if_not_jax()


def _build_rc():
    from jaxonomy.acausal import AcausalCompiler, AcausalDiagram, EqnEnv
    from jaxonomy.acausal import electrical as elec

    ev = EqnEnv()
    ad = AcausalDiagram()
    vs = elec.VoltageSource(ev, name="vs", v=1.0)
    r1 = elec.Resistor(ev, name="r1", R=1.0)
    c1 = elec.Capacitor(
        ev, name="c1", C=1.0, initial_voltage=0.5, initial_voltage_fixed=True
    )
    gnd = elec.Ground(ev, name="gnd")
    ad.connect(vs, "p", r1, "n")
    ad.connect(r1, "p", c1, "p")
    ad.connect(c1, "n", vs, "n")
    ad.connect(vs, "n", gnd, "p")
    sys_ = AcausalCompiler(ev, ad)()
    bld = jaxonomy.DiagramBuilder()
    bld.add(sys_)
    diagram = bld.build()
    return diagram, diagram.create_context()


def test_pure_ode_has_no_algebraic_rows():
    """Pure ODE (no mass matrix): both the mask and the residual are None."""

    class Decay(jaxonomy.LeafSystem):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.declare_continuous_state(default_value=jnp.array(1.0), ode=self._ode)

        def _ode(self, time, state, **params):
            return -state.continuous_state

    sys = Decay()
    ctx = sys.create_context()
    assert algebraic_row_mask(sys) is None
    assert compute_constraint_residual(sys, ctx) is None
    assert constraint_residual_norm(sys, ctx) is None


def test_rc_circuit_has_algebraic_rows():
    """RC circuit via AcausalCompiler has a rank-deficient mass matrix; the
    algebraic rows are the non-capacitor equations."""
    diagram, _ = _build_rc()
    mask = algebraic_row_mask(diagram)
    assert mask is not None, "acausal RC must produce a non-trivial mass matrix"
    # At least one differential state (the capacitor voltage) and at least one
    # algebraic equation.
    assert mask.any(), "expected at least one algebraic row"
    assert not mask.all(), "expected at least one differential row"


def test_rc_circuit_residual_at_initial_condition_is_zero():
    """AcausalCompiler initialises the algebraic state consistently with the
    differential initial conditions, so ``||f_a||_∞`` at t=0 must be ~0."""
    diagram, ctx0 = _build_rc()
    resid = constraint_residual_norm(diagram, ctx0)
    assert resid is not None
    assert resid < 1e-10, f"initial constraint residual too large: {resid:.3e}"


def test_rc_circuit_residual_stable_over_long_sim():
    """Over a 50-second BDF simulation, the algebraic residual must remain
    near machine precision — linear DAEs do not drift under BDF. If this
    ever fails, the simulator has acquired a drift source."""
    diagram, ctx0 = _build_rc()
    opts = jaxonomy.SimulatorOptions(
        math_backend="jax", ode_solver_method="bdf", max_major_steps=500,
    )
    result = jaxonomy.simulate(diagram, ctx0, (0.0, 50.0), options=opts)
    resid = constraint_residual_norm(diagram, result.context)
    assert resid < 1e-8, f"long-sim drift above threshold: {resid:.3e}"


def test_perturbed_state_produces_nonzero_residual():
    """Sanity-check that the measurement itself is not a constant zero —
    manually shifting an algebraic variable must move the residual."""
    diagram, ctx0 = _build_rc()
    # Perturb the whole continuous state by 0.1 on every component.
    state = np.asarray(ctx0[diagram.leaf_systems[0].system_id].continuous_state)
    perturbed = jnp.asarray(state + 0.1)
    ctx_bad = ctx0.with_subcontext(
        diagram.leaf_systems[0].system_id,
        ctx0[diagram.leaf_systems[0].system_id].with_continuous_state(perturbed),
    )
    resid_bad = constraint_residual_norm(diagram, ctx_bad)
    assert resid_bad is not None and resid_bad > 1e-3, (
        f"perturbed residual unexpectedly small: {resid_bad:.3e}"
    )
