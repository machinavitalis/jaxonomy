# SPDX-License-Identifier: MIT

"""Regression tests for T-113-followup-dae-adjoint-sign-bug (FIXED 2026-05-20).

The BDF-DAE adjoint path (``SimulatorOptions(math_backend='jax',
enable_autodiff=True, ode_solver_method='bdf')``) used to return a
*wrong* gradient with respect to acausal parameters that participate in
the semi-explicit DAE through the algebraic constraint block — including
the ``Insulator.R`` thermal-resistance parameter used pervasively in the
battery/thermal demos. On the minimal single-cell repro the AD gradient
was ~25× too small; on the 2-cell pack the relative error exceeded 90%.

Root cause (two compounding bugs in ``_wrapped_advance_to_adj``):
  1. The Cao et al. (2003) terminal-condition correction read and patched
     ``adjoints.ode_solver_state.y``, which is *zero* on this path — the
     objective seed ``∂J/∂x(T)`` actually flows through
     ``adjoints.context.continuous_state``. The correction was therefore
     a no-op operating on zeros.
  2. Even with the seed read correctly, the objective's dependence on the
     ALGEBRAIC terminal states ``x_a(T)`` — which are pinned by the
     constraint ``0 = f_a(x_d, x_a, p)`` — was only partly handled: the
     differential consistent-IC correction was present in spirit but the
     direct terminal boundary term ``-g_{x_a} f_{a,x_a}^{-1} f_{a,p}`` was
     missing, and the raw algebraic seed corrupted the reverse-solve
     quadrature.

The fix (``jaxonomy/simulation/autodiff_rules.py``) reads the seed from
``context.continuous_state``, applies the Cao consistent-IC correction to
the differential seed, ZEROES the algebraic seed, and adds the direct
boundary term to the parameter cotangent. Validated against
forward-mode (``simulate_jacfwd``) and central differences.

Two regression tests:

1. ``test_single_cell_thermal_dJdR_matches_fd`` — smallest reproducer
   (1 ``HeatCapacitor`` + 1 ``Insulator`` driven by a
   ``TemperatureSource``).
2. ``test_two_cell_thermal_pack_dJdR_matches_fd`` — 2-cell version
   mirroring the topology of the original pack report.

Both assert AD-vs-FD agreement on ``dJ/dR`` to within 5%.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import jaxonomy
from jaxonomy.acausal import AcausalCompiler, AcausalDiagram, EqnEnv
from jaxonomy.acausal import thermal as ht
from jaxonomy.simulation import SimulatorOptions


pytestmark = pytest.mark.minimal


# ---------------------------------------------------------------------
# Shared simulation knobs
# ---------------------------------------------------------------------


_OPTS_BDF = SimulatorOptions(
    math_backend="jax",
    ode_solver_method="bdf",
    enable_tracing=True,
    enable_autodiff=True,
    rtol=1e-8,
    atol=1e-10,
)
_T_AMB = 273.15
_C_VAL = 10.0
_T_INIT = 350.0
_T_REF = 280.0
_T_SIM = 1.0


# ---------------------------------------------------------------------
# Single-cell repro (smallest expression of the bug)
# ---------------------------------------------------------------------


def _single_cell_factory():
    """src → Insulator(R) → HeatCapacitor → final-state objective."""

    def _make(R_val: float):
        ev = EqnEnv()
        ad = AcausalDiagram()
        src = ht.TemperatureSource(ev, name="src", temperature=_T_AMB)
        cap = ht.HeatCapacitor(
            ev, name="cap", C=_C_VAL,
            initial_temperature=_T_INIT,
            initial_temperature_fixed=True,
        )
        ins = ht.Insulator(ev, name="ins", R=float(R_val))
        ad.connect(src, "port", ins, "port_a")
        ad.connect(ins, "port_b", cap, "port")
        ac = AcausalCompiler(ev, ad)
        system = ac()
        builder = jaxonomy.DiagramBuilder()
        s = builder.add(system)
        return builder.build(), s

    return _make


def test_single_cell_thermal_dJdR_matches_fd():
    """AD vs central-difference on dJ/dR for a 1-cell thermal RC.

    Objective: ``J = sum((T_final - T_REF)^2)``. Both AD and FD must
    agree to within ~5% on the same point; pre-fix the AD path is
    off by an order of magnitude.
    """
    make = _single_cell_factory()
    R0 = 0.5
    h = 1e-4

    def _J(R_val: float) -> float:
        diagram, s = make(R_val)
        ctx = diagram.create_context()
        res = jaxonomy.simulate(
            diagram, ctx, (0.0, _T_SIM), options=_OPTS_BDF,
        )
        xfinal = np.asarray(res.context[s.system_id].continuous_state)
        return float(np.sum((xfinal - _T_REF) ** 2))

    dJ_dR_fd = (_J(R0 + h) - _J(R0 - h)) / (2.0 * h)

    diagram, s = make(R0)
    ctx0 = diagram.create_context()
    subctx = ctx0[s.system_id]
    assert "ins_R" in subctx.parameters, (
        f"Insulator parameter expected as 'ins_R'; got: "
        f"{list(subctx.parameters.keys())}"
    )

    @jax.jit
    def _J_ad(R_val):
        new_params = dict(subctx.parameters)
        new_params["ins_R"] = R_val
        new_subctx = subctx.with_parameters(new_params)
        new_ctx = ctx0.with_subcontext(s.system_id, new_subctx)
        res = jaxonomy.simulate(
            diagram, new_ctx, (0.0, _T_SIM), options=_OPTS_BDF,
        )
        xfinal = res.context[s.system_id].continuous_state
        return jnp.sum((xfinal - _T_REF) ** 2)

    dJ_dR_ad = float(jax.grad(_J_ad)(jnp.array(R0)))

    rel_err = abs(dJ_dR_ad - dJ_dR_fd) / (abs(dJ_dR_fd) + 1e-12)
    assert rel_err < 0.05, (
        f"AD vs FD disagreement on dJ/dR: FD={dJ_dR_fd:.6f}, "
        f"AD={dJ_dR_ad:.6f}, rel_err={rel_err:.4f}"
    )


# ---------------------------------------------------------------------
# 2-cell pack repro (structural shape requested by the spec)
# ---------------------------------------------------------------------


def _two_cell_factory():
    """Two parallel cooling branches from a single ambient source.

    Topology mirrors a multi-cell battery pack's cooling tree: one
    ambient ``TemperatureSource`` feeds two independent
    ``Insulator → HeatCapacitor`` branches. Each branch's ``R`` is an
    independent parameter so the adjoint must propagate sensitivity
    through two distinct algebraic equations.
    """

    def _make(R_vec: np.ndarray):
        R0_val, R1_val = float(R_vec[0]), float(R_vec[1])
        ev = EqnEnv()
        ad = AcausalDiagram()
        src = ht.TemperatureSource(ev, name="src", temperature=_T_AMB)
        c0 = ht.HeatCapacitor(
            ev, name="c0", C=_C_VAL,
            initial_temperature=_T_INIT,
            initial_temperature_fixed=True,
        )
        c1 = ht.HeatCapacitor(
            ev, name="c1", C=_C_VAL,
            initial_temperature=_T_INIT,
            initial_temperature_fixed=True,
        )
        r0 = ht.Insulator(ev, name="r0", R=R0_val)
        r1 = ht.Insulator(ev, name="r1", R=R1_val)
        ad.connect(src, "port", r0, "port_a")
        ad.connect(r0, "port_b", c0, "port")
        ad.connect(src, "port", r1, "port_a")
        ad.connect(r1, "port_b", c1, "port")
        ac = AcausalCompiler(ev, ad)
        system = ac()
        builder = jaxonomy.DiagramBuilder()
        s = builder.add(system)
        return builder.build(), s

    return _make


def test_two_cell_thermal_pack_dJdR_matches_fd():
    """AD vs FD on dJ/dR for a 2-cell parallel-cooling pack."""
    make = _two_cell_factory()
    R_base = np.array([0.5, 0.5])
    h = 1e-3

    def _J(R_vec: np.ndarray) -> float:
        diagram, s = make(R_vec)
        ctx = diagram.create_context()
        res = jaxonomy.simulate(
            diagram, ctx, (0.0, _T_SIM), options=_OPTS_BDF,
        )
        xfinal = np.asarray(res.context[s.system_id].continuous_state)
        return float(np.sum((xfinal - _T_REF) ** 2))

    def _fd(i: int) -> float:
        Rp = R_base.copy(); Rp[i] += h
        Rm = R_base.copy(); Rm[i] -= h
        return (_J(Rp) - _J(Rm)) / (2.0 * h)

    dJ_dR_fd = np.array([_fd(0), _fd(1)])

    diagram, s = make(R_base)
    ctx0 = diagram.create_context()
    subctx = ctx0[s.system_id]

    @jax.jit
    def _J_ad(R_vec):
        new_params = dict(subctx.parameters)
        new_params["r0_R"] = R_vec[0]
        new_params["r1_R"] = R_vec[1]
        new_subctx = subctx.with_parameters(new_params)
        new_ctx = ctx0.with_subcontext(s.system_id, new_subctx)
        res = jaxonomy.simulate(
            diagram, new_ctx, (0.0, _T_SIM), options=_OPTS_BDF,
        )
        xfinal = res.context[s.system_id].continuous_state
        return jnp.sum((xfinal - _T_REF) ** 2)

    dJ_dR_ad = np.asarray(jax.grad(_J_ad)(jnp.asarray(R_base)))

    rel_err = np.abs(dJ_dR_ad - dJ_dR_fd) / (np.abs(dJ_dR_fd) + 1e-12)
    assert np.all(rel_err < 0.05), (
        f"AD vs FD disagreement on dJ/dR: FD={dJ_dR_fd!s}, "
        f"AD={dJ_dR_ad!s}, rel_err={rel_err!s}"
    )
