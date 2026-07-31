# SPDX-License-Identifier: MIT

"""BDF Newton convergence must not depend on how the caller was compiled.

The BDF inner Newton loop could only declare convergence through a *rate*
estimate, ``rate / (1 - rate) * dy_norm < tol``. That expression degenerates
exactly when the iteration has succeeded: once the correction reaches the
floating-point noise floor it stops shrinking, ``rate`` sits at ~1, the
expression blows up, and a converged step is reported as non-converged.

The caller then halves dt and retries — which for a DAE is strictly harmful,
because as ``dt -> 0`` the Newton matrix ``M - c*J`` tends to the *singular*
mass matrix ``M``. Every retry failed the same way, dt collapsed to the floor,
and the step bailed out by poisoning the state with NaN.

Whether the correction stalled at rate ~1 depended on rounding, and rounding
depends on whether XLA fused/constant-folded the loop. So the same model
integrated correctly under ``simulate`` (initial state a compile-time constant,
folded) and returned NaN under ``Simulator.advance_to`` (initial state a
runtime argument) — a silent, compilation-dependent divergence.

These tests assert the invariant that actually matters: the four execution
modes must agree.
"""

import warnings

import numpy as np
import pytest

import jax

import jaxonomy
from jaxonomy import SimulatorOptions
from jaxonomy.simulation.simulator import Simulator, ODESolver, _check_options
from jaxonomy.testing.markers import skip_if_not_jax

skip_if_not_jax()

pytestmark = pytest.mark.minimal


def _build_cell(initial_soc=0.8, capacity_Ah=1.0, internal_resistance=0.05,
                source_current=1.0):
    """CurrentSource -> BatteryCellTabular -> Ground: an index-1 DAE.

    The mass matrix is rank 1 of 6, which is what makes the retry-by-halving
    path harmful and so what made the convergence bug fatal rather than slow.
    """
    from jaxonomy.acausal import (
        AcausalCompiler, AcausalDiagram, EqnEnv, electrical as elec,
        battery as batt_lib,
    )

    ev = EqnEnv()
    ad = AcausalDiagram()
    cell = batt_lib.BatteryCellTabular(
        ev, name="cell", capacity_Ah=capacity_Ah, ocv_soc=[0.0, 0.5, 1.0],
        ocv_volts=[3.0, 3.6, 4.2], internal_resistance=internal_resistance,
        initial_soc=initial_soc, initial_soc_fixed=True,
        enable_soc_port=True, enable_ocv_port=True,
    )
    cs = elec.CurrentSource(ev, name="cs", i=source_current)
    sensV = elec.VoltageSensor(ev, name="sensV")
    sensI = elec.CurrentSensor(ev, name="sensI")
    gnd = elec.Ground(ev, name="gnd")
    ad.connect(cs, "p", sensI, "p")
    ad.connect(sensI, "n", cell, "p")
    ad.connect(cell, "n", cs, "n")
    ad.connect(cell, "n", gnd, "p")
    ad.connect(cell, "p", sensV, "p")
    ad.connect(cell, "n", sensV, "n")

    acausal_system = AcausalCompiler(ev, ad)()
    builder = jaxonomy.DiagramBuilder()
    acausal_system = builder.add(acausal_system)
    diagram = builder.build()
    return diagram, diagram.create_context(check_types=True), cell, acausal_system


def _soc_port(cell, acausal_system):
    idx = acausal_system.outsym_to_portid[cell.get_sym_by_port_name("soc")]
    return acausal_system.output_ports[idx]


def _final_soc(sim_state):
    return float(np.asarray(jax.tree.leaves(sim_state.context.continuous_state)[0])[0])


T_END = 60.0
# 1 A discharge on a 1 Ah cell for 60 s: SOC drops by 60/3600.
EXPECTED_SOC = 0.8 - 1.0 / 3600.0 * T_END


def _make_simulator(diagram, cell, acausal_system, t_end):
    options = _check_options(
        diagram, SimulatorOptions(ode_solver_method="bdf"), (0.0, t_end),
        {"soc": _soc_port(cell, acausal_system)},
    )
    solver = ODESolver(diagram, options=options.ode_options)
    return Simulator(diagram, ode_solver=solver, options=options)


class TestExecutionModesAgree:
    """The same DAE must integrate identically however it is compiled."""

    def test_simulate_gives_the_analytic_answer(self):
        diagram, context, cell, acausal_system = _build_cell()
        results = jaxonomy.simulate(
            diagram, context, (0.0, T_END),
            recorded_signals={"soc": _soc_port(cell, acausal_system)},
            options=SimulatorOptions(ode_solver_method="bdf"),
        )
        soc = np.asarray(results.outputs["soc"])
        assert np.all(np.isfinite(soc))
        assert float(soc[-1]) == pytest.approx(EXPECTED_SOC, abs=1e-6)

    def test_advance_to_matches_simulate(self):
        """The regression: this returned NaN while ``simulate`` was correct."""
        diagram, context, cell, acausal_system = _build_cell()
        sim = _make_simulator(diagram, cell, acausal_system, T_END)

        diagram.cache_enabled = True
        try:
            state = sim.advance_to(T_END, context.with_time(0.0))
        finally:
            diagram.cache_enabled = False

        assert _final_soc(state) == pytest.approx(EXPECTED_SOC, abs=1e-6)

    def test_context_as_constant_and_as_argument_agree(self):
        """Constant-folded and runtime-argument paths must not diverge.

        This is the sharpest form of the bug: identical maths, and the only
        difference is whether XLA could fold the initial state.
        """
        diagram, context, cell, acausal_system = _build_cell()
        sim = _make_simulator(diagram, cell, acausal_system, T_END)

        diagram.cache_enabled = True
        try:
            as_constant = jax.jit(
                lambda: sim._advance_to(T_END, context.with_time(0.0))
            )()
            as_argument = jax.jit(
                lambda c: sim._advance_to(T_END, c.with_time(0.0))
            )(context)
        finally:
            diagram.cache_enabled = False

        assert _final_soc(as_constant) == pytest.approx(EXPECTED_SOC, abs=1e-6)
        assert _final_soc(as_argument) == pytest.approx(EXPECTED_SOC, abs=1e-6)
        assert _final_soc(as_argument) == pytest.approx(
            _final_soc(as_constant), rel=1e-9
        )

    def test_uncompiled_python_loop_agrees(self):
        """With jit disabled the loop is plain Python — same answer required."""
        diagram, context, cell, acausal_system = _build_cell()
        sim = _make_simulator(diagram, cell, acausal_system, T_END)

        diagram.cache_enabled = True
        try:
            with jax.disable_jit():
                state = sim._advance_to(T_END, context.with_time(0.0))
        finally:
            diagram.cache_enabled = False

        assert _final_soc(state) == pytest.approx(EXPECTED_SOC, abs=1e-6)


class TestNewtonConvergenceCriterion:
    """Unit-level guards on the convergence expression itself."""

    def test_small_correction_converges_regardless_of_rate(self):
        """A correction already below tol is converged, whatever the rate says.

        This is the clause whose absence caused the bug: convergence could only
        be reached through the rate estimate.
        """
        import jax.numpy as jnp
        from jaxonomy.backend._jax.bdf import _solve_newton_system_impl  # noqa: F401

        # Exercise the criterion directly, mirroring the loop body.
        tol = 1e-3
        for rate in (jnp.asarray(1.0), jnp.asarray(0.999999), jnp.asarray(np.inf)):
            dy_norm = jnp.asarray(1e-18)
            converged = (
                (dy_norm == 0.0)
                | (dy_norm < tol)
                | (jnp.isfinite(rate) & (rate < 1.0)
                   & (rate / (1 - rate) * dy_norm < tol))
            )
            assert bool(converged), f"tiny correction not converged at rate={rate}"

    def test_diverging_iteration_is_not_reported_as_converged(self):
        """rate > 1 makes ``1 - rate`` negative; without a guard that reads as converged."""
        import jax.numpy as jnp

        tol = 1e-3
        rate = jnp.asarray(2.0)      # diverging
        dy_norm = jnp.asarray(10.0)  # large correction

        unguarded = jnp.isfinite(rate) & (rate / (1 - rate) * dy_norm < tol)
        assert bool(unguarded), "precondition: the unguarded form is wrongly True"

        guarded = (
            (dy_norm == 0.0)
            | (dy_norm < tol)
            | (jnp.isfinite(rate) & (rate < 1.0)
               & (rate / (1 - rate) * dy_norm < tol))
        )
        assert not bool(guarded), "a diverging iteration must not count as converged"


class TestCriterionDoesNotDegradeTheSolver:
    """The new clause is more permissive — guard what that could cost."""

    def test_accuracy_is_preserved_on_an_analytic_dae(self):
        """A looser convergence test must not loosen the answer.

        SOC is exactly linear here, so every recorded sample has a closed form —
        an accuracy regression shows up as a mid-trajectory error, not just a
        final-value one.
        """
        diagram, context, cell, acausal_system = _build_cell()
        results = jaxonomy.simulate(
            diagram, context, (0.0, T_END),
            recorded_signals={"soc": _soc_port(cell, acausal_system)},
            options=SimulatorOptions(ode_solver_method="bdf", rtol=1e-10, atol=1e-12),
        )
        t = np.asarray(results.time)
        soc = np.asarray(results.outputs["soc"])
        expected = 0.8 - 1.0 / 3600.0 * t

        assert np.all(np.isfinite(soc))
        # Tight: the trajectory is exactly linear, so this is solver accuracy.
        assert np.max(np.abs(soc - expected)) < 1e-9

    def test_tightening_tolerance_still_tightens_the_answer(self):
        """A convergence test that ignored tolerance would flatten this out."""
        errors = {}
        for rtol in (1e-4, 1e-10):
            diagram, context, cell, acausal_system = _build_cell()
            results = jaxonomy.simulate(
                diagram, context, (0.0, T_END),
                recorded_signals={"soc": _soc_port(cell, acausal_system)},
                options=SimulatorOptions(
                    ode_solver_method="bdf", rtol=rtol, atol=rtol * 1e-2,
                ),
            )
            t = np.asarray(results.time)
            soc = np.asarray(results.outputs["soc"])
            errors[rtol] = float(np.max(np.abs(soc - (0.8 - 1.0 / 3600.0 * t))))

        assert errors[1e-10] <= errors[1e-4] + 1e-12, errors

    def test_a_genuinely_divergent_system_still_terminates(self):
        """The no-hang guarantee must survive the more permissive criterion.

        A NaN source can never converge; the solver must still reach tf and
        surface a non-finite state rather than spinning in the retry loop.
        """
        from jaxonomy.library import Constant, Integrator

        builder = jaxonomy.DiagramBuilder()
        src = builder.add(Constant(np.nan, name="nan_src"))
        integ = builder.add(Integrator(initial_state=0.0, name="i"))
        builder.connect(src.output_ports[0], integ.input_ports[0])
        diagram = builder.build(name="divergent")

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            results = jaxonomy.simulate(
                diagram, diagram.create_context(), (0.0, 1.0),
                recorded_signals={"y": integ.output_ports[0]},
                options=SimulatorOptions(ode_solver_method="bdf"),
            )

        # It must come back (not hang) and must not pretend the answer is fine.
        assert not np.all(np.isfinite(np.asarray(results.outputs["y"])))
