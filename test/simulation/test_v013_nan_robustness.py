# SPDX-License-Identifier: MIT

"""V-013: NaN / Inf robustness of the simulation loop.

A NaN can enter a simulation trivially — a NaN parameter from bad input
data, a 0/0 in a user callback — and what the engine does next differs
sharply by solver (verified empirically 2026-07-09):

- **Fixed-step rk4** completes the horizon and propagates NaN through the
  state, exactly what a user debugging bad data wants. Tested below.
- **Adaptive dopri5/bdf** used to hang forever (NaN error norm → the
  accept-step test never passes; no NaN guard — the T-005/T-008 root gap).
  Both solvers now carry NaN / step-underflow termination guards in their
  accept loops (dopri5.attempt_rk_step, bdf.attempt_bdf_step): the step is
  force-accepted, the non-finite state propagates, and the simulation
  terminates promptly instead of spinning.
"""

import numpy as np
import pytest

import jaxonomy
from jaxonomy.library import Constant, Integrator
from jaxonomy.simulation import SimulatorOptions

pytestmark = pytest.mark.minimal


def _source_driven_integrator(value):
    builder = jaxonomy.DiagramBuilder()
    src = builder.add(Constant(value))
    integ = builder.add(Integrator(initial_state=0.0))
    builder.connect(src.output_ports[0], integ.input_ports[0])
    return builder.build(), integ


_RK4 = dict(ode_solver_method="rk4", max_minor_step_size=0.01)


class TestFixedStepNaNPropagation:
    def test_nan_source_completes_and_propagates(self):
        """rk4 must reach tf and yield NaN state — not hang, not raise."""
        diagram, integ = _source_driven_integrator(float("nan"))
        ctx = diagram.create_context()
        results = jaxonomy.simulate(
            diagram,
            ctx,
            (0.0, 1.0),
            options=SimulatorOptions(**_RK4),
            recorded_signals={"x": integ.output_ports[0]},
        )
        x = np.asarray(results.outputs["x"])
        assert float(results.time[-1]) == pytest.approx(1.0)
        # everything after the initial sample is poisoned
        assert np.isnan(x[1:]).all()

    def test_nan_initial_state_completes_and_propagates(self):
        diagram, integ = _source_driven_integrator(1.0)
        ctx = diagram.create_context()
        ctx = ctx.with_subcontext(
            integ.system_id,
            ctx[integ.system_id].with_continuous_state(np.float64("nan")),
        )
        results = jaxonomy.simulate(
            diagram,
            ctx,
            (0.0, 1.0),
            options=SimulatorOptions(**_RK4),
            recorded_signals={"x": integ.output_ports[0]},
        )
        assert float(results.time[-1]) == pytest.approx(1.0)
        assert np.isnan(np.asarray(results.outputs["x"])).all()

    def test_inf_source_completes(self):
        """+inf drive: rk4 completes; state is inf (or NaN once inf-inf
        arithmetic appears). Either is fine — the contract is completion."""
        diagram, integ = _source_driven_integrator(float("inf"))
        ctx = diagram.create_context()
        results = jaxonomy.simulate(
            diagram,
            ctx,
            (0.0, 1.0),
            options=SimulatorOptions(**_RK4),
            recorded_signals={"x": integ.output_ports[0]},
        )
        x = np.asarray(results.outputs["x"])
        assert float(results.time[-1]) == pytest.approx(1.0)
        assert not np.isfinite(x[-1])


@pytest.mark.parametrize("method", ["dopri5", "bdf"])
def test_adaptive_solver_nan_source_terminates(method):
    """Was skip-quarantined: a NaN in the RHS made the adaptive accept-step
    loop spin forever (NaN error norm never accepted, no NaN guard — same
    root gap as the diverging-ODE hangs, T-005/T-008). The solvers now
    force-accept on a NaN error norm, so the simulation *terminates* with the
    non-finite state visible to the caller. The trajectory is truncated at
    the point NaN struck (dopri5) or jumped to the boundary (bdf) — the
    contract asserted here is prompt termination + a non-finite final state,
    not a full recording."""
    diagram, integ = _source_driven_integrator(float("nan"))
    ctx = diagram.create_context()
    results = jaxonomy.simulate(
        diagram,
        ctx,
        (0.0, 1.0),
        options=SimulatorOptions(ode_solver_method=method),
        recorded_signals={"x": integ.output_ports[0]},
    )
    xf = float(results.context[integ.system_id].continuous_state)
    assert np.isnan(xf), f"expected NaN final state, got {xf}"
