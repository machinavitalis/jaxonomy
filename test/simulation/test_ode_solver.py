# SPDX-License-Identifier: MIT

import pytest
import matplotlib.pyplot as plt

import numpy as np
from scipy.integrate import solve_ivp
import jaxonomy
from jaxonomy.models import EulerRigidBody, ArenstorfOrbit, Lorenz, Pleiades
from jaxonomy.backend import ODESolver, numpy_api as npa
from jaxonomy.testing import set_backend

pytestmark = pytest.mark.minimal

ODE_SOLVERS = ["RK45", "BDF"]


class TestHairerSystems:
    def _run_ode_test(self, system, t_span, method, rtol, atol):
        set_backend("jax")
        context = system.create_context()
        recorded_signals = {"x": system.output_ports[0]}
        options = jaxonomy.SimulatorOptions(
            rtol=rtol,
            atol=atol,
            ode_solver_method=method,
        )
        results = jaxonomy.simulate(
            system,
            context,
            t_span,
            recorded_signals=recorded_signals,
            options=options,
        )
        x = results.outputs["x"]
        t = results.time

        # Compare with the SciPy solution
        # Since we can't directly match time stamps using the `simulate` interface,
        # instead call the SciPy solve with results interpolated at the specified points

        set_backend("numpy")
        context = system.create_context()
        scipy_solver = ODESolver(system)
        scipy_solver.initialize(context)

        def f(t, y):
            return scipy_solver.flat_ode_rhs(y, t, context)

        xc0 = scipy_solver._ravel(context.continuous_state)
        scipy_sol = solve_ivp(
            f, t_span, xc0, atol=atol, rtol=rtol, method=method, t_eval=t
        )
        x_scipy = scipy_sol.y.T

        assert np.allclose(x_scipy, x, rtol=1e-4, atol=1e-4)
        return t, x, x_scipy

    @pytest.mark.parametrize("method", ODE_SOLVERS)
    def test_euler(self, method, show_plot=False):
        # Euler's equation of rotation for a rigid body
        system = EulerRigidBody()
        t_span = (0.0, 20.0)
        t, x, x_scipy = self._run_ode_test(system, t_span, method, rtol=1e-8, atol=1e-6)

        if show_plot:
            fig, axs = plt.subplots(2, 1, figsize=(7, 4), sharex=True)
            axs[0].plot(t, x_scipy, ".-")
            axs[0].set_title("Scipy")
            axs[1].plot(t, x, ".-")
            axs[1].set_title("Jaxonomy")
            plt.show()

    @pytest.mark.parametrize("method", ODE_SOLVERS)
    def test_arenstorf(self, method, show_plot=False):
        # Restricted three-body problem
        system = ArenstorfOrbit()
        t_span = (0.0, 17.0652165601579625588917206249)

        t, x, x_scipy = self._run_ode_test(
            system, t_span, method, rtol=1e-10, atol=1e-12
        )

        if show_plot:
            fig, axs = plt.subplots(figsize=(4, 4), sharex=True)
            axs.plot(x_scipy[:, 0], x_scipy[:, 1], "-", label="Scipy")
            axs.plot(x[:, 0], x[:, 1], "--", label="Jaxonomy")
            plt.show()

    @pytest.mark.parametrize("method", ODE_SOLVERS)
    def test_lorenz(self, method, show_plot=False):
        system = Lorenz()

        # Very chaotic - only compare over short times
        t_span = (0.0, 1.0)
        t, x, x_scipy = self._run_ode_test(
            system, t_span, method, rtol=1e-12, atol=1e-14
        )

        if show_plot:
            fig, axs = plt.subplots(3, 1, figsize=(7, 4), sharex=True)
            for i in range(3):
                axs[i].plot(t, x_scipy[:, i], label="scipy")
                axs[i].plot(t, x[:, i], "--", label="jaxonomy")
            plt.show()

    @pytest.mark.slow
    @pytest.mark.timeout(600)
    def test_pleiades(self, show_plot=False):
        t_span = (0.0, 3.0)
        system = Pleiades()
        t, x, x_scipy = self._run_ode_test(
            system, t_span, "RK45", rtol=1e-8, atol=1e-10
        )

        if show_plot:
            fig, axs = plt.subplots(figsize=(4, 4), sharex=True)
            for i in range(7):
                axs.plot(x_scipy[:, i], x_scipy[:, i + 7], "-", label="Scipy")
                axs.plot(x[:, i], x[:, i + 7], "--", label="Jaxonomy")
            plt.show()


class DivergingODE(jaxonomy.LeafSystem):
    def __init__(self, name="DivergingODE"):
        super().__init__(name=name)
        self.declare_continuous_state(default_value=1.0, ode=self.ode)

    def ode(self, t, state):
        x = state.continuous_state
        return npa.exp(x)


# Runs in a fresh subprocess: this guards the T-005/T-008 anti-hang
# property (a regression hangs uninterruptibly — the 120s subprocess kill
# converts that into a clean failure), and the long diverging adaptive
# kernel slows down unboundedly when tensorflow is resident in the parent
# pytest process (see test/conftest.py's early TF import and the fuller
# rationale in test_v008_solver_stress.TestDivergingSystem).
_DIVERGING_SNIPPET = """
import json
import numpy as np
import jax.numpy as jnp
import jaxonomy
from jaxonomy import LeafSystem
from jaxonomy.backend import numpy_api as npa

class DivergingODE(LeafSystem):
    def __init__(self, name="DivergingODE"):
        super().__init__(name=name)
        self.declare_continuous_state(default_value=1.0, ode=self.ode)

    def ode(self, t, state):
        x = state.continuous_state
        return npa.exp(x)

system = DivergingODE()
context = system.create_context()
options = jaxonomy.SimulatorOptions(ode_solver_method="{method}")
results = jaxonomy.simulate(system, context, (0.0, 0.5), options=options)
print("VERDICT " + json.dumps({{
    "t_final": float(results.context.time),
    "x_finite": bool(np.all(np.isfinite(np.asarray(results.context.continuous_state)))),
}}))
"""


@pytest.mark.parametrize("method", ODE_SOLVERS)
def test_diverging_solution(method):
    """Was skip-quarantined: this used to hang the adaptive inner loop
    (no step-underflow guard), and the RuntimeError it originally asserted
    was removed in T-002b for vmap compatibility. With the T-005/T-008
    termination guards the solve terminates promptly; the divergence
    manifests as a non-finite final state (or early termination), never a
    hang and never a silently-finite answer past the singularity."""
    import json
    import subprocess
    import sys
    from pathlib import Path

    solver = {"RK45": "dopri5", "BDF": "bdf"}[method]
    repo_root = Path(__file__).resolve().parents[2]
    try:
        proc = subprocess.run(
            [sys.executable, "-c", _DIVERGING_SNIPPET.format(method=solver)],
            capture_output=True,
            text=True,
            timeout=120,
            cwd=repo_root,
        )
    except subprocess.TimeoutExpired:
        pytest.fail(
            f"Diverging ODE under {solver} hung for >120s — the T-005/T-008 "
            "solver termination guards have regressed."
        )
    verdict_lines = [
        ln for ln in proc.stdout.splitlines() if ln.startswith("VERDICT ")
    ]
    assert verdict_lines, (
        f"subprocess produced no verdict (rc={proc.returncode}):\n"
        f"stdout:\n{proc.stdout[-2000:]}\nstderr:\n{proc.stderr[-2000:]}"
    )
    verdict = json.loads(verdict_lines[-1][len("VERDICT "):])
    terminated_early = verdict["t_final"] < 0.5 - 1e-9
    assert terminated_early or not verdict["x_finite"], (
        f"{solver}: reached t={verdict['t_final']} with finite state past "
        "the finite-time singularity — silent step clipping."
    )


if __name__ == "__main__":
    # TestHairerSystems().test_euler("rk45", show_plot=True)
    # TestHairerSystems().test_arenstorf(show_plot=True)
    # TestHairerSystems().test_lorenz(show_plot=True)
    # TestHairerSystems().test_pleiades(show_plot=True)
    # test_diverging_solution("RK45")
    test_diverging_solution("BDF")
