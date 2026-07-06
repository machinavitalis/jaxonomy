# SPDX-License-Identifier: MIT

"""
V-012: Real-world model regression suite.

Maintains a corpus of nontrivial models that simulate correctly today.
Reference values for representative scalar quantities are checked in here;
each test runs the canonical short simulation and asserts that current
output matches the reference within ``RTOL``. CI fails on any deviation.

Scope: this campaign uses only models already shipped in
``jaxonomy/models/`` (no new models written here). Categories that don't
yet have an in-repo runnable end-to-end model (battery, EV powertrain,
MPC closed-loop, FMU, MuJoCo, neural-net plant) are explicitly skipped
so the gap remains visible.

Reference values were captured during development by running each model
with the JAX backend (DEFAULT_BACKEND="jax") and the default solver
options. ``RTOL`` is loose enough to absorb tiny floating-point drift
across platforms but tight enough to catch real regressions.
"""

import numpy as np
import pytest

import jaxonomy
from jaxonomy.models import (
    ArenstorfOrbit,
    BouncingBall,
    EulerRigidBody,
    FitzHughNagumo,
    LotkaVolterra,
    Lorenz,
    PendulumDiagram,
    VanDerPol,
)

pytestmark = pytest.mark.slow

# Relative tolerance for reference comparisons. Picked so that ordinary
# floating-point drift between machines won't trip the suite, but a real
# numerical regression in the simulator (e.g. step-size logic, solver
# tableau, event handling) will.
RTOL = 1e-5


# ---------------------------------------------------------------------------
# Reference values (locked in from a clean development run).
#
# Each entry stores three scalars per model:
#   - final_state_norm: L2 norm of the final continuous-state vector
#   - time_count:       number of recorded sample times (integer; exact match)
#   - y_max:            max of recorded output port "y" over the trajectory
#
# These three together cover state evolution, solver step pattern, and
# observable signal magnitude, which is enough to catch the regressions
# this suite is designed to flag.
# ---------------------------------------------------------------------------
REFERENCES = {
    "pendulum_diagram": {
        "final_state_norm": 1.7832340006789238,
        "time_count": 133,
        "y_max": 1.0,
    },
    "lotka_volterra": {
        "final_state_norm": 1.1593213666,
        "time_count": 41,
        "y_max": 10.7781466235,
    },
    "fitzhugh_nagumo": {
        "final_state_norm": 2.0936001060,
        "time_count": 121,
        "y_max": 1.9969096721,
    },
    "van_der_pol": {
        "final_state_norm": 1.7417896540,
        "time_count": 157,
        "y_max": 2.6783435448,
    },
    "lorenz": {
        "final_state_norm": 40.1556366717,
        "time_count": 217,
        "y_max": 42.1239628480,
    },
    "arenstorf_orbit": {
        "final_state_norm": 0.9711695372,
        "time_count": 61,
        "y_max": 1.1420621727,
    },
    "euler_rigid_body": {
        "final_state_norm": 1.3323316040,
        "time_count": 30,
        "y_max": 1.0,
    },
    "bouncing_ball": {
        "final_state_norm": 2.7765846715,
        "time_count": 21,
        "y_max": 10.0,
    },
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _simulate_and_summarize(system, context, t_end):
    """Run a canonical simulation and reduce it to scalar summaries.

    The output port at index 0 is recorded under the name "y". For
    ``LeafSystem``-rooted models this is the continuous-state output; for
    diagram-rooted models it's the exported output port. Either way it
    gives us a stable observable to hash.
    """
    recorded = {"y": system.output_ports[0]}
    result = jaxonomy.simulate(
        system,
        context,
        (0.0, t_end),
        recorded_signals=recorded,
    )

    final_state = np.asarray(result.context.continuous_state)
    y = np.asarray(result.outputs["y"])
    return {
        "final_state_norm": float(np.linalg.norm(final_state)),
        "time_count": int(len(result.time)),
        "y_max": float(np.max(y)),
    }


def _assert_matches_reference(name, summary):
    ref = REFERENCES[name]
    # time_count is an integer count of sample points; require exact match
    # (a drift here means the solver took a different number of steps,
    # which is itself a regression worth flagging).
    assert summary["time_count"] == ref["time_count"], (
        f"{name}: time_count drifted "
        f"(got {summary['time_count']}, expected {ref['time_count']})"
    )

    np.testing.assert_allclose(
        summary["final_state_norm"],
        ref["final_state_norm"],
        rtol=RTOL,
        err_msg=f"{name}: final_state_norm regression",
    )
    np.testing.assert_allclose(
        summary["y_max"],
        ref["y_max"],
        rtol=RTOL,
        err_msg=f"{name}: y_max regression",
    )


# ---------------------------------------------------------------------------
# Module-scoped fixtures: build each model + context once and reuse.
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def pendulum_system():
    sys = PendulumDiagram(x0=[1.0, 0.0], m=1.0, g=9.81, L=1.0, b=0.1)
    return sys, sys.create_context()


@pytest.fixture(scope="module")
def lotka_volterra_system():
    sys = LotkaVolterra()
    return sys, sys.create_context()


@pytest.fixture(scope="module")
def fitzhugh_nagumo_system():
    sys = FitzHughNagumo()
    return sys, sys.create_context()


@pytest.fixture(scope="module")
def van_der_pol_system():
    sys = VanDerPol(x0=[1.0, 0.0])
    return sys, sys.create_context()


@pytest.fixture(scope="module")
def lorenz_system():
    sys = Lorenz()
    return sys, sys.create_context()


@pytest.fixture(scope="module")
def arenstorf_system():
    sys = ArenstorfOrbit()
    return sys, sys.create_context()


@pytest.fixture(scope="module")
def euler_rigid_body_system():
    sys = EulerRigidBody()
    return sys, sys.create_context()


@pytest.fixture(scope="module")
def bouncing_ball_system():
    sys = BouncingBall(g=10.0, e=0.7)
    ctx = sys.create_context().with_continuous_state(np.array([10.0, 0.0]))
    return sys, ctx


# ---------------------------------------------------------------------------
# Regression cases: in-repo models that simulate end-to-end today.
# ---------------------------------------------------------------------------
def test_regression_pendulum_diagram(pendulum_system):
    """Damped pendulum diagram (composed of primitives)."""
    sys, ctx = pendulum_system
    summary = _simulate_and_summarize(sys, ctx, t_end=10.0)
    _assert_matches_reference("pendulum_diagram", summary)


def test_regression_lotka_volterra(lotka_volterra_system):
    """Predator-prey ODE."""
    sys, ctx = lotka_volterra_system
    summary = _simulate_and_summarize(sys, ctx, t_end=10.0)
    _assert_matches_reference("lotka_volterra", summary)


def test_regression_fitzhugh_nagumo(fitzhugh_nagumo_system):
    """Excitable-membrane ODE (stiff-ish neural model)."""
    sys, ctx = fitzhugh_nagumo_system
    summary = _simulate_and_summarize(sys, ctx, t_end=50.0)
    _assert_matches_reference("fitzhugh_nagumo", summary)


def test_regression_van_der_pol(van_der_pol_system):
    """Van der Pol oscillator (limit cycle)."""
    sys, ctx = van_der_pol_system
    summary = _simulate_and_summarize(sys, ctx, t_end=20.0)
    _assert_matches_reference("van_der_pol", summary)


def test_regression_lorenz(lorenz_system):
    """Lorenz attractor (chaotic ODE; Hairer benchmark)."""
    sys, ctx = lorenz_system
    summary = _simulate_and_summarize(sys, ctx, t_end=5.0)
    _assert_matches_reference("lorenz", summary)


def test_regression_arenstorf_orbit(arenstorf_system):
    """Restricted three-body Arenstorf orbit (Hairer benchmark)."""
    sys, ctx = arenstorf_system
    summary = _simulate_and_summarize(sys, ctx, t_end=5.0)
    _assert_matches_reference("arenstorf_orbit", summary)


def test_regression_euler_rigid_body(euler_rigid_body_system):
    """Euler's rigid body equations with time-windowed forcing."""
    sys, ctx = euler_rigid_body_system
    summary = _simulate_and_summarize(sys, ctx, t_end=5.0)
    _assert_matches_reference("euler_rigid_body", summary)


def test_regression_bouncing_ball(bouncing_ball_system):
    """Hybrid system: ball with restitution-event reset."""
    sys, ctx = bouncing_ball_system
    summary = _simulate_and_summarize(sys, ctx, t_end=5.0)
    _assert_matches_reference("bouncing_ball", summary)


# ---------------------------------------------------------------------------
# Explicitly-skipped categories.
#
# These are domains the V-012 corpus is meant to cover long-term, but the
# repo doesn't yet ship a ready-to-instantiate end-to-end model for any
# of them. We skip with a visible reason so the gap is tracked in CI
# rather than silently ignored. As soon as a corresponding model lands,
# the relevant test should be replaced with a real regression case.
# ---------------------------------------------------------------------------
def test_regression_battery_pack():
    pytest.skip(
        "battery-pack closed-loop scenario not yet in repo: "
        "T-XX (jaxonomy.models.Battery exists as a cell ECM, but no "
        "pack-level driving-cycle model is shipped)."
    )


def test_regression_ev_powertrain():
    pytest.skip(
        "full EV powertrain regression not yet in repo: T-XX "
        "(jaxonomy.models.CompactEV exists but no canonical drive-cycle "
        "harness is shipped to lock reference outputs against)."
    )


def test_regression_mpc_closed_loop():
    pytest.skip(
        "MPC / NMPC closed-loop variant regression not yet in repo: "
        "T-XX (trajopt primitives exist in jaxonomy.library.nmpc, but no "
        "standalone canonical MPC scenario model is shipped)."
    )


def test_regression_fmu_import():
    pytest.skip(
        "FMU-imported model regression not yet in repo: T-XX "
        "(no checked-in FMU asset to load end-to-end)."
    )


def test_regression_mujoco_bridge():
    pytest.skip(
        "MuJoCo-bridge model regression not yet in repo: T-XX "
        "(no MuJoCo-backed plant shipped under jaxonomy.models)."
    )


def test_regression_neural_net_plant():
    pytest.skip(
        "neural-network surrogate / learned-plant regression not yet "
        "in repo: T-XX (MLP primitives exist but no end-to-end NN-plant "
        "scenario model is shipped)."
    )
