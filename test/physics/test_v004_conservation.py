# SPDX-License-Identifier: MIT
"""V-004: Conservation law preservation.

Closed conservative systems must preserve their conserved quantities (energy,
angular momentum) within solver tolerance over long horizons. Each test:

  1. Builds a `LeafSystem` with a continuous-state ODE.
  2. Simulates over many natural periods.
  3. Records the state trajectory (recorded_signals).
  4. Computes the conserved quantity along the trajectory.
  5. Asserts max relative drift |q_t - q_0| / |q_0| stays below a documented
     bound depending on solver:

        Dopri5 (rtol=1e-8, atol=1e-10):  drift < 1e-3
        BDF    (rtol=1e-8, atol=1e-10):  drift < 1e-2

Note: Dopri5 is non-symplectic, so secular drift over very long horizons is
expected even at tight tolerances; the bound below is intentionally generous.
"""

import jax.numpy as jnp
import numpy as np
import pytest

import jaxonomy
from jaxonomy import LeafSystem, SimulatorOptions
from jaxonomy.testing.markers import skip_if_not_jax

skip_if_not_jax()
pytestmark = pytest.mark.slow


# ── helpers ──────────────────────────────────────────────────────────────────


def _opts(method: str) -> SimulatorOptions:
    return SimulatorOptions(
        math_backend="jax",
        ode_solver_method=method,
        rtol=1e-8,
        atol=1e-10,
        save_time_series=True,
    )


def _drift_bound(method: str) -> float:
    return 1e-3 if method == "dopri5" else 1e-2


def _max_rel_drift(q: np.ndarray) -> float:
    q = np.asarray(q)
    q0 = q[0]
    if abs(q0) < 1e-12:
        return float(np.max(np.abs(q - q0)))
    return float(np.max(np.abs(q - q0) / np.abs(q0)))


SOLVERS = ["dopri5", "bdf"]


# ═══════════════════════════════════════════════════════════════════════════
# 1. Simple harmonic oscillator
# ═══════════════════════════════════════════════════════════════════════════


@pytest.mark.parametrize("method", SOLVERS)
def test_harmonic_oscillator_energy(method):
    """SHO: dx/dt = v, dv/dt = -ω²x. E = 0.5 v² + 0.5 ω² x².

    50 natural periods at ω=2 ⇒ T_end ≈ 50·π.
    """
    omega = 2.0

    class SHO(LeafSystem):
        def __init__(self):
            super().__init__()
            self.declare_dynamic_parameter("omega", omega)
            self.declare_continuous_state(
                default_value=jnp.array([1.0, 0.0]), ode=self._ode
            )
            self.declare_output_port(self._out, default_value=jnp.zeros(2))

        def _ode(self, time, state, **params):
            x, v = state.continuous_state
            return jnp.array([v, -(params["omega"] ** 2) * x])

        def _out(self, time, state, **params):
            return state.continuous_state

    sys = SHO()
    ctx = sys.create_context()
    t_end = 50.0 * 2.0 * np.pi / omega
    res = jaxonomy.simulate(
        sys, ctx, (0.0, t_end), options=_opts(method),
        recorded_signals={"state": sys.output_ports[0]},
    )

    xs = np.asarray(res.outputs["state"])
    E = 0.5 * xs[:, 1] ** 2 + 0.5 * omega**2 * xs[:, 0] ** 2
    drift = _max_rel_drift(E)
    assert drift < _drift_bound(method), f"{method} energy drift={drift:.3e}"


# ═══════════════════════════════════════════════════════════════════════════
# 2. Undamped pendulum
# ═══════════════════════════════════════════════════════════════════════════


@pytest.mark.parametrize("method", SOLVERS)
def test_pendulum_energy(method):
    """Pendulum: θ'' = -(g/L) sin θ. E = 0.5 v² + (g/L)(1 - cos θ).

    Small-angle initial condition gives period ≈ 2π√(L/g). Run 50 periods.
    """
    g, L = 9.81, 1.0

    class Pendulum(LeafSystem):
        def __init__(self):
            super().__init__()
            self.declare_dynamic_parameter("g_over_L", g / L)
            self.declare_continuous_state(
                default_value=jnp.array([0.3, 0.0]), ode=self._ode
            )
            self.declare_output_port(self._out, default_value=jnp.zeros(2))

        def _ode(self, time, state, **params):
            theta, v = state.continuous_state
            return jnp.array([v, -params["g_over_L"] * jnp.sin(theta)])

        def _out(self, time, state, **params):
            return state.continuous_state

    sys = Pendulum()
    ctx = sys.create_context()
    period = 2 * np.pi * np.sqrt(L / g)
    t_end = 50.0 * period
    res = jaxonomy.simulate(
        sys, ctx, (0.0, t_end), options=_opts(method),
        recorded_signals={"state": sys.output_ports[0]},
    )

    xs = np.asarray(res.outputs["state"])
    theta, v = xs[:, 0], xs[:, 1]
    E = 0.5 * v**2 + (g / L) * (1.0 - np.cos(theta))
    drift = _max_rel_drift(E)
    assert drift < _drift_bound(method), f"{method} energy drift={drift:.3e}"


# ═══════════════════════════════════════════════════════════════════════════
# 3. Two-body Kepler orbit
# ═══════════════════════════════════════════════════════════════════════════


@pytest.mark.parametrize("method", SOLVERS)
def test_kepler_orbit_energy_and_angular_momentum(method):
    """Planar Kepler: r'' = -r/|r|³, normalized G·M = 1.

    Conserved: total energy E = 0.5 |v|² - 1/|r|, angular momentum L = x·vy - y·vx.
    Initial (1, 0) and v=(0, 1) gives a circular orbit, period 2π.
    """

    class Kepler(LeafSystem):
        def __init__(self):
            super().__init__()
            self.declare_continuous_state(
                default_value=jnp.array([1.0, 0.0, 0.0, 1.0]), ode=self._ode
            )
            self.declare_output_port(self._out, default_value=jnp.zeros(4))

        def _ode(self, time, state, **params):
            x, y, vx, vy = state.continuous_state
            r3 = (x * x + y * y) ** 1.5
            return jnp.array([vx, vy, -x / r3, -y / r3])

        def _out(self, time, state, **params):
            return state.continuous_state

    sys = Kepler()
    ctx = sys.create_context()
    t_end = 50.0 * 2.0 * np.pi  # 50 periods
    res = jaxonomy.simulate(
        sys, ctx, (0.0, t_end), options=_opts(method),
        recorded_signals={"state": sys.output_ports[0]},
    )

    xs = np.asarray(res.outputs["state"])
    x, y, vx, vy = xs[:, 0], xs[:, 1], xs[:, 2], xs[:, 3]
    r = np.sqrt(x * x + y * y)
    E = 0.5 * (vx * vx + vy * vy) - 1.0 / r
    Lz = x * vy - y * vx

    e_drift = _max_rel_drift(E)
    l_drift = _max_rel_drift(Lz)
    bound = _drift_bound(method)
    assert e_drift < bound, f"{method} energy drift={e_drift:.3e}"
    assert l_drift < bound, f"{method} L_z drift={l_drift:.3e}"


# ═══════════════════════════════════════════════════════════════════════════
# 4. LC circuit (no resistance)
# ═══════════════════════════════════════════════════════════════════════════


@pytest.mark.parametrize("method", SOLVERS)
def test_lc_circuit_energy(method):
    """Lossless LC: L·di/dt = -q/C, dq/dt = i. E = 0.5·L·i² + 0.5·q²/C.

    State is [q, i]. Natural frequency ω = 1/√(LC).
    """
    L_ind, C = 1.0, 1.0  # ω = 1 rad/s

    class LC(LeafSystem):
        def __init__(self):
            super().__init__()
            self.declare_dynamic_parameter("L", L_ind)
            self.declare_dynamic_parameter("C", C)
            self.declare_continuous_state(
                default_value=jnp.array([1.0, 0.0]), ode=self._ode
            )
            self.declare_output_port(self._out, default_value=jnp.zeros(2))

        def _ode(self, time, state, **params):
            q, i = state.continuous_state
            return jnp.array([i, -q / (params["L"] * params["C"])])

        def _out(self, time, state, **params):
            return state.continuous_state

    sys = LC()
    ctx = sys.create_context()
    omega = 1.0 / np.sqrt(L_ind * C)
    t_end = 50.0 * 2.0 * np.pi / omega
    res = jaxonomy.simulate(
        sys, ctx, (0.0, t_end), options=_opts(method),
        recorded_signals={"state": sys.output_ports[0]},
    )

    xs = np.asarray(res.outputs["state"])
    q, i = xs[:, 0], xs[:, 1]
    E = 0.5 * L_ind * i**2 + 0.5 * q**2 / C
    drift = _max_rel_drift(E)
    assert drift < _drift_bound(method), f"{method} energy drift={drift:.3e}"


# ═══════════════════════════════════════════════════════════════════════════
# 5. Frictionless mass on incline (1-D)
# ═══════════════════════════════════════════════════════════════════════════


@pytest.mark.parametrize("method", SOLVERS)
def test_incline_energy(method):
    """A point mass slides frictionlessly on a 1-D incline of angle α.

    State along the slope: ds/dt = v, dv/dt = -g sin α. Height h = s sin α.
    E = 0.5 v² + g h. With v0 > 0, the mass decelerates, stops, returns;
    motion is bounded (oscillatory) because the slope is treated as a 1-D
    line with a turning point at s=0 (we wrap into a finite track by
    choosing v0 small enough that it reverses naturally — purely 1-D
    decel/accel).

    We integrate over a finite range where v reverses; over many cycles
    using a parabolic motion, energy is conserved exactly under exact ODE.
    """
    g = 9.81
    alpha = np.deg2rad(30.0)
    g_par = g * np.sin(alpha)  # along-slope deceleration when moving up

    class Incline(LeafSystem):
        def __init__(self):
            super().__init__()
            self.declare_dynamic_parameter("g_par", g_par)
            # Start at s=0 with v0 upslope; motion is one-up-one-down cycle.
            self.declare_continuous_state(
                default_value=jnp.array([0.0, 5.0]), ode=self._ode
            )
            self.declare_output_port(self._out, default_value=jnp.zeros(2))

        def _ode(self, time, state, **params):
            s, v = state.continuous_state
            return jnp.array([v, -params["g_par"]])

        def _out(self, time, state, **params):
            return state.continuous_state

    sys = Incline()
    ctx = sys.create_context()
    # Apex at t* = v0/g_par; full cycle 2 t*. Run many cycles equivalently
    # (system is unbounded since gravity always pulls down; energy still
    # constant because no dissipation).
    v0 = 5.0
    t_cycle = 2.0 * v0 / g_par
    t_end = 50.0 * t_cycle
    res = jaxonomy.simulate(
        sys, ctx, (0.0, t_end), options=_opts(method),
        recorded_signals={"state": sys.output_ports[0]},
    )

    xs = np.asarray(res.outputs["state"])
    s, v = xs[:, 0], xs[:, 1]
    h = s * np.sin(alpha)
    E = 0.5 * v**2 + g * h
    drift = _max_rel_drift(E)
    assert drift < _drift_bound(method), f"{method} energy drift={drift:.3e}"


# ═══════════════════════════════════════════════════════════════════════════
# 6. Free rigid body in zero gravity (Euler equations)
# ═══════════════════════════════════════════════════════════════════════════


@pytest.mark.parametrize("method", SOLVERS)
def test_rigid_body_angular_momentum(method):
    """Torque-free rigid body, principal axes:

        I1 ω1' = (I2 - I3) ω2 ω3
        I2 ω2' = (I3 - I1) ω3 ω1
        I3 ω3' = (I1 - I2) ω1 ω2

    Conserved (in body frame): |L|² = (I1 ω1)² + (I2 ω2)² + (I3 ω3)².
    Also kinetic energy 0.5·(I1 ω1² + I2 ω2² + I3 ω3²) — we check L.
    """
    I1, I2, I3 = 1.0, 2.0, 3.0

    class RigidBody(LeafSystem):
        def __init__(self):
            super().__init__()
            self.declare_dynamic_parameter("I1", I1)
            self.declare_dynamic_parameter("I2", I2)
            self.declare_dynamic_parameter("I3", I3)
            self.declare_continuous_state(
                default_value=jnp.array([1.0, 0.5, 0.2]), ode=self._ode
            )
            self.declare_output_port(self._out, default_value=jnp.zeros(3))

        def _ode(self, time, state, **params):
            w1, w2, w3 = state.continuous_state
            i1, i2, i3 = params["I1"], params["I2"], params["I3"]
            return jnp.array(
                [
                    (i2 - i3) * w2 * w3 / i1,
                    (i3 - i1) * w3 * w1 / i2,
                    (i1 - i2) * w1 * w2 / i3,
                ]
            )

        def _out(self, time, state, **params):
            return state.continuous_state

    sys = RigidBody()
    ctx = sys.create_context()
    t_end = 200.0  # long horizon; tumbling motion has no fixed period
    res = jaxonomy.simulate(
        sys, ctx, (0.0, t_end), options=_opts(method),
        recorded_signals={"omega": sys.output_ports[0]},
    )

    ws = np.asarray(res.outputs["omega"])
    L1 = I1 * ws[:, 0]
    L2 = I2 * ws[:, 1]
    L3 = I3 * ws[:, 2]
    Lmag = np.sqrt(L1 * L1 + L2 * L2 + L3 * L3)
    drift = _max_rel_drift(Lmag)
    assert drift < _drift_bound(method), f"{method} |L| drift={drift:.3e}"
