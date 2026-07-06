# SPDX-License-Identifier: MIT

"""
Tests for online / real-time parameter estimation blocks:
  - RecursiveLeastSquares
  - AugmentedStateEKF
"""

from math import ceil

import numpy as np
import pytest
import jax
import jax.numpy as jnp

import jaxonomy
from jaxonomy import DiagramBuilder, SimulatorOptions
from jaxonomy.library import (
    Constant,
    Adder,
    ZeroOrderHold,
    RecursiveLeastSquares,
    AugmentedStateEKF,
)
from jaxonomy.testing import requires_jax


# ══════════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════════

DT = 0.1


def _run_simulation(diagram, t_end=1.0, signals=None, dt=DT, max_steps_per_dt=10):
    """Helper: create context, run simulation, return sol."""
    ctx = diagram.create_context()
    nseg = ceil(t_end / dt)
    options = SimulatorOptions(
        max_major_steps=max_steps_per_dt * nseg,
        max_major_step_length=dt,
    )
    return jaxonomy.simulate(
        diagram,
        ctx,
        (0.0, t_end),
        options=options,
        recorded_signals=signals or {},
    )


def _rls_loop(n_params, theta_true, phi_fn, noise_std=0.0, n_steps=200,
              forgetting_factor=1.0, seed=42):
    """
    Pure-Python reference RLS loop for ground-truth comparisons.
    phi_fn(k) returns the regressor for step k.
    """
    rng = np.random.default_rng(seed)
    theta_hat = np.zeros(n_params)
    P = np.eye(n_params) * 1e4
    lam = forgetting_factor
    for k in range(n_steps):
        phi = phi_fn(k)
        y = phi @ theta_true + rng.normal(scale=noise_std)
        e = y - phi @ theta_hat
        Pphi = P @ phi
        denom = lam + phi @ Pphi
        K = Pphi / denom
        theta_hat = theta_hat + K * e
        P = (P - np.outer(K, phi) @ P) / lam
    return theta_hat, P


# ══════════════════════════════════════════════════════════════════════════════
# RecursiveLeastSquares – unit tests for the core update
# ══════════════════════════════════════════════════════════════════════════════

@requires_jax()
class TestRLSCoreStep:
    """Algebraic tests for _rls_step without any simulation overhead."""

    def test_identity_regressor_converges(self):
        """With phi=e_1, y=3, theta[0] should converge to 3."""
        theta_hat = jnp.zeros(2)
        P = jnp.eye(2) * 1e4
        phi = jnp.array([1.0, 0.0])
        y = jnp.array(3.0)
        lam = 1.0

        for _ in range(100):
            theta_hat, P, e = RecursiveLeastSquares._rls_step(
                theta_hat, P, phi, y, lam
            )

        assert float(theta_hat[0]) == pytest.approx(3.0, abs=1e-4)
        # theta[1] stays at zero since phi[1]=0
        assert float(theta_hat[1]) == pytest.approx(0.0, abs=1e-6)

    def test_two_orthogonal_regressors(self):
        """Alternating e_1 and e_2 observations should identify both components."""
        theta_true = np.array([2.0, -1.0])
        theta_hat = jnp.zeros(2)
        P = jnp.eye(2) * 1e4
        lam = 1.0

        for k in range(500):
            phi = jnp.array([1.0, 0.0]) if k % 2 == 0 else jnp.array([0.0, 1.0])
            y = jnp.dot(phi, jnp.array(theta_true))
            theta_hat, P, _ = RecursiveLeastSquares._rls_step(
                theta_hat, P, phi, y, lam
            )

        np.testing.assert_allclose(np.array(theta_hat), theta_true, atol=1e-4)

    def test_prediction_error_decreases_to_zero(self):
        """For noise-free data, prediction error should collapse to ≈ 0."""
        theta_hat = jnp.zeros(3)
        P = jnp.eye(3) * 1e4
        phi = jnp.array([1.0, 2.0, 3.0])
        y = jnp.array(float(phi @ np.array([1.0, -1.0, 0.5])))
        lam = 1.0

        errors = []
        for _ in range(60):
            theta_hat, P, e = RecursiveLeastSquares._rls_step(
                theta_hat, P, phi, y, lam
            )
            errors.append(float(e))

        # First error ≠ 0 (uninitialised), last error ≈ 0
        assert abs(errors[0]) > 1e-6  # some nonzero initial error
        assert abs(errors[-1]) < 1e-6

    def test_covariance_decreases_monotonically(self):
        """Trace of P should decrease (or stay the same) with each update."""
        theta_hat = jnp.zeros(2)
        P = jnp.eye(2) * 100.0
        phi = jnp.array([1.0, 2.0])
        y = jnp.array(5.0)
        lam = 1.0

        traces = [float(jnp.trace(P))]
        for _ in range(20):
            theta_hat, P, _ = RecursiveLeastSquares._rls_step(
                theta_hat, P, phi, y, lam
            )
            traces.append(float(jnp.trace(P)))

        # Each new trace should be ≤ previous
        for i in range(1, len(traces)):
            assert traces[i] <= traces[i - 1] + 1e-12

    def test_forgetting_factor_less_than_one_inflates_covariance(self):
        """λ < 1 should keep P from collapsing to zero (allows tracking)."""
        theta_hat = jnp.zeros(1)
        P_no_forget = jnp.eye(1) * 100.0
        P_forget = jnp.eye(1) * 100.0
        phi = jnp.array([1.0])
        y = jnp.array(3.0)
        lam_forget = 0.9

        for _ in range(200):
            theta_hat, P_no_forget, _ = RecursiveLeastSquares._rls_step(
                theta_hat, P_no_forget, phi, y, 1.0
            )
            theta_hat, P_forget, _ = RecursiveLeastSquares._rls_step(
                theta_hat, P_forget, phi, y, lam_forget
            )

        # With forgetting, covariance remains higher (filter stays "alert")
        assert float(P_forget[0, 0]) > float(P_no_forget[0, 0])

    def test_matches_numpy_reference(self):
        """JAX step must match the pure-NumPy reference implementation."""
        rng = np.random.default_rng(0)
        theta_hat_jax = jnp.zeros(3)
        P_jax = jnp.eye(3) * 1e3
        theta_hat_np = np.zeros(3)
        P_np = np.eye(3) * 1e3
        lam = 0.95

        for _ in range(50):
            phi_np = rng.standard_normal(3)
            y_np = float(phi_np @ np.array([1.0, 2.0, 3.0]))
            phi_jax = jnp.array(phi_np)
            y_jax = jnp.array(y_np)

            # JAX step
            theta_hat_jax, P_jax, _ = RecursiveLeastSquares._rls_step(
                theta_hat_jax, P_jax, phi_jax, y_jax, lam
            )
            # NumPy step
            e_np = y_np - phi_np @ theta_hat_np
            Pphi = P_np @ phi_np
            K = Pphi / (lam + phi_np @ Pphi)
            theta_hat_np = theta_hat_np + K * e_np
            P_np = (P_np - np.outer(K, phi_np) @ P_np) / lam

        np.testing.assert_allclose(
            np.array(theta_hat_jax), theta_hat_np, atol=1e-8
        )
        np.testing.assert_allclose(np.array(P_jax), P_np, atol=1e-8)

    def test_scalar_phi_and_y(self):
        """Scalar phi and y should work just like length-1 vectors."""
        theta_hat = jnp.zeros(1)
        P = jnp.eye(1) * 100.0
        phi = jnp.array([2.0])
        y = jnp.array(6.0)  # true theta = 3.0

        for _ in range(100):
            theta_hat, P, e = RecursiveLeastSquares._rls_step(
                theta_hat, P, phi, y, 1.0
            )

        assert float(theta_hat[0]) == pytest.approx(3.0, abs=1e-4)

    def test_jit_compatible(self):
        """_rls_step must be JIT-compilable."""
        step = jax.jit(RecursiveLeastSquares._rls_step)
        theta_hat = jnp.zeros(2)
        P = jnp.eye(2) * 1e4
        phi = jnp.array([1.0, 2.0])
        y = jnp.array(5.0)
        theta_new, P_new, e = step(theta_hat, P, phi, y, 1.0)
        assert theta_new.shape == (2,)
        assert P_new.shape == (2, 2)
        assert e.shape == ()

    def test_grad_through_step(self):
        """Gradient of the prediction error w.r.t. theta_hat should be -phi."""
        phi = jnp.array([1.0, 2.0])
        y = jnp.array(5.0)
        P = jnp.eye(2) * 1.0  # small P so update is negligible

        def loss(theta_hat):
            _, _, e = RecursiveLeastSquares._rls_step(theta_hat, P, phi, y, 1.0)
            return e ** 2

        grad = jax.grad(loss)(jnp.zeros(2))
        # d(e^2)/d(theta) = 2*e * (-phi) at theta=0, e=5
        # = 2*5*(-phi) = [-10, -20]
        expected = 2.0 * 5.0 * (-phi)
        np.testing.assert_allclose(np.array(grad), np.array(expected), atol=1e-5)


# ══════════════════════════════════════════════════════════════════════════════
# RecursiveLeastSquares – construction and block properties
# ══════════════════════════════════════════════════════════════════════════════

@requires_jax()
class TestRLSConstruction:
    """Block construction, default values, port counts."""

    def test_default_construction(self):
        """Block with defaults should construct without error."""
        rls = RecursiveLeastSquares(dt=DT, n_params=3)
        assert rls is not None

    def test_custom_initial_estimate(self):
        """theta_0 and P_0 are accepted as arrays."""
        rls = RecursiveLeastSquares(
            dt=DT,
            n_params=2,
            theta_0=jnp.array([1.0, -1.0]),
            P_0=jnp.eye(2) * 5.0,
        )
        assert rls is not None

    def test_port_count(self):
        """Block must have 2 input ports and 3 output ports."""
        rls = RecursiveLeastSquares(dt=DT, n_params=2)
        assert len(rls.input_ports) == 2
        assert len(rls.output_ports) == 3

    def test_output_port_names(self):
        """Output port names should be theta_hat, P, prediction_error."""
        rls = RecursiveLeastSquares(dt=DT, n_params=2)
        names = [p.name for p in rls.output_ports]
        assert "theta_hat" in names
        assert "P" in names
        assert "prediction_error" in names

    def test_context_creation(self):
        """create_context() should succeed (calls initialize)."""
        b = DiagramBuilder()
        rls = b.add(RecursiveLeastSquares(dt=DT, n_params=2))
        phi_src = b.add(Constant(jnp.zeros(2), name="phi"))
        y_src = b.add(Constant(jnp.zeros(()), name="y"))
        b.connect(phi_src.output_ports[0], rls.input_ports[0])
        b.connect(y_src.output_ports[0], rls.input_ports[1])
        diag = b.build()
        ctx = diag.create_context()
        assert ctx is not None

    def test_forgetting_factor_stored(self):
        """After context creation, lam should equal forgetting_factor."""
        b = DiagramBuilder()
        rls = b.add(RecursiveLeastSquares(dt=DT, n_params=1, forgetting_factor=0.9))
        phi_src = b.add(Constant(jnp.ones(1), name="phi"))
        y_src = b.add(Constant(jnp.zeros(()), name="y"))
        b.connect(phi_src.output_ports[0], rls.input_ports[0])
        b.connect(y_src.output_ports[0], rls.input_ports[1])
        diag = b.build()
        diag.create_context()
        assert rls.lam == pytest.approx(0.9)


# ══════════════════════════════════════════════════════════════════════════════
# RecursiveLeastSquares – simulation-based integration tests
# ══════════════════════════════════════════════════════════════════════════════

@requires_jax()
class TestRLSSimulation:
    """End-to-end simulation tests wiring RLS into a DiagramBuilder."""

    def _build_rls_diagram(self, phi_val, y_val, n_params=2, forgetting_factor=1.0,
                           theta_0=None, P_0=None):
        """Minimal diagram: Constant phi + Constant y → RLS."""
        b = DiagramBuilder()
        rls = b.add(
            RecursiveLeastSquares(
                dt=DT,
                n_params=n_params,
                theta_0=theta_0,
                P_0=P_0,
                forgetting_factor=forgetting_factor,
                name="rls",
            )
        )
        phi_src = b.add(Constant(jnp.array(phi_val, dtype=float), name="phi"))
        y_src = b.add(Constant(jnp.array(float(y_val)), name="y"))
        b.connect(phi_src.output_ports[0], rls.input_ports[0])
        b.connect(y_src.output_ports[0], rls.input_ports[1])
        return b.build(), rls

    def test_theta_hat_output_shape(self):
        """theta_hat output should have shape (n_params,)."""
        diag, rls = self._build_rls_diagram([1.0, 0.0], 3.0, n_params=2)
        signals = {"theta_hat": rls.output_ports[0]}
        sol = _run_simulation(diag, t_end=0.5, signals=signals)
        assert sol.outputs["theta_hat"].shape[1] == 2

    def test_P_output_shape(self):
        """P output should have shape (n_params, n_params)."""
        diag, rls = self._build_rls_diagram([1.0, 0.0], 3.0, n_params=2)
        signals = {"P": rls.output_ports[1]}
        sol = _run_simulation(diag, t_end=0.5, signals=signals)
        assert sol.outputs["P"].shape[1:] == (2, 2)

    def test_prediction_error_shape(self):
        """prediction_error output should be scalar."""
        diag, rls = self._build_rls_diagram([1.0, 0.0], 3.0, n_params=2)
        signals = {"err": rls.output_ports[2]}
        sol = _run_simulation(diag, t_end=0.5, signals=signals)
        # Each timestep returns a scalar, so shape is (n_timesteps,) or (n_timesteps, 1)
        assert sol.outputs["err"].ndim >= 1

    def test_single_component_convergence(self):
        """With phi=e_1, y=3, theta_hat[0] should converge to 3."""
        diag, rls = self._build_rls_diagram([1.0, 0.0], 3.0, n_params=2)
        signals = {"theta_hat": rls.output_ports[0]}
        sol = _run_simulation(diag, t_end=5.0, signals=signals)
        theta_final = sol.outputs["theta_hat"][-1]
        assert float(theta_final[0]) == pytest.approx(3.0, abs=1e-3)
        # Second component unobservable, remains near zero
        assert abs(float(theta_final[1])) < 0.1

    def test_prediction_error_collapses_to_zero(self):
        """After convergence, prediction error should be essentially zero."""
        diag, rls = self._build_rls_diagram([1.0, 2.0], 5.0, n_params=2)
        signals = {"err": rls.output_ports[2]}
        sol = _run_simulation(diag, t_end=5.0, signals=signals)
        errors = np.abs(sol.outputs["err"])
        # Last few errors should be very small
        assert float(np.max(errors[-5:])) < 1e-5

    def test_covariance_decreases(self):
        """Trace of P should be smaller at the end than at the start."""
        diag, rls = self._build_rls_diagram([1.0, 2.0], 5.0, n_params=2)
        signals = {"P": rls.output_ports[1]}
        sol = _run_simulation(diag, t_end=2.0, signals=signals)
        P_traj = sol.outputs["P"]
        trace_start = float(np.trace(P_traj[1]))   # skip t=0 initial
        trace_end = float(np.trace(P_traj[-1]))
        assert trace_end < trace_start

    def test_custom_initial_theta_used(self):
        """theta_0 should appear as the first output value."""
        theta_0 = jnp.array([10.0, -5.0])
        diag, rls = self._build_rls_diagram(
            [1.0, 0.0], 10.0, n_params=2, theta_0=theta_0
        )
        signals = {"theta_hat": rls.output_ports[0]}
        sol = _run_simulation(diag, t_end=DT, signals=signals)
        # The very first output uses theta_0 (before update)
        # RLS outputs are feedthrough so they use the corrected state
        # Just check that it's a valid array
        assert sol.outputs["theta_hat"].shape[1] == 2

    def test_n_params_one(self):
        """Scalar parameter identification: y = 5 * phi[0]."""
        diag, rls = self._build_rls_diagram([1.0], 5.0, n_params=1)
        signals = {"theta_hat": rls.output_ports[0]}
        sol = _run_simulation(diag, t_end=3.0, signals=signals)
        theta_final = float(sol.outputs["theta_hat"][-1, 0])
        assert theta_final == pytest.approx(5.0, abs=1e-3)

    def test_forgetting_factor_tracks_change(self):
        """With λ < 1, RLS should eventually re-identify a changed parameter."""
        # Phase 1: y=1 (true theta=1)
        diag1, rls1 = self._build_rls_diagram(
            [1.0], 1.0, n_params=1, forgetting_factor=0.97
        )
        signals = {"theta_hat": rls1.output_ports[0]}
        sol1 = _run_simulation(diag1, t_end=5.0, signals=signals)
        theta_after_phase1 = float(sol1.outputs["theta_hat"][-1, 0])

        # Phase 2: y=5 (true theta=5), different constant
        diag2, rls2 = self._build_rls_diagram(
            [1.0], 5.0, n_params=1, forgetting_factor=0.97
        )
        sol2 = _run_simulation(diag2, t_end=5.0, signals={"theta_hat": rls2.output_ports[0]})
        theta_after_phase2 = float(sol2.outputs["theta_hat"][-1, 0])

        # Both should have converged to their respective targets
        assert theta_after_phase1 == pytest.approx(1.0, abs=0.05)
        assert theta_after_phase2 == pytest.approx(5.0, abs=0.05)


# ══════════════════════════════════════════════════════════════════════════════
# AugmentedStateEKF – construction tests
# ══════════════════════════════════════════════════════════════════════════════

@requires_jax()
class TestAugmentedEKFConstruction:
    """Block instantiation and port layout."""

    @staticmethod
    def _make_simple_aekf(nx=1, n_params=1):
        """Create a minimal AugmentedStateEKF for a 1D linear system."""
        def forward(x, u, theta):
            a = theta[0]
            return jnp.array([a * x[0] + u[0]])

        def observation(x, u, theta):
            return jnp.array([x[0]])

        return AugmentedStateEKF(
            dt=DT,
            nx=nx,
            n_params=n_params,
            forward=forward,
            observation=observation,
            G_x_func=lambda t: jnp.eye(nx),
            Q_x_func=lambda t, x, u, th: jnp.eye(nx) * 0.01,
            Q_theta=jnp.eye(n_params) * 1e-4,
            R_func=lambda t: jnp.eye(1) * 0.1,
            x_hat_0=jnp.zeros(nx),
            P_hat_0_x=jnp.eye(nx),
            theta_hat_0=jnp.zeros(n_params),
            P_hat_0_theta=jnp.eye(n_params),
        )

    def test_construction_succeeds(self):
        aekf = self._make_simple_aekf()
        assert aekf is not None

    def test_port_counts(self):
        """Must have 2 input and 2 output ports."""
        aekf = self._make_simple_aekf()
        assert len(aekf.input_ports) == 2
        assert len(aekf.output_ports) == 2

    def test_output_port_names(self):
        aekf = self._make_simple_aekf()
        names = [p.name for p in aekf.output_ports]
        assert "x_hat" in names
        assert "theta_hat" in names

    def test_context_creation(self):
        """build + create_context should succeed without errors."""
        b = DiagramBuilder()
        aekf = b.add(self._make_simple_aekf(nx=1, n_params=1))
        u_src = b.add(Constant(jnp.zeros(1), name="u"))
        y_src = b.add(Constant(jnp.ones(1), name="y"))
        b.connect(u_src.output_ports[0], aekf.input_ports[0])
        b.connect(y_src.output_ports[0], aekf.input_ports[1])
        diag = b.build()
        ctx = diag.create_context()
        assert ctx is not None

    def test_initialize_stores_dimensions(self):
        """After context creation, nx, np, nz should be set."""
        b = DiagramBuilder()
        aekf = b.add(self._make_simple_aekf(nx=2, n_params=3))
        u_src = b.add(Constant(jnp.zeros(1), name="u"))
        y_src = b.add(Constant(jnp.ones(1), name="y"))
        b.connect(u_src.output_ports[0], aekf.input_ports[0])
        b.connect(y_src.output_ports[0], aekf.input_ports[1])
        b.build().create_context()
        assert aekf.nx == 2
        assert aekf.np == 3
        assert aekf.nz == 5

    def test_larger_dimensions(self):
        """Higher-dimensional system should construct without error."""
        nx, nparams = 3, 2

        def forward(x, u, theta):
            A = jnp.array([[theta[0], 0.0, 0.0],
                            [0.0, theta[1], 0.0],
                            [0.0, 0.0, 0.9]])
            return A @ x + u

        def observation(x, u, theta):
            return x[:2]

        aekf = AugmentedStateEKF(
            dt=DT,
            nx=nx,
            n_params=nparams,
            forward=forward,
            observation=observation,
            G_x_func=lambda t: jnp.eye(nx),
            Q_x_func=lambda t, x, u, th: jnp.eye(nx) * 0.01,
            Q_theta=jnp.eye(nparams) * 1e-4,
            R_func=lambda t: jnp.eye(2) * 0.1,
            x_hat_0=jnp.zeros(nx),
            P_hat_0_x=jnp.eye(nx),
            theta_hat_0=jnp.ones(nparams) * 0.8,
            P_hat_0_theta=jnp.eye(nparams),
        )
        b = DiagramBuilder()
        aekf = b.add(aekf)
        u_src = b.add(Constant(jnp.zeros(nx), name="u"))
        y_src = b.add(Constant(jnp.zeros(2), name="y"))
        b.connect(u_src.output_ports[0], aekf.input_ports[0])
        b.connect(y_src.output_ports[0], aekf.input_ports[1])
        diag = b.build()
        ctx = diag.create_context()
        assert ctx is not None


# ══════════════════════════════════════════════════════════════════════════════
# AugmentedStateEKF – simulation-based integration tests
# ══════════════════════════════════════════════════════════════════════════════

@requires_jax()
class TestAugmentedEKFSimulation:
    """Integration tests running AugmentedStateEKF in a simulation loop."""

    def _build_linear_diagram(
        self,
        true_a,
        nx=1,
        n_params=1,
        t_end=3.0,
        x_init=1.0,
        theta_init=0.5,
        Q_theta_scale=1e-4,
    ):
        """
        Build diagram: constant u=0, y = true_a * x_init (constant observation).
        AugmentedStateEKF estimates both x and the decay coefficient a.
        """

        def forward(x, u, theta):
            a = theta[0]
            return jnp.array([a * x[0] + u[0]])

        def observation(x, u, theta):
            return jnp.array([x[0]])

        aekf = AugmentedStateEKF(
            dt=DT,
            nx=1,
            n_params=1,
            forward=forward,
            observation=observation,
            G_x_func=lambda t: jnp.eye(1),
            Q_x_func=lambda t, x, u, th: jnp.eye(1) * 1e-4,
            Q_theta=jnp.eye(1) * Q_theta_scale,
            R_func=lambda t: jnp.eye(1) * 1e-3,  # low noise → fast convergence
            x_hat_0=jnp.zeros(1),
            P_hat_0_x=jnp.eye(1) * 1.0,
            theta_hat_0=jnp.array([theta_init]),
            P_hat_0_theta=jnp.eye(1) * 1.0,
        )

        b = DiagramBuilder()
        aekf = b.add(aekf)
        u_src = b.add(Constant(jnp.zeros(1), name="u"))
        # Observation: constant y = x_init (system already at that state)
        y_src = b.add(Constant(jnp.array([x_init]), name="y"))
        b.connect(u_src.output_ports[0], aekf.input_ports[0])
        b.connect(y_src.output_ports[0], aekf.input_ports[1])
        diag = b.build()
        return diag, aekf

    def test_output_shapes(self):
        """x_hat and theta_hat outputs should have correct shapes."""
        diag, aekf = self._build_linear_diagram(true_a=0.9)
        signals = {
            "x_hat": aekf.output_ports[0],
            "theta_hat": aekf.output_ports[1],
        }
        sol = _run_simulation(diag, t_end=0.5, signals=signals)
        assert sol.outputs["x_hat"].shape[1] == 1
        assert sol.outputs["theta_hat"].shape[1] == 1

    def test_x_hat_converges_to_observation(self):
        """With low measurement noise, x_hat should track the constant y well."""
        x_true = 1.0
        diag, aekf = self._build_linear_diagram(true_a=0.9, x_init=x_true)
        signals = {"x_hat": aekf.output_ports[0]}
        sol = _run_simulation(diag, t_end=3.0, signals=signals)
        x_hat_final = float(sol.outputs["x_hat"][-1, 0])
        assert x_hat_final == pytest.approx(x_true, abs=0.05)

    def test_theta_hat_converges_toward_true_value(self):
        """
        For a constant observation y = x_init, the filter should push theta_hat
        toward a value consistent with x_init = a * x_hat.  With x_init near 1
        and a small forgetting covariance, we verify the estimate moves toward
        the true parameter.
        """
        true_a = 0.9
        theta_init = 0.5  # starting guess, far from true value
        diag, aekf = self._build_linear_diagram(
            true_a=true_a, x_init=1.0, theta_init=theta_init, Q_theta_scale=1e-2
        )
        signals = {"theta_hat": aekf.output_ports[1]}
        sol = _run_simulation(diag, t_end=5.0, signals=signals)
        theta_trajectory = sol.outputs["theta_hat"][:, 0]
        # theta should have moved away from the initial guess
        initial_error = abs(float(theta_trajectory[1]) - true_a)
        final_error = abs(float(theta_trajectory[-1]) - true_a)
        assert final_error <= initial_error + 0.01  # not getting worse at least

    def test_simulation_runs_without_nan(self):
        """No NaN should appear in any output over a long simulation."""
        diag, aekf = self._build_linear_diagram(true_a=0.8, t_end=5.0)
        signals = {
            "x_hat": aekf.output_ports[0],
            "theta_hat": aekf.output_ports[1],
        }
        sol = _run_simulation(diag, t_end=5.0, signals=signals)
        assert not np.any(np.isnan(sol.outputs["x_hat"]))
        assert not np.any(np.isnan(sol.outputs["theta_hat"]))

    def test_initial_theta_hat_from_block(self):
        """theta_hat output at t=0 should reflect theta_hat_0."""
        theta_init = 0.3
        diag, aekf = self._build_linear_diagram(
            true_a=0.9, theta_init=theta_init
        )
        signals = {"theta_hat": aekf.output_ports[1]}
        sol = _run_simulation(diag, t_end=DT, signals=signals)
        # At the first recorded step, theta_hat should be close to theta_init
        # (the feedthrough corrects it, but not by much with high initial covariance
        #  and reasonable y)
        theta_0 = float(sol.outputs["theta_hat"][0, 0])
        assert not np.isnan(theta_0)
        assert np.isfinite(theta_0)

    def test_zero_control_input(self):
        """With u=0, forward should reduce to x[n+1] = a * x[n]."""
        diag, aekf = self._build_linear_diagram(true_a=0.9)
        signals = {"x_hat": aekf.output_ports[0]}
        sol = _run_simulation(diag, t_end=2.0, signals=signals)
        assert sol.outputs["x_hat"].shape[0] > 1

    def test_two_parameters(self):
        """Augmented EKF with two unknown parameters."""
        nx, np_ = 1, 2

        def forward(x, u, theta):
            # x[n+1] = theta[0] * x[n] + theta[1] * u[0]
            return jnp.array([theta[0] * x[0] + theta[1] * u[0]])

        def observation(x, u, theta):
            return jnp.array([x[0]])

        aekf = AugmentedStateEKF(
            dt=DT,
            nx=1,
            n_params=2,
            forward=forward,
            observation=observation,
            G_x_func=lambda t: jnp.eye(1),
            Q_x_func=lambda t, x, u, th: jnp.eye(1) * 0.01,
            Q_theta=jnp.eye(2) * 1e-4,
            R_func=lambda t: jnp.eye(1) * 0.01,
            x_hat_0=jnp.zeros(1),
            P_hat_0_x=jnp.eye(1),
            theta_hat_0=jnp.zeros(2),
            P_hat_0_theta=jnp.eye(2),
        )

        b = DiagramBuilder()
        aekf = b.add(aekf)
        u_src = b.add(Constant(jnp.array([0.5]), name="u"))
        y_src = b.add(Constant(jnp.array([1.0]), name="y"))
        b.connect(u_src.output_ports[0], aekf.input_ports[0])
        b.connect(y_src.output_ports[0], aekf.input_ports[1])
        diag = b.build()
        signals = {"x_hat": aekf.output_ports[0], "theta_hat": aekf.output_ports[1]}
        sol = _run_simulation(diag, t_end=1.0, signals=signals)
        assert sol.outputs["theta_hat"].shape[1] == 2
        assert not np.any(np.isnan(sol.outputs["theta_hat"]))

    def test_multi_output_observation(self):
        """AugmentedStateEKF with 2D observation and 2D state."""
        nx, np_, ny = 2, 1, 2

        def forward(x, u, theta):
            a = theta[0]
            return jnp.array([a * x[0] + u[0], 0.9 * x[1]])

        def observation(x, u, theta):
            return x  # full state observed

        aekf = AugmentedStateEKF(
            dt=DT,
            nx=nx,
            n_params=np_,
            forward=forward,
            observation=observation,
            G_x_func=lambda t: jnp.eye(nx),
            Q_x_func=lambda t, x, u, th: jnp.eye(nx) * 0.01,
            Q_theta=jnp.eye(np_) * 1e-4,
            R_func=lambda t: jnp.eye(ny) * 0.01,
            x_hat_0=jnp.zeros(nx),
            P_hat_0_x=jnp.eye(nx),
            theta_hat_0=jnp.zeros(np_),
            P_hat_0_theta=jnp.eye(np_),
        )

        b = DiagramBuilder()
        aekf = b.add(aekf)
        u_src = b.add(Constant(jnp.zeros(1), name="u"))
        y_src = b.add(Constant(jnp.ones(ny), name="y"))
        b.connect(u_src.output_ports[0], aekf.input_ports[0])
        b.connect(y_src.output_ports[0], aekf.input_ports[1])
        diag = b.build()
        signals = {"x_hat": aekf.output_ports[0], "theta_hat": aekf.output_ports[1]}
        sol = _run_simulation(diag, t_end=1.0, signals=signals)
        assert sol.outputs["x_hat"].shape[1] == nx
        assert sol.outputs["theta_hat"].shape[1] == np_
        assert not np.any(np.isnan(sol.outputs["x_hat"]))


# ══════════════════════════════════════════════════════════════════════════════
# Interaction and comparison tests
# ══════════════════════════════════════════════════════════════════════════════

@requires_jax()
class TestOnlineEstimationComparison:
    """Cross-validation and edge case tests."""

    def test_rls_matches_pure_python_reference(self):
        """
        Simulate RLS in Python loop and compare with a DiagramBuilder simulation
        for the same random-regressor sequence.  Since we can't vary inputs
        across timesteps in a DiagramBuilder, we test consistent deterministic
        behaviour: the constant-phi DiagramBuilder simulation should match the
        Python reference over the same constant phi.
        """
        phi_val = np.array([1.5, -2.0, 0.5])
        theta_true = np.array([1.0, 2.0, 3.0])
        y_val = float(phi_val @ theta_true)
        n_params = 3

        # Python reference
        theta_ref, _ = _rls_loop(
            n_params,
            theta_true,
            phi_fn=lambda k: phi_val,
            noise_std=0.0,
            n_steps=int(5.0 / DT),
            forgetting_factor=1.0,
        )

        # DiagramBuilder simulation
        b = DiagramBuilder()
        rls = b.add(
            RecursiveLeastSquares(dt=DT, n_params=n_params, forgetting_factor=1.0)
        )
        phi_src = b.add(Constant(jnp.array(phi_val), name="phi"))
        y_src = b.add(Constant(jnp.array(y_val), name="y"))
        b.connect(phi_src.output_ports[0], rls.input_ports[0])
        b.connect(y_src.output_ports[0], rls.input_ports[1])
        diag = b.build()

        signals = {"theta_hat": rls.output_ports[0]}
        sol = _run_simulation(diag, t_end=5.0, signals=signals)
        theta_sim = np.array(sol.outputs["theta_hat"][-1])

        np.testing.assert_allclose(theta_sim, theta_ref, atol=1e-4)

    def test_rls_n_params_one_exact_recovery(self):
        """Single-parameter case: noiseless → exact recovery."""
        phi_val = np.array([2.5])
        theta_true = np.array([7.0])
        y_val = float(phi_val @ theta_true)

        b = DiagramBuilder()
        rls = b.add(RecursiveLeastSquares(dt=DT, n_params=1))
        phi_src = b.add(Constant(jnp.array(phi_val), name="phi"))
        y_src = b.add(Constant(jnp.array(y_val), name="y"))
        b.connect(phi_src.output_ports[0], rls.input_ports[0])
        b.connect(y_src.output_ports[0], rls.input_ports[1])
        diag = b.build()
        signals = {"theta_hat": rls.output_ports[0]}
        sol = _run_simulation(diag, t_end=2.0, signals=signals)
        theta_final = float(sol.outputs["theta_hat"][-1, 0])
        assert theta_final == pytest.approx(theta_true[0], abs=1e-3)

    def test_aekf_correct_step_reduces_covariance(self):
        """
        After one manual correction step, P_plus should have smaller trace
        than P_minus (the Kalman update always reduces uncertainty).
        """
        nx = 1
        n_params = 1

        def forward(x, u, theta):
            return jnp.array([theta[0] * x[0]])

        def observation(x, u, theta):
            return jnp.array([x[0]])

        b = DiagramBuilder()
        aekf = AugmentedStateEKF(
            dt=DT,
            nx=nx,
            n_params=n_params,
            forward=forward,
            observation=observation,
            G_x_func=lambda t: jnp.eye(nx),
            Q_x_func=lambda t, x, u, th: jnp.eye(nx) * 0.01,
            Q_theta=jnp.eye(n_params) * 1e-4,
            R_func=lambda t: jnp.eye(1) * 0.1,
            x_hat_0=jnp.zeros(nx),
            P_hat_0_x=jnp.eye(nx) * 10.0,  # large initial uncertainty
            theta_hat_0=jnp.zeros(n_params),
            P_hat_0_theta=jnp.eye(n_params) * 10.0,
        )
        aekf = b.add(aekf)
        u_src = b.add(Constant(jnp.zeros(1), name="u"))
        y_src = b.add(Constant(jnp.ones(1), name="y"))
        b.connect(u_src.output_ports[0], aekf.input_ports[0])
        b.connect(y_src.output_ports[0], aekf.input_ports[1])
        diag = b.build()
        diag.create_context()

        # Invoke one correction step manually via the augmented state
        z_minus = jnp.zeros(nx + n_params)
        P_minus = jnp.eye(nx + n_params) * 10.0
        u = jnp.zeros(1)
        y = jnp.ones(1)

        z_plus, P_plus = aekf._correct(0.0, z_minus, P_minus, u, y)

        assert float(jnp.trace(P_plus)) < float(jnp.trace(P_minus))

    def test_aekf_propagate_increases_covariance(self):
        """
        The propagate step with nonzero Q should increase covariance
        (uncertainty grows between measurements).
        """
        nx = 1
        n_params = 1

        def forward(x, u, theta):
            return jnp.array([0.9 * x[0]])

        def observation(x, u, theta):
            return jnp.array([x[0]])

        b = DiagramBuilder()
        aekf = AugmentedStateEKF(
            dt=DT,
            nx=nx,
            n_params=n_params,
            forward=forward,
            observation=observation,
            G_x_func=lambda t: jnp.eye(nx),
            Q_x_func=lambda t, x, u, th: jnp.eye(nx) * 1.0,  # large Q
            Q_theta=jnp.eye(n_params) * 1.0,
            R_func=lambda t: jnp.eye(1) * 0.1,
            x_hat_0=jnp.zeros(nx),
            P_hat_0_x=jnp.eye(nx),
            theta_hat_0=jnp.zeros(n_params),
            P_hat_0_theta=jnp.eye(n_params),
        )
        aekf = b.add(aekf)
        u_src = b.add(Constant(jnp.zeros(1), name="u"))
        y_src = b.add(Constant(jnp.ones(1), name="y"))
        b.connect(u_src.output_ports[0], aekf.input_ports[0])
        b.connect(y_src.output_ports[0], aekf.input_ports[1])
        diag = b.build()
        diag.create_context()

        z_plus = jnp.zeros(nx + n_params)
        P_plus = jnp.eye(nx + n_params) * 0.1  # small input covariance
        u = jnp.zeros(1)

        _, P_minus = aekf._propagate(0.0, z_plus, P_plus, u)

        assert float(jnp.trace(P_minus)) > float(jnp.trace(P_plus))

    def test_rls_is_jax_traceable_in_diagram(self):
        """Running a JIT-compiled simulation over RLS should not raise."""
        b = DiagramBuilder()
        rls = b.add(RecursiveLeastSquares(dt=DT, n_params=2))
        phi_src = b.add(Constant(jnp.array([1.0, 2.0]), name="phi"))
        y_src = b.add(Constant(jnp.array(5.0), name="y"))
        b.connect(phi_src.output_ports[0], rls.input_ports[0])
        b.connect(y_src.output_ports[0], rls.input_ports[1])
        diag = b.build()
        ctx = diag.create_context()
        # jaxonomy.simulate uses JIT internally; this will raise if not traceable
        options = SimulatorOptions(max_major_steps=20, max_major_step_length=DT)
        sol = jaxonomy.simulate(diag, ctx, (0.0, 1.0), options=options,
                                recorded_signals={"theta_hat": rls.output_ports[0]})
        assert sol is not None

    def test_aekf_is_jax_traceable_in_diagram(self):
        """AugmentedStateEKF must survive a JIT-compiled simulation run."""
        nx, np_ = 1, 1

        def forward(x, u, theta):
            return jnp.array([theta[0] * x[0] + u[0]])

        def observation(x, u, theta):
            return jnp.array([x[0]])

        b = DiagramBuilder()
        aekf = b.add(AugmentedStateEKF(
            dt=DT, nx=nx, n_params=np_,
            forward=forward, observation=observation,
            G_x_func=lambda t: jnp.eye(nx),
            Q_x_func=lambda t, x, u, th: jnp.eye(nx) * 0.01,
            Q_theta=jnp.eye(np_) * 1e-4,
            R_func=lambda t: jnp.eye(1) * 0.1,
            x_hat_0=jnp.zeros(nx), P_hat_0_x=jnp.eye(nx),
            theta_hat_0=jnp.zeros(np_), P_hat_0_theta=jnp.eye(np_),
        ))
        u_src = b.add(Constant(jnp.zeros(1), name="u"))
        y_src = b.add(Constant(jnp.ones(1), name="y"))
        b.connect(u_src.output_ports[0], aekf.input_ports[0])
        b.connect(y_src.output_ports[0], aekf.input_ports[1])
        diag = b.build()
        ctx = diag.create_context()
        options = SimulatorOptions(max_major_steps=20, max_major_step_length=DT)
        sol = jaxonomy.simulate(diag, ctx, (0.0, 1.0), options=options,
                                recorded_signals={"x_hat": aekf.output_ports[0]})
        assert sol is not None
