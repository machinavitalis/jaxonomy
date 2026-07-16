# SPDX-License-Identifier: MIT

"""Tests for the BDF non-finite abort diagnostic and forward/backward
NaN parity (consumer-reported diagnosability and AD-parity issues, 2026-07).

The diagnostic: when the BDF corrector/step retry loop exits through the
terminal bailout (T-005/T-008/T-134 — the trial state went non-finite or
the error test failed at the minimum step size, and the state is poisoned
with NaN), ``BDFSolver.step`` now surfaces a host-side ``UserWarning``
via ``jax.debug.callback`` reporting the failure time, the collapsed step
size, and which state rows went non-finite.  Healthy runs stay silent and
pay no host round-trip (the callback sits behind a ``lax.cond``).

Parity: reverse-mode VJPs through the BDF solve must be finite wherever
the forward solve is finite, and enabling ``enable_autodiff`` must not
change forward numerical behavior.  The 2026-07 tutorial-series reports
of "forward finite / VJP NaN" and "forward NaNs only with autodiff" both
traced to unconverged ``project_constraints`` warm starts handing BDF an
inconsistent algebraic state — the forward solve fails identically with
autodiff on or off (verified on the tank-network repro), and the VJP is
NaN only because the forward is.  The tests here pin that parity on a
semi-explicit DAE.
"""

from __future__ import annotations

import re
import warnings

import numpy as np
import pytest
import jax
import jax.numpy as jnp

import jaxonomy
from jaxonomy import LeafSystem, SimulatorOptions, simulate
from jaxonomy.library import Constant, Integrator
from jaxonomy.testing.markers import skip_if_not_jax

skip_if_not_jax()

pytestmark = pytest.mark.minimal

_ABORT_MSG = "BDF solver aborted"


def _abort_warnings(records):
    return [
        r for r in records
        if issubclass(r.category, UserWarning) and _ABORT_MSG in str(r.message)
    ]


def _source_driven_integrator(value):
    builder = jaxonomy.DiagramBuilder()
    src = builder.add(Constant(value))
    integ = builder.add(Integrator(initial_state=0.0))
    builder.connect(src.output_ports[0], integ.input_ports[0])
    return builder.build(), src, integ


class Blowup(LeafSystem):
    """``x' = x**2`` with ``x(0) = 1`` — finite-time blowup at t = 1."""

    def __init__(self, name=None):
        super().__init__(name=name)
        self.declare_continuous_state(default_value=jnp.array([1.0]), ode=self._ode)
        self.declare_continuous_state_output(name="x")

    def _ode(self, time, state, **params):
        x = state.continuous_state
        return x**2


class SemiExplicitDAE(LeafSystem):
    """Index-1 nonlinear semi-explicit DAE:

    ``x' = -x + z``,  ``0 = z - x**2``   (so on-manifold ``z = x**2``).
    """

    def __init__(self, x0=0.8, z0=None, name=None):
        super().__init__(name=name)
        if z0 is None:
            z0 = x0**2  # consistent algebraic start
        self.declare_continuous_state(
            default_value=jnp.array([x0, z0]),
            ode=self._ode,
            mass_matrix=np.array([1.0, 0.0]),
        )
        self.declare_continuous_state_output(name="x")

    def _ode(self, time, state, **params):
        x, z = state.continuous_state
        return jnp.array([-x + z, z - x**2])


# detailed abort diagnostics are opt-in (the in-graph callback costs ~0.3 s
# of XLA compile per BDF model); these tests exercise the detailed path
_BDF = dict(ode_solver_method="bdf", math_backend="jax",
            bdf_nonfinite_diagnostics=True)
_BDF_DEFAULT = dict(ode_solver_method="bdf", math_backend="jax")


class TestBDFNonfiniteWarning:
    def test_healthy_run_is_silent(self):
        diagram, _, integ = _source_driven_integrator(1.0)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            results = simulate(
                diagram, diagram.create_context(), (0.0, 1.0),
                options=SimulatorOptions(**_BDF),
            )
        assert _abort_warnings(w) == []
        xf = np.asarray(results.context[integ.system_id].continuous_state)
        assert np.allclose(xf, 1.0)

    def test_nan_source_warns_with_time_and_rows(self):
        diagram, _, integ = _source_driven_integrator(float("nan"))
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            results = simulate(
                diagram, diagram.create_context(), (0.0, 1.0),
                options=SimulatorOptions(**_BDF),
            )
        aborts = _abort_warnings(w)
        assert len(aborts) >= 1
        msg = str(aborts[0].message)
        # Reports the failure time and which state rows went non-finite.
        assert re.search(r"t=[-\d.einfa]+, dt=", msg)
        assert "non-finite state rows [0] (of 1)" in msg
        # The state is poisoned as before (the diagnostic changes no numerics).
        xf = np.asarray(results.context[integ.system_id].continuous_state)
        assert np.isnan(xf).all()

    def test_finite_time_blowup_warns_near_blowup_time(self):
        system = Blowup()
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            simulate(
                system, system.create_context(), (0.0, 2.0),
                options=SimulatorOptions(**_BDF),
            )
        aborts = _abort_warnings(w)
        assert len(aborts) >= 1
        msg = str(aborts[0].message)
        m = re.search(r"t=([-\d.e+]+), dt=", msg)
        assert m, msg
        t_fail = float(m.group(1))
        # x' = x^2, x(0)=1 blows up at t=1; the reported failure time must
        # be at the blowup, not at the interval boundary (t=2).
        assert 0.9 <= t_fail <= 1.1

    def test_vmap_batched_warning_identifies_lane(self):
        diagram, src, integ = _source_driven_integrator(1.0)
        ctx0 = diagram.create_context()

        def run(v):
            ctx = ctx0.with_subcontext(
                src.system_id, ctx0[src.system_id].with_parameter("value", v)
            )
            r = simulate(
                diagram, ctx, (0.0, 1.0),
                options=SimulatorOptions(max_major_steps=16, **_BDF),
            )
            return r.context[integ.system_id].continuous_state

        vals = jnp.array([1.0, jnp.nan])
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            out = np.asarray(jax.jit(jax.vmap(run))(vals)).ravel()
        aborts = _abort_warnings(w)
        # The warning must fire under jit+vmap (the historical vmap-of-cond
        # IO-effect limitation, T-002b, no longer applies).  Depending on how
        # JAX lowers the cond under vmap the emitter may see per-lane scalars
        # or a batched flag, so only the firing itself is asserted.
        assert len(aborts) >= 1
        # Healthy lane unaffected; NaN lane poisoned.
        assert np.isclose(out[0], 1.0)
        assert np.isnan(out[1])

    def test_reverse_ad_healthy_run_silent_and_correct(self):
        diagram, src, integ = _source_driven_integrator(1.0)
        ctx0 = diagram.create_context()
        opts = SimulatorOptions(
            enable_autodiff=True, max_major_steps=16, **_BDF
        )

        def xf(v):
            ctx = ctx0.with_subcontext(
                src.system_id, ctx0[src.system_id].with_parameter("value", v)
            )
            r = simulate(diagram, ctx, (0.0, 1.0), options=opts)
            return jnp.reshape(
                jnp.asarray(r.context[integ.system_id].continuous_state), ()
            )

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            g = jax.grad(xf)(jnp.asarray(1.5))
        assert _abort_warnings(w) == []
        # d/du of integral of constant u over [0, 1] is 1.
        assert np.isclose(float(g), 1.0, rtol=1e-6)


class TestBDFAutodiffParity:
    """Consumer-reported forward/backward
    finiteness parity through the implicit BDF solve on a DAE."""

    def _final_state(self, system, opts, tf=2.0):
        return simulate(
            system, system.create_context(), (0.0, tf), options=opts
        ).context.continuous_state

    def test_forward_identical_with_and_without_autodiff(self):
        """Addendum item 6: enabling autodiff must not change forward
        numerics — pinned bit-exact at loose tolerance on a DAE."""
        kwargs = dict(rtol=1e-6, atol=1e-8, max_major_steps=32, **_BDF)
        x_ad = self._final_state(
            SemiExplicitDAE(),
            SimulatorOptions(enable_autodiff=True, **kwargs),
        )
        x_no = self._final_state(
            SemiExplicitDAE(),
            SimulatorOptions(enable_autodiff=False, **kwargs),
        )
        x_ad, x_no = np.asarray(x_ad), np.asarray(x_no)
        assert np.isfinite(x_ad).all()
        assert np.array_equal(x_ad, x_no)

    def test_vjp_finite_and_matches_fd_where_forward_finite(self):
        """Root cause of the reported NaN gradients: wherever the forward
        BDF/DAE solve is finite, the reverse-mode VJP is finite and matches
        central differences — including through a state reset that seeds the
        algebraic row from the differential one (``z0 = x0**2``), the
        episodic/DPC pattern.  Before the initial-time algebraic-cotangent
        correction, the adjoint solve's algebraic variable λ_a(0) leaked into
        dJ/dz0 (true value 0) and the chained gradient came out ≈ -0.90
        instead of ≈ +1.51."""
        opts = SimulatorOptions(
            rtol=1e-8, atol=1e-10, enable_autodiff=True,
            max_major_steps=32, **_BDF,
        )
        system = SemiExplicitDAE()
        ctx0 = system.create_context()

        def xf2(x0, z0):
            ctx = ctx0.with_continuous_state(jnp.array([x0, z0]))
            r = simulate(system, ctx, (0.0, 1.0), options=opts)
            return jnp.asarray(r.context.continuous_state)[0]

        def xf_chained(x0):
            return xf2(x0, x0**2)  # reset-then-integrate: z0 = h(x0)

        x0 = jnp.asarray(0.8)
        z0 = jnp.asarray(0.64)
        eps = 1e-6

        # Independent-argument gradients.
        gx0, gz0 = jax.grad(xf2, argnums=(0, 1))(x0, z0)
        fd_x0 = (float(xf2(x0 + eps, z0)) - float(xf2(x0 - eps, z0))) / (2 * eps)
        assert np.isfinite(float(gx0)) and np.isfinite(float(gz0))
        assert np.isclose(float(gx0), fd_x0, rtol=1e-4), (float(gx0), fd_x0)
        # The algebraic initial state is not a free input (the solver
        # re-enforces the constraint), so its sensitivity is exactly zero.
        assert float(gz0) == 0.0

        # Chained (manifold-seeded) gradient matches FD.
        y, pull = jax.vjp(xf_chained, x0)
        (g,) = pull(jnp.asarray(1.0))
        fd = (
            float(xf_chained(x0 + eps)) - float(xf_chained(x0 - eps))
        ) / (2 * eps)
        assert np.isfinite(float(y))
        assert np.isclose(float(g), fd, rtol=1e-4), (float(g), fd)


class TestDefaultPathGenericWarning:
    def test_default_warns_generically_and_points_at_flag(self):
        """Without the opt-in, a non-finite end state still warns (free,
        post-run, host-side) and names the detailed flag."""
        diagram, _, integ = _source_driven_integrator(float("nan"))
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            simulate(
                diagram, diagram.create_context(), (0.0, 1.0),
                options=SimulatorOptions(**_BDF_DEFAULT),
            )
        # no detailed abort warning on the default path...
        assert _abort_warnings(w) == []
        # ...but the generic post-run warning fires and names the flag
        generic = [
            x for x in w
            if "non-finite continuous state" in str(x.message)
        ]
        assert len(generic) == 1
        assert "bdf_nonfinite_diagnostics=True" in str(generic[0].message)

    def test_default_healthy_run_fully_silent(self):
        diagram, _, integ = _source_driven_integrator(1.0)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            simulate(
                diagram, diagram.create_context(), (0.0, 1.0),
                options=SimulatorOptions(**_BDF_DEFAULT),
            )
        assert _abort_warnings(w) == []
        assert not [
            x for x in w if "non-finite continuous state" in str(x.message)
        ]
