# SPDX-License-Identifier: MIT

"""Tests for T-110-followup-attach-on-jacfwd.

Covers attaching a :class:`ProvenanceManifest` to the result of
:func:`jaxonomy.simulate_jacfwd` when ``record_provenance=True``.

The natural return of :func:`simulate_jacfwd` is a JAX array (or pytree
of arrays); there is no ``.provenance`` field to attach to.  The
followup therefore changes the return shape when ``record_provenance=True``
to a tuple ``(jacobian, manifest)``.  The default-off path stays
byte-equivalent — just the jacobian.

Pins:

* Default (``record_provenance=False``) → bare jacobian, numerically
  identical to the historical contract.
* Opt-in → ``(jacobian, manifest)`` tuple.  The jacobian is the same
  value as on the default path; the manifest has populated
  ``jaxonomy_version`` / ``jax_version`` / ``system`` / ``options``
  / ``config_hash`` fields.
* Reproducibility: re-running the same setup yields identical
  ``config_hash`` values (timestamp / git metadata excluded by design).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaxonomy import LeafSystem, SimulatorOptions, simulate_jacfwd
from jaxonomy.simulation.provenance import ProvenanceManifest
from jaxonomy.testing.markers import skip_if_not_jax

skip_if_not_jax()


pytestmark = pytest.mark.minimal


# ---------------------------------------------------------------------------
# Tiny decay system reused across every test.  Identical in shape to the
# fixtures in ``test/autodiff/test_simulate_jacfwd.py`` so the same Jacobian
# value is asserted across files.
# ---------------------------------------------------------------------------


class _ScalarDecay(LeafSystem):
    """dx/dt = -a*x; output = x."""

    def __init__(self, a: float = 1.5):
        super().__init__()
        self.declare_dynamic_parameter("a", a)
        self.declare_continuous_state(
            default_value=jnp.array(1.0), ode=self._ode
        )
        self.declare_output_port(
            lambda t, s, **p: s.continuous_state,
            default_value=jnp.zeros(()),
        )

    def _ode(self, time, state, **params):
        return -params["a"] * state.continuous_state


def _opts() -> SimulatorOptions:
    return SimulatorOptions(
        math_backend="jax",
        ode_solver_method="dopri5",
        rtol=1e-9,
        atol=1e-12,
        max_major_steps=200,
    )


def _make_ctx_factory(sys, x0: float):
    def make_ctx(a):
        ctx = sys.create_context()
        ctx = ctx.with_continuous_state(jnp.array(x0))
        ctx.parameters["a"] = a
        return ctx
    return make_ctx


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestSimulateJacfwdProvenance:
    def test_default_off_returns_bare_jacobian(self):
        """Default ``record_provenance=False`` keeps the historical
        single-return contract (a bare jacobian, not a tuple)."""
        sys = _ScalarDecay(a=1.5)
        make_ctx = _make_ctx_factory(sys, x0=4.0)
        a = jnp.array(1.5)
        J = simulate_jacfwd(sys, make_ctx, (0.0, 2.0), a, options=_opts())
        # Bare jacobian (a JAX array) — must NOT be a tuple.
        assert not isinstance(J, tuple)
        # Sanity check the analytic value: -T * x0 * exp(-a*T).
        expected = -2.0 * 4.0 * jnp.exp(-1.5 * 2.0)
        np.testing.assert_allclose(J, expected, rtol=1e-5)

    def test_opt_in_returns_tuple_with_manifest(self):
        """``record_provenance=True`` switches to a ``(J, manifest)``
        tuple return.  The jacobian is bit-for-bit the same as on the
        default-off path; the manifest has populated fields."""
        sys = _ScalarDecay(a=1.5)
        make_ctx = _make_ctx_factory(sys, x0=4.0)
        a = jnp.array(1.5)

        J_bare = simulate_jacfwd(sys, make_ctx, (0.0, 2.0), a, options=_opts())
        J_paired, manifest = simulate_jacfwd(
            sys, make_ctx, (0.0, 2.0), a,
            options=_opts(), record_provenance=True,
        )

        # Jacobian value is unchanged by the provenance kwarg.
        np.testing.assert_array_equal(np.asarray(J_bare), np.asarray(J_paired))

        # Manifest is populated.
        assert isinstance(manifest, ProvenanceManifest)
        assert manifest.jaxonomy_version
        assert manifest.jax_version
        assert manifest.numpy_version
        assert "default_float_dtype" in manifest.precision_info
        # System fingerprint is recorded.
        assert manifest.system["hash"] is not None
        assert "a" in manifest.system["parameter_names"]
        # The options snapshot records ``record_provenance=True`` so an
        # auditor can tell the manifest was opt-in (mirrors the
        # simulate_batch followup).
        assert manifest.options["record_provenance"] is True
        # ``simulate_jacfwd`` forces ``enable_autodiff=False`` — the
        # manifest captures the value actually used during the run.
        assert manifest.options["enable_autodiff"] is False
        # Config hash is populated.
        assert manifest.config_hash
        assert len(manifest.config_hash) == 64  # sha256 hex digest

    def test_same_setup_same_config_hash(self):
        """Re-running the same diagram + options yields manifests with
        identical ``config_hash`` values.  Timestamp / git metadata are
        excluded from the hash by design, so they may differ — but the
        run identity should not."""
        sys = _ScalarDecay(a=1.5)
        make_ctx = _make_ctx_factory(sys, x0=4.0)
        a = jnp.array(1.5)

        _, m1 = simulate_jacfwd(
            sys, make_ctx, (0.0, 2.0), a,
            options=_opts(), record_provenance=True,
        )
        _, m2 = simulate_jacfwd(
            sys, make_ctx, (0.0, 2.0), a,
            options=_opts(), record_provenance=True,
        )

        assert m1.config_hash == m2.config_hash
        assert m1.system["hash"] == m2.system["hash"]
        assert m1.options == m2.options

    def test_different_options_different_config_hash(self):
        """Changing a recorded option (e.g. ``rtol``) flips the
        ``config_hash`` — confirming the hash actually depends on the
        captured configuration."""
        sys = _ScalarDecay(a=1.5)
        make_ctx = _make_ctx_factory(sys, x0=4.0)
        a = jnp.array(1.5)

        _, m_tight = simulate_jacfwd(
            sys, make_ctx, (0.0, 2.0), a,
            options=SimulatorOptions(
                math_backend="jax",
                ode_solver_method="dopri5",
                rtol=1e-9,
                atol=1e-12,
                max_major_steps=200,
            ),
            record_provenance=True,
        )
        _, m_loose = simulate_jacfwd(
            sys, make_ctx, (0.0, 2.0), a,
            options=SimulatorOptions(
                math_backend="jax",
                ode_solver_method="dopri5",
                rtol=1e-6,
                atol=1e-9,
                max_major_steps=200,
            ),
            record_provenance=True,
        )

        assert m_tight.config_hash != m_loose.config_hash
