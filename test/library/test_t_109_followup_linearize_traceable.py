# SPDX-License-Identifier: MIT

"""Tests for T-109-followup-linearize-traceable.

Pre-fix, :func:`linearize` body contained two host-side guards that
forced concrete-value extraction:

* ``bool(jnp.all(jnp.isfinite(xc0_flat)))`` — finite-state pre-check
* ``float(jnp.max(jnp.abs(xdot0_flat)))`` — equilibrium-residual warning

Either would trip under ``jax.jit`` / ``jax.grad`` / ``jax.vmap`` with
a ``TracerBoolConversionError`` (or similar), so the linearization
path could not be embedded in any composed JAX transformation.
Downstream helpers (``frequency_response``, ``bode_data``, etc.) were
already traceable, so this closes the T-109 differentiability story
end-to-end.

Post-fix the guards are gated behind ``jax.core.is_concrete`` and are
no-ops when called under a trace; A/B/C/D Jacobians are computed
identically in both paths.

Tests:
* :func:`linearize` runs under ``jax.jit`` end-to-end on a simple
  pendulum at equilibrium.
* :func:`linearize` runs under ``jax.grad`` of a downstream scalar
  function of A; the gradient is finite and shape-correct.
* Eager-mode equilibrium-warning behaviour is unchanged (still fires
  when called outside any trace and the residual exceeds threshold).
* Eager-mode finite-state ``ValueError`` is unchanged.
* Traced path silently skips the equilibrium warning (diagnostic, not
  invariant) but still returns matching A/B/C/D.
"""

from __future__ import annotations

import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaxonomy.library import linearize
from jaxonomy.models.pendulum import Pendulum


pytestmark = pytest.mark.minimal


# ---------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------


def _make_pendulum_context():
    """Pendulum at the stable equilibrium (theta=0, omega=0, u=0)."""
    sys = Pendulum(m=1.0, L=1.0, b=0.0, input_port=True)
    sys.input_ports[0].fix_value(jnp.array([0.0]))
    base_ctx = sys.create_context()
    eq_ctx = base_ctx.with_continuous_state(jnp.array([0.0, 0.0]))
    return sys, eq_ctx


# ---------------------------------------------------------------------
# Traceability tests
# ---------------------------------------------------------------------


class TestLinearizeTraceable:
    def test_linearize_under_jit(self):
        """``jax.jit(linearize)`` runs end-to-end on a pendulum.

        Pre-fix this raised ``TracerBoolConversionError`` from the
        ``bool(jnp.all(jnp.isfinite(...)))`` host-side guard.
        """
        sys, base_ctx = _make_pendulum_context()

        def _linearize_at(x_seed):
            new_ctx = base_ctx.with_continuous_state(x_seed)
            return linearize(sys, new_ctx).A

        x0 = jnp.array([0.0, 0.0])
        A_eager = _linearize_at(x0)
        A_jit = jax.jit(_linearize_at)(x0)
        np.testing.assert_allclose(
            np.asarray(A_jit), np.asarray(A_eager), atol=1e-12,
        )

    def test_linearize_under_grad(self):
        """``jax.grad`` of a scalar function of ``linearize(...).A``
        returns a finite, shape-correct gradient."""
        sys, base_ctx = _make_pendulum_context()

        def _scalar_of_A(x_seed):
            new_ctx = base_ctx.with_continuous_state(x_seed)
            return jnp.sum(linearize(sys, new_ctx).A ** 2)

        x0 = jnp.array([0.0, 0.0])
        g = jax.grad(_scalar_of_A)(x0)
        assert g.shape == x0.shape
        assert np.all(np.isfinite(np.asarray(g)))

    def test_linearize_eager_equilibrium_warning_unchanged(self):
        """The equilibrium-residual warning still fires when called
        eagerly and the residual exceeds threshold."""
        sys, base_ctx = _make_pendulum_context()
        # Off-equilibrium: theta=pi/4 → sin(pi/4) != 0 → residual nonzero
        far_ctx = base_ctx.with_continuous_state(jnp.array([np.pi / 4.0, 0.0]))
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            linearize(sys, far_ctx)
        eq_warns = [
            w for w in caught
            if issubclass(w.category, UserWarning)
            and "does not appear to be an equilibrium" in str(w.message)
        ]
        assert len(eq_warns) >= 1, (
            "Eager-mode equilibrium warning must still fire on a "
            "non-equilibrium operating point."
        )

    def test_linearize_eager_finite_check_unchanged(self):
        """The finite-state ``ValueError`` still fires eagerly."""
        sys, base_ctx = _make_pendulum_context()
        nan_ctx = base_ctx.with_continuous_state(
            jnp.array([float("nan"), 0.0]),
        )
        with pytest.raises(ValueError, match="non-finite"):
            linearize(sys, nan_ctx)

    def test_linearize_traced_path_skips_warning_silently(self):
        """Under ``jit`` the equilibrium warning is suppressed even
        for an off-equilibrium seed — it's a diagnostic, not an
        invariant. The Jacobians still match the eager call."""
        sys, base_ctx = _make_pendulum_context()

        def _A_at(x_seed):
            new_ctx = base_ctx.with_continuous_state(x_seed)
            return linearize(sys, new_ctx).A

        x_far = jnp.array([np.pi / 4.0, 0.0])
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            A_jit = jax.jit(_A_at)(x_far)
        eq_warns = [
            w for w in caught
            if "does not appear to be an equilibrium" in str(w.message)
        ]
        assert eq_warns == [], (
            "Equilibrium warning should be suppressed under jax.jit; "
            f"got: {[str(w.message) for w in eq_warns]}"
        )
        A_eager = _A_at(x_far)
        np.testing.assert_allclose(
            np.asarray(A_jit), np.asarray(A_eager), atol=1e-12,
        )
