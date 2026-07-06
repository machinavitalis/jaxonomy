# SPDX-License-Identifier: MIT

"""Regression test for T-002-followup-tracing-downgrade-warn.

Before the fix, ``SimulatorOptions(math_backend="jax", enable_tracing=False)``
silently swapped the backend to numpy with a single ``logger.warning`` (invisible
under the default log config) and the user's ``ode_solver_method="dopri5"`` then
failed late in ``ScipySolver._finalize`` with a message that pointed at the
solver method, not the backend swap.

After the fix, the option-validation pass emits a ``UserWarning`` (visible at
default warning levels) up front, telling the user that the backend was swapped
to numpy and why — so the downstream ``ode_solver_method`` error is no longer
ambush. Backwards compatible: existing callers that intentionally use
``enable_tracing=False`` for eager numpy execution still work; they just see
the warning unless they explicitly set ``math_backend="numpy"``.
"""

from __future__ import annotations

import warnings

import pytest

import jax.numpy as jnp

import jaxonomy


class _Trivial(jaxonomy.LeafSystem):
    def __init__(self):
        super().__init__()
        self.declare_continuous_state(1, ode=self._ode)
        self.declare_continuous_state_output()

    def _ode(self, time, state, **params):
        return -state.continuous_state


def test_jax_with_tracing_disabled_emits_user_warning():
    """The backend swap must be loud — a ``UserWarning`` so the user sees it
    even without log-config tweaking."""
    sys = _Trivial()
    ctx = sys.create_context().with_continuous_state(jnp.array([1.0]))
    opts = jaxonomy.SimulatorOptions(
        math_backend="jax",
        enable_tracing=False,
    )
    with pytest.warns(UserWarning, match="enable_tracing=False"):
        jaxonomy.simulate(sys, ctx, (0.0, 1.0), options=opts)


def test_warning_message_names_the_actual_culprit():
    """The warning text must mention the swap cause (tracing) and the scipy
    solver methods, otherwise users still chase the wrong knob."""
    sys = _Trivial()
    ctx = sys.create_context().with_continuous_state(jnp.array([1.0]))
    opts = jaxonomy.SimulatorOptions(
        math_backend="jax",
        enable_tracing=False,
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        jaxonomy.simulate(sys, ctx, (0.0, 1.0), options=opts)
    swap_warnings = [w for w in caught if "enable_tracing=False" in str(w.message)]
    assert swap_warnings, "expected a UserWarning naming enable_tracing=False"
    msg = str(swap_warnings[0].message)
    assert "numpy" in msg
    assert "scipy" in msg or "dopri5" in msg


def test_explicit_numpy_backend_is_silent():
    """When the user explicitly opts into ``math_backend='numpy'`` with
    ``enable_tracing=False``, no swap-warning should fire — that's the
    documented eager-numpy path."""
    sys = _Trivial()
    ctx = sys.create_context().with_continuous_state(jnp.array([1.0]))
    opts = jaxonomy.SimulatorOptions(
        math_backend="numpy",
        enable_tracing=False,
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        jaxonomy.simulate(sys, ctx, (0.0, 1.0), options=opts)
    swap_warnings = [w for w in caught if "enable_tracing=False" in str(w.message)]
    assert not swap_warnings, (
        f"unexpected swap warning when math_backend='numpy' was explicit: {swap_warnings}"
    )


def test_jax_backend_with_tracing_still_works():
    """The happy path is unchanged."""
    sys = _Trivial()
    ctx = sys.create_context().with_continuous_state(jnp.array([1.0]))
    opts = jaxonomy.SimulatorOptions(math_backend="jax", enable_tracing=True)
    results = jaxonomy.simulate(sys, ctx, (0.0, 1.0), options=opts)
    assert float(results.context.continuous_state[0]) == pytest.approx(
        float(jnp.exp(-1.0)), rel=1e-3
    )
