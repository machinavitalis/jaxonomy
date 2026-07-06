# SPDX-License-Identifier: MIT

"""T-109 phase 4 (diagram-level lift) — ``discretize(diagram, dt, base_context=...)``.

The LTI sub-piece (``discretize(linsys, dt, method)``) was shipped
first. Phase 4 completion makes ``discretize`` polymorphic so callers
can pass a continuous-time system directly and skip the explicit
``linearize`` step.

Two equivalent calls now produce the same result:

    # Explicit linearize → discretize
    linsys = linearize(system, base_context)
    discrete = discretize(linsys, dt)

    # Diagram-level shortcut
    discrete = discretize(system, dt, base_context=base_context)
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from jaxonomy import discretize
from jaxonomy.library import LTISystem, LinearizedSystem, linearize


def _first_order_lti():
    """``G(s) = 1/(s+1)``."""
    sys = LTISystem(
        A=jnp.array([[-1.0]]),
        B=jnp.array([[1.0]]),
        C=jnp.array([[1.0]]),
        D=jnp.array([[0.0]]),
    )
    sys.input_ports[0].fix_value(jnp.array([0.0]))
    return sys


# ---------------------------------------------------------------------------
# Diagram-level path: positional dispatch on first argument type.
# ---------------------------------------------------------------------------


def test_diagram_path_matches_explicit_linearize_then_discretize():
    """``discretize(system, dt, base_context=ctx)`` produces the same
    discrete LinearizedSystem as the two-step form."""
    system = _first_order_lti()
    ctx = system.create_context()
    dt = 0.1

    # Explicit form (works as before — the LTI path).
    linsys = linearize(system, ctx)
    explicit = discretize(linsys, dt)

    # Polymorphic form (phase-4 completion).
    diagram_form = discretize(system, dt, base_context=ctx)

    np.testing.assert_allclose(np.asarray(explicit.A), np.asarray(diagram_form.A))
    np.testing.assert_allclose(np.asarray(explicit.B), np.asarray(diagram_form.B))
    np.testing.assert_allclose(np.asarray(explicit.C), np.asarray(diagram_form.C))
    np.testing.assert_allclose(np.asarray(explicit.D), np.asarray(diagram_form.D))
    assert diagram_form.dt == pytest.approx(dt)
    assert diagram_form.is_discrete()


def test_diagram_path_default_method_is_zoh():
    """Method dispatch should reach the same ZOH path as the LTI form."""
    system = _first_order_lti()
    ctx = system.create_context()
    dt = 0.1

    via_diagram = discretize(system, dt, base_context=ctx)
    via_diagram_explicit_zoh = discretize(system, dt, base_context=ctx, method="zoh")
    np.testing.assert_array_equal(via_diagram.A, via_diagram_explicit_zoh.A)
    np.testing.assert_array_equal(via_diagram.B, via_diagram_explicit_zoh.B)


def test_diagram_path_propagates_method_kwarg():
    """``method='euler'`` should produce the Euler matrices, not ZOH."""
    system = _first_order_lti()
    ctx = system.create_context()
    dt = 0.1

    zoh = discretize(system, dt, base_context=ctx, method="zoh")
    euler = discretize(system, dt, base_context=ctx, method="euler")

    # First-order plant: ZOH gives A_d = exp(-dt) ≈ 0.9048; Euler gives
    # A_d = 1 + (-1)·dt = 0.9. Distinct enough to confirm method dispatch.
    assert not np.allclose(np.asarray(zoh.A), np.asarray(euler.A), atol=1e-3)
    np.testing.assert_allclose(float(euler.A[0, 0]), 0.9, rtol=1e-15)


def test_diagram_path_requires_base_context():
    """Forgetting base_context= must produce a clear error, not a
    silently broken linearization."""
    system = _first_order_lti()
    with pytest.raises(ValueError, match="base_context"):
        discretize(system, 0.1)


# ---------------------------------------------------------------------------
# LTI path remains byte-equivalent.
# ---------------------------------------------------------------------------


def test_lti_path_still_works_unchanged():
    """Pre-existing callers passing a LinearizedSystem must continue
    to work without supplying base_context= (positional dispatch)."""
    linsys = LinearizedSystem(
        A=jnp.array([[-1.0]]),
        B=jnp.array([[1.0]]),
        C=jnp.array([[1.0]]),
        D=jnp.array([[0.0]]),
        operating_point={"x": jnp.zeros(1), "u": jnp.zeros(1)},
    )
    out = discretize(linsys, 0.1)
    assert out.is_discrete()
    assert out.dt == pytest.approx(0.1)


def test_lti_path_ignores_base_context_kwarg():
    """When the first arg is already a LinearizedSystem, base_context
    is ignored (the linearization has already happened)."""
    linsys = LinearizedSystem(
        A=jnp.array([[-1.0]]),
        B=jnp.array([[1.0]]),
        C=jnp.array([[1.0]]),
        D=jnp.array([[0.0]]),
        operating_point={"x": jnp.zeros(1), "u": jnp.zeros(1)},
    )
    a = discretize(linsys, 0.1)
    b = discretize(linsys, 0.1, base_context="anything-here-is-ignored")
    np.testing.assert_array_equal(a.A, b.A)


# ---------------------------------------------------------------------------
# Nonlinear plant: linearize-and-discretize works end-to-end.
# ---------------------------------------------------------------------------


def test_diagram_path_handles_nonlinear_system():
    """Linearizing the downward-pendulum equilibrium then ZOH-discretizing
    gives a stable discrete LTI (|eig| < 1)."""
    from jaxonomy.models.pendulum import Pendulum

    system = Pendulum(m=1.0, L=1.0, b=0.1, input_port=True)
    system.input_ports[0].fix_value(jnp.array([0.0]))
    base = system.create_context()
    down = base.with_continuous_state(jnp.array([0.0, 0.0]))

    out = discretize(system, 0.05, base_context=down)
    assert out.is_discrete()
    # is_stable() on a discrete linsys checks |eig| < 1.
    assert out.is_stable()
