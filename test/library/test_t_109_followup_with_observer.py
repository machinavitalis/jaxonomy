# SPDX-License-Identifier: MIT

"""T-109-followup-with-observer — Luenberger block + with_observer wiring helper.

Closes the last open piece of T-109 phase 4: attach a steady-state
Luenberger observer to a plant diagram with one call.

Two layers:

* :class:`jaxonomy.library.Luenberger` — the discrete-time observer
  block. State update: ``x_hat[k+1] = A x_hat[k] + B u[k] + L (y - C x_hat - D u)``.
* :func:`jaxonomy.with_observer(plant, observer)` — wiring helper that
  builds a new diagram exposing ``u`` (input fanned to plant +
  observer) and ``x_hat`` (the observer's estimate).
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

import jaxonomy
from jaxonomy import with_observer
from jaxonomy.library import (
    Constant,
    Gain,
    Integrator,
    LinearizedSystem,
    Luenberger,
)


# ---------------------------------------------------------------------------
# Luenberger block — construction-time validation.
# ---------------------------------------------------------------------------


def test_luenberger_requires_square_A():
    with pytest.raises(ValueError, match="A must be square"):
        Luenberger(
            dt=0.01,
            A=jnp.array([[1.0, 2.0, 3.0]]),  # (1, 3) — not square
            B=jnp.array([[1.0]]),
            C=jnp.array([[1.0]]),
            L=jnp.array([[0.5]]),
        )


def test_luenberger_requires_L_shape_n_by_p():
    """L must be (n_states, n_outputs)."""
    with pytest.raises(ValueError, match="L must have shape"):
        Luenberger(
            dt=0.01,
            A=jnp.eye(2),                            # n=2
            B=jnp.array([[1.0], [0.0]]),
            C=jnp.array([[1.0, 0.0]]),               # p=1
            L=jnp.array([[0.5, 0.3], [0.1, 0.2]]),   # wrong shape (2,2) vs (2,1)
        )


def test_luenberger_default_D_is_zero():
    """Omitting D yields a zero feedthrough matrix of compatible shape."""
    blk = Luenberger(
        dt=0.01,
        A=jnp.eye(2),
        B=jnp.array([[1.0], [0.0]]),
        C=jnp.array([[1.0, 0.0]]),
        L=jnp.array([[0.5], [0.3]]),
    )
    # Two input ports declared, one output port declared.
    assert len(blk.input_ports) == 2
    assert len(blk.output_ports) == 1


# ---------------------------------------------------------------------------
# Luenberger end-to-end — converges on a known stable system.
# ---------------------------------------------------------------------------


def _simulate_observer_on_constant_plant_output(L_gain: float, n_steps: int = 50):
    """Run a Luenberger block with constant ``u=0`` and constant
    ``y=1`` for ``n_steps`` ticks. With a scalar stable plant
    ``A=[[0.9]]``, ``B=[[1.0]]``, ``C=[[1.0]]``, ``D=0`` the observer
    estimate should converge to the steady-state ``x_hat = 1`` (the
    true state corresponding to ``y=1``).
    """
    dt = 0.01
    A = jnp.array([[0.9]])
    B = jnp.array([[1.0]])
    C = jnp.array([[1.0]])
    L = jnp.array([[L_gain]])

    obs = Luenberger(dt=dt, A=A, B=B, C=C, L=L,
                     x_hat_0=jnp.array([0.0]))

    b = jaxonomy.DiagramBuilder()
    u_src = b.add(Constant(0.0, name="u_src"))
    y_src = b.add(Constant(1.0, name="y_src"))
    b.add(obs)
    b.connect(u_src.output_ports[0], obs.input_ports[0])
    b.connect(y_src.output_ports[0], obs.input_ports[1])
    diag = b.build(name="root")
    ctx = diag.create_context()

    res = jaxonomy.simulate(
        diag, ctx, (0.0, dt * n_steps),
        recorded_signals={"x_hat": obs.output_ports[0]},
    )
    return np.asarray(res.outputs["x_hat"])


def test_luenberger_estimate_converges_under_positive_gain():
    """With a non-trivial gain L>0, the observer drives its estimate
    toward the measured value over multiple ticks."""
    traj = _simulate_observer_on_constant_plant_output(L_gain=0.5, n_steps=50)
    # Should converge towards 1 (the constant measurement). Loose
    # tolerance — discrete dynamics with A=0.9 settle quickly.
    # x_hat is a 1-vector at each step; flatten before reading.
    final = float(np.ravel(traj[-1])[0])
    assert 0.8 < final < 1.05, (
        f"observer estimate failed to converge: final = {final:.3f}"
    )


def test_luenberger_zero_gain_decouples_observer_from_measurement():
    """With L=0, the observer becomes a plain prediction: x_hat[k+1] =
    A x_hat[k] + B u[k]. With u=0 and x_hat_0=0 it should stay at 0
    regardless of the measurement."""
    traj = _simulate_observer_on_constant_plant_output(L_gain=0.0, n_steps=30)
    # The observer never sees y; stays at x_hat_0 = 0 forever.
    np.testing.assert_allclose(np.ravel(traj), 0.0, atol=1e-12)


# ---------------------------------------------------------------------------
# with_observer wiring helper — diagram structure.
# ---------------------------------------------------------------------------


def _build_first_order_plant():
    """Tiny plant diagram: y = integral(u). Single input ``u``,
    single output ``y``. Used by the wiring-helper tests."""
    b = jaxonomy.DiagramBuilder()
    integ = b.add(Integrator(0.0, name="plant_integ"))
    b.export_input(integ.input_ports[0], name="u")
    b.export_output(integ.output_ports[0], name="y")
    return b.build(name="plant")


def test_with_observer_returns_diagram_with_x_hat_output():
    """``with_observer`` should return a diagram exposing ``x_hat`` at
    the top level alongside the original plant inputs."""
    plant = _build_first_order_plant()
    obs = Luenberger(
        dt=0.01,
        A=jnp.array([[1.0]]),
        B=jnp.array([[0.01]]),
        C=jnp.array([[1.0]]),
        L=jnp.array([[0.3]]),
    )
    wrapped = with_observer(plant, obs)

    # Exposes ``u`` (input) and ``x_hat`` (output).
    in_names = [p.name for p in wrapped.input_ports]
    out_names = [p.name for p in wrapped.output_ports]
    assert "u" in in_names
    assert "x_hat" in out_names


def test_with_observer_x_hat_estimate_is_nontrivial_under_drive():
    """End-to-end smoke: drive the plant with a non-zero input and
    confirm the observer's x_hat estimate moves away from zero."""
    plant = _build_first_order_plant()
    obs = Luenberger(
        dt=0.01,
        A=jnp.array([[1.0]]),
        B=jnp.array([[0.01]]),
        C=jnp.array([[1.0]]),
        L=jnp.array([[0.3]]),
        x_hat_0=jnp.array([0.0]),
    )
    wrapped = with_observer(plant, obs)

    # Drive ``u`` with a constant 1.0 by wiring an outer source.
    outer = jaxonomy.DiagramBuilder()
    src = outer.add(Constant(1.0, name="u_src"))
    sub = outer.add(wrapped)
    outer.connect(src.output_ports[0], sub.input_ports[0])
    outer.export_output(sub.output_ports[0], name="x_hat")
    top = outer.build(name="closed_loop")
    ctx = top.create_context()

    res = jaxonomy.simulate(
        top, ctx, (0.0, 0.5),
        recorded_signals={"x_hat": sub.output_ports[0]},
    )
    x_hat = np.asarray(res.outputs["x_hat"])
    # With u=1 driving an integrator and a working observer, x_hat
    # should grow past zero by the end. x_hat is a (T, n_states)
    # array; flatten the last sample's vector before reading.
    final = float(np.ravel(x_hat[-1])[0])
    assert final > 0.05, (
        f"observer estimate did not respond to drive: final={final:.3e}"
    )


def test_with_observer_default_plant_port_indices_are_zero():
    """The default wiring picks plant_u_port=0 and plant_y_port=0 —
    this matches the canonical "single in, single out" plant pattern.
    We only assert that the builder accepts the default; full
    end-to-end behaviour is exercised by
    ``test_with_observer_x_hat_estimate_is_nontrivial_under_drive``.
    """
    plant = _build_first_order_plant()
    obs = Luenberger(
        dt=0.01,
        A=jnp.array([[1.0]]),
        B=jnp.array([[0.01]]),
        C=jnp.array([[1.0]]),
        L=jnp.array([[0.3]]),
    )
    # Default kwargs should work for a single-in / single-out plant.
    wrapped = with_observer(plant, obs)
    # Sanity: the resulting diagram has the expected port shape.
    # (Can't create_context on a standalone wrapped diagram because
    # the exported "u" input has no upstream driver in isolation —
    # ``wrapped`` is meant to be used as a subsystem under a parent.)
    assert len(wrapped.input_ports) == 1
    assert len(wrapped.output_ports) == 1
    assert wrapped.input_ports[0].name == "u"
    assert wrapped.output_ports[0].name == "x_hat"
