# SPDX-License-Identifier: MIT

"""Adjoint checkpointing must cover the full horizon — the tail-skip bug.

``_odeint``'s forward pass stored the final state at
``jnp.maximum(index + 1, max_checkpoints - 1)``: when the checkpoint array was
exactly full (``index + 1 == max_checkpoints``) that produced an out-of-bounds
scatter which JAX silently drops, so ``ts[-1] < tf`` and the backward sweep
never integrated the costate over the final ``(ts[-1], tf]`` sub-interval —
precisely where the adjoint (initialized to dJ/dx(T)) is largest.  Whether the
bug fired depended on the forward step count landing the checkpoint index
exactly at full, which is why the resulting gradient error looked erratically
tolerance- and problem-dependent from the outside (jaxterity's
``double_pendulum-damping-dopri5-float64`` was ~2-3.5% off at rtol=1e-8 with
``max_minor_step_size=0.01``, while nearby configurations were exact).

The sweep below varies ``tf`` with a capped uniform step so the step count
walks across the checkpoint-full pattern; every gradient must match a central
finite difference tightly, for both the dopri5 and bdf adjoints (the
checkpoint machinery is solver-independent).
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import jaxonomy
from jaxonomy import SimulatorOptions
from jaxonomy.testing.markers import requires_jax, skip_if_not_jax

skip_if_not_jax()

G = 9.81
X0 = jnp.array([0.4, 0.2, 0.0, 0.0])


class DampedDoublePendulum(jaxonomy.LeafSystem):
    """Point-mass double pendulum (m=l=1) with joint damping ``b``."""

    def __init__(self, b=0.5, name="dp"):
        super().__init__(name=name)
        self.declare_dynamic_parameter("b", float(b))
        self.declare_continuous_state(default_value=X0, ode=self.ode)
        self.declare_continuous_state_output(name="x")

    def ode(self, time, state, **params):
        th1, th2, w1, w2 = state.continuous_state
        b = params["b"]
        d = th1 - th2
        den = 3.0 - jnp.cos(2.0 * d)
        dw1 = (
            -3.0 * G * jnp.sin(th1)
            - G * jnp.sin(th1 - 2.0 * th2)
            - 2.0 * jnp.sin(d) * (w2**2 + w1**2 * jnp.cos(d))
        ) / den - b * w1
        dw2 = (
            2.0 * jnp.sin(d) * (2.0 * w1**2 + 2.0 * G * jnp.cos(th1) + w2**2 * jnp.cos(d))
        ) / den - b * w2
        return jnp.array([w1, w2, dw1, dw2])


MODEL = DampedDoublePendulum()
B0 = 0.5
FD_H = 1e-4


def _make_fwd(method, tf):
    opts = SimulatorOptions(
        math_backend="jax",
        ode_solver_method=method,
        rtol=1e-8,
        atol=1e-10,
        enable_autodiff=True,
        max_minor_step_size=0.01,  # uniform capped steps -> controlled step count
        max_major_steps=300,
    )
    ctx0 = MODEL.create_context()

    def fwd(b):
        ctx = ctx0.with_parameter("b", b)
        res = jaxonomy.simulate(MODEL, ctx, (0.0, tf), options=opts)
        return jnp.sum(res.context.continuous_state**2)

    return fwd


# tf sweep: with hmax=0.01 the forward step count walks 57..63, crossing the
# exactly-full checkpoint pattern (16 slots, depth doubling) at least once —
# tf=0.60 is the configuration measured at 3.5% gradient error pre-fix.
@requires_jax()
@pytest.mark.parametrize("tf", [0.57, 0.60, 0.63])
def test_dopri5_adjoint_covers_tail_interval(tf):
    fwd = jax.jit(_make_fwd("dopri5", tf))
    grad = float(jax.grad(fwd)(jnp.float64(B0)))
    fd = float(
        (fwd(jnp.float64(B0 + FD_H)) - fwd(jnp.float64(B0 - FD_H))) / (2 * FD_H)
    )
    assert grad == pytest.approx(fd, rel=1e-5), (
        f"dopri5 adjoint gradient {grad:+.10f} vs FD {fd:+.10f} at tf={tf} "
        "(tail checkpoint interval skipped?)"
    )


@requires_jax()
def test_bdf_adjoint_covers_tail_interval():
    # Same machinery, different solver: the checkpoint tail-skip hit BDF too.
    fwd = jax.jit(_make_fwd("bdf", 0.60))
    grad = float(jax.grad(fwd)(jnp.float64(B0)))
    fd = float(
        (fwd(jnp.float64(B0 + FD_H)) - fwd(jnp.float64(B0 - FD_H))) / (2 * FD_H)
    )
    assert grad == pytest.approx(fd, rel=1e-4)
