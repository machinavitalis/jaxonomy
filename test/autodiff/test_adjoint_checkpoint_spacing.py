# SPDX-License-Identifier: MIT

"""Adjoint accuracy over horizons much longer than a decaying time constant.

The backward pass of ``_odeint`` re-integrates the state backward in time
from each forward checkpoint.  A state that decays forward grows backward,
so drift between checkpoints is amplified roughly as exp(gap / tau), and once
the state enters the adjoint's Jacobian the gradient is lost (KNOWN_GAPS,
"Autodiff — long horizons need more adjoint checkpoints").

The model is a nonlinear decay driven by a parameter, tau ~ 1 s, over a
200 s horizon.  With 64 checkpoints the gap stays a few tau and the gradient
matches a central finite difference; the default 16 do not, which is the
documented limitation (strict xfail, so a fix shows up here).
"""

import jax
import jax.numpy as jnp
import pytest

import jaxonomy
from jaxonomy import SimulatorOptions
from jaxonomy.testing.markers import requires_jax, skip_if_not_jax

skip_if_not_jax()

TF = 200.0
P0 = 1.0
FD_H = 1e-5


class DrivenDecay(jaxonomy.LeafSystem):
    """x' = -x (1 + x^2) + p (1 + sin t): decays with tau ~ 1 s."""

    def __init__(self, name="decay"):
        super().__init__(name=name)
        self.declare_dynamic_parameter("p", P0)
        self.declare_continuous_state(default_value=jnp.array(0.0), ode=self.ode)

    def ode(self, time, state, **params):
        x = state.continuous_state
        return -x * (1.0 + x**2) + params["p"] * (1.0 + jnp.sin(time))


MODEL = DrivenDecay()


def _grad_and_fd(max_checkpoints):
    opts = SimulatorOptions(
        enable_autodiff=True,
        max_major_steps=4,
        max_checkpoints=max_checkpoints,
        rtol=1e-10,
        atol=1e-12,
    )
    ctx0 = MODEL.create_context()

    @jax.jit
    def fwd(p):
        ctx = ctx0.with_parameter("p", p)
        return jaxonomy.simulate(MODEL, ctx, (0.0, TF), options=opts).context.continuous_state

    grad = float(jax.grad(fwd)(jnp.float64(P0)))
    fd = float((fwd(jnp.float64(P0 + FD_H)) - fwd(jnp.float64(P0 - FD_H))) / (2 * FD_H))
    return grad, fd


@requires_jax()
def test_adjoint_accurate_with_closely_spaced_checkpoints():
    grad, fd = _grad_and_fd(64)
    assert grad == pytest.approx(fd, rel=1e-6)


@requires_jax()
@pytest.mark.xfail(
    strict=True,
    reason="known gap: default checkpoints too sparse for a 200-tau horizon",
)
def test_adjoint_accurate_with_default_checkpoints():
    grad, fd = _grad_and_fd(16)
    assert grad == pytest.approx(fd, rel=1e-6)
