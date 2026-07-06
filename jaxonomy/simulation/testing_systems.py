import jax.numpy as jnp
from ..framework import LeafSystem

class Decay(LeafSystem):
    """xdot = -k * x; output = x."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.declare_dynamic_parameter("k", 1.0)
        self.declare_continuous_state(
            default_value=jnp.array(1.0), ode=self._ode,
        )
        self.declare_continuous_state_output(name="x")

    def _ode(self, t, state, **p):
        return -p["k"] * state.continuous_state
