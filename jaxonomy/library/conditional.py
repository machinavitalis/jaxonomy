# SPDX-License-Identifier: MIT
"""
Conditional container block (T-009).

``Conditional`` wraps a submodel (typically a subdiagram exposed via
:func:`jaxonomy.submodel_function` or a plain JAX-traceable callable)
and controls whether its output is visible based on a boolean enable
input.  When the submodel is disabled, three output behaviours are
supported:

- ``"reset"``   — output emits a fixed value (default 0).
- ``"hold"``    — output holds the last value produced while enabled;
                  held across the disabled stretch as a discrete state.
- ``"passthrough"`` — output equals the first forwarded input (useful
                  as an explicit bypass, and serves as the
                  "not-clocked-down marker" path for recording: the
                  forwarded input is the user's choice of sentinel).

All three modes are autodiff-safe: the disabled path uses ``jnp.where``,
so gradients through the disabled branch are zero rather than undefined.

The submodel is evaluated on every step regardless of enable state (JAX
cannot truly skip a traced computation); only the output is masked.
Users who need to avoid submodel computation on disabled steps should
use ``jax.lax.cond`` at the application level instead.

Usage::

    bld = jaxonomy.DiagramBuilder()
    plant = bld.add(MyPlant())                # or any subdiagram
    bld.export_input(plant.input_ports[0], name="u")
    bld.export_output(plant.output_ports[0], name="y")
    submodel = bld.build()
    f = jaxonomy.submodel_function(submodel)

    cond = Conditional(
        submodel=lambda ctx, u: f(ctx, u),
        n_inputs=1,
        when_disabled="hold",
        initial_value=jnp.array(0.0),
    )

The typical pattern in a bigger diagram is:

    outer = jaxonomy.DiagramBuilder()
    plant_block = outer.add(cond)
    # Connect enable signal to input 0, user input to input 1, etc.
    outer.connect(enable_source.output_ports[0], plant_block.input_ports[0])
    outer.connect(u_source.output_ports[0], plant_block.input_ports[1])
    ...
"""

from __future__ import annotations

from typing import Callable, Iterable

import jax.numpy as jnp

from ..framework import LeafSystem


__all__ = ["Conditional", "WhenDisabled"]


class WhenDisabled:
    """Allowed string values for the ``when_disabled`` kwarg."""

    RESET = "reset"
    HOLD = "hold"
    PASSTHROUGH = "passthrough"

    @classmethod
    def valid(cls) -> tuple[str, ...]:
        return (cls.RESET, cls.HOLD, cls.PASSTHROUGH)


class Conditional(LeafSystem):
    """Container block that enables/disables a submodel.

    Args:
        submodel: Callable taking ``*inputs`` and returning a single
            output array.  For a subdiagram, use
            ``jaxonomy.submodel_function`` to build a compatible
            callable, then wrap with a context-capturing lambda:
            ``lambda *u: f(context, *u)``.
        n_inputs: Number of non-enable inputs the submodel takes.
            Input port 0 is always the enable signal; ports 1..n_inputs
            carry the submodel's inputs in order.
        when_disabled: ``"reset"``, ``"hold"``, or ``"passthrough"``.
        initial_value: Output value when disabled (reset or hold mode's
            initial state).  Also used to infer output shape/dtype when
            the submodel has not been evaluated yet.
        name: Optional block name.
    """

    def __init__(
        self,
        submodel: Callable,
        n_inputs: int = 1,
        when_disabled: str = WhenDisabled.RESET,
        initial_value=0.0,
        hold_period: float | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        if when_disabled not in WhenDisabled.valid():
            raise ValueError(
                f"Conditional: when_disabled must be one of "
                f"{WhenDisabled.valid()!r}, got {when_disabled!r}"
            )
        if when_disabled == WhenDisabled.HOLD and not hold_period:
            raise ValueError(
                "Conditional(when_disabled='hold') requires a positive "
                "hold_period to determine the snapshot sample rate."
            )

        self._submodel = submodel
        self._when_disabled = when_disabled
        self._initial = jnp.asarray(initial_value)

        # Port 0 is always enable; remaining ports are submodel inputs.
        self.declare_input_port(name="enable")
        for i in range(n_inputs):
            self.declare_input_port(name=f"u_{i}")

        if when_disabled == WhenDisabled.HOLD:
            # Discrete state holding the last snapshot of the submodel
            # output while the block was enabled.  Updated at the
            # user-supplied ``hold_period``; between snapshots the output
            # reads from the last store.
            self.declare_discrete_state(default_value=self._initial)
            self.declare_periodic_update(
                self._hold_update, period=float(hold_period), offset=0.0,
            )
        self.declare_output_port(
            self._compute_output,
            prerequisites_of_calc=[port.ticket for port in self.input_ports],
        )

    # ── callbacks ─────────────────────────────────────────────────────────

    def _submodel_output(self, inputs):
        """Evaluate the wrapped submodel on the forwarded inputs."""
        user_inputs = inputs[1:]  # skip enable
        return jnp.asarray(self._submodel(*user_inputs))

    def _hold_update(self, time, state, *inputs, **params):
        enable = inputs[0]
        y_sub = self._submodel_output(inputs)
        # On disabled steps, keep the previous held value.
        return jnp.where(
            jnp.asarray(enable).astype(bool),
            y_sub,
            state.discrete_state,
        )

    def _compute_output(self, time, state, *inputs, **params):
        enable = jnp.asarray(inputs[0]).astype(bool)
        y_sub = self._submodel_output(inputs)

        if self._when_disabled == WhenDisabled.RESET:
            return jnp.where(enable, y_sub, self._initial)

        if self._when_disabled == WhenDisabled.HOLD:
            held = state.discrete_state
            return jnp.where(enable, y_sub, held)

        # passthrough: output = first user input when disabled.  This
        # requires the submodel output and the first user input to have
        # compatible shapes.
        passthrough = jnp.asarray(inputs[1]) if len(inputs) > 1 else self._initial
        return jnp.where(enable, y_sub, passthrough)
