# SPDX-License-Identifier: MIT
"""
Replicated-submodel container block (T-010).

``ReplicatedFunction`` evaluates a submodel callable N times in parallel
using :func:`jax.vmap`.  Inputs can be either single (broadcast to all
N instances) or batched (shape leads with N).  The output is a single
batched array of shape ``(N, *submodel_output_shape)``.

Per-instance vs shared parameters: the submodel is a closure the user
provides, so any parameters baked into the closure are shared across
instances.  To have per-instance parameters, encode them as an
``params`` argument with leading axis N and let the submodel receive
them via the inputs channel (``vmap`` over that axis).

Usage::

    bld = jaxonomy.DiagramBuilder()
    plant = bld.add(MyPlant())
    bld.export_input(plant.input_ports[0], name="u")
    bld.export_output(plant.output_ports[0], name="y")
    submodel = bld.build()

    f = jaxonomy.submodel_function(submodel)
    ctx = submodel.create_context()

    replicated = ReplicatedFunction(
        submodel=lambda u: f(ctx, u),
        n=16,
        n_inputs=1,
        in_axes=(0,),   # input is batched along axis 0; use None to broadcast
    )

The block's input port ``u`` expects either shape ``(16, *u_shape)`` or
``u_shape`` (broadcast); the output port ``y`` emits shape
``(16, *y_shape)``.

Gradients flow through ``vmap`` naturally.  Using this inside a
simulation diagram works so long as the submodel is JAX-traceable
(no ``CustomPythonBlock`` / FMU blocks inside).
"""

from __future__ import annotations

from typing import Callable, Sequence

import jax
import jax.numpy as jnp

from ..framework import LeafSystem


__all__ = ["ReplicatedFunction"]


class ReplicatedFunction(LeafSystem):
    """Container block: evaluate a submodel N times in parallel via vmap.

    Args:
        submodel: Callable ``f(*inputs) -> output``.  Must be
            JAX-traceable so ``vmap`` can transform it.
        n: Number of replicas.
        n_inputs: Number of input ports the block should declare (and
            the number of positional inputs the submodel takes).
        in_axes: Tuple of length ``n_inputs``, matching the ``vmap``
            ``in_axes`` convention: ``0`` means the corresponding input
            is already batched along axis 0; ``None`` means broadcast
            the single-instance input to all N replicas.  Default is
            ``(0,) * n_inputs`` (all inputs batched).
        name: Optional block name.
    """

    def __init__(
        self,
        submodel: Callable,
        n: int,
        n_inputs: int = 1,
        in_axes: Sequence[int | None] | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if n < 1:
            raise ValueError(f"ReplicatedFunction: n must be >= 1, got {n}")
        if n_inputs < 1:
            raise ValueError(
                f"ReplicatedFunction: n_inputs must be >= 1, got {n_inputs}"
            )
        if in_axes is None:
            in_axes = (0,) * n_inputs
        if len(in_axes) != n_inputs:
            raise ValueError(
                f"ReplicatedFunction: len(in_axes) must equal n_inputs "
                f"({n_inputs}), got {len(in_axes)}"
            )
        for ax in in_axes:
            if ax is not None and ax != 0:
                raise ValueError(
                    "ReplicatedFunction: in_axes entries must be 0 or None "
                    f"(got {ax}).  If you need a non-zero axis, transpose "
                    "the input upstream."
                )

        self._n = int(n)
        self._in_axes = tuple(in_axes)
        # axis_size is needed when every in_axes entry is None (all broadcast):
        # JAX vmap cannot infer N from the inputs in that case.
        self._vmapped = jax.vmap(
            submodel, in_axes=self._in_axes, axis_size=self._n,
        )

        for i in range(n_inputs):
            self.declare_input_port(name=f"u_{i}")

        self.declare_output_port(
            self._compute_output,
            prerequisites_of_calc=[port.ticket for port in self.input_ports],
        )

    def _compute_output(self, time, state, *inputs, **params):
        # ``inputs`` are in port-declaration order; shape rules:
        #   in_axes[i] == 0    → inputs[i] must have leading dim N
        #   in_axes[i] is None → inputs[i] is broadcast as a single value
        # We let JAX's vmap rule validate shapes.
        return self._vmapped(*inputs)
