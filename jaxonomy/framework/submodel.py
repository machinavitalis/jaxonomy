# SPDX-License-Identifier: MIT
"""
Submodel-as-pure-function public API (T-008).

Wraps the existing ``port.eval(context)`` mechanism as a documented,
pure, autodiff- and vmap-compatible function.  The intended use cases
are MPC cost functions, RL environment step functions, and any
higher-order construct that needs a submodel as an ordinary JAX-traceable
callable rather than a simulator run.

Preparation.  The API expects a ``Diagram`` (or ``LeafSystem``) whose
unconnected input ports have been exported at the diagram level
(``builder.export_input``) and, before ``create_context`` is called,
seeded with placeholder values so the diagram's type inference succeeds.
In practice:

    bld = jaxonomy.DiagramBuilder()
    plant = bld.add(MyPlant())
    bld.export_input(plant.input_ports[0], name="u")
    bld.export_output(plant.output_ports[0], name="y")
    diagram = bld.build()
    for p in diagram.input_ports:
        p.fix_value(jnp.zeros(p.default_value.shape if p.default_value is not None else ()))
    context = diagram.create_context()

    f = jaxonomy.submodel_function(diagram)
    y = f(context, u)

The ``submodel_function`` helper itself supplies default zero-placeholders
when the caller didn't pre-seed the ports, so the snippet above collapses
to::

    f = jaxonomy.submodel_function(diagram)   # auto-placeholders if unfixed
    context = diagram.create_context()
    y = f(context, u)

Signature: ``f(context, *inputs) -> outputs``.  The callable:

  - evaluates the requested output ports,
  - fixes the requested input ports to the supplied values for the
    duration of the call and unfixes them on return,
  - is ``jax.grad`` / ``jax.jit`` / ``jax.vmap`` compatible (the
    supplied inputs participate as traced values; the port's
    ``_callback`` captures a closure over the tracer but is cleared by
    the ``finally`` clause before the trace exits),
  - returns a single array when one output port is selected, a tuple
    in declaration order otherwise.
"""

from __future__ import annotations

from contextlib import ExitStack
from typing import TYPE_CHECKING, Callable, Sequence

import jax.numpy as jnp

if TYPE_CHECKING:
    from .context import ContextBase
    from .port import InputPort, OutputPort
    from .system_base import SystemBase


__all__ = ["submodel_function"]


def _seed_placeholder(port):
    """If ``port`` is not fixed and has no upstream, fix it to zeros so
    ``create_context`` can evaluate the port during type inference.

    No-op if the port is already fixed or is an internal port that will
    be evaluated from an upstream connection.
    """
    if port.is_fixed:
        return
    shape = ()
    dtype = jnp.float64
    if port.default_value is not None:
        arr = jnp.asarray(port.default_value)
        shape, dtype = arr.shape, arr.dtype
    port.fix_value(jnp.zeros(shape, dtype=dtype))


def submodel_function(
    system: "SystemBase",
    output_ports: "Sequence[OutputPort] | None" = None,
    input_ports: "Sequence[InputPort] | None" = None,
    auto_seed: bool = True,
) -> Callable:
    """Wrap ``system``'s ports as a pure function of (context, *inputs).

    Args:
        system: The ``LeafSystem`` or ``Diagram`` to wrap.
        output_ports: Output ports whose values to return.  Defaults to
            all of ``system.output_ports``.
        input_ports: Input ports that the closure will feed.  Defaults to
            all of ``system.input_ports``.  Inputs not listed here are
            assumed already connected or fixed.
        auto_seed: If True (default), any input port in ``input_ports``
            that is not already fixed or connected is pre-fixed to a
            zero placeholder so ``create_context`` succeeds on systems
            with dangling exported inputs.  Set to False if you have
            seeded placeholders yourself.

    Returns:
        ``f(context, *inputs) -> outputs``.  When a single output port
        is selected the return is a scalar / array; otherwise a tuple
        in ``output_ports`` declaration order.

    Example::

        bld = jaxonomy.DiagramBuilder()
        plant = bld.add(MyPlant())
        bld.export_input(plant.input_ports[0], name="u")
        bld.export_output(plant.output_ports[0], name="y")
        diagram = bld.build()

        f = jaxonomy.submodel_function(diagram)
        ctx = diagram.create_context()    # auto-seeded placeholders
        y = f(ctx, u)
        dy_du = jax.grad(lambda u: f(ctx, u))(u0)
        y_batch = jax.vmap(f, in_axes=(None, 0))(ctx, u_batch)

    Performance envelope (T-008, follow-up finding 2026-05-16):
        Each call invokes the diagram's full evaluation machinery —
        port-fix context managers, dependency-tracked output evaluation,
        cache invalidation. That overhead is fine for **one-shot
        rollouts**, **batched evaluation** (where the cost amortises
        across the batch via ``jax.vmap``), and **gradient computation
        via ``jax.grad``** (the closure is traced once, then the
        compiled XLA program runs at native speed).

        It is **not** fine for tight Python-side loops that call ``f``
        thousands of times per simulated second — typical MPC inner
        loops where the prediction model is re-evaluated at every
        sample of a ``jax.lax.scan``-style rollout. There the per-call
        Python overhead dominates and the wall-clock blows up by 100×
        or more relative to closing over the underlying primitive
        directly (e.g. ``interp_2d``, ``lookup_table_nd``, or a
        hand-rolled JAX function). The canonical workaround in that
        case is to skip ``submodel_function`` entirely for the inner
        loop and call the primitive directly inside the scan body. See
        ``docs/examples/engine_map_fitting_to_mpc.ipynb`` for an
        example of the hand-rolled-scan pattern.

        Rule of thumb: if the closure will be invoked from a
        Python-level loop more than ~100 times per simulation, profile
        first. ``jax.jit(f)`` + ``jax.vmap`` over the entire batch
        usually beats a Python loop by orders of magnitude.
    """
    out_ports = tuple(output_ports) if output_ports is not None else tuple(system.output_ports)
    in_ports = tuple(input_ports) if input_ports is not None else tuple(system.input_ports)

    if not out_ports:
        raise ValueError(
            f"submodel_function({system.name!r}): the system has no output ports "
            "to evaluate.  Supply output_ports= explicitly if you want to evaluate "
            "intermediate ports."
        )

    if auto_seed:
        for p in in_ports:
            _seed_placeholder(p)

    def _call(context: "ContextBase", *inputs):
        if len(inputs) != len(in_ports):
            raise TypeError(
                f"submodel_function({system.name!r}) expected {len(in_ports)} "
                f"input values (one per input port), got {len(inputs)}."
            )
        with ExitStack() as stack:
            for port, value in zip(in_ports, inputs):
                stack.enter_context(port.fixed(value))
            ys = tuple(p.eval(context) for p in out_ports)
        return ys[0] if len(ys) == 1 else ys

    _call.__name__ = f"submodel_{system.name}"
    _call.__doc__ = (
        f"Evaluate {system.name!r} as a pure function of inputs "
        f"{[p.name for p in in_ports]} → outputs "
        f"{[p.name for p in out_ports]}.\n\n"
        "Signature: f(context, *inputs) -> outputs.  See "
        "jaxonomy.submodel_function for details."
    )
    return _call
