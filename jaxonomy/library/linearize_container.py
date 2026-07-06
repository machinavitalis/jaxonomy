# SPDX-License-Identifier: MIT
"""
Linearize-container helper (T-011).

A "LinearizeContainer" in the classical sense is a container block that
wraps a subdiagram and swaps it for its linearization at build time.
Jaxonomy does not have a first-class container-block concept — it has
``LeafSystem`` and ``Diagram``.  The pragmatic equivalent is a helper
that takes a subdiagram + operating point and returns an
``LTISystem`` block ready to drop into the outer ``DiagramBuilder``
wherever the original subdiagram would have gone.

Usage::

    subdiagram = build_plant_subdiagram()
    op_point = subdiagram.create_context()  # set state/params to taste
    lti_block = jaxonomy.library.linearize_to_lti(subdiagram, op_point)

    outer = jaxonomy.DiagramBuilder()
    outer.add(lti_block)
    # ... wire lti_block.input_ports / output_ports into the larger system
    diagram = outer.build()

The operating point is the same context you'd hand to
:func:`jaxonomy.linearize`.  For a fixed point use the result of
:func:`jaxonomy.trim` (in ``jaxonomy.optimization``) to compute the
operating-point state + inputs before calling this helper.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from ..framework import SystemBase, ContextBase
    from .linear_system import LTISystem

from .linear_system import linearize


__all__ = ["linearize_to_lti"]


def linearize_to_lti(
    system: "SystemBase",
    base_context: "ContextBase",
    input_port=None,
    output_port=None,
    name: Optional[str] = None,
) -> "LTISystem":
    """Linearize ``system`` at ``base_context`` and return an ``LTISystem``.

    Args:
        system: A ``LeafSystem`` or ``Diagram`` to linearize.  If it has
            multiple input or output ports, ``input_port`` and
            ``output_port`` must be supplied explicitly.
        base_context: The operating-point context.  State, inputs, and
            parameters read from this context define the point about
            which the linearization is performed.
        input_port: Input port to linearize against.  Required when
            ``system`` has more than one input port.
        output_port: Output port to linearize against.  Required when
            ``system`` has more than one output port.
        name: Optional name for the returned ``LTISystem`` block.

    Returns:
        An ``LTISystem`` block with the derived ``(A, B, C, D)``
        matrices.  Drop this into a ``DiagramBuilder`` wherever the
        original subdiagram would go; downstream blocks should be
        wired to ``lti.output_ports[0]`` and upstream blocks to
        ``lti.input_ports[0]``.
    """
    lin = linearize(
        system,
        base_context,
        input_port=input_port,
        output_port=output_port,
        name=name,
    )
    return lin.to_lti()
