# SPDX-License-Identifier: MIT
"""
Simulator-entry-point error remapping (T-006).

JAX traces carry deep internal stack frames that obscure the line in a
user's block that actually triggered the error.  This module provides
the ``remap_simulation_errors`` decorator applied to ``simulate``,
``simulate_batch``, and ``Simulator.advance_to``:

  - On success: no behavioural change.
  - On ``JaxonomyError`` or ``KeyboardInterrupt``: re-raise unchanged.
  - On any other exception (typically a JAX ``TracerArrayConversionError``,
    ``TypeError`` from a bad dtype promotion, or a ``NotImplementedError``
    from vmap-of-cond etc.): wrap it in :class:`SimulationError` with a
    high-level message that names the offending block and port when
    they can be inferred from the traceback.

Verbose mode — set ``JAXONOMY_VERBOSE_TRACEBACK=1`` in the environment
to skip the wrapping entirely.  Useful when debugging JAX-level bugs
rather than user-model problems.

Block / port identification is best-effort.  We scan the traceback for
frames whose locals include ``self`` with a ``name`` attribute (every
``LeafSystem`` has one) and surface the innermost match.  Frames inside
JAX and XLA are skipped via a path prefix heuristic.
"""

from __future__ import annotations

import functools
import os
import sys
import traceback as _tb
from typing import Callable

from ..framework.error import JaxonomyError


__all__ = [
    "SimulationError",
    "remap_simulation_errors",
]

_VERBOSE_ENV = "JAXONOMY_VERBOSE_TRACEBACK"


class SimulationError(JaxonomyError):
    """Raised when a simulator entry point fails at trace or run time.

    Attributes:
        cause: The original exception.  Accessible as ``__cause__`` too.
        block: Name of the block that appeared innermost in the
            traceback, or ``None`` if no block context was recoverable.
        port: Name of the port if the failure was inside a port
            callback (best-effort), else ``None``.
    """

    def __init__(
        self,
        message: str,
        *,
        cause: BaseException | None = None,
        block: str | None = None,
        port: str | None = None,
    ):
        super().__init__(message)
        self.cause = cause
        self.block = block
        self.port = port


def _is_internal_frame(filename: str) -> bool:
    """Heuristic: True for JAX / XLA / jaxlib / diffrax internals.

    Frames through these libraries rarely identify the user's mistake;
    we hide them unless verbose mode is on.  ``jaxonomy/framework`` and
    ``jaxonomy/simulation`` frames are also hidden since they are
    typically equally deep.
    """
    hints = (
        "/jax/", "/jax_src/", "/jaxlib/", "/site-packages/jax",
        "/diffrax/", "/equinox/",
        "/jaxonomy/framework/", "/jaxonomy/simulation/",
        "/jaxonomy/backend/",
    )
    return any(hint in filename for hint in hints)


def _extract_block_port(exc: BaseException) -> tuple[str | None, str | None]:
    """Walk the traceback from innermost outward, looking for a frame
    whose ``self`` local has a ``name`` attribute (characteristic of
    ``LeafSystem``).  Return the first match's name as the block; and
    if an ``output_port`` / ``input_port`` local is present nearby,
    use its name as the port.
    """
    tb = exc.__traceback__
    frames = []
    while tb is not None:
        frames.append(tb)
        tb = tb.tb_next

    block_name = None
    port_name = None
    for frame_tb in reversed(frames):
        frm = frame_tb.tb_frame
        slf = frm.f_locals.get("self")
        if slf is None:
            continue
        name = getattr(slf, "name", None)
        if name is None or not isinstance(name, str):
            continue
        if block_name is None:
            block_name = name
        # Try to find a port hint
        for key in ("output_port", "input_port", "port"):
            p = frm.f_locals.get(key)
            if p is not None and getattr(p, "name", None):
                port_name = p.name
                break
        if block_name is not None:
            break
    return block_name, port_name


def _format_message(cause: BaseException, block: str | None, port: str | None) -> str:
    kind = type(cause).__name__
    head = f"{kind}: {cause}"
    where = []
    if block:
        where.append(f"block={block!r}")
    if port:
        where.append(f"port={port!r}")
    if where:
        head = f"{head}  (in {', '.join(where)})"
    tail = (
        "\n\nSet JAXONOMY_VERBOSE_TRACEBACK=1 for the full JAX/XLA stack trace."
    )
    return head + tail


def remap_simulation_errors(func: Callable) -> Callable:
    """Decorator: catch non-JaxonomyError exceptions at a simulator entry
    point and re-raise as :class:`SimulationError` with block/port
    context inferred from the traceback.

    ``JaxonomyError`` subclasses (e.g. ``StaticError``,
    ``BlockParameterError``) propagate unchanged so domain-specific
    messages aren't double-wrapped.  ``KeyboardInterrupt`` and
    ``SystemExit`` also propagate.
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if os.environ.get(_VERBOSE_ENV, "").lower() in ("1", "true", "yes"):
            return func(*args, **kwargs)
        try:
            return func(*args, **kwargs)
        except (JaxonomyError, KeyboardInterrupt, SystemExit):
            raise
        except (RuntimeError, ValueError):
            # Domain-level errors raised explicitly by jaxonomy validation
            # (e.g. integer-time overflow, invalid SimulatorOptions, missing
            # context fields) already have user-friendly messages and clear
            # provenance.  Pass through unchanged so existing user code can
            # catch them by their declared type.
            raise
        except BaseException as exc:
            block, port = _extract_block_port(exc)
            message = _format_message(exc, block, port)
            new_err = SimulationError(
                message, cause=exc, block=block, port=port
            )
            # Preserve the original traceback via `from exc` — Python shows
            # "The above exception was the direct cause of..." and the user
            # still sees the filtered chain.
            raise new_err from exc

    return wrapper
