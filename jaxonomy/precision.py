# SPDX-License-Identifier: MIT
"""
Precision introspection utility (T-005) and global precision-policy
context manager (T-038a-followup-mixed-precision-cascade).

Jaxonomy enables ``JAX_ENABLE_X64`` by default at import time
(``jaxonomy/_init.py``). Users who opt into float32 must set the env var
*before* importing jaxonomy.  The function :func:`precision_info` returns
the current resolved precision so users can assert their expectation in
a script.

For the full policy, including solver error bounds per (solver, dtype),
see ``test/precision/POLICY.md``.

T-038a-followup-mixed-precision-cascade
---------------------------------------
The follow-up issue: end-to-end f32 simulations require setting
``dtype=`` on every block, which is boilerplate.  :func:`precision_policy`
is a ``contextvars.ContextVar``-backed context manager that lets a user
declare a *default* dtype for blocks built inside the ``with`` block:

    with jaxonomy.precision_policy(jnp.float32):
        sys = MyDiagram()  # all dtype-aware blocks default to float32
        result = jaxonomy.simulate(sys, ctx, ...)

Per-block ``dtype=`` kwargs from T-038a take precedence over the policy
(explicit-over-implicit).  Default-off (no active policy) is byte-equivalent
to the pre-T-038a behavior — :func:`active_precision_policy` returns
``None`` and dtype-aware blocks fall through their existing default branch.

Currently consulted by ``LookupTable1d`` only.  Coverage extends to other
dtype-aware blocks as ``T-038a-followup-other-blocks`` lands.
"""

from __future__ import annotations

import contextvars
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Iterator, Optional

import jax
import jax.numpy as jnp
import numpy as np


__all__ = [
    "PrecisionInfo",
    "precision_info",
    "assert_float64_active",
    "precision_policy",
    "active_precision_policy",
]


@dataclass(frozen=True)
class PrecisionInfo:
    """Resolved floating-point configuration at the time of the call."""

    x64_enabled: bool
    default_float_dtype: str  # "float32" or "float64"
    machine_eps: float  # machine epsilon of the default float type
    integer_time_dtype: str  # always "int64" in current jaxonomy

    def __str__(self) -> str:  # pragma: no cover - diagnostic
        return (
            f"PrecisionInfo(x64={self.x64_enabled}, "
            f"default_float={self.default_float_dtype}, "
            f"eps={self.machine_eps:.3e}, int_time={self.integer_time_dtype})"
        )


def precision_info() -> PrecisionInfo:
    """Return the current precision configuration.

    ``x64_enabled`` reflects ``jax.config.jax_enable_x64``.  The default
    float dtype is looked up via ``jnp.zeros(()).dtype`` so it matches
    whatever JAX is actually producing, rather than trusting the env var
    to be in sync.
    """
    x64 = bool(jax.config.read("jax_enable_x64"))
    sample = jnp.zeros(())
    dtype = jnp.result_type(sample)  # canonical float dtype
    eps = float(np.finfo(dtype).eps)
    return PrecisionInfo(
        x64_enabled=x64,
        default_float_dtype=str(dtype),
        machine_eps=eps,
        integer_time_dtype="int64",
    )


def assert_float64_active(*, msg: str | None = None) -> None:
    """Raise ``RuntimeError`` if the process is not running in float64.

    Use this at the start of scripts that rely on x64 precision.  It is
    cheap to call and catches silent downgrades early.
    """
    info = precision_info()
    if info.default_float_dtype != "float64":
        raise RuntimeError(
            msg
            or (
                "Jaxonomy is running in "
                f"{info.default_float_dtype!r}.  Set JAX_ENABLE_X64=true "
                "before importing jaxonomy to use double precision (the "
                "default).  See test/precision/POLICY.md for context."
            )
        )


# ── T-038a-followup-mixed-precision-cascade: precision_policy ─────────────
#
# A ContextVar holds the currently-active dtype policy.  ``None`` means
# no policy is active (the default-off path: dtype-aware blocks fall
# through to their existing default branch, byte-equivalent to pre-
# follow-up behavior).  Nested ``with`` blocks push and pop tokens,
# so inner contexts override outer ones; on exit the previous value is
# restored exactly.
#
# We deliberately use ``contextvars`` rather than a class-level stack on
# ``LeafSystem`` because (a) it's the standard library idiom for
# scoped configuration, (b) it composes with asyncio/threads correctly,
# and (c) it requires no refactor of how LeafSystem is built — the only
# touch point is consulting :func:`active_precision_policy` from
# block constructors that already track ``self._dtype``.

_active_precision_policy: contextvars.ContextVar[Optional[Any]] = (
    contextvars.ContextVar("jaxonomy_active_precision_policy", default=None)
)


def active_precision_policy() -> Optional[Any]:
    """Return the currently-active precision-policy dtype, or ``None``.

    Block ``__init__`` methods that already track an explicit per-block
    ``dtype=`` kwarg (e.g. ``LookupTable1d._dtype``) should call this
    when ``self._dtype is None`` to fall back to the context-manager
    default.  Explicit per-block dtype always wins (explicit-over-implicit).
    """
    return _active_precision_policy.get()


@contextmanager
def precision_policy(dtype: Any) -> Iterator[None]:
    """Context manager that sets a default dtype for dtype-aware blocks.

    Usage:

        with jaxonomy.precision_policy(jnp.float32):
            sys = MyDiagram()  # blocks default to f32 unless overridden
            result = jaxonomy.simulate(sys, ctx, ...)

    Semantics:
        * Blocks already supporting a per-block ``dtype=`` kwarg
          (T-038a) fall back to the context value when their own
          ``_dtype is None``.
        * Per-block ``dtype=`` explicitly passed to a constructor
          *overrides* the context (explicit-over-implicit).
        * Nested ``with precision_policy(...)`` blocks override outer
          contexts; the previous value is restored on exit.
        * Outside any active context, :func:`active_precision_policy`
          returns ``None`` and the default float dtype path holds —
          T-005 default-float64 policy preserved.

    Args:
        dtype: A JAX/NumPy dtype-like (e.g. ``jnp.float32``,
            ``jnp.float64``, ``jnp.float16``).  Stored verbatim;
            interpretation is up to the consuming block.

    Note:
        Only ``LookupTable1d`` consults this context as of T-038a-
        followup-mixed-precision-cascade.  Coverage extends to other
        dtype-aware blocks (``Gain``, ``Constant``, ``Adder``, ...) as
        ``T-038a-followup-other-blocks`` lands per-block ``_dtype``
        attributes for them.
    """
    token = _active_precision_policy.set(dtype)
    try:
        yield
    finally:
        _active_precision_policy.reset(token)
