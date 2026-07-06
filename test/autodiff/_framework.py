# SPDX-License-Identifier: MIT
"""
Gradient-correctness property-test harness (T-001).

``assert_grad_matches_fd(fwd, *inputs, solver, dtype, ...)`` is the primary
entry point. It runs reverse-mode autodiff on ``fwd``, runs a central-difference
FD approximation on each input, and asserts the two match within the tolerance
policy in ``tolerances.py``. Failures raise a ``GradientMismatch`` with enough
context to diagnose which block, solver, dtype, and input element misbehaved.

The harness is deliberately small and has no pytest imports so it can be used
from property-based tests (hypothesis) and conventional parametrized tests alike.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

import numpy as np
import jax
import jax.numpy as jnp

from jaxonomy.simulation import SimulatorOptions

from .tolerances import GradTolerance, get_tol, STATELESS_TOL


# ── exception ────────────────────────────────────────────────────────────────


class GradientMismatch(AssertionError):
    """Raised when AD and FD gradients disagree beyond the declared tolerance."""


@dataclass
class MismatchReport:
    block: str
    solver: str
    dtype: str
    arg_index: int
    element: tuple[int, ...] | None
    ad: float
    fd: float
    rtol: float
    atol: float
    extra: dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        elem = f"[{self.element}]" if self.element is not None else ""
        absd = abs(self.ad - self.fd)
        reld = absd / (abs(self.fd) + 1e-30)
        lines = [
            f"gradient mismatch: block={self.block}  solver={self.solver}  dtype={self.dtype}",
            f"  input[{self.arg_index}]{elem}: AD={self.ad:.8g}  FD={self.fd:.8g}",
            f"  abs_err={absd:.3g}  rel_err={reld:.3g}  allowed rtol={self.rtol:.1e} atol={self.atol:.1e}",
        ]
        if self.extra:
            lines.append(f"  extra: {self.extra}")
        return "\n".join(lines)


# ── FD helper ────────────────────────────────────────────────────────────────


def _as_np(x):
    return np.asarray(x)


def fd_central(fwd: Callable, inputs: Sequence, eps: float) -> list[np.ndarray]:
    """Central-difference gradient of a scalar-output function.

    Accepts arbitrary-shape numpy/jax array inputs.  Scalars are supported via
    ``np.asarray``.  Returns a list of numpy arrays matching each input's shape.
    """
    grads: list[np.ndarray] = []
    base_inputs = [_as_np(x).astype(np.float64) for x in inputs]
    for i, x0 in enumerate(base_inputs):
        g = np.zeros_like(x0)
        flat = x0.reshape(-1)
        for k in range(flat.size):
            plus = [np.array(a, copy=True) for a in base_inputs]
            minus = [np.array(a, copy=True) for a in base_inputs]
            plus[i].reshape(-1)[k] = flat[k] + eps
            minus[i].reshape(-1)[k] = flat[k] - eps
            fp = float(fwd(*[jnp.asarray(a) for a in plus]))
            fm = float(fwd(*[jnp.asarray(a) for a in minus]))
            g.reshape(-1)[k] = (fp - fm) / (2.0 * eps)
        grads.append(g)
    return grads


# ── core assertion ───────────────────────────────────────────────────────────


def assert_grad_matches_fd(
    fwd: Callable,
    *inputs,
    solver: str | None = None,
    dtype: str = "float64",
    block: str = "<unknown>",
    argnums: int | Sequence[int] | None = None,
    extra: dict[str, Any] | None = None,
    tol: GradTolerance | None = None,
) -> None:
    """Assert that ``jax.grad(fwd)`` matches central-difference FD.

    Args:
        fwd: Scalar-output function, ``fwd(*inputs) -> scalar``.
        inputs: Positional inputs (numpy/jax arrays or scalars).
        solver: Solver name (rk4/dopri5/bdf) if a simulation is involved.
            ``None`` uses the stateless tolerance.
        dtype: "float32" or "float64".
        block: Block or scenario name for error reporting.
        argnums: Which args to differentiate. Default: all.
        extra: Extra fields to surface in the failure message.
        tol: Override the tolerance policy (for debugging).
    """
    n = len(inputs)
    if argnums is None:
        argnums = tuple(range(n))
    elif isinstance(argnums, int):
        argnums = (argnums,)

    if tol is None:
        tol = get_tol(solver, dtype) if solver is not None else STATELESS_TOL[dtype]

    grad_fn = jax.grad(fwd, argnums=argnums)
    ad_grads = grad_fn(*inputs)
    if len(argnums) == 1:
        ad_grads = (ad_grads,)

    fd_grads_full = fd_central(fwd, inputs, eps=tol.fd_eps)
    fd_grads = [fd_grads_full[i] for i in argnums]

    extra = extra or {}
    for ai, (ad_g, fd_g) in enumerate(zip(ad_grads, fd_grads)):
        ad_arr = _as_np(ad_g).astype(np.float64)
        fd_arr = _as_np(fd_g)
        # Compare element-wise when total sizes match; JAX sometimes returns a
        # leading unit dim when inputs are stored inside a pytree parameter.
        if ad_arr.shape != fd_arr.shape:
            if ad_arr.size == fd_arr.size:
                ad_arr = ad_arr.reshape(fd_arr.shape)
            else:
                raise GradientMismatch(
                    f"AD/FD size mismatch on arg {argnums[ai]}: ad={ad_arr.shape} fd={fd_arr.shape}"
                )
        abs_err = np.abs(ad_arr - fd_arr)
        allowed = tol.atol + tol.rtol * np.abs(fd_arr)
        # NaN abs_err indicates the FD forward pass hit a numerically
        # degenerate input; surface it as a mismatch rather than silently
        # masking with >.
        bad = np.isnan(abs_err) | (abs_err > allowed)
        if np.any(bad):
            # Worst offender
            worst_flat = int(np.argmax(abs_err / (allowed + 1e-30)))
            shape = fd_arr.shape
            element = np.unravel_index(worst_flat, shape) if shape else None
            report = MismatchReport(
                block=block,
                solver=solver or "stateless",
                dtype=dtype,
                arg_index=int(argnums[ai]),
                element=element,
                ad=float(ad_arr.reshape(-1)[worst_flat]) if shape else float(ad_arr),
                fd=float(fd_arr.reshape(-1)[worst_flat]) if shape else float(fd_arr),
                rtol=tol.rtol,
                atol=tol.atol,
                extra=extra,
            )
            raise GradientMismatch(str(report))


# ── simulation helpers ───────────────────────────────────────────────────────

_SOLVER_METHOD = {
    "rk4": "rk4",
    "dopri5": "dopri5",
    "bdf": "bdf",
}

# RK4 is a fixed-step solver; adaptive rtol/atol have no effect. The simulator
# reads the step size from ``max_minor_step_size``. This constant is used by
# ``sim_options`` to ensure RK4 uses a sensibly-small step for the short
# simulation horizons used in the gradient tests.
_RK4_DEFAULT_STEP = 0.01


def sim_options(solver: str, dtype: str, **overrides) -> SimulatorOptions:
    """Build a SimulatorOptions with solver+dtype-consistent settings."""
    tol = get_tol(solver, dtype)
    kwargs: dict[str, Any] = dict(
        math_backend="jax",
        enable_autodiff=True,
        ode_solver_method=_SOLVER_METHOD[solver.lower()],
        rtol=tol.sim_rtol,
        atol=tol.sim_atol,
    )
    # RK4 is fixed-step: rtol/atol are ignored. Provide a small step so
    # accuracy matches the (solver, dtype) tolerance bucket over typical
    # sub-second test horizons.
    if solver.lower() == "rk4":
        kwargs["max_minor_step_size"] = _RK4_DEFAULT_STEP
    kwargs.update(overrides)
    return SimulatorOptions(**kwargs)
