# SPDX-License-Identifier: MIT

"""Implicit-function differentiation for iterative solvers (T-131).

``jax.grad`` cannot reverse-differentiate a ``lax.while_loop`` (no static
trip count to tape), so any dynamics callback that runs an iterative
solver — a Newton loop for an implicit actuator model, a projected
Gauss-Seidel contact/constraint solver, a fixed-point flash calculation —
forecloses reverse-mode autodiff through :func:`jaxonomy.simulate`.

:func:`implicit_solver` removes the loop from the tape using the implicit
function theorem. Given a black-box solver ``x* = solver(theta)`` and the
residual ``g(x*, theta) = 0`` its output satisfies, the wrapped solver:

* **forward**: runs ``solver`` exactly as-is (any control flow allowed —
  ``while_loop``, host iteration counts, line searches);
* **backward**: never differentiates the loop. By the IFT,
  ``dx*/dtheta = -(∂g/∂x)⁻¹ ∂g/∂theta``, so the VJP solves one linear
  system with the residual Jacobian transpose and routes the cotangent
  through ``∂g/∂theta`` — both obtained from the *residual*, which must
  therefore be JAX-differentiable (it almost always is: it is the
  equation, not the algorithm).

Accuracy note: the gradient is exact for the *equation* ``g = 0``. If the
solver returns an approximate root (loose tolerance), the gradient
corresponds to the exact root near it — tighten the solver tolerance if
gradients and primal must be consistent to machine precision.

Reverse-mode oriented: the wrapper uses ``jax.custom_vjp``, which
forecloses forward-mode (``jvp``) through it — call the raw solver
directly for forward-mode workflows (e.g. ``parameter_jacobian``-style
sweeps).
"""

from __future__ import annotations

from typing import Callable, Optional

import jax
import jax.numpy as jnp

__all__ = ["implicit_solver"]


def implicit_solver(
    solver: Callable,
    residual: Callable,
    linear_solve: Optional[Callable] = None,
):
    """Make an iterative solver reverse-mode differentiable via the IFT.

    Args:
        solver: ``solver(theta) -> x_star`` — any (jit-compatible)
            function returning a solution as a **flat array** (shape
            ``(n,)`` or scalar). Internal control flow is unrestricted;
            it is never differentiated. ``theta`` may be any pytree.
        residual: ``residual(x, theta) -> r`` with ``r`` the same shape
            as ``x`` and ``residual(solver(theta), theta) ≈ 0``. Must be
            JAX-differentiable — this is the equation the solution
            satisfies, used to construct both sides of the IFT.
        linear_solve: optional ``linear_solve(A, b) -> w`` used for the
            adjoint system ``(∂g/∂x)ᵀ w = b``. Defaults to the dense
            :func:`jnp.linalg.solve` — right for the small systems that
            appear inside dynamics callbacks (constraint dimensions of
            tens). Supply a matrix-free solver (e.g. CG) for large ``n``.

    Returns:
        A function ``wrapped(theta) -> x_star`` that is byte-equivalent
        to ``solver`` in the forward pass and reverse-differentiable.

    Example — an implicit velocity law solved by Newton iteration::

        def solve_v(theta):                       # while_loop inside
            def newton(v):
                g = v + jnp.tanh(theta * v) - 1.0
                dg = 1.0 + theta / jnp.cosh(theta * v) ** 2
                return v - g / dg
            def cond(carry):
                v, i = carry
                return (jnp.abs(v + jnp.tanh(theta * v) - 1.0) > 1e-12) & (i < 50)
            def body(carry):
                v, i = carry
                return newton(v), i + 1
            v, _ = jax.lax.while_loop(cond, body, (jnp.asarray(0.5), 0))
            return v

        def residual(v, theta):
            return v + jnp.tanh(theta * v) - 1.0

        solve_v_diff = implicit_solver(solve_v, residual)
        jax.grad(solve_v_diff)(0.3)               # works; matches FD
    """
    if linear_solve is None:
        def linear_solve(A, b):  # noqa: ANN001 - simple default
            return jnp.linalg.solve(A, b)

    @jax.custom_vjp
    def wrapped(theta):
        return solver(theta)

    def fwd(theta):
        x_star = solver(theta)
        return x_star, (x_star, theta)

    def bwd(saved, x_bar):
        x_star, theta = saved
        x_arr = jnp.asarray(x_star)
        scalar_x = x_arr.ndim == 0
        x_flat = jnp.atleast_1d(x_arr)
        xbar_flat = jnp.atleast_1d(jnp.asarray(x_bar)).astype(x_flat.dtype)

        def g_of_x(x):
            r = residual(x[0] if scalar_x else x, theta)
            return jnp.atleast_1d(jnp.asarray(r))

        # Adjoint linear system: (∂g/∂x)ᵀ w = -x̄. Dense Jacobian by
        # default — the systems this wraps are small (constraint dims of
        # tens); pass linear_solve= for matrix-free treatment.
        J_x = jax.jacobian(g_of_x)(x_flat)
        w = linear_solve(jnp.transpose(J_x), -xbar_flat)

        # θ̄ = (∂g/∂θ)ᵀ w via a VJP of the residual in its second slot.
        def g_of_theta(th):
            r = residual(x_star, th)
            return jnp.atleast_1d(jnp.asarray(r))

        _, vjp_theta = jax.vjp(g_of_theta, theta)
        (theta_bar,) = vjp_theta(w)
        return (theta_bar,)

    wrapped.defvjp(fwd, bwd)
    return wrapped
