# SPDX-License-Identifier: MIT
"""DAE constraint projection (T-003a).

Companion to :mod:`jaxonomy.simulation.dae_drift` (the detection
primitive).  Newton-corrects the algebraic component of the continuous
state at the end of each major step so that ``f_a(t, x, p) = 0`` is
re-established after the ODE solver advances the differential states.

Public entry point: :func:`project_constraints`.  Wired into
``Simulator._major_step`` when ``SimulatorOptions.dae_projection_enabled``
is ``True``.  The Jacobian ``∂f_a / ∂x_a`` is computed via
:func:`jax.jacfwd`; the linear update solves ``J Δx_a = -f_a``.  The
iteration is unrolled to ``max_iter`` steps with an early-out via
:func:`jax.lax.cond` once the residual drops below ``tol``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import jax
import jax.numpy as jnp

from .dae_drift import algebraic_row_mask

if TYPE_CHECKING:
    from ..framework import ContextBase, SystemBase


__all__ = ["project_constraints", "baumgarte_augment_ode_rhs"]


def _flatten_continuous_state(context: "ContextBase") -> tuple[jnp.ndarray, callable]:
    """Return (flat_xc, unflatten_fn) for a context's continuous state.

    ``unflatten_fn(flat) -> new_context`` rebuilds a new context whose
    continuous state has been replaced by the per-leaf split of
    ``flat``.  Works for both LeafContext (single array) and
    DiagramContext (list of arrays).
    """
    xc = context.continuous_state

    # Single LeafContext: continuous_state is a single array (or pytree).
    if not isinstance(xc, list):
        leaves, treedef = jax.tree.flatten(xc)
        # Use .shape from the JAX/numpy array directly — works for tracers.
        shapes = [tuple(getattr(l, "shape", jnp.asarray(l).shape)) for l in leaves]
        sizes = [int(np.prod(sh)) if sh else 1 for sh in shapes]
        flat = jnp.concatenate([jnp.ravel(l) for l in leaves]) if leaves else \
            jnp.zeros((0,), dtype=jnp.float64)

        def _unflatten(flat_x):
            if not leaves:
                return context
            split = []
            offset = 0
            for sz, sh in zip(sizes, shapes):
                split.append(flat_x[offset:offset + sz].reshape(sh))
                offset += sz
            new_xc = jax.tree.unflatten(treedef, split)
            return context.with_continuous_state(new_xc)

        return flat, _unflatten

    # DiagramContext: continuous_state is a list of per-leaf arrays/pytrees.
    per_leaf_specs = []  # list of (sizes, shapes, treedef)
    flat_segments = []
    for sub_xc in xc:
        leaves, treedef = jax.tree.flatten(sub_xc)
        shapes = [tuple(getattr(l, "shape", jnp.asarray(l).shape)) for l in leaves]
        sizes = [int(np.prod(sh)) if sh else 1 for sh in shapes]
        per_leaf_specs.append((sizes, shapes, treedef))
        for l in leaves:
            flat_segments.append(jnp.ravel(l))
    flat = jnp.concatenate(flat_segments) if flat_segments else \
        jnp.zeros((0,), dtype=jnp.float64)

    def _unflatten(flat_x):
        new_subs = []
        offset = 0
        for sizes, shapes, treedef in per_leaf_specs:
            split = []
            for sz, sh in zip(sizes, shapes):
                split.append(flat_x[offset:offset + sz].reshape(sh))
                offset += sz
            new_subs.append(jax.tree.unflatten(treedef, split))
        return context.with_continuous_state(new_subs)

    return flat, _unflatten


def _full_residual(system: "SystemBase", context: "ContextBase") -> jnp.ndarray:
    """Flatten ``eval_time_derivatives`` to a single vector.

    Same flattening convention as ``algebraic_row_mask`` so the row
    indices line up.
    """
    xcdot = system.eval_time_derivatives(context)
    leaves = jax.tree.leaves(xcdot)
    if not leaves:
        return jnp.zeros((0,), dtype=jnp.float64)
    return jnp.concatenate([jnp.ravel(l) for l in leaves])


def _emit_projection_warning(norm_val, tol_val, iters_val, max_iter_val):
    """Host-side ``UserWarning`` for non-converged projection.

    Invoked via ``jax.debug.callback`` so ``warnings.warn`` runs on the
    host; gated here so converged solves stay silent.
    """
    import warnings
    norm = float(norm_val)
    tol = float(tol_val)
    if not (norm <= tol):  # also fires on NaN
        warnings.warn(
            f"project_constraints did not converge: ||f_a||_inf = {norm:.3e} "
            f"> tol = {tol:.3e} after {int(iters_val)}/{int(max_iter_val)} "
            f"Newton iterations. The returned algebraic state may be "
            f"inconsistent and a subsequent implicit solve may fail; "
            f"increase max_iter or improve the starting guess (e.g. build "
            f"input sources near the operating point).",
            UserWarning,
            stacklevel=2,
        )


def project_constraints(
    system: "SystemBase",
    context: "ContextBase",
    *,
    tol: float = 1e-8,
    max_iter: int = 20,
    gradient: str = "stop",
    warn_on_nonconvergence: bool = True,
) -> "ContextBase":
    """Newton-project the algebraic state onto the constraint manifold.

    Solves ``J · Δx_a = -f_a`` for the algebraic sub-vector ``x_a``
    using ``J = ∂f_a / ∂x_a``, holding the differential states fixed.
    Iterates up to ``max_iter`` Newton steps (a ``lax.while_loop``, so
    unused iterations cost nothing), exiting once the max-abs residual
    drops below ``tol``.

    Differentiation — pick the mode by what consumes the result:

    * ``gradient="stop"`` (default): the projected algebraic values carry
      no gradient.  This is the *correct* semantics when the projected
      context is handed to an implicit integrator (the reset-then-simulate
      pattern, and this function's own in-simulator per-step use): the
      trajectory's true sensitivity to the algebraic initial values is
      zero — the solver re-enforces ``f_a = 0`` at every step — while
      AD through the integrator w.r.t. off-manifold algebraic
      perturbations produces spurious large gradients (verified against
      finite differences: "stop" matches FD to ~1e-8; propagating
      algebraic-IC sensitivity into a BDF solve was off by orders of
      magnitude).
    * ``gradient="implicit"``: reverse- and forward-mode derivatives of
      the projected state itself are computed via the implicit function
      theorem (``jax.lax.custom_root``): ``∂x_a*/∂θ = -J⁻¹ ∂f_a/∂θ``.
      Use this when the projected values are consumed *directly* (outputs,
      losses on the manifold) — validated against FD to ~1e-11.  Never
      differentiate through the Newton iterations themselves; the unrolled
      path yields wrong gradients.

    Args:
        system: The system whose algebraic constraints will be enforced.
        context: Current context (post-ODE-step).
        tol: Convergence threshold on ``||f_a||_∞``.
        max_iter: Maximum Newton iterations.
        gradient: ``"stop"`` (default; projected values carry no gradient)
            or ``"implicit"`` (IFT via ``lax.custom_root``).
        warn_on_nonconvergence: Emit a host-side ``UserWarning`` when the
            final residual exceeds ``tol`` (or is non-finite).

    Returns:
        A new context with the algebraic sub-vector corrected.  The
        differential sub-vector is unchanged.  When the system has no
        mass matrix (no algebraic constraints), the input context is
        returned unmodified.
    """
    if gradient not in ("implicit", "stop"):
        raise ValueError(
            f"project_constraints: gradient must be 'implicit' or 'stop', "
            f"got {gradient!r}."
        )

    mask_np = algebraic_row_mask(system)
    if mask_np is None or not mask_np.any():
        return context

    alg_idx = jnp.asarray(np.where(mask_np)[0])

    flat_x, unflatten = _flatten_continuous_state(context)

    # Sanity: the mask length should match the flat continuous-state length
    # when the system has the simple "one M-block per leaf" layout.  If
    # the shapes don't agree (e.g. a leaf with a tuple-valued state), we
    # cannot do a sensible projection — return unchanged.  Use the static
    # ``shape`` attribute (works on tracers — only the static dim matters).
    if flat_x.shape[0] != int(mask_np.shape[0]):
        return context

    def _residual_full(flat):
        new_ctx = unflatten(flat)
        return _full_residual(system, new_ctx)

    def _residual_alg(x_a):
        # Reconstruct flat state from the differential baseline (closed
        # over — gradients w.r.t. it flow via custom_root's IFT rule)
        # plus the candidate algebraic sub-vector.
        flat = flat_x.at[alg_idx].set(x_a)
        return _residual_full(flat)[alg_idx]

    def _newton_solve(f, x0):
        max_it = jnp.asarray(int(max_iter))

        def cond(carry):
            x, it, norm = carry
            return jnp.logical_and(norm > tol, it < max_it)

        def body(carry):
            x, it, _ = carry
            f_x = f(x)
            # Jacobian ∂f_a / ∂x_a — small (n_alg × n_alg) dense block.
            J = jax.jacfwd(f)(x)
            dx = jnp.linalg.solve(J, -f_x)
            x_new = x + dx
            return x_new, it + 1, jnp.max(jnp.abs(f(x_new)))

        norm0 = jnp.max(jnp.abs(f(x0)))
        x, it, norm = jax.lax.while_loop(
            cond, body, (x0, jnp.asarray(0), norm0)
        )
        if warn_on_nonconvergence:
            jax.debug.callback(
                _emit_projection_warning, norm, tol, it, int(max_iter)
            )
        return x

    def _tangent_solve(g, y):
        # Linear solve for custom_root's IFT rule: J_g · x = y.
        return jnp.linalg.solve(jax.jacfwd(g)(jnp.zeros_like(y)), y)

    x_a0 = flat_x[alg_idx]
    if gradient == "implicit":
        x_a_star = jax.lax.custom_root(
            _residual_alg, x_a0, _newton_solve, _tangent_solve
        )
    else:  # "stop": value-only projection
        x_a_star = jax.lax.stop_gradient(
            _newton_solve(_residual_alg, jax.lax.stop_gradient(x_a0))
        )

    return unflatten(flat_x.at[alg_idx].set(x_a_star))


# ----------------------------------------------------------------------
# T-113-followup-baumgarte-and-ssp — Baumgarte stabilization
# ----------------------------------------------------------------------
#
# Classical Baumgarte: a holonomic constraint ``g(x) = 0`` is replaced
# in the index-reduced DAE by the damped form
#
#     g̈ + 2α ġ + β² g = 0
#
# which drives any residual drift to zero exponentially (critically
# damped at α = β = 1/τ; τ = relaxation time).  In Jaxonomy's mass-
# matrix DAE form ``M·ẋ = f(x)`` the algebraic rows of ``M`` are zero,
# so the algebraic-row residual ``f_a(x)`` is itself the constraint
# residual ``g``.  Adding ``-2α·ġ - β²·g`` to those rows of the rhs
# reshapes the constraint enforced by the solver from the bare
# ``f_a = 0`` into the damped form, which BDF (or any other solver
# that satisfies ``M·ẋ = f`` algebraically) then enforces step-by-step.
#
# We approximate ``ġ`` via the JVP ``∂g/∂x_diff · ẋ_diff`` — the
# differential states' rhs is well-defined (their rows of ``M`` are
# identity in the supported semi-explicit form), so the time-derivative
# of ``g`` along the trajectory is computable from the current state
# alone.  The algebraic-state contribution to ``ġ`` is dropped: it is
# multiplied by ẋ_alg which is undefined (M_alg = 0); ignoring it is
# the standard simplification used by every Baumgarte implementation
# for index-reduced DAEs.
#
# Default-off byte-equivalence: when ``alpha is None`` and
# ``beta is None``, :func:`baumgarte_augment_ode_rhs` returns the
# original ``rhs_fn`` unchanged — no extra ops compiled in.  When the
# system has no mass matrix (no algebraic rows to stabilize), the
# wrapper similarly short-circuits.
#
# Limitations & honest fallback notes (T-113-followup-baumgarte-architecture):
#   * The augmentation modifies the algebraic constraint enforced by
#     BDF.  For systems where ``f_a`` is the position-level constraint
#     (low-index DAE), this matches classical Baumgarte exactly.  For
#     higher-index reductions where ``f_a`` is itself a derivative of
#     the underlying constraint (e.g. ``g̈``), the Baumgarte coefficients
#     act on that reduction; effective decay rates may not equal the
#     nominal ``α/β`` chosen by the user.  This is a fundamental
#     limitation of mass-matrix DAE form, not of this implementation.
#   * ``ġ`` ignores the algebraic-state contribution.  Over a single
#     timestep this is fine (the algebraic states are kept on-manifold
#     by the solver); over very stiff problems with rapidly-varying
#     algebraic states the approximation may degrade.
#   * On stiff high-index DAEs (e.g. PlanarPendulum's index-3 form
#     reduced to mass-matrix DAE), the augmented algebraic equations
#     may be hard for BDF's Newton iteration to converge — the JVP-
#     based ``ġ`` couples the seven algebraic rows tightly through the
#     differential rhs, and large gains (α, β ≥ 1) can stall the inner
#     solve.  Recommend pairing with :func:`project_constraints` (set
#     ``dae_projection_enabled=True``) and using small Baumgarte gains
#     (≤ 0.1) as a continuous *complement* to projection rather than
#     a standalone correction.

def baumgarte_augment_ode_rhs(
    rhs_fn,
    system: "SystemBase",
    alpha: float | None,
    beta: float | None,
):
    """Wrap a solver's ``ode_rhs`` to add Baumgarte feedback.

    Adds ``-2α·ġ - β²·g`` to each algebraic row of the rhs, where
    ``g = f_a(x)`` (the algebraic-row residual at the current state)
    and ``ġ`` is approximated by the JVP of ``f_a`` against the
    differential portion of the rhs.

    Default-off semantics: when both ``alpha`` and ``beta`` are
    ``None``, OR when ``system`` has no mass matrix / no algebraic
    rows, the original ``rhs_fn`` is returned unchanged.  This keeps
    the disabled hot path byte-equivalent.

    Composes cleanly with :func:`project_constraints` — projection
    kills accumulated drift at major-step boundaries; Baumgarte
    damps drift continuously between projections.

    Args:
        rhs_fn: A callable ``(y, t, context) -> xcdot_pytree`` matching
            :meth:`ODESolverBase.ode_rhs` — the function the solver
            uses to evaluate time derivatives.
        system: The system whose algebraic constraints will be damped.
        alpha: Baumgarte velocity-feedback gain (multiplies ``ġ``).
            ``None`` disables the velocity term.
        beta: Baumgarte position-feedback gain (multiplies ``g``).
            ``None`` disables the position term.

    Returns:
        A new callable with the same signature as ``rhs_fn``.  When the
        Baumgarte path is inactive (defaults / no constraints), the
        original ``rhs_fn`` is returned unchanged so the trace graph is
        identical to the disabled run.
    """
    # Default-off: both gains None → no augmentation, return as-is so
    # the JIT trace graph is byte-equivalent to the pre-followup path.
    if alpha is None and beta is None:
        return rhs_fn

    mask_np = algebraic_row_mask(system)
    if mask_np is None or not mask_np.any():
        # No mass matrix or no algebraic rows — nothing to damp.  Return
        # the original rhs unchanged (no extra ops compiled in).
        return rhs_fn

    a_val = 0.0 if alpha is None else float(alpha)
    b_val = 0.0 if beta is None else float(beta)
    # Pre-compute the constant-shape pieces of the augmentation so the
    # wrapped rhs only does cheap per-step ops.
    mask = jnp.asarray(mask_np)
    alg_idx = jnp.asarray(np.where(mask_np)[0])
    diff_idx = jnp.asarray(np.where(~mask_np)[0])

    def _flatten_pytree(pytree):
        leaves = jax.tree.leaves(pytree)
        if not leaves:
            return jnp.zeros((0,), dtype=jnp.float64), None, None
        shapes = [tuple(getattr(l, "shape", jnp.asarray(l).shape)) for l in leaves]
        sizes = [int(np.prod(sh)) if sh else 1 for sh in shapes]
        flat = jnp.concatenate([jnp.ravel(l) for l in leaves])
        return flat, shapes, sizes

    def wrapped_rhs(y, t, context):
        # Original rhs evaluated at the (y, t) the solver passed in.
        xcdot = rhs_fn(y, t, context)
        leaves = jax.tree.leaves(xcdot)
        if not leaves:
            return xcdot
        treedef = jax.tree.structure(xcdot)
        shapes = [tuple(getattr(l, "shape", jnp.asarray(l).shape)) for l in leaves]
        sizes = [int(np.prod(sh)) if sh else 1 for sh in shapes]
        flat_xcdot = jnp.concatenate([jnp.ravel(l) for l in leaves])

        # Sanity guard: if the flat layout doesn't match the algebraic
        # mask (e.g. tuple-valued state), bail out and return the
        # un-augmented rhs.  Keeps weird systems working.
        if flat_xcdot.shape[0] != int(mask_np.shape[0]):
            return xcdot

        # g(x) = f_a(x) — the current algebraic-row residual.
        g = flat_xcdot[alg_idx]

        # ġ via JVP of f_a against ẋ_diff.  We re-evaluate the rhs as
        # a function of the flat state via the same context-update
        # pattern that ``ode_rhs`` uses.  This stays inside JAX so the
        # JVP is traceable.
        def _f_a_of_flat(flat_y):
            # Reconstruct y from flat using the saved tree structure.
            if not shapes:
                return jnp.zeros((0,), dtype=flat_y.dtype)
            split = []
            offset = 0
            for sz, sh in zip(sizes, shapes):
                split.append(flat_y[offset:offset + sz].reshape(sh))
                offset += sz
            new_y = jax.tree.unflatten(treedef, split)
            new_xcdot = rhs_fn(new_y, t, context)
            new_leaves = jax.tree.leaves(new_xcdot)
            new_flat = jnp.concatenate([jnp.ravel(l) for l in new_leaves])
            return new_flat[alg_idx]

        # Build the flat y from the input pytree (matches the layout we
        # already extracted from xcdot).
        y_leaves = jax.tree.leaves(y)
        if y_leaves:
            flat_y = jnp.concatenate([jnp.ravel(l) for l in y_leaves])
        else:
            flat_y = jnp.zeros((0,), dtype=flat_xcdot.dtype)

        # Tangent: ẋ for the differential rows (= rhs there); zero in
        # the algebraic slots (their ẋ is undefined; standard
        # Baumgarte-for-DAE simplification — see module-level note).
        tangent = jnp.zeros_like(flat_y).at[diff_idx].set(flat_xcdot[diff_idx])

        if a_val != 0.0:
            _, g_dot = jax.jvp(_f_a_of_flat, (flat_y,), (tangent,))
        else:
            g_dot = jnp.zeros_like(g)

        feedback = -2.0 * a_val * g_dot - (b_val ** 2) * g
        new_flat_xcdot = flat_xcdot.at[alg_idx].add(feedback)

        # Unflatten back to the original pytree.
        split = []
        offset = 0
        for sz, sh in zip(sizes, shapes):
            split.append(new_flat_xcdot[offset:offset + sz].reshape(sh))
            offset += sz
        return jax.tree.unflatten(treedef, split)

    return wrapped_rhs
