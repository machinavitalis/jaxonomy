# SPDX-License-Identifier: MIT

"""Custom JAX autodiff (VJP) rules for the Simulator.

Extracts the VJP factory functions that were previously methods on the Simulator
class. These define custom reverse-mode differentiation rules for:

  - ``advance_to``: captures variation with respect to the simulation end time
  - ``guarded_integrate``: captures variation with respect to zero-crossing times

Both functions return wrapped callables with ``jax.custom_vjp`` rules defined.
"""

from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING

import numpy as np
import jax
import jax.numpy as jnp

from .types import StepEndReason

if TYPE_CHECKING:
    from typing import Callable
    from ..framework import ContextBase
    from .types import SimulatorState


def make_advance_to_vjp(simulator) -> "Callable":
    """Build ``advance_to`` with a custom VJP for end-time differentiation.

    If autodiff is not enabled, returns the unwrapped ``_advance_to`` method.

    The custom VJP correctly accounts for variation with respect to the simulation
    end time, which the standard JAX VJP would not capture (since ``boundary_time``
    is a dynamic value that affects the control flow termination condition).

    Args:
        simulator: The Simulator instance.

    Returns:
        A callable with the same signature as ``Simulator._advance_to``.
    """
    if not simulator.enable_autodiff:
        return simulator._advance_to

    def _wrapped_advance_to(sim, boundary_time: float, context: "ContextBase"):
        return sim._advance_to(boundary_time, context)

    def _wrapped_advance_to_fwd(sim, boundary_time: float, context: "ContextBase"):
        primals, vjp_fun = jax.vjp(sim._advance_to, boundary_time, context)

        # Keep the final continuous time derivative for the adjoint
        xdot = sim.system.eval_time_derivatives(primals.context)

        # Store final solver state (yf = flat CT state at T) and context for the
        # mass-matrix adjoint IC correction applied once in the backward rule.
        final_solver_state = primals.ode_solver_state
        final_context = primals.context

        res = (vjp_fun, xdot, primals.step_end_reason, final_solver_state, final_context)
        return primals, res

    def _wrapped_advance_to_adj(_sim, res: tuple, adjoints: "SimulatorState"):
        vjp_fun, xdot, reason, final_solver_state, final_context = res

        # T-113-followup-dae-adjoint-sign-bug: correct semi-explicit-DAE
        # adjoint terminal handling. The objective J = g(x(T)) generally
        # depends on the ALGEBRAIC terminal states x_a(T), but x_a(T) is not a
        # free state — it is pinned by the constraint 0 = f_a(x_d, x_a, p).
        # Eliminating x_a(T) = h(x_d(T), p) via the implicit function theorem
        # (faa = ∂f_a/∂x_a):
        #     ∂x_a/∂x_d = -faa^{-1} f_{a,x_d},   ∂x_a/∂p = -faa^{-1} f_{a,p}
        # splits the algebraic seed g_{x_a} into two contributions the naive
        # block_diag(M, M.T, I) adjoint dropped:
        #   (1) a correction to the differential terminal costate (Cao et al.
        #       2003 consistent-IC), and
        #   (2) a DIRECT terminal boundary term on dJ/dp.
        # The algebraic seed is then ZEROED — its whole effect is carried by
        # (1) and (2).
        #
        # CRITICAL: the terminal seed g_x = ∂J/∂x(T) flows through
        # ``adjoints.context.continuous_state`` (the differentiated objective
        # reads ``results.context[...].continuous_state``), NOT through
        # ``adjoints.ode_solver_state.y`` (which is zero on this path). The
        # earlier implementation patched the latter and so was a no-op.
        solver = _sim.ode_solver
        ctx_boundary = None
        # Preserve the ORIGINAL objective seed for the tf_adj computation below;
        # the DAE correction patches adjoints.context.continuous_state in place.
        vc = adjoints.context.continuous_state
        if (
            solver.supports_mass_matrix
            and solver.mass is not None
            and final_solver_state is not None
        ):
            from jax.flatten_util import ravel_pytree

            M = solver.mass
            n_total = M.shape[0]
            alg_mask = np.all(M == 0, axis=1)
            diff_indices = np.where(~alg_mask)[0]
            alg_indices = np.where(alg_mask)[0]
            n_ode = len(diff_indices)
            perm = np.concatenate([diff_indices, alg_indices])
            inv_perm = np.argsort(perm)
            needs_perm = not np.array_equal(perm, np.arange(n_total))

            if n_ode < n_total:  # at least one algebraic state -> DAE
                # Flatten the objective seed (per-subsystem continuous_state
                # cotangent) into the global flat order matching solver.mass.
                g_x_flat, cs_unravel = ravel_pytree(
                    adjoints.context.continuous_state
                )
                yf = final_solver_state.y
                tf = final_solver_state.t

                if needs_perm:
                    _orig_rhs = solver.flat_ode_rhs

                    def _perm_rhs(y_p, t, ctx):
                        return _orig_rhs(y_p[inv_perm], t, ctx)[perm]

                    _rhs_for_jac = _perm_rhs
                    yf_p = yf[perm]
                    g_x_p = g_x_flat[perm]
                else:
                    _rhs_for_jac = solver.flat_ode_rhs
                    yf_p = yf
                    g_x_p = g_x_flat

                J = jax.jacfwd(_rhs_for_jac)(yf_p, tf, final_context)
                dg_ode = J[n_ode:, :n_ode]   # ∂f_a/∂x_d
                dg_alg = J[n_ode:, n_ode:]    # ∂f_a/∂x_a  (= faa)

                g_ode = g_x_p[:n_ode]
                g_alg = g_x_p[n_ode:]         # g_{x_a}
                faaT_g_alg = jnp.linalg.solve(dg_alg.T, g_alg)  # faa^{-T} g_{x_a}

                # (1) Cao consistent-IC correction of the differential seed.
                g_ode_corr = g_ode - dg_ode.T @ faaT_g_alg
                # Zero the algebraic seed (its effect is in (1) and (2)).
                g_x_p_new = jnp.concatenate(
                    [g_ode_corr, jnp.zeros_like(g_alg)]
                )
                g_x_new = g_x_p_new[inv_perm] if needs_perm else g_x_p_new
                adjoints = adjoints._replace(
                    context=adjoints.context.with_continuous_state(
                        cs_unravel(g_x_new)
                    )
                )

                # (2) Direct terminal boundary term dJ/dp += -g_{x_a} faa^{-1}
                # f_{a,p} = w^T f_p, with w zero on differential rows and
                # w_a = -faa^{-T} g_{x_a} on the algebraic rows. Realise it as a
                # VJP of the terminal RHS w.r.t. the context (parameters live in
                # the context).
                w_alg = -faaT_g_alg
                w_p = jnp.concatenate(
                    [jnp.zeros(n_ode, dtype=w_alg.dtype), w_alg]
                )
                w = w_p[inv_perm] if needs_perm else w_p
                _primals_b, _vjp_b = jax.vjp(
                    lambda c: solver.flat_ode_rhs(yf, tf, c), final_context
                )
                (ctx_boundary,) = _vjp_b(w)

        # Standard adjoint variables (ignoring the auto-derived tf_adj)
        _, context_adj = vjp_fun(adjoints)

        if ctx_boundary is not None:
            context_adj = jax.tree_util.tree_map(
                lambda a, b: a + b, context_adj, ctx_boundary
            )

        # Manual tf_adj: dot product of the ORIGINAL objective seed with final
        # time derivatives (vc captured before the DAE correction patched it).
        vT_xdot = jax.tree_util.tree_map(
            lambda xdot_leaf, vc_leaf: jnp.dot(xdot_leaf, vc_leaf), xdot, vc
        )

        # Zero out tf_adj if simulation ended due to terminal event
        tf_adj = jnp.where(
            reason == StepEndReason.TerminalEventTriggered,
            0.0,
            sum(jax.tree_util.tree_leaves(vT_xdot)),
        )

        return (tf_adj, context_adj)

    advance_to = jax.custom_vjp(_wrapped_advance_to, nondiff_argnums=(0,))
    advance_to.defvjp(_wrapped_advance_to_fwd, _wrapped_advance_to_adj)

    # Preserve docstring and annotations
    advance_to.__doc__ = simulator._advance_to.__doc__
    advance_to.__annotations__ = simulator._advance_to.__annotations__

    return partial(advance_to, simulator)


def _cross_block_saltation_correction(
    sim, triggered, ctx_pre, f_minus, f_plus, lam_plus_flat, zc_events_out, context_adj
):
    """Global event-time (saltation) correction for events in *stateless* blocks.

    The per-block saltation adjoint (``leaf_system._wrap_reset_map``) corrects
    the event-time gradient when the event's owning block has continuous state
    — it localizes the crossing against that block's own dynamics jump.  When
    the owning block has *no* continuous state (e.g. a ``StateMachine`` whose
    transition merely re-points an output that feeds a downstream integrator),
    that per-block rule short-circuits and the event-time sensitivity of the
    *downstream* block's dynamics is lost.

    This helper supplies exactly that missing term at the simulator level, where
    the full continuous costate ``λ⁺`` is available.  For each triggered event
    in a stateless block it forms the global event-time cotangent

        c = (f⁻ − f⁺) · λ⁺

    (the localized crossing time's cotangent contributed by every continuous
    state whose RHS jumps across the event) and redistributes it to parameters
    via the implicit-function theorem ``dt_e/dp = −∇g/D``, where ``∇g`` is the
    guard gradient taken against a *refreshed* port cache (so the guard's
    dependence on upstream parameters / signals is live) and ``D`` is the guard's
    total time-derivative along the trajectory.  Only parameter leaves are
    corrected: parameter cotangents are final (parameters do not evolve),
    whereas a continuous-state correction here would be at the wrong time point
    and the time leaf belongs to the end-time gradient.

    Events whose block *has* continuous state are skipped (already handled
    per-block), so this is purely additive and never double-counts.
    """
    from jax.flatten_util import ravel_pytree

    # Event-time cotangent magnitude from the GLOBAL dynamics jump.  Zero when
    # nothing fired (then f⁻ == f⁺), so the whole correction self-disables.
    jump_dot = jnp.dot(f_minus - f_plus, lam_plus_flat)
    trig_f = jnp.asarray(triggered, dtype=lam_plus_flat.dtype)

    for event in zc_events_out:
        # Prefer the smooth ``grad_guard`` residual when present (the trigger
        # ``guard`` may be non-smooth, e.g. a boolean StateMachine predicate
        # whose gradient is identically zero); fall back to ``guard``.
        guard = getattr(event, "grad_guard", None) or getattr(event, "guard", None)
        if guard is None:
            continue
        sid = event.system_id
        # Only blocks WITHOUT continuous state are unhandled per-block.
        if ctx_pre[sid].has_continuous_state:
            continue

        # Live guard gradient: ∇g with the port cache refreshed so the guard's
        # dependence on upstream parameters / signals (and its total time
        # derivative) is captured rather than frozen.
        def _live_guard(c, _g=guard):
            return jnp.asarray(_g(c.refresh_port_cache())).reshape(())

        grad_g = jax.grad(_live_guard, allow_int=True)(ctx_pre)

        # Denominator D = total time-derivative of the guard along the
        # trajectory = ∂g/∂t (incl. source du/dt, from the refresh) plus the
        # state-driven part Σ ∂g/∂x · ẋ.  ``ravel_pytree`` of the guard's
        # continuous-state cotangent matches the solver's flat RHS ordering.
        # NOTE: a non-smooth guard (e.g. a boolean ``where(x>c, 1, -1)`` such
        # as the one a ``StateMachine`` emits) has ∇g ≡ 0, hence D ≡ 0 and the
        # correction is identically skipped — the event-time gradient is simply
        # not recoverable from a guard with no gradient.
        g_x_flat, _ = ravel_pytree(grad_g.continuous_state)
        D = grad_g.time + jnp.dot(g_x_flat, f_minus)
        safe_D = jnp.where(D != 0, D, 1.0)

        active = event.event_data.active & event.event_data.triggered
        active_f = jnp.asarray(active, dtype=lam_plus_flat.dtype)
        # dt_e/dp = −∇g/D ; the redistributed event-time cotangent is c·dt_e/dp,
        # i.e. scale·∇g with scale = −c/D, gated to the firing event.
        scale = jnp.where(D != 0, -jump_dot / safe_D, 0.0) * active_f * trig_f

        # Redistribute scale·∇g to the parameter leaves of every subcontext
        # (the upstream source's parameter — e.g. a Sine amplitude — lives in
        # its own subcontext's parameters).
        float0 = jax.dtypes.float0

        def _add(a, b, _s=scale):
            bd = getattr(b, "dtype", None)
            if bd is not None and bd != float0 and jnp.issubdtype(bd, jnp.inexact):
                return a + _s * b
            return a

        for s_id in list(context_adj.subcontexts.keys()):
            dgp = grad_g[s_id].parameters
            cur = context_adj[s_id].parameters
            new_p = jax.tree_util.tree_map(_add, cur, dgp)
            context_adj = context_adj.with_subcontext(
                s_id, context_adj[s_id].with_parameters(new_p)
            )

    return context_adj


def make_guarded_integrate_vjp(simulator) -> "Callable":
    """Build ``guarded_integrate`` with a custom VJP for zero-crossing time.

    If autodiff is not enabled, returns the unwrapped ``_guarded_integrate`` method.

    The custom VJP defines a differentiable forward pass that integrates to the
    actual zero-crossing time (determined in the primal pass), then applies
    the reset maps. The adjoint pass uses the chain rule to recover gradients
    through the event handling.

    Args:
        simulator: The Simulator instance.

    Returns:
        A callable with the same signature as ``Simulator._guarded_integrate``.
    """
    if not simulator.enable_autodiff:
        return simulator._guarded_integrate

    # Import here to avoid circular imports — _odeint is a module-level function
    # in simulator.py with its own custom VJP.
    from .simulator import _odeint

    def _wrapped_solve(sim, solver_state, results_data, tf, context, zc_events):
        return sim._guarded_integrate(
            solver_state, results_data, tf, context, zc_events
        )

    def _wrapped_solve_fwd(
        sim, solver_state, _results_data, tf, context, zc_events
    ):
        # Primal calculation (no results recording)
        t0 = solver_state.t

        (
            triggered,
            solver_state_out,
            context_out,
            _,
            zc_events_out,
        ) = sim._guarded_integrate(solver_state, None, tf, context, zc_events)
        tf = solver_state_out.t

        # Differentiable forward pass with known zero-crossing location
        solver = sim.ode_solver
        func = solver.flat_ode_rhs

        def _forward(solver_state, tf, context, zc_events_out):
            solver_state_out = _odeint(solver, func, solver_state, tf, context)
            context = context.with_time(solver_state_out.t)
            context = context.with_continuous_state(
                solver_state_out.unraveled_state
            )
            context = sim.system.handle_zero_crossings(zc_events_out, context)
            return context

        _primals, vjp_fun = jax.vjp(
            _forward, solver_state, tf, context, zc_events_out
        )

        # T-001c-followup #1d (cross-block saltation): residuals for the global
        # event-time correction applied to events whose owning block has *no*
        # continuous state (so the per-block saltation adjoint in
        # ``leaf_system._wrap_reset_map`` short-circuits and the event-time
        # sensitivity of a *downstream* block's dynamics is otherwise lost).
        # ``ctx_pre``/``ctx_post`` are the system context immediately before /
        # after the reset is applied at the localized crossing time; ``f_minus``
        # / ``f_plus`` are the flattened global RHS evaluated there.  These are
        # cheap primal evaluations (no recording / no autodiff) and are unused
        # unless an event in a stateless block actually fired.
        yf = solver_state_out.y
        te = solver_state_out.t
        ctx_pre = context.with_time(te).with_continuous_state(
            solver_state_out.unraveled_state
        )
        ctx_pre = ctx_pre.refresh_port_cache()
        ctx_post = context_out.refresh_port_cache()
        f_minus = func(yf, te, ctx_pre)
        f_plus = func(yf, te, ctx_post)

        primals = (triggered, solver_state_out, context_out, None, zc_events_out)
        residuals = (
            triggered, solver_state_out, t0, tf, context, vjp_fun,
            zc_events_out, ctx_pre, f_minus, f_plus,
        )
        return primals, residuals

    def _wrapped_solve_adj(sim, residuals, adjoints):
        (
            triggered, primal_solver_state, t0, tf, context, vjp_fun,
            zc_events_out, ctx_pre, f_minus, f_plus,
        ) = residuals
        (
            _triggered_adj,
            solver_state_adj,
            context_adj,
            _results_data_adj,
            _zc_events_adj,
        ) = adjoints

        # Capture the incoming costate on the post-event continuous state (λ⁺)
        # before ``vjp_fun`` consumes it — needed by the cross-block saltation
        # correction below.
        lam_plus_flat = solver_state_adj.y

        context_adj = context_adj.with_time(solver_state_adj.t)
        context_adj = context_adj.with_continuous_state(
            solver_state_adj.unraveled_state
        )

        # Compute adjoints through the forward function
        solver_state_adj, tf_adj, context_adj, zc_events_adj = vjp_fun(context_adj)

        # tf_adj is the time derivative of the state at the final time
        yf = primal_solver_state.y
        yf_bar = solver_state_adj.y
        func = sim.ode_solver.flat_ode_rhs
        tf_adj = jnp.dot(func(yf, tf, context), yf_bar)

        tf_adj = jnp.where(
            triggered,
            tf_adj,
            jnp.zeros_like(tf_adj),
        )

        # T-001c-followup #1d (cross-block saltation): add the event-time
        # (saltation) gradient for events whose owning block has no continuous
        # state — the per-block saltation adjoint cannot reach the dynamics
        # jump those events induce in a *downstream* block, so supply it here at
        # the global level (where the full continuous costate λ⁺ is available).
        context_adj = _cross_block_saltation_correction(
            sim, triggered, ctx_pre, f_minus, f_plus, lam_plus_flat,
            zc_events_out, context_adj,
        )

        return (solver_state_adj, None, tf_adj, context_adj, zc_events_adj)

    guarded_integrate = jax.custom_vjp(_wrapped_solve, nondiff_argnums=(0,))
    guarded_integrate.defvjp(_wrapped_solve_fwd, _wrapped_solve_adj)

    # Preserve docstring and annotations
    guarded_integrate.__doc__ = simulator._guarded_integrate.__doc__
    guarded_integrate.__annotations__ = simulator._guarded_integrate.__annotations__

    return partial(guarded_integrate, simulator)
