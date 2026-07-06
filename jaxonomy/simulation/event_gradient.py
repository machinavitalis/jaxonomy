# SPDX-License-Identifier: MIT

"""Public event-time gradient helpers (T-125 phase 1).

Event-time gradient — also known as the *saltation* or *shock* gradient — is
the sensitivity of the time at which a zero-crossing event fires with respect
to parameters.  For a bouncing ball with state ``(y, v)`` and guard
``g(y) = y``, this is ``∂t_bounce/∂h0``.

Math
----

At the event boundary the guard satisfies ``g(t_e, x(t_e; p), p) = 0``.
Total derivative w.r.t. ``p`` (treating ``t_e`` as an implicit function of
``p``) gives::

    0 = ∂g/∂t · dt_e/dp
        + ∂g/∂x · (ẋ · dt_e/dp + ∂x/∂p)
        + ∂g/∂p

Solving for ``dt_e/dp``::

    dt_e/dp = -(∂g/∂x · ∂x/∂p + ∂g/∂p) / (∂g/∂x · ẋ + ∂g/∂t)

where the denominator is the directional derivative of the guard along
the trajectory.  Vanishing denominators correspond to grazing / tangential
crossings — the implicit-function theorem fails there.

The :func:`event_time_gradient` helper computes the formula via
``jax.grad`` of an implicit residual so that arbitrary ``params`` PyTrees
are handled uniformly.

Usage
-----

This is a side-car helper: callers supply the guard, rhs, recorded event
time / state, and (optionally) the trajectory sensitivity ``∂x_e/∂p`` in
the form of a function ``state_at_event_fn(p) -> state``.  No internal
simulator state is required.  The deeper integration of event-time
gradients into ``simulate``'s custom-VJP path is deferred to
``T-125-followup-custom-vjp``.

The helper is fully JAX-traceable: callers can chain it under
``jax.grad`` / ``jax.jacfwd`` of downstream cost functions.

See ``test/autodiff/test_t_125_event_time_grad_phase1.py`` for the
worked bouncing-ball example.
"""

from __future__ import annotations

import functools
from typing import Any, Callable

import jax
import jax.numpy as jnp

from ..backend import numpy_api as npa


__all__ = [
    "event_time_gradient",
    "event_time_jacobian",
    "event_times_gradient",
    "multi_event_time_gradient",
    "simulate_with_event_time_grad",
    "vmap_event_time_gradient",
    "vmap_event_times_gradient",
]


def event_time_gradient(
    guard_fn: Callable[[float, Any, Any], jnp.ndarray],
    ode_rhs_fn: Callable[[float, Any, Any], Any],
    t_event: jnp.ndarray,
    state_at_event_fn: Callable[[Any], Any] | Any,
    params: Any,
    *,
    eps: float = 1e-30,
) -> Any:
    """Compute ``∂t_event/∂params`` via the implicit-function theorem.

    Args:
        guard_fn: ``(t, state, params) -> scalar`` — the zero-crossing
            guard.  Must be JAX-traceable in all three arguments.
        ode_rhs_fn: ``(t, state, params) -> dstate/dt`` — the continuous
            RHS evaluated at the event boundary.  Same PyTree structure
            as the state.
        t_event: Scalar time at which the guard fires.
        state_at_event_fn: Either
            * a callable ``params -> state`` that reconstructs the
              recorded event state from the parameters (so JAX can
              propagate the trajectory sensitivity ``∂x_e/∂p``), or
            * a constant PyTree of state values (no implicit dependence
              on ``params``).
            The callable form is the general case; the constant form is
            equivalent to passing ``lambda p: <constant>`` and is useful
            when the user only wants the explicit ``∂g/∂p`` contribution.
        params: Parameter PyTree to differentiate with respect to.  May
            be a scalar, ndarray, or any nested container.
        eps: Floor used to clip the denominator
            ``(∂g/∂x · ẋ + ∂g/∂t)`` away from zero before division — keeps
            ``jax.grad`` finite at grazing crossings.  Sign-preserving.

    Returns:
        The PyTree of ``∂t_event/∂params`` with the same structure as
        ``params``.
    """
    t_e = jnp.asarray(t_event)

    # Normalise state_at_event_fn into a callable.
    if callable(state_at_event_fn):
        _state_fn = state_at_event_fn
    else:
        _const = state_at_event_fn
        def _state_fn(_p):  # noqa: ANN001
            return _const

    # Evaluate state at event for use in the rhs.
    x_e = _state_fn(params)

    # Denominator: dg/dx · ẋ + dg/dt, evaluated at (t_e, x_e, params).
    f_at_event = ode_rhs_fn(t_e, x_e, params)

    def _g_of_t(t):
        return guard_fn(t, x_e, params)

    dg_dt = jax.grad(_g_of_t)(t_e)

    # Compute ∂g/∂x · f via a JVP — avoids materialising the full Jacobian.
    _, inner = jax.jvp(
        lambda x: guard_fn(t_e, x, params),
        (x_e,),
        (f_at_event,),
    )
    denom = inner + dg_dt

    # Clip denominator away from exact zero — keeps grad finite at
    # grazing crossings.  Sign-preserving floor.
    safe_denom = jnp.where(
        jnp.abs(denom) < eps,
        jnp.where(denom >= 0, eps, -eps),
        denom,
    )

    # Numerator: total derivative of the residual ``R(p) = guard(t_e,
    # state_fn(p), p)`` w.r.t. ``p`` — JAX handles the chain rule across
    # the ``state_fn`` dependency and the explicit ``params`` dependency
    # uniformly.
    def _residual(p):
        return guard_fn(t_e, _state_fn(p), p)

    dR_dp = jax.grad(_residual)(params)

    # dt_e/dp = -dR_dp / safe_denom.
    return jax.tree_util.tree_map(lambda r: -r / safe_denom, dR_dp)


def event_time_jacobian(
    guard_fn: Callable[[float, Any, Any], jnp.ndarray],
    ode_rhs_fn: Callable[[float, Any, Any], Any],
    t_event: jnp.ndarray,
    state_at_event_fn: Callable[[Any], Any] | Any,
    params: jnp.ndarray,
    *,
    eps: float = 1e-30,
) -> jnp.ndarray:
    """Vector-valued convenience wrapper of :func:`event_time_gradient`.

    Identical semantics, but returns a flat ndarray so that the result
    composes cleanly with downstream linear-algebra (Sobol sampling,
    Fisher information, etc.).  ``params`` should be a 1-D array.

    For a 1-D ``params`` array of length ``n_p``, returns shape ``(n_p,)``.
    """
    grad = event_time_gradient(
        guard_fn,
        ode_rhs_fn,
        t_event,
        state_at_event_fn,
        params,
        eps=eps,
    )
    return npa.asarray(grad)


def event_times_gradient(
    results: Any,
    params: Any,
    guards: Any,
    ode_rhs_fn: Callable[[float, Any, Any], Any],
    state_at_event_fn: Callable[[float, Any], Any],
    *,
    event_indices: Any = None,
    eps: float = 1e-30,
) -> dict:
    """Batch event-time gradients across all firings recorded by
    ``simulate(..., options=SimulatorOptions(record_event_times=True))``.

    For each recorded event in ``results.event_times``, applies the
    implicit-function theorem (T-125 phase 1) to every firing instant
    and returns the per-firing gradient PyTrees keyed by event index.

    Args:
        results: A :class:`SimulationResults` whose ``event_times`` is
            populated (i.e., the simulation was run with
            ``SimulatorOptions(record_event_times=True)``).  Calling
            this helper on a ``results`` whose ``event_times is None``
            raises ``ValueError`` with the remediation hint — the
            default-off path is preserved by simply not invoking this
            helper.
        params: Parameter PyTree to differentiate with respect to.
            Same semantics as :func:`event_time_gradient`.
        guards: Either a single guard callable
            ``(t, state, params) -> scalar`` applied to every recorded
            event, or a mapping ``{event_index: guard_fn}`` providing
            a distinct guard per event slot.  The latter form is
            intended for multi-event diagrams where each event index
            has its own zero-crossing function.
        ode_rhs_fn: ``(t, state, params) -> dstate/dt`` — same as in
            :func:`event_time_gradient`.  Reused across all firings.
        state_at_event_fn: ``(t_e, params) -> state`` — reconstructs
            the trajectory state at firing time ``t_e`` parametrized by
            ``params``.  This is the simpler ``state_fn`` form noted in
            the T-125-followup-multi-event task spec: callers express
            per-event-class behaviour via the ``t_e`` argument rather
            than per-event callables.  The deeper per-event-class
            form is a deferred followup.
        event_indices: Optional iterable of event indices to compute
            gradients for.  When ``None`` (default), every event index
            present in ``results.event_times`` is processed.  Indices
            not present in ``results.event_times`` raise ``KeyError``.
        eps: Forwarded to :func:`event_time_gradient` — denominator
            floor for grazing crossings.

    Returns:
        ``{event_index: stacked_gradient}`` — for each event index,
        the per-firing gradients stacked along a leading axis (so a
        gradient that is itself a PyTree leaf of shape ``S`` becomes
        an array of shape ``(n_firings,) + S``; PyTree containers are
        preserved by mapping the stack over leaves).  Events that
        fired zero times yield an empty leading axis.
    """
    event_times_dict = getattr(results, "event_times", None)
    if event_times_dict is None:
        raise ValueError(
            "event_times_gradient: results.event_times is None. "
            "Re-run simulate(...) with "
            "SimulatorOptions(record_event_times=True) so the firing "
            "instants are captured."
        )

    if callable(guards):
        # Single guard for every event index.
        def _guard_for(_idx):  # noqa: ANN001
            return guards
    else:
        # Per-event mapping.
        guards_map = dict(guards)
        def _guard_for(idx):
            if idx not in guards_map:
                raise KeyError(
                    f"event_times_gradient: no guard supplied for event "
                    f"index {idx}.  Provide guards[{idx}] = <fn> or pass "
                    f"a single callable to apply uniformly."
                )
            return guards_map[idx]

    if event_indices is None:
        selected = list(event_times_dict.keys())
    else:
        selected = list(event_indices)
        for idx in selected:
            if idx not in event_times_dict:
                raise KeyError(
                    f"event_times_gradient: event index {idx} not present "
                    f"in results.event_times (have: "
                    f"{sorted(event_times_dict.keys())})."
                )

    out: dict = {}
    for idx in selected:
        firings = jnp.asarray(event_times_dict[idx])
        guard_fn = _guard_for(idx)
        n_firings = int(firings.shape[0]) if firings.ndim >= 1 else 0

        if n_firings == 0:
            # Empty firing set — surface a structurally-correct empty
            # leading axis by computing a single dummy gradient at
            # ``t_e=0`` and slicing it off.  Avoids special-casing the
            # PyTree shape downstream.
            template = event_time_gradient(
                guard_fn,
                ode_rhs_fn,
                jnp.asarray(0.0),
                lambda p: state_at_event_fn(jnp.asarray(0.0), p),
                params,
                eps=eps,
            )
            out[idx] = jax.tree_util.tree_map(lambda leaf: leaf[None][:0], template)
            continue

        per_firing_grads = []
        for k in range(n_firings):
            t_e = firings[k]
            # Bind the firing time into the state callable so the
            # T-125 helper sees the standard ``state_fn(p) -> state``
            # signature.
            def _state_fn_for_firing(p, _t_e=t_e):
                return state_at_event_fn(_t_e, p)
            g_k = event_time_gradient(
                guard_fn,
                ode_rhs_fn,
                t_e,
                _state_fn_for_firing,
                params,
                eps=eps,
            )
            per_firing_grads.append(g_k)

        # Stack per-firing gradients leafwise so the PyTree structure
        # is preserved with a new leading axis of length n_firings.
        out[idx] = jax.tree_util.tree_map(
            lambda *leaves: jnp.stack(leaves, axis=0),
            *per_firing_grads,
        )

    return out


# ──────────────────────────────────────────────────────────────────────────
# T-125-followup-multi-event-saltation: forward-sensitivity propagation
# through reset maps.
# ──────────────────────────────────────────────────────────────────────────
#
# :func:`event_time_gradient` and :func:`event_times_gradient` rely on a
# user-supplied ``state_at_event_fn(p) -> x_e`` to carry the trajectory
# sensitivity ``∂x_e/∂p`` into the implicit-function-theorem formula.  That
# works for a *single* event because the user can write the closed-form arc
# (e.g. a free-fall drop).  For events *past the first* — every firing that
# is reached only after one or more reset maps have fired — the user would
# have to compose every prior reset map analytically in JAX-traceable form,
# which is exactly the thing that is not generally possible.  Without the
# correct ``∂x_e/∂p`` the saltation gradient collapses to ~zero where finite
# differences report a materially nonzero value (the original
# TiltedFloorBall report).
#
# :func:`multi_event_time_gradient` closes the gap by propagating the
# forward sensitivity ``S(t) = ∂x(t;p)/∂p`` along the *recorded* trajectory:
#
#   * within each smooth arc, ``S`` obeys the variational equation
#     ``Ṡ = (∂f/∂x) S + ∂f/∂p`` (integrated alongside the state), and
#   * at each event the saltation jump
#     ``S⁺ = R_x S⁻ + R_p + (R_t + R_x f⁻ − f⁺) (dt_e/dp)``
#     corrects ``S`` for both the reset map's explicit dependence and the
#     fact that the firing instant ``t_e(p)`` itself moves with ``p``
#     (Hiskens & Pai 2000, trajectory sensitivities of hybrid systems).
#
# The per-event ``dt_e/dp`` then comes from the same implicit-function
# formula as :func:`event_time_gradient`, but with the *correctly
# propagated* ``S⁻(t_e)`` instead of the user's closed-form guess.


def multi_event_time_gradient(
    guard_fn: Callable[[float, Any, Any], jnp.ndarray] | Any,
    ode_rhs_fn: Callable[[float, Any, Any], Any],
    reset_map_fn: Callable[[float, Any, Any], Any] | Any,
    initial_state: Callable[[Any], Any] | Any,
    event_times: Any,
    params: Any,
    *,
    t0: float = 0.0,
    eps: float = 1e-30,
    rtol: float = 1e-10,
    atol: float = 1e-12,
    return_state_sensitivity: bool = False,
) -> Any:
    """Saltation gradient ``dt_e/dp`` for *every* firing along a hybrid
    trajectory, propagating the forward sensitivity through reset maps.

    Unlike :func:`event_time_gradient` — which needs the caller to supply a
    closed-form ``state_at_event_fn`` for the trajectory sensitivity, and so
    only gets the first firing right — this helper reconstructs
    ``∂x_e/∂p`` itself by integrating the variational equation along each
    recorded arc and applying the saltation jump at each event.  It is the
    correct path for multi-bounce / repeated-event problems where each
    firing re-initialises the arc from the previous reset map.

    Args:
        guard_fn: ``(t, state, params) -> scalar`` zero-crossing guard, or a
            sequence of such callables aligned with ``event_times`` (one per
            firing) for heterogeneous events.
        ode_rhs_fn: ``(t, state, params) -> dstate/dt`` continuous RHS,
            shared across all arcs.  Must be JAX-traceable.
        reset_map_fn: ``(t_e, state_minus, params) -> state_plus`` reset map
            applied at each firing, or a sequence aligned with
            ``event_times``.  Use the identity map
            (``lambda t, x, p: x``) for events that only *observe* a
            crossing without resetting state.
        initial_state: either a callable ``params -> x0`` (so the seed
            sensitivity ``S(t0) = ∂x0/∂p`` is captured) or a constant state
            PyTree (seed sensitivity is then zero).
        event_times: ordered sequence / array of recorded firing instants
            ``[t_1, ..., t_n]`` (strictly increasing, all ``> t0``).  These
            are the *recorded* primal event times — e.g. from
            ``results.event_times``.
        params: parameter PyTree to differentiate with respect to.
        t0: trajectory start time (default ``0.0``).
        eps: sign-preserving floor on the implicit-function denominator
            ``(∂g/∂x · ẋ + ∂g/∂t)`` — guards grazing crossings.
        rtol: relative tolerance for the augmented (state + sensitivity)
            arc integration.
        atol: absolute tolerance for the augmented arc integration.
        return_state_sensitivity: when ``True``, also return the list of
            pre-event forward sensitivities ``S⁻(t_e)`` (flat ``(n_x, n_p)``
            arrays) for inspection / debugging.

    Returns:
        The per-firing ``dt_e/dp`` stacked along a leading axis of length
        ``n`` and shaped like ``params`` (a scalar parameter yields shape
        ``(n,)``; a PyTree parameter yields the same PyTree with each leaf
        carrying a leading firing axis).  If ``return_state_sensitivity`` is
        ``True``, returns ``(grads, [S_minus_1, ..., S_minus_n])``.

    Notes:
        Fully JAX-traceable (the arc integration uses
        ``jax.experimental.ode.odeint``).  Default-off and purely additive:
        the simulator path is untouched and callers who don't import this
        helper pay zero cost.
    """
    from jax.flatten_util import ravel_pytree
    from jax.experimental.ode import odeint

    t_list = [jnp.asarray(t) for t in jnp.asarray(event_times)]
    n_events = len(t_list)

    # Resolve per-firing guard / reset callables.
    def _as_per_event(obj, name):
        if callable(obj):
            return [obj] * n_events
        seq = list(obj)
        if len(seq) != n_events:
            raise ValueError(
                f"multi_event_time_gradient: {name} has {len(seq)} entries "
                f"but there are {n_events} event_times; supply one callable "
                f"to apply uniformly or a sequence of matching length."
            )
        return seq

    guards = _as_per_event(guard_fn, "guard_fn")
    resets = _as_per_event(reset_map_fn, "reset_map_fn")

    # Normalise the initial-state spec into a callable.
    if callable(initial_state):
        _x0_fn = initial_state
    else:
        _const_x0 = initial_state
        def _x0_fn(_p):  # noqa: ANN001
            return _const_x0

    p_flat, unravel_p = ravel_pytree(params)
    n_p = p_flat.shape[0]

    # Seed state + sensitivity.  Flatten the state PyTree and capture the
    # unravel so guard/rhs/reset can be called in their native structure.
    x0 = _x0_fn(unravel_p(p_flat))
    x_flat, unravel_x = ravel_pytree(x0)
    n_x = x_flat.shape[0]

    def _x0_flat(pf):
        return ravel_pytree(_x0_fn(unravel_p(pf)))[0]

    # S(t0) = ∂x0/∂p  (zeros when initial_state is constant).
    S = jax.jacfwd(_x0_flat)(p_flat)  # (n_x, n_p)

    # Flat-coordinate adapters around the user callables.
    def _f_flat(t, xf, pf):
        return ravel_pytree(ode_rhs_fn(t, unravel_x(xf), unravel_p(pf)))[0]

    def _g_flat(guard, t, xf, pf):
        return jnp.asarray(guard(t, unravel_x(xf), unravel_p(pf)))

    def _r_flat(reset, t, xf, pf):
        return ravel_pytree(reset(t, unravel_x(xf), unravel_p(pf)))[0]

    def _integrate_arc(xf, Smat, ta, tb):
        z0 = jnp.concatenate([xf, Smat.reshape(-1)])

        def _aug(z, t):
            xx = z[:n_x]
            SS = z[n_x:].reshape(n_x, n_p)
            f = _f_flat(t, xx, p_flat)
            # Variational equation: Ṡ = (∂f/∂x) S + ∂f/∂p.
            Jx = jax.jacfwd(lambda a: _f_flat(t, a, p_flat))(xx)
            Jp = jax.jacfwd(lambda b: _f_flat(t, xx, b))(p_flat)
            dS = Jx @ SS + Jp
            return jnp.concatenate([f, dS.reshape(-1)])

        zf = odeint(_aug, z0, jnp.stack([ta, tb]), rtol=rtol, atol=atol)[-1]
        return zf[:n_x], zf[n_x:].reshape(n_x, n_p)

    grads_flat: list = []
    S_minus_list: list = []
    t_prev = jnp.asarray(t0)
    for k in range(n_events):
        t_e = t_list[k]
        guard = guards[k]
        reset = resets[k]

        # Advance state + sensitivity to the firing instant.
        x_flat, S = _integrate_arc(x_flat, S, t_prev, t_e)
        S_minus_list.append(S)

        # Implicit-function-theorem denominator + numerator at the event,
        # using the CORRECTLY propagated S⁻(t_e).
        f_minus = _f_flat(t_e, x_flat, p_flat)
        g_x = jax.grad(lambda a: _g_flat(guard, t_e, a, p_flat))(x_flat)
        g_p = jax.grad(lambda b: _g_flat(guard, t_e, x_flat, b))(p_flat)
        g_t = jax.grad(lambda tt: _g_flat(guard, tt, x_flat, p_flat))(t_e)

        denom = g_x @ f_minus + g_t
        safe_denom = jnp.where(
            jnp.abs(denom) < eps,
            jnp.where(denom >= 0, eps, -eps),
            denom,
        )
        dtau_dp = -(g_x @ S + g_p) / safe_denom  # (n_p,)
        grads_flat.append(dtau_dp)

        # Apply the reset map and the saltation jump to S so the next arc
        # starts from the consistent post-event sensitivity.
        x_plus = _r_flat(reset, t_e, x_flat, p_flat)
        R_x = jax.jacfwd(lambda a: _r_flat(reset, t_e, a, p_flat))(x_flat)
        R_p = jax.jacfwd(lambda b: _r_flat(reset, t_e, x_flat, b))(p_flat)
        R_t = jax.jacfwd(lambda tt: _r_flat(reset, tt, x_flat, p_flat))(t_e)
        f_plus = _f_flat(t_e, x_plus, p_flat)
        S = R_x @ S + R_p + jnp.outer(R_t + R_x @ f_minus - f_plus, dtau_dp)
        x_flat = x_plus
        t_prev = t_e

    # Re-shape each per-firing flat gradient back into the params PyTree and
    # stack leafwise so the output mirrors event_time_gradient's structure
    # with a leading firing axis.
    if n_events == 0:
        template = jax.tree_util.tree_map(
            lambda leaf: leaf[None][:0], unravel_p(jnp.zeros(n_p))
        )
        grads = template
    else:
        per_firing = [unravel_p(g) for g in grads_flat]
        grads = jax.tree_util.tree_map(
            lambda *leaves: jnp.stack(leaves, axis=0), *per_firing
        )

    if return_state_sensitivity:
        return grads, S_minus_list
    return grads


# ──────────────────────────────────────────────────────────────────────────
# T-125-followup-custom-vjp: `jax.custom_vjp` integration
# ──────────────────────────────────────────────────────────────────────────
#
# Wraps a normal :func:`simulate` call so that the recorded event time
# becomes a *differentiable* output: ``jax.grad(simulate_with_event_time_grad
# )(params)`` directly returns the implicit-function-theorem gradient
# ``dt_e/dp``.  The forward pass runs ``simulate(...)`` (with
# ``record_event_times=True``) as a black-box ``jax.pure_callback`` so the
# Python-side event recorder fires reliably; the backward pass invokes
# :func:`event_time_gradient` to compute the saltation gradient and scales
# it by the upstream cotangent.
#
# Why ``pure_callback``?  ``simulate`` populates ``results.event_times``
# via a Python-side aggregator fed by ``jax.debug.callback``.  Under
# nested tracing (e.g. ``jax.grad`` over ``jax.jit``), the aggregator
# wouldn't be visible at trace time.  ``pure_callback`` lets us treat the
# whole forward simulate as an opaque op that returns a single scalar
# ``t_event``; the gradient is supplied by the implicit-function rule, so
# JAX never needs to differentiate *through* ``simulate``.
#
# Defaults preserve byte-equivalence: the wrapper does not touch
# ``simulate``'s code path; opting out (i.e. not calling the wrapper) is
# zero-cost.


def _default_sim_runner(
    diagram,
    ctx,
    t_span,
    params,  # noqa: ARG001 — passed for symmetry; user injects into ctx upstream
    event_index,
    options,
):
    """Internal helper — runs ``simulate(...)`` with
    ``record_event_times=True`` and returns ``t_event[event_index][0]`` as
    a float scalar.

    Splits cleanly out of :func:`simulate_with_event_time_grad` so the
    forward path can be substituted by a synthetic ``sim_runner`` in
    tests that don't depend on the full ``simulate`` machinery.
    """
    # Lazy import — avoids a hard circular dependency between
    # ``event_gradient`` and ``simulator`` at import time.
    from .simulator import simulate
    from .types import SimulatorOptions

    # Merge the caller's options with ``record_event_times=True``.  Mutating
    # a dataclass via ``replace`` keeps the user's other knobs intact and
    # avoids an in-place mutation on a (possibly shared) options object.
    import dataclasses

    if options is None:
        merged = SimulatorOptions(record_event_times=True)
    else:
        merged = dataclasses.replace(options, record_event_times=True)

    results = simulate(diagram, ctx, t_span, options=merged)
    event_times = getattr(results, "event_times", None)
    if event_times is None or event_index not in event_times:
        raise RuntimeError(
            "simulate_with_event_time_grad: event index "
            f"{event_index!r} not present in results.event_times "
            f"(have: {None if event_times is None else sorted(event_times.keys())}). "
            "Did the guard ever fire within the simulated time span?"
        )
    firings = event_times[event_index]
    if firings is None or len(firings) == 0:
        raise RuntimeError(
            "simulate_with_event_time_grad: event index "
            f"{event_index!r} recorded zero firings.  Phase 1 of this "
            "wrapper only supports the FIRST firing; the guard must "
            "trigger at least once within the simulated time span."
        )
    return float(firings[0])


def simulate_with_event_time_grad(
    diagram,
    ctx,
    t_span,
    params,
    event_index: int,
    guard_fn: Callable[[float, Any, Any], jnp.ndarray],
    ode_rhs_fn: Callable[[float, Any, Any], Any],
    state_at_event_fn: Callable[[Any], Any] | Any,
    options=None,
    *,
    sim_runner: Callable[..., float] | None = None,
    eps: float = 1e-30,
) -> jnp.ndarray:
    """Differentiable wrapper around ``simulate`` for event-time gradients.

    Returns the scalar firing time ``t_event`` of the FIRST recorded
    firing of ``event_index`` and registers a ``jax.custom_vjp`` rule
    that uses the implicit-function theorem (T-125 phase 1) for the
    reverse-mode gradient.  As a consequence::

        jax.grad(simulate_with_event_time_grad)(diagram, ctx, t_span,
                                                params, event_index,
                                                guard_fn, ode_rhs_fn,
                                                state_at_event_fn)

    yields ``∂t_event/∂params`` without the caller having to invoke
    :func:`event_time_gradient` manually.

    Args:
        diagram: SystemBase passed straight to :func:`simulate`.
        ctx: ContextBase passed straight to :func:`simulate`.  The
            caller is responsible for injecting ``params`` into ``ctx``
            (e.g. via ``ctx.with_parameter(...)``) so the forward sim
            uses the requested parameter values.
        t_span: ``(t0, t1)`` tuple passed to :func:`simulate`.
        params: Parameter PyTree to differentiate with respect to.  Same
            semantics as :func:`event_time_gradient` — the wrapper does
            not modify ``ctx`` from this value; it is used only for the
            backward rule.
        event_index: Integer event slot whose firing time is returned.
        guard_fn: ``(t, state, params) -> scalar`` — zero-crossing
            guard used by the implicit-function backward rule.
        ode_rhs_fn: ``(t, state, params) -> dstate/dt`` — continuous
            RHS evaluated at the event boundary.
        state_at_event_fn: Either
            * ``(t_e, params) -> state`` — preferred signature, matches
              :func:`event_times_gradient`.  The wrapper passes the
              recorded ``t_e`` as a *concrete* Python float so the
              implicit-function-theorem chain rule sees a non-trivial
              ``∂x_e/∂p`` (in particular, ``y(t_e_fixed, h0) =
              h0 - g t_e²/2`` has ``∂/∂h0 = 1`` even though
              ``y(t_e(h0), h0) ≡ 0``).
            * ``params -> state`` — single-arg form, identical to the
              one accepted by :func:`event_time_gradient`.  Useful when
              the caller has already bound ``t_e`` into a closure.
            The wrapper auto-detects which form was passed by argument
            count.  See :func:`event_time_gradient` for the full
            contract on the constant-state-PyTree form.
        options: Optional :class:`SimulatorOptions`.  The wrapper
            forwards a copy with ``record_event_times=True`` to
            :func:`simulate`; ``options is None`` (default) constructs
            a fresh ``SimulatorOptions(record_event_times=True)``.
        sim_runner: ``(diagram, ctx, t_span, params, event_index, options)
            -> float`` — optional override for the forward simulate
            call.  Defaults to the standard :func:`simulate` path.
            Tests use this hook to substitute analytic forward
            trajectories where wiring a full ``simulate`` call would be
            disproportionate.
        eps: Floor for the implicit-function denominator (forwarded to
            :func:`event_time_gradient`).

    Returns:
        Scalar ``jnp.ndarray`` holding ``t_event``.

    Notes:
        Composes with ``jax.jit`` and ``jax.vmap``: the forward pass
        runs as a ``jax.pure_callback`` (black-box w.r.t. JAX), and the
        backward pass uses :func:`event_time_gradient` which is itself
        JAX-traceable.  Default-off byte-equivalence is preserved — the
        existing :func:`event_time_gradient` and :func:`simulate` are
        not touched by this wrapper.
    """
    runner = sim_runner if sim_runner is not None else _default_sim_runner

    # Bind all non-pytree / non-traced args via closure; ``custom_vjp``
    # only differentiates with respect to ``params``.
    @jax.custom_vjp
    def _wrapped(params_):
        return _fwd_event_time(params_, diagram, ctx, t_span,
                               event_index, options, runner)

    def _fwd(params_):
        t_e = _fwd_event_time(params_, diagram, ctx, t_span,
                              event_index, options, runner)
        # Save everything the backward rule needs.  ``params_`` flows in
        # as a JAX value; ``t_e`` is the scalar firing time.
        return t_e, (params_, t_e)

    def _bwd(residuals, cotangent):
        params_, t_e = residuals
        # Normalize the ``state_at_event_fn`` argument count so that
        # :func:`event_time_gradient` always sees its standard
        # ``state_fn(params) -> state`` form.  Critically, we bind ``t_e``
        # in as a *concrete* Python float when the user supplied the
        # ``(t_e, params)`` form — this guarantees ``∂x_e/∂p`` is taken
        # at the recorded firing instant rather than along the
        # parameter-dependent ``t_e(p)`` curve (which would zero out
        # the implicit dependence; see the note in the API docstring).
        if callable(state_at_event_fn):
            try:
                import inspect
                n_args = len(inspect.signature(state_at_event_fn).parameters)
            except (TypeError, ValueError):
                n_args = 1
            if n_args >= 2:
                # Capture ``t_e`` by closure; ``jax.lax.stop_gradient``
                # severs any cotangent path *through* ``t_e`` so that
                # the implicit-function gradient is taken at the
                # recorded firing instant rather than along the
                # ``t_e(p)`` curve.  Works whether ``t_e`` is a concrete
                # scalar (eager) or a JIT tracer (under ``jax.jit`` /
                # ``jax.vmap``) — the standalone
                # :func:`event_time_gradient` already treats ``t_event``
                # as a constant w.r.t. the differentiation parameter
                # for the same reason.
                t_e_bound = jax.lax.stop_gradient(t_e)
                def _state_fn_bound(_p, _t=t_e_bound):
                    return state_at_event_fn(_t, _p)
            else:
                _state_fn_bound = state_at_event_fn
        else:
            _state_fn_bound = state_at_event_fn  # constant state PyTree

        # Implicit-function-theorem gradient (T-125 phase 1).
        dt_dp = event_time_gradient(
            guard_fn,
            ode_rhs_fn,
            t_e,
            _state_fn_bound,
            params_,
            eps=eps,
        )
        # Chain rule: upstream cotangent (scalar) times dt_e/dp.
        scaled = jax.tree_util.tree_map(lambda g: cotangent * g, dt_dp)
        return (scaled,)

    _wrapped.defvjp(_fwd, _bwd)
    return _wrapped(params)


def _fwd_event_time(params, diagram, ctx, t_span, event_index, options, runner):
    """Forward black-box: returns the scalar event firing time.

    Runs the simulator via ``jax.pure_callback`` so the forward path is
    opaque to JAX tracing (callback fires concretely each grad / jit
    invocation).  The output dtype follows the T-005 default-float64
    policy via ``jnp.asarray`` upcast — ``pure_callback`` is told the
    output is a float64 scalar.
    """
    def _run(_p):
        # ``_p`` here is the concrete numpy view of ``params`` at the
        # moment the callback fires.  We forward it to the runner so
        # tests using a synthetic ``sim_runner`` can read it; the
        # default ``simulate`` runner ignores it (the user is expected
        # to have already injected the values into ``ctx``).
        return npa.asarray(runner(diagram, ctx, t_span, _p, event_index, options),
                           dtype=jnp.float64).reshape(())

    # ``pure_callback`` is a JAX-tracing op: under ``jit``/``grad`` it
    # behaves as a black box returning the declared shape/dtype.
    # ``vmap_method="sequential"`` makes the callback ``vmap``-able by
    # looping over batched inputs — phase 1's correctness contract holds
    # per-element, which matches the wrapper's single-event semantics.
    result_shape = jax.ShapeDtypeStruct((), jnp.float64)
    return jax.pure_callback(
        _run, result_shape, params, vmap_method="sequential",
    )


# ──────────────────────────────────────────────────────────────────────────
# T-125-followup-batch-event-grad: vmap-batch event-time gradient wrapper
# ──────────────────────────────────────────────────────────────────────────
#
# Convenience helper for Monte-Carlo / parameter-sweep workflows.
#
# The standalone :func:`event_time_gradient` already composes with
# ``jax.vmap`` when the caller is careful with their closures:
#
#   * The closed-over ``t_event`` must be a JAX value (no Python ``float``
#     cast), and
#   * any ``state_fn`` that depends on ``t_event`` should treat it as a
#     constant w.r.t. the differentiation parameter via
#     ``jax.lax.stop_gradient`` — same convention as
#     :func:`simulate_with_event_time_grad`.
#
# The :func:`vmap_event_time_gradient` wrapper takes a *batched* parameter
# PyTree (leading axis = sample axis) plus a per-sample ``t_event_array``
# and a 2-argument ``state_at_event_fn(t_e, params) -> state`` (same
# signature as :func:`event_times_gradient` and
# :func:`simulate_with_event_time_grad`).  It returns the stacked per-sample
# gradients with the same leading axis.  This eliminates the per-sample
# closure boilerplate users would otherwise have to write themselves.
#
# Default-off byte-equivalence: the wrapper is purely additive — callers
# who don't import it pay zero cost.  Internally it is a thin ``jax.vmap``
# over :func:`event_time_gradient`; if vmap composition ever regresses,
# pass ``use_python_loop=True`` for an honest fallback that calls the
# single-sample helper ``N`` times.


def vmap_event_time_gradient(
    guard_fn: Callable[[float, Any, Any], jnp.ndarray],
    ode_rhs_fn: Callable[[float, Any, Any], Any],
    t_event_array: jnp.ndarray,
    state_at_event_fn: Callable[[Any, Any], Any],
    params_batch: Any,
    *,
    eps: float = 1e-30,
    use_python_loop: bool = False,
) -> Any:
    """Vectorised event-time gradient over a batch of parameter samples.

    For ``N`` samples, computes ``∂t_event/∂params`` for each in turn and
    stacks the results along the leading axis — the same shape contract
    Monte-Carlo / Sobol workflows expect from :func:`simulate_batch`.

    Args:
        guard_fn: ``(t, state, params) -> scalar`` — zero-crossing guard.
            Same contract as :func:`event_time_gradient`; shared across
            the batch.
        ode_rhs_fn: ``(t, state, params) -> dstate/dt`` — RHS at the
            event boundary; shared across the batch.
        t_event_array: ``(N,)`` array of per-sample firing instants.
            Treated as a constant w.r.t. the differentiation parameter
            inside the wrapper (``jax.lax.stop_gradient``) so the
            implicit-function chain rule is taken at the recorded
            instant — same convention as
            :func:`simulate_with_event_time_grad`.
        state_at_event_fn: ``(t_e, params) -> state`` — reconstructs
            the trajectory state at firing time ``t_e`` parametrised by
            a single-sample ``params`` slice.  Identical signature to
            the one accepted by :func:`event_times_gradient`; the
            wrapper composes it with each ``t_event_array[i]`` and the
            ``i``-th slice of ``params_batch`` under ``jax.vmap``.
        params_batch: Batched parameter PyTree.  All leaves must share
            a leading axis of length ``N`` matching ``t_event_array``.
            May be a scalar batch (``shape (N,)`` ndarray), a vector
            batch (``shape (N, n_p)``), or a PyTree thereof.
        eps: Forwarded to :func:`event_time_gradient` — denominator floor
            for grazing crossings.
        use_python_loop: When ``True``, iterate explicitly over the
            sample axis instead of using ``jax.vmap``.  Slower but
            byte-identical; useful as a fallback if vmap composition
            ever breaks (e.g. under future JAX versions where a closure
            inside :func:`event_time_gradient` becomes non-vmap-friendly).

    Returns:
        Per-sample gradients with the same leading axis as
        ``params_batch``.  PyTree structure of each sample's gradient
        matches the single-sample :func:`event_time_gradient`.

    Notes:
        Default-off: the wrapper is purely additive and does not modify
        the simulator path.  Composes cleanly with ``jax.jit`` and
        downstream ``jax.grad`` of a scalar cost over the batch axis.
    """
    t_event_array = jnp.asarray(t_event_array)

    def _single_sample(t_e, params_i):
        # Bind ``t_e`` into the state callable so the inner helper sees
        # the canonical ``state_fn(params) -> state`` form.  Use
        # ``stop_gradient`` so the implicit-function gradient is taken
        # at the recorded firing instant rather than along the
        # ``t_e(params)`` curve — same convention as the custom-VJP
        # wrapper.
        t_e_const = jax.lax.stop_gradient(t_e)

        def _state_fn(p, _t=t_e_const):
            return state_at_event_fn(_t, p)

        return event_time_gradient(
            guard_fn,
            ode_rhs_fn,
            t_e,
            _state_fn,
            params_i,
            eps=eps,
        )

    if use_python_loop:
        # Honest fallback path — iterate the sample axis in Python and
        # stack the per-sample gradients leafwise.  Slower than vmap but
        # robust against any future vmap-composition regression inside
        # :func:`event_time_gradient`.
        n = int(t_event_array.shape[0])
        per_sample = []
        for i in range(n):
            params_i = jax.tree_util.tree_map(lambda leaf, _i=i: leaf[_i], params_batch)
            per_sample.append(_single_sample(t_event_array[i], params_i))
        if n == 0:
            # Empty batch — produce a structurally-correct empty leading
            # axis by computing a single dummy gradient and slicing it
            # off.  Mirrors :func:`event_times_gradient`'s empty-firing
            # convention.
            template_params = jax.tree_util.tree_map(
                lambda leaf: leaf[:1].reshape((1,) + leaf.shape[1:]) if leaf.ndim >= 1 else leaf,
                params_batch,
            )
            template_params_0 = jax.tree_util.tree_map(lambda leaf: leaf[0], template_params)
            template = _single_sample(jnp.asarray(0.0), template_params_0)
            return jax.tree_util.tree_map(lambda leaf: leaf[None][:0], template)
        return jax.tree_util.tree_map(
            lambda *leaves: jnp.stack(leaves, axis=0),
            *per_sample,
        )

    # Default path: jax.vmap.  Sample axis is the leading axis of every
    # leaf of ``params_batch`` and of ``t_event_array``.
    return jax.vmap(_single_sample, in_axes=(0, 0))(t_event_array, params_batch)


# ──────────────────────────────────────────────────────────────────────────
# T-125-followup-multi-events-batched-vmap: cross product of multi-event +
# batched parameter sweep.
# ──────────────────────────────────────────────────────────────────────────
#
# :func:`event_times_gradient` handles "many firings, one parameter sample";
# :func:`vmap_event_time_gradient` handles "one firing per sample, many
# samples".  Monte-Carlo / Sobol workflows that observe multi-event diagrams
# need both axes at once: for each recorded event index, compute the
# per-firing implicit-function-theorem gradient at every parameter sample.
#
# :func:`vmap_event_times_gradient` ships exactly that cross product.  The
# implementation is a Python-loop over the per-event axis (mirroring
# :func:`event_times_gradient`) wrapping a ``jax.vmap`` over the sample axis
# (mirroring :func:`vmap_event_time_gradient`).  Each per-event entry in
# the output dict has a leading "sample" axis matching ``params_batch`` and
# a secondary "firing" axis matching ``results.event_times[idx]``.
#
# Honest fallback: if the inner vmap composition ever regresses, pass
# ``use_python_loop=True`` to iterate explicitly over the sample axis as
# well.  Slower but byte-identical.
#
# Default-off byte-equivalence: purely additive — the simulator path is not
# touched, and callers who don't import this helper pay zero cost.


def vmap_event_times_gradient(
    results: Any,
    params_batch: Any,
    guards: Any,
    ode_rhs_fn: Callable[[float, Any, Any], Any],
    state_at_event_fn: Callable[[float, Any], Any],
    *,
    event_indices: Any = None,
    eps: float = 1e-30,
    use_python_loop: bool = False,
) -> dict:
    """Cross-product of multi-event + batched-parameter event-time gradient.

    For each event index recorded in ``results.event_times``, computes the
    implicit-function-theorem gradient ``∂t_event/∂params`` at every
    (sample, firing) pair and returns the result keyed by event index.

    Output contract::

        {event_index: gradient_batch}

    where ``gradient_batch`` has leading axes ``(N, n_firings, ...)`` for
    array-valued ``params_batch`` leaves and is itself a PyTree mirroring
    the structure of ``params_batch`` for nested batches.  ``N`` is the
    sample-axis length (shared across all batch leaves); ``n_firings`` is
    the per-event firing count read from ``results.event_times[idx]``.

    Args:
        results: A :class:`SimulationResults` whose ``event_times`` is
            populated (i.e., the simulation was run with
            ``SimulatorOptions(record_event_times=True)``).  Same
            requirement as :func:`event_times_gradient`.  A ``results``
            whose ``event_times is None`` raises ``ValueError`` with the
            remediation hint.
        params_batch: Batched parameter PyTree.  All leaves must share a
            leading axis of length ``N``.  Same contract as
            :func:`vmap_event_time_gradient`.
        guards: Either a single guard callable ``(t, state, params) ->
            scalar`` applied to every event, or a mapping
            ``{event_index: guard_fn}``.  Same semantics as
            :func:`event_times_gradient`.
        ode_rhs_fn: ``(t, state, params) -> dstate/dt`` — shared across
            firings and samples.
        state_at_event_fn: ``(t_e, params) -> state`` — reconstructs the
            trajectory state at firing time ``t_e`` parametrised by a
            single-sample ``params`` slice.  Same signature as
            :func:`event_times_gradient` and
            :func:`vmap_event_time_gradient`.
        event_indices: Optional iterable of event indices to compute
            gradients for.  When ``None``, every recorded event index is
            processed.  Indices not present in ``results.event_times``
            raise ``KeyError``.
        eps: Forwarded to :func:`event_time_gradient` — denominator floor
            for grazing crossings.
        use_python_loop: When ``True``, iterate explicitly over both the
            firing and sample axes in Python.  Slower but byte-identical;
            honest fallback when vmap composition is invasive.

    Returns:
        ``{event_index: gradient_batch}`` — one entry per processed
        event index.  For each entry, leaves carry leading axes
        ``(N, n_firings, ...)``.  Empty firing lists yield a structurally
        correct ``(N, 0, ...)`` leading-axis pair.

    Notes:
        Default-off: purely additive.  Composes with ``jax.jit``.  The
        firing times read from ``results.event_times`` are treated as
        constants w.r.t. ``params_batch`` (the implicit-function theorem
        is applied at the recorded instants — same convention as
        :func:`vmap_event_time_gradient` and
        :func:`simulate_with_event_time_grad`).
    """
    event_times_dict = getattr(results, "event_times", None)
    if event_times_dict is None:
        raise ValueError(
            "vmap_event_times_gradient: results.event_times is None. "
            "Re-run simulate(...) with "
            "SimulatorOptions(record_event_times=True) so the firing "
            "instants are captured."
        )

    if callable(guards):
        def _guard_for(_idx):  # noqa: ANN001
            return guards
    else:
        guards_map = dict(guards)
        def _guard_for(idx):
            if idx not in guards_map:
                raise KeyError(
                    f"vmap_event_times_gradient: no guard supplied for "
                    f"event index {idx}.  Provide guards[{idx}] = <fn> "
                    f"or pass a single callable to apply uniformly."
                )
            return guards_map[idx]

    if event_indices is None:
        selected = list(event_times_dict.keys())
    else:
        selected = list(event_indices)
        for idx in selected:
            if idx not in event_times_dict:
                raise KeyError(
                    f"vmap_event_times_gradient: event index {idx} not "
                    f"present in results.event_times (have: "
                    f"{sorted(event_times_dict.keys())})."
                )

    # Validate that ``params_batch`` carries a sample axis on every leaf
    # and infer ``N`` from the first leaf.  Mirrors the in_axes=0 contract
    # of :func:`vmap_event_time_gradient`.
    leaves = jax.tree_util.tree_leaves(params_batch)
    if not leaves:
        raise ValueError(
            "vmap_event_times_gradient: params_batch has no array leaves; "
            "cannot infer batch size."
        )
    n_samples = int(jnp.asarray(leaves[0]).shape[0])

    out: dict = {}
    for idx in selected:
        firings = jnp.asarray(event_times_dict[idx])
        guard_fn = _guard_for(idx)
        n_firings = int(firings.shape[0]) if firings.ndim >= 1 else 0

        if n_firings == 0:
            # Empty firing set — produce a structurally-correct
            # ``(N, 0, ...)`` leading pair by computing a single dummy
            # gradient via :func:`vmap_event_time_gradient` at ``t_e=0``
            # for every sample, then slicing out the firing axis.
            dummy_t = jnp.zeros((n_samples,), dtype=jnp.float64)
            template_batch = vmap_event_time_gradient(
                guard_fn,
                ode_rhs_fn,
                dummy_t,
                state_at_event_fn,
                params_batch,
                eps=eps,
                use_python_loop=use_python_loop,
            )
            # template_batch leaves have shape (N, ...).  Insert an empty
            # firing axis at position 1 -> (N, 0, ...).
            out[idx] = jax.tree_util.tree_map(
                lambda leaf: leaf[:, None, ...][:, :0, ...],
                template_batch,
            )
            continue

        # Per-firing pass: for each firing instant, vmap the single-shot
        # gradient over the sample axis.  Stack the per-firing results
        # leafwise along axis=1 so the final leaf shape is
        # ``(N, n_firings, ...)``.
        per_firing_batches = []
        for k in range(n_firings):
            t_e_k = firings[k]
            # Broadcast the scalar firing time across the sample axis so
            # the inner vmap sees a ``(N,)`` ``t_event_array`` (matching
            # the leaf shape contract of :func:`vmap_event_time_gradient`).
            t_e_broadcast = jnp.broadcast_to(t_e_k, (n_samples,))

            grad_batch_k = vmap_event_time_gradient(
                guard_fn,
                ode_rhs_fn,
                t_e_broadcast,
                state_at_event_fn,
                params_batch,
                eps=eps,
                use_python_loop=use_python_loop,
            )
            per_firing_batches.append(grad_batch_k)

        # Stack along the new firing axis (position 1, since position 0 is
        # the sample axis).  ``tree_map(*pytrees)`` operates leafwise.
        out[idx] = jax.tree_util.tree_map(
            lambda *leaves: jnp.stack(leaves, axis=1),
            *per_firing_batches,
        )

    return out
