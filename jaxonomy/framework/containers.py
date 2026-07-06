# SPDX-License-Identifier: MIT

"""Container blocks.

This module ships the standard *container block* family:

- ``EnabledSubsystem(submodel, n_inputs, mode={"reset","passthrough","hold"})``
- ``TriggeredSubsystem(submodel, n_inputs, edge={"rising","falling","either"})``
- ``ForEach(submodel, n, n_inputs, in_axes=...)`` — a block-diagram-vocabulary
  alias for the existing ``ReplicatedFunction`` (T-010).
- ``ForLoop(body_fn, n_iter, ...)`` — runs ``body_fn`` ``n_iter`` times
  via ``jax.lax.fori_loop``.
- ``WhileLoop(body_fn, cond_fn, max_iter=...)`` — runs ``body_fn`` until
  ``cond_fn(carry)`` is False, capped at ``max_iter`` via
  ``jax.lax.while_loop``.

The phase-1 wrappers are thin layers over the existing ``Conditional``
(T-009) / ``ReplicatedFunction`` (T-010) / ``LeafSystem`` primitives.
The follow-up loop blocks call ``jax.lax.fori_loop`` /
``jax.lax.while_loop`` directly (these are JAX primitives; using them
through ``npa`` would just re-export them). We deliberately avoid
touching ``jaxonomy/library/primitives.py`` (parallel work in T-112,
T-121, T-127 and T-118 operates in that file) and reuse the existing
``jnp.where``-based masking pattern so:

- Default-off / non-touched-block path is byte-equivalent to the
  pre-T-120 codebase: nothing in ``jaxonomy/framework`` or
  ``jaxonomy/library`` re-exports any of these unless the user imports
  them explicitly from ``jaxonomy.library`` (we add the names to the
  library ``__init__`` for ergonomics, but the underlying classes live
  here).

- Gradients through the disabled / non-triggered branch are zero (via
  ``jnp.where``) rather than NaN, matching the existing ``Conditional``
  contract.

- ``ForLoop`` is differentiable through ``body_fn`` parameters as long
  as the body itself is pure; ``WhileLoop`` is differentiable through
  the carry but not through the loop count (which is data-dependent),
  matching standard JAX semantics for ``lax.while_loop``.
"""

from __future__ import annotations

import inspect
from typing import Callable, Literal

import jax
import jax.numpy as jnp

from .leaf_system import LeafSystem


def _accepts_extra_args(fn: Callable) -> bool:
    """Return True if ``fn`` accepts more than one positional argument.

    Used by :class:`WhileLoop` to decide whether ``cond_fn`` / ``body_fn``
    expect ``(carry, *inputs)`` or just ``(carry,)``.

    Strategy: introspect via :func:`inspect.signature`. The detection is
    intentionally conservative — anything that *can* take more than one
    positional argument (multiple POSITIONAL_OR_KEYWORD, POSITIONAL_ONLY,
    or a VAR_POSITIONAL ``*args``) is treated as "extra-args mode".
    Builtins / C-level callables where ``inspect.signature`` raises
    ``ValueError`` fall back to single-arg mode (backwards-compatible).
    """
    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError):
        # Builtins or C-level callables: assume legacy single-arg.
        return False
    positional_count = 0
    for param in sig.parameters.values():
        if param.kind in (
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
        ):
            positional_count += 1
        elif param.kind is inspect.Parameter.VAR_POSITIONAL:
            return True
    return positional_count > 1


__all__ = [
    "EnabledSubsystem",
    "TriggeredSubsystem",
    "ZeroCrossingTriggeredSubsystem",
    "ForEach",
    "ForLoop",
    "WhileLoop",
    "EnabledMode",
    "EnabledStateMode",
    "TriggerEdge",
]


# ---------------------------------------------------------------------------
# Allowed string values, mirroring the ``WhenDisabled`` style in
# ``jaxonomy.library.conditional``.
# ---------------------------------------------------------------------------


class EnabledMode:
    """Allowed string values for ``EnabledSubsystem.mode``."""

    RESET = "reset"
    HOLD = "hold"
    PASSTHROUGH = "passthrough"

    @classmethod
    def valid(cls) -> tuple[str, ...]:
        return (cls.RESET, cls.HOLD, cls.PASSTHROUGH)


class EnabledStateMode:
    """Allowed string values for ``EnabledSubsystem.state_mode``.

    Controls how the *continuous state* (declared via ``state_dynamics``)
    evolves while the enable signal is false:

    - ``HOLD`` (default): freeze the state at its current value
      (``xdot = 0`` while disabled). Resumes integration on re-enable.
    - ``RESET``: snap the state back to ``initial_state`` on every
      disable→enable transition (so each enable window starts from the
      configured initial value). While disabled, the state is held.
    - ``FREE``: the state evolves according to ``state_dynamics``
      regardless of enable. Only the *output* is masked per ``mode=``.
    """

    HOLD = "hold"
    RESET = "reset"
    FREE = "free"

    @classmethod
    def valid(cls) -> tuple[str, ...]:
        return (cls.HOLD, cls.RESET, cls.FREE)


class TriggerEdge:
    """Allowed string values for ``TriggeredSubsystem.edge``."""

    RISING = "rising"
    FALLING = "falling"
    EITHER = "either"

    @classmethod
    def valid(cls) -> tuple[str, ...]:
        return (cls.RISING, cls.FALLING, cls.EITHER)


# ---------------------------------------------------------------------------
# EnabledSubsystem
# ---------------------------------------------------------------------------


class EnabledSubsystem(LeafSystem):
    """Container block: run a submodel only while an enable signal is true.

    This is the subsystem-framing wrapper around the existing
    :class:`jaxonomy.library.Conditional` primitive (T-009). It exists
    as a separate class so that:

    - The block-diagram-vocabulary name ``EnabledSubsystem`` is discoverable
      next to the rest of the container family.
    - We can later extend the ``mode="hold"`` path with subsystem-state
      semantics (per-block discrete-state binding) without disturbing
      the lighter ``Conditional`` primitive.

    Args:
        submodel: Callable ``f(*inputs) -> output`` (single output per
            phase 1). Must be JAX-traceable.
        n_inputs: Number of submodel inputs (does NOT include the
            enable port). Input port 0 is always the enable signal;
            ports 1..n_inputs carry the submodel inputs.
        mode: One of ``"reset"`` / ``"passthrough"`` / ``"hold"``.

            - ``reset``: when disabled, output = ``initial_value``.
            - ``passthrough``: when disabled, output = first user input
              (input port 1). Submodel and passthrough output must
              broadcast-compatibly.
            - ``hold``: when disabled, output holds the most recent
              snapshot taken at ``hold_period``. Requires a positive
              ``hold_period``.
        initial_value: Output value when disabled in reset mode, and
            the seed for the held discrete state in hold mode. Used to
            infer output shape/dtype.
        hold_period: Sample period (seconds) for the held snapshot in
            hold mode. Required iff ``mode == "hold"``.
        state_mode: One of ``"hold"`` / ``"reset"`` / ``"free"``.
            Controls the *continuous-state* behaviour while disabled
            (independent of ``mode=`` which gates only the output).
            See :class:`EnabledStateMode`. Default ``"hold"``. Only has
            an effect when ``state_dynamics`` is provided; for the
            stateless submodel default this kwarg is validated but
            otherwise a no-op (so the default-off path is byte-equivalent
            to phase 1).
        state_dynamics: Optional callable
            ``f(t, x, *user_inputs) -> xdot`` defining a continuous
            state for the EnabledSubsystem itself. When provided, the
            block declares a continuous state seeded by ``initial_state``
            (or ``initial_value`` if ``initial_state`` is None) and
            applies ``state_mode`` semantics around it. When omitted,
            the block has no continuous state and behaves exactly as in
            T-120 phase 1.
        initial_state: Initial value of the continuous state. Required
            when ``state_dynamics`` is provided.
        name: Optional block name.
    """

    def __init__(
        self,
        submodel: Callable,
        n_inputs: int = 1,
        mode: Literal["reset", "passthrough", "hold"] = EnabledMode.RESET,
        initial_value=0.0,
        hold_period: float | None = None,
        state_mode: Literal["hold", "reset", "free"] = EnabledStateMode.HOLD,
        state_dynamics: Callable | None = None,
        initial_state=None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        if mode not in EnabledMode.valid():
            raise ValueError(
                f"EnabledSubsystem: mode must be one of "
                f"{EnabledMode.valid()!r}, got {mode!r}"
            )
        if state_mode not in EnabledStateMode.valid():
            raise ValueError(
                f"EnabledSubsystem: state_mode must be one of "
                f"{EnabledStateMode.valid()!r}, got {state_mode!r}"
            )
        if mode == EnabledMode.HOLD and not hold_period:
            raise ValueError(
                "EnabledSubsystem(mode='hold') requires a positive "
                "hold_period to determine the snapshot sample rate."
            )
        if n_inputs < 0:
            raise ValueError(
                f"EnabledSubsystem: n_inputs must be >= 0, got {n_inputs}"
            )
        if mode == EnabledMode.PASSTHROUGH and n_inputs < 1:
            raise ValueError(
                "EnabledSubsystem(mode='passthrough') requires at least "
                "one user input to use as the bypass signal."
            )
        if state_dynamics is not None and initial_state is None:
            raise ValueError(
                "EnabledSubsystem: state_dynamics= requires an "
                "initial_state= value (the seed for the continuous "
                "state)."
            )

        self._submodel = submodel
        self._mode = mode
        self._initial = jnp.asarray(initial_value)
        self._state_mode = state_mode
        self._state_dynamics = state_dynamics
        self._initial_state = (
            jnp.asarray(initial_state) if initial_state is not None else None
        )

        # Port 0 is always enable; remaining ports are submodel inputs.
        self.declare_input_port(name="enable")
        for i in range(n_inputs):
            self.declare_input_port(name=f"u_{i}")

        if mode == EnabledMode.HOLD:
            self.declare_discrete_state(default_value=self._initial)
            self.declare_periodic_update(
                self._hold_update,
                period=float(hold_period),
                offset=0.0,
            )

        # T-120-followup-enabled-cont-state: declare a continuous state on
        # this block when the user supplied a state_dynamics callable. The
        # state_mode kwarg controls how the state evolves vs. the enable
        # signal. The default-off path (state_dynamics=None) bypasses this
        # block entirely → byte-equivalent to phase 1.
        if self._state_dynamics is not None:
            self.declare_continuous_state(
                default_value=self._initial_state,
                ode=self._wrapped_ode,
            )

            if self._state_mode == EnabledStateMode.RESET:
                # Snap the continuous state back to its initial value on
                # every enable transition (rising or falling), so each
                # disable window leaves the state at the seed and each
                # re-enable starts from the same point. Pair this with
                # the "hold" ode-zeroing during the disabled window so
                # the state actually stays at the seed instead of drifting.
                #
                # Treat the enable signal as a centred continuous guard
                # (``enable - 0.5``): zero-crossings of this expression
                # match enable transitions in either direction. The
                # framework's continuous detector preserves the float
                # carry-type for the guard signal — using
                # ``direction="edge_detection"`` with a boolean guard
                # would conflict with the simulator's float-typed
                # internal carry, so we go through the continuous path.
                self.declare_zero_crossing(
                    guard=self._enable_guard,
                    reset_map=self._reset_continuous_state,
                    direction="crosses_zero",
                    name="enable_reset",
                )

        self.declare_output_port(
            self._compute_output,
            prerequisites_of_calc=[port.ticket for port in self.input_ports],
        )

    # ── callbacks ─────────────────────────────────────────────────────────

    def _submodel_output(self, inputs):
        user_inputs = inputs[1:]  # skip enable
        return jnp.asarray(self._submodel(*user_inputs))

    def _hold_update(self, time, state, *inputs, **params):
        enable = jnp.asarray(inputs[0]).astype(bool)
        y_sub = self._submodel_output(inputs)
        return jnp.where(enable, y_sub, state.discrete_state)

    def _compute_output(self, time, state, *inputs, **params):
        enable = jnp.asarray(inputs[0]).astype(bool)
        y_sub = self._submodel_output(inputs)

        if self._mode == EnabledMode.RESET:
            return jnp.where(enable, y_sub, self._initial)

        if self._mode == EnabledMode.HOLD:
            return jnp.where(enable, y_sub, state.discrete_state)

        # passthrough
        bypass = jnp.asarray(inputs[1])
        return jnp.where(enable, y_sub, bypass)

    # ── continuous-state callbacks (T-120-followup-enabled-cont-state) ───

    def _wrapped_ode(self, time, state, *inputs, **params):
        """Apply ``state_mode`` semantics to the user's ``state_dynamics``.

        - ``hold``: multiply the user's xdot by the enable flag. While
          disabled the ode returns zero, so the integrator preserves the
          state across the disabled window.
        - ``reset``: same as ``hold`` for the per-step ode (state stays
          at the seed during the disabled window); the reset to the
          initial value is enforced by a zero-crossing reset_map on the
          enable-edge.
        - ``free``: pass the user's xdot through untouched. The state
          evolves regardless of enable (only the output port is masked).
        """
        enable = jnp.asarray(inputs[0]).astype(bool)
        user_inputs = inputs[1:]
        xc = state.continuous_state
        xdot = self._state_dynamics(time, xc, *user_inputs)
        xdot = jnp.asarray(xdot)

        if self._state_mode == EnabledStateMode.FREE:
            return xdot

        # HOLD and RESET both gate the per-step derivative. RESET layers
        # the additional snap-to-initial behaviour via the zero-crossing
        # event registered in __init__.
        scale = jnp.asarray(enable, dtype=xdot.dtype)
        return xdot * scale

    def _enable_guard(self, time, state, *inputs, **params):
        """Continuous guard for enable transitions.

        Returns ``enable - 0.5`` so any 0↔1 transition crosses zero. Used
        with ``direction="crosses_zero"`` to fire the continuous-state
        reset on either a rising or falling enable edge.
        """
        return jnp.asarray(inputs[0]) - 0.5

    def _reset_continuous_state(self, time, state, *inputs, **params):
        """Snap the continuous state back to its initial value."""
        return state.with_continuous_state(
            jnp.asarray(self._initial_state, dtype=state.continuous_state.dtype)
        )


# ---------------------------------------------------------------------------
# TriggeredSubsystem
# ---------------------------------------------------------------------------


class TriggeredSubsystem(LeafSystem):
    """Container block: latch the submodel output on edge transitions
    (the child still RUNS every step — only the *output* is gated).

    Important: this does **not** skip execution of the submodel on
    non-triggered steps. The submodel is evaluated on every step so its
    inputs participate in the JAX trace; the trigger only controls
    whether a fresh result is *latched* into the held output. If you
    need to actually skip computation between triggers, gate it yourself
    with ``jax.lax.cond`` at the application level.

    Phase-1 implementation runs the submodel on every step (so the
    inputs participate in the trace) but only *latches* a new output
    on an edge transition of the trigger signal. Between transitions
    the output holds the most recently latched value.

    The trigger signal is sampled at ``sample_period``. Edges are
    detected by comparing the current trigger sample against the
    previously-stored sample held in discrete state.

    This is *not* the eventual zero-crossing-driven ``TriggeredSubsystem``
    described in the T-120 architecture notes (that requires hooking
    into the continuous-time event detector); but it is functionally
    correct for any sample-rate use case and matches the behaviour
    documented in the test fixtures.

    Args:
        submodel: Callable ``f(*inputs) -> output`` taking the
            non-trigger user inputs. Must be JAX-traceable.
        n_inputs: Number of user inputs (NOT counting the trigger).
        edge: ``"rising"`` (low→high), ``"falling"`` (high→low) or
            ``"either"``.
        sample_period: Period (seconds) at which the trigger signal is
            sampled and the latch is updated. Must be positive.
        initial_value: Latched output value before any edge has been
            detected. Defines output shape/dtype.
        name: Optional block name.

    Limitations (phase 1):
        - Trigger detection runs on the periodic sample grid, not on
          continuous-time zero crossings. Trigger pulses shorter than
          ``sample_period`` may be missed.
        - The latch is a single discrete state; the submodel must
          produce a single output array.
        - The submodel runs on every output evaluation; only the
          *output* is gated. Users who need to skip computation on
          non-triggered steps should use ``jax.lax.cond`` at the
          application level.
    """

    def __init__(
        self,
        submodel: Callable,
        n_inputs: int = 1,
        edge: Literal["rising", "falling", "either"] = TriggerEdge.RISING,
        sample_period: float = 0.0,
        initial_value=0.0,
        **kwargs,
    ):
        super().__init__(**kwargs)

        if edge not in TriggerEdge.valid():
            raise ValueError(
                f"TriggeredSubsystem: edge must be one of "
                f"{TriggerEdge.valid()!r}, got {edge!r}"
            )
        if sample_period is None or float(sample_period) <= 0.0:
            raise ValueError(
                "TriggeredSubsystem requires a positive sample_period "
                "(seconds) for trigger sampling."
            )
        if n_inputs < 0:
            raise ValueError(
                f"TriggeredSubsystem: n_inputs must be >= 0, got {n_inputs}"
            )

        self._submodel = submodel
        self._edge = edge
        self._sample_period = float(sample_period)
        self._initial = jnp.asarray(initial_value)

        # Port 0 is the trigger signal; ports 1..n_inputs are the user
        # inputs forwarded to the submodel.
        self.declare_input_port(name="trigger")
        for i in range(n_inputs):
            self.declare_input_port(name=f"u_{i}")

        # Discrete state pair: (latched_output, previous_trigger).
        # Pack as a flat 1-D array so the existing scalar-friendly
        # discrete-state machinery handles them uniformly. The two
        # pieces have different shapes in general, so use a tuple-like
        # tree via two separate periodic updates? Simpler: pack as a
        # dict. ``LeafSystem.declare_discrete_state`` only accepts a
        # single default_value, so we encode (latch, prev_trigger) as
        # a flat concatenation when both are scalar. For phase 1 we
        # require a scalar trigger so this packing is safe.
        #
        # Layout: discrete_state[..., 0] holds the previous trigger
        # sample; discrete_state[..., 1:] holds the latched output
        # (flattened). For a scalar latch this collapses to length 2.
        flat_init = jnp.concatenate(
            [
                jnp.zeros((1,), dtype=self._initial.dtype),
                jnp.atleast_1d(self._initial).reshape(-1),
            ]
        )
        self._latch_shape = self._initial.shape
        self._latch_size = int(jnp.atleast_1d(self._initial).reshape(-1).size)
        self.declare_discrete_state(default_value=flat_init)
        self.declare_periodic_update(
            self._latch_update,
            period=self._sample_period,
            offset=0.0,
        )
        self.declare_output_port(
            self._compute_output,
            prerequisites_of_calc=[port.ticket for port in self.input_ports],
        )

    # ── helpers ───────────────────────────────────────────────────────────

    def _unpack(self, ds):
        prev_trig = ds[0]
        latch_flat = ds[1 : 1 + self._latch_size]
        latch = latch_flat.reshape(self._latch_shape)
        return prev_trig, latch

    def _pack(self, prev_trig, latch):
        return jnp.concatenate(
            [
                jnp.atleast_1d(prev_trig).reshape(-1)[:1],
                jnp.atleast_1d(latch).reshape(-1),
            ]
        )

    def _edge_detected(self, prev_trig, cur_trig):
        prev_b = jnp.asarray(prev_trig).astype(bool)
        cur_b = jnp.asarray(cur_trig).astype(bool)
        if self._edge == TriggerEdge.RISING:
            return jnp.logical_and(jnp.logical_not(prev_b), cur_b)
        if self._edge == TriggerEdge.FALLING:
            return jnp.logical_and(prev_b, jnp.logical_not(cur_b))
        # either
        return jnp.not_equal(prev_b, cur_b)

    # ── callbacks ─────────────────────────────────────────────────────────

    def _submodel_output(self, inputs):
        user_inputs = inputs[1:]  # skip trigger
        return jnp.asarray(self._submodel(*user_inputs))

    def _latch_update(self, time, state, *inputs, **params):
        ds = state.discrete_state
        prev_trig, latch = self._unpack(ds)
        cur_trig = jnp.asarray(inputs[0])
        edge = self._edge_detected(prev_trig, cur_trig)
        y_sub = self._submodel_output(inputs)
        new_latch = jnp.where(edge, y_sub, latch)
        # Store the current trigger sample (cast to the same dtype as
        # the rest of the discrete-state vector).
        new_prev = jnp.asarray(cur_trig).astype(ds.dtype).reshape(())
        return self._pack(new_prev, new_latch)

    def _compute_output(self, time, state, *inputs, **params):
        # Output reads the latched value. The submodel is *also* invoked
        # via the latch_update path on the periodic sample grid; here we
        # additionally consult the current trigger so a same-step rising
        # edge surfaces immediately rather than one sample later.
        ds = state.discrete_state
        prev_trig, latch = self._unpack(ds)
        cur_trig = jnp.asarray(inputs[0])
        edge = self._edge_detected(prev_trig, cur_trig)
        y_sub = self._submodel_output(inputs)
        return jnp.where(edge, y_sub, latch)


# ---------------------------------------------------------------------------
# ForEach
# ---------------------------------------------------------------------------


def ForEach(
    submodel: Callable,
    n: int,
    n_inputs: int = 1,
    in_axes=None,
    name: str | None = None,
):
    """Container block: evaluate a submodel ``n`` times in parallel.

    ``ForEach`` is a block-diagram-vocabulary alias for the existing
    :class:`jaxonomy.library.ReplicatedFunction` (T-010). It exists so
    that users familiar with the ``ForEach`` block name can find it
    without paying a duplication tax: the implementation
    is exactly :class:`ReplicatedFunction` under the hood.

    Args:
        submodel: Callable ``f(*inputs) -> output``. Must be
            JAX-traceable.
        n: Number of replicas (the iteration count).
        n_inputs: Number of input ports the block declares.
        in_axes: As in :func:`jax.vmap` / ``ReplicatedFunction``: a
            length-``n_inputs`` tuple of ``0`` (input is batched along
            the leading axis) or ``None`` (input is broadcast). Default
            is all-batched.
        name: Optional block name.

    Returns:
        A configured :class:`ReplicatedFunction` instance, ready to be
        wired into a :class:`DiagramBuilder`.
    """
    # Lazy import: ReplicatedFunction lives in jaxonomy.library, which
    # imports the framework. Importing it eagerly here would create a
    # cycle. The lazy import is exercised only when a user actually
    # constructs a ForEach block.
    from ..library.replicated import ReplicatedFunction

    kwargs: dict = {}
    if name is not None:
        kwargs["name"] = name
    return ReplicatedFunction(
        submodel=submodel,
        n=n,
        n_inputs=n_inputs,
        in_axes=in_axes,
        **kwargs,
    )


# ---------------------------------------------------------------------------
# ForLoop
# ---------------------------------------------------------------------------


class ForLoop(LeafSystem):
    """Container block: run ``body_fn`` ``n_iter`` times per major step.

    ``ForLoop`` wraps :func:`jax.lax.fori_loop`. The block declares a
    single input port carrying the *initial carry value* and a single
    output port returning the carry after ``n_iter`` iterations.

    Args:
        body_fn: Callable ``(i: int, carry) -> carry``. Must be
            JAX-traceable. The carry pytree must have a fixed structure
            and shape across iterations (this is a ``lax.fori_loop``
            requirement, not a Jaxonomy choice).
        n_iter: Number of iterations. Must be a non-negative Python int
            (static); a runtime-traced ``n_iter`` would force
            ``lax.while_loop`` semantics and is not supported here —
            use :class:`WhileLoop` for that case.
        name: Optional block name.

    Differentiability:
        :func:`jax.grad` flows through ``body_fn``'s parameters and
        through the initial carry. The loop count ``n_iter`` is static
        and not differentiable.

    Example:
        A body that accumulates ``i`` into the carry over 10
        iterations yields ``carry_initial + (0+1+...+9) = carry + 45``.

    Notes:
        - ``body_fn`` must close over any constants it needs; the
          ``i``-th iteration receives only ``(i, carry)``.
        - Per T-005, default float dtype is float64 unless the active
          precision policy says otherwise; ``ForLoop`` does not cast.
    """

    def __init__(
        self,
        body_fn: Callable,
        n_iter: int,
        **kwargs,
    ):
        super().__init__(**kwargs)

        if not isinstance(n_iter, int):
            raise TypeError(
                f"ForLoop: n_iter must be a Python int (static), got "
                f"{type(n_iter).__name__}"
            )
        if n_iter < 0:
            raise ValueError(
                f"ForLoop: n_iter must be >= 0, got {n_iter}"
            )

        self._body_fn = body_fn
        self._n_iter = int(n_iter)

        # Single input port: the initial carry value.
        self.declare_input_port(name="carry_init")
        self.declare_output_port(
            self._compute_output,
            prerequisites_of_calc=[port.ticket for port in self.input_ports],
        )

    # ── callbacks ─────────────────────────────────────────────────────────

    def _compute_output(self, time, state, *inputs, **params):
        initial_carry = inputs[0]
        return jax.lax.fori_loop(
            0, self._n_iter, self._body_fn, initial_carry
        )


# ---------------------------------------------------------------------------
# WhileLoop
# ---------------------------------------------------------------------------


class WhileLoop(LeafSystem):
    """Container block: run ``body_fn`` until ``cond_fn`` is False.

    ``WhileLoop`` wraps :func:`jax.lax.while_loop` with a built-in
    iteration counter that caps execution at ``max_iter`` to guarantee
    termination under jit.

    The block declares an input port for the *initial carry value*
    (port 0) plus ``n_inputs`` additional ports for upstream signals
    that the loop body / condition can consume. A single output port
    returns the carry after the loop exits (either because ``cond_fn``
    returned False, or because ``max_iter`` was hit).

    Args:
        body_fn: Callable. Either ``carry -> carry`` (legacy) or
            ``(carry, *inputs) -> carry`` when ``n_inputs > 0``. Must be
            JAX-traceable. The signature is detected via
            :func:`inspect.signature`; functions that accept more than
            one positional argument (or ``*args``) receive *all*
            upstream input values. Callables that only need a subset
            should either accept the rest as throwaway positional
            args, or use ``*args`` and index into it.
        cond_fn: Callable. Either ``carry -> bool`` (legacy) or
            ``(carry, *inputs) -> bool``. Loop continues while this
            returns True. Signature detection mirrors ``body_fn``; the
            same all-inputs-or-none rule applies.
        max_iter: Positive integer cap on iterations. Required to
            keep traces bounded under jit. Defaults to 1000.
        n_inputs: Number of additional upstream input ports (default 0).
            When ``n_inputs > 0`` the block exposes ports
            ``u_0..u_{n_inputs-1}`` after the ``carry_init`` port. The
            current values of these inputs are passed to ``cond_fn`` /
            ``body_fn`` (if they accept them) on every iteration — so
            the condition can compare the carry against a live upstream
            signal (e.g. "iterate until the input exceeds a threshold").
        name: Optional block name.

    Differentiability:
        ``jax.grad`` flows through the carry as long as ``body_fn`` and
        ``cond_fn`` are pure. The number of iterations is data-dependent
        and not differentiable; ``lax.while_loop`` is itself
        non-differentiable in reverse mode (use ``jax.jvp`` for forward
        mode, or refactor with :func:`jax.lax.scan` if you need a
        reverse-mode-friendly bounded loop).

    Notes:
        - On hitting ``max_iter`` the loop exits silently. Users who
          want a runtime warning should ``jax.debug.callback`` from
          ``body_fn`` or test the post-loop carry.
        - The carry pytree structure must be invariant across
          iterations (a ``lax.while_loop`` requirement).
        - The condition is re-evaluated against the *current* upstream
          input values inside the loop trace — the inputs are captured
          once at output-evaluation time and held constant for the
          duration of the loop (the diagram doesn't re-tick during a
          single major step).
    """

    def __init__(
        self,
        body_fn: Callable,
        cond_fn: Callable,
        max_iter: int = 1000,
        n_inputs: int = 0,
        **kwargs,
    ):
        super().__init__(**kwargs)

        if not isinstance(max_iter, int):
            raise TypeError(
                f"WhileLoop: max_iter must be a Python int (static), got "
                f"{type(max_iter).__name__}"
            )
        if max_iter <= 0:
            raise ValueError(
                f"WhileLoop: max_iter must be > 0 to keep traces bounded, "
                f"got {max_iter}"
            )
        if not isinstance(n_inputs, int):
            raise TypeError(
                f"WhileLoop: n_inputs must be a Python int, got "
                f"{type(n_inputs).__name__}"
            )
        if n_inputs < 0:
            raise ValueError(
                f"WhileLoop: n_inputs must be >= 0, got {n_inputs}"
            )

        self._body_fn = body_fn
        self._cond_fn = cond_fn
        self._max_iter = int(max_iter)
        self._n_inputs = int(n_inputs)

        # Signature inspection: detect whether the user-supplied
        # callables want the upstream inputs forwarded. We do this once
        # at construction so the JIT-compiled hot path doesn't pay any
        # introspection cost. The detection is intentionally permissive
        # — anything with more than one positional argument or a
        # ``*args`` gets the extra-args treatment. Legacy single-arg
        # callables are wrapped to ignore the inputs, preserving the
        # T-120-followup-loop-blocks contract byte-for-byte.
        self._body_takes_inputs = _accepts_extra_args(body_fn)
        self._cond_takes_inputs = _accepts_extra_args(cond_fn)

        # Port 0 is always the initial carry; ports 1..n_inputs carry
        # the upstream input signals forwarded to ``cond_fn`` /
        # ``body_fn``.
        self.declare_input_port(name="carry_init")
        for i in range(self._n_inputs):
            self.declare_input_port(name=f"u_{i}")

        self.declare_output_port(
            self._compute_output,
            prerequisites_of_calc=[port.ticket for port in self.input_ports],
        )

    # ── callbacks ─────────────────────────────────────────────────────────

    def _compute_output(self, time, state, *inputs, **params):
        initial_carry = inputs[0]
        user_inputs = inputs[1:]  # the upstream signals (may be empty)
        max_iter = self._max_iter
        body_fn = self._body_fn
        cond_fn = self._cond_fn
        body_takes_inputs = self._body_takes_inputs
        cond_takes_inputs = self._cond_takes_inputs

        def safe_cond(loop_state):
            count, carry = loop_state
            if cond_takes_inputs:
                user_cond = cond_fn(carry, *user_inputs)
            else:
                user_cond = cond_fn(carry)
            user_cond = jnp.asarray(user_cond).astype(bool)
            return jnp.logical_and(count < max_iter, user_cond)

        def safe_body(loop_state):
            count, carry = loop_state
            if body_takes_inputs:
                new_carry = body_fn(carry, *user_inputs)
            else:
                new_carry = body_fn(carry)
            return (count + 1, new_carry)

        _final_count, final_carry = jax.lax.while_loop(
            safe_cond, safe_body, (jnp.asarray(0, dtype=jnp.int32), initial_carry)
        )
        return final_carry


# ---------------------------------------------------------------------------
# ZeroCrossingTriggeredSubsystem (T-120-followup-zc-trigger)
# ---------------------------------------------------------------------------


# Map the user-facing edge vocabulary onto the framework's
# ``declare_zero_crossing(direction=...)`` strings. The framework
# direction names describe the *guard sign transition*, which for a
# trigger interpreted as "fires when the input crosses zero" maps:
#
# - ``rising`` (low → high, i.e. negative → non-negative trigger value)
#   → ``negative_then_non_negative``
# - ``falling`` (high → low, i.e. positive → non-positive)
#   → ``positive_then_non_positive``
# - ``either`` (any zero crossing)
#   → ``crosses_zero``
_EDGE_TO_DIRECTION = {
    TriggerEdge.RISING: "negative_then_non_negative",
    TriggerEdge.FALLING: "positive_then_non_positive",
    TriggerEdge.EITHER: "crosses_zero",
}


class ZeroCrossingTriggeredSubsystem(LeafSystem):
    """Container block: latch the submodel output at zero-crossings.

    Like :class:`TriggeredSubsystem`, but uses the framework's continuous
    zero-crossing detector rather than a periodic sample grid. The
    submodel fires *exactly* when the trigger signal crosses zero in the
    configured direction — this gives sub-sample-period precision for
    the latched event time, which is the property normally expected
    from a triggered subsystem driven by a continuous signal.

    Wiring matches :class:`TriggeredSubsystem`:

    - Input port 0 is the trigger signal (a continuous scalar; the
      block monitors its sign).
    - Input ports 1..n_inputs are the submodel inputs.
    - The single output port returns the most recently latched
      submodel output (initialized to ``initial_value``).

    Args:
        submodel: Callable ``f(*inputs) -> output`` taking the
            non-trigger user inputs. Must be JAX-traceable.
        n_inputs: Number of user inputs (NOT counting the trigger).
        edge: ``"rising"`` (low→high zero crossing of the trigger
            signal), ``"falling"`` (high→low), or ``"either"``.
        initial_value: Latched output value before the first crossing
            fires. Also defines the output shape/dtype.
        name: Optional block name.

    Differentiability:
        ``jax.grad`` flows through the submodel inputs along the path
        through the latch (so when the latched value depends on a
        differentiable input, the gradient propagates). The trigger
        signal itself is consumed by the zero-crossing event detector;
        the gradient through the discontinuity at the firing instant
        is zero by design (the latched value is constant between
        crossings).

    Notes:
        - The framework localizes the zero crossing to within the
          integrator's tolerance, so the latched output reflects the
          submodel inputs *at the crossing instant*, not at the next
          periodic sample. Compare with the phase-1
          :class:`TriggeredSubsystem`, which can only resolve the edge
          to the nearest ``sample_period``.
        - The latch is a single discrete-state component; the submodel
          must produce a single output array of fixed shape.
        - This is a leaf block (no nested mode machinery), so the
          ``"hold between crossings"`` semantics fall out naturally:
          the output port simply returns ``state.discrete_state``.
    """

    def __init__(
        self,
        submodel: Callable,
        n_inputs: int = 1,
        edge: Literal["rising", "falling", "either"] = TriggerEdge.RISING,
        initial_value=0.0,
        **kwargs,
    ):
        super().__init__(**kwargs)

        if edge not in TriggerEdge.valid():
            raise ValueError(
                f"ZeroCrossingTriggeredSubsystem: edge must be one of "
                f"{TriggerEdge.valid()!r}, got {edge!r}"
            )
        if n_inputs < 0:
            raise ValueError(
                f"ZeroCrossingTriggeredSubsystem: n_inputs must be >= 0, "
                f"got {n_inputs}"
            )

        self._submodel = submodel
        self._edge = edge
        self._initial = jnp.asarray(initial_value)

        # Port 0 is the trigger signal; ports 1..n_inputs are the user
        # inputs forwarded to the submodel.
        self.declare_input_port(name="trigger")
        for i in range(n_inputs):
            self.declare_input_port(name=f"u_{i}")

        # Discrete state: the latched submodel output.
        self.declare_discrete_state(default_value=self._initial)

        # Zero-crossing event: guard returns the trigger-signal value;
        # reset_map evaluates the submodel and writes the result to the
        # discrete state.
        self.declare_zero_crossing(
            guard=self._guard,
            reset_map=self._on_trigger,
            direction=_EDGE_TO_DIRECTION[edge],
            name="zc_trigger",
        )

        # Output reads the latched value.
        self.declare_output_port(
            self._compute_output,
            prerequisites_of_calc=[port.ticket for port in self.input_ports],
            default_value=self._initial,
        )

    # ── callbacks ─────────────────────────────────────────────────────────

    def _guard(self, time, state, *inputs, **params):
        # The trigger is input port 0. Cast to the framework's working
        # dtype so the zero-crossing solver sees a plain scalar.
        return jnp.asarray(inputs[0])

    def _on_trigger(self, time, state, *inputs, **params):
        # Run the submodel on the user inputs (everything past the
        # trigger) and latch the result into the discrete state.
        user_inputs = inputs[1:]
        new_value = jnp.asarray(self._submodel(*user_inputs))
        # Preserve dtype/shape of the discrete state; jnp.broadcast_to
        # is safe for the scalar-latch case and also handles the
        # rank-preserving case when initial_value supplied a shape.
        new_value = jnp.asarray(new_value, dtype=state.discrete_state.dtype)
        new_value = jnp.reshape(new_value, state.discrete_state.shape)
        return state.with_discrete_state(new_value)

    def _compute_output(self, time, state, *inputs, **params):
        return state.discrete_state
