# SPDX-License-Identifier: MIT

"""Self-contained systems with no subsystems.

A LeafSystem is a minimal component of a system model in jaxonomy, containing no
subsystems.  If it has inputs or outputs it can be connected to other LeafSystems
as "blocks" to form Diagrams.  If not, it is a self-contained dynamical system.

The LeafSystem class defines the interface for these blocks, specifying various
options for configuring the block.  For example, a LeafSystem can declare that
it has a continuous state and provide an ODE function governing the time evolution
of that state.  It can also declare discrete states, parameters, and update events.
The built-in blocks in jaxonomy.library are all subclasses of LeafSystem, as are
any custom blocks defined by the user.

After declaring states, parameters, ODE, updates, etc., the LeafSystem comprises a set
of pure functions that can be evaluated given a Context, which contains the actual
numeric values for the states, parameters, etc.
"""

from __future__ import annotations
import copy
import dataclasses
from abc import ABCMeta
from functools import partial, wraps
from typing import List, Set, Tuple, Callable, TYPE_CHECKING

import numpy as np

import jax
import jax.numpy as jnp
from jax import tree_util
from jax.typing import ArrayLike

from . import build_recorder

from ..logging import logger

# Import the switchable backend dispatcher as "jaxonomy.numpy" or "npa"
from ..backend import utils, cond, numpy_api as npa, IS_JAXLITE

from .cache import SystemCallback, CallbackTracer
from .state import LeafState
from .port import OutputPort

from .system_base import SystemBase, UpstreamEvalError
from .context_factory import LeafContextFactory
from .dependency_graph import (
    mark_cache,
    LeafDependencyGraphFactory,
    DependencyTicket,
)

from .event import (
    FlatEventCollection,
    LeafEventCollection,
    PeriodicEventData,
    DiscreteUpdateEvent,
    ZeroCrossingEvent,
    ZeroCrossingEventData,
)
from .parameter import Parameter, with_resolved_parameters

if TYPE_CHECKING:
    from ..backend.typing import (
        Array,
        Scalar,
        ShapeLike,
        DTypeLike,
    )
    from .context import ContextBase, LeafContext
    from .state import LeafStateComponent

__all__ = ["LeafSystem"]


# Helper functons used in feedthrough determination
_mark_up_to_date = partial(mark_cache, is_out_of_date=False)
_mark_out_of_date = partial(mark_cache, is_out_of_date=True)


def _wrap_leaf_user_callback(context, owner, user_callback, collect_inputs):
    """Context wrapper for leaf updates/outputs; used via :func:`functools.partial` so
    :func:`copy.deepcopy` can replace ``owner`` (see :meth:`LeafSystem.wrap_callback`).
    """
    if isinstance(collect_inputs, bool):
        port_indices = None if collect_inputs else []
    else:
        port_indices = collect_inputs

    inputs = owner.collect_inputs(context, port_indices)
    leaf_context = context[owner.system_id]
    leaf_state = leaf_context.state
    params = leaf_context.parameters
    try:
        return user_callback(context.time, leaf_state, *inputs, **params)
    except IndexError as exc:
        # T-A4-followup-requires-inputs-infer: a callback that reads
        # ``inputs[i]`` while ``requires_inputs=False`` (so inputs were
        # trimmed to an empty tuple) raises an opaque
        # "tuple index out of range". Re-raise with the actual cause and
        # the fix, instead of leaving the user to decode the bare IndexError.
        n_inputs = len(inputs)
        n_declared = len(owner.input_ports)
        if n_inputs < n_declared and "index out of range" in str(exc):
            raise IndexError(
                f"{exc}. The output/update callback of block {owner.name!r} "
                f"indexed an input, but only {n_inputs} of {n_declared} "
                f"declared input ports were collected — almost always because "
                f"`requires_inputs=False` (or a "
                f"`prerequisites_of_calc=[DependencyTicket.nothing]` that "
                f"resolves to it) was set on a callback that does read inputs. "
                f"Pass `requires_inputs=True` (or list the input indices it "
                f"reads) to collect them."
            ) from exc
        raise


def _array_like(a, b):
    return isinstance(a, ArrayLike) and isinstance(b, ArrayLike)


def _coerce_scalar_for_jax(value):
    """Promote Python scalars / 0-d numpy arrays to ``jnp.asarray`` on the JAX
    backend.

    Why: ``with_parameter`` / ``with_parameters`` previously raised
    ``AttributeError: 'float' object has no attribute 'shape'`` on the canonical
    sweep loop ``for c in cs: diag.with_parameters({"k": float(c)})``.
    Promoting also keeps the abstract pytree leaf type stable across calls, so
    ``jax.jit`` caches the compiled simulator instead of retracing on every new
    parameter value (T-008-followup-with-parameter-trace-cache).
    """
    if isinstance(value, (bool, int, float, complex, np.generic, np.ndarray)):
        return jnp.asarray(value)
    return value


def _check_values_compatible(original_val, new_val):
    """Validate that ``new_val`` can replace ``original_val`` as a parameter /
    default-state value. Returns the (possibly coerced) new value.

    On the JAX backend, Python scalars are auto-promoted via ``jnp.asarray``
    so the shape/dtype check below succeeds and the stored leaf stays an
    array (preventing a jit retrace on each fresh scalar value)."""

    if npa.active_backend != "jax":
        return new_val

    new_val = _coerce_scalar_for_jax(new_val)

    if _array_like(new_val, original_val):
        if new_val.shape != original_val.shape:
            raise ValueError(
                f"Cannot change default value shape from {original_val.shape} to {new_val.shape}"
            )
        if new_val.dtype != original_val.dtype:
            raise ValueError(
                f"Cannot change default value dtype from {original_val.dtype} to {new_val.dtype}"
            )
        return new_val

    from jax.core import Tracer
    if not isinstance(new_val, (type(original_val), Tracer)):
        raise ValueError(
            f"Cannot change default value type from {type(original_val)} to {type(new_val)}"
        )
    return new_val


class InitializeParameterResolver(ABCMeta):
    """Wrapper for the LeafSystem for proper handling of parameters.

    1) wraps initialize() method such that parameters are always resolved when
    the function is called.
    2) automatically call initialize() after __init__.
    """

    def __new__(cls, name, bases, dct):
        if "initialize" in dct:
            orig_initialize = dct["initialize"]
            dct["initialize"] = with_resolved_parameters(orig_initialize)

        if "reset_default_values" in dct:
            orig_reset_default_values = dct["reset_default_values"]
            dct["reset_default_values"] = with_resolved_parameters(
                orig_reset_default_values
            )

        if "__init__" in dct:
            orig_init = dct["__init__"]

            @wraps(orig_init)
            def new_init(self, *args, **kwargs):
                orig_init(self, *args, **kwargs)
                build_recorder.create_block(self, orig_init, *args, **kwargs)

            dct["__init__"] = new_init

        return super().__new__(cls, name, bases, dct)


class LeafSystem(SystemBase, metaclass=InitializeParameterResolver):
    """Basic building block for dynamical systems.

    A LeafSystem is a minimal component of a system model in jaxonomy, containing no
    subsystems.  Inputs, outputs, state, parameters, updates, etc. can be added to the
    block using the various `declare_*` methods.  The built-in blocks in
    jaxonomy.library are all subclasses of LeafSystem, as are any custom blocks defined
    by the user."""

    # SystemBase is a dataclass, so we need to call __post_init__ explicitly
    def __post_init__(self):
        super().__post_init__()
        logger.debug(f"Initializing {self.name} [{self.system_id}]")

        # If not None, this defines the shape and data type of the continuous state
        # component.  This value will be used to initialize the context, so it will
        # also serve as the initial value unless explicitly overridden. It will
        # typically be an array, but it can be any PyTree-structured object (list,
        # dict, namedtuple, etc.), provided that the ODE function returns a PyTree
        # of the same structure.
        self._default_continuous_state: LeafStateComponent = None
        self._mass_matrix: Array = None
        self._continuous_state_output_port_idx: int = None

        # The SystemCallback associated with time derivatives of the continuous state.
        # This is initialized in the `declare_continuous_state` method.
        self.ode_callback: SystemCallback = None

        # If not empty, this defines the shape and data type of the discrete state.
        # This value will be used to initialize the context, so it will also serve
        # as the initial value unless explicitly overridden. This will often be an
        # array, but as for the continuous state it can be any PyTree-structured
        # object (list, dict, namedtuple, etc.), provided that the update functions
        # return a PyTree of the same structure.
        self._default_discrete_state: LeafStateComponent = None

        # If not None, the system has a "mode" or "stage" component of the state.
        # In a "state machine" paradigm, this represents the current state of the
        # system (although "state" is obviously used for other things in this case).
        # The mode is an integer value, and the system can declare transitions between
        # modes using the `declare_zero_crossing` method, which in addition to the
        # guard function and reset map also takes optional `start_mode` and `end_mode`
        # arguments.
        self._default_mode: int = None
        self._mode_output_port_idx: int = None

        # Set of "template" values for the sample-and-hold output ports, if known.
        # If not known, these will be `None`, in which case an appropriate value is
        # inferred from upstream during static analysis.
        self._default_cache: List[LeafStateComponent] = []

        # Transition map from (start_mode -> [*end_modes]) indicating which
        # transition events are active in each mode.  This is not used by
        # any logic in the system, but can be useful for debugging.
        self.transition_map: dict[int, List[Tuple[int, ZeroCrossingEvent]]] = {}

        # Set of events that updates at a fixed rate.  Each event has its own period
        # and offset, so "fires" independently of the other events. These can be
        # created using the `declare_periodic_update` method.
        self._state_update_events: List[DiscreteUpdateEvent] = []

        # Set of events that update when a zero-crossing occurs.  Each event has its
        # own guard function and, optionally, reset map, start mode, and end mode.
        # These can be created using the `declare_zero_crossing` method.
        self._zero_crossing_events: List[ZeroCrossingEvent] = []

        # T-115-followup-saturate-rate-classification: count only ZC events
        # that have *behavioral* side effects (a user-supplied ``reset_map``
        # or a mode transition via ``start_mode`` / ``end_mode``). Pure
        # solver-hint ZC events (e.g. :class:`Saturate` declaring clip
        # boundaries so the ODE integrator can localise the discontinuity)
        # should not flip the block's rate-group classification to
        # ``event_driven`` — they exist only to help the solver, not to
        # represent an asynchronous trigger. The rate-groups inference uses
        # this counter to distinguish the two cases.
        self._n_behavioral_zc_events: int = 0

        # T-027: per-event Zeno-hold tracking for `declare_zero_crossing(zeno_tolerance=...)`.
        # Each entry is a dict {slot, tol, name} where `slot` is the index into the
        # private Zeno discrete state. The discrete state is a NamedTuple with fields
        # `zeno` (bool array, one slot per protected event) and `tprev` (float array).
        # See `_install_zeno_protection` for details. Empty by default; the existing
        # `Integrator` Zeno path uses its own private discrete state and is unaffected.
        #
        # T-027a: when the user also calls `declare_discrete_state(...)`, the framework
        # packs the Zeno tracker alongside the user's value as `_DiscreteWithZeno(
        # user=..., zeno=_ZenoState(zeno=..., tprev=...))`. User callbacks (ode, guard,
        # reset) are wrapped to see only their own `user` slot via `state.discrete_state`,
        # and the framework re-packs on the way out. When the user has no discrete
        # state, the bare `_ZenoState` is stored as before (no wrapper, no overhead).
        self._zeno_protected_events: List[dict] = []
        self._zeno_state_type = None  # Set on first protected event.
        self._zeno_combined_type = None  # Set when user has discrete state too.
        self._zeno_user_default = None  # User's declared default (if any) at install time.
        self._zeno_ode_wrapped = False

    def initialize(self, **parameters):
        """Hook for initializing a system. Called during context creation.

        If the parameters are instances of Parameter, they will be resolved.
        If implemented, the function signature should contain all the declared
        parameters.

        This function should not be called directly. It will be called implicitly
        after __init__ with the resolved parameters.
        """
        pass

    @property
    def has_feedthrough_side_effects(self) -> bool:
        # See explanation in `SystemBase.has_feedthrough_side_effects`.  This will
        # almost always be False, but can be overridden in special cases where a
        # feedthrough output is computed via use of `io_callback`.
        return False

    @property
    def has_ode_side_effects(self) -> bool:
        # This will almost always be False for a LeafSystem - Diagram systems
        # have some special logic to do this determination.
        return False

    @property
    def has_continuous_state(self) -> bool:
        return self._default_continuous_state is not None

    @property
    def continuous_state_default(self) -> "LeafStateComponent":
        """The declared default continuous-state value (read-only).

        This is the ``default_value`` passed to ``declare_continuous_state``
        (or the array inferred from ``shape`` / ``dtype``), i.e. the value
        that seeds ``context.continuous_state`` before any user override.
        Returns ``None`` when the block has no continuous state. Exposed as a
        documented accessor so callers don't have to reach into the private
        ``_default_continuous_state`` attribute (T-C2-followup).
        """
        return self._default_continuous_state

    @property
    def has_discrete_state(self) -> bool:
        return self._default_discrete_state is not None

    @property
    def has_zero_crossing_events(self) -> bool:
        return len(self._zero_crossing_events) > 0

    #
    # Event handling
    #
    def wrap_callback(
        self, callback: Callable, collect_inputs: bool | list[int] = True
    ) -> Callable:
        """Wrap an update function to unpack local variables and block inputs.

        The callback should have the signature
        `callback(time, state, *inputs, **params) -> result`
        and will be wrapped to have the signature `callback(context) -> result`,
        as expected by the event handling logic.

        This is used internally for declaration methods like
        `declare_periodic_update` so that users can write more intuitive
        block-level update functions without worrying about the "context", and have
        them automatically wrapped to have the right interface.  It can also be
        called directly by users to wrap their own update functions, for example to
        create a callback function for `declare_output_port`.

        The context and state are strictly immutable, so the callback should not
        attempt to change any values in the context or state.  Even in cases where
        it is impossible to _enforce_ this (e.g. a state component is a list, which
        is always mutable in Python), the callback should be careful to avoid direct
        modification of the context or state, which may lead to unexpected behavior
        or JAX tracer errors.

        Args:
            callback (Callable):
                The (pure) function to be wrapped. See above for expected signature.
            collect_inputs (bool):
                If True, the callback will eval input ports to gather input values.
                Normally this should be True, but it can be set to False if the
                return value depends only on the state but not inputs, for
                instance. This helps reduce the number of expressions that need to
                be JIT compiled. Can also be specified as a list of integer port indices.
                Default is True (collect all inputs).

        Returns:
            Callable:
                The wrapped function, with signature `callback(context) -> result`.
        """
        return partial(
            _wrap_leaf_user_callback,
            owner=self,
            user_callback=callback,
            collect_inputs=collect_inputs,
        )

    def _passthrough(self, context: ContextBase) -> LeafState:
        """Dummy callback for inactive events."""
        return context[self.system_id].state

    @property
    def state_update_events(self) -> FlatEventCollection:
        return FlatEventCollection(tuple(self._state_update_events))

    @property
    def zero_crossing_events(self) -> LeafEventCollection:
        # The default is for all to be active. Use the `determine_active_guards`
        # method to determine which are active conditioned on the current "mode"
        # or "stage" of the system.
        return LeafEventCollection(tuple(self._zero_crossing_events)).mark_all_active()

    def with_parameter(self, name: str, value) -> LeafSystem:
        """Return a copy of this system with one dynamic parameter replaced.

        The returned system is a new instance. The original is unchanged.

        Args:
            name: Parameter name (must exist as a dynamic parameter).
            value: New value (typically a JAX array for ``jax.grad`` / ``jax.vmap``).

        Raises:
            KeyError: If ``name`` is not a dynamic parameter.
            TypeError: If ``name`` is a static parameter.
        """
        if name in self._static_parameters:
            raise TypeError(
                f"Parameter {name!r} is static on {self.name!r}; static parameters "
                "cannot be replaced at runtime without recompilation."
            )
        if name not in self._dynamic_parameters:
            available = sorted(
                {*self._static_parameters.keys(), *self._dynamic_parameters.keys()}
            )
            raise KeyError(
                f"Parameter {name!r} is not a dynamic parameter on {self.name!r}. "
                f"Available: {available}"
            )

        old_param = self._dynamic_parameters[name]
        old_val = Parameter.unwrap(old_param)
        try:
            value = _check_values_compatible(old_val, value)
        except ValueError as e:
            raise ValueError(f"{e} (parameter {name!r} on {self.name!r})") from None

        new = copy.deepcopy(self)
        new.parent = None
        new._dependency_graph = None
        new.feedthrough_pairs = None
        new._cache_update_events = None
        new._cached_input_ports.clear()
        new._cached_output_ports.clear()

        if isinstance(old_param, Parameter):
            new._dynamic_parameters[name] = dataclasses.replace(
                old_param,
                value=value,
                name=name,
                system=new,
            )
        else:
            new._dynamic_parameters[name] = Parameter(
                value=value,
                name=name,
                system=new,
            )

        return new

    # Inherits docstring from SystemBase
    def eval_zero_crossing_updates(
        self,
        context: ContextBase,
        events: LeafEventCollection,
    ) -> LeafState:
        local_events = events[self.system_id]
        state = context[self.system_id].state

        logger.debug(f"Eval update events for {self.name}")
        logger.debug(f"local events: {local_events}")

        for event in local_events:
            # This is evaluated conditionally on event_data.active
            state = event.handle(context)

            # Store the updated state in the context for this block
            leaf_context = context[self.system_id].with_state(state)

            # Update the context for this block in the overall context
            context = context.with_subcontext(self.system_id, leaf_context)

        # Now `context` contains the updated "plus" state for this block, but
        # this needs to be discarded so that other block updates can also be
        # processed using the "minus" state. This is done by simply returning the
        # "plus" state and discarding the rest of the updated context.
        return state

    # Inherits docstring from SystemBase
    def determine_active_guards(self, root_context: ContextBase) -> LeafEventCollection:
        mode = root_context[self.system_id].mode  # Current system mode

        def _conditionally_activate(
            event: ZeroCrossingEvent,
        ) -> ZeroCrossingEvent:
            # Check to see if the event corresponds to a mode transition
            # If not, just return the event unchanged (will be active)
            if event.active_mode is None:
                return event
            # If the event does correspond to a mode transition, check to see
            # if the event is active in the current mode
            return cond(
                mode == event.active_mode,
                lambda e: e.mark_active(),
                lambda e: e.mark_inactive(),
                event,
            )

        # Apply the conditional activation to all events
        zero_crossing_events = LeafEventCollection(
            tuple(_conditionally_activate(e) for e in self.zero_crossing_events)
        )

        logger.debug(f"Zero-crossing events for {self.name}: {zero_crossing_events}")
        return zero_crossing_events

    @property
    def _flat_callbacks(self) -> List[OutputPort]:
        """Return all of the sample-and-hold output ports in this system."""
        return self.callbacks

    def declare_cache(
        self,
        callback: Callable,
        period: float | Parameter = None,
        offset: float | Parameter = 0.0,
        name: str = None,
        prerequisites_of_calc: List[DependencyTicket] = None,
        default_value: Array = None,
        requires_inputs: bool = True,
    ) -> int:
        """Declare a stored computation for the system.

        This method accepts a callback function with the block-level signature
            `callback(time, state, *inputs, **parameters) -> value`
        and wraps it to have the signature
            `callback(context) -> value`

        This callback can optionally be used to define a periodic update event that
        refreshes the cached value.  Other calculations (e.g. sample-and-hold output
        ports) can then depend on the cached value.

        Args:
            callback (Callable):
                The callback function defining the cached computation.
            period (float, optional):
                If not None, the callback function will be used to define a periodic
                update event that refreshes the value. Defaults to None.
            offset (float, optional):
                The offset of the periodic update event. Defaults to 0.0.  Will be ignored
                unless `period` is not None.
            name (str, optional):
                The name of the cached value. Defaults to None.
            default_value (Array, optional):
                The default value of the result, if known. Defaults to None.
            requires_inputs (bool, optional):
                If True, the callback will eval input ports to gather input values.
                This will add a bit to compile time, so setting to False where possible
                is recommended. Defaults to True.
            prerequisites_of_calc (List[DependencyTicket], optional):
                The dependency tickets for the computation. Defaults to None, in which
                case the default is to assume dependency on either (inputs) if
                `requires_inputs` is True, or (nothing) otherwise.

        Returns:
            int: The index of the callback in `system.callbacks`.  The cache index can
                recovered from `system.callbacks[callback_index].cache_index`.
        """
        # The index in the list of system callbacks
        callback_index = len(self.callbacks)

        # This is the index that this cached value will have in state.cache
        cache_index = len(self._default_cache)
        self._default_cache.append(default_value)

        # To help avoid unnecessary flagging of algebraic loops, trim the inputs as a
        # default prereq if the update callback doesn't use them
        if prerequisites_of_calc is None:
            if requires_inputs:
                prerequisites_of_calc = [DependencyTicket.u]
            else:
                prerequisites_of_calc = [DependencyTicket.nothing]

        def _update_callback(
            time: Scalar, state: LeafState, *inputs, **parameters
        ) -> LeafState:
            output = callback(time, state, *inputs, **parameters)
            return state.with_cached_value(cache_index, output)

        _update_callback = self.wrap_callback(
            _update_callback, collect_inputs=requires_inputs
        )

        if period is None:
            event = None

        else:
            # The cache has a periodic event updating its value defined by the callback
            event = DiscreteUpdateEvent(
                system_id=self.system_id,
                event_data=PeriodicEventData(
                    period=period, offset=offset, active=False
                ),
                name=f"{self.name}:cache_update_{cache_index}_",
                callback=_update_callback,
                passthrough=self._passthrough,
            )

        if name is None:
            name = f"cache_{cache_index}"

        sys_callback = SystemCallback(
            callback=_update_callback,
            system=self,
            callback_index=callback_index,
            name=name,
            prerequisites_of_calc=prerequisites_of_calc,
            event=event,
            default_value=default_value,
            cache_index=cache_index,
        )
        self.callbacks.append(sys_callback)

        return callback_index

    # NOTE: we can only declare one continuous state per system because each
    # call will overwrite self._default_continuous_state
    def declare_continuous_state(
        self,
        shape: ShapeLike = None,
        default_value: Array = None,
        dtype: DTypeLike = None,
        ode: Callable = None,
        mass_matrix: Array = None,
        as_array: bool = True,
        requires_inputs: bool = True,
        prerequisites_of_calc: List[DependencyTicket] = None,
        substeps: int = 1,
        project: Callable = None,
    ):
        """Declare a continuous state component for the system.

        The continuous state value is read inside callbacks as
        ``state.continuous_state`` (the ``state`` argument of the ``ode`` /
        output callbacks). **Unpack contract** (T-C3-followup): the shape of
        ``state.continuous_state`` mirrors exactly what you passed as
        ``default_value`` (or the zeros array implied by ``shape`` /
        ``dtype``):

        - A scalar default (``jnp.array(0.0)``) gives a scalar
          ``state.continuous_state`` — read it directly, do **not** index.
        - A vector default (``jnp.zeros(3)``) gives a length-3 array — index
          / unpack as ``x, y, z = state.continuous_state`` or
          ``state.continuous_state[i]``.
        - A PyTree default (tuple / NamedTuple / dict) gives back the same
          PyTree structure; your ``ode`` must return ``xcdot`` with the
          identical structure.

        The ``ode`` callback's return value must match the
        ``default_value`` structure element-for-element, since it is added to
        the state during integration. A common error is declaring a scalar
        state but returning ``jnp.array([xdot])`` (shape ``(1,)``) from the
        ode — keep both scalar or both vector.

        Multirate substepping (T-133): ``substeps=N`` declares that this
        block's continuous dynamics have a fast time constant needing ``N``
        inner integration steps per outer solver step (e.g. a motor's
        electrical winding inside a 1 kHz control loop). Honored by the
        fixed-step ``rk4`` solver (``SimulatorOptions(ode_solver_method=
        "rk4")``): the block's states advance with ``N`` RK4 substeps of
        ``h/N`` while the rest of the diagram takes one step of ``h``,
        with first-order (zero-order-hold) coupling at the boundary —
        each side sees the other's start-of-step values, matching the
        semantics of a hand-rolled JIT-safe substep loop. Adaptive solvers
        (``dopri5``/``bdf``) ignore the declaration — they control
        stiffness through global step adaptation. ``N`` must be a static
        Python ``int >= 1``; the default 1 is byte-equivalent to the
        pre-T-133 behavior.

        Reverse-mode autodiff (``enable_autodiff=True``) is supported —
        the substep loop has a static trip count and the checkpointed
        adjoint substeps the costates alongside their primals. Gradient
        accuracy carries the scheme's first-order coupling error: the
        adjoint converges to the true sensitivity linearly in the outer
        step ``h`` (exact FD agreement is only recovered as ``h`` is
        refined), and for dynamics *unstable at the outer step* the
        adjoint's reverse-time primal re-integration further limits
        accuracy. Reduce the outer step when gradients through the
        coupling interface need to be tight.

        Declared state projection (T-132): ``project=fn`` declares that
        this block's continuous state lives on a manifold and supplies
        the retraction back onto it — e.g. unit-quaternion
        renormalization for an attitude state (``nq=4`` integrated
        componentwise drifts off the unit sphere under any one-step
        integrator). ``fn(x) -> x`` receives the state in its declared
        structure, must be shape-preserving and jit-safe, and is applied
        by the simulator **at the end of every major step** (composing
        with, and independent of, the T-003a DAE projection). Within-step
        drift is bounded by the step size; the recorded trajectory and
        all values other blocks see at major-step boundaries are on the
        manifold. Differentiable: the projection participates in
        reverse-mode AD as ordinary traced ops.
        """
        if not isinstance(substeps, (int, np.integer)) or isinstance(
            substeps, bool
        ) or substeps < 1:
            raise ValueError(
                f"declare_continuous_state: substeps must be a static Python "
                f"int >= 1, got {substeps!r}. (It sets a compile-time inner "
                "loop count and cannot be traced or fractional.)"
            )
        self._continuous_substeps = int(substeps)
        if project is not None and not callable(project):
            raise ValueError(
                f"declare_continuous_state: project must be a callable "
                f"x -> x (shape-preserving, jit-safe), got {project!r}."
            )
        self._continuous_projection = project

        self.ode_callback = SystemCallback(
            callback=None,
            system=self,
            callback_index=len(self.callbacks),
            name=f"{self.name}_ode",
            prerequisites_of_calc=prerequisites_of_calc,
        )
        self.callbacks.append(self.ode_callback)
        callback_idx = len(self.callbacks) - 1

        # FIXME: this is to preserve some backward compatibility while we decouple
        # declaration from configuration. Declaration should not have to call
        # configuration.
        if default_value is not None or shape is not None:
            self.configure_continuous_state(
                callback_idx,
                shape=shape,
                default_value=default_value,
                dtype=dtype,
                ode=ode,
                mass_matrix=mass_matrix,
                as_array=as_array,
                requires_inputs=requires_inputs,
                prerequisites_of_calc=prerequisites_of_calc,
            )

        return callback_idx

    def configure_continuous_state(
        self,
        callback_idx: int,
        shape: ShapeLike = None,
        default_value: Array = None,
        dtype: DTypeLike = None,
        ode: Callable = None,
        mass_matrix: Array = None,
        as_array: bool = True,
        requires_inputs: bool = True,
        prerequisites_of_calc: List[DependencyTicket] = None,
    ):
        """Configure a continuous state component for the system.

        The `ode` callback computes the time derivative of the continuous state based on the
        current time, state, and any additional inputs. If `ode` is not provided, a default
        zero vector of the same size as the continuous state is used. If provided, the `ode`
        callback should have the signature `ode(time, state, *inputs, **params) -> xcdot`.

        Args:
            callback_idx (int):
                The index of the callback in the system's callback list.
            shape (ShapeLike, optional):
                The shape of the continuous state vector. Defaults to None.
            default_value (Array, optional):
                The initial value of the continuous state vector. Defaults to None.
            dtype (DTypeLike, optional):
                The data type of the continuous state vector. Defaults to None.
            ode (Callable, optional):
                The callback for computing the time derivative of the continuous state.
                Should have the signature:
                    `ode(time, state, *inputs, **parameters) -> xcdot`.
                Defaults to None.
            mass_matrix (Array, optional):
                The mass matrix for the continuous state. Defaults to None. If
                provided, must be a square matrix with the same shape as the
                continuous state.  Using a mass matrix different from the identity
                in any LeafSystem will require the use of a compatible continuous-time
                solver (currently only BDF is supported).  Currently mass matrices are
                also only supported for scalar- or vector-valued continuous states (
                i.e. no matrices or other PyTree-structured states).
            as_array (bool, optional):
                If True, treat the default_value as an array-like (cast if necessary).
                Otherwise, it will be stored as the default state without modification.
            requires_inputs (bool, optional):
                If True, indicates that the ODE computation requires inputs.
            prerequisites_of_calc (List[DependencyTicket], optional):
                The dependency tickets for the ODE computation. Defaults to None, in
                which case the assumption is a dependency on either (time, continuous
                state) if `requires_inputs` is False, otherwise (time, continuous state,
                inputs.

        Raises:
            AssertionError:
                If neither shape nor default_value is provided, or if the mass matrix
                is inconsistent with the continuous state.

        Notes:
            (1) Only one of `shape` and `default_value` should be provided. If `default_value`
            is provided, it will be used as the initial value of the continuous state. If
            `shape` is provided, the initial value will be a zero vector of the given shape
            and specified dtype.
        """

        if prerequisites_of_calc is None:
            prerequisites_of_calc = [DependencyTicket.time, DependencyTicket.xc]
            if requires_inputs:
                prerequisites_of_calc.append(DependencyTicket.u)

        if as_array:
            default_value = utils.make_array(default_value, dtype=dtype, shape=shape)

        logger.debug(f"In block {self.name} [{self.system_id}]: {default_value=}")

        # Tree-map the default value to ensure that it is an array-like with the
        # correct shape and dtype. This is necessary because the default value
        # may be a list, tuple, or other PyTree-structured object.
        default_value = tree_util.tree_map(npa.asarray, default_value)

        self._default_continuous_state = default_value
        if self._continuous_state_output_port_idx is not None:
            port = self.output_ports[self._continuous_state_output_port_idx]
            port.default_value = default_value
            self._default_cache[port.cache_index] = default_value

        if ode is None:
            # If no ODE is specified, return a zero vector of the same size as the
            # continuous state. This will break if the continuous state is
            # a named tuple, in which case a custom ODE must be provided.
            assert as_array, "Must provide custom ODE for non-array continuous state"

            def ode(time, state, *inputs, **parameters):
                return npa.zeros_like(default_value)

        # Wrap the ode function to accept a context and return the time derivatives.
        ode = self.wrap_callback(ode)

        # Declare the time derivative function as a system callback so that its
        # dependencies can be tracked in the system dependency graph
        self.ode_callback._callback = ode
        self.ode_callback.prerequisites_of_calc = prerequisites_of_calc

        # Override the default `eval_time_derivatives` to use the wrapped ODE function
        self.eval_time_derivatives = self.ode_callback.eval

        # T-027: if Zeno-protected events were registered before this, wrap the
        # ode so a Zeno-hold freezes the continuous state.
        if self._zeno_protected_events:
            self._zeno_ode_wrapped = False
            self._wrap_ode_for_zeno()

        if mass_matrix is not None:
            # Check that the state is a vector or scalar
            assert as_array, "Mass matrix only supported for array-valued states"
            assert (
                len(default_value.shape) <= 1
            ), "Mass matrix only supported for scalar or vector continuous states"
            n = default_value.size
            assert mass_matrix.shape in ((n, n), (n,)), (
                "Mass matrix must be either a square matrix or vector of the same "
                f"size as the continuous state, but got {mass_matrix.shape} for "
                f"continuous state of shape {default_value.shape}."
            )
            if len(mass_matrix.shape) == 1:
                mass_matrix = np.diag(mass_matrix)
            else:
                mass_matrix = np.asarray(mass_matrix)

            # If we end up with an identity matrix, we can just ignore the mass
            # matrix and use the default mass matrix (which is None).  This will
            # allow us to continue using explicit ODE solvers.
            nontrivial_mass_matrix = not np.allclose(mass_matrix, np.eye(n))
            if not nontrivial_mass_matrix:
                mass_matrix = None

        self._mass_matrix = mass_matrix

    @property
    def mass_matrix(self) -> Array:
        # When this is called, an array return value is expected, so we can safely
        # return the mass matrix as an array, even if the internal value is None.
        if self._default_continuous_state is None:
            return None

        if self._mass_matrix is not None:
            return self._mass_matrix

        # Currently only scalar- or vector-valued continuous states are supported,
        # so check that the continuous state (or all tree leaves if tree-structured)
        # is a scalar or vector, and return corresponding identity matrices.
        xc_leaves = tree_util.tree_leaves(self._default_continuous_state)
        if not all(len(xc.shape) <= 1 for xc in xc_leaves):
            raise ValueError(
                "Mass matrix DAEs are only supported when the continuous state is "
                f"scalar- or vector-valued.  System {self.name} has non-vector "
                "continuous state with default value "
                f"{self._default_continuous_state}."
            )

        # Now we are guaranteed that the continuous state is a scalar or vector, so
        # we can return the corresponding (tree-structured) identity matrix.
        return jax.tree.map(lambda x: np.eye(x.size), self._default_continuous_state)

    @property
    def has_mass_matrix(self) -> bool:
        # Does the system have a nontrivial mass matrix?  This will return
        # False if the mass matrix is None or the identity matrix, since
        # the internal _mass_matrix attribute is set to None during
        # continuous state creation in the case where the mass matrix is
        # the identity.
        return self._mass_matrix is not None

    @property
    def continuous_substep_vector(self):
        """T-133: per-entry multirate substep factors for this block.

        Returns pytree-structured int vectors aligned with the flattened
        continuous state (same leaves-concatenation ordering the ODE
        solvers use for ``mass_matrix``), or ``None`` when the block has
        no continuous state. Every entry carries the block-level factor
        declared via ``declare_continuous_state(substeps=N)`` (default 1).
        """
        if self._default_continuous_state is None:
            return None
        factor = int(getattr(self, "_continuous_substeps", 1))
        return jax.tree.map(
            lambda x: np.full((np.asarray(x).size,), factor, dtype=np.int32),
            self._default_continuous_state,
        )

    @property
    def has_multirate_substeps(self) -> bool:
        """True when this block declared ``substeps > 1`` (T-133)."""
        return (
            self._default_continuous_state is not None
            and int(getattr(self, "_continuous_substeps", 1)) > 1
        )

    def declare_discrete_state(
        self,
        shape: ShapeLike = None,
        default_value: Array | Parameter = None,
        dtype: DTypeLike = None,
        as_array: bool = True,
        name: str = None,
    ):
        """Declare a discrete state component for the system.

        The discrete state is a component of the system's state that can be updated
        at specific events, such as zero-crossings or periodic updates.

        .. note::
            Currently only **one** discrete state component is supported per
            ``LeafSystem``.  If ``declare_discrete_state`` is called more than once,
            the second call will silently overwrite the first.  To store several
            independent values, pack them into a single array and split inside your
            update callback.

        Args:
            shape (ShapeLike, optional):
                The shape of the discrete state. Defaults to None.
            default_value (Array, optional):
                The initial value of the discrete state. Defaults to None.
            dtype (DTypeLike, optional):
                The data type of the discrete state. Defaults to None.
            as_array (bool, optional):
                If True, treat the default_value as an array-like (cast if necessary).
                Otherwise, it will be stored as the default state without modification.
            name (str, optional):
                Readability label for the discrete state (parity with
                ``declare_continuous_state_output(name=...)``). Stored as
                ``self.discrete_state_name`` for diagnostics/debugging; it
                does not change runtime behaviour, and the state is still
                read as ``state.discrete_state``.

        Raises:
            AssertionError:
                If as_array is True and neither shape nor default_value is provided.

        Notes:
            (1) Only one of `shape` and `default_value` should be provided. If
            `default_value` is provided, it will be used as the initial value of the
            continuous state. If `shape` is provided, the initial value will be a
            zero vector of the given shape and specified dtype.

            (2) Use `declare_periodic_update` to declare an update event that
            modifies the discrete state at a recurring interval.
        """
        self.discrete_state_name = name
        if as_array:
            default_value = utils.make_array(default_value, dtype=dtype, shape=shape)

        # Tree-map the default value to ensure that it is an array-like with the
        # correct shape and dtype. This is necessary because the default value
        # may be a list, tuple, or other PyTree-structured object.
        default_value = tree_util.tree_map(npa.asarray, default_value)

        # T-027a: if Zeno protection is already installed, pack the user's
        # value alongside the existing Zeno tracker rather than overwriting it.
        if self._zeno_protected_events:
            self._zeno_user_default = default_value
            current = self._default_discrete_state
            zeno_xd = (
                current.zeno
                if isinstance(current, self._zeno_combined_type)
                else current
            )
            self._default_discrete_state = self._zeno_combined_type(
                user=default_value, zeno=zeno_xd
            )
        else:
            self._default_discrete_state = default_value

    def configure_discrete_state_default_value(
        self, default_value: Array, as_array: bool = True
    ):
        if as_array:
            dtype = self._default_discrete_state.dtype
            shape = self._default_discrete_state.shape
            default_value = utils.make_array(default_value, dtype=dtype, shape=shape)

        # Tree-map the default value to ensure that it is an array-like with the
        # correct shape and dtype. This is necessary because the default value
        # may be a list, tuple, or other PyTree-structured object.
        default_value = tree_util.tree_map(npa.asarray, default_value)

        _check_values_compatible(self._default_discrete_state, default_value)

        self._default_discrete_state = default_value

    #
    # I/O declaration
    #
    def _resolve_requires_inputs(
        self,
        requires_inputs: bool | list[int] | None,
        prerequisites_of_calc: List[DependencyTicket] | None,
    ) -> bool | list[int]:
        """Resolve the ``requires_inputs`` flag for an output port.

        Inference is deliberately conservative (T-A4-followup-requires-inputs-infer):
        we only infer ``requires_inputs=False`` when the caller explicitly
        declared ``prerequisites_of_calc=[DependencyTicket.nothing]`` — an
        unambiguous "this output depends on nothing" signal. Every other
        unset case keeps the legacy default of ``True`` (collect all inputs),
        because the established convention is that ``prerequisites_of_calc``
        may list *upstream / transitive* tickets (e.g. ``xcdot`` for a
        derivative output whose callback still reads ``u``, or ``xd`` for a
        sample-and-hold port whose *update* event reads ``u``) while
        ``requires_inputs`` independently controls input collection. Auto-
        flipping to ``False`` from a non-input prereq list would silently
        starve those callbacks of their inputs.
        """
        if requires_inputs is not None:
            return requires_inputs
        if prerequisites_of_calc is not None:
            # The only unambiguous "no inputs" declaration.
            if list(prerequisites_of_calc) == [DependencyTicket.nothing]:
                return False
        # Legacy default: collect all inputs.
        return True

    def declare_output_port(
        self,
        callback: Callable = None,
        period: float = None,
        offset: float = 0.0,
        name: str = None,
        prerequisites_of_calc: List[DependencyTicket] = None,
        default_value: Array = None,
        requires_inputs: bool | list[int] | None = None,
        units=None,
    ) -> int:
        """Declare an output port in the LeafSystem.

        This method accepts a callback function with the block-level signature
            `callback(time, state, *inputs, **parameters) -> value`
        and wraps it to the signature expected by SystemBase.declare_output_port:
            `callback(context) -> value`

        Args:
            callback (Callable):
                The callback function defining the output port.
            period (float, optional):
                If not None, the port will act as a "sample-and-hold", with the
                callback function used to define a periodic update event that refreshes
                the value that will be returned by the port. Typically this should
                match the update period of some associated update event in the system.
                Defaults to None.
            offset (float, optional):
                The offset of the periodic update event. Defaults to 0.0.  Will be ignored
                unless `period` is not None.
            name (str, optional):
                The name of the output port. Defaults to None.
            default_value (Array, optional):
                The default value of the output port, if known. Defaults to None.
            requires_inputs (bool | list[int] | None, optional):
                Whether the callback reads input port values.

                **Defaults to ``None``.** ``None`` resolves to ``True``
                (collect all inputs) in every case except the unambiguous
                ``prerequisites_of_calc=[DependencyTicket.nothing]``
                declaration, which resolves to ``False``
                (T-A4-followup-requires-inputs-infer). The inference is
                deliberately conservative: ``prerequisites_of_calc`` may list
                *upstream / transitive* tickets (e.g. ``xcdot`` for a
                derivative output whose callback still reads ``u``, or ``xd``
                for a sample-and-hold port whose *update* event reads ``u``),
                so a non-input prereq list does **not** imply the callback is
                input-free — only ``[nothing]`` does.

                **Set this to ``False`` explicitly whenever the output does NOT
                depend on any input port** (e.g. a ZOH output that returns a
                stored discrete state, or a CT output that only reads continuous
                state).  This serves two purposes:
                  1. **Eliminates false-positive algebraic-loop detection.**  The
                     diagram-level algebraic-loop checker conservatively assumes every
                     output with ``requires_inputs=True`` has direct feedthrough from
                     all connected inputs.  Declaring ``requires_inputs=False`` tells
                     the checker there is no feedthrough from inputs to this output,
                     which is required to break apparent cycles in discrete feedback
                     topologies (A→B→A) that are valid because updates use x⁻.
                  2. **Reduces compile time** by avoiding unnecessary input collection.

                Can also be specified as a list of integer port indices to declare
                selective feedthrough (only the listed inputs feed through to this
                output).  Defaults to ``True`` (collect all inputs, assume full
                feedthrough).
            prerequisites_of_calc (List[DependencyTicket], optional):
                The dependency tickets for the output port computation.  Defaults to
                None, in which case the assumption is a dependency on either (nothing)
                if `requires_inputs` is False otherwise (inputs).

        Returns:
            int: The index of the declared output port.
        """

        # T-A4-followup-requires-inputs-infer: when the caller leaves
        # ``requires_inputs`` unset (None) but supplies ``prerequisites_of_calc``,
        # infer whether inputs are needed from the prerequisites rather than
        # forcing the user to keep the two arguments consistent by hand.
        requires_inputs = self._resolve_requires_inputs(
            requires_inputs, prerequisites_of_calc
        )

        if default_value is not None:
            default_value = npa.array(default_value)

        cache_index = None
        if period is not None:
            # The output port will be of "sample-and-hold" type, so we have to declare a
            # periodic event to update the value.  The callback will be used to define the
            # update event, and the output callback will simply return the stored value.

            # This is the index that this port value will have in state.cache
            cache_index = len(self._default_cache)
            self._default_cache.append(default_value)

        output_port_idx = super().declare_output_port(
            callback, name=name, cache_index=cache_index, units=units
        )

        self.configure_output_port(
            output_port_idx,
            callback,
            period=period,
            offset=offset,
            prerequisites_of_calc=prerequisites_of_calc,
            default_value=default_value,
            requires_inputs=requires_inputs,
        )

        return output_port_idx

    def configure_output_port(
        self,
        port_index: int,
        callback: Callable,
        period: float = None,
        offset: float = 0.0,
        prerequisites_of_calc: List[DependencyTicket] = None,
        default_value: Array = None,
        requires_inputs: bool | list[int] | None = None,
    ):
        """Configure an output port in the LeafSystem.

        See `declare_output_port` for a description of the arguments.

        Args:
            port_index (int):
                The index of the output port to configure.

        Returns:
            None
        """
        if default_value is not None:
            default_value = npa.array(default_value)

        # Infer requires_inputs from prerequisites when left unset, so a
        # standalone configure_output_port call gets the same ergonomics as
        # declare_output_port. (T-A4-followup-requires-inputs-infer)
        requires_inputs = self._resolve_requires_inputs(
            requires_inputs, prerequisites_of_calc
        )

        # To help avoid unnecessary flagging of algebraic loops, trim the inputs as a
        # default prereq if the output callback doesn't use them
        if prerequisites_of_calc is None:
            if requires_inputs:
                prerequisites_of_calc = [DependencyTicket.u]
            else:
                prerequisites_of_calc = [DependencyTicket.nothing]

        if period is None:
            event = None
            _output_callback = self.wrap_callback(
                callback, collect_inputs=requires_inputs
            )
            cache_index = None

        else:
            # The output port will be of "sample-and-hold" type, so we have to declare a
            # periodic event to update the value.  The callback will be used to define the
            # update event, and the output callback will simply return the stored value.

            # This is the index that this port value will have in state.cache
            cache_index = self.output_ports[port_index].cache_index
            if cache_index is None:
                cache_index = len(self._default_cache)
                self._default_cache.append(default_value)

            def _output_callback(context: ContextBase) -> Array:
                state = context[self.system_id].state
                return state.cache[cache_index]

            def _update_callback(
                time: Scalar, state: LeafState, *inputs, **parameters
            ) -> LeafState:
                output = callback(time, state, *inputs, **parameters)
                return state.with_cached_value(cache_index, output)

            _update_callback = self.wrap_callback(
                _update_callback, collect_inputs=requires_inputs
            )

            # Create the associated update event
            event = DiscreteUpdateEvent(
                system_id=self.system_id,
                event_data=PeriodicEventData(
                    period=period, offset=offset, active=False
                ),
                name=f"{self.name}:output_{cache_index}",
                callback=_update_callback,
                passthrough=self._passthrough,
            )

            # Note that in this case the "prerequisites of calc" will correspond to the
            # prerequisites of the update event, not the literal output callback itself.
            # However, these can be used to determine dependencies for the update event
            # via the output port.

        super().configure_output_port(
            port_index,
            _output_callback,
            prerequisites_of_calc=prerequisites_of_calc,
            default_value=default_value,
            event=event,
            cache_index=cache_index,
        )

    def configure_continuous_state_default_value(
        self, callback_idx: int, default_value: Array, as_array: bool = True
    ):
        if as_array:
            dtype = self._default_continuous_state.dtype
            shape = self._default_continuous_state.shape
            default_value = utils.make_array(default_value, dtype=dtype, shape=shape)

        # Tree-map the default value to ensure that it is an array-like with the
        # correct shape and dtype. This is necessary because the default value
        # may be a list, tuple, or other PyTree-structured object.
        default_value = tree_util.tree_map(npa.asarray, default_value)

        _check_values_compatible(self._default_continuous_state, default_value)

        self._default_continuous_state = default_value
        if self._continuous_state_output_port_idx is not None:
            port = self.output_ports[self._continuous_state_output_port_idx]
            port.default_value = default_value
            self._default_cache[port.cache_index] = default_value

    def configure_output_port_default_value(
        self,
        port_index: int,
        default_value: Array,
    ):
        port = self.output_ports[port_index]
        if port.event is None:
            # T-107-followup-transport-delay-warn-quiet — demoted from
            # ``logger.warning`` to ``logger.debug``. ``TransportDelay``
            # (and similar event-less ports) emit this on every
            # construction; the behaviour is correct (the default really
            # is unused because the port has no periodic event), but the
            # WARNING level reads like a user-facing bug. The information
            # is still available at DEBUG for anyone wiring up a new
            # event-less port that *intended* to use ``default_value``.
            logger.debug(
                "period is None so default_value is not used for port %d in block %s",
                port_index,
                self.name,
            )
            return
        default_value = npa.array(default_value)
        cache_index = self.output_ports[port_index].cache_index

        if cache_index is None:
            raise ValueError(
                "Output port does not have a cache index, so default value cannot be set"
            )

        _check_values_compatible(self._default_cache[cache_index], default_value)
        self._default_cache[cache_index] = default_value

    def declare_continuous_state_output(
        self,
        name: str = None,
    ) -> int:
        """Declare a continuous state output port in the system.

        This method creates a new block-level output port which returns the full
        continuous state of the system.

        Args:
            name (str, optional):
                The name of the output port. Defaults to None (autogenerate name).

        Returns:
            int: The index of the new output port.
        """
        if self._continuous_state_output_port_idx is not None:
            raise ValueError("Continuous state output port already declared")

        def _callback(time: Scalar, state: LeafState, *inputs, **parameters):
            return state.continuous_state

        self._continuous_state_output_port_idx = self.declare_output_port(
            _callback,
            name=name,
            prerequisites_of_calc=[DependencyTicket.xc],
            default_value=self._default_continuous_state,
            requires_inputs=False,
        )
        return self._continuous_state_output_port_idx

    def declare_mode_output(self, name: str = None) -> int:
        """Declare a mode output port in the system.

        This method creates a new block-level output port which returns the component
        of the system's state corresponding to the discrete "mode" or "stage".

        Args:
            name (str, optional):
                The name of the output port. Defaults to None.

        Returns:
            int:
                The index of the declared mode output port.
        """

        def _callback(time: Scalar, state: LeafState, *inputs, **parameters):
            return state.mode

        self._mode_output_port_idx = self.declare_output_port(
            _callback,
            name=name,
            prerequisites_of_calc=[DependencyTicket.mode],
            default_value=self._default_mode,
            requires_inputs=False,
        )

        return self._mode_output_port_idx

    #
    # Event declaration
    #
    def declare_periodic_update(
        self,
        callback: Callable = None,
        period: Scalar | Parameter = None,
        offset: Scalar | Parameter = None,
        enable_tracing: bool = None,
    ):
        self._state_update_events.append(None)
        event_idx = len(self._state_update_events) - 1

        # FIXME: this is to preserve some backward compatibility while we decouple
        # declaration from configuration. Declaration should not have to call
        # configuration.
        if callback is not None:
            # Default ``offset`` to 0.0 when only ``period`` is supplied. Leaving
            # offset=None would propagate into PeriodicEventData and trigger an
            # opaque ``npa.minimum(None, ...)`` TypeError at the first scheduler
            # tick rather than at construction.
            if period is not None and offset is None:
                offset = 0.0
            self.configure_periodic_update(
                event_idx,
                callback,
                period,
                offset,
                enable_tracing=enable_tracing,
            )
        return event_idx

    def configure_periodic_update(
        self,
        event_index: int,
        callback: Callable,
        period: Scalar | Parameter,
        offset: Scalar | Parameter,
        enable_tracing: bool = None,
    ):
        """Configure an existing periodic update event.

        The event will be triggered at regular intervals defined by the period and
        offset parameters. The callback should have the signature
        `callback(time, state, *inputs, **params) -> xd_plus`, where `xd_plus` is the
        updated value of the discrete state.

        This callback should be written to compute the "plus" value of the discrete
        state component given the "minus" values of all state components and inputs.

        Args:
            event_index (int):
                The index of the event to configure.
            callback (Callable):
                The callback function defining the update.
            period (Scalar):
                The period at which the update event occurs.
            offset (Scalar):
                The offset at which the first occurrence of the event is triggered.
            enable_tracing (bool, optional):
                If True, enable tracing for this event. Defaults to None.
        """
        _wrapped_callback = self.wrap_callback(callback)

        def _callback(context: ContextBase) -> LeafState:
            xd = _wrapped_callback(context)
            return context[self.system_id].state.with_discrete_state(xd)

        if enable_tracing is None:
            enable_tracing = True

        event = DiscreteUpdateEvent(
            system_id=self.system_id,
            name=f"{self.name}:periodic_update",
            event_data=PeriodicEventData(period=period, offset=offset, active=False),
            callback=_callback,
            passthrough=self._passthrough,
            enable_tracing=enable_tracing,
            is_state_update=True,
        )
        self._state_update_events[event_index] = event

    def declare_default_mode(self, mode: int):
        self._default_mode = mode

    def configure_default_mode(self, mode: int):
        self._default_mode = mode
        if self._mode_output_port_idx:
            self.configure_output_port_default_value(self._mode_output_port_idx, mode)

    def declare_zero_crossing(
        self,
        guard: Callable,
        reset_map: Callable = None,
        start_mode: int = None,
        end_mode: int = None,
        direction: str = "crosses_zero",
        terminal: bool = False,
        name: str = None,
        enable_tracing: bool = None,
        zeno_tolerance: float | None = None,
        grad_guard: Callable = None,
    ):
        """Declare an event triggered by a zero-crossing of a guard function.

        Optionally, the system can also transition between discrete modes
        If `start_mode` and `end_mode` are specified, the system will transition
        from `start_mode` to `end_mode` when the event is triggered according to `guard`.
        This event will be active conditionally on `state.mode == start_mode` and when
        triggered will result in applying the reset map. In addition, the mode will be
        updated to `end_mode`.

        If `start_mode` and `end_mode` are not specified, the event will always be active
        and will not result in a mode transition.

        The guard function should have the signature:
            `guard(time, state, *inputs, **parameters) -> float`

        and the reset map should have the signature of an unrestricted update:
            `reset_map(time, state, *inputs, **parameters) -> state`

        Args:
            guard (Callable):
                The guard function which triggers updates on zero crossing.
            reset_map (Callable, optional):
                The reset map which is applied when the event is triggered. If None
                (default), no reset is applied.
            start_mode (int, optional):
                The mode or stage of the system in which the guard will be
                actively monitored. If None (default), the event will always be
                active.
            end_mode (int, optional):
                The mode or stage of the system to which the system will transition
                when the event is triggered. If start_mode is None, this is ignored.
                Otherwise it _must_ be specified, though it can be the same as
                start_mode.
            direction (str, optional):
                The direction of the zero crossing. Options are "crosses_zero"
                (default), "positive_then_non_positive", "negative_then_non_negative",
                and "edge_detection".  All except edge detection operate on continuous
                signals; edge detection operates on boolean signals and looks for a
                jump from False to True or vice versa.
            terminal (bool, optional):
                If True, the event will halt simulation if and when the zero-crossing
                occurs. If this event is triggered the reset map will still be applied
                as usual prior to termination. Defaults to False.
            name (str, optional):
                The name of the event. Defaults to None.
            enable_tracing (bool, optional):
                If True, enable tracing for this event. Defaults to None.

        Notes:
            By default the system state does not have a "mode" component, so in
            order to declare "state transitions" with non-null start and end modes,
            the user must first call `declare_default_mode` to set the default mode
            to be some integer (initial condition for the system).
        """

        logger.debug(
            f"Declaring transition for {self.name} with guard {guard} and reset map {reset_map}"
        )

        if enable_tracing is None:
            enable_tracing = True

        if start_mode is not None or end_mode is not None:
            assert (
                self._default_mode is not None
            ), "System has no mode: call `declare_default_mode` before transitions."
            assert isinstance(start_mode, int) and isinstance(end_mode, int)

        # T-027: optional Zeno-hold protection. If `zeno_tolerance` is set,
        # wrap the user's reset_map to flag a Zeno entry, declare a companion
        # exit event, and freeze the continuous-state ODE while held.
        if zeno_tolerance is not None:
            assert (
                isinstance(zeno_tolerance, (float, int)) and float(zeno_tolerance) > 0.0
            ), "zeno_tolerance must be a positive float"
            reset_map, _zeno_companion = self._install_zeno_protection(
                reset_map=reset_map,
                guard=guard,
                direction=direction,
                tol=float(zeno_tolerance),
                name=name,
            )
        else:
            _zeno_companion = None

        # Wrap the reset map with a mode update if necessary
        def _reset_and_update_mode(
            time: Scalar, state: LeafState, *inputs, **parameters
        ) -> LeafState:
            if reset_map is not None:
                state = reset_map(time, state, *inputs, **parameters)
            logger.debug(f"Updating mode from {state.mode} to {end_mode}")

            # If the start and end modes are declared, update the mode
            if start_mode is not None:
                logger.debug(f"Updating mode from {state.mode} to {end_mode}")
                state = state.with_mode(end_mode)

            return state

        _wrapped_guard = self.wrap_callback(guard)
        # Optional smooth guard residual for the event-time (saltation) gradient
        # only — wrapped the same way as the trigger guard.  ``None`` keeps the
        # legacy behaviour (the saltation paths fall back to ``guard``).
        _wrapped_grad_guard = (
            self.wrap_callback(grad_guard) if grad_guard is not None else None
        )
        _wrapped_reset = _wrap_reset_map(
            self, _reset_and_update_mode, _wrapped_guard, terminal,
            grad_guard=_wrapped_grad_guard,
        )

        event = ZeroCrossingEvent(
            system_id=self.system_id,
            guard=_wrapped_guard,
            grad_guard=_wrapped_grad_guard,
            reset_map=_wrapped_reset,
            passthrough=self._passthrough,
            direction=direction,
            is_terminal=terminal,
            name=name,
            event_data=ZeroCrossingEventData(active=True, triggered=False),
            enable_tracing=enable_tracing,
            active_mode=start_mode,
        )

        event_index = len(self._zero_crossing_events)
        self._zero_crossing_events.append(event)

        # T-115-followup-saturate-rate-classification: bump the
        # behavioral-ZC counter only when this event actually does
        # something on trigger — has a user-supplied reset map or
        # participates in a mode transition. Pure guard-only events
        # (Saturate / DeadZone clip boundaries) are solver hints with
        # no behavioral effect, so they should not flip the block's
        # rate-group classification to ``event_driven``.
        if (
            reset_map is not None
            or start_mode is not None
            or end_mode is not None
        ):
            self._n_behavioral_zc_events += 1

        # Record the transition in the transition map (for debugging or analysis)
        if start_mode is not None:
            if start_mode not in self.transition_map:
                self.transition_map[start_mode] = []
            self.transition_map[start_mode].append((event_index, event))

        # T-027: register the companion `_exit_zeno` event AFTER the main event
        # so the slot index is finalized first.
        if _zeno_companion is not None:
            _zeno_companion(event_index)

    def _install_zeno_protection(
        self,
        reset_map: Callable | None,
        guard: Callable,
        direction: str,
        tol: float,
        name: str | None,
    ) -> Tuple[Callable, Callable]:
        """T-027: install per-event Zeno-hold tracking on a `declare_zero_crossing`.

        Returns ``(wrapped_reset_map, register_companion)``.

        - ``wrapped_reset_map`` runs the user's reset, then sets the per-event
          Zeno flag if `(time - tprev) < tol`.
        - ``register_companion(event_index)`` declares the partner ``_exit_zeno``
          event whose reset clears the flag. Called by ``declare_zero_crossing``
          once the main event index is known.

        Storage layout:
        - If the host LeafSystem has NO user discrete state, the discrete state
          slot holds the bare ``_ZenoState(zeno, tprev)`` NamedTuple of arrays
          sized to the number of protected events.
        - T-027a: if the host LeafSystem ALSO calls ``declare_discrete_state``,
          the discrete state is packed as ``_DiscreteWithZeno(user, zeno)``.
          User callbacks (ode, guard, reset) are wrapped to see only their own
          ``user`` slot via ``state.discrete_state``; the framework re-packs on
          the way out. Order of declarations does not matter.

        The host's continuous-state ode is wrapped exactly once: when ANY
        protected event has ``zeno=True``, the ode output is multiplied by
        0 to freeze the state.
        """
        from collections import namedtuple

        # Reverse the direction for the exit companion event.
        _reverse = {
            "positive_then_non_positive": "negative_then_non_negative",
            "negative_then_non_negative": "positive_then_non_positive",
            "crosses_zero": "crosses_zero",
        }
        if direction not in _reverse:
            raise ValueError(
                f"zeno_tolerance is not supported for direction={direction!r}"
            )
        exit_direction = _reverse[direction]

        # Allocate this event's slot. We grow the discrete state lazily so the
        # number of protected events does not need to be known up-front.
        slot = len(self._zeno_protected_events)
        self._zeno_protected_events.append(
            {"slot": slot, "tol": tol, "name": name}
        )

        # On first installation: set up types and capture any pre-existing
        # user discrete state so we can pack it alongside the Zeno tracker.
        if slot == 0:
            self._zeno_state_type = namedtuple("_ZenoState", ["zeno", "tprev"])
            self._zeno_combined_type = namedtuple(
                "_DiscreteWithZeno", ["user", "zeno"]
            )
            # If the user already declared discrete state, capture its default
            # so we can preserve it inside the combined wrapper. Subsequent
            # `declare_discrete_state` calls also flow through this path
            # (they update `_default_discrete_state` directly; we re-pack on
            # `create_state`).
            self._zeno_user_default = self._default_discrete_state
        # (Re)build the default value so it always matches the current count.
        n = slot + 1
        zeno_default = self._zeno_state_type(
            zeno=npa.zeros(n, dtype=bool),
            tprev=npa.zeros(n, dtype=float),
        )
        if self._zeno_user_default is not None:
            self._default_discrete_state = self._zeno_combined_type(
                user=self._zeno_user_default, zeno=zeno_default
            )
        else:
            self._default_discrete_state = zeno_default

        zeno_type = self._zeno_state_type
        combined_type = self._zeno_combined_type

        def _split_xd(xd):
            """Return (user_xd_or_None, zeno_xd) given the framework discrete state."""
            if isinstance(xd, combined_type):
                return xd.user, xd.zeno
            return None, xd

        def _pack_xd(user_xd, zeno_xd):
            """Re-pack the framework discrete state from updated parts."""
            if user_xd is None:
                return zeno_xd
            return combined_type(user=user_xd, zeno=zeno_xd)

        # Wrap the user's reset_map: present a "user view" of state.discrete_state
        # (just their own xd, not the wrapper), run the reset, and re-pack with
        # the updated Zeno tracker.
        user_reset = reset_map

        def _zeno_aware_reset(time, state, *inputs, **params):
            user_xd, zeno_xd = _split_xd(state.discrete_state)
            if user_reset is not None:
                if user_xd is not None:
                    # Present a user-facing view: replace the wrapper with just
                    # the user's slot. Bypass `with_discrete_state`'s
                    # tree-shape coercion (the wrapper has a different
                    # structure from the user's xd).
                    user_view = dataclasses.replace(state, discrete_state=user_xd)
                else:
                    user_view = state
                out_state = user_reset(time, user_view, *inputs, **params)
                new_user_xd = out_state.discrete_state if user_xd is not None else None
                # Carry over continuous_state / mode / cache from the user's return.
                state = out_state
            else:
                new_user_xd = user_xd
            dt = time - zeno_xd.tprev[slot]
            entered = (dt - tol) <= 0.0
            new_zeno = zeno_xd.zeno.at[slot].set(
                npa.logical_or(zeno_xd.zeno[slot], entered)
            )
            new_tprev = zeno_xd.tprev.at[slot].set(time)
            new_zeno_xd = zeno_type(zeno=new_zeno, tprev=new_tprev)
            return dataclasses.replace(
                state, discrete_state=_pack_xd(new_user_xd, new_zeno_xd)
            )

        # Companion `_exit_zeno` event: its guard is the user's guard re-used,
        # but with reversed direction so it fires when the trigger condition
        # disappears. The reset clears the slot's `zeno` flag (and leaves the
        # user's discrete state unchanged).
        def _register_companion(main_event_index: int):
            def _exit_reset(time, state, *_inputs, **_params):
                user_xd, zeno_xd = _split_xd(state.discrete_state)
                new_zeno = zeno_xd.zeno.at[slot].set(False)
                new_zeno_xd = zeno_type(zeno=new_zeno, tprev=zeno_xd.tprev)
                return dataclasses.replace(
                    state, discrete_state=_pack_xd(user_xd, new_zeno_xd)
                )

            self.declare_zero_crossing(
                guard=guard,
                reset_map=_exit_reset,
                direction=exit_direction,
                name=(f"{name}__exit_zeno" if name else "_exit_zeno"),
            )

        # Wrap the ode once: multiply by `(1 - any_zeno)` so any protected
        # event being held freezes the entire host system.
        if not self._zeno_ode_wrapped and self.ode_callback is not None:
            self._wrap_ode_for_zeno()
        # Mark a deferred wrap if `declare_continuous_state` happens later.
        self._zeno_ode_wrapped = self._zeno_ode_wrapped or (
            self.ode_callback is not None
        )

        return _zeno_aware_reset, _register_companion

    def _wrap_ode_for_zeno(self) -> None:
        """T-027: replace the ode_callback to freeze when any Zeno flag is set.

        T-027a: handle the combined ``_DiscreteWithZeno(user, zeno)`` layout.
        """
        if self.ode_callback is None or self.ode_callback._callback is None:
            return
        original = self.ode_callback._callback
        sys_id = self.system_id
        combined_type = self._zeno_combined_type

        def _frozen_ode(context):
            xdot = original(context)
            xd = context[sys_id].state.discrete_state
            zeno_xd = xd.zeno if isinstance(xd, combined_type) else xd
            any_zeno = npa.any(zeno_xd.zeno)
            scale = npa.where(any_zeno, 0.0, 1.0)
            return tree_util.tree_map(lambda v: v * scale, xdot)

        self.ode_callback._callback = _frozen_ode
        # Keep `eval_time_derivatives` pointing at the (now-wrapped) callback.
        self.eval_time_derivatives = self.ode_callback.eval
        self._zeno_ode_wrapped = True

    #
    # Initialization
    #
    @property
    def context_factory(self) -> LeafContextFactory:
        return LeafContextFactory(self)

    @property
    def dependency_graph_factory(self) -> LeafDependencyGraphFactory:
        return LeafDependencyGraphFactory(self)

    def create_state(self) -> LeafState:
        # Hook for context creation: get the default state for this system.
        # Users should not need to call this directly - the state will be created
        # as part of the context.  Generally, `system.create_context()` should
        # be all that's necessary for initialization.
        self.reset_default_values(**self.dynamic_parameters)
        return LeafState(
            name=self.name,
            continuous_state=self._default_continuous_state,
            discrete_state=self._default_discrete_state,
            mode=self._default_mode,
            cache=tuple(self._default_cache),
        )

    def initialize_static_data(self, context: ContextBase):
        # Try to infer any missing default values for "sample-and-hold" output ports
        # and any other cached computations.
        cached_callbacks: list[SystemCallback] = [
            cb for cb in self.callbacks if cb.cache_index is not None
        ]

        for callback in cached_callbacks:
            i = callback.cache_index
            if self._default_cache[i] is None:
                try:
                    if isinstance(callback, OutputPort):
                        # Try to eval the callback for the _event_ (not the output
                        # port return function), which would return a value of the
                        # right data type for the output port, provided it is connected
                        _eval = callback.event.callback
                    else:
                        # If it's not an output port, the callback function evaluation
                        # should return the correct data type.
                        _eval = callback.eval

                    state: LeafState = _eval(context)
                    y = state.cache[i]
                    self._default_cache[i] = y
                    local_context = context[self.system_id].with_cached_value(i, y)
                    context = context.with_subcontext(self.system_id, local_context)
                except UpstreamEvalError:
                    logger.debug(
                        "%s.initialize_static_data: UpstreamEvalError. "
                        "Continuing without default value initialization.",
                        self.name,
                    )

        return context

    def _create_dependency_cache(self) -> dict[int, CallbackTracer]:
        cache = {}
        for source in self.callbacks:
            cache[source.callback_index] = CallbackTracer(ticket=source.ticket)
        return cache

    # Inherits docstring from SystemBase.get_feedthrough
    def get_feedthrough(self) -> List[Tuple[int, int]]:
        # NOTE: This implementation is basically a direct port of the Drake algorithm

        if self.dependency_graph is None:
            raise ValueError("Must create dependency graph first.")

        # If we already did this or it was set manually, return the stored value
        if self.feedthrough_pairs is not None:
            return self.feedthrough_pairs

        feedthrough = []  # Confirmed feedthrough pairs (input, output)

        # First collect all possible feedthrough pairs
        unknown: Set[Tuple[int, int]] = set()
        for iport in self.input_ports:
            for oport in self.output_ports:
                unknown.add((iport.index, oport.index))

        if len(unknown) == 0:
            return feedthrough

        # Create a local context and "cache".  The cache here just contains CallbackTracer
        # objects that can be used to trace dependencies through the system, but
        # otherwise don't store any actual values.  This is different from any "cached"
        # computations that might be stored in the state for reuse by multiple ports or
        # downstream calculations within the system.
        #
        # This cache will only contain local sources - this is fine since we're just
        # testing local input -> output paths for this system.
        cache = self._create_dependency_cache()

        original_unknown = unknown.copy()
        for pair in original_unknown:
            u, v = pair
            output_port = self.output_ports[v]
            input_port = self.input_ports[u]

            # If output prerequisites are unspecified, this tells us nothing
            if DependencyTicket.all_sources in output_port.prerequisites_of_calc:
                continue

            # Determine feedthrough dependency via cache invalidation
            cache = _mark_up_to_date(cache, output_port.callback_index)

            # Notify subscribers of a value change in the input, invalidating all
            # downstream cache values
            input_tracker = self.dependency_graph[input_port.ticket]
            cache = input_tracker.notify_subscribers(
                cache, self.dependency_graph, local_only=True
            )

            # If the output cache is now out of date, this is a feedthrough path
            if cache[output_port.callback_index].is_out_of_date:
                feedthrough.append(pair)

            # Regardless of the result of the caching, the pair is no longer unknown
            unknown.remove(pair)

            # Reset the output cache to out-of-date in case other inputs also
            # feed through to this output.
            cache = _mark_out_of_date(cache, output_port.callback_index)

        logger.debug(f"{self.name} feedthrough pairs: {feedthrough}")

        # Conservatively assume everything still unknown is feedthrough
        for pair in unknown:
            feedthrough.append(pair)

        self.feedthrough_pairs = feedthrough
        return self.feedthrough_pairs

    def reset_default_values(self, **dynamic_parameters):
        """This function is used to reset default values for
        continuous/discrete states, ports and mode based on dynamic parameters.
        It is called in `create_state()` and used to reset states in ensemble sims
        and optimization with the context method `with_new_state()`.

        Note that dtypes and shapes can't be changed after initialization because
        the diagram may already have been jax-compiled. Only values may change.
        """
        pass


#
# Zero-crossing event handling with custom adjoint definitions
#
def _saltation_correct(reset_adj: "ContextBase", grad_g: "ContextBase", gamma):
    """Add the rank-1 saltation correction ``gamma · ∇g`` to the reset-map
    cotangent over the differentiable (float) leaves of the root context.

    ``reset_adj`` is the bare reset-map VJP result ``(dR)^T λ⁺`` and ``grad_g``
    is the total guard gradient ``∇g`` (both cotangents of the same root
    context, so they share a pytree structure).  Real-valued leaves get
    ``a + gamma * b``; integer / boolean / ``float0`` leaves (discrete modes,
    counters, and the float0 cotangents of integer state) keep the bare
    reset-map value ``a`` unchanged, so cotangent dtypes stay consistent across
    the event ``cond`` branches (T-001c-followup #1b).
    """
    float0 = jax.dtypes.float0

    def _leaf(a, b):
        bd = getattr(b, "dtype", None)
        if bd is not None and bd != float0 and jnp.issubdtype(bd, jnp.inexact):
            return a + gamma * b
        return a

    return jax.tree_util.tree_map(_leaf, reset_adj, grad_g)


def _wrap_reset_map(
    system: SystemBase,
    reset_map: Callable,
    guard: Callable,
    is_terminal: bool,
    grad_guard: Callable = None,
) -> Callable:
    """Wrap the reset map with a custom adjoint definition.

    Ignoring "saltation" effects, we could just use the built-in autodiff.
    However, we need to correct for variations in the "time of impact", which
    has ramifications for the continuous state, time, and parameters in the
    _local_ context.  Only the local context is affected since we assume that the
    guard and reset maps can only see and act on the local state. We need to
    compute these corrections to the adjoints and then store them in an updated
    _root_ context (since this was the original input).

    The adjoint update to time is also overridden at the Simulator level
    because it is most naturally defined as t_adj = dot(xf_dot, vf) at the
    end of the `advance` call. However, we also need to update it here in case the
    adjoint time variable is used by other things farther in the graph.

    For a helpful tutorial related to this implementation, see:
    "Saltation Matrices: The Essential Tool for Linearizing Hybrid Dynamical Systems"
    https://arxiv.org/abs/2306.06862
    This is a bit more complicated because we allow for differentiation with respect to
    more than just the initial state, but it introduces the "saltation" idea.
    """

    system_id = system.system_id

    # Transform the signature from (time, state, *inputs, **params) -> state to
    # context -> state.
    reset_map = system.wrap_callback(reset_map)

    if IS_JAXLITE:
        return reset_map

    # Evaluate the ODE RHS function.  This is needed to get the time sensitivity
    # of the continuous state, which is used in the adjoint update to time.
    def _ode(x: LeafState, context: ContextBase) -> Array:
        local_context = context[system_id].with_state(x)
        context = context.with_subcontext(system_id, local_context)
        return system.eval_time_derivatives(context)

    def _reset_map_fwd(context: ContextBase) -> LeafState:
        """Compute the "forward pass" of the zero-crossing update function."""

        # This basically just wraps the event handler function, but it also computes various
        # "residual" information that will be necessary for the backwards pass. The way to
        # understand what residuals are needed is to start with the adjoint function and then
        # see what information used there can be more efficiently computed in the forward
        # pass.

        # Evaluate the _local_ system dynamics before the transition
        x_minus = context[system_id].state
        xdot_minus = _ode(x_minus, context)  # This is the _local_ xdot

        # Total gradient of the guard w.r.t. the *root* context, computed
        # against a context whose input-port cache is refreshed so that the
        # guard's dependence on signals it reads through input ports is *live*
        # rather than frozen.  Differentiating the cached guard (the previous
        # behaviour) captured only the guard's explicit local arguments and so
        # missed, whenever the guard reads an upstream signal
        # (T-001c-followup #1d), both:
        #   * the event-time denominator's du/dt contribution (so the
        #     denominator collapsed to ~0 and the saltation correction
        #     vanished for guards driven by a source), and
        #   * the upstream-parameter numerator ∂g/∂p_upstream (the crossing
        #     time's dependence on a parameter entering through the port).
        # Refreshing the port cache inside the differentiated function re-
        # expresses each input as a live function of the (time, state,
        # parameters) it is computed from, so ``jax.grad`` propagates through
        # the port into the producing block.  ``dg_droot`` then carries ∂g/∂·
        # on every root-context leaf the guard depends on — exactly the ∇g the
        # full saltation rank-1 correction needs.
        # Use the smooth ``grad_guard`` residual for ∇g when one was supplied
        # (the trigger ``guard`` may be non-smooth, e.g. a boolean predicate);
        # falls back to ``guard`` otherwise.
        _guard_for_grad = grad_guard if grad_guard is not None else guard

        def _live_guard(c: ContextBase) -> Scalar:
            return _guard_for_grad(c.refresh_port_cache())

        dg_droot: ContextBase = jax.grad(_live_guard, allow_int=True)(context)

        # Reset map VJP, and compute the transition (primal values)
        # Returns the _local_ updated state x_plus, and the function to compute
        # the vjp of the reset map given the adjoint state. This will be used in the
        # backwards pass.
        x_plus, reset_vjp = jax.vjp(reset_map, context)

        # Recompute the _local_ ODE values after the transition.  This will be zero
        # if the event is terminal, since the state does not advance after the event.
        # Note that this can use standard Python control flow because it is known
        # at compile time whether an event is terminal or not.
        if not context[system_id].has_continuous_state:
            # Can't "zero out" the continuous state if it doesn't exist
            xdot_plus = None
        elif is_terminal:
            xdot_plus = 0 * xdot_minus
        else:
            xdot_plus = _ode(x_plus, context)

        # Combine all the "residuals" necessary for the backwards pass into a tuple.
        res = (dg_droot, xdot_minus, xdot_plus, reset_vjp)
        return x_plus, res

    def _reset_map_adj(res: tuple, state_adj: LeafState) -> ContextBase:
        """Compute the "backward pass" of the zero-crossing update function."""

        # Unpack the residuals from the forward pass
        dg_droot, xdot_minus, xdot_plus, reset_vjp = res

        # Compute vjp with reset Jacobian (return is a tuple of length 1). If we did not
        # account for saltation effects we could just return this directly.
        root_context_adj: ContextBase = reset_vjp(state_adj)[0]

        # Short-circuit the rank-1 saltation correction when the local block
        # has no continuous state (T-001c-followup #1). In that case
        # `xdot_minus`, `xdot_plus`, `dg_dx`, `vT_dR_dx`, and `vc` are all
        # None/empty, so the rank-1 update is ill-defined. The saltation matrix
        # is identity for blocks without a continuous component (no local
        # dynamics jump to localize the event time against), so returning the
        # bare reset-map cotangent is exactly correct (no correction needed).
        local_dg_dx = dg_droot[system_id].continuous_state
        if xdot_minus is None or local_dg_dx is None:
            return (root_context_adj,)

        # Saltation scalar gamma = num / den, built from purely *local*
        # quantities: the local guard state-gradient, the local dynamics jump
        # (xdot_minus -> xdot_plus), the reset-map's local Jacobian, and the
        # incoming continuous cotangent.
        vc = state_adj.continuous_state
        vT_dR_dx = root_context_adj[system_id].continuous_state
        vT_dR_dt = root_context_adj.time  # time is only defined at the root

        # Denominator: TOTAL time-derivative of the guard along the trajectory.
        # ``dg_droot.time`` already carries the explicit-time + input-signal
        # du/dt contributions (from the live port-cache refresh in the forward
        # pass); add the local-state contribution dg_dx·xdot_minus.
        den = dg_droot.time + jnp.dot(local_dg_dx, xdot_minus)
        num = jnp.dot(xdot_plus, vc) - jnp.dot(vT_dR_dx, xdot_minus) - vT_dR_dt
        safe_den = jnp.where(den != 0, den, 1.0)
        gamma = jnp.where(den != 0, num / safe_den, 0.0 * num)

        # Full saltation adjoint: Ξ^T λ⁺ = (dR)^T λ⁺ + gamma · ∇g, applied to
        # every differentiable (float) leaf of the *root* context the guard
        # depends on — local state / parameters / time AND any upstream
        # parameters, states, or input signals the guard reads through its
        # input ports (which ``dg_droot`` now carries thanks to the live
        # refresh).  Non-float leaves (discrete modes, integer counters, the
        # float0 cotangents of integer state) keep the bare reset-map cotangent
        # so cotangent dtypes stay consistent across the event ``cond``
        # branches (T-001c-followup #1b).
        root_context_adj = _saltation_correct(root_context_adj, dg_droot, gamma)

        return (root_context_adj,)

    _reset_map = jax.custom_vjp(reset_map)
    _reset_map.defvjp(_reset_map_fwd, _reset_map_adj)
    return _reset_map
