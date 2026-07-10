# SPDX-License-Identifier: MIT

"""Python DSL for constructing :class:`jaxonomy.library.state_machine.StateMachine` blocks."""

from __future__ import annotations

import ast
import uuid
from dataclasses import dataclass, field
from typing import Optional

__all__ = [
    "State",
    "StateMachineBuilder",
    "Transition",
]

_KEYWORD_LIKE = frozenset({"and", "or", "not", "in", "is", "lambda"})
_EXTRA_GLOBALS = frozenset({"jnp", "np", "npa"})


@dataclass
class State:
    """A state-machine state.

    T-024 / T-024a: nested-state support.  ``parent`` is the optional
    enclosing composite state.  ``children`` is populated by the
    builder when ``add_state`` is called with a parent.  ``during`` is a
    list of action strings — folded into the on_entry chain of every
    transition that *enters* this state's region (a transition between
    two siblings inside this composite does NOT re-fire ``during``,
    matching UML "stay in state, no re-entry" semantics).
    ``initial_child`` (T-024a) is the default sub-state to auto-descend
    to when a transition targets this composite.  Defaults to the first
    child added; can be overridden via ``builder.set_initial_child``.
    """

    name: str
    uuid: str = field(default_factory=lambda: str(uuid.uuid4()))
    on_entry: list[str] = field(default_factory=list)
    on_exit: list[str] = field(default_factory=list)
    during: list[str] = field(default_factory=list)
    parent: Optional["State"] = None
    children: list["State"] = field(default_factory=list)
    initial_child: Optional["State"] = None

    @property
    def is_leaf(self) -> bool:
        return not self.children

    def ancestors(self) -> list["State"]:
        """Outermost-first list of strict ancestors (self excluded)."""
        chain = []
        node = self.parent
        while node is not None:
            chain.append(node)
            node = node.parent
        return list(reversed(chain))


@dataclass
class Transition:
    """A transition between two states."""

    source: State
    dest: State
    guard: str
    actions: list[str] = field(default_factory=list)
    uuid: str = field(default_factory=lambda: str(uuid.uuid4()))


def _names_in_eval_expr(expr: str) -> set[str]:
    tree = ast.parse(expr, mode="eval")
    out = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            out.add(node.id)
    return out - _KEYWORD_LIKE - _EXTRA_GLOBALS


def _action_assigned_and_free(stmt: str) -> tuple[set[str], set[str]]:
    tree = ast.parse(stmt.strip())
    assigned: set[str] = set()
    free: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name):
                    assigned.add(t.id)
        elif isinstance(node, ast.AugAssign) and isinstance(node.target, ast.Name):
            assigned.add(node.target.id)
        elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
            free.add(node.id)
    free -= assigned
    free -= _KEYWORD_LIKE
    free -= _EXTRA_GLOBALS
    return assigned, free


class StateMachineBuilder:
    """Build a :class:`~jaxonomy.library.state_machine.StateMachine` from Python.

    Guards and actions are Python expression/statement strings, evaluated like JSON-defined
    state machines (see :mod:`jaxonomy.dashboard.serialization.block_interface`).
    The default ``time_mode`` is ``\"agnostic\"`` (zero-crossing guards); connect a
    :class:`~jaxonomy.library.primitives.Clock` (or other time source) when guards use ``t``.

    **Transition priority:** when multiple guards on the same source state are simultaneously
    true, the *first* transition added via :meth:`add_transition` fires (insertion order).

    **Entry / exit actions:** :attr:`State.on_entry` and :attr:`State.on_exit` action-string
    lists are fully supported for all states.  At :meth:`build` time they are folded into
    transition actions in UML order::

        on_exit(source) → transition actions → on_entry(dest)

    The initial state's ``on_entry`` list also runs at diagram initialisation time.
    """

    def __init__(self):
        self._states: list[State] = []
        self._transitions: list[Transition] = []
        self._initial_state: Optional[State] = None

    def add_state(self, name: str, parent: Optional[State] = None) -> State:
        """Add a state and return it for use in transitions.

        T-024: pass ``parent=composite_state`` to nest the new state
        inside an enclosing composite.  Use :meth:`add_compound_state`
        as a synonym when the state is intended to host children only.
        """
        if parent is not None and parent not in self._states:
            raise ValueError(
                f"parent state {parent.name!r} was not added to this builder"
            )
        state = State(name=name, parent=parent)
        if parent is not None:
            parent.children.append(state)
            # T-024a: first child added becomes the composite's default
            # initial child if none has been set yet.  Override with
            # ``set_initial_child(parent, child)`` if a different default
            # is desired.
            if parent.initial_child is None:
                parent.initial_child = state
        self._states.append(state)
        return state

    def set_initial_child(self, parent: State, child: State) -> None:
        """T-024a: override the composite ``parent``'s default initial
        child to ``child``.  Used at transition resolution time when a
        transition targets the composite directly — the runtime
        descends through ``initial_child`` chains until it lands on a
        leaf state.

        ``child`` must be a direct child of ``parent``.
        """
        if child not in parent.children:
            raise ValueError(
                f"State {child.name!r} is not a child of {parent.name!r}; "
                "use add_state(name, parent=parent) to declare it first."
            )
        parent.initial_child = child

    def add_compound_state(
        self, name: str, parent: Optional[State] = None,
    ) -> State:
        """Add a composite (non-leaf) state.  Same as :meth:`add_state`.

        Use this when the state's role is purely structural — it will
        have children added under it via ``add_state(name, parent=it)``.
        Composite states' ``on_entry`` / ``on_exit`` / ``during`` lists
        are folded into every leaf descendant's action chain at build
        time.
        """
        return self.add_state(name, parent=parent)

    def set_initial_state(self, state: State) -> None:
        """Set the initial / entry state."""
        if state not in self._states:
            raise ValueError(
                f"State {state.name!r} was not added to this builder (use add_state first)."
            )
        self._initial_state = state

    def add_transition(
        self,
        source: State,
        dest: State,
        guard: str,
        actions: Optional[list[str]] = None,
    ) -> Transition:
        """Add a prioritized exit from ``source`` to ``dest`` when ``guard`` holds.

        Args:
            source: Source state (must have been created with :meth:`add_state`).
            dest:   Destination state (same).
            guard:  A Python **expression string** that evaluates to a bool, e.g.
                    ``"x > 1.0"`` or ``"mode == 2 and error < 0.01"``.
                    Do **not** pass a callable — pass the expression as a string.
            actions: Optional list of Python statement strings to execute when the
                     transition fires (after ``source.on_exit``, before ``dest.on_entry``).
        """
        if not isinstance(guard, str):
            raise TypeError(
                f"guard must be a Python expression string (e.g. 'x > 1.0'), "
                f"not {type(guard).__name__!r}. "
                "Callables / lambdas are not accepted; pass the expression as a string."
            )
        if source not in self._states or dest not in self._states:
            raise ValueError("source and dest must be states returned by add_state()")
        t = Transition(
            source=source,
            dest=dest,
            guard=guard,
            actions=list(actions or []),
        )
        self._transitions.append(t)
        return t

    def _extract_guard_variables(self) -> set[str]:
        """Names appearing in guard strings (excluding keywords / common globals)."""
        names: set[str] = set()
        for t in self._transitions:
            names |= _names_in_eval_expr(t.guard)
        return names

    def _collect_io_names(self) -> tuple[list[str], list[str]]:
        """Return ``(input_names, output_names)`` for the underlying block.

        Scans transition actions *and* all state ``on_entry``/``on_exit`` action
        lists so that variables assigned only in entry/exit hooks are still
        promoted to output ports.
        """
        assigned: set[str] = set()
        free_in_actions: set[str] = set()

        # Transition actions
        for t in self._transitions:
            for stmt in t.actions:
                a, f = _action_assigned_and_free(stmt)
                assigned |= a
                free_in_actions |= f

        # State on_entry / on_exit action lists
        for s in self._states:
            for stmt in s.on_entry + s.on_exit:
                a, f = _action_assigned_and_free(stmt)
                assigned |= a
                free_in_actions |= f

        guard_names = self._extract_guard_variables()
        output_names = sorted(assigned)
        output_set = set(output_names)
        inputs_set = (guard_names | free_in_actions) - output_set
        input_names = sorted(inputs_set)
        return input_names, output_names

    @staticmethod
    def _lca(a: State, b: State) -> Optional[State]:
        """Lowest common ancestor of ``a`` and ``b``, or None if they
        share no common parent.  Returns the outermost state common
        to both ancestor chains (closest to the root)."""
        a_chain = a.ancestors() + [a]
        b_chain = b.ancestors() + [b]
        common = None
        for x, y in zip(a_chain, b_chain):
            if x is y:
                common = x
            else:
                break
        return common

    def _entry_chain_to(
        self, state: State, stop_at: Optional[State] = None,
        *, include_during: bool = False,
    ) -> list[str]:
        """Concatenated ``on_entry`` strings for the path from
        ``stop_at`` (exclusive) down to and including ``state``.
        ``stop_at=None`` means walk the entire chain from the root.

        T-024a: ``include_during=True`` appends each ancestor's
        ``during`` actions after that state's ``on_entry`` — folding
        UML "during" semantics into the entry chain so they fire once
        when the machine enters that state's region.
        """
        chain = state.ancestors() + [state]
        if stop_at is not None:
            try:
                idx = chain.index(stop_at)
                chain = chain[idx + 1:]
            except ValueError:
                # stop_at not on the chain — defensive.
                pass
        actions: list[str] = []
        for s in chain:
            actions.extend(s.on_entry)
            if include_during:
                actions.extend(s.during)
        return actions

    @staticmethod
    def _descend_to_leaf(state: State) -> tuple[State, list[State]]:
        """T-024a: starting from ``state``, walk ``initial_child``
        until a leaf is reached.  Returns ``(leaf, descended_path)``
        where ``descended_path`` is the list of composites traversed
        AFTER ``state`` (so a transition targeting ``state`` directly
        ends up at ``leaf`` and fires entry chains for the composites
        in ``descended_path``)."""
        if state.is_leaf:
            return state, []
        path: list[State] = []
        node = state
        while not node.is_leaf:
            if node.initial_child is None:
                raise ValueError(
                    f"Composite state {node.name!r} has no initial child; "
                    "set one via builder.set_initial_child(parent, child) "
                    "or add at least one child to it before building."
                )
            node = node.initial_child
            path.append(node)
        return node, path

    def _exit_chain_from(
        self, state: State, stop_at: Optional[State] = None,
    ) -> list[str]:
        """Concatenated ``on_exit`` strings for the path from ``state``
        up to ``stop_at`` (exclusive).  ``stop_at=None`` walks all the
        way to the root."""
        chain = [state] + state.ancestors()[::-1]  # innermost-first
        if stop_at is not None:
            try:
                idx = chain.index(stop_at)
                chain = chain[:idx]
            except ValueError:
                pass
        actions: list[str] = []
        for s in chain:
            actions.extend(s.on_exit)
        return actions

    def _entry_actions(self, output_names: list[str]) -> list[str]:
        from jaxonomy.dashboard.serialization.block_interface import (
            _extract_assigned_vars,
        )

        # T-024 + T-024a: include the full ancestor on_entry chain (and
        # during actions) when the initial state is nested.  If the
        # initial state is itself composite, descend to a leaf and
        # include the descended composites' entry/during actions too.
        actions = list(self._entry_chain_to(
            self._initial_state, include_during=True,
        ))
        _, descended = self._descend_to_leaf(self._initial_state)
        for inner in descended:
            actions.extend(inner.on_entry)
            actions.extend(inner.during)
        covered: set[str] = set()
        for stmt in actions:
            covered |= _extract_assigned_vars(stmt)
        for name in output_names:
            if name not in covered:
                actions.append(f"{name} = 0.0")
                covered.add(name)
        return actions

    def _to_model_json(self, entry_actions: list[str]):
        from uuid import UUID, uuid4

        from jaxonomy.dashboard.serialization import model_json

        nodes = []
        for s in self._states:
            exit_ids = [UUID(t.uuid) for t in self._transitions if t.source is s]
            nodes.append(
                model_json.StateMachineState(
                    name=s.name,
                    uuid=UUID(s.uuid),
                    exit_priority_list=exit_ids,
                )
            )

        links = []
        for t in self._transitions:
            # T-024 + T-024a:
            #   - LCA semantics for entry/exit chains (T-024).
            #   - If t.dest is a composite, auto-descend via
            #     initial_child chain to a leaf; entry chain extends
            #     down through descended composites.
            #   - Each entered state's `during` actions (T-024a) fire
            #     after its on_entry — UML "fire once on entry to
            #     region" semantics.
            resolved_dest, descended = self._descend_to_leaf(t.dest)
            lca = self._lca(t.source, t.dest)
            exit_chain = self._exit_chain_from(t.source, stop_at=lca)
            entry_chain = self._entry_chain_to(
                t.dest, stop_at=lca, include_during=True,
            )
            # Append entry/during for any composites we descended into
            # past t.dest (only relevant when t.dest was a composite).
            for inner in descended:
                entry_chain.extend(inner.on_entry)
                entry_chain.extend(inner.during)
            merged_actions = (
                exit_chain + list(t.actions) + entry_chain
            )
            links.append(
                model_json.StateMachineTransition(
                    uuid=UUID(t.uuid),
                    sourceNodeId=UUID(t.source.uuid),
                    destNodeId=UUID(resolved_dest.uuid),
                    guard=t.guard,
                    actions=merged_actions,
                )
            )

        # T-024a: if the initial state is composite, the runtime entry
        # point lands on its descended initial-child leaf.
        initial_leaf, _ = self._descend_to_leaf(self._initial_state)
        entry = model_json.StateMachineEntryPoint(
            dest_id=UUID(initial_leaf.uuid),
            actions=list(entry_actions),
        )

        return model_json.StateMachine(
            uuid=uuid4(),
            nodes=nodes,
            links=links,
            entry_point=entry,
        )

    def build(
        self,
        name: str = "state_machine",
        time_mode: str = "agnostic",
        dt: float | None = None,
    ):
        """Compile to a :class:`~jaxonomy.library.state_machine.StateMachine` leaf block.

        Args:
            name: Block name.
            time_mode: ``"agnostic"`` (default — runs whenever a guard
                fires) or ``"discrete"`` (fixed-rate, requires ``dt``).
                The underlying ``StateMachine`` block does not currently
                expose a "continuous" time mode. Without this kwarg
                users had to reach into ``built_block._sm`` and
                reconstruct a ``StateMachine`` to get the discrete-time
                variant (T-118-followup-state-machine-time-mode).
            dt: Sample time in seconds. Required when
                ``time_mode="discrete"`` and disallowed otherwise.
        """
        if self._initial_state is None:
            raise ValueError("Must call set_initial_state() before build()")
        if not self._states:
            raise ValueError("Must add at least one state before build()")

        valid_modes = ("agnostic", "discrete")
        if time_mode not in valid_modes:
            raise ValueError(
                f"StateMachineBuilder.build: time_mode must be one of "
                f"{valid_modes!r}, got {time_mode!r}."
            )
        if time_mode == "discrete" and dt is None:
            raise ValueError(
                "StateMachineBuilder.build(time_mode='discrete', ...) requires dt=."
            )
        if time_mode != "discrete" and dt is not None:
            raise ValueError(
                f"StateMachineBuilder.build: dt= is only valid with "
                f"time_mode='discrete'; got time_mode={time_mode!r}."
            )

        input_names, output_names = self._collect_io_names()
        entry_actions = self._entry_actions(output_names)

        load_sm = self._to_model_json(entry_actions)

        from jaxonomy.dashboard.serialization import block_interface as sm_block

        sm_data = sm_block._create_state_machine_data(
            load_sm,
            input_names,
            output_names,
            False,
            # Agnostic (zero-crossing) guards/actions are evaluated under
            # JAX tracing; rewrite python and/or/not to trace-safe
            # logical_* calls there.
            trace_safe_bool_ops=(time_mode == "agnostic"),
        )

        from jaxonomy.library.state_machine import StateMachine

        return StateMachine(
            sm_data=sm_data,
            inputs=input_names,
            outputs=output_names,
            dt=dt,
            time_mode=time_mode,
            name=name,
            accelerate_with_jax=False,
        )
