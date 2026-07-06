# SPDX-License-Identifier: MIT

"""Variants / configurable diagrams.

The Variant DSL provides a *build-time* variant selector that picks one
of N sub-diagrams to instantiate. Unselected variants are never built,
so they never show up in the JIT trace -- the "active variant only"
code-generation behavior familiar from established block-diagram tools.

``T-111-followup-runtime-switch`` adds ``RuntimeVariantSubsystem``: a
SIMULATE-time switch between pre-built choices, driven by a discrete
selector input. This is the simulate-time counterpart to the build-time
"label-mode active variant" selector above. This is the
runtime counterpart to the build-time selector above; it pays the cost
of building all branches in exchange for the ability to swap between
them mid-simulation.

``T-111-followup-with-config`` adds the post-build configurator pattern:
``apply_variant_config(diagram, **overrides) -> Diagram`` (also exposed as
``Diagram.with_config(**overrides)``). Build the diagram once with
variants in their default state; later, swap variant choices by name to
produce a fresh diagram bound to a new configuration. This is useful when
the variant set is small and known up front, the diagram is built once
and re-configured between runs, and the configuration isn't tied to a
runtime selector signal.

Phase 1 explicitly avoids touching the `Diagram` / `DiagramBuilder` core
class hierarchy (T-118 and T-119 are running in parallel and edit the same
files). Everything here is additive: existing diagrams without variants
are unchanged.

Public API
==========

- ``Variant(choices=..., default=..., name=...)``: a thin frozen container
  describing N variant *choices*, each of which is a zero-argument builder
  callable that returns a fully-built ``Diagram`` (or, more generally, a
  ``SystemBase``). At least one choice must be marked as default so that
  resolving without a selection still produces a sensible diagram.

- ``select_variant(variant, name=None)``: build-time resolver. Returns the
  ``SystemBase`` produced by the named choice (or the default). The other
  choices are *never invoked*, so any heavy block construction (and any
  side effects like parameter declaration) inside their builders is
  skipped entirely.

- ``variant_subsystem(choices, name=None, default=None)``: a one-liner
  convenience that wraps ``Variant`` + ``select_variant`` for the common
  case where a caller already has a dict of pre-built diagrams (or
  builder callables) and just wants to pick one.

- ``RuntimeVariantSubsystem(choices, n_inputs=..., default_choice=...)``:
  a ``LeafSystem`` block that evaluates ALL branches every step (so each
  branch sees the same input trajectory and gradients flow correctly)
  but only exposes the selected branch's output. The selector is a
  discrete-valued input port -- it can change at runtime and the active
  branch follows. See its docstring for the full contract.

Phase 2+ (JSON round-trip, CLI helper, "all-variants" sweeps) is left
for follow-up work tracked under T-111.

Design notes
============

We deliberately keep ``Variant`` a tiny dataclass-style object rather than
a ``SystemBase`` subclass. A real ``Variant`` block (in the block-diagram
sense) would need to live in the registered-systems list and participate in port
wiring, which would force changes to ``DiagramBuilder.add`` /
``DiagramBuilder.build`` -- exactly the collision territory we agreed to
avoid in this phase. Build-time resolution sidesteps the question entirely:
by the time the diagram lands in the builder, the selection has already
collapsed to a single concrete sub-system.

The builders held by a ``Variant`` are zero-argument callables that
return a ``SystemBase``. They are typically ``lambda: build_pid()`` or
``lambda: build_lqr()`` where each ``build_*`` constructs and returns a
fresh ``Diagram``. Because we never call the unselected builders, any
``LeafSystem`` they would have created is never instantiated, and any
parameters they would have declared never enter the JIT trace.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping, Optional

import jax.numpy as jnp

from ..logging import logger
from .leaf_system import LeafSystem
from .system_base import SystemBase

__all__ = [
    "Variant",
    "VariantError",
    "select_variant",
    "variant_subsystem",
    "RuntimeVariantSubsystem",
    "apply_variant_config",
    "list_variants",
    "get_variant_choices",
    "get_active_variant",
    "VARIANT_METADATA_ATTR",
    # T-111 phase 2: JSON round-trip helpers.
    "dump_variant_config",
    "load_variant_config",
    "dump_variant_config_to_json",
    "load_variant_config_from_json",
    "apply_variant_config_from_dict",
    # T-111 phase 4: multi-variant resolution policies.
    "expand_all_variant_configs",
    "iter_variant_configurations",
]


# Sentinel attribute name used to tag a SystemBase produced by
# ``select_variant`` with the originating ``Variant`` (so post-build
# configurators like ``apply_variant_config`` can find / swap it). Kept
# as a public constant for users who need to introspect or strip the
# metadata.
VARIANT_METADATA_ATTR = "_variant_metadata"


class VariantError(ValueError):
    """Raised when a variant configuration is invalid or a selection is bad."""


# ---------------------------------------------------------------------------
# Variant container
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Variant:
    """A frozen description of N variant choices for build-time selection.

    Args:
        choices:
            Mapping from choice name to a zero-argument builder callable.
            Each callable, when invoked, must return a ``SystemBase``
            (typically a fully-built ``Diagram``). Unselected callables are
            never invoked.
        default:
            Name of the choice to use when ``select_variant`` is called
            without an explicit ``name``. Required (no implicit "first
            choice") so that adding a new variant later doesn't silently
            change the default. Must be a key in ``choices``.
        name:
            Optional human-readable label for diagnostics / logging. Does
            not affect resolution.

    Raises:
        VariantError: If ``choices`` is empty, ``default`` is not in
            ``choices``, or any choice is not callable.
    """

    choices: Mapping[str, Callable[[], SystemBase]]
    default: str
    name: Optional[str] = None

    def __post_init__(self):
        if not self.choices:
            raise VariantError(
                f"Variant {self.name!r}: choices must contain at least one entry"
            )
        for choice_name, builder in self.choices.items():
            if not callable(builder):
                raise VariantError(
                    f"Variant {self.name!r}: choice {choice_name!r} is not "
                    f"callable (got {type(builder).__name__}). Pass a "
                    f"zero-argument builder, e.g. ``lambda: build_pid()``."
                )
        if self.default not in self.choices:
            raise VariantError(
                f"Variant {self.name!r}: default {self.default!r} is not in "
                f"choices {list(self.choices)!r}"
            )

    @property
    def choice_names(self) -> tuple[str, ...]:
        """Stable tuple of available choice names (for introspection / CLI)."""
        return tuple(self.choices.keys())


# ---------------------------------------------------------------------------
# Build-time resolver
# ---------------------------------------------------------------------------


def select_variant(
    variant: Variant,
    name: Optional[str] = None,
) -> SystemBase:
    """Resolve a ``Variant`` at build time and return the active sub-system.

    Only the chosen builder is invoked; the others are never called. This
    matches the "active variant only" code-generation behavior familiar
    from established block-diagram tools -- nothing about the unselected
    branches enters the JIT trace, the parameter pytree, or the diagram's
    registered-systems list.

    Args:
        variant: The ``Variant`` to resolve.
        name:
            Name of the choice to activate. If ``None``, ``variant.default``
            is used.

    Returns:
        The ``SystemBase`` returned by the chosen builder.

    Raises:
        VariantError: If ``name`` is not one of ``variant.choices``, or if
            the chosen builder returns something that isn't a ``SystemBase``.
    """
    chosen = variant.default if name is None else name
    if chosen not in variant.choices:
        raise VariantError(
            f"Variant {variant.name!r}: unknown choice {chosen!r}; "
            f"available: {list(variant.choices)!r}"
        )
    builder = variant.choices[chosen]
    result = builder()
    if not isinstance(result, SystemBase):
        raise VariantError(
            f"Variant {variant.name!r}: choice {chosen!r} builder returned "
            f"{type(result).__name__}, expected a SystemBase (Diagram or LeafSystem)."
        )
    # T-111-followup-with-config: tag the resolved subsystem with its
    # originating variant so post-build configurators (apply_variant_config /
    # Diagram.with_config) can locate and swap it later. Tagging is a
    # no-op for the legacy / phase-1 path: nothing in the simulator,
    # context factory, or pytree machinery reads VARIANT_METADATA_ATTR;
    # it's purely a hint for the configurator walker.
    setattr(
        result,
        VARIANT_METADATA_ATTR,
        _VariantMetadata(variant=variant, active_choice=chosen),
    )
    return result


# ---------------------------------------------------------------------------
# One-liner convenience
# ---------------------------------------------------------------------------


def variant_subsystem(
    choices: Mapping[str, Callable[[], SystemBase]],
    name: Optional[str] = None,
    default: Optional[str] = None,
) -> Callable[..., SystemBase]:
    """Build a resolver closure for a one-shot variant point.

    Convenience wrapper around ``Variant`` + ``select_variant`` for the
    common case::

        controller = variant_subsystem(
            choices={
                "pid": lambda: build_pid(),
                "lqr": lambda: build_lqr(),
            },
            default="pid",
        )

        # Later, at "configure" time:
        active = controller(name="lqr")   # returns the lqr Diagram
        active = controller()             # returns the pid Diagram (default)

    Args:
        choices: See ``Variant.choices``.
        name: See ``Variant.name``.
        default:
            Choice name to use when the returned closure is called without
            an argument. If ``None``, the *first* key of ``choices`` is
            used (insertion order, which is guaranteed in Python 3.7+).

    Returns:
        A closure ``select(name=None) -> SystemBase`` that, on each call,
        resolves to a freshly-built sub-system for the named choice.
    """
    if not choices:
        raise VariantError(
            f"variant_subsystem {name!r}: choices must contain at least one entry"
        )
    chosen_default = default if default is not None else next(iter(choices))
    variant = Variant(choices=dict(choices), default=chosen_default, name=name)

    def _select(name: Optional[str] = None) -> SystemBase:
        return select_variant(variant, name=name)

    # Surface the underlying Variant for introspection / tests.
    _select.variant = variant  # type: ignore[attr-defined]
    return _select


# ---------------------------------------------------------------------------
# T-111-followup-with-config: post-build configurator
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _VariantMetadata:
    """Internal tag attached to a SystemBase produced by ``select_variant``.

    Records the originating ``Variant`` and the choice currently active
    on the tagged subsystem. ``apply_variant_config`` reads this tag to
    decide which subsystems to swap when reconfiguring a built diagram.
    """

    variant: Variant
    active_choice: str


def _iter_tagged(
    diagram,  # framework.Diagram (avoid import cycle)
):
    """Yield ``(parent_diagram, child_index, tag)`` for every variant-tagged
    direct child anywhere in the diagram tree.

    Walks ``diagram.nodes`` recursively. Children that are themselves
    Diagrams are descended into AFTER yielding their own tag (if any),
    so a Variant whose chosen branch is itself a Diagram can be swapped
    at the outer level without first reconfiguring the inner level.
    """
    # Local import to avoid an import cycle (diagram imports leaf_system,
    # which is imported here at module load time).
    from .diagram import Diagram

    if not isinstance(diagram, Diagram):
        return
    for idx, child in enumerate(diagram.nodes):
        tag = getattr(child, VARIANT_METADATA_ATTR, None)
        if isinstance(tag, _VariantMetadata):
            yield diagram, idx, tag
        if isinstance(child, Diagram):
            yield from _iter_tagged(child)


def apply_variant_config(diagram, **overrides):
    """Return a copy of ``diagram`` with named variants reconfigured.

    Walks the diagram tree, finds every subsystem that was produced by
    :func:`select_variant` from a named ``Variant``, and -- for each
    ``override_name=choice`` keyword -- replaces matching subsystems with
    a freshly-built copy from ``select_variant(variant, name=choice)``.
    All other diagram structure (non-variant blocks, connections,
    exported ports) is preserved.

    The original diagram is not modified.

    Example::

        builder = DiagramBuilder()
        ctrl = select_variant(controller_variant, name="pid")  # default
        plant = select_variant(plant_variant, name="lti")
        builder.add(ctrl)
        builder.add(plant)
        ...
        diagram = builder.build()

        # Reconfigure post-build:
        runtime_a = apply_variant_config(diagram, controller="pid", plant="lti")
        runtime_b = apply_variant_config(diagram, controller="lqr", plant="lti")

    Args:
        diagram: A built ``Diagram`` (typically the output of
            ``DiagramBuilder.build``).
        **overrides: Map from a variant's ``name`` (the ``name=`` kwarg
            passed to :class:`Variant`) to the choice name to activate.
            Variants whose name is not mentioned in ``overrides`` keep
            their currently-active choice.

    Returns:
        A new ``Diagram`` with the requested variant choices resolved.
        If ``overrides`` is empty, returns a structurally identical deep
        copy of ``diagram`` (the same default-off semantics as
        :meth:`Diagram.with_parameters` with no updates).

    Raises:
        VariantError: If an override name does not match any variant in
            the diagram, or if the requested choice is not in that
            variant's ``choices``.
    """
    # Local imports to avoid an import cycle at module load time.
    import copy as _copy
    from .diagram import (
        Diagram,
        _diagram_rewrite_child_refs,
        _diagram_refresh_exported_outputs_for_child,
        _diagram_rebuild_leaf_systems,
    )
    from .system_base import next_system_id

    if not isinstance(diagram, Diagram):
        raise VariantError(
            f"apply_variant_config: expected a Diagram, got "
            f"{type(diagram).__name__}."
        )

    new = _copy.deepcopy(diagram)
    new.system_id = next_system_id()
    new.parent = None
    new._dependency_graph = None
    new.feedthrough_pairs = None
    new._cache_update_events = None

    if not overrides:
        # Default-off: no overrides → identical-equivalent diagram.
        _diagram_rebuild_leaf_systems(new)
        return new

    # Index every tagged subsystem in the (deep-copied) tree by variant name.
    found_by_name: dict[str, list[tuple[Diagram, int, _VariantMetadata]]] = {}
    for parent_d, idx, tag in _iter_tagged(new):
        vname = tag.variant.name
        if vname is None:
            # Anonymous variant — can't be addressed by name. Skip; the
            # error path below will still fire for any unmatched override.
            continue
        found_by_name.setdefault(vname, []).append((parent_d, idx, tag))

    # Validate: every override must match at least one tagged variant.
    unknown = [k for k in overrides if k not in found_by_name]
    if unknown:
        raise VariantError(
            f"apply_variant_config: no Variant with name in {unknown!r} "
            f"found in diagram {diagram.name!r}. Available variant "
            f"names: {sorted(found_by_name)!r}. (Anonymous Variants "
            f"-- those built without a ``name=`` kwarg -- cannot be "
            f"addressed by ``apply_variant_config``.)"
        )

    # Apply each override.
    for vname, choice in overrides.items():
        for parent_d, idx, tag in found_by_name[vname]:
            if choice not in tag.variant.choices:
                raise VariantError(
                    f"apply_variant_config: variant {vname!r}: unknown "
                    f"choice {choice!r}; available: "
                    f"{list(tag.variant.choices)!r}"
                )
            old_child = parent_d.nodes[idx]
            # Build a fresh subsystem from the requested choice. This
            # re-tags the result with updated metadata, so subsequent
            # apply_variant_config calls keep working.
            repl = select_variant(tag.variant, name=choice)
            parent_d.nodes[idx] = repl
            repl.parent = parent_d
            _diagram_rewrite_child_refs(parent_d, old_child, repl)
            _diagram_refresh_exported_outputs_for_child(parent_d, repl)

    _diagram_rebuild_leaf_systems(new)
    return new


# ---------------------------------------------------------------------------
# T-111-followup-variant-introspection: discovery helpers
# ---------------------------------------------------------------------------
#
# These walk the diagram tree looking for ``select_variant``-produced
# subsystems (tagged via ``VARIANT_METADATA_ATTR``) and surface their
# metadata so user code -- e.g. a CLI ``variants list`` helper -- can
# enumerate the variant structure of a built or partially-built diagram
# without rebuilding it.
#
# Anonymous Variants (those built with no ``name=`` kwarg) are still
# returned by :func:`list_variants` (their ``name`` slot reads as
# ``None`` in the tuple), but they cannot be addressed by the
# name-keyed helpers :func:`get_variant_choices` /
# :func:`get_active_variant`. This mirrors the addressability semantics
# of :func:`apply_variant_config`.


def list_variants(diagram) -> list[tuple]:
    """List every variant point found in a (possibly nested) diagram.

    Walks ``diagram`` recursively and returns a metadata triple for
    every subsystem that was produced by :func:`select_variant`. Each
    triple has the shape ``(name, choice_names, active_choice)``:

    - ``name`` is the variant's human-readable label (``Variant.name``).
      ``None`` for anonymous Variants.
    - ``choice_names`` is the stable tuple of available choice names
      (``Variant.choice_names``).
    - ``active_choice`` is the name of the choice currently bound at
      this point in the diagram.

    Iteration order follows the diagram's tree-traversal order (parent
    before children, siblings in registration order). If the same
    ``Variant`` instance is reused at multiple points in the diagram,
    each occurrence yields its own entry — callers that want a deduped
    view should collapse on ``name``.

    Args:
        diagram: A built ``Diagram`` (typically the output of
            ``DiagramBuilder.build``) or any ``SystemBase``. Passing a
            non-Diagram subsystem returns ``[]`` (no children to walk;
            a tagged-leaf-as-root case is not produced by the current
            API surface, but the helper degrades gracefully).

    Returns:
        A list of ``(name, choice_names, active_choice)`` tuples. Empty
        if ``diagram`` contains no variant points (default-off path).
    """
    # Avoid an import cycle (diagram imports leaf_system, which is
    # imported here at module load time).
    from .diagram import Diagram

    if not isinstance(diagram, Diagram):
        return []

    out: list[tuple] = []
    for _parent, _idx, tag in _iter_tagged(diagram):
        out.append((tag.variant.name, tag.variant.choice_names, tag.active_choice))
    return out


def _find_first_tag_by_name(diagram, variant_name: str):
    """Return the first ``_VariantMetadata`` whose ``variant.name`` matches.

    Walks the diagram tree in the same order as :func:`list_variants`
    and stops at the first hit. Returns ``None`` if no match is found.
    """
    from .diagram import Diagram

    if not isinstance(diagram, Diagram):
        return None
    for _parent, _idx, tag in _iter_tagged(diagram):
        if tag.variant.name == variant_name:
            return tag
    return None


def get_variant_choices(diagram, variant_name: str) -> tuple:
    """Return the choice names of the named variant in ``diagram``.

    Args:
        diagram: A built ``Diagram``.
        variant_name: The human-readable label of the variant to look
            up (the ``name=`` kwarg passed to :class:`Variant`).

    Returns:
        A tuple of choice names (``Variant.choice_names``), in
        insertion order.

    Raises:
        VariantError: If no variant with the given name is found in the
            diagram. Anonymous Variants (built without ``name=``) are
            never matched and so cannot be queried via this helper.
    """
    tag = _find_first_tag_by_name(diagram, variant_name)
    if tag is None:
        available = sorted(
            {n for n, _, _ in list_variants(diagram) if n is not None}
        )
        raise VariantError(
            f"get_variant_choices: no Variant with name {variant_name!r} "
            f"found in diagram. Available variant names: {available!r}. "
            f"(Anonymous Variants -- those built without a ``name=`` "
            f"kwarg -- cannot be addressed by name.)"
        )
    return tag.variant.choice_names


def get_active_variant(diagram, variant_name: str):
    """Return the currently-selected choice for the named variant.

    Args:
        diagram: A built ``Diagram``.
        variant_name: The human-readable label of the variant to look
            up (the ``name=`` kwarg passed to :class:`Variant`).

    Returns:
        The name of the active choice (a string in
        ``Variant.choice_names``), or ``None`` if no variant with the
        given name is found in the diagram. The ``None`` sentinel lets
        CLI / introspection code treat "no such variant" as a soft
        miss; use :func:`get_variant_choices` if you want a hard error
        for unknown names.
    """
    tag = _find_first_tag_by_name(diagram, variant_name)
    if tag is None:
        return None
    return tag.active_choice


# ---------------------------------------------------------------------------
# T-111-followup-runtime-switch: RuntimeVariantSubsystem
# ---------------------------------------------------------------------------


class RuntimeVariantSubsystem(LeafSystem):
    """Switch between pre-built submodel choices via a discrete selector input.

    This is the runtime counterpart to ``select_variant`` / ``Variant``. Unlike
    the build-time selector (which never instantiates the unselected branches),
    ``RuntimeVariantSubsystem`` builds *every* choice and routes the selected
    branch's output through. The selector is a normal input port, so it can be
    driven by any discrete control signal in the diagram and the active branch
    follows at simulate time. This is the runtime-controlled variant pattern,
    as opposed to the label-mode build-time variant.

    Implementation: the block stacks all branches' outputs along a new leading
    axis and picks out the selected slice with integer indexing. This is the
    same mechanism used by ``MultiPortSwitch`` (T-118), reused here at the
    framework level so it does not pull a library dependency.

    Contract — "all branches integrated each step"
    -----------------------------------------------
    Because the underlying ``stack`` traces every branch, every choice's
    submodel runs on every step and sees the same input trajectory. The
    consequences:

    - Pure (memoryless) branches behave exactly as you'd expect: only the
      selected branch's output is exposed; gradients w.r.t. the active
      branch's parameters are non-zero, and gradients w.r.t. the others
      are zero (matching ``MultiPortSwitch``'s data-input semantics).

    - The selector is non-differentiable (``round`` + ``clip`` zero out
      its gradient), as expected for a control signal.

    - If a branch holds internal discrete state (e.g. a hold latch) the
      caller is responsible for supplying that state. ``RuntimeVariantSubsystem``
      itself is stateless; if you need stateful sub-Diagrams, hoist the
      state out, or build the runtime switch by composing ``MultiPortSwitch``
      (T-118) with N pre-built sub-diagrams in a parent ``DiagramBuilder``.

    All branches must return outputs that are broadcast-compatible (the
    stack op requires a common shape/dtype after broadcasting).

    Args:
        choices:
            Either a sequence of submodel callables ``[f0, f1, ..., f_{N-1}]``
            *or* a mapping ``{int: callable}`` (integer keys
            must be ``0..N-1``). Each callable has signature
            ``f(*inputs) -> output`` and must be JAX-traceable.
        n_inputs:
            Number of user inputs forwarded to every branch. Input port 0
            is always the selector; ports ``1..n_inputs`` are the user
            inputs. Defaults to 1.
        default_choice:
            Index of the choice used as the default. Stored for
            introspection / documentation; the runtime selector value
            still controls which branch is exposed each step. Defaults
            to 0.
        name:
            Optional block name.

    Input ports:
        (0) selector  — scalar integer-valued signal in ``[0, N-1]``.
            Floating values are rounded and clipped.
        (1..n_inputs) user inputs forwarded to every branch.

    Output ports:
        (0) The selected branch's output.

    Raises:
        VariantError:
            If ``choices`` is empty / non-callable / has bad keys, or if
            ``default_choice`` is out of range.
    """

    def __init__(
        self,
        choices,
        n_inputs: int = 1,
        default_choice: int = 0,
        **kwargs,
    ):
        super().__init__(**kwargs)

        # Normalize choices to an indexable tuple of callables.
        normalized = self._normalize_choices(choices, name=kwargs.get("name"))
        self._choices: tuple[Callable, ...] = normalized
        n_choices = len(normalized)

        if not (0 <= int(default_choice) < n_choices):
            raise VariantError(
                f"RuntimeVariantSubsystem {kwargs.get('name')!r}: "
                f"default_choice={default_choice} is out of range "
                f"[0, {n_choices - 1}]."
            )
        if int(n_inputs) < 0:
            raise VariantError(
                f"RuntimeVariantSubsystem {kwargs.get('name')!r}: "
                f"n_inputs must be >= 0, got {n_inputs}."
            )

        self._n_choices = n_choices
        self._n_inputs = int(n_inputs)
        self._default_choice = int(default_choice)

        # Port 0 = selector; ports 1..n_inputs forwarded to every branch.
        self.declare_input_port(name="selector")
        for i in range(self._n_inputs):
            self.declare_input_port(name=f"u_{i}")

        self.declare_output_port(
            self._compute_output,
            prerequisites_of_calc=[port.ticket for port in self.input_ports],
        )

    # ── helpers ────────────────────────────────────────────────────────────

    @staticmethod
    def _normalize_choices(choices, name=None) -> tuple[Callable, ...]:
        if isinstance(choices, Mapping):
            if not choices:
                raise VariantError(
                    f"RuntimeVariantSubsystem {name!r}: choices mapping is empty."
                )
            try:
                keys = sorted(int(k) for k in choices.keys())
            except (TypeError, ValueError) as exc:
                raise VariantError(
                    f"RuntimeVariantSubsystem {name!r}: choices mapping keys "
                    f"must be integers; got {list(choices.keys())!r}."
                ) from exc
            if keys != list(range(len(keys))):
                raise VariantError(
                    f"RuntimeVariantSubsystem {name!r}: choices mapping keys "
                    f"must be a contiguous 0..N-1 range; got {keys!r}."
                )
            ordered = tuple(choices[k] for k in keys)
        else:
            ordered = tuple(choices)
            if not ordered:
                raise VariantError(
                    f"RuntimeVariantSubsystem {name!r}: choices sequence is empty."
                )
        for i, fn in enumerate(ordered):
            if not callable(fn):
                raise VariantError(
                    f"RuntimeVariantSubsystem {name!r}: choice [{i}] is not "
                    f"callable (got {type(fn).__name__}). Pass a submodel "
                    f"function f(*inputs) -> output."
                )
        return ordered

    # ── output computation ────────────────────────────────────────────────

    def _compute_output(self, _time, _state, *inputs, **_params):
        selector = inputs[0]
        user_inputs = inputs[1 : 1 + self._n_inputs]
        # Evaluate every branch — this is the "all branches integrated each
        # step" contract. ``jnp.stack`` requires a common shape/dtype across
        # branch outputs.
        branch_outputs = [
            jnp.asarray(fn(*user_inputs)) for fn in self._choices
        ]
        stacked = jnp.stack(branch_outputs, axis=0)
        idx = jnp.clip(
            jnp.round(selector).astype(jnp.int32), 0, self._n_choices - 1
        )
        return stacked[idx]

    # ── introspection ─────────────────────────────────────────────────────

    @property
    def n_choices(self) -> int:
        """Number of variant choices held by this block."""
        return self._n_choices

    @property
    def default_choice(self) -> int:
        """Default choice index (documentary; runtime selector still rules)."""
        return self._default_choice


# ---------------------------------------------------------------------------
# Bind ``with_config`` onto Diagram.
# ---------------------------------------------------------------------------
#
# We attach ``with_config`` to ``Diagram`` from this module rather than
# editing ``diagram.py`` directly to keep the variant feature additive
# (per the file-ownership constraint in T-111). The free function
# ``apply_variant_config`` is the canonical API; ``Diagram.with_config``
# is a thin one-line wrapper that simply forwards to it.


def _diagram_with_config(self, **overrides):
    """Return a new diagram with named variants reconfigured.

    Thin instance-method wrapper around :func:`apply_variant_config`.
    See its docstring for the full contract. Implements the post-build
    configurator pattern: build the diagram once, then call
    ``.with_config(name=choice, ...)`` to bind a new variant configuration.
    """
    return apply_variant_config(self, **overrides)


def _install_with_config():
    """Attach ``with_config`` to the ``Diagram`` class once at import time."""
    from .diagram import Diagram as _Diagram

    if not hasattr(_Diagram, "with_config"):
        _Diagram.with_config = _diagram_with_config  # type: ignore[attr-defined]


_install_with_config()


# ---------------------------------------------------------------------------
# T-111 phase 2: JSON serialization round-trip
# ---------------------------------------------------------------------------
#
# Phase 1 shipped the Python-level Variant API. Phase 2 is about
# persisting *which choice is active* for each named variant in a
# diagram so that downstream tooling (model JSON, CLI helpers, CI
# reproducibility manifests) can capture and re-apply a variant
# configuration without re-running the diagram builder.
#
# What is NOT serialised: the builder callables themselves.  Variant
# choices are zero-arg Python callables and cannot round-trip through
# JSON in general.  The configuration captured here is the *binding*
# of a variant name to its active choice name — both are plain
# strings — so callers replay it against a freshly-built diagram of
# the same shape via ``apply_variant_config``.
#
# Anonymous Variants (those built without ``name=``) are skipped: they
# have no stable identifier across builds, so persisting their
# "active choice" would be ambiguous.  ``dump_variant_config`` will
# debug-log when it skips one (so the omission is debuggable).

def dump_variant_config(diagram) -> dict[str, str]:
    """Return ``{variant_name: active_choice}`` for every named variant in ``diagram``.

    Walks the diagram tree in the same order as :func:`list_variants`
    and collects the currently-active choice name for each variant
    whose ``Variant.name`` is set. Anonymous Variants are skipped
    (debug-logged so the omission is discoverable).

    When the same ``Variant.name`` appears at multiple points in the
    diagram (an unusual but supported pattern), every occurrence must
    already bind the same active choice — which they do under the
    :func:`apply_variant_config` invariant. The first occurrence's
    choice is the one persisted.

    Args:
        diagram: A built ``Diagram``.

    Returns:
        Dict mapping variant name to active choice name. Empty if
        ``diagram`` contains no named variant points.

    See also:
        :func:`load_variant_config` — apply a config dict back to a
            freshly-built diagram.
        :func:`dump_variant_config_to_json` — convenience wrapper that
            returns a JSON string for direct file persistence.
    """
    from .diagram import Diagram

    if not isinstance(diagram, Diagram):
        return {}

    config: dict[str, str] = {}
    for _parent, _idx, tag in _iter_tagged(diagram):
        name = tag.variant.name
        if name is None:
            logger.debug(
                "dump_variant_config: skipping anonymous Variant "
                "(active_choice=%r)",
                tag.active_choice,
            )
            continue
        if name not in config:
            config[name] = tag.active_choice
    return config


def load_variant_config(diagram, config: dict[str, str]):
    """Apply a previously-dumped variant configuration to ``diagram``.

    Thin wrapper around :func:`apply_variant_config` for the
    JSON-round-trip use case: takes the dict produced by
    :func:`dump_variant_config` and re-binds every named variant.

    Args:
        diagram: A built ``Diagram`` matching the structure of the
            diagram the config was dumped from.
        config: Mapping of ``{variant_name: active_choice}``. Variant
            names not present in the diagram raise ``VariantError``;
            extras in the diagram (not mentioned in ``config``) keep
            their current binding.

    Returns:
        A new ``Diagram`` with the requested variant choices applied.
        The original ``diagram`` is left untouched (per the
        :func:`apply_variant_config` contract).

    Raises:
        VariantError: If a key in ``config`` is not a Variant name in
            the diagram, or if a value is not one of the available
            choices for that Variant.
    """
    if not config:
        # Nothing to apply — return the input untouched. apply_variant_config
        # with no kwargs would also work but allocates an unnecessary copy.
        return diagram
    return apply_variant_config(diagram, **config)


# Backwards-compatibility alias: ``apply_variant_config_from_dict`` was
# the originally-considered name.
apply_variant_config_from_dict = load_variant_config


def dump_variant_config_to_json(diagram, *, indent: Optional[int] = 2) -> str:
    """Dump the variant configuration of ``diagram`` as a JSON string.

    Args:
        diagram: A built ``Diagram``.
        indent: JSON ``indent`` value (default 2 for diff-friendly
            output; pass ``None`` for the compact one-line form).

    Returns:
        JSON object string mapping variant names to active choice
        names. Empty object ``"{}"`` if no named variants are present.
    """
    import json

    return json.dumps(dump_variant_config(diagram), indent=indent, sort_keys=True)


def load_variant_config_from_json(diagram, json_str: str):
    """Apply a variant configuration loaded from a JSON string.

    Convenience wrapper around :func:`load_variant_config` that parses
    ``json_str`` first.

    Args:
        diagram: A built ``Diagram``.
        json_str: JSON object string in the shape produced by
            :func:`dump_variant_config_to_json`.

    Returns:
        A new ``Diagram`` with the requested variant choices applied.

    Raises:
        ValueError: If ``json_str`` is not a JSON object.
        VariantError: See :func:`load_variant_config`.
    """
    import json

    config = json.loads(json_str)
    if not isinstance(config, dict):
        raise ValueError(
            "load_variant_config_from_json: expected a JSON object "
            f"({{...}}) at the top level; got {type(config).__name__}."
        )
    return load_variant_config(diagram, config)


# ---------------------------------------------------------------------------
# T-111 phase 4: multi-variant resolution policies
# ---------------------------------------------------------------------------
#
# A diagram with N named variants {V_1: M_1 choices, ..., V_N: M_N
# choices} encodes ``∏ M_i`` distinct configurations.  Phase 4 surfaces
# the obvious "all configurations" expansion as a first-class helper,
# so parameter-sweep / dispersion-study callers don't have to roll
# their own Cartesian-product loop and risk drift from
# :func:`list_variants` ordering.
#
# Two flavours:
#
#   * ``expand_all_variant_configs(diagram)`` returns a *list of
#     dicts*, each suitable as the second argument to
#     :func:`load_variant_config`.  Cheap; the caller decides what to
#     do with each binding (apply, simulate, dispatch over an
#     executor pool).
#
#   * ``iter_variant_configurations(diagram)`` is a generator yielding
#     ``(config_dict, configured_diagram)`` pairs — applies each
#     config via :func:`load_variant_config` lazily, so callers that
#     want a fresh diagram per binding don't have to thread the
#     loader call themselves.
#
# Anonymous Variants (no ``name=``) are skipped, matching the
# :func:`dump_variant_config` policy — they have no stable identifier
# to bind in the resulting config dict.


def expand_all_variant_configs(diagram) -> "list[dict[str, str]]":
    """Enumerate every named-variant configuration of ``diagram``.

    Walks the diagram once to collect the named variant points + their
    available choices, then returns the full Cartesian product of
    ``{variant_name: choice_name}`` bindings as a list of dicts.

    For a diagram with N named variants of sizes ``M_1, ..., M_N`` the
    returned list has length ``∏ M_i``. Result ordering: outer-most
    variant in :func:`list_variants` order; within each variant,
    choices in ``Variant.choice_names`` (insertion) order. This makes
    the expansion deterministic and easy to diff across builds.

    Args:
        diagram: A built ``Diagram``.

    Returns:
        A list of dicts. Each dict maps ``variant_name`` to a chosen
        ``active_choice``. An empty diagram (no named variants)
        returns ``[{}]`` — the single trivial configuration (no
        bindings to make), matching the convention of an empty
        Cartesian product.

    See also:
        :func:`iter_variant_configurations` — the generator form that
            also yields the configured diagram for each binding.
    """
    from itertools import product
    from .diagram import Diagram

    if not isinstance(diagram, Diagram):
        return [{}]

    # Collect named variants in tree-traversal order; deduplicate on
    # name so a Variant reused at multiple points contributes one axis
    # to the product.
    seen: dict[str, tuple[str, ...]] = {}
    for _parent, _idx, tag in _iter_tagged(diagram):
        name = tag.variant.name
        if name is None:
            continue
        if name not in seen:
            seen[name] = tag.variant.choice_names

    if not seen:
        return [{}]

    names = list(seen.keys())
    choice_axes = [seen[n] for n in names]
    return [
        dict(zip(names, combo)) for combo in product(*choice_axes)
    ]


def iter_variant_configurations(diagram):
    """Yield ``(config, configured_diagram)`` for every configuration.

    Generator counterpart to :func:`expand_all_variant_configs`.
    Applies each config via :func:`load_variant_config` lazily so the
    caller doesn't have to thread the loader through their own loop.

    Useful pattern: a parameter-sweep that wants to simulate every
    variant configuration once

    .. code-block:: python

        for config, diag in iter_variant_configurations(root):
            ctx = diag.create_context()
            results[tuple(sorted(config.items()))] = jaxonomy.simulate(
                diag, ctx, (0.0, t_final)
            )

    Args:
        diagram: A built ``Diagram``.

    Yields:
        Tuples of ``(config_dict, configured_diagram)``. ``config_dict``
        is the binding that produced ``configured_diagram`` — exactly
        what :func:`dump_variant_config` would return on it.
    """
    for cfg in expand_all_variant_configs(diagram):
        yield cfg, load_variant_config(diagram, cfg)
