# SPDX-License-Identifier: MIT

"""Builder class for constructing block diagrams."""

from __future__ import annotations

import warnings
from typing import Tuple, List, Mapping, TYPE_CHECKING, overload

from ..backend import numpy_api as npa
from ..logging import logger

from . import build_recorder
from .system_base import SystemBase
from .diagram import Diagram
from .error import InputNotConnectedError, StaticError
from .parameter import Parameter
from .units import (
    assert_unit_compatible,
    assert_units_compatible_with_scale,
    resolve_unit,
)

if TYPE_CHECKING:
    from .port import (
        InputPort,
        OutputPort,
        InputPortLocator,
        OutputPortLocator,
    )

    ExportedInputData = Tuple[InputPortLocator, str]  # (locator, port_name)

__all__ = [
    "DiagramBuilder",
]


class BuilderError(StaticError):
    """Errors related to constructing diagrams."""

    pass


def _install_unit_conversion(dest_port, factor: float) -> None:
    """Wrap a destination input port's evaluation callback to multiply
    the upstream value by ``factor``.

    Used by :meth:`DiagramBuilder.connect` under
    ``unit_conversion="auto"`` / ``"warn"`` when the source and
    destination units share base-dim exponents but differ by a scalar
    ``scale``.

    The wrapping preserves the dependency-tracking metadata on the port
    (``prerequisites_of_calc`` etc.) — we only intercept the cached
    ``_callback``.  Multiple chained conversions on the same port are
    additive (the factors compose) because each call wraps the previously
    installed callback.
    """
    original_callback = dest_port._callback
    factor_value = float(factor)

    def _scaled_callback(context):
        value = original_callback(context)
        # Use the active numpy backend so the multiplication remains
        # JAX-traceable / vmap-safe.  Using a Python float on the right
        # avoids dtype promotion surprises for integer-typed signals
        # (the multiplication will promote to floating-point only when
        # the upstream value already is).
        return npa.asarray(value) * factor_value

    dest_port._callback = _scaled_callback
    # Track the cumulative conversion factor on the port for diagnostics
    # / introspection. Multiplicative so chained connects compose.
    prior = getattr(dest_port, "_unit_conversion_factor", 1.0)
    dest_port._unit_conversion_factor = prior * factor_value


class SystemNameNotUniqueError(BuilderError):
    def __init__(self, system: SystemBase):
        super().__init__(f"System name {system.name} is not unique", system=system)


class DisconnectedInputError(InputNotConnectedError, BuilderError):
    def __init__(self, input_port_locator: InputPortLocator):
        system, port_index = input_port_locator
        super().__init__(
            f"Input port {system.name}[{port_index}] is not connected",
            system=system,
            port_index=port_index,
            port_direction="in",
        )


class EmptyDiagramError(BuilderError):
    def __init__(self, name: str):
        super().__init__(f"Cannot compile an empty diagram: {name}")


class DiagramBuilder:
    """Class for constructing block diagram systems.

    The `DiagramBuilder` class is responsible for building a diagram by adding systems, connecting ports,
    and exporting inputs and outputs. It keeps track of the registered systems, input and output ports,
    and the connection map between input and output ports of the child systems.
    """

    def __init__(
        self,
        *,
        validate_rates_at_connect: str | bool | None = None,
        unit_conversion: str = "auto",
        auto_insert_rate_transitions: bool = False,
    ):
        """Construct a DiagramBuilder.

        Args:
            validate_rates_at_connect: T-105 Phase 2 — opt-in connect-time
                multirate consistency check.  When set to ``"warn"`` (or
                ``True``, treated as ``"warn"``), each :meth:`connect`
                call routes through
                :func:`jaxonomy.simulation.rate_groups.check_connection_rate_compat`
                and emits a :class:`RateMismatchWarning` for adjacent
                blocks of incompatible discrete rates.  ``"error"``
                raises :class:`RateMismatchError` on the first offender.
                Default ``None`` keeps the legacy path completely off
                (byte-equivalent to the pre-T-105-Phase-2 behaviour).
            unit_conversion: T-104 followup — controls behaviour when two
                connected ports share base-dimensions but differ only by a
                scalar ``scale`` (e.g. ``meter`` vs ``kilometer``):
                  * ``"auto"`` (default) silently inserts the conversion
                    factor on the destination input port;
                  * ``"warn"`` inserts the factor and emits a
                    :class:`UserWarning`;
                  * ``"error"`` refuses the connection (preserves the
                    Phase-1 strict-equal behaviour).
                Genuine dimensional mismatches (e.g. ``meter`` vs
                ``second``) always raise regardless of mode.
            auto_insert_rate_transitions: T-105-followup-phase3 — when
                ``True``, :meth:`connect` automatically synthesises a
                :func:`jaxonomy.library.RateTransition` block (a
                ``ZeroOrderHold`` for slow→fast, a ``Decimator`` for
                fast→slow) between any two adjacent leaves whose
                inferred discrete sample times differ.  The rewritten
                wiring is ``src → rate_transition → dst`` and an
                informational log line documents the insertion.
                Composes with ``validate_rates_at_connect``: when both
                are enabled, the warning still fires and the transition
                still gets inserted.  Default ``False`` keeps the legacy
                code path byte-equivalent (the strict mode that surfaces
                rate mismatches rather than silently inserting transitions).
        """
        # Child input ports that are exported as diagram-level inputs
        self._input_port_ids: List[InputPortLocator] = []
        self._input_port_names: List[str] = []
        # Child output ports that are exported as diagram-level outputs
        self._output_port_ids: List[OutputPortLocator] = []
        self._output_port_names: List[str] = []

        # Connection map between input and output ports of the child systems
        self._connection_map: Mapping[InputPortLocator, OutputPortLocator] = {}

        # List of registered systems
        self._registered_systems: List[SystemBase] = []

        # Name lookup for input ports
        self._diagram_input_indices: Mapping[str, InputPortLocator] = {}

        # All input ports of child systems (for use in ensuring proper connectivity)
        self._all_input_ports: List[InputPortLocator] = []

        # Each DiagramBuilder can only be used to build a single diagram.  This is to
        # avoid creating multiple diagrams that reference the same LeafSystem. Doing so
        # may or may not actually lead to problems, since the LeafSystems themselves
        # should act like a collection of pure functions, but best practice is to have
        # each leaf system be fully unique.
        self._already_built = False
        self._built_as_name = None

        # T-105 Phase 2: normalise the connect-time-rate-validation flag
        # to the same string vocabulary that ``check_connection_rate_compat``
        # understands.  ``None`` means "off" (byte-equivalent default);
        # ``True`` is sugar for ``"warn"``.
        if validate_rates_at_connect is True:
            self._validate_rates_at_connect: str | None = "warn"
        elif validate_rates_at_connect in (None, False):
            self._validate_rates_at_connect = None
        elif validate_rates_at_connect in ("warn", "error"):
            self._validate_rates_at_connect = validate_rates_at_connect
        else:
            raise BuilderError(
                "validate_rates_at_connect must be None, True/False, "
                "'warn', or 'error'; got "
                f"{validate_rates_at_connect!r}"
            )

        # T-104 followup: unit-conversion mode.  Validated up front so
        # typos surface at construction time rather than from the first
        # ``connect`` call.
        if unit_conversion not in ("auto", "warn", "error"):
            raise BuilderError(
                "unit_conversion must be 'auto', 'warn', or 'error'; "
                f"got {unit_conversion!r}"
            )
        self._unit_conversion = unit_conversion

        # T-105-followup-phase3: opt-in auto-insertion of RateTransition
        # blocks at connect time.  Default ``False`` preserves the legacy
        # behaviour (byte-equivalent).  When enabled, ``connect`` will
        # synthesise a RateTransition block between any two adjacent
        # leaves whose inferred discrete sample times differ.  The
        # counter below gives auto-inserted blocks unique names without
        # leaking into the user-visible name space when the flag is off.
        self._auto_insert_rate_transitions = bool(auto_insert_rate_transitions)
        self._auto_rate_transition_counter = 0

    @overload
    def add(self, system: SystemBase) -> SystemBase: ...

    @overload
    def add(self, system: SystemBase, *systems: SystemBase) -> List[SystemBase]: ...

    def add(self, *systems: SystemBase) -> List[SystemBase] | SystemBase:
        """Add one or more systems to the diagram.

        Args:
            *systems SystemBase:
                System(s) to add to the diagram.

        Returns:
            List[SystemBase] | SystemBase:
                The added system(s). Will return a single system if there is only
                a single system in the argument list.

        Raises:
            BuilderError: If the diagram has already been built.
            BuilderError: If the system is already registered.
            BuilderError: If the system name is not unique.
        """
        for system in systems:
            self._check_not_already_built()
            self._check_system_not_registered(system)
            self._check_system_name_is_unique(system)
            self._registered_systems.append(system)

            # Add the system's input ports to the list of all input ports
            # So that we can make sure they're all connected before building.
            self._all_input_ports.extend([port.locator for port in system.input_ports])

            logger.debug("Added system %s to DiagramBuilder", system.name)
            logger.debug(
                "    Registered systems: %s",
                [s.name for s in self._registered_systems],
            )
        build_recorder.add_block(self, systems)

        return systems[0] if len(systems) == 1 else systems

    def connect(self, src: OutputPort, dest: InputPort):
        """Connect an output port to an input port.

        The input port and output port must both belong to systems that have
        already been added to the diagram.  The input port must not already be
        connected to another output port.

        Args:
            src (OutputPort): The output port to connect.
            dest (InputPort): The input port to connect.

        Raises:
            BuilderError: If the diagram has already been built.
            BuilderError: If the source system is not registered.
            BuilderError: If the destination system is not registered.
            BuilderError: If the input port is already connected.
            BuilderError: If src is an InputPort or dest is an OutputPort.
        """
        # Local imports to avoid a hard import cycle at module load.
        from .port import InputPort as _InputPort, OutputPort as _OutputPort

        # Direction validation -- catches the common (input, input) /
        # (output, output) miswiring at connect time rather than letting it
        # surface as an opaque "input not connected" error during simulation.
        src_is_input = isinstance(src, _InputPort)
        dest_is_output = isinstance(dest, _OutputPort)
        if src_is_input and isinstance(dest, _InputPort):
            raise BuilderError(
                f"Cannot connect input-to-input: "
                f"'{src.system.name}.in[{src.index}]' -> "
                f"'{dest.system.name}.in[{dest.index}]'. "
                f"The first argument must be an output port."
            )
        if isinstance(src, _OutputPort) and dest_is_output:
            raise BuilderError(
                f"Cannot connect output-to-output: "
                f"'{src.system.name}.out[{src.index}]' -> "
                f"'{dest.system.name}.out[{dest.index}]'. "
                f"The second argument must be an input port."
            )
        if src_is_input:
            raise BuilderError(
                f"connect() expected an OutputPort as the first argument, got "
                f"InputPort '{src.system.name}.in[{src.index}]'."
            )
        if dest_is_output:
            raise BuilderError(
                f"connect() expected an InputPort as the second argument, got "
                f"OutputPort '{dest.system.name}.out[{dest.index}]'."
            )

        self._check_not_already_built()
        self._check_system_is_registered(src.system)
        self._check_system_is_registered(dest.system)
        self._check_input_not_connected(dest.locator)

        # T-104 phase 1 / followup: connect-time unit consistency check.
        # Ports that never declared a unit are treated as dimensionless and
        # connect to anything (default-off byte-equivalence).  When both
        # sides declare units:
        #   * dimensional mismatch (e.g. m vs s) -> always UnitMismatchError;
        #   * scalar-scale mismatch (e.g. m vs km):
        #       - "error": raise (Phase-1 strict behaviour);
        #       - "warn":  apply factor + emit UserWarning;
        #       - "auto":  silently apply factor.
        src_units = getattr(src, "units", None)
        dst_units = getattr(dest, "units", None)
        src_label = f"'{src.system.name}.out[{src.index}]' ({src.name})"
        dst_label = f"'{dest.system.name}.in[{dest.index}]' ({dest.name})"

        if self._unit_conversion == "error":
            # Preserve the Phase-1 strict-equal behaviour.
            assert_unit_compatible(
                src_units,
                dst_units,
                src_label=src_label,
                dst_label=dst_label,
            )
        else:
            # Returns the multiplicative factor (1.0 for matched / wildcard
            # units, src.scale / dst.scale otherwise).  Raises on genuine
            # dimensional mismatch.
            factor = assert_units_compatible_with_scale(
                src_units,
                dst_units,
                src_label=src_label,
                dst_label=dst_label,
            )
            if factor != 1.0:
                src_u = resolve_unit(src_units)
                dst_u = resolve_unit(dst_units)
                msg = (
                    f"Unit conversion: applying factor {factor!r} to "
                    f"connection {src_label} ({src_u!r}) -> "
                    f"{dst_label} ({dst_u!r})."
                )
                if self._unit_conversion == "warn":
                    warnings.warn(msg, UserWarning, stacklevel=2)
                else:
                    # "auto": informational log line; no warning.
                    logger.info(msg)
                _install_unit_conversion(dest, factor)

        # T-105 Phase 2: opt-in connect-time multirate consistency check.
        # Default-off (``self._validate_rates_at_connect is None``) keeps
        # the legacy code path byte-equivalent.  When the builder was
        # constructed with ``validate_rates_at_connect=...``, route the
        # source/dest pair through ``check_connection_rate_compat`` which
        # honours the ``_jaxonomy_rate_transition`` marker (T-123) and
        # the universal-sample-time rule (constant / inherited bridge
        # any rates).
        if self._validate_rates_at_connect is not None:
            # Local import: avoids a hard import cycle between
            # ``framework`` and ``simulation`` at module load.
            from ..simulation.rate_groups import check_connection_rate_compat

            check_connection_rate_compat(
                src.system,
                src.index,
                dest.system,
                dest.index,
                on_mismatch=self._validate_rates_at_connect,
            )

        # T-105-followup-phase3: opt-in connect-time auto-insertion of
        # RateTransition blocks between adjacent leaves with differing
        # discrete sample times.  Default-off keeps the legacy code
        # path byte-equivalent.  Composes with the validate hook above:
        # when both are enabled, the warning still fires (above) and
        # the bridge still gets inserted (here).
        if self._auto_insert_rate_transitions:
            inserted = self._maybe_auto_insert_rate_transition(src, dest)
            if inserted is not None:
                # ``_maybe_auto_insert_rate_transition`` already wrote
                # both legs of ``src → bridge → dest`` into the
                # connection map (and recorded both connections with
                # the build recorder), so we are done here.
                return

        build_recorder.connect_ports(self, src, dest)

        self._connection_map[dest.locator] = src.locator

        logger.debug(
            f"Connected port {src.name} of system {src.system.name} to port {dest.name} of system {dest.system.name}"
        )

    def _maybe_auto_insert_rate_transition(self, src, dest):
        """T-105-followup-phase3 helper.

        Inspect the inferred sample times of ``src.system`` and
        ``dest.system``; if they are both discrete with different
        periods, synthesise a :func:`jaxonomy.library.RateTransition`
        block, register it with the builder, and rewrite the connection
        as ``src → rate_transition → dest``.

        Returns the freshly added bridge block, or ``None`` if no
        insertion was appropriate (matched rates, universal source/dest,
        or either side already a rate-transition bridge).
        """
        # Local imports: avoid hard import cycles between framework and
        # simulation/library at module load.
        from ..simulation.rate_groups import infer_block_sample_time

        # If either side is already a rate-transition bridge, do nothing
        # — the user (or a previous auto-insertion) has already handled
        # the transition.
        if getattr(src.system, "_jaxonomy_rate_transition", False):
            return None
        if getattr(dest.system, "_jaxonomy_rate_transition", False):
            return None

        src_st = infer_block_sample_time(src.system)
        dst_st = infer_block_sample_time(dest.system)

        # Only auto-insert when both sides are discrete with different
        # periods.  Universal sample times (constant/inherited) and
        # continuous-to-continuous matches do not need a bridge.
        if not (src_st.is_discrete() and dst_st.is_discrete()):
            return None
        if src_st.matches(dst_st):
            return None

        from ..library import RateTransition

        self._auto_rate_transition_counter += 1
        bridge_name = (
            f"_auto_rate_transition_{self._auto_rate_transition_counter}_"
            f"{src.system.name}_to_{dest.system.name}"
        )
        bridge = RateTransition(
            input_dt=src_st.period,
            output_dt=dst_st.period,
            name=bridge_name,
        )
        # Belt-and-suspenders: ``RateTransition`` already tags the
        # returned block with ``_jaxonomy_rate_transition = True`` on
        # the slow→fast (ZOH) and fast→slow (Decimator) branches.  Set
        # it again here so future auto-insertion calls definitely skip
        # this block even if the factory's tagging changes.
        bridge._jaxonomy_rate_transition = True

        self.add(bridge)

        logger.info(
            "T-105-followup-phase3 auto-inserted RateTransition '%s' "
            "between '%s.out[%d]' (dt=%s) and '%s.in[%d]' (dt=%s).",
            bridge_name,
            src.system.name,
            src.index,
            src_st.period,
            dest.system.name,
            dest.index,
            dst_st.period,
        )

        # Wire ``src → bridge.in[0]`` and ``bridge.out[0] → dest``.
        # We bypass the public ``connect`` method on these two legs to
        # avoid recursing into auto-insertion (the bridge is tagged so
        # it would short-circuit anyway, but going through the recorder
        # / connection-map directly is simpler and matches the
        # documented invariant that the auto-inserted block is exactly
        # one bridge between the two original ports).
        bridge_in = bridge.input_ports[0]
        bridge_out = bridge.output_ports[0]

        build_recorder.connect_ports(self, src, bridge_in)
        self._connection_map[bridge_in.locator] = src.locator
        logger.debug(
            f"Connected port {src.name} of system {src.system.name} to port {bridge_in.name} of system {bridge.name}"
        )

        build_recorder.connect_ports(self, bridge_out, dest)
        self._connection_map[dest.locator] = bridge_out.locator
        logger.debug(
            f"Connected port {bridge_out.name} of system {bridge.name} to port {dest.name} of system {dest.system.name}"
        )

        return bridge

    def export_input(self, port: InputPort, name: str = None) -> int:
        """Export an input port of a child system as a diagram-level input.

        The input port must belong to a system that has already been added to the
        diagram. The input port must not already be connected to another output port.

        Args:
            port (InputPort): The input port to export.
            name (str, optional):
                The name to assign to the exported input port. If not provided, a
                unique name will be generated.

        Returns:
            int: The index (in the to-be-built diagram) of the exported input port.

        Raises:
            BuilderError: If the diagram has already been built.
            BuilderError: If the system is not registered.
            BuilderError: If the input port is already connected.
            BuilderError: If the input port name is not unique.
        """
        self._check_not_already_built()
        self._check_system_is_registered(port.system)
        self._check_input_not_connected(port.locator)

        if name is None:
            # Since the system names are unique, auto-generated port names are also unique
            # at the level of _this_ diagram (subsystems can have ports with the same name)
            name = f"{port.system.name}_{port.name}"
        elif name in self._diagram_input_indices:
            raise BuilderError(
                f"Input port name {name} is not unique",
                system=port.system,
                port_index=port.index,
                port_direction="in",
            )

        # Index at the diagram (not subsystem) level
        port_index = len(self._input_port_ids)
        self._input_port_ids.append(port.locator)
        self._input_port_names.append(name)

        self._diagram_input_indices[name] = port_index

        build_recorder.export_port(self, port.system, "input", port.index, name)

        return port_index

    def export_output(self, port: OutputPort, name: str = None) -> int:
        """Export an output port of a child system as a diagram-level output.

        The output port must belong to a system that has already been added to the
        diagram.

        Args:
            port (OutputPort): The output port to export.
            name (str, optional):
                The name to assign to the exported output port. If not provided, a
                unique name will be generated.

        Returns:
            int: The index (in the to-be-built diagram) of the exported output port.

        Raises:
            BuilderError: If the diagram has already been built.
            BuilderError: If the system is not registered.
            BuilderError: If the output port name is not unique.
        """
        self._check_not_already_built()
        self._check_system_is_registered(port.system)

        if name is None:
            # Since the system names are unique, auto-generated port names are also unique
            # at the level of _this_ diagram (subsystems can have ports with the same name)
            name = f"{port.system.name}_{port.name}"
        elif name in self._output_port_names:
            raise BuilderError(
                f"Output port name {name} is not unique",
                system=port.system,
                port_index=port.index,
                port_direction="out",
            )

        # Index at the diagram (not subsystem) level
        port_index = len(self._output_port_ids)
        self._output_port_ids.append(port.locator)
        self._output_port_names.append(name)

        build_recorder.export_port(self, port.system, "output", port.index, name)

        return port_index

    def _check_not_already_built(self):
        if self._already_built:
            raise BuilderError(
                "DiagramBuilder: build has already been called to "
                "create a diagram; this DiagramBuilder may no longer be used: "
                f"{self._built_as_name}"
            )

    def _check_system_name_is_unique(self, system: SystemBase):
        if system.name in map(lambda s: s.name, self._registered_systems):
            raise SystemNameNotUniqueError(system)

    def _system_is_registered(self, system: SystemBase) -> bool:
        # return (system is not None) and (system in self._registered_systems)
        if system.system_id is None:  # system.__init__ is not done yet
            return False
        return system.system_id in map(lambda s: s.system_id, self._registered_systems)

    def _check_system_not_registered(self, system: SystemBase):
        if self._system_is_registered(system):
            raise BuilderError(
                f"System {system.name} is already registered",
                system=system,
            )

    def _check_system_is_registered(self, system: SystemBase):
        if not self._system_is_registered(system):
            raise BuilderError(
                f"System {system.name} is not registered",
                system=system,
            )

    def _check_input_not_connected(self, input_port_locator: InputPortLocator):
        if not (
            (input_port_locator not in self._input_port_ids)
            and (input_port_locator not in self._connection_map)
        ):
            system, port_index = input_port_locator
            raise BuilderError(
                f"Input port {port_index} for {system} is already connected",
                system=system,
                port_index=port_index,
                port_direction="in",
            )

    def _check_input_is_connected(self, input_port_locator: InputPortLocator):
        if not (
            (input_port_locator in self._input_port_ids)
            or (input_port_locator in self._connection_map)
        ):
            raise DisconnectedInputError(input_port_locator)

    def _check_contents_are_complete(self):
        # Make sure all the systems referenced in the builder attributes are registered

        # Check that systems and registered_systems have the same elements
        for system in self._registered_systems:
            self._check_system_is_registered(system)

        # Check that connection_map only refers to registered systems
        for (
            input_port_locator,
            output_port_locator,
        ) in self._connection_map.items():
            self._check_system_is_registered(input_port_locator[0])
            self._check_system_is_registered(output_port_locator[0])

        # Check that input_port_ids and output_port_ids only refer to registered systems
        for port_locator in [*self._input_port_ids, *self._output_port_ids]:
            self._check_system_is_registered(port_locator[0])

    def _check_ports_are_valid(self):
        for dst, src in self._connection_map.items():
            dst_sys, dst_idx = dst
            if (dst_idx < 0) or (dst_idx >= dst_sys.num_input_ports):
                raise BuilderError(
                    f"Input port index {dst_idx} is out of range "
                    f"(0-{dst_sys.num_input_ports-1})",
                    system=dst_sys,
                    port_index=dst_idx,
                    port_direction="in",
                )
            src_sys, src_idx = src
            if (src_idx < 0) or (src_idx >= src_sys.num_output_ports):
                raise BuilderError(
                    f"Output port index {src_idx} is out of range "
                    f"(0-{src_sys.num_output_ports-1})",
                    system=src_sys,
                    port_index=src_idx,
                    port_direction="out",
                )

    def build(
        self,
        name: str = "root",
        ui_id: str = None,
        parameters: dict[str, Parameter] = None,
    ) -> Diagram:
        """Builds a Diagram system with the specified name and system ID.

        Args:
            name (str, optional): The name of the diagram. Defaults to "root".
            ui_id (str, optional): The unique identifier for the diagram.
            parameters (dict[str, Parameter], optional):
                A dictionary of dynamic parameters to declare for the diagram.

        Returns:
            Diagram: The newly constructed diagram.

        Raises:
            EmptyDiagramError: If no systems are registered in the diagram.
            BuilderError: If the diagram has already been built.
            AlgebraicLoopError: If an algebraic loop is detected in the diagram.
            DisconnectedInputError: If an input port is not connected.
        """
        self._check_not_already_built()
        self._check_contents_are_complete()
        self._check_ports_are_valid()

        # Check that all internal input ports are connected
        for input_port_locator in self._input_port_ids:
            self._check_input_is_connected(input_port_locator)

        if len(self._registered_systems) == 0:
            raise EmptyDiagramError(name)

        diagram = Diagram(
            nodes=self._registered_systems,
            name=name,
            connection_map=self._connection_map,
            ui_id=ui_id,
        )

        build_recorder.build_diagram(self, diagram, parameters)

        if parameters:
            for name, parameter in parameters.items():
                diagram.declare_dynamic_parameter(name, parameter)
                diagram.instance_parameters.add(name)

        # Export diagram-level inputs
        for locator, port_name in zip(self._input_port_ids, self._input_port_names):
            diagram.export_input(locator, port_name)

        # Export diagram-level outputs
        assert len(self._output_port_ids) == len(self._output_port_names)
        for locator, port_name in zip(self._output_port_ids, self._output_port_names):
            diagram.export_output(locator, port_name)

        self._already_built = True  # Prevent further use of this builder
        self._built_as_name = name
        return diagram
