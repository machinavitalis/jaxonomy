# SPDX-License-Identifier: MIT
"""
T-025a — :class:`JaxonomyDiagramSlave` base class for binding a Jaxonomy
:class:`~jaxonomy.framework.diagram.Diagram` into a
:class:`pythonfmu.Fmi2Slave`.

Usage in a user-authored slave script::

    from jaxonomy.library.fmu_slave import JaxonomyDiagramSlave
    import jaxonomy
    from jaxonomy.library import Constant

    def build_diagram():
        bld = jaxonomy.DiagramBuilder()
        # ... wire up your blocks ...
        return bld.build()

    class MyModel(JaxonomyDiagramSlave):
        DIAGRAM_FACTORY = staticmethod(build_diagram)
        DT = 0.01
        # The slave introspects the diagram's input/output ports at
        # __init__ time and registers them as FMI variables.

The slave drives the diagram via :func:`jaxonomy.simulate` segments — one
``[t, t + step_size]`` segment per ``do_step``. Internal state is carried
across steps in ``self._ctx``.

Inputs:

T-025c — the slave auto-detects every ``Constant`` block in the
diagram and exposes its ``name`` as an FMI input variable. When the
master writes that variable, ``apply_inputs`` updates the Constant's
``value`` parameter in the next step's context. This covers the
common case where an FMU input is a setpoint/gain/reference signal
fed into the rest of the model. For more elaborate routing
(multi-port wiring, cross-block parameter coupling), override
:meth:`apply_inputs` to plumb your own context updates.

Outputs are derived from the diagram's exported output ports
(``bld.export_output(...)``), so anything you want the master to
read needs to be on ``diagram.output_ports``.
"""

from __future__ import annotations

from typing import Callable, Iterable

# Lazy: pythonfmu is only needed when actually running inside an FMU.
# Import at class-definition time is fine because the slave script is
# only ever loaded inside the FMU's embedded Python.
from pythonfmu import Fmi2Slave, Fmi2Causality, Real

import numpy as np


class JaxonomyDiagramSlave(Fmi2Slave):
    """Wraps a Jaxonomy diagram as a pythonfmu ``Fmi2Slave``."""

    #: Subclass-overridable: a zero-argument callable that returns a
    #: built :class:`~jaxonomy.framework.diagram.Diagram`. Use
    #: ``staticmethod`` when binding a free function.
    DIAGRAM_FACTORY: Callable | None = None

    #: Communication step size used as the default. Has no effect on
    #: solver internals — the simulator picks its own minor steps.
    DT: float = 0.01

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if self.DIAGRAM_FACTORY is None:
            raise RuntimeError(
                f"{type(self).__name__} must define DIAGRAM_FACTORY"
            )
        # Build the diagram and an initial context. The diagram lives
        # for the FMU instance's lifetime; ``self._ctx`` is replayed
        # forward across do_step calls.
        self._diagram = self.DIAGRAM_FACTORY()
        self._ctx = self._diagram.create_context()
        self._dt = float(self.DT)

        # Register an FMI Real for every diagram I/O port. Vector
        # ports become one Real per element with structured names
        # (``port[i]`` for 1-D, ``port[i,j]`` for higher).
        # All values are stashed in ``self._values`` keyed by the FMI
        # variable name; the registered ``Real`` carries a getter /
        # setter pair targeting that dict, so variable names with
        # bracket / comma syntax (vector elements) don't require a
        # valid Python identifier.
        self._values: dict[str, float] = {}
        self._input_specs: list[tuple[object, tuple[int, ...]]] = []
        self._output_specs: list[tuple[object, tuple[int, ...]]] = []
        # T-025c: name-keyed map from auto-discovered Constant blocks
        # to ``(system_id, parameter_name)`` pairs. Used by the
        # default ``apply_inputs`` to translate FMI writes into
        # ``ctx.with_subcontext(...).with_parameter(...)`` updates.
        self._constant_inputs: dict[str, tuple[int, str]] = {}

        for port in self._diagram.input_ports:
            for varname, idx in _expand_port(port):
                self._values[varname] = 0.0
                self._register_real(varname, 0.0, Fmi2Causality.input)
                self._input_specs.append((port, idx))

        for port in self._diagram.output_ports:
            for varname, idx in _expand_port(port):
                self._values[varname] = 0.0
                # Outputs default to initial=calculated; FMI forbids
                # passing a start= value alongside that, so register
                # without one.
                self._register_real(varname, None, Fmi2Causality.output)
                self._output_specs.append((port, idx))

        # T-025c: walk leaf systems and expose any Constant block as
        # an FMI input variable named after the block. The block's
        # ``value`` parameter gets overridden via ``apply_inputs``.
        # Skip Constants whose name is already taken by an exported
        # output port (the master would see a duplicate variable).
        existing_names = set(self._values.keys())
        for leaf in _iter_constants(self._diagram):
            block_name = leaf.name
            if not block_name or block_name in existing_names:
                continue
            self._constant_inputs[block_name] = (leaf.system_id, "value")
            initial = float(self._ctx[leaf.system_id].parameters["value"])
            self._values[block_name] = initial
            self._register_real(block_name, initial, Fmi2Causality.input)
            existing_names.add(block_name)

    def _register_real(self, name: str, start, causality):
        """Register one Real variable with a closure-based
        getter/setter that targets ``self._values[name]``.

        ``start`` may be ``None`` (omits the start attribute, required
        when ``initial=calculated``, the default for outputs).
        """
        def _g(_n=name):
            return self._values[_n]
        def _s(v, _n=name):
            self._values[_n] = float(v)
        kwargs: dict = {"causality": causality, "getter": _g, "setter": _s}
        if start is not None:
            kwargs["start"] = start
        self.register_variable(Real(name, **kwargs))

        # On entering simulation we'll prime the output attributes so
        # that the importer reading at t=0 sees real defaults rather
        # than the placeholder zeros above.
        self._t = 0.0
        self._primed = False

    # ── overridable hooks ─────────────────────────────────────────────

    def apply_inputs(self, ctx, input_values: dict[str, float]):
        """Hook: fold ``input_values`` into ``ctx`` and return a new
        context.

        T-025c default: any FMI input variable whose name matches a
        ``Constant`` block in the diagram is treated as a write to
        that block's ``value`` parameter. Subclasses can override to
        plumb non-Constant inputs (e.g. parameter overrides on other
        block types, or values that fan out across multiple blocks);
        call ``super().apply_inputs(ctx, input_values)`` first to keep
        the default Constant routing.

        ``input_values`` maps the FMI variable name (post-flatten) to
        the float the importer just set on us.
        """
        for name, (sys_id, param) in self._constant_inputs.items():
            if name not in input_values:
                continue
            sub = ctx[sys_id].with_parameter(param, input_values[name])
            ctx = ctx.with_subcontext(sys_id, sub)
        return ctx

    def read_outputs(self, ctx) -> dict[str, float]:
        """Hook: read every FMI output variable's current value from
        ``ctx``. Default uses ``port.eval(ctx)`` and unpacks elements
        by index. Override to customize."""
        out: dict[str, float] = {}
        for port, idx in self._output_specs:
            value = port.eval(ctx)
            varname = _flat_name(port, idx)
            out[varname] = float(_index(value, idx))
        return out

    # ── Fmi2Slave implementation ──────────────────────────────────────

    def do_step(self, current_time: float, step_size: float) -> bool:
        # Ingest input values the importer wrote on us. Two sources:
        # (a) flattened diagram input ports (rare; the FMU contract
        # rarely exposes hierarchical input ports), and (b) T-025c
        # auto-discovered Constant-block names. All values flow
        # through self._values, which the registered Real getter /
        # setter pairs target.
        input_values: dict[str, float] = {}
        for port, idx in self._input_specs:
            varname = _flat_name(port, idx)
            input_values[varname] = float(self._values[varname])
        for block_name in self._constant_inputs:
            input_values[block_name] = float(self._values[block_name])
        try:
            self._ctx = self.apply_inputs(self._ctx, input_values)

            if step_size > 0:
                # Lazy import to keep this module light at parse time
                # and so that pythonfmu's bundling doesn't have to
                # see jaxonomy.simulate.
                import jaxonomy
                from jaxonomy.simulation import SimulatorOptions
                results = jaxonomy.simulate(
                    self._diagram,
                    self._ctx,
                    (current_time, current_time + step_size),
                    options=SimulatorOptions(return_context=True),
                )
                if results.context is not None:
                    self._ctx = results.context

            outs = self.read_outputs(self._ctx)
            for varname, value in outs.items():
                self._values[varname] = float(value)
            self._t = current_time + step_size
            return True
        except Exception as exc:
            self.log(f"do_step failed at t={current_time}: {exc}")
            return False


# ── helpers ───────────────────────────────────────────────────────────


def _iter_constants(diagram) -> Iterable[object]:
    """Yield every ``Constant`` leaf system in the diagram tree.

    Identifies Constants by class name + module to avoid taking a
    hard import dependency on ``jaxonomy.library`` at parse time
    (which would fight pythonfmu's slave-module discovery).
    Recursively descends nested diagrams via their ``leaf_systems``
    attribute.

    The module-name check matches both the legacy ``primitives``
    module and the post-split ``jaxonomy.library.sources`` module
    where ``Constant`` actually lives now (it is re-exported from
    primitives.py for backward compatibility).
    """
    leaves = getattr(diagram, "leaf_systems", None)
    if leaves is None:
        return
    for leaf in leaves:
        cls = type(leaf)
        mod = cls.__module__ or ""
        if (cls.__name__ == "Constant"
                and ("primitives" in mod or "jaxonomy.library.sources" in mod)):
            yield leaf


def _expand_port(port) -> Iterable[tuple[str, tuple[int, ...]]]:
    """Yield ``(flat_name, multi_index)`` for each element of ``port``.
    Scalar ports yield one entry with multi_index ``()``."""
    name = port.name or f"port_{getattr(port, 'index', 0)}"
    default = getattr(port, "default_value", None)
    if default is None:
        yield name, ()
        return
    arr = np.asarray(default)
    if arr.ndim == 0:
        yield name, ()
        return
    for i in range(arr.size):
        idx = np.unravel_index(i, arr.shape)
        flat = f"{name}[{','.join(str(k) for k in idx)}]"
        yield flat, idx


def _flat_name(port, idx: tuple[int, ...]) -> str:
    name = port.name or f"port_{getattr(port, 'index', 0)}"
    if not idx:
        return name
    return f"{name}[{','.join(str(k) for k in idx)}]"


def _index(value, idx: tuple[int, ...]):
    if not idx:
        return value
    arr = np.asarray(value)
    return arr[idx]
