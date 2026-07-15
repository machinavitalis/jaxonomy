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

The slave drives the diagram forward one ``[t, t + step_size]`` segment
per ``do_step``. Internal state is carried across steps in ``self._ctx``.
By default a single :class:`~jaxonomy.simulation.simulator.Simulator`
kernel is built once and reused for every step (``REUSE_SIMULATOR``);
set the flag to ``False`` to fall back to one :func:`jaxonomy.simulate`
call per segment.

Inputs:

Two mechanisms feed FMI inputs into the diagram:

- **Exported diagram input ports** (``bld.export_input(...)``) — the
  slave wraps the diagram so that each exported input port is fed by an
  injected ``Constant`` block named after the port, and registers that
  name as an FMI input. Master writes are applied to the injected
  Constant before every step, so exported input ports behave as real
  FMI inputs.
- **T-025c auto-discovery** — every ``Constant`` block in the diagram
  is exposed as an FMI input variable named after the block (vector
  Constants flatten to one Real per element, ``name[i]``). When the
  master writes the variable, ``apply_inputs`` updates the Constant's
  ``value`` parameter in the next step's context. For more elaborate
  routing (cross-block parameter coupling), override
  :meth:`apply_inputs` to plumb your own context updates.

Outputs are derived from the diagram's exported output ports
(``bld.export_output(...)``), so anything you want the master to
read needs to be on ``diagram.output_ports``. Per FMI 2.0 §4.2.4 the
outputs are primed during ``exit_initialization_mode`` so a master
reading right after initialization sees the true t=0 values.

Initial states:

Set ``EXPOSE_INITIAL_STATES = {"fmi_param_name": "block_name", ...}``
on the subclass to register the named leaf systems' continuous states
as FMI parameters (variability ``fixed``); the values the master sets
during initialization mode are applied to the context when
``exit_initialization_mode`` runs.

Logging:

``import jaxonomy`` configures the ``jaxonomy`` logger from the
``LOG_LEVEL`` environment variable (default INFO), which floods an FMI
master's console with per-step messages. The slave therefore runs its
embedded jaxonomy calls with the logger temporarily set to the class
attribute ``LOG_LEVEL`` (default ``"ERROR"``). Setting the environment
variable ``LOG_LEVEL`` opts out (the slave then leaves logging alone),
as does setting the class attribute to ``None``. Logging outside the
slave's own calls is never touched.
"""

from __future__ import annotations

import contextlib
import logging as _logging
import os
from typing import Callable, Iterable

# Lazy: pythonfmu is only needed when actually running inside an FMU.
# Import at class-definition time is fine because the slave script is
# only ever loaded inside the FMU's embedded Python.
from pythonfmu import Fmi2Slave, Fmi2Causality, Fmi2Variability, Real

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

    #: Log level applied to the ``jaxonomy`` logger while the slave
    #: executes its embedded jaxonomy calls (diagram build, per-step
    #: simulation). ``None`` disables the override entirely; a
    #: ``LOG_LEVEL`` environment variable set by the user always wins
    #: (the slave then leaves logging alone). The previous level is
    #: restored after every call, so library logging outside the slave
    #: is unaffected.
    LOG_LEVEL: str | int | None = "ERROR"

    #: Opt-in map of FMI parameter name -> leaf-system name whose
    #: continuous state is exposed as (a) ``fixed``-variability FMI
    #: parameter(s), applied to the context when
    #: ``exit_initialization_mode`` runs. Vector states flatten to one
    #: Real per element (``name[i]``).
    EXPOSE_INITIAL_STATES: dict[str, str] | None = None

    #: Reuse one built Simulator + JIT kernel across ``do_step`` calls
    #: (the kernel takes the segment endpoints as traced arguments, so
    #: advancing time hits the JAX compile cache). Cuts the per-step
    #: overhead from ~100 ms (re-trace + XLA re-compile per segment) to
    #: well under 1 ms. Set to ``False`` to fall back to one
    #: :func:`jaxonomy.simulate` call per segment.
    REUSE_SIMULATOR: bool = True

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if self.DIAGRAM_FACTORY is None:
            raise RuntimeError(
                f"{type(self).__name__} must define DIAGRAM_FACTORY"
            )
        # Build the diagram and an initial context. The diagram lives
        # for the FMU instance's lifetime; ``self._ctx`` is replayed
        # forward across do_step calls. Exported diagram input ports
        # are wired to injected Constant blocks (named after the port)
        # so master writes actually reach the diagram.
        with self._log_scope():
            self._diagram = self.DIAGRAM_FACTORY()
            injected_input_constants: list = []
            if self._diagram.input_ports:
                self._diagram, injected_input_constants = (
                    _wrap_exported_inputs(self._diagram)
                )
            self._ctx = self._diagram.create_context()
        self._dt = float(self.DT)
        self._t = 0.0

        # Persistent-kernel state (REUSE_SIMULATOR). The kernel is
        # (re)built lazily on the first step whose size exceeds the
        # one the cached options were derived for.
        self._kernel = None
        self._kernel_step_size = 0.0

        # Register an FMI Real for every diagram I/O surface. Vector
        # values become one Real per element with structured names
        # (``name[i]`` for 1-D, ``name[i,j]`` for higher).
        # All values are stashed in ``self._values`` keyed by the FMI
        # variable name; the registered ``Real`` carries a getter /
        # setter pair targeting that dict, so variable names with
        # bracket / comma syntax (vector elements) don't require a
        # valid Python identifier.
        self._values: dict[str, float] = {}
        self._output_specs: list[tuple[object, tuple[int, ...]]] = []
        # T-025c: FMI-variable-name-keyed maps from Constant blocks to
        # context locations. Scalars map to ``(system_id, parameter)``;
        # vector elements map to ``(system_id, multi_index)``. Used by
        # the default ``apply_inputs`` to translate FMI writes into
        # ``ctx.with_subcontext(...).with_parameter(...)`` updates.
        self._constant_inputs: dict[str, tuple[int, str]] = {}
        self._constant_array_inputs: dict[str, tuple[int, tuple[int, ...]]] = {}
        # EXPOSE_INITIAL_STATES: FMI parameter name (post-flatten) ->
        # ``(system_id, multi_index)`` into that leaf's continuous state.
        self._initial_state_params: dict[str, tuple[int, tuple[int, ...]]] = {}

        # Seed the name set with the output variable names up front so
        # the input registrations below can detect collisions with
        # outputs that only register afterwards.
        existing_names: set[str] = {
            varname
            for port in self._diagram.output_ports
            for varname, _idx in _expand_port(port)
        }

        # Injected input-port Constants first, so exported input ports
        # keep their FMI variable names. A name collision here means
        # the diagram exports an input and something else under the
        # same name — fail loudly rather than silently dropping the
        # input (FMI forbids duplicate variable names).
        for leaf in injected_input_constants:
            registered = self._register_constant_input(leaf, existing_names)
            if not registered:
                raise RuntimeError(
                    f"exported input port {leaf.name!r} collides with "
                    f"another FMI variable name; rename the port or the "
                    f"conflicting block"
                )

        for port in self._diagram.output_ports:
            for varname, idx in _expand_port(port):
                self._values[varname] = 0.0
                # Outputs default to initial=calculated; FMI forbids
                # passing a start= value alongside that, so register
                # without one.
                self._register_real(varname, None, Fmi2Causality.output)
                self._output_specs.append((port, idx))

        # T-025c: walk leaf systems and expose any Constant block as
        # (an) FMI input variable(s) named after the block. The block's
        # ``value`` parameter gets overridden via ``apply_inputs``.
        # Skip Constants whose name is already taken (e.g. by an
        # exported output port — the master would see a duplicate
        # variable — or by an injected input-port Constant above).
        for leaf in _iter_constants(self._diagram):
            self._register_constant_input(leaf, existing_names)

        if self.EXPOSE_INITIAL_STATES:
            self._register_initial_state_params(existing_names)

    def _register_real(self, name: str, start, causality, variability=None):
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
        if variability is not None:
            kwargs["variability"] = variability
        self.register_variable(Real(name, **kwargs))

    def _register_constant_input(self, leaf, existing_names: set[str]) -> bool:
        """Register one Constant block as FMI input variable(s).

        Scalar values register under the block name; vector values
        flatten to ``name[i]`` / ``name[i,j]`` elements. Names already
        taken are skipped. Returns True if every variable registered.
        """
        block_name = leaf.name
        if not block_name:
            return False
        value = np.asarray(self._ctx[leaf.system_id].parameters["value"])
        all_registered = True
        if value.ndim == 0:
            entries = [(block_name, ())]
        else:
            entries = [
                (_flat_element_name(block_name, idx), idx)
                for idx in (np.unravel_index(i, value.shape)
                            for i in range(value.size))
            ]
        for varname, idx in entries:
            if varname in existing_names:
                all_registered = False
                continue
            initial = float(value[idx]) if idx else float(value)
            self._values[varname] = initial
            self._register_real(varname, initial, Fmi2Causality.input)
            if idx:
                self._constant_array_inputs[varname] = (leaf.system_id, idx)
            else:
                self._constant_inputs[varname] = (leaf.system_id, "value")
            existing_names.add(varname)
        return all_registered

    def _register_initial_state_params(self, existing_names: set[str]):
        """Register EXPOSE_INITIAL_STATES entries as FMI parameters
        (variability ``fixed``: settable during initialization mode,
        applied by :meth:`exit_initialization_mode`)."""
        leaves = {leaf.name: leaf for leaf in (self._diagram.leaf_systems or [])}
        for param_name, block_name in self.EXPOSE_INITIAL_STATES.items():
            leaf = leaves.get(block_name)
            if leaf is None:
                raise RuntimeError(
                    f"EXPOSE_INITIAL_STATES: no leaf system named "
                    f"{block_name!r} in the diagram (have: "
                    f"{sorted(leaves)})"
                )
            xc = self._ctx[leaf.system_id].continuous_state
            if xc is None:
                raise RuntimeError(
                    f"EXPOSE_INITIAL_STATES: leaf system {block_name!r} "
                    f"has no continuous state"
                )
            xc = np.asarray(xc)
            if xc.ndim == 0:
                entries = [(param_name, ())]
            else:
                entries = [
                    (_flat_element_name(param_name, idx), idx)
                    for idx in (np.unravel_index(i, xc.shape)
                                for i in range(xc.size))
                ]
            for varname, idx in entries:
                if varname in existing_names:
                    raise RuntimeError(
                        f"EXPOSE_INITIAL_STATES: FMI variable name "
                        f"{varname!r} is already registered"
                    )
                initial = float(xc[idx]) if idx else float(xc)
                self._values[varname] = initial
                self._register_real(
                    varname, initial, Fmi2Causality.parameter,
                    variability=Fmi2Variability.fixed,
                )
                self._initial_state_params[varname] = (leaf.system_id, idx)
                existing_names.add(varname)

    # ── overridable hooks ─────────────────────────────────────────────

    def apply_inputs(self, ctx, input_values: dict[str, float]):
        """Hook: fold ``input_values`` into ``ctx`` and return a new
        context.

        T-025c default: any FMI input variable whose name matches a
        ``Constant`` block in the diagram (or a vector element of one,
        ``name[i]``) is treated as a write to that block's ``value``
        parameter. Exported diagram input ports arrive here too — they
        are fed by injected Constants named after the port. Subclasses
        can override to plumb non-Constant inputs (e.g. parameter
        overrides on other block types, or values that fan out across
        multiple blocks); call ``super().apply_inputs(ctx,
        input_values)`` first to keep the default Constant routing.

        ``input_values`` maps the FMI variable name (post-flatten) to
        the float the importer just set on us.

        The replacement parameter values preserve the shape/dtype of
        the ones they replace, so the persistent JIT kernel
        (``REUSE_SIMULATOR``) keeps hitting its compile cache.
        """
        for name, (sys_id, param) in self._constant_inputs.items():
            if name not in input_values:
                continue
            old = ctx[sys_id].parameters[param]
            new = np.full_like(np.asarray(old), input_values[name])
            sub = ctx[sys_id].with_parameter(param, new)
            ctx = ctx.with_subcontext(sys_id, sub)
        pending: dict[int, np.ndarray] = {}
        for name, (sys_id, idx) in self._constant_array_inputs.items():
            if name not in input_values:
                continue
            arr = pending.get(sys_id)
            if arr is None:
                arr = np.array(ctx[sys_id].parameters["value"])
            arr[idx] = input_values[name]
            pending[sys_id] = arr
        for sys_id, arr in pending.items():
            sub = ctx[sys_id].with_parameter("value", arr)
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

    def exit_initialization_mode(self):
        """Apply initialization-mode writes and prime the outputs.

        FMI 2.0 §4.2.4: outputs with ``initial=calculated`` must be
        readable right after ``exitInitializationMode`` — before the
        first ``doStep`` — so fold the master's initialization-mode
        input/parameter writes into the context and run the output
        readout here.
        """
        with self._log_scope():
            self._ctx = self._apply_initial_states(self._ctx)
            self._ctx = self.apply_inputs(self._ctx, self._collect_input_values())
            self._write_outputs(self._ctx)

    def do_step(self, current_time: float, step_size: float) -> bool:
        # Ingest input values the importer wrote on us (exported input
        # ports and T-025c auto-discovered Constant-block names — all
        # flow through self._values, which the registered Real getter /
        # setter pairs target), advance the diagram one segment, then
        # refresh the outputs. step_size == 0 skips integration but
        # still runs the readout.
        try:
            with self._log_scope():
                self._ctx = self.apply_inputs(
                    self._ctx, self._collect_input_values()
                )
                if step_size > 0:
                    if self.REUSE_SIMULATOR:
                        self._ctx = self._advance_cached(
                            current_time, step_size
                        )
                    else:
                        self._ctx = self._advance_simulate(
                            current_time, step_size
                        )
                self._write_outputs(self._ctx)
            self._t = current_time + step_size
            return True
        except Exception as exc:
            self.log(f"do_step failed at t={current_time}: {exc}")
            return False

    # ── internals ─────────────────────────────────────────────────────

    def _collect_input_values(self) -> dict[str, float]:
        names = list(self._constant_inputs) + list(self._constant_array_inputs)
        return {name: float(self._values[name]) for name in names}

    def _write_outputs(self, ctx):
        for varname, value in self.read_outputs(ctx).items():
            self._values[varname] = float(value)

    def _apply_initial_states(self, ctx):
        """Fold EXPOSE_INITIAL_STATES parameter values into the
        context's continuous state (shape/dtype-preserving)."""
        pending: dict[int, np.ndarray] = {}
        for varname, (sys_id, idx) in self._initial_state_params.items():
            arr = pending.get(sys_id)
            if arr is None:
                arr = np.array(ctx[sys_id].continuous_state)
            arr[idx] = self._values[varname]
            pending[sys_id] = arr
        for sys_id, arr in pending.items():
            sub = ctx[sys_id].with_continuous_state(arr)
            ctx = ctx.with_subcontext(sys_id, sub)
        return ctx

    def _advance_simulate(self, current_time: float, step_size: float):
        """One fresh :func:`jaxonomy.simulate` call per segment
        (``REUSE_SIMULATOR = False`` fallback path)."""
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
        return results.context if results.context is not None else self._ctx

    def _advance_cached(self, current_time: float, step_size: float):
        """Advance one segment through a persistent JIT kernel.

        ``simulate`` jits a fresh closure with the segment endpoints
        baked in, so every ``do_step`` re-traces and re-compiles
        (~100 ms/step). Building the Simulator once and passing
        ``(t0, tf)`` as traced kernel arguments makes subsequent steps
        hit the compile cache (<1 ms/step, numerically identical). The
        kernel is rebuilt only when the communication step size grows
        beyond the one the resolved options (max_major_steps heuristic)
        were derived for.
        """
        if self._kernel is None or step_size > self._kernel_step_size:
            self._build_kernel(step_size)
        tf = current_time + step_size
        ctx = self._kernel(self._ctx, current_time, tf)
        t_end = float(ctx.time)
        if not np.isclose(t_end, tf, rtol=1e-9, atol=1e-12):
            raise RuntimeError(
                f"simulation stopped at t={t_end} before reaching {tf}"
            )
        return ctx

    def _build_kernel(self, step_size: float):
        import jax
        from jaxonomy.backend import ODESolver
        from jaxonomy.simulation import SimulatorOptions
        from jaxonomy.simulation.simulator import Simulator, _check_options

        options = _check_options(
            self._diagram,
            SimulatorOptions(return_context=True),
            (0.0, step_size),
            None,
        )
        ode_solver = ODESolver(self._diagram, options=options.ode_options)
        sim = Simulator(self._diagram, ode_solver=ode_solver, options=options)

        @jax.jit
        def _kernel(context, t0, tf):
            return sim.advance_to(tf, context.with_time(t0)).context

        self._kernel = _kernel
        self._kernel_step_size = step_size

    @contextlib.contextmanager
    def _log_scope(self):
        """Temporarily set the ``jaxonomy`` logger to ``LOG_LEVEL`` for
        the slave's embedded calls. No-op when the class attribute is
        ``None`` or the user set a ``LOG_LEVEL`` environment variable
        (``jaxonomy._init`` then already honored their choice)."""
        level = self.LOG_LEVEL
        if level is None or "LOG_LEVEL" in os.environ:
            yield
            return
        logger = _logging.getLogger("jaxonomy")
        prev_level = logger.level
        logger.setLevel(level)
        try:
            yield
        finally:
            logger.setLevel(prev_level)


# ── helpers ───────────────────────────────────────────────────────────


def _wrap_exported_inputs(diagram):
    """Wrap ``diagram`` so every exported input port is fed by an
    injected ``Constant`` named after the port.

    Returns ``(wrapper_diagram, injected_constants)``. The wrapper
    re-exports the original output ports under their own names, so the
    FMU's output surface is unchanged; the input surface becomes the
    injected Constants (which the default ``apply_inputs`` drives via
    context parameter updates — the mechanism that actually reaches
    the diagram, unlike writes to an unconnected exported port).
    """
    import jaxonomy
    from jaxonomy.library import Constant

    input_ports = list(diagram.input_ports)
    output_ports = list(diagram.output_ports)
    bld = jaxonomy.DiagramBuilder()
    bld.add(diagram)
    injected = []
    for port in input_ports:
        name = port.name or f"port_{getattr(port, 'index', 0)}"
        default = getattr(port, "default_value", None)
        value = np.asarray(default, dtype=float) if default is not None else 0.0
        constant = bld.add(Constant(value, name=name))
        bld.connect(constant.output_ports[0], port)
        injected.append(constant)
    for port in output_ports:
        bld.export_output(port, name=port.name)
    return bld.build(), injected


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
        yield _flat_element_name(name, idx), idx


def _flat_element_name(name: str, idx: tuple[int, ...]) -> str:
    return f"{name}[{','.join(str(k) for k in idx)}]"


def _flat_name(port, idx: tuple[int, ...]) -> str:
    name = port.name or f"port_{getattr(port, 'index', 0)}"
    if not idx:
        return name
    return _flat_element_name(name, idx)


def _index(value, idx: tuple[int, ...]):
    if not idx:
        return value
    arr = np.asarray(value)
    return arr[idx]
