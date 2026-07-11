# SPDX-License-Identifier: MIT

"""
basic FMU import block based on FMPy package and the example:
https://github.com/CATIA-Systems/FMPy/blob/main/fmpy/examples/custom_input.py
"""

from collections import namedtuple
from math import prod
from typing import TYPE_CHECKING

import numpy as np

import jax

from ..framework.error import BlockRuntimeError
from ..framework import (
    LeafSystem,
    BlockInitializationError,
    DependencyTicket,
    parameters,
)
from ..logging import logger
from ..backend import io_callback, numpy_api as npa
from ..lazy_loader import LazyLoader, LazyModuleAccessor

if TYPE_CHECKING:
    import fmpy
    from fmpy import fmi1, fmi2, fmi3
    from fmpy.model_description import ScalarVariable
else:
    fmpy = LazyLoader("fmpy", globals(), "fmpy")
    fmi1 = LazyModuleAccessor(fmpy, "fmi1")
    fmi2 = LazyModuleAccessor(fmpy, "fmi2")
    fmi3 = LazyModuleAccessor(fmpy, "fmi3")
    model_description = LazyModuleAccessor(fmpy, "model_description")
    ScalarVariable = LazyModuleAccessor(model_description, "ScalarVariable")


def _is_fmi3(model_description) -> bool:
    """T-026: Detect FMI 3.0 from a parsed fmpy ``model_description``.

    fmpy stores ``fmiVersion`` as a string like ``"2.0"`` or ``"3.0"``.
    """
    return str(getattr(model_description, "fmiVersion", "")).startswith("3.")


def _fmi_call_exception():
    """Lazy resolution of fmpy's ``FMICallException``.

    fmpy defines this once in ``fmpy.fmi1``; both v2 and v3 raise it.
    Older code referenced ``fmi2.FMICallException`` / ``fmi3.FMICallException``
    which never existed and broke at runtime when an FMU error fired.
    """
    try:
        return (fmi1.FMICallException,)
    except Exception:
        return ()


def _fmi3_exceptions():
    """Back-compat alias for T-026's exception-tuple helper.

    Original T-026 logic looked up ``fmi3.FMICallException`` (which
    doesn't exist); T-026a corrected this to ``fmi1.FMICallException``.
    Kept under the original name so prior tests keep importing it.
    """
    return _fmi_call_exception()

ValueReference = int


# T-026a: per-FMI-version typed accessor table.  fmpy exposes typed
# get/set methods on the slave instance; we dispatch through these names
# rather than peppering the body of exec_step with branches.
#
# Keys are the ``ScalarVariable.type`` strings produced by fmpy's
# model_description parser; values are (getter, setter, numpy_dtype).
# A getter takes (vr, nValues) and returns a flat tuple/list; a setter
# takes (vr, values).  ``None`` for either means "unsupported in this
# FMI version".  numpy_dtype is the dtype used for output state arrays
# and for casting input values to a list the C API will accept.

_FMI2_ACCESSORS = {
    "Real":        ("getReal",     "setReal",     np.float64),
    "Integer":     ("getInteger",  "setInteger",  np.int32),
    "Enumeration": ("getInteger",  "setInteger",  np.int32),
    "Boolean":     ("getBoolean",  "setBoolean",  np.bool_),
    "String":      ("getString",   "setString",   object),
}

_FMI3_ACCESSORS = {
    "Float64":     ("getFloat64",  "setFloat64",  np.float64),
    "Float32":     ("getFloat32",  "setFloat32",  np.float32),
    "Real":        ("getFloat64",  "setFloat64",  np.float64),
    "Int8":        ("getInt8",     "setInt8",     np.int8),
    "UInt8":       ("getUInt8",    "setUInt8",    np.uint8),
    "Int16":       ("getInt16",    "setInt16",    np.int16),
    "UInt16":      ("getUInt16",   "setUInt16",   np.uint16),
    "Int32":       ("getInt32",    "setInt32",    np.int32),
    "UInt32":      ("getUInt32",   "setUInt32",   np.uint32),
    "Int64":       ("getInt64",    "setInt64",    np.int64),
    "UInt64":      ("getUInt64",   "setUInt64",   np.uint64),
    "Integer":     ("getInt32",    "setInt32",    np.int32),
    # FMI 3 models Enumeration as Int64 (verified against
    # Reference-FMUs/Feedthrough). T-026 originally dispatched it as
    # Int32 — that was a bug that didn't surface until T-026a
    # exercised it against a real FMU.
    "Enumeration": ("getInt64",    "setInt64",    np.int64),
    "Boolean":     ("getBoolean",  "setBoolean",  np.bool_),
    "String":      ("getString",   "setString",   object),
    "Binary":      ("getBinary",   "setBinary",   object),
}


def _accessor_table(fmi_version: str):
    return _FMI3_ACCESSORS if fmi_version == "3.0" else _FMI2_ACCESSORS


def _variable_n_values(variable) -> int:
    """Total scalar count for a ScalarVariable. 1 for plain scalars, the
    product of dimensions for arrays. fmpy populates ``.shape`` with
    a tuple of ints for resolved dimensions and ``()`` for scalars."""
    shape = getattr(variable, "shape", ()) or ()
    if not shape:
        return 1
    return int(prod(shape))


def _variable_shape(variable) -> tuple:
    return tuple(getattr(variable, "shape", ()) or ())


class ModelicaFMU(LeafSystem):
    # Should we pass parameter overrides via kwargs? Sounds like it could conflict
    # in some rare cases (eg. dt, name...). The corresponding definition in
    # block_interface.py is pretty fragile in this regard.
    def __init__(
        self,
        file_name,
        dt,
        name=None,
        input_names: list[str] = None,
        output_names: list[str] = None,
        parameters: dict = None,
        start_time: float = 0.0,
        first_step_at_zero: bool = False,
        **kwargs,
    ):
        """Load and execute an FMU for Co-Simulation.

        .. warning:: **One instance per FMU per process** for FMUs built
            with pythonfmu (e.g. via :func:`jaxonomy.library.build_fmu`).
            The embedded-Python wrapper holds a process-wide
            ``Py_Initialize`` singleton, so instantiating the same
            ``.fmu`` dylib twice in one Python process fails. For
            multi-start or batched co-simulation, isolate each instance
            in its own process (``multiprocessing`` /
            ``concurrent.futures.ProcessPoolExecutor`` with the *spawn*
            start method, or a ``subprocess`` running a small driver
            script) and aggregate results afterwards. This is an
            upstream pythonfmu limitation, not a Jaxonomy one; FMUs from
            other exporters (OpenModelica, Dymola, Reference-FMUs) do
            not carry it.

        Args:
            file_name (str): path to FMU file
            dt (float): stepsize for FMU simulation
            name (str, optional): name of block
            input_names (list[str], optional): if set, only expose these inputs
            output_names (list[str], optional): if set, only expose these outputs
            parameters (dict, optional): dictionary of parameter overrides
            start_time (float, optional): FMU experiment start time.
            first_step_at_zero (bool, optional): Default ``False``. The
                first FMU step fires at ``t=dt`` (Modelica clocked-block
                convention: the periodic update is offset by one sample),
                so the block's outputs at ``t=0`` reflect the FMU's
                initial state (post-``setupExperiment``, pre-step) and
                only switch to "post-first-step" values from ``t=dt``
                onward. This introduces a one-sample phase lag versus an
                FMU exported with ``offset=0`` semantics, which weakens
                FMU round-trip byte-equivalence. Pass
                ``first_step_at_zero=True`` to fire the first step at
                ``t=0`` so the outputs leave their initial value as soon
                as the simulation begins. Surfaced as the FMU offset
                asymmetry in a follow-up finding.
            kwargs: ignored
        """
        try:
            super().__init__(name=name)
            self._init(
                file_name,
                dt,
                name=name or f"fmu_{self.system_id}",
                input_names=input_names,
                output_names=output_names,
                parameters=parameters,
                start_time=start_time,
                first_step_at_zero=first_step_at_zero,
            )
        except Exception as e:
            logger.error(
                "Failed to initialize FMU block %s (%s): %s", name, self.system_id, e
            )
            raise BlockInitializationError(str(e), system=self)

    @parameters(static=["file_name"])
    def _init(
        self,
        file_name,
        dt,
        name: str,
        input_names: list[str] = None,
        output_names: list[str] = None,
        parameters: dict = None,
        start_time: float = 0.0,
        first_step_at_zero: bool = False,
    ):
        self.dt = dt

        # read the model description
        model_description = fmpy.read_model_description(file_name)

        # extract the FMU
        unzipdir = fmpy.extract(file_name)

        # T-026: dispatch on FMI version.  FMI 3.0 has a different slave
        # class and uses type-specific getters (getFloat64, getInt32, ...)
        # in place of FMI 2.0's untyped getReal / getInteger.
        self._fmi_version = "3.0" if _is_fmi3(model_description) else "2.0"
        if self._fmi_version == "3.0":
            cs = model_description.coSimulation
            if cs is None:
                # Reference-FMUs Clocks.fmu, e.g., is scheduledExecution-only —
                # a different fmpy class (FMU3ScheduledExecution) and a
                # different stepping protocol. Out of scope for the
                # co-simulation block.
                raise BlockInitializationError(
                    f"FMU {file_name} has no co-simulation interface "
                    f"(only modelExchange / scheduledExecution). "
                    f"ModelicaFMU only supports co-simulation FMUs.",
                    system=self,
                )
            self.fmu = fmu = fmi3.FMU3Slave(
                guid=model_description.guid,
                unzipDirectory=unzipdir,
                modelIdentifier=cs.modelIdentifier,
                instanceName=name,
            )
            fmu.instantiate()
            # FMI 3.0 collapses setupExperiment into enterInitializationMode
            # via keyword args.
            fmu.enterInitializationMode(startTime=start_time)
        else:
            self.fmu = fmu = fmi2.FMU2Slave(
                guid=model_description.guid,
                unzipDirectory=unzipdir,
                modelIdentifier=model_description.coSimulation.modelIdentifier,
                instanceName=name,
            )
            fmu.instantiate()
            # setup and set startTime before entering initialization mode
            # per FMI 2.0.4 section 2.1.6.
            fmu.setupExperiment(startTime=start_time)
            # enter initialization mode before get/set params per FMI 2.0.4 section 4.2.4.
            fmu.enterInitializationMode()

        # collect the value references
        self.fmu_inputs: list[ValueReference] = []
        self.fmu_outputs: list[ValueReference] = []
        # T-026a: parallel ScalarVariable arrays so exec_step can dispatch
        # each port to the right typed getter/setter (mixed-type FMUs)
        # and reshape array I/O.
        self.fmu_input_vars: list = []
        self.fmu_output_vars: list = []

        inputs_by_name: dict[str, ScalarVariable] = {}
        outputs_by_name: dict[str, ScalarVariable] = {}
        variable_by_id: dict[int, ScalarVariable] = {}

        # FIXME: we rely on the XML file here, but jaxonomy uses a similar
        # JSON file with altered variable names.
        # TODO: implement support for parsing that file and mapping from
        # jaxonomy json name to/from xml name properly.
        def _compatible_param_name(name):
            return name.replace(".", "_")

        for variable in model_description.modelVariables:
            if variable.causality == "input":
                variable_by_id[variable.valueReference] = variable
                inputs_by_name[variable.name] = variable
            elif variable.causality == "output":
                variable_by_id[variable.valueReference] = variable
                outputs_by_name[variable.name] = variable
            elif variable.causality == "parameter" and parameters is not None:
                compat_name = _compatible_param_name(variable.name)
                parameter_value = parameters.get(compat_name, None)
                if parameter_value is None:
                    continue

                logger.debug(
                    "Setting parameter #%d '%s' <%s>: %s %s",
                    variable.valueReference,
                    variable.name,
                    variable.type,
                    parameter_value,
                    type(parameter_value),
                )

                # Values at this point have been wrapped into np.ndarray of
                # shape () via jaxonomy's JSON parsing. Enumerations are ints.
                # T-026: dispatch to v3 setter names where applicable.
                self._set_value(fmu, variable, parameter_value, name)

        # If input_names or output_names are set, we filter out the variables
        # exposed as I/O ports to match those. This so that the ports in model.json
        # actually match those in the FMU.
        # NOTE: Maybe this is unnecessarily complicated.
        # T-026a: types we can't represent inside the JAX-traced state.
        # Exclude them from the *default* port set; users who actually want
        # them must opt in via input_names/output_names and handle the
        # object dtype themselves.
        _NON_JAX_TYPES = {"String", "Binary"}

        def _accept_default(variable, role):
            if variable.type in _NON_JAX_TYPES:
                logger.warning(
                    "FMU %s: skipping %s port %r (type %s — not "
                    "representable as a JAX array; pass via "
                    "%s_names to expose it explicitly)",
                    name, role, variable.name, variable.type, role,
                )
                return False
            return True

        if input_names is not None:
            for in_name in input_names:
                if in_name not in inputs_by_name:
                    raise BlockInitializationError(
                        f"Input port {in_name} found on the block { name} "
                        + f"but not found in FMU {file_name}",
                        system=self,
                    )
                variable = inputs_by_name[in_name]
                self.fmu_inputs.append(variable.valueReference)
                self.fmu_input_vars.append(variable)
                self.declare_input_port(name=variable.name)
        else:
            for in_name, variable in inputs_by_name.items():
                if not _accept_default(variable, "input"):
                    continue
                self.fmu_inputs.append(variable.valueReference)
                self.fmu_input_vars.append(variable)
                self.declare_input_port(name=in_name)

        if output_names is not None:
            for out_name in output_names:
                if out_name not in outputs_by_name:
                    raise BlockInitializationError(
                        f"Input port {out_name} found on the block { name} "
                        + f"but not found in FMU {file_name}",
                        system=self,
                    )
                variable = outputs_by_name[out_name]
                self.fmu_outputs.append(variable.valueReference)
                self.fmu_output_vars.append(variable)
        else:
            for out_name, variable in outputs_by_name.items():
                if not _accept_default(variable, "output"):
                    continue
                self.fmu_outputs.append(variable.valueReference)
                self.fmu_output_vars.append(variable)

        # T-026a: pre-compute per-type read/write groupings so the
        # io_callback at every step can dispatch with one batched call
        # per type instead of one per port.
        self._output_groups = self._build_groups(self.fmu_output_vars, "get")
        self._input_groups = self._build_groups(self.fmu_input_vars, "set")

        # exit initialization mode after get/set params per FMI 2.0.4 section 4.2.4.
        fmu.exitInitializationMode()

        # Declare a discrete state component for each of the output
        # variables. T-026a fix: index by position into fmu_output_vars,
        # not by valueReference — FMI 3 alias variables (e.g. BouncingBall's
        # ``h`` and ``h_ft`` sharing vr=1) collapse a dict-by-vr lookup.
        self.state_names = [v.name for v in self.fmu_output_vars]
        self.DiscreteStateType = namedtuple("DiscreteState", self.state_names)

        # Create the default discrete state values
        default_values = {}
        for variable in self.fmu_output_vars:
            start_value = self._get_value(fmu, variable)
            default_values[variable.name] = start_value

        # Map the default values to array-like types so that they have shape and dtype
        default_state = jax.tree_util.tree_map(
            npa.asarray, self.DiscreteStateType(**default_values)
        )
        self.declare_discrete_state(default_value=default_state, as_array=False)

        # Declare an output port for each of the output variables
        def _make_output_callback(o_port_name):
            def _output(time, state, *inputs, **parameters):
                return getattr(state.discrete_state, o_port_name)

            return _output

        for o_port_name in default_values:
            self.declare_output_port(
                _make_output_callback(o_port_name),
                name=o_port_name,
                prerequisites_of_calc=[DependencyTicket.xd],
                requires_inputs=False,
            )

        # The step function acts as a periodic update that will update all components
        # of the discrete state.
        #
        # A5 (jax.grad-through-FMU): an FMU co-simulation step is an opaque
        # external call routed through ``io_callback`` — JAX has no derivative
        # rule for it, and a naive ``jax.grad`` otherwise dies with the generic
        # "IO callbacks do not support JVP". Wrap the step in a ``custom_jvp``
        # whose JVP rule raises a clear, FMU-specific error with concrete
        # workarounds, so the failure names the cause instead of leaving the
        # user to decode a backend message. The forward (primal) path is
        # unchanged — ``custom_jvp`` only intercepts differentiation.
        block_name = self.name

        @jax.custom_jvp
        def _fmu_step(time, state, inputs_tuple):
            return io_callback(
                self.exec_step, default_state, time, state, *inputs_tuple
            )

        @_fmu_step.defjvp
        def _fmu_step_jvp(primals, tangents):
            raise BlockRuntimeError(
                "jax.grad / jax.jvp through a ModelicaFMU block "
                f"({block_name!r}) is not supported: an FMU co-simulation step "
                "is an opaque external call (via io_callback / fmpy) with no "
                "analytic derivative. To obtain sensitivities, either (a) use "
                "finite differences over the FMU inputs/parameters (perturb in "
                "plain numpy outside jax.grad, or use jaxonomy.uq Monte Carlo / "
                "Sobol), or (b) replace the FMU with a native jaxonomy model "
                "for the part of the system you need to differentiate. The "
                "forward simulation (no jax.grad) works fine.",
                system=self,
            )

        def _step(time, state, *inputs):
            # Use the io_callback (wrapped for a clear grad-time error) so that
            # we can call the untraceable FMU object.
            return _fmu_step(time, state, tuple(inputs))

        # ``offset=dt`` (default) honors the Modelica clocked-block
        # convention so the block's outputs at ``t=0`` reflect the FMU's
        # ``setupExperiment`` state. ``first_step_at_zero=True`` fires
        # the first step at ``t=0``, eliminating the one-sample phase
        # lag for users who exported the FMU with that semantics. See
        # the constructor docstring.
        self.declare_periodic_update(
            _step,
            period=dt,
            offset=0.0 if first_step_at_zero else dt,
        )

    # T-026 / T-026a: type-dispatch helpers covering both FMI 2 and FMI 3.
    def _set_value(self, fmu, variable, value, block_name):
        """Set one variable on the FMU using the right typed setter.

        Supports scalar and array variables. ``value`` may be a scalar,
        a numpy/jax array, or any iterable; it is flattened to length
        ``prod(variable.shape)`` before the C call.
        """
        table = _accessor_table(self._fmi_version)
        vt = variable.type
        if vt not in table or table[vt][1] is None:
            raise BlockInitializationError(
                f"Unsupported FMI {self._fmi_version} variable type "
                f"{vt!r} for parameter {variable.name} in FMU block "
                f"{block_name}",
                system=self,
            )
        _, setter_name, dtype = table[vt]
        ref = [variable.valueReference]
        n = _variable_n_values(variable)
        try:
            if dtype is object:  # String / Binary — pass through as-is
                if n == 1:
                    fmu_values = [value if isinstance(value, (str, bytes))
                                  else str(value)]
                else:
                    fmu_values = list(value)
            elif dtype is np.bool_:
                arr = np.asarray(value).reshape(-1).astype(np.bool_)
                fmu_values = [bool(v) for v in arr]
            else:
                arr = np.asarray(value).reshape(-1).astype(dtype, copy=False)
                fmu_values = arr.tolist()
            if len(fmu_values) != n:
                raise ValueError(
                    f"variable {variable.name} expects {n} values "
                    f"(shape={_variable_shape(variable)}), got {len(fmu_values)}"
                )
            getattr(fmu, setter_name)(ref, fmu_values)
        except Exception as e:
            raise BlockInitializationError(
                f"Failed to set parameter {variable.name}: {e}", system=self,
            ) from e

    def _get_value(self, fmu, variable):
        """Read one variable from the FMU using the version-correct getter.

        Returns a Python scalar for shape-() variables and a numpy
        ndarray of the right dtype/shape for array variables.
        """
        table = _accessor_table(self._fmi_version)
        vt = variable.type
        if vt not in table or table[vt][0] is None:
            raise NotImplementedError(
                f"Unsupported FMI {self._fmi_version} variable type {vt!r} for "
                f"output port {variable.name}"
            )
        getter_name, _, dtype = table[vt]
        ref = [variable.valueReference]
        shape = _variable_shape(variable)
        n = _variable_n_values(variable)
        # FMI 3 typed getters take nValues for arrays. FMI 2 has no array
        # type at the C level, so n is always 1 there.
        if self._fmi_version == "3.0" and n != 1:
            raw = getattr(fmu, getter_name)(ref, n)
        else:
            raw = getattr(fmu, getter_name)(ref)
        if not shape:
            return raw[0]
        if dtype is object:
            return np.asarray(list(raw), dtype=object).reshape(shape)
        return np.asarray(raw, dtype=dtype).reshape(shape)

    def _build_groups(self, variables, mode):
        """T-026a: bucket variables by FMI type for one batched call per
        type. Returns a list of (accessor_name, dtype, port_indices,
        value_refs, n_values_per_ref, shapes).

        ``mode`` selects the column from the type table — ``"get"`` or
        ``"set"``. Per-port reshapes happen in exec_step using the
        recorded shapes; cumulative offsets are recomputed from
        n_values_per_ref to keep the structure light.
        """
        col = 0 if mode == "get" else 1
        table = _accessor_table(self._fmi_version)
        by_type: dict[str, dict] = {}
        for idx, var in enumerate(variables):
            vt = var.type
            if vt not in table or table[vt][col] is None:
                raise BlockInitializationError(
                    f"Unsupported FMI {self._fmi_version} {mode}-port type "
                    f"{vt!r} on variable {var.name}", system=self,
                )
            accessor_name, _setter, dtype = table[vt]
            if mode == "set":
                accessor_name = table[vt][1]
            bucket = by_type.setdefault(vt, {
                "accessor": accessor_name, "dtype": dtype,
                "indices": [], "refs": [], "nvals": [], "shapes": [],
            })
            bucket["indices"].append(idx)
            bucket["refs"].append(var.valueReference)
            bucket["nvals"].append(_variable_n_values(var))
            bucket["shapes"].append(_variable_shape(var))
        return list(by_type.values())

    def _create_discrete_state_type(self, fmu, fmu_outputs, variables):
        self.state_names = [variables[output_ref].name for output_ref in fmu_outputs]
        self.DiscreteStateType = namedtuple("DiscreteState", self.state_names)

    def exec_step(self, time, state, *inputs, **parameters):
        # NOTE: We should get the fmu from the context in order to build a pure
        # function but it is very unlikely this would ever work with FMUs since
        # they have their own internal hidden state. More context here:
        # https://github.com/machinavitalis/jaxonomy/pull/5330/files#r1419062533
        # Also look at that PR to see the previous implementation (it worked with
        # a single I/O port).

        try:
            fmu = self.fmu

            # Note: although it may appear that the order of operations below is
            # backwards, e.g. 1] get_outputs, 2] set_inputs, 3] step, this is
            # actually intentional.
            # Explanation by example assuming 1sec update intervals.
            # The reason get_outputs happens before set_inputs and 'step, is that
            # at t=0, the fmu outputs are already at t=0, so we can just read them.
            # Then, the fmu should get inputs at t=0, and use those to take a step
            # to t=1. The step operation, using inputs at t=0, puts the fmu in a
            # state where it outputs are now at t=1. This we cannot read them until
            # next update interval at t=1.

            # T-026a: read every output type group, then write every input
            # type group. One C call per type. Arrays are reshaped on read
            # and flattened on write.
            xd: dict = {}
            is_v3 = self._fmi_version == "3.0"
            for grp in self._output_groups:
                refs = grp["refs"]
                total_n = sum(grp["nvals"])
                if is_v3 and total_n != len(refs):
                    raw = getattr(fmu, grp["accessor"])(refs, total_n)
                else:
                    raw = getattr(fmu, grp["accessor"])(refs)
                # Walk the flat result and slice per port.
                offset = 0
                for port_idx, n, shape in zip(grp["indices"], grp["nvals"], grp["shapes"]):
                    chunk = raw[offset:offset + n]
                    offset += n
                    name = self.state_names[port_idx]
                    if not shape:
                        xd[name] = chunk[0]
                    elif grp["dtype"] is object:
                        xd[name] = np.asarray(list(chunk), dtype=object).reshape(shape)
                    else:
                        xd[name] = np.asarray(chunk, dtype=grp["dtype"]).reshape(shape)

            # Group inputs by type and flatten each port's value to that
            # type's batched ``set...`` call.
            for grp in self._input_groups:
                values_flat = []
                for port_idx, n, shape in zip(grp["indices"], grp["nvals"], grp["shapes"]):
                    val = inputs[port_idx]
                    if grp["dtype"] is object:
                        if n == 1:
                            values_flat.append(
                                val if isinstance(val, (str, bytes)) else str(val)
                            )
                        else:
                            values_flat.extend(list(val))
                    elif grp["dtype"] is np.bool_:
                        arr = np.asarray(val).reshape(-1).astype(np.bool_)
                        values_flat.extend(bool(v) for v in arr)
                    else:
                        arr = np.asarray(val).reshape(-1).astype(grp["dtype"], copy=False)
                        values_flat.extend(arr.tolist())
                getattr(fmu, grp["accessor"])(grp["refs"], values_flat)

            # Advance the FMU in time. The periodic update fires at t=dt,
            # 2dt, ..., but doStep expects currentCommunicationPoint to be
            # the *start* of the step interval — i.e. the FMU's current
            # internal time, which is one period earlier. Strict FMUs
            # (e.g. Reference-FMUs/BouncingBall) error otherwise.
            fmu.doStep(
                currentCommunicationPoint=float(time) - self.dt,
                communicationStepSize=self.dt,
            )

        except (*_fmi_call_exception(),) as e:
            logger.error(
                "Failed to run FMU block %s (%s): %s", self.name, self.system_id, e
            )
            raise BlockRuntimeError(str(e), system=self) from e

        xd = jax.tree_util.tree_map(npa.asarray, xd)

        return self.DiscreteStateType(**xd)
