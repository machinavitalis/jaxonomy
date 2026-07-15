# SPDX-License-Identifier: MIT

from typing import TYPE_CHECKING
import logging as _logging
import numpy as np

logger = _logging.getLogger(__name__)
from jaxonomy.backend import numpy_api as npa
from jaxonomy.acausal.component_library.base import EqnEnv, SymKind
from jaxonomy.acausal.diagram_processing import DiagramProcessing
from jaxonomy.acausal.acausal_diagram import AcausalDiagram
from jaxonomy.acausal.index_reduction.index_reduction import (
    IndexReduction,
    SemiExplicitDAE,
)
from jaxonomy.acausal.index_reduction.equation_utils import (
    compute_initial_conditions,
)
from jaxonomy.acausal.error import (
    AcausalModelError,
    AcausalCompilerError,
)
from jaxonomy.framework import DependencyTicket, LeafSystem, build_recorder
from jaxonomy.framework.error import BlockParameterError
from jaxonomy.framework.system_base import UpstreamEvalError
from jaxonomy.lazy_loader import LazyLoader
from jaxonomy.framework.system_base import Parameter

if TYPE_CHECKING:
    import sympy as sp
else:
    sp = LazyLoader("sp", globals(), "sympy")


if TYPE_CHECKING:
    from jaxonomy.dashboard.serialization.from_model_json import AcausalNetwork


def _lambdify_with_diagnostics(sym_args, expr, modules, context: str, dp: DiagramProcessing):
    try:
        return sp.lambdify(sym_args, expr, modules=modules)
    except Exception as err:
        fcns = sorted({str(f.func) for f in expr.atoms(sp.Function)})
        dp._update_dpd()
        raise AcausalCompilerError(
            message=(
                f"Failed to lambdify {context}. "
                f"Functions present in expression: {fcns}. "
                f"Original error: {type(err).__name__}: {err}"
            ),
            dpd=dp.dpd,
        ) from err


class AcausalSystem(LeafSystem):
    # These class-level attributes are populated by the dashboard serialization layer
    # (from_model_json.py) after compilation, to map dashboard port IDs to system port
    # indices. They must be class-level (not instance-level) because the serializer
    # accesses them on the class before __init__ runs. If multiple AcausalSystem
    # instances are created, the last one's maps will overwrite earlier ones; this is
    # acceptable in the current single-diagram serialization context.
    acausal_network: "AcausalNetwork" = None
    outports_maps: dict[str, dict[int, int]] = None
    inports_maps: dict[str, dict[int, int]] = None

    def __init__(
        self, dp: DiagramProcessing, sed: SemiExplicitDAE, name: str, leaf_backend="jax"
    ):
        super().__init__(name=name)
        self.dp = dp
        self.sed = sed

        # Input Ports
        self._configure_inputs(dp.diagram.input_syms)

        # Parameters in Context
        self._configure_parameters(dp.params)

        # Set up callbacks using lambdify
        time = self.dp.eqn_env.t
        inputs = [s.s for s in self.dp.diagram.input_syms]
        params = [s.s for s in self.dp.params.keys()]
        self.n_ode = sed.n_ode
        self.n_alg = sed.n_alg
        sym_args = (time, sed.x, sed.y, *inputs, *params)

        # Continuous State
        self._configure_continuous_state(sym_args, leaf_backend)

        # Output Ports
        if sed.is_scaled:
            outp_exprs = {
                sym: expr.subs(sed.vars_to_scaled_vars)
                for sym, expr in dp.outp_exprs.items()
            }
        else:
            outp_exprs = dp.outp_exprs
        self._configure_outputs(sym_args, outp_exprs, leaf_backend)

        # Zero Crossing
        if sed.is_scaled:
            zcs = {}
            for idx, zc_tuple in dp.zcs.items():
                zc_expr, direction, is_bool_expr = zc_tuple
                zcs[idx] = (
                    zc_expr.subs(sed.vars_to_scaled_vars),
                    direction,
                    is_bool_expr,
                )
        else:
            zcs = dp.zcs

        self._configure_zcs(sym_args, zcs, leaf_backend)

    def _configure_inputs(self, input_syms):
        # this ensure that inports of the acausal_system are in the same order as
        # the 'inputs' portion of the lambdify args
        insym_to_portid = {}
        for sym in input_syms:
            idx = self.declare_input_port(name=sym.name)
            insym_to_portid[sym] = idx

        self.insym_to_portid = insym_to_portid

    def continuous_state_layout(self) -> list[dict]:
        """Describe each row of this system's continuous-state vector.

        The compiled state vector is ``[x; y]``: ``n_ode`` differential
        rows (the integrated states) followed by ``n_alg`` algebraic
        rows (constraint unknowns such as node potentials and flows).
        This mapping is otherwise only recoverable by inspecting the
        compiler's internals — use it when setting or reading the state
        directly (e.g. ``context.with_continuous_state`` for episodic
        resets; pair with ``SimulatorOptions(dae_initial_projection=True)``
        so the algebraic rows are re-solved).

        Returns:
            One dict per row, in state-vector order:
            ``{"row": int, "kind": "differential"|"algebraic",
            "name": str, "scaled_name": str | None}``.  ``name`` is the
            physical variable name; when the system was compiled with
            ``scale=True`` the state itself holds the *scaled* quantity
            and ``scaled_name`` is its symbol name.
        """
        sed = self.sed
        scaled_to_var = getattr(sed, "scaled_vars_to_vars", None) or {}

        def _describe(sym):
            if sed.is_scaled and sym in scaled_to_var:
                return str(scaled_to_var[sym]), str(sym)
            return str(sym), (str(sym) if sed.is_scaled else None)

        layout = []
        for i, sym in enumerate(list(sed.x) + list(sed.y)):
            name, scaled_name = _describe(sym)
            layout.append({
                "row": i,
                "kind": "differential" if i < self.n_ode else "algebraic",
                "name": name,
                "scaled_name": scaled_name,
            })
        return layout

    def initialize(self, *args, **kwargs):
        # presently, this is unused.
        # resolved_args = [
        #     arg.get() if isinstance(arg, Parameter) else arg for arg in args
        # ]
        if args:
            self.dp._update_dpd()
            raise AcausalCompilerError(
                message="AcausalSystem initialize method detected unamed args. This is not supported.",
                dpd=self.dp.dpd,
            )

        # all the acausal component params are passed in through kwargs
        resolved_kwargs = {
            k: kwarg.get() if isinstance(kwarg, Parameter) else kwarg
            for k, kwarg in kwargs.items()
        }
        for k, v in self.dp.params.items():
            if k.validator:
                # not all params have validation, so skip if the validator is None.
                resolved_val = resolved_kwargs[k.name]
                is_tracer = hasattr(resolved_val, "aval") or type(resolved_val).__name__.endswith("Tracer")
                if not is_tracer and not k.validator(resolved_val):
                    raise AcausalModelError(k.invalid_msg)

    def _configure_parameters(self, params):
        for k, v in params.items():
            self.declare_dynamic_parameter(k.name, v)

    def parameter_names_for(self, component) -> list[str]:
        """Return the parameter keys this ``AcausalSystem`` exposes for one
        acausal component.

        Compiled-system parameters follow the convention
        ``f"{component.name}_{symbol_name}"``, so a user doing
        ``jax.grad`` / ``jax.vmap`` over a component's parameters can look up
        the right ``ctx.parameters`` keys directly instead of grepping or
        guessing.

        Args:
            component: The acausal component instance (e.g. an
                ``Insulator(name=...)``) whose parameter keys you want.

        Returns:
            Sorted list of parameter-name strings registered on this system.
        """
        comp_name = getattr(component, "name", None)
        if comp_name is None:
            raise TypeError(
                f"parameter_names_for: expected a component with a .name attribute, "
                f"got {type(component).__name__}"
            )
        prefix = f"{comp_name}_"
        return sorted(
            name for name in self._dynamic_parameters if name.startswith(prefix)
        )

    def _configure_continuous_state(self, sym_args, leaf_backend):
        sp_rhs = _lambdify_with_diagnostics(
            sym_args,
            self.sed.f + self.sed.g,
            modules=[leaf_backend, {"npa": npa}],
            context=f"{self.name} continuous-state RHS",
            dp=self.dp,
        )
        mass_matrix = np.concatenate((np.ones(self.n_ode), np.zeros(self.n_alg)))
        # T-017b: hoist ``[str(k) for k in self.dp.params.keys()]`` out of
        # the per-trace path.  ``self.dp.params`` is a dict whose keys are
        # sympy ``Sym`` objects (their stringification involves attribute
        # lookups on each call), and the ``_rhs`` closure below is
        # re-entered multiple times per JIT trace (forward pass, jacfwd
        # AD, etc.).  Pre-computing the string key list once at compile
        # time is bit-equivalent and shaves a few ms off the trace.
        n_ode = self.n_ode
        param_keys_str = tuple(str(k) for k in self.dp.params.keys())

        def _rhs(time, state, *u, **params):
            cstate = state.continuous_state
            param_values = [params[k] for k in param_keys_str]
            x = cstate[:n_ode]
            y = cstate[n_ode:]  # noqa
            return npa.array(sp_rhs(time, x, y, *u, *param_values))

        cs_idx = self.declare_continuous_state(
            shape=(self.n_ode + self.n_alg), ode=_rhs, mass_matrix=mass_matrix
        )
        # T-044 (NeuralDAEBlock, phase 1): expose the compiled differential-row
        # RHS and its callback index so a post-hoc neural correction can be
        # added to the *differential* rows without touching the symbolic
        # (sympy / Pantelides) path.  See
        # ``jaxonomy.library.neural_dae.add_neural_correction``.
        self._cs_callback_idx = cs_idx
        self._cs_base_ode = _rhs
        self._cs_mass_matrix = mass_matrix

    def _configure_outputs(self, sym_args, outp_exprs, leaf_backend):
        outsym_to_portid = {}
        if self.dp.diagram.num_outputs == 0:
            # if not output, output the state vector
            self.declare_continuous_state_output(name=f"{self.name}:output")
            outsym_to_portid = None
        else:
            # T-017b: same hoisting as in ``_configure_continuous_state``.
            n_ode = self.n_ode
            param_keys_str = tuple(str(k) for k in self.dp.params.keys())

            def _make_outp_callback(outp_expr):
                lambdify_output = _lambdify_with_diagnostics(
                    sym_args,
                    outp_expr,
                    modules=[leaf_backend, {"npa": npa}],
                    context=f"{self.name} output `{sym.name}`",
                    dp=self.dp,
                )

                def _output_fun(time, state, *u, **params):
                    cstate = state.continuous_state
                    param_values = [params[k] for k in param_keys_str]
                    x, y = cstate[:n_ode], cstate[n_ode:]  # noqa
                    return npa.array(lambdify_output(time, x, y, *u, *param_values))

                return _output_fun

            # declaring acausal_system output ports in this order means that the ordering
            # 'source of truth' is self.model.output_syms which can be used to link
            # back to the acausal sensors causal port for diagram link src point remapping.
            for sym, outp_expr in outp_exprs.items():
                _output = _make_outp_callback(outp_expr)
                idx = self.declare_output_port(
                    _output,
                    name=sym.name,  # FIXME: not the name from the block port
                    prerequisites_of_calc=[DependencyTicket.xc],
                    requires_inputs=True,
                )
                outsym_to_portid[sym] = idx

        self.outsym_to_portid = outsym_to_portid

    def _configure_zcs(self, sym_args, zcs, leaf_backend):
        # T-017b: hoist ``[str(k) for k in self.dp.params.keys()]`` out of
        # the per-trace path; see ``_configure_continuous_state``.
        n_ode = self.n_ode
        param_keys_str = tuple(str(k) for k in self.dp.params.keys())

        def _make_zc_callback(zc_expr, is_bool_expr):
            lambdify_zc = _lambdify_with_diagnostics(
                sym_args,
                zc_expr,
                modules=[leaf_backend, {"npa": npa}],
                context=f"{self.name} zero-crossing",
                dp=self.dp,
            )

            if is_bool_expr:
                # zero crossing are always expecting a float, so when the zero crossing condition
                # is defined by a boolean, we have this extra npa.where() function whihc maps it
                # to a float. the same is done in the IfThenEsle block.
                def _zc_fun(time, state, *u, **params):
                    cstate = state.continuous_state
                    param_values = [params[k] for k in param_keys_str]
                    x, y = cstate[:n_ode], cstate[n_ode:]  # noqa
                    return npa.where(
                        npa.array(lambdify_zc(time, x, y, *u, *param_values)), 1.0, -1.0
                    )

            else:

                def _zc_fun(time, state, *u, **params):
                    cstate = state.continuous_state
                    param_values = [params[k] for k in param_keys_str]
                    x, y = cstate[:n_ode], cstate[n_ode:]  # noqa
                    return npa.array(lambdify_zc(time, x, y, *u, *param_values))

            return _zc_fun

        # just copying what we do for outputs.
        for idx, zc_tuple in zcs.items():
            zc_expr, direction, is_bool_expr = zc_tuple
            _zc = _make_zc_callback(zc_expr, is_bool_expr)
            self.declare_zero_crossing(_zc, direction=direction)

    def _validate_parameters(self, context):
        """Check each acausal component parameter against its declared
        validator, using the context's current (possibly runtime-updated)
        value. Runs at ``create_context()`` time — via
        ``initialize_static_data`` — so a parameter changed after compile
        (e.g. ``Parameter.set(...)``) is still caught, not just the value
        baked in at compile time.
        """
        params = context[self.system_id].parameters
        for sym in self.dp.params:
            if sym.validator is None:
                continue
            val = params.get(sym.name)
            if val is None:
                continue
            # Skip abstract/traced values (e.g. under jit/vmap); validation
            # only makes sense on concrete parameter values. Note a *concrete*
            # jax array also has ``.aval``, so test the type name, not hasattr.
            if type(val).__name__.endswith("Tracer"):
                continue
            if not bool(sym.validator(val)):
                raise BlockParameterError(sym.invalid_msg, system=self)

    def initialize_static_data(self, context):
        self._validate_parameters(context)
        dp = self.dp
        sed = self.sed
        try:
            u = self.collect_inputs(context)
            knowns_new = {}
            for known in sed.knowns.keys():
                sym = dp.syms_map[known]
                if sym.kind == SymKind.inp:
                    port_idx = self.insym_to_portid[sym]
                    knowns_new[known] = u[port_idx]

            knowns = sed.knowns.copy()
            knowns.update(knowns_new)

            X_ic_mapping = compute_initial_conditions(
                sed.t,
                sed.eqs,
                sed.X,
                sed.ics,
                sed.ics_weak,
                knowns,
                verbose=self.dp.verbose,
            )

            x_ic = [X_ic_mapping[sed.dae_X_to_X_mapping[var]] for var in sed.x]
            y_ic = [X_ic_mapping[sed.dae_X_to_X_mapping[var]] for var in sed.y]

            if sed.is_scaled:
                x_ic = [val / sed.Ss[idx] for idx, val in enumerate(x_ic)]
                y_ic = [val / sed.Ss[idx + sed.n_ode] for idx, val in enumerate(y_ic)]

            x0 = np.array(x_ic + y_ic, dtype=float)

            self._default_continuous_state = x0
            local_context = context[self.system_id].with_continuous_state(x0)
            context = context.with_subcontext(self.system_id, local_context)

        except UpstreamEvalError:
            logger.warning(
                "AcausalSystem.initialize_static_data: upstream port evaluation failed "
                "before initial conditions could be set. Initial state will use default "
                "zeros. If the simulation diverges immediately, check that all input "
                "ports are connected."
            )
        return super().initialize_static_data(context)


class AcausalCompiler:
    """
    This class ochestrates the compilation of Acausal models to Acausal Phleafs.

    There are 3 primary stages:
        1] diagram_processing. AcausalDiagram -> DAEs
        2] index_reduction. DAEs -> index-1 DAEs
        3] generate_acausal_system. index-1 DAEs -> pleaf

    **Parameter naming convention.** When the compiled ``AcausalSystem`` is
    inserted into a diagram, each acausal component contributes its parameters
    to the system's dynamic-parameter dict under the key
    ``f"{component.name}_{symbol_name}"``. For example, an
    ``Insulator(name="cool_lo_m0", R=...)`` exposes its resistance as
    ``ctx[acausal_system.system_id].parameters["cool_lo_m0_R"]``. This is what
    you'd reach for when doing ``jax.grad`` / ``jax.vmap`` over component
    parameters. Use :func:`AcausalSystem.parameter_names_for` for a typed
    accessor instead of grepping by substring.
    """

    def __init__(
        self,
        eqn_env: EqnEnv,
        diagram: AcausalDiagram,
        scale: bool = False,
        verbose: bool = False,
        condition_number_threshold: float = 1e4,
    ):
        """T-038: ``condition_number_threshold`` (default 1e4) tunes the
        ill-conditioned-Jacobian warning emitted by the index-reduction pass.
        Increase it for models where the natural Jacobian has large but
        well-understood scale separation (e.g. multi-domain electrical/thermal),
        or pass a smaller value to fail loudly on borderline-singular setups.
        """
        self.dp = DiagramProcessing(
            eqn_env,
            diagram,
            verbose=verbose,
        )
        self.index_reduction_done = False
        self.scale = scale
        self.verbose = verbose
        self.condition_number_threshold = condition_number_threshold

        build_recorder.create_acausal_compiler()

    def diagram_processing(self):
        self.dp()

    def index_reduction(self):
        self.ir = IndexReduction(
            ir_inputs_from_dp=self.dp.index_reduction_inputs,
            dpd=self.dp.dpd,
            verbose=self.verbose,
            condition_number_threshold=self.condition_number_threshold,
        )
        self.sed = self.ir(scale=self.scale)
        self.index_reduction_done = True

    def generate_acausal_system(
        self,
        name="acausal_system",
        leaf_backend="jax",
    ):
        """
        This function is used for generating AcasualSystem from an AcausalDiagram.
        """
        if not self.dp.diagram_processing_done:
            self.diagram_processing()
        if not self.index_reduction_done:
            # NOTE: presently in from_model_json.py, self.index_reduction_done is set to True
            # this is only to temporarily skip index reduction when testing from json.
            self.index_reduction()

        with build_recorder.paused():
            system = AcausalSystem(self.dp, self.sed, name, leaf_backend)
            # T-044 phase 2: inject diagram-authored NeuralDAEBlocks at the
            # differential-row RHS site. Done here (post index reduction, with
            # the system's _cs_base_ode in place) so the neural term never
            # touches the symbolic / Pantelides path. No-op without blocks.
            if getattr(self.dp.diagram, "neural_blocks", None):
                from jaxonomy.library.neural_dae import apply_neural_blocks

                apply_neural_blocks(system, self.dp.diagram, self.dp, self.sed)
        build_recorder.compile_acausal_diagram(system)

        return system

    # execute compilation
    def __call__(self, name="acausal_system", leaf_backend="jax", return_sed=False):
        self.diagram_processing()
        self.index_reduction()
        system = self.generate_acausal_system(name=name, leaf_backend=leaf_backend)
        if return_sed:
            return system, self.sed
        return system
