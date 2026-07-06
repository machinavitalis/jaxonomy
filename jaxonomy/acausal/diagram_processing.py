# SPDX-License-Identifier: MIT

import copy
from typing import TYPE_CHECKING
import warnings
from jaxonomy.acausal.component_library.base import (
    Eqn,
    Sym,
    SymKind,
    EqnKind,
    EqnEnv,
)
from .acausal_diagram import AcausalDiagram
from jaxonomy.acausal.component_library.base import nodpot_qty
from jaxonomy.acausal.component_library.fluid import (
    FluidProperties,
    M_FLOW_EPS,
)
from jaxonomy.acausal.component_library.hydraulic import (
    HydraulicProperties,
)
from jaxonomy.acausal.error import (
    AcausalModelError,
    AcausalCompilerError,
)
from .types import DiagramProcessingData, IndexReductionInputs
from jaxonomy.lazy_loader import LazyLoader
from jaxonomy.framework.system_base import Parameter

if TYPE_CHECKING:
    import sympy as sp
    import numpy as np
else:
    sp = LazyLoader("sp", globals(), "sympy")
    np = LazyLoader("np", globals(), "numpy")


class DiagramProcessing:
    """
    This class transforms an AcausalDiagram object into a set of differential algebraic equations.
    The output from this class is the input for index reduction.

    The stages of diagram_processing (in order):
    - identify the AcausalDiagram nodes
    - identify the fluid domain networks. A 'network' is subset of component ports that are all connected together sharing the
        same fluid. e.g. a coolant loop, or a hydraulic system.
    - finalize the AcausalDiagram by calling finalize() method of all components. Presently this propagates fluid properties to all
        appropriate ports.
    - generate node flow equations.
    - generate node potential variables and equations.
    - add derivative relation equations. e.g. ff(t) = Derivative(f(t))
    - 'finalize', i.e. update sets of equations and symbols.
    - alias elimination
    - initial condition consistency check
    - prune unnecesary derivative relations. some f(t) were eliminated by alias elimination, their derivative relations are not needed.
    - identify params, initial conditions, and inputs
    - collect expressions for evlauting output values.
    - prepare inputs for index reduction
    """

    def __init__(
        self,
        eqn_env: EqnEnv,
        diagram: AcausalDiagram,
        verbose: bool = False,
    ):
        self.eqn_env = eqn_env
        self.diagram = diagram

        # ordered(indexed) symbols and equations sets
        self.syms = {}
        self.next_sym_idx = 0
        self.syms_append(eqn_env.syms)
        self.eqs = {}
        # dict{sympy symbol: parent 'Sym' object} used to dereference to the parent.
        self.syms_map = {}

        self.eqs_original = {}  # copy of eqs, but never perform any substitutions.
        self.syms_map_original = {}

        self.nodes = {}  # dict of node_id to set of ports
        self.node_domains = {}  # dict of node_id to node domain
        self.port_to_node = {}  # dict(cmp:dict(port_id:node_id))
        self.pot_alias_map = {}  # dict{node_id:dict{der_idx:pot_sym}}
        self.alias_map = {}  # dict{aliasee:aliasee_sub_expr}
        self.alias_eqs = []  # list of the equation in which aliases were found.
        self.aliaser_map = {}  # dict{aliaser:[aliasees]} *ees replaced by *ers.
        self.sym_ic = {}  # dict from Sympy.symbol to it's initial condition value
        self.params = {}  # dict from param symbol to param value
        self.inputs = []  # list of input Sym objects
        self.outp_exprs = {}  # dict{sym:expr} where expr syms are all in self.eqs
        self.zcs = {}  # dict{idx:(zc_expr,zc_direction)}
        self.fld_nws = {}  # dict{fluid_prop_cmp:set of all nodes associated with it}
        self.diagram_processing_done = False
        self.verbose = verbose

        self._update_dpd()

    # helper functions for diagram processing
    def pp_eqs(self, eqs_all=False, tabs=""):
        if not self.verbose:
            return
        if self.eqs_original and eqs_all:
            print(tabs + "original equations")
            for idx, eq in self.eqs_original.items():
                print(tabs + f"\t{idx}: {eq}")
        print(tabs + "equations")
        for eq_idx, eq in self.eqs.items():
            print(tabs + f"\t{eq_idx}: {eq}")

    def sanitize_solve(self, s, eq):
        # handle the various options for what can be returned from solve().
        exprs = sp.solve(eq.e, s)
        if isinstance(exprs, list):
            if len(exprs) > 1:
                message = f"[sanitize_solve] Solving {eq} for {s} produced more than one solution:{exprs}."
                raise AcausalCompilerError(message=message, dpd=self.dpd)
            elif len(exprs) == 0:
                message = (
                    f"[sanitize_solve] Solving {eq} for {s} produced no solutions."
                )
                raise AcausalCompilerError(message=message, dpd=self.dpd)
            else:
                return exprs[0]
        else:
            # It's unclear whether a non-list result is ever an acceptable
            # outcome here, so block it for now.
            message = f"[sanitize_solve] Sanitize_solve failed: s={s}, eq={eq}. Did not return a list."
            raise AcausalCompilerError(message=message, dpd=self.dpd)

    def pp(self):
        if not self.verbose:
            return
        print(f"DiagramProcessing {self.diagram.name}:")

        print(f"\tComponents:{[c.name for c in self.diagram.comps]}")

        print("\tConnections:")
        for port_tuple_a, port_tuple_b in self.diagram.connections:
            ca, pa = port_tuple_a
            cb, pb = port_tuple_b
            print(f"\t\t{ca.name},{pa}\tto {cb.name},{pb}")

        print(f"\tsyms:{self.syms}")

        self.pp_eqs(eqs_all=True, tabs="\t")

        print("\tnode_sets:")
        for id, nset in self.nodes.items():
            if len(nset) > 3:
                print(f"\t\t{id}:")
                for n in nset:
                    print(f"\t\t\t{n}:")
            else:
                print(f"\t\t{id}:{nset}")

        print(f"\tsym_ic={self.sym_ic}")
        print(f"\tparams={self.params}")
        print(f"\tinputs={self.inputs}")
        print("\toutp_exprs:")
        for outp_var, outp_expr in self.outp_exprs.items():
            print(f"\t\t{outp_var}:{outp_expr}")
        print(f"\tzcs={self.zcs}")
        print(f"end {self.diagram.name}\n")

    def pp_nodepots(self):
        if not self.verbose:
            return
        print("=================== pp nodepots")
        for nodepots in self.pot_alias_map.values():
            for der_idx, n in nodepots.items():
                print(f"{n.name}. ic={n.ic}")

    def eqs_append(self, eq):
        self.eqs[len(self.eqs)] = eq

    def syms_append(self, syms_list):
        for sym in syms_list:
            self.syms[self.next_sym_idx] = sym
            self.next_sym_idx += 1

    def update_syms(self):
        # find all unique symbols in all active equations,
        # rebuild self.syms from the symbols found.
        remaining_syms = set()
        for eq in self.eqs.values():
            symbols = eq.e.atoms(sp.Symbol)
            symbols.discard(self.eqn_env.t)
            fcns = eq.e.atoms(sp.Function)
            remaining_syms.update(symbols)
            remaining_syms.update(fcns)

        remaining_syms = list(remaining_syms)
        self.syms = {i: s for i, s in self.syms.items() if s.s in remaining_syms}

    def get_some_syms(self, eqn_kind_filter: list = None):
        # similar to update_syms, but has filter to only find
        # a subset of the symbols based on their SymKind.
        remaining_syms = set()
        for eq in self.eqs.values():
            if eq.kind in eqn_kind_filter:
                continue
            symbols = eq.e.atoms(sp.Symbol)
            symbols.discard(self.eqn_env.t)
            fcns = eq.e.atoms(sp.Function)
            remaining_syms.update(symbols)
            remaining_syms.update(fcns)

        remaining_syms = list(remaining_syms)
        syms_new = {i: s for i, s in self.syms.items() if s.s in remaining_syms}

        return syms_new

    def update_syms_map(self, do_original=False):
        # update the inverse mapping from Sympy Symbol to Sym object.
        if do_original:
            self.syms_map_original = {s.s: s for s in self.syms.values()}
        else:
            self.syms_map = {s.s: s for s in self.syms.values()}

    def eqs_subs(self, sym, sub_expr):
        # given a substitution pair, perform the substitution in any
        # active equation where the substitution is applicable.
        # for any instances of function symbols appearing in an equation
        # which have as their args, the symbol being replace, this sub
        # will be applied. this means that the instance of the function
        # symbol will differ from the symbol key in the self.syms dict.
        # syms_subs() below is meant to fix this discrepancy.
        for i, eq in self.eqs.items():
            self.eqs[i] = eq.subs(sym, sub_expr)

    def syms_subs(self, sym, sub_expr):
        # given a substitution pair, perform the subsitution in any
        # symbol where the substitution is applicable.
        # for example, the symbol for a lookup table function is:
        #   interp(x,xp,fp)
        # if the substitution pair is x->a, then this function will
        # update the value in self.syms so that the value matches any
        # instance of the value's symbol in the equations.
        for i, s in self.syms.items():
            if s.kind in [SymKind.lut, SymKind.cond]:
                del self.syms_map[s.s]
                s_new = s.subs(sym, sub_expr)
                self.syms[i] = s_new
                self.syms_map[s.s] = s_new

    def zcs_subs(self, sym, sub_expr):
        # ZC expressions are kept separate from system equations
        # because they serve a different purpose. ZC expressions
        # are written in the components symbols, some of which may
        # get replaced during alias elimination. As such, we need
        # to perform the same substitutions in the ZC exprs such
        # that they also can be evauated in the lambdify functions.
        zcs_new = {}
        for idx, zc_tuple in self.zcs.items():
            zc_expr, zc_dir, is_bool_expr = zc_tuple
            zc_expr_new = zc_expr.subs(sym, sub_expr)
            zc_tuple = (zc_expr_new, zc_dir, is_bool_expr)
            zcs_new[idx] = zc_tuple

        self.zcs = zcs_new

    def check_port_set_domain(self, port_set, node_id):
        # T-104-followup-acausal-pantelides-units: also check
        # ``flow_units`` / ``pot_units`` consistency alongside the
        # domain enum check. Today every same-domain port carries
        # identical class-level units (e.g. all ThermalPorts share
        # ``flow_units=ACAUSAL_UNIT_WATT, pot_units=ACAUSAL_UNIT_KELVIN``),
        # so this is defensive infrastructure rather than a check that
        # fires on current code; threading it now means per-instance
        # unit overrides (e.g. a temperature sensor reporting in °C
        # connected to a thermal mass in K) will be caught at compile
        # time rather than silently miscompiling.
        domain = None
        flow_units = None
        pot_units = None
        reference_comp = None
        reference_port_name = None
        for port_tuple in port_set:
            comp, port_name = port_tuple
            port = comp.ports[port_name]
            if domain is None:
                domain = port.domain
                flow_units = getattr(port, "flow_units", None)
                pot_units = getattr(port, "pot_units", None)
                reference_comp = comp
                reference_port_name = port_name
            else:
                if domain != port.domain:
                    ports = list(port_set)
                    message = "These connected component ports have mismatched domains."
                    raise AcausalModelError(
                        message=message,
                        ports=ports,
                        include_port_domain=True,
                        dpd=self.dpd,
                    )
                this_flow = getattr(port, "flow_units", None)
                this_pot = getattr(port, "pot_units", None)
                if flow_units is not None and this_flow is not None and flow_units != this_flow:
                    raise AcausalModelError(
                        message=(
                            f"Connected ports in the {domain.name if hasattr(domain, 'name') else domain} "
                            f"domain disagree on flow units: "
                            f"{reference_comp.name}.{reference_port_name} uses "
                            f"{flow_units.name if hasattr(flow_units, 'name') else flow_units!r} but "
                            f"{comp.name}.{port_name} uses "
                            f"{this_flow.name if hasattr(this_flow, 'name') else this_flow!r}."
                        ),
                        ports=list(port_set),
                        include_port_domain=True,
                        dpd=self.dpd,
                    )
                if pot_units is not None and this_pot is not None and pot_units != this_pot:
                    raise AcausalModelError(
                        message=(
                            f"Connected ports in the {domain.name if hasattr(domain, 'name') else domain} "
                            f"domain disagree on potential units: "
                            f"{reference_comp.name}.{reference_port_name} uses "
                            f"{pot_units.name if hasattr(pot_units, 'name') else pot_units!r} but "
                            f"{comp.name}.{port_name} uses "
                            f"{this_pot.name if hasattr(this_pot, 'name') else this_pot!r}."
                        ),
                        ports=list(port_set),
                        include_port_domain=True,
                        dpd=self.dpd,
                    )

        return domain

    def add_sym(self, sym):
        self.syms[self.next_sym_idx] = sym
        self.next_sym_idx += 1

    def _update_dpd(self):
        # update dpd with latest
        self.dpd = DiagramProcessingData(
            self.diagram,
            self.syms,
            self.syms_map_original,
            self.nodes,
            self.node_domains,
            self.pot_alias_map,
            self.alias_map,
            self.aliaser_map,
            self.params,
        )

    def _print_ics(self, step="0"):
        # use for debugging only
        if not self.verbose:
            return
        print("\n\n" + step + "ICS " * 20)
        for i, sym in self.syms.items():
            print(f"{i=} {sym.name} {type(sym.ic)=}")

    # methods for diagram processing start here.
    def identify_nodes(self):
        """
        This function does the following;
         - identify the nodes of the network. nodes are junctions between two or more components.
         - verify that all ports connected to the node are of the same domain.

        Naive algorithm to sort port-pairs into sets of ports belonging to a node in the network.
        Start by assuming each port-pair is its own node, i.e. a network with components connected in a line.
        Initialize a dict 'nodes' by enumerating each of these into a dict{1:set1,2:set2, ...etc.}
        Initialize 'workset' as a list nodes.keys(), e.g. a list fo all the node IDs.
        Then, pop an ID from workset, and pop the corresponding node(N) from 'nodes'.
            check if any other node shares a port with N,
                if say node M shares a port with N, merge all ports of node N into node M, leaving M in 'nodes'.
                if no other nodes share a port with N, re-add N to workset, because it is a complete node.
        If workset empty, we are done.
        """
        # print("AcausalCompiler.identify_nodes()")
        nodes = {
            id: set(port_pair) for id, port_pair in enumerate(self.diagram.connections)
        }
        workset = list(nodes.keys())
        while workset:
            this_id = workset.pop()
            this_port_set = nodes.pop(this_id)
            # print(f'this_id={this_id}, this_port_set={this_port_set}, workset={workset}')
            this_set_grouped = False
            for node_id, port_set in nodes.items():
                if not port_set.isdisjoint(this_port_set):
                    nodes[node_id].update(this_port_set)
                    this_set_grouped = True

            if not this_set_grouped:
                nodes[this_id] = this_port_set
                self.node_domains[this_id] = self.check_port_set_domain(
                    this_port_set, this_id
                )

        # print(f"self.node_domains={self.node_domains}")
        if self.verbose:
            print(f"{nodes=}")
        self.nodes = nodes
        # create effectively the inverse mapping as well
        for node_id, port_set in self.nodes.items():
            for port_tuple in port_set:
                cmp, port_id = port_tuple
                cmp_dict = self.port_to_node.get(cmp, {})
                cmp_dict[port_id] = node_id
                self.port_to_node[cmp] = cmp_dict

    def identify_fluid_networks(self):
        """
        Networks of fluid components are special relative to other domains in that
        they are incomplete unless they have fluid properties defined.
        It is allowed to have 2 or more fluid networks interacting (e.g. exchanging energy, but not mixing fluids),
        and each network needs its fluid properties defined. To make this easy for the
        user, there is a FluidProperties component which the user 'connects'
        to the fluid network just like any other block. There must be one and
        only one FluidProperties block per fluid network. Some fluid components
        have equations which rely on fluid properties like density, etc. Therefore
        each component connected to fluid network is passed the fluid_spec from
        the FluidProperties component connected to its network. In order to pass
        the fluid_spec to each compoenent, we need to:
            1] identify all segregated fluid networks
            2] collect a list/set of the fluid components in each network
            3] ensure one and only one FluidProperties component is connected to a given network
            4] X. copy the fluid_spec from the FluidProperties component to the
            other components.
            4] create fluid state equations for each node in a fluid network.
            these are sort of like node_pot variables, but

        In the case of multiple fluids interacting, this requires components that have ports
        for fluid_A and fluid_B, and that these ports be correctly identified in the components'
        fluid_port_sets attribute.

        Eventually, multi-substance fluid models will be supported. But this doens't mean there aren't
        use cases for segregated fluid networks: e.g. glycol cooling netrwork, and refrigerant cooling
        network, interacting through a chiller, i.e. fluid-fluid heat exchanger.
        """
        dbg_print = self.verbose and False
        fprop_clss = (FluidProperties, HydraulicProperties)
        fp_cmps = [c for c in self.diagram.comps if isinstance(c, fprop_clss)]
        if dbg_print:
            print(f"DiagramProcessing.identify_fluid_networks()\n{fp_cmps=}")
        fld_nodes = set()
        for fp_cmp in fp_cmps:
            if dbg_print:
                print(f"\n{fp_cmp=}")
            node_id = self.port_to_node[fp_cmp]["prop"]
            node_set = set([node_id])
            # we need to walk the graph to find all node, and their port_sets, for which
            # this FluidProp has domain over.
            while_lim = 10
            while_cnt = 0
            workset = set([node_id])
            while workset:
                if dbg_print:
                    print(f"{workset=}")
                if while_cnt >= while_lim:
                    msg = "fluid properties propagation too many loops."
                    raise AcausalCompilerError(message=msg, dpd=self.dpd)
                node_id = workset.pop()
                port_set = self.nodes[node_id]
                for port_tuple in port_set:
                    cmp, port_id = port_tuple
                    if dbg_print:
                        print(f"\t{cmp}.{port_id}")
                    for cmp_port_set in cmp.fluid_port_sets:
                        if dbg_print:
                            print(f"\t\t{cmp_port_set=}")
                        if port_id in cmp_port_set:
                            if dbg_print:
                                print(f"\t\t{port_id=}")
                            for other_port_id in cmp_port_set:
                                if dbg_print:
                                    print(f"\t\t\t{other_port_id=}")
                                new_node = self.port_to_node[cmp][other_port_id]
                                if new_node not in node_set:
                                    # FIXME: we should be adding already added node in theb first place.
                                    workset.add(new_node)
                                node_set.add(new_node)
                while_cnt += 1
            # check if any of the nodes for this fluid network have been assigned to
            # another fluid network already, if so, raise error.
            if fld_nodes & node_set:
                message = "Detected fluid components with ports connected to multiple FluidProperties."
                ports = []
                for nid in node_set:
                    ports = ports + list(self.nodes[nid])
                raise AcausalModelError(
                    message=message,
                    ports=ports,
                    dpd=self.dpd,
                )
            fld_nodes.update(node_set)
            self.fld_nws[fp_cmp] = node_set

        if dbg_print:
            print(f"{self.fld_nws=}")

        # assign the fluid props to all the ports connected to each network.
        for fp, node_ids in self.fld_nws.items():
            for node_id in node_ids:
                port_set = self.nodes[node_id]
                # print(f"{node_id=} {port_set=}")
                for port_tuple in port_set:
                    cmp, port_id = port_tuple
                    # print(f"{cmp=} {port_id=}")
                    cmp.ports[port_id].assign_fluid(fp.fluid)

    def finalize_diagram(self):
        """
        1] Calls the finalize() method for each component.
        2] Populates symbol and equation data for diagram_processing.
        """
        for cmp in self.diagram.comps:
            cmp.finalize(self.eqn_env)
            self.diagram.add_cmp_sympy_syms(cmp)
            self.diagram.syms.update(cmp.syms)
            self.diagram.eqs.update(cmp.eqs)

        for fp in self.fld_nws:
            # now that all components have been finalized, this means all fluid models
            # have added their fluid state equations in all the necessary places. Some
            # fluid models create new Sym objects when do that, so we need to add those
            # newly created Sym objects to diagram processing list of 'syms'.
            if fp.fluid.syms:
                self.syms_append(fp.fluid.syms)

        syms = list(self.diagram.syms)
        self.syms_append(syms)
        self.next_sym_idx = len(self.syms.keys())
        self.eqs = {idx: e for idx, e in enumerate(self.diagram.eqs)}
        self.update_syms_map()

        # print(f"\n{self.syms=}")
        # print(f"\n{self.eqs=}")
        # print(f"\n{self.syms_map=}")

    def add_node_flow_eqs(self):
        """
        For each node in the system, generate the flow equations.
        sum(all flow syms) = 0
        """

        if self.verbose:
            print("\n\nAcausalCompiler.add_node_flow_eqs()")
            print(f"{self.nodes=}")
        for node_id, port_set in self.nodes.items():
            flow_syms_on_node = set()

            for port_tuple in port_set:
                cmp, port_id = port_tuple
                # HACK: this is a bit hacky. FluidProperties components have no
                # flow nor potential symbols, so we allow skipping them here.
                if isinstance(cmp, FluidProperties):
                    continue

                port = cmp.ports[port_id]

                # collect the flow symbols to create balancing equation for node
                flow_syms_on_node.add(port.flow.s)

            # create and save the balancing equation for the node
            sum_expr = sp.core.add.Add(*flow_syms_on_node)
            eq = Eqn(e=sp.Eq(0, sum_expr), kind=EqnKind.flow, node_id=node_id)
            self.eqs_append(eq)

    def add_stream_balance_eqs(self):
        if self.verbose:
            print("\n\nAcausalCompiler.add_stream_balance_eqs()")
            print(f"{self.nodes=}")
        for fp, node_ids in self.fld_nws.items():
            if self.verbose:
                print(f"\n\n{fp=} {node_ids=}")
            if isinstance(fp, FluidProperties):
                for node_id in node_ids:
                    port_set = self.nodes[node_id]

                    # remove any FluidProperties components from port_sets.
                    # FIXME: maybe this should be done when fluid props are assigned to comp ports?
                    port_set_mflow = set()
                    port_set_no_mflow = set()
                    for port_tuple in port_set:
                        cmp, port_id = port_tuple
                        if not isinstance(cmp, FluidProperties):
                            if cmp.ports[port_id].no_mflow:
                                port_set_no_mflow.add(port_tuple)
                            else:
                                port_set_mflow.add(port_tuple)

                    if self.verbose:
                        print(
                            f"\t{node_id=}\n\t\t{port_set=}\n\t\t{port_set_mflow=}\n\t\t{port_set_no_mflow=}\n"
                        )
                    if len(port_set_mflow) <= 1:
                        # each fluid connection must have >=2 ports that allow mass flow
                        # this is sort of like special case D.3.1 of https://web.mit.edu/crlaugh/Public/stream-docs.pdf
                        # I'm chosing not to implement support for this case right now.
                        raise AcausalModelError(
                            message="not enough ports have non-zero mass flow allowed.",
                            ports=list(port_set),
                            dpd=self.dpd,
                        )
                    elif len(port_set_mflow) == 2:
                        if self.verbose:
                            print("\t\tspecial case for cmp1<=>cmp2 connection")
                        # special case for cmp1<=>cmp2 connection
                        # covered in D.3.2 of https://web.mit.edu/crlaugh/Public/stream-docs.pdf
                        port_list = list(port_set_mflow)
                        port_tuple_1 = port_list[0]
                        port_tuple_2 = port_list[1]
                        cmp_1, port_id_1 = port_tuple_1
                        cmp_2, port_id_2 = port_tuple_2
                        h_outflow_1 = cmp_1.ports[port_id_1].h_outflow.s
                        h_inStream_1 = cmp_1.ports[port_id_1].h_inStream.s

                        h_outflow_2 = cmp_2.ports[port_id_2].h_outflow.s
                        h_inStream_2 = cmp_2.ports[port_id_2].h_inStream.s

                        eq = Eqn(
                            e=sp.Eq(h_inStream_1, h_outflow_2),
                            kind=EqnKind.stream,
                            node_id=node_id,
                        )
                        self.eqs_append(eq)
                        eq = Eqn(
                            e=sp.Eq(h_inStream_2, h_outflow_1),
                            kind=EqnKind.stream,
                            node_id=node_id,
                        )
                        self.eqs_append(eq)
                        if port_set_no_mflow:
                            if self.verbose:
                                print("\t\tspecial case for no mflow sensors")
                            # there is one or more sensor ports connected here. this is
                            # special case D.3.3 of https://web.mit.edu/crlaugh/Public/stream-docs.pdf
                            # note, every sensor port connected here has the same inStream equation, so
                            # we make just one equation, and a bunch of aliases.
                            mflow_1 = cmp_1.ports[port_id_1].mflow.s
                            mflow_2 = cmp_2.ports[port_id_2].mflow.s
                            mflow_max_expr_1 = sp.Piecewise(
                                (-mflow_1, -mflow_1 > M_FLOW_EPS), (M_FLOW_EPS, True)
                            )
                            mflow_max_expr_2 = sp.Piecewise(
                                (-mflow_2, -mflow_2 > M_FLOW_EPS), (M_FLOW_EPS, True)
                            )
                            num_expr = (
                                mflow_max_expr_1 * h_outflow_1
                                + mflow_max_expr_2 * h_outflow_2
                            )
                            den_expr = mflow_max_expr_1 + mflow_max_expr_2

                            # identified as SymKind.stream to minimize chance it is replaced by an alias
                            sensor_inStream = Sym(
                                self.eqn_env,
                                name=f"np{node_id}_sensor_inStream",
                                kind=SymKind.stream,
                            )
                            self.add_sym(sensor_inStream)
                            eq = Eqn(
                                e=sp.Eq(sensor_inStream.s, num_expr / den_expr),
                                kind=EqnKind.stream,
                                node_id=node_id,
                            )
                            self.eqs_append(eq)
                            # ... and the aliases.
                            for sensor_port_tuple in port_set_no_mflow:
                                sensor, sensor_port_id = sensor_port_tuple
                                if self.verbose:
                                    print(f"\t\t\t{sensor=} {sensor_port_id=}")
                                eq = Eqn(
                                    e=sp.Eq(
                                        sensor_inStream.s,
                                        sensor.ports[sensor_port_id].h_inStream.s,
                                    ),
                                    kind=EqnKind.stream,
                                    node_id=node_id,
                                )
                                self.eqs_append(eq)

                    else:
                        if self.verbose:
                            print("\t\tgeneral case cover in section D.2")
                        # general case cover in section D.2
                        for port_tuple_i in port_set_mflow:
                            cmp_i, port_id_i = port_tuple_i
                            if self.verbose:
                                print(f"\t\t{cmp_i=} {port_id_i=}")
                            h_inStream_i = cmp_i.ports[port_id_i].h_inStream.s
                            port_set_j = port_set_mflow.difference({port_tuple_i})
                            num_terms = []
                            den_terms = []
                            for port_tuple_j in port_set_j:
                                cmp_j, port_id_j = port_tuple_j
                                if self.verbose:
                                    print(f"\t\t\t{cmp_j=} {port_id_j=}")
                                # mflow_syms.append(cmp_j.ports[port_id_j].mflow)
                                # h_outflow_syms.append(cmp_j.ports[port_id_j].h_outflow)
                                mflow_j = cmp_j.ports[port_id_j].mflow.s
                                h_outflow_j = cmp_j.ports[port_id_j].h_outflow.s

                                mflow_max_expr_j = sp.Piecewise(
                                    (-mflow_j, -mflow_j > M_FLOW_EPS),
                                    (M_FLOW_EPS, True),
                                )

                                num_term = mflow_max_expr_j * h_outflow_j
                                num_terms.append(num_term)
                                den_terms.append(mflow_max_expr_j)

                            num_expr = sp.core.add.Add(*num_terms)
                            den_expr = sp.core.add.Add(*den_terms)
                            if self.verbose:
                                print(f"\t\t{num_expr=}\n{den_expr=}")
                            eq = Eqn(
                                e=sp.Eq(h_inStream_i, num_expr / den_expr),
                                kind=EqnKind.stream,
                                node_id=node_id,
                            )
                            self.eqs_append(eq)

    def add_node_potential_eqs(self):
        """
        For each node in the system, generate the potential variable constraint
        equations.

        Constraints are made between the potential variable of the node, and the
        potential variable of a component connected to the node. So if components
        A, B, C are connected to a node, we will generate the following contraint
        equations: Np=Ap, Np=Bp, Np=Cp. Where Np is the node potential variable.

        Additionally, we need to create these constraints for each of the variables
        in the 'derivative index'.

        The 'derivative index' of a potential variable is a measure of how many derivatives there
        are of the underlying variable, for which the potential variable is either that underlying
        variable, or one of its derivatives.

        This is best explained by examples.
        MechTrans: the potential variable is velocity; however, the underlying variable is position,
        and the derivative index includes the acceleration.
        Elec: the potential variable is the pin voltage (not the voltage across the component),
        this is also the underlying variable, and there are no further derivatives.

        If we think of derivatives in an ordinal sense, and say that the potential variable is
        0, then for the examples above, the 'derivative index' are:
        MechTrans: [-1,1] i.e. position is -1 because it is an integral of potential variable velocity,
        and acceleration is 1 because it is a derivative of velocity.
        Elec: [0,0] i.e. the terminal/pin voltage has no integrals nor derivatives defined.

        The 'derivative index' of potential variables is required because it defines the set of
        contraint equations required for a given node. Continuing with the exmaples:
        MechTrans: if components A and B are connected at the node, we need 3 constraint equations,
        Ax=Bx, Av=Bv, Aa=Ba. This means that the initial conditions for these must be consistent as well.
        Elec: if components C and D are connected at the node, we need 1 constraint equation,
        C_volts = D_volts. Recall, C_volts and D_volts are pin voltages, not voltages across components.

        However, rather than create constraints between components directly, we do so between the node
        variables and the components variables. These 'constrains' are also alias equations, which
        means that during alias elimination, we will remove all the components potential variables,
        retaining only the node potential variables. Of course nothing is lost, because via the
        compiler alias_map, we can always know the value of any component potential variable.

        """
        if self.verbose:
            print("\n\nAcausalCompiler.add_node_potential_eqs()")
            print(f"{self.nodes=}")
        for node_id, port_set in self.nodes.items():
            node_domain = self.node_domains[node_id]
            np_qty_name = nodpot_qty(node_domain)
            np_base_name = f"np{node_id}_{str(node_domain)}_" + np_qty_name
            # start defining potential symbols family for the node
            node_alias_map = {}
            if self.verbose:
                print(f"\t{np_base_name=} {port_set=}")
            for port_tuple in port_set:
                cmp, port_id = port_tuple
                if self.verbose:
                    print(f"\t\t{cmp=} {port_id=}")
                # HACK: this is a bit hacky. FluidProperties components have no
                # flow nor potential symbols, so we allow skipping them here.
                if isinstance(cmp, FluidProperties):
                    continue

                port = cmp.ports[port_id]

                # then all the 'integrals' of the potential variable
                this_var = port.pot
                der_idx = -1
                while this_var.int_sym is not None:
                    if der_idx not in node_alias_map.keys():
                        node_pot = Sym(
                            self.eqn_env,
                            name=np_base_name + f"_n{abs(der_idx)}",
                            kind=SymKind.node_pot,
                        )
                        self.add_sym(node_pot)
                        node_alias_map[der_idx] = node_pot
                    else:
                        node_pot = node_alias_map[der_idx]
                    self.eqs_append(
                        Eqn(
                            e=sp.Eq(node_pot.s, this_var.int_sym.s),
                            kind=EqnKind.pot,
                            node_id=node_id,
                        )
                    )
                    der_idx -= 1
                    this_var = this_var.int_sym

                # do the pot var here so dict indices are naturally in order.
                # this is temporary. eventually we can sort them.
                if 0 not in node_alias_map.keys():
                    node_pot = Sym(
                        self.eqn_env,
                        name=np_base_name + "_0",
                        kind=SymKind.node_pot,
                    )
                    self.add_sym(node_pot)
                    node_alias_map[0] = node_pot
                else:
                    node_pot = node_alias_map[0]
                # constraint equation for the potential variable
                self.eqs_append(
                    Eqn(
                        e=sp.Eq(node_pot.s, port.pot.s),
                        kind=EqnKind.pot,
                        node_id=node_id,
                    )
                )

                # then all the 'derivatives' of the potential variable
                this_var = port.pot
                der_idx = 1
                while this_var.der_sym is not None:
                    if der_idx not in node_alias_map.keys():
                        node_pot = Sym(
                            self.eqn_env,
                            name=np_base_name + f"_p{abs(der_idx)}",
                            kind=SymKind.node_pot,
                        )
                        self.add_sym(node_pot)
                        node_alias_map[der_idx] = node_pot
                    else:
                        node_pot = node_alias_map[der_idx]
                    self.eqs_append(
                        Eqn(
                            e=sp.Eq(node_pot.s, this_var.der_sym.s),
                            kind=EqnKind.pot,
                            node_id=node_id,
                        )
                    )
                    der_idx += 1
                    this_var = this_var.der_sym

            self.pot_alias_map[node_id] = node_alias_map

            # update the int_sym and der_sym of the newly created pot vars
            if self.verbose:
                print(f"{node_alias_map=}")
            pot_min = min(node_alias_map.keys())
            pot_max = max(node_alias_map.keys())
            # go from underlying variable to highest derivative, adding the der_sym
            for der_idx in range(pot_min, pot_max):
                this_pot = node_alias_map[der_idx]
                this_pot.der_sym = node_alias_map[der_idx + 1]
            # go from highest derivative, to underlying variable, adding the int_sym
            for der_idx in range(pot_max, pot_min, -1):
                this_pot = node_alias_map[der_idx]
                this_pot.int_sym = node_alias_map[der_idx - 1]

            # denug print
            # print("\n+++++++++++++++++++++++++")
            # for idx, np in node_alias_map.items():
            #     print(
            #         f"\tder_idx={der_idx} der_sym={np.der_sym} np={np} int_sym={np.int_sym}"
            #     )

        # print(f"pot_alias_map={self.pot_alias_map}")

    def add_derivative_relations(self):
        """
        iterate over symbols and collect any derivative relation equations.
        """
        for sym in self.syms.values():
            if sym.der_relation is not None:
                self.eqs_append(eq=sym.der_relation)

    def finalize_equations(self):
        """
        The addition of equations to the system is complete.
        This function just records the status such that at any
        point we can always see what were all the equations.
        """
        self.update_syms_map()
        self.update_syms_map(do_original=True)
        self.eqs_original = copy.deepcopy(self.eqs)

    def collect_zcs(self):
        idx = 0
        for cmp in self.diagram.comps:
            for zc_tuple in cmp.zc.values():
                self.zcs[idx] = zc_tuple
                idx += 1

    def _t036f_check_alias_conflicts(self):
        """T-036f-followup-alias-conflicts: pre-elimination conflict scan.

        Walks ``self.eqs`` and identifies every equation the downstream
        loop would treat as an alias declaration (forms ``a=b``, ``a=-b``,
        ``0=a±b``, ``a±b=0``).  Tentatively computes each candidate's
        aliasee + sub_expr and groups by aliasee with the same
        other-symbol set.  Two distinct sub_exprs (under ``sp.expand``)
        for the same (aliasee, other-syms) pair raise ``AcausalModelError``
        naming both component sources, surfacing the root cause instead of
        a downstream IndexReduction count mismatch.  Equivalent
        declarations are treated as harmlessly redundant.

        Synthesised flow / stream / pot eqs are skipped: the elimination
        loop chains them across the network by design (e.g. ``0=a+b`` and
        ``0=a+c`` collapse to ``b=c`` — not a user-level conflict).
        """
        eq_id_to_cmp = {
            id(eq): cmp.name
            for cmp in self.diagram.comps
            for eq in cmp.eqs.keys()
        }
        sp_bool_types = (sp.logic.boolalg.BooleanTrue, sp.logic.boolalg.BooleanFalse)

        def _pri(symbols):
            inp_or_param = (SymKind.inp, SymKind.param)
            ordered = sorted(symbols, key=str)
            if len(ordered) < 2:
                return ordered[0]
            sym0, sym1 = ordered[0], ordered[1]
            s0s = self.syms_map.get(sym0)
            s1s = self.syms_map.get(sym1)
            if s0s is None or s1s is None:
                return sym0
            if s0s.kind in inp_or_param:
                return sym1
            if s1s.kind in inp_or_param:
                return sym0
            if s0s.kind == SymKind.node_pot:
                return sym1
            if s1s.kind == SymKind.node_pot:
                return sym0
            return sym0

        # candidates[alias_sym] = list[(cmp_name, sub_expr, eq, other_syms)]
        candidates: dict = {}
        for eq in self.eqs.values():
            if eq.kind in (EqnKind.outp, EqnKind.der_relation, EqnKind.pot):
                continue
            if isinstance(eq.e, sp_bool_types):
                continue
            if isinstance(eq.e.lhs, sp_bool_types) or isinstance(
                eq.e.rhs, sp_bool_types
            ):
                continue
            try:
                lhs_s = eq.e.lhs.atoms(sp.Symbol)
                lhs_s.discard(self.eqn_env.t)
                lhs_f = eq.e.lhs.atoms(sp.Function)
                rhs_s = eq.e.rhs.atoms(sp.Symbol)
                rhs_s.discard(self.eqn_env.t)
                rhs_f = eq.e.rhs.atoms(sp.Function)
            except (AttributeError, TypeError):
                continue
            ln, rn = len(lhs_s) + len(lhs_f), len(rhs_s) + len(rhs_f)
            t0 = ln == 1 and rn == 1
            t1 = eq.e.lhs == 0 and getattr(eq.e.rhs, "is_Add", False) and rn == 2
            t2 = eq.e.rhs == 0 and getattr(eq.e.lhs, "is_Add", False) and ln == 2
            t3 = eq.e.lhs == 0 and rn == 1
            t4 = eq.e.rhs == 0 and ln == 1
            if not (t0 or t1 or t2 or t3 or t4):
                continue
            syms = lhs_s | rhs_s | lhs_f | rhs_f
            try:
                a = _pri(syms)
                sub = self.sanitize_solve(a, eq)
            except Exception:  # pragma: no cover - best-effort
                continue
            cmp_name = eq_id_to_cmp.get(id(eq))
            if cmp_name is None:
                continue
            candidates.setdefault(a, []).append(
                (cmp_name, sub, eq, frozenset(syms - {a}))
            )

        # Conflict iff same (aliasee, other-syms) bucket carries multiple
        # distinct sub_exprs (canonicalised via ``sp.expand``).
        conflicts = []
        for a, decls in candidates.items():
            if len(decls) < 2:
                continue
            by_others: dict = {}
            for d in decls:
                by_others.setdefault(d[3], []).append(d)
            for group in by_others.values():
                if len(group) < 2:
                    continue
                buckets: dict = {}
                for cmp_name, sub, eq, _ in group:
                    try:
                        key = str(sp.expand(sub))
                    except Exception:  # pragma: no cover - best-effort
                        key = str(sub)
                    buckets.setdefault(key, []).append((cmp_name, sub, eq))
                if len(buckets) > 1:
                    conflicts.append((a, buckets))

        if not conflicts:
            return

        lines = []
        for alias_sym, buckets in conflicts:
            lines.append(
                f"Inconsistent alias declarations for canonical variable "
                f"'{alias_sym}':"
            )
            for decls in buckets.values():
                for cmp_name, sub_expr, eq in decls:
                    lines.append(
                        f"  - {cmp_name or '<unknown>'} declares "
                        f"'{alias_sym} = {sub_expr}' (from equation: {eq.e})"
                    )
            lines.append(
                "These expressions are not equivalent under sympy.expand. "
                "Either remove the redundant alias from one component, or "
                "align the expressions (e.g. flip a sign convention)."
            )
        message = "\n".join(lines)
        raise AcausalModelError(message=message, dpd=self.dpd)

    def _t036f_check_write_time_alias_conflict(
        self, aliaser, aliasee, aliasee_expr, eq, aliasee_list
    ):
        """T-036f-followup-cross-component-aliases: write-time guard.

        Called from inside ``alias_elimination`` immediately before each
        ``aliaser_map[aliaser].append((aliasee, aliasee_expr))``.  After
        pot/flow elimination collapses pin-potentials into a shared
        ``np_node`` (or otherwise canonical) symbol, a *cross-component*
        conflict may emerge: two component-internal equations both alias
        the same downstream variable to the same aliaser but with
        contradictory expressions (e.g. ``+aliaser`` vs ``-aliaser``).
        The pre-elimination scan in ``_t036f_check_alias_conflicts``
        cannot see this case — at pre-pass time the two component pin
        symbols are still distinct and only become identified after the
        pot/flow elimination loop substitutes them away.

        This check compares the incoming ``aliasee_expr`` against any
        existing entry in ``aliasee_list`` whose ``aliasee`` symbol
        matches the new one.  If the same aliasee already has a
        structurally non-equal expression under ``sp.expand``, raise an
        ``AcausalModelError`` naming the source component of each
        conflicting equation and the canonical aliaser variable.

        Honest scope: a "true" cross-component sign-flip (component A:
        ``V = Vp - Vn``; component B: ``V = Vn - Vp`` connected to the
        same nodes) does NOT trip this guard, because A and B alias
        *distinct* aliasees (``a_V`` vs ``b_V``) — the contradiction
        manifests downstream as a count mismatch, which T-036f's
        parallel-redundancy section already names.  The guard fires
        when two component equations alias the SAME canonical aliasee
        to incompatible expressions (e.g. via a shared parameter or a
        shared connector variable).
        """
        if not aliasee_list:
            return
        # Symbolic equality of the two expressions.  Use ``sp.expand`` so
        # ``a + b`` matches ``b + a`` and ``-(-x)`` matches ``x``.
        try:
            new_canon = sp.expand(aliasee_expr)
        except Exception:  # pragma: no cover - defensive
            new_canon = aliasee_expr
        for existing_aliasee, existing_expr in aliasee_list:
            if existing_aliasee is not aliasee:
                continue
            try:
                old_canon = sp.expand(existing_expr)
            except Exception:  # pragma: no cover - defensive
                old_canon = existing_expr
            try:
                equal = sp.simplify(new_canon - old_canon) == 0
            except Exception:  # pragma: no cover - defensive
                equal = str(new_canon) == str(old_canon)
            if equal:
                continue
            # Found a conflict — name the components for both sides.
            new_cmp = self._t036f_lookup_component_for_eq(eq)
            # We don't have the originating eq for ``existing_expr`` (the
            # write site discards it), so name the component of the
            # ``existing_aliasee`` symbol via ``sym_to_cmp`` as a
            # second-best signal.
            old_cmp = self._t036f_lookup_component_for_sym(existing_aliasee)
            aliaser_name = getattr(aliaser, "name", None) or str(
                getattr(aliaser, "s", aliaser)
            )
            aliasee_name = getattr(aliasee, "name", None) or str(
                getattr(aliasee, "s", aliasee)
            )
            lines = [
                f"Inconsistent alias declarations for canonical node "
                f"'{aliaser_name}':",
                f"  - {old_cmp or '<unknown>'} contributes "
                f"'{aliasee_name} = {existing_expr}'",
                f"  - {new_cmp or '<unknown>'} contributes "
                f"'{aliasee_name} = {aliasee_expr}'",
                "",
                "These two pin assignments are not equivalent under "
                "sympy.expand. Likely cause: the components' pin sign "
                "conventions disagree.  Either reverse the relevant "
                "`connect()` order, or align the components' pin polarity.",
            ]
            raise AcausalModelError(
                message="\n".join(lines), dpd=self.dpd
            )

    def _t036f_lookup_component_for_eq(self, eq):
        """Best-effort: find the component whose ``eqs`` dict owns ``eq``."""
        for cmp in self.diagram.comps:
            for ce in cmp.eqs.keys():
                if ce is eq:
                    return cmp.name
        return None

    def _t036f_lookup_component_for_sym(self, sym):
        """Best-effort: dereference a Sym to its owning component name."""
        sym_to_cmp = getattr(self.diagram, "sym_to_cmp", {}) or {}
        cmp = sym_to_cmp.get(sym)
        if cmp is not None:
            return cmp.name
        return None

    def _t036f_check_post_elim_sign_flip(self):
        """T-036f-followup-cross-component-aliases-architecture:
        post-elimination sign-flip scan over ``aliaser_map``.

        The semantic sign-flip case — two ElecTwoPin variants both
        connected as voltage sources to the same nodes but with
        opposite pin polarity (component A: ``Vp`` at node_top, ``Vn``
        at node_bot; component B: ``Vp`` at node_bot, ``Vn`` at
        node_top) — is NOT visible at any single ``alias_elimination``
        write site (the original entries alias *distinct* aliasees,
        e.g. ``a_V`` vs ``b_V``).  The contradiction surfaces only
        AFTER the elimination loop chains pot/flow substitutions to
        identify the two component-internal value parameters with each
        other through opposite signs, e.g. ``vs_a_v = +vs_a_V`` and
        ``vs_a_v = -vs_b_v`` end up as two entries on the SAME aliaser.

        Detection: walk ``aliaser_map`` for aliasers whose ``kind`` is
        ``param`` or ``inp`` (i.e. genuinely-known canonical values,
        the components are externally constraining via these symbols).
        For each such aliaser, examine its ``aliasee_list``: extract
        the sign of ``aliaser`` in every entry's ``aliasee_expr``
        relative to the dominant single-symbol sub-expression.  If the
        same aliaser has aliasees contributed by two DIFFERENT
        components with OPPOSITE signs, raise an
        ``AcausalModelError`` naming both components, the canonical
        aliaser variable, and listing each conflicting entry — this is
        the genuine over-determined sign-flip.

        False-positive guards:
          - Restricted to ``param``/``inp`` aliasers — sensors and
            other observation-only components produce ``outp``/``var``
            aliasees that don't constrain the system, so a sign-flip
            on a ``node_pot`` aliaser is normal (components were just
            wired with opposite pin orientations) and is intentionally
            skipped here.
          - Requires aliasees from ≥ 2 DIFFERENT components.  A single
            component aliasing both itself and a derived expression
            with mixed signs is an internal artefact of the
            elimination loop, not a user-level conflict.
          - If the originating components cannot be resolved via
            ``sym_to_cmp`` (e.g. synthesised aliasees), skip silently
            rather than misattribute.
        """
        sym_to_cmp = getattr(self.diagram, "sym_to_cmp", {}) or {}
        if not sym_to_cmp:
            return
        for aliaser, aliasee_list in list(self.aliaser_map.items()):
            if not aliasee_list or len(aliasee_list) < 2:
                continue
            if aliaser.kind not in (SymKind.param, SymKind.inp):
                continue
            # Collect (sign, aliasee, expr, component_name) per entry.
            entries = []
            for aliasee, aliasee_expr in aliasee_list:
                sign = self._t036f_expr_sign_relative_to(
                    aliasee_expr, aliasee
                )
                if sign == 0:
                    continue
                cmp_name = sym_to_cmp.get(aliasee)
                if cmp_name is not None:
                    cmp_name = getattr(cmp_name, "name", str(cmp_name))
                entries.append((sign, aliasee, aliasee_expr, cmp_name))
            if not entries:
                continue
            # Find any pair from DIFFERENT components with opposite signs.
            pos = [e for e in entries if e[0] > 0 and e[3] is not None]
            neg = [e for e in entries if e[0] < 0 and e[3] is not None]
            if not pos or not neg:
                continue
            conflict_pair = None
            for p_entry in pos:
                for n_entry in neg:
                    if p_entry[3] != n_entry[3]:
                        conflict_pair = (p_entry, n_entry)
                        break
                if conflict_pair:
                    break
            if conflict_pair is None:
                continue
            (_, p_aliasee, p_expr, p_cmp), (
                _,
                n_aliasee,
                n_expr,
                n_cmp,
            ) = conflict_pair
            aliaser_name = getattr(aliaser, "name", str(aliaser))
            lines = [
                f"Inconsistent alias declarations through canonical "
                f"variable '{aliaser_name}':",
                f"  - {p_cmp} constrains "
                f"'{aliaser_name} = +{p_expr}' "
                f"(via {getattr(p_aliasee, 'name', p_aliasee)})",
                f"  - {n_cmp} constrains "
                f"'{aliaser_name} = {n_expr}' "
                f"(via {getattr(n_aliasee, 'name', n_aliasee)})",
                "",
                f"Both components are externally constrained "
                f"(e.g. as voltage sources). The system is "
                f"over-determined: '{aliaser_name}' is forced to be "
                f"both +X and -X, which is consistent only at zero.",
                "",
                "Likely cause: the components' pin polarity disagrees "
                "(e.g. one ElecTwoPin variant sees `Vp - Vn` while the "
                "other sees `Vn - Vp` against the same node pair).  "
                "Either reverse the relevant `connect()` order, or "
                "align the components' pin polarity.",
            ]
            raise AcausalModelError(
                message="\n".join(lines), dpd=self.dpd
            )

    def _t036f_expr_sign_relative_to(self, expr, aliasee):
        """Return +1 if ``expr`` is structurally a positive multiple of
        a single symbol/function, -1 if a negative multiple, 0 if
        ambiguous (constants, multi-term sums, or unrecognised forms).

        Used to classify the sign of an ``aliasee_expr`` in an
        ``aliaser_map`` entry.  The elimination loop produces entries
        whose expression is one of:
          - the aliasee itself: ``aliasee(t)``           → +1
          - the negated aliasee: ``-aliasee(t)``         → -1
          - a single param/var symbol: ``some_sym(t)``   → +1
          - a negated single symbol: ``-some_sym(t)``    → -1
        Anything more complex (constant, multi-term, with offsets) is
        treated as ambiguous (returns 0) and excluded from the scan.
        """
        try:
            e = sp.expand(expr)
        except Exception:  # pragma: no cover - defensive
            return 0
        if e.is_Number:
            return 0
        # Single symbol / function form.
        atoms_s = e.atoms(sp.Symbol) - {self.eqn_env.t}
        atoms_f = e.atoms(sp.Function)
        if len(atoms_s) + len(atoms_f) != 1:
            return 0
        only = next(iter(atoms_s | atoms_f))
        try:
            coeff = sp.simplify(e / only)
        except Exception:  # pragma: no cover - defensive
            return 0
        if coeff == 1:
            return 1
        if coeff == -1:
            return -1
        return 0

    def alias_elimination(self):
        """
        Find equations of the form:
            type 0: a=b, a=-b
            type 1: 0=a+b, 0=a-b
            type 2: a+b=0, a-b=0
            type 3: 0=a
            type 4: a=0
        Replaces all a with b, or replaces a with 0,and removes the equation from system.
        The substitution process changes all equations which include a, this means it is
        possible the change transforms an equation that previously did not match any
        of the types above, but now does. This means we need to use a workset to ensure
        the process converges to a 'fixed point' where no more equations in the system match
        any of the types above.

        Note: it's not deterministic because Sympy returns sets,
        and therefore the it's not always the same replacements that happen, even for
        sequential runs with the same diagram.

        what about equations of the form:
            type 5: 1=a/g
            type 6: a=1/g
            type 5: const=a, a=const

        What about initial conditions? This is handled later, and since we keep a record of
        what aliased what, we can always go back and ensure any alias set has initial
        conditions verified for consistency and completeness.
        """

        self._update_dpd()

        # T-036f-followup-alias-conflicts: detect conflicting alias declarations
        # BEFORE running the elimination loop.  If two component equations both
        # alias the same canonical variable to non-equivalent expressions, the
        # existing loop silently lets the second equation collapse (e.g. into
        # ``V = -V`` → ``V = 0``), which downstream surfaces only as a generic
        # IndexReduction count mismatch.  This pre-pass groups alias-form
        # equations by their would-be aliasee and raises an AcausalModelError
        # naming the offending components when expressions disagree.
        self._t036f_check_alias_conflicts()

        def _pick_deterministic_symbol(symbols):
            # Keep aliasing deterministic across runs/platforms.
            return sorted(symbols, key=str)[0]

        def alias_priorities(symbols):
            # 'alias_sym' is the one being replaced everywhere by something else.
            # never replace Syms with kind = "params", or "in"
            inp_or_param = [SymKind.inp, SymKind.param]
            ordered = sorted(symbols, key=str)
            sym0 = ordered[0]
            if len(ordered) > 1:
                sym1 = ordered[1]
                s0s = self.syms_map[sym0]
                s1s = self.syms_map[sym1]
                # print(f"\t\t[alias_priorities] {s0s}:{s0s.kind} {s1s}:{s1s.kind}")
                if s0s.kind in inp_or_param:
                    return sym1
                elif s1s.kind in inp_or_param:
                    return sym0
                elif s0s.kind == SymKind.node_pot:
                    return sym1
                elif s1s.kind == SymKind.node_pot:
                    return sym0
                # elif s0s.kind == SymKind.stream:
                #     return sym1
                # elif s1s.kind == SymKind.stream:
                #     return sym0
                else:
                    return sym0  # arbitrary
            else:
                # the alias equation only had one sym, the other side was 0
                return sym0

        sp_bool_types = (sp.logic.boolalg.BooleanTrue, sp.logic.boolalg.BooleanFalse)

        workset = list(self.eqs.keys())
        while workset:
            eq_idx = workset.pop()
            eq = self.eqs[eq_idx]
            alias_sym = None
            if eq.kind == EqnKind.pot:
                # when dealing with potential variable aliases, always chose the
                # component var to be replaced. when potential equations are created,
                # they always have the node_pot on LHS, and comp_pot on RHS.
                rhs_fcns = eq.e.rhs.atoms(sp.Function)
                alias_sym = _pick_deterministic_symbol(rhs_fcns)
            elif isinstance(eq.e, sp_bool_types):
                # alias elimination has resulted in a equation like 0 = a - a = 0 -> BooleanTrue;
                # therefore, remove eqn from eqs.
                del self.eqs[eq_idx]
            elif eq.kind == EqnKind.der_relation:
                # derivative relation equations are alias equations by design.
                # derivative relations are alias equations that might need to be in the
                # final set of equations, so we do not consider them as opportunities
                # for simplification.
                pass
            elif eq.kind == EqnKind.outp:
                # output equations are alias equations by design. we want to keep them in the
                # system equations, but we want to perform all possible substitutions of the
                # output expression, so that output equations eventually are of the form
                # outp_sym = expr, where all symbols in expr are in the final equations
                pass
            else:
                # print("\n\n===========================")
                # print(f"{eq_idx} eq={eq}")
                # we should never have Derivative(f(t),t) in any 'system equations'.
                # 'system equations should use the alias symbol for Derivative(f(t),t),
                # and this alias relationship is resolved in the derivative relation equations.
                if len(eq.e.atoms(sp.Derivative)) > 0:
                    message = f"{eq_idx}: {eq} has symbols of type Sympy.Derivative. This is not allowed."
                    raise AcausalCompilerError(message=message, dpd=self.dpd)
                if isinstance(eq.e.lhs, sp_bool_types):
                    # when the equation is blah = a >= b, lhs is a boolean with no symbols
                    lhs_symbols = set()
                    lhs_fcns = set()
                else:
                    lhs_symbols = eq.e.lhs.atoms(sp.Symbol)
                    lhs_symbols.discard(self.eqn_env.t)
                    lhs_fcns = eq.e.lhs.atoms(sp.Function)

                if isinstance(eq.e.rhs, sp_bool_types):
                    # when the equation is a >= b = blah, rhs is a boolean with no symbols
                    rhs_symbols = set()
                    rhs_fcns = set()
                else:
                    rhs_symbols = eq.e.rhs.atoms(sp.Symbol)
                    rhs_symbols.discard(self.eqn_env.t)
                    rhs_fcns = eq.e.rhs.atoms(sp.Function)

                # print(f"lhs_symbols={lhs_symbols} rhs_symbols={rhs_symbols}")
                # print(f"lhs_fcns={lhs_fcns} rhs_fcns={rhs_fcns}")

                # conditions for type0: a=b, a=-b
                lhs_1_sym = len(lhs_symbols) + len(lhs_fcns) == 1
                rhs_1_sym = len(rhs_symbols) + len(rhs_fcns) == 1
                is_type0 = lhs_1_sym and rhs_1_sym

                # conditions for type1: 0=a+b, 0=a-b
                type1_sym_or_fun = len(rhs_symbols) + len(rhs_fcns) == 2
                is_type1 = eq.e.rhs.is_Add and eq.e.lhs == 0 and type1_sym_or_fun

                # conditions for type2: a+b=0, a-b=0
                type2_sym_or_fun = len(lhs_symbols) + len(lhs_fcns) == 2
                is_type2 = eq.e.lhs.is_Add and eq.e.rhs == 0 and type2_sym_or_fun

                # conditions for types3: 0=a
                is_type3 = eq.e.lhs == 0 and len(rhs_symbols) + len(rhs_fcns) == 1

                # conditions for types4: a=0
                is_type4 = eq.e.rhs == 0 and len(lhs_symbols) + len(lhs_fcns) == 1

                # alias elimination
                if is_type0 or is_type1 or is_type2 or is_type3 or is_type4:
                    self.alias_eqs.append(eq_idx)
                    eq_symbols = lhs_symbols | rhs_symbols | lhs_fcns | rhs_fcns
                    alias_sym = alias_priorities(eq_symbols)

            if alias_sym is not None:
                # print(f"alias_sym={alias_sym} eqn {eq.e}. ")
                # track the relationship: alias->substitution_expression
                alias_sub_expr = self.sanitize_solve(alias_sym, eq)
                self.alias_map[alias_sym] = alias_sub_expr

                # track the inverse relationship: substituter->alias_expression
                # it happens that the substituter is 0, in this case we don't track it.
                # the main reason we need this is as input to initial_condition_validation().
                # see the method documentation for more details.
                aliasee = self.syms_map[alias_sym]
                sub_expr_fcns = alias_sub_expr.atoms(sp.Function)
                sub_expr_symbols = alias_sub_expr.atoms(sp.Symbol)
                sub_expr_symbols.discard(self.eqn_env.t)
                aliaser_sym = None
                if sub_expr_fcns:
                    aliaser_sym = sub_expr_fcns.pop()
                elif sub_expr_symbols:
                    aliaser_sym = sub_expr_symbols.pop()

                if aliaser_sym is not None:
                    aliaser = self.syms_map[aliaser_sym]
                    # print(f"aliaser={aliaser} eqn {eq.e}. ")
                    if aliaser.kind == SymKind.node_pot:
                        aliasee_expr = alias_sym
                    else:
                        aliasee_expr = self.sanitize_solve(aliaser_sym, eq)
                    aliasee_list = self.aliaser_map.get(aliaser, [])
                    # T-036f-followup-cross-component-aliases: write-time
                    # conflict guard.  After pot/flow elimination collapses
                    # pin-potentials into a shared aliaser symbol, two
                    # component-internal eqs may both alias the same
                    # downstream variable to that aliaser but with
                    # contradictory expressions (e.g. one with `+aliaser`,
                    # another with `-aliaser`).  Without this guard the
                    # silent overwrite produces only a downstream
                    # IndexReduction count mismatch.  Compare the new
                    # `(aliasee, aliasee_expr)` against existing entries
                    # for the same aliaser; if the same aliasee carries a
                    # different expression under `sp.expand`, raise an
                    # AcausalModelError naming both component sources.
                    self._t036f_check_write_time_alias_conflict(
                        aliaser, aliasee, aliasee_expr, eq, aliasee_list
                    )
                    aliasee_list.append((aliasee, aliasee_expr))
                    self.aliaser_map[aliaser] = aliasee_list

                # perform the substitution
                self.eqs_subs(alias_sym, alias_sub_expr)
                self.syms_subs(alias_sym, alias_sub_expr)  # LUT and COND syms
                self.zcs_subs(alias_sym, alias_sub_expr)  # for ZC exprs

                # remove eqn from eqs.
                del self.eqs[eq_idx]
                # print(f"substitute {alias_sym} with {alias_sub_expr}")

                # After substitution, equations that contained alias_sym may now
                # be alias equations themselves or may have collapsed to BooleanTrue/
                # False. We preserve the existing workset (equations not yet examined)
                # and union it with any equations whose content just changed so they
                # get re-examined.  Replacing the workset entirely would discard
                # unprocessed equations that do NOT contain alias_sym — which was a
                # bug that caused alias elimination to stop after processing only the
                # last potential equation encountered.
                #
                # BooleanTrue/False equations (Eq(x,x) after substitution) cannot
                # safely be checked via eq.expr (which calls .lhs/.rhs), so we
                # include them directly in the changed set so they get deleted on
                # the next pop.
                changed_idxs = set()
                for _idx, _eq in self.eqs.items():
                    if isinstance(_eq.e, sp_bool_types):
                        changed_idxs.add(_idx)
                    elif alias_sym in _eq.expr.atoms():
                        changed_idxs.add(_idx)
                workset = list(set(workset) | changed_idxs)

        # T-036f-followup-cross-component-aliases-architecture:
        # post-elimination scan over aliaser_map for the sign-flip case
        # the write-time guard cannot catch (two components contributing
        # entries with opposite-sign expressions to the same canonical
        # aliaser).  Fires only on `param`/`inp` aliasers (the genuine
        # over-determined sign-flip pattern); naturally skips sensor-style
        # observation components.
        self._t036f_check_post_elim_sign_flip()

        # prune syms to only those found in the remaining equations.
        self.update_syms()
        if self.verbose:
            print("##### DiagramProcessing.alias_elimination #####", "\n")
            print("alias_map:")
            for ae, ar in self.alias_map.items():
                aes = self.syms_map_original[ae]
                print(f"\t{ae}:{aes.kind}->{ar}")
            print("syms_map_original:")
            for ss, sym in self.syms_map_original.items():
                print(f"\t{ss}:{sym}")
            print("syms_map:")
            for ss, sym in self.syms_map.items():
                print(f"\t{ss}:{sym}")
            self.pp_eqs()

    def initial_condition_validation(self):
        """
        For all alias sets, verify that the initial conditions for all symbols in
        an alias set are consistent and complete.

        Definitions: aliasees are replaced by aliasers

        When an aliaser replaces several
        aliasees, there may be one or more aliasees that have initial conditions
        specificed in the diagram. To ensure that these initial conditions are
        all respected and consistent, we will evaluate each aliasees' initial
        condition in its aliasee_expr, to get the initial condition from the
        view of the aliaser. Then, including any initial condition directly assigned
        to the aliaser, we now have a set of source initial conditions
        for the aliaser, and can chose an appropriate initial condition for the set,
        by checking it for consistency (all values are the same).
        Although the above may seem like it is relying on too many assumptions about
        aliasee<->aliaser relationship, this is not the case, because alias elimination
        only choses these pairs from equations which have been established to define
        these simple relationships between exactly 2 symbols.

        if initial condition of the aliaser and all aliasses are none, we just leave it as that.

        Note that there might be aliasers which are also aliasees. Ideally this would not be the case,
        but diagram_processing is not at the stage where it is resolving all the alias mapping
        such no aliasers are not also aliasees.
        As such, it is necessary to use a workset to ensure the initial condition propagation
        has converge to a fixed point.
        """
        if self.verbose:
            print("##### DiagramProcessing.initial_condition_validation #####", "\n")

        self._update_dpd()

        def ic_mismatch(ic1, ic2):
            return not np.allclose(ic1, ic2)

        # Propagate ICs to fixed point. We track (aliaser_id, ic_value) pairs seen
        # during the iteration; if the same pair appears twice the algorithm has
        # entered a cycle and we terminate (the conflict check earlier guarantees
        # any genuine inconsistency raises before we reach a cycle).
        seen_ic_states: dict = {}  # maps aliaser → set of ic values visited
        workset = set(self.aliaser_map.keys())
        while workset:
            aliaser = workset.pop()
            aliasee_pairs = self.aliaser_map[aliaser]
            # print(f"\n{aliaser=}++++++++++++++++++++++++++++++ {while_cnt=}")
            aliaser_ic_pre = aliaser.ic
            aliasees_ics = []
            aliasees_weak_ics = []
            for alias, alias_expr in aliasee_pairs:
                # print(f"{alias=}. {alias.ic=}. {alias.ic_fixed=} {alias_expr=}")
                alias_ic = alias.ic
                if alias_ic is not None:
                    alias_to_aliaser_ic = float(alias_expr.subs(alias.s, alias_ic))
                    if alias.ic_fixed:
                        aliasees_ics.append(alias_to_aliaser_ic)
                    else:
                        aliasees_weak_ics.append(alias_to_aliaser_ic)

            # print(f"{aliasees_ics=}")
            # print(f"{aliasees_weak_ics=}")
            if aliaser.ic is not None and not aliaser.ic_fixed and aliasees_ics:
                # this aliaser has a weak IC, but some aliasees have strong ICs, so reset
                # this aliaser IC to None.
                aliaser.ic = None
            for ic in aliasees_ics:
                aliaser_ic = aliaser.ic
                if aliaser_ic is None and ic is not None:
                    aliaser.ic = ic
                    aliaser.ic_fixed = True
                    if self.verbose:
                        print(f"for {aliaser} assign ic={ic}")
                elif ic is not None and ic_mismatch(ic, aliaser_ic):
                    message = (
                        f"Detected conflicting initial conditions."
                        f" Values are: {ic} and {aliaser_ic}."
                    )
                    raise AcausalModelError(
                        message=message, variables=[aliaser], dpd=self.dpd
                    )
            if aliaser.ic is None and aliasees_weak_ics:
                for ic in aliasees_weak_ics:
                    aliaser_ic = aliaser.ic
                    if aliaser_ic is None and ic is not None:
                        aliaser.ic = ic
                        if self.verbose:
                            print(f"for {aliaser} assign weak ic={ic}")
                    elif ic is not None and ic_mismatch(ic, aliaser_ic):
                        msg = f"Conflicting weak ICs for {aliaser}={aliaser_ic} and aliasee ic={ic}, ignoring the latter."
                        warnings.warn(msg, UserWarning)

            aliaser_ic = aliaser.ic
            if aliaser_ic_pre != aliaser_ic:
                # only re-add any other_aliasers which are aliasers of this aliaser.
                # good luck making sense of that senstence!
                other_aliasers = []
                for other_aliaser, aliasee_pairs in self.aliaser_map.items():
                    aliasees = [a for (a, e) in aliasee_pairs]
                    if aliaser in aliasees:
                        other_aliasers.append(other_aliaser)

                # Cycle detection: if we have seen this (aliaser, ic) combination before,
                # skip re-adding to workset to break the cycle.
                if aliaser not in seen_ic_states:
                    seen_ic_states[aliaser] = set()
                ic_key = (id(aliaser), round(float(aliaser.ic), 10) if aliaser.ic is not None else None)
                if ic_key in seen_ic_states[aliaser]:
                    continue  # already processed this state, break cycle
                seen_ic_states[aliaser].add(ic_key)

                workset.update(set(other_aliasers))

    def prune_derivative_relations(self):
        """
        iterate over equations and remove any derivative relations that do not define the system.
        """
        if self.verbose:
            print("======================== prune_derivative_relations")
        # get the syms in the system equations. do not get symbols from derivative relation equations.
        eqs_syms = self.get_some_syms(eqn_kind_filter=[EqnKind.der_relation])
        eqs_syms = list(eqs_syms.values())
        # print(f"eqs_syms={eqs_syms}")

        # get all their derivative family
        syms_relatives = set()
        for sym in eqs_syms:
            this_var = sym
            while this_var.int_sym is not None:
                syms_relatives.add(this_var.int_sym)
                this_var = this_var.int_sym
            this_var = sym
            while this_var.der_sym is not None:
                syms_relatives.add(this_var.der_sym)
                this_var = this_var.der_sym

        # print(f"syms_relatives={syms_relatives}")

        # get a list of eqn syms and their relatives. no duplicates, so use set1.union(set2).
        eqs_syms = list(set(eqs_syms) | syms_relatives)
        eqs_syms = set([s.s for s in eqs_syms])
        # print(f"eqs_syms={eqs_syms}")

        # iterate over the derivative relation equations, and identify all those which are
        # NOT needed to define a time derivative relationship between two sp.Function symbols
        # appearing in the 'system equations'.
        remove_eq_ids = []
        lhss = set()
        rhss = set()
        for eq_idx, eq in self.eqs.items():
            if eq.kind == EqnKind.der_relation:
                if self.verbose:
                    print(f"\n\t {eq_idx}: {eq}")
                # Remove trivial derivative relations of the form Eq(0, Derivative(0,t))
                # or any equation where both sides are structurally zero.
                lhs_zero = eq.e.lhs.is_zero or eq.e.lhs == sp.S.Zero
                rhs_zero = (
                    eq.e.rhs.is_zero
                    or eq.e.rhs == sp.S.Zero
                    or (
                        isinstance(eq.e.rhs, sp.Derivative)
                        and eq.e.rhs.expr.is_zero
                    )
                )
                if lhs_zero and rhs_zero:
                    remove_eq_ids.append(eq_idx)
                    continue

                lhs_fcns = eq.e.lhs.atoms(sp.Function)
                rhs_fcns = eq.e.rhs.atoms(sp.Function)
                if self.verbose:
                    print(f"\t\t {lhs_fcns=} {rhs_fcns=}")
                    print(f"\t\t {lhss=} {rhss=}")
                if lhs_fcns.issubset(lhss) and rhs_fcns.issubset(rhss):
                    # the intent is to remove derivative_relations which have been made
                    # identical to each other through alias eliminaton. it is important to check both sides
                    # since alias elimination may have put the same var in the LSH of 2 der_relations, but
                    # both are still required.
                    remove_eq_ids.append(eq_idx)
                    continue

                fncs = lhs_fcns | rhs_fcns
                # intersection = eqs_syms.intersection(fncs)
                intersection = fncs.issubset(eqs_syms)
                # print(f"\t intersection={intersection}")
                if not intersection:
                    # the intent is to remove der_relations for which neither of their symbols
                    # appear in the final 'system equations'.
                    # print("\t remove because no matching symbols")
                    remove_eq_ids.append(eq_idx)
                    continue

                # if we cant remove this derivative relation, keep track of its lhs,
                # that way, if we find a duplicate der_relation, we can remove the duplicate.
                lhss.update(lhs_fcns)
                rhss.update(rhs_fcns)

        if self.verbose:
            print(f"remove_eq_ids={remove_eq_ids}")

        for idx in remove_eq_ids:
            del self.eqs[idx]

    @staticmethod
    def _are_proportional_exprs(e1: sp.Expr, e2: sp.Expr) -> bool:
        """
        Return True if ``e1`` and ``e2`` represent the same constraint
        (i.e. ``e1 = k * e2`` for some nonzero constant *k*).

        ``sp.monic()`` was used here previously, but it is a univariate
        polynomial operation and raises an exception whenever the expression
        contains ``Derivative`` objects or more than one free symbol — both
        of which are routine in mechanical/rotational equations.

        The replacement uses ``as_coefficients_dict()``, which treats every
        non-numeric atom (including ``Derivative`` instances) as an opaque
        key, so the coefficient dictionary is well-defined for any expression
        that is linear in its atoms after ``sp.expand()``.
        """
        e1 = sp.expand(e1)
        e2 = sp.expand(e2)

        # Fast path: structurally identical or negated
        if (e1 - e2) == sp.S.Zero:
            return True
        if (e1 + e2) == sp.S.Zero:
            return True

        # General proportionality via coefficient dictionaries
        c1 = e1.as_coefficients_dict()
        c2 = e2.as_coefficients_dict()
        keys1 = frozenset(k for k, v in c1.items() if v != sp.S.Zero)
        keys2 = frozenset(k for k, v in c2.items() if v != sp.S.Zero)
        if keys1 != keys2:
            return False
        if not keys1:
            return True  # both structurally zero

        # All coefficient ratios must be equal and numeric
        ratio = None
        for key in keys1:
            coef1 = c1[key]
            coef2 = c2[key]
            if coef2 == sp.S.Zero:
                return False
            r = coef1 / coef2
            if not r.is_number:
                return False
            if ratio is None:
                ratio = r
            elif sp.nsimplify(r - ratio) != sp.S.Zero:
                return False
        return True

    def remove_duplicate_eqs(self):
        """
        After alias elimination some equations may become mathematically
        equivalent (identical up to a nonzero scalar multiple).  This method
        removes all but the first occurrence of each such equation.

        Equations are compared with ``_are_proportional_exprs``, which uses
        ``as_coefficients_dict()`` so that ``Derivative`` objects and
        multi-variable expressions are handled correctly.  The earlier
        ``sp.monic()``-based approach was limited to univariate polynomials
        and raised exceptions for mechanical/rotational models.
        """

        if self.verbose:
            print("##### DiagramProcessing.remove_duplicate_eqs #####", "\n")
            self.pp_eqs()
            print("\n")

        unique_exprs = {}
        duplicates = []

        for idx, eqn in self.eqs.items():
            expr = sp.expand(sp.nsimplify(eqn.expr))
            if self.verbose:
                print(f"{idx=} {expr=} ")
            is_duplicate = False
            for unique_idx, unique_expr in unique_exprs.items():
                if self.verbose:
                    print(f"\t{unique_idx=} {unique_expr=}")
                if self._are_proportional_exprs(expr, unique_expr):
                    duplicates.append(idx)
                    is_duplicate = True
                    break
            if not is_duplicate:
                unique_exprs[idx] = expr

        if duplicates:
            for duplicate in duplicates:
                self.eqs.pop(duplicate, None)

        if self.verbose:
            print(f"\n{duplicates=}\n")
            self.pp_eqs()
            print("\n")

    def get_params_ic_interface(self):
        """
        This collects the symbols that meet certain conditions. This is
        just for conveniece of getting these at some later stage.
        """
        # it seems correct to prioritize grouping by SymKind, before
        # the presence of ic.
        for sym in self.syms.values():
            if sym.kind == SymKind.param:
                self.params[sym] = sym.val
            elif sym.kind == SymKind.inp:
                self.inputs.append(sym)
            elif sym.ic is not None:
                self.sym_ic[sym.s] = sym.ic

        if self.verbose:
            print("##### DiagramProcessing.get_params_ic_interface #####", "\n")
            print(f"{self.sym_ic=}")
            print(f"{self.params=}")
            print(f"{self.inputs=}")

    def get_output_exprs(self):
        """
        this removes output equations from the equtaions set, and creates a map
        from output_symbol to output_expression.
        we remove output equations from the equations set because these equations
        should not be processed by index reduction.

        NOTE: if index_reduction changes any of the symbols used by any of the
        output_expressions, then these subsitutions will have to be done after
        index_reduction has completed.
        e.g. expr.subs(index_reduction.dummy_vars)
        """
        idx_to_remove = []
        for idx, eq in self.eqs.items():
            if eq.kind == EqnKind.outp:
                idx_to_remove.append(idx)
                eq_symbols = eq.e.atoms(sp.Symbol)
                eq_symbols.discard(self.eqn_env.t)
                eq_fcns = eq.e.atoms(sp.Function)
                all_eq_syms = list(eq_symbols | eq_fcns)
                if self.verbose:
                    print(f"{eq_symbols=}")
                    print(f"{eq_fcns=}")
                    print(f"{all_eq_syms=}")
                for s in all_eq_syms:
                    outp_sym = self.syms_map[s]
                    if self.verbose:
                        print(f"{outp_sym=} {outp_sym.kind=}")
                    if outp_sym.kind == SymKind.outp:
                        outp_expr = self.sanitize_solve(outp_sym.s, eq)
                        self.outp_exprs[outp_sym] = outp_expr
                        if self.verbose:
                            print(f"{outp_sym=} {hex(id(outp_sym))}")

        for idx in idx_to_remove:
            del self.eqs[idx]
        self.update_syms()

    def index_reduction_inputs_f(self):
        """
        collect the various lists, dicts, sets of info required by index reduction.
        nothing new computed really, just repackaging/organizing existing data.

        t = EqnEnv.t
        x = list of all differetiated variables
        x_dot = list of RHS of each der_relation
        y = list of all algebraic variables
        X = list(y,x,x_dot)
        exprs = list[eq.expr for eq in self.eqs.values()]
        vars_in_exprs = dict{expr:[all_members_of_X_in_eq]}
        exprs_idx = dict{expr:idx for idx,expr in self.eqs.values()}
        knowns = dict{sym.s:val} when sym.kind in [in,out,param,lut]
        knowns_set = set(self.knowns.keys())
        ics = dict{sym.s:sym.ic for sym in X when sym.ic is not None}
        """
        self._update_dpd()
        if self.verbose:
            print("##### DiagramProcessing.index_reduction_inputs_f #####", "\n")
        # collect x and x_dot
        x_set = set()
        x_dot_set = set()

        # for collecting the weak ICs. These are declared here because
        # they are added to in several places.
        DEFAULT_IC = 0.0
        ics_weak2 = {}
        x_to_x_dot_ic = {}

        for eq_idx, eq in self.eqs.items():
            if eq.kind == EqnKind.der_relation:
                if self.verbose:
                    print(f"\n{eq_idx=} {eq=}")
                fcn_set = eq.e.rhs.atoms(sp.Function)
                if not fcn_set:
                    raise AcausalCompilerError(
                        message="[index_reduction_inputs_f] equation RHS has no functions.",
                        dpd=self.dpd,
                    )
                # self.x_dot.append(sp.simplify(eq.e.rhs))
                # the line above was the original idea, but this occassionally results
                # one of: -Der(f(t),t) or Der(-f(t),t), and those '-' seem to
                # break index reduction because they make the Der() symbol more
                # like an expression, which is not what is needed.
                # so we landed on the operation below which seems to work for now.
                # T-002a: `.atoms(...).pop()` picks a hash-order element,
                # causing x_el / x_dot_el to vary run-to-run and propagating
                # non-determinism into the state vector layout.  Sort by
                # string representation for a stable choice.
                x_el = sorted(eq.e.rhs.atoms(sp.Function), key=str)[0]
                if self.verbose:
                    print(f"\t{x_el=}")
                x_set.add(x_el)

                # some machinery for assisting with initial condition for symbols of the form: Der(f(t))
                # !!!!! the old way.
                x_dot_el = x_el.diff(self.eqn_env.t)
                x_dot_set.add(x_dot_el)
                lhs_sym = sorted(eq.e.lhs.atoms(sp.Function), key=str)[0]
                x_to_x_dot_ic[lhs_sym] = x_dot_el

                # !!!!! the supposed 'right' way
                # all Der(f(t)) need to be weak because of where these symbols
                # appear in the system equations, e.g. g(t) = Der(f(t)), they will always
                # appear in those equations. because we get the IC from the IC of g(t),
                # it is not apropriate to have both g(t) IC and Der(f(t)) IC both be strong
                # since they are equated to each other. One needs to be left weak. Since g(t)
                # is typically where users apply ICs, we choose g(t) to be allowed to be
                # strong or weak, and force Der(f(t)) to always be weak.
                # Note, due to alias elimination, the der_relation might be like: -g(t),Der(f(t)),
                # this means we cannot blindly apply the IC for g(t) to Der(f(t)), but rather we
                # have to substitute the IC for g(t) into the LHS, evaluate to a float. This is
                # the same procedure applied in initial_condition_validation() above.
                lhs_fcn = sorted(eq.e.lhs.atoms(sp.Function), key=str)[0]
                # if lhs_sym not in x_to_x_dot_ic.keys():
                #     x_to_x_dot_ic[lhs_sym] = []
                # x_to_x_dot_ic[lhs_sym] = x_dot_el
                lhs_sym = self.syms_map.get(lhs_fcn, None)
                lhs_sym_ic = lhs_sym.ic if lhs_sym.ic is not None else DEFAULT_IC
                x_dot_el_ic = float(eq.e.lhs.subs(lhs_sym.s, lhs_sym_ic))
                ics_weak2[x_dot_el] = x_dot_el_ic
                if self.verbose:
                    print(f"add weak IC for {x_dot_el=}")

        if self.verbose:
            print(f"{ics_weak2=}")
            print(f"{x_dot_set=}")
            print(f"{x_to_x_dot_ic=}")

        knowns_syms = (
            [s for s in self.params]
            + [s for s in self.inputs]
            + [s for s in self.syms.values() if s.kind in [SymKind.lut, SymKind.cond]]
        )
        # Build a substitution map from parameter symbols → numeric values so
        # that cond (Piecewise) symbols can be evaluated at t=0 for accurate
        # initial conditions.
        param_subs_map = {}
        for ps in self.params:
            if isinstance(ps.val, Parameter):
                pval = ps.val.get()
            else:
                pval = ps.val
            if pval is not None:
                param_subs_map[ps.s] = pval

        knowns = {}
        for s in knowns_syms:
            if s.kind == SymKind.param:
                if isinstance(s.val, Parameter):
                    val = s.val.get()
                else:
                    val = s.val
            elif s.kind == SymKind.cond:
                # Evaluate the Piecewise expression at t=0 with known parameter
                # values to obtain a numerically correct initial condition.
                # Fall back to 1.0 if the expression cannot be evaluated to a
                # float (e.g. branch conditions that remain symbolic).
                try:
                    val_expr = s.s.subs(param_subs_map).subs(self.eqn_env.t, sp.Integer(0))
                    val = float(val_expr)
                except Exception:
                    val = 1.0
            elif s.kind == SymKind.lut:
                # LUT symbols can appear inside Abs() ops; taking Der() results
                # in div by 0, so keep a safe structural placeholder.
                val = 1.0
            else:
                # presently this path is only for inputs.
                val = 0.0
            knowns[s.s] = val

        knowns_set = set(knowns.keys())
        not_y_list = list(x_set | x_dot_set | knowns_set)

        # collect y by getting all other syms
        y = [s.s for s in self.syms.values() if s.s not in not_y_list]

        X = x_set | x_dot_set | set(y)

        # create a list of all syms that cant be in y
        x = list(x_set)
        x_dot = list(x_dot_set)

        exprs = []
        exprs_idx = {}
        vars_in_exprs = {}
        for eq_idx, eq in self.eqs.items():
            if eq.kind == EqnKind.outp:
                # output equations are only for computing outputs and should not be present here.
                raise AcausalCompilerError(
                    message="[index_reduction_inputs_f] equations with EqnKind.outp should have been removed before calling index_reduction_inputs_f().",
                    dpd=self.dpd,
                )
            # list of expressions, from the equations
            # Normalize negated derivatives: Derivative(-f(t),t) → -Derivative(f(t),t)
            # sp.expand distributes negation through Derivative, which sp.simplify also
            # handles but at ~100x the cost.
            expr = sp.expand(eq.expr)
            exprs.append(expr)
            exprs_idx[expr] = eq_idx

            # list of which variables appear in which expression

            ders_fcns = set()
            if eq.kind == EqnKind.der_relation:
                ders = expr.atoms(sp.Derivative)
                # find the sp.Functions in the sp.Derivative symbols, call these der_fncs
                # sp.Derivative symbols by design only ever appear in der_relattion equations,
                # so leave the set empty for all other equation kinds.
                for der in ders:
                    der_fcns = der.atoms(sp.Function)
                    ders_fcns.update(der_fcns)
            else:
                ders = set()

            # now find all other symbols we expect might be present
            symbols = expr.atoms(sp.Symbol)
            symbols.discard(self.eqn_env.t)
            fcns = expr.atoms(sp.Function)

            syms_set = ders | symbols | fcns
            syms_set = syms_set - ders_fcns  # remove the der_fcns if any
            vars_in_exprs[expr] = X & syms_set  # set intersection

        X = list(X)
        if self.verbose:
            print(f"{X=}")
        # collect the strong and weak ICs
        ics = {}
        ics_weak = {}
        if self.verbose:
            print("collect strong and weak ICs:")
        for s in X:
            sym = self.syms_map.get(s, None)
            if self.verbose:
                print(f"\nfind ICs for {sym=} {s=}")
            if sym is not None:
                if self.verbose:
                    print(f"\t{sym.ic=} {sym.ic_fixed=}")
                if sym.ic is not None:
                    if sym.ic_fixed:
                        ics[s] = sym.ic
                    else:
                        ics_weak[s] = sym.ic
                else:
                    ics_weak[s] = DEFAULT_IC  # this just applies the default.
                    sym.ic = DEFAULT_IC
            # else:
            #     # case where variable did not come from a component or
            #     # from node equation addition
            #     if s not in x_to_x_dot_ic.values():
            #         raise AcausalCompilerError(
            #             message=f"Unexpected variable {s} in X.", dpd=self.dpd
            #         )
            # duplicate the IC assignment for the 'Derivative(f(t)) symbols.
            if s in x_to_x_dot_ic.keys():
                x_dot_s = x_to_x_dot_ic[s]
                # print(f"adding IC={sym.ic} for x_dot var {x_dot_s} because {s=}")
                # all Der(f(t)) need to be weak because of where these symbols
                # appear in the system equations, e.g. g(t) = Der(f(t)), they will always
                # appear in those equations. because we get the IC from the IC of g(t),
                # it is not apropriate to have both g(t) IC and Der(f(t)) IC both be strong
                # since they are equated to each other. One needs to be left weak. Since g(t)
                # is typically where users apply ICs, we choose g(t) to be allowed to be
                # strong or weak, and force Der(f(t)) to always be weak.
                if self.verbose:
                    print(f"\tadd weak IC for {x_dot_s=}")
                ics_weak[x_dot_s] = sym.ic

        # Merge ics_weak2 into ics_weak. The primary collection (above) is correct
        # for most cases. ics_weak2 captures der_relation symbols whose LHS appears
        # in multiple derivative relations — a case the primary loop misses because
        # it iterates over X (integration variables) rather than der_relations.
        # Merging with update() adds missing entries and corrects any stale values
        # from the primary loop without discarding correct primary entries.
        ics_weak.update(ics_weak2)

        if self.verbose:
            print(f"{x_dot=}")
            print(f"{x=}")
            print(f"{y=}")
            print(f"{X=}")
            print("\nexprs:")
            for i, expr in enumerate(exprs):
                print(f"\t{i}: {expr}")

            print("\nvars_in_exprs:")
            for expr, vars in vars_in_exprs.items():
                print(f"\t{expr=}")
                for var in vars:
                    print(f"\t\t{var=}")
            print(f"{knowns=}")
            print(f"{ics=}")
            print(f"{ics_weak=}")
            print(f"{ics_weak2=}")
            print(f"num exprs: {len(exprs)=} num variables: {len(x)+len(y)=} ")

        total_ics = len(ics) + len(ics_weak)
        if total_ics != len(X):
            raise AcausalCompilerError(
                message=f"Mismatch between number of expected ics {len(X)} and total ics collected {total_ics}.",
                dpd=self.dpd,
            )

        # store the shared interface as a NamedTuple
        self.index_reduction_inputs = IndexReductionInputs(
            self.eqn_env.t,
            x,
            x_dot,
            y,
            X,
            exprs,
            vars_in_exprs,
            exprs_idx,
            knowns,
            knowns_set,
            ics,
            ics_weak,
        )

    def prep_dpd(self):
        self.dpd = DiagramProcessingData(
            self.diagram,
            self.syms,
            self.syms_map_original,
            self.nodes,
            self.node_domains,
            self.pot_alias_map,
            self.alias_map,
            self.aliaser_map,
            self.params,
        )

    def diagram_processing(self):
        self.identify_nodes()
        self.identify_fluid_networks()
        self.finalize_diagram()
        if self.verbose:
            self.pp()
        self.add_node_flow_eqs()
        self.add_stream_balance_eqs()
        self.add_node_potential_eqs()
        self.add_derivative_relations()
        self.finalize_equations()  # just book keeping
        if self.verbose:
            self.pp()
        self.collect_zcs()  # do this *before* alias elimination
        self.alias_elimination()
        self.initial_condition_validation()
        self.prune_derivative_relations()
        self.remove_duplicate_eqs()
        self.get_params_ic_interface()
        self.get_output_exprs()
        if self.verbose:
            self.pp()
        self.index_reduction_inputs_f()
        if self.verbose:
            self.pp()
        self.prep_dpd()
        self.diagram_processing_done = True

    # execute compilation
    def __call__(self):
        self.diagram_processing()
