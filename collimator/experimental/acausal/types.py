# Copyright (C) 2025 Collimator, Inc
# SPDX-License-Identifier: MIT

from typing import NamedTuple
from .acausal_diagram import AcausalDiagram
from .component_library.base import Sym


# data structure for capturing the useful data object output from DiagramProcessing
# defined here since it is used in DiagramProcessing and error.py, and hence
# deffining in DiagramProcessing would create circulr imports
class DiagramProcessingData(NamedTuple):
    ad: AcausalDiagram

    # see DiagramProcessing for description
    syms: dict
    syms_map_original: dict
    nodes: dict
    node_domains: dict
    pot_alias_map: dict
    alias_map: dict
    aliaser_map: dict
    params: dict


class IndexReductionInputs(NamedTuple):
    t: Sym  # Symbol for time
    x: list  # list of all differential variables
    x_dot: list  # [sp.Der(x_) for x_ in x]
    y: list  # list of all algebraic variables
    X: list  # x + x_dot + y
    exprs: list  # the equations of the system in the '0 = expr' form
    vars_in_exprs: dict  # dict{expr:set(vars)}, the variables in each expr
    exprs_idx: dict  # map from expr to the DiagramProcessing equation index
    knowns: dict  # the symbols that are either params or inputs
    knowns_set: set  # set(knowns)
    ics: dict  # dict{var:value} for all 'strong' initial conditions
    ics_weak: dict  # dict{var:value} for all 'weak' initial conditions
