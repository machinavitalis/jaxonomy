# Copyright (C) 2025 Collimator, Inc
# SPDX-License-Identifier: MIT

"""Base classes for nonlinear MPCs with Ipopt"""

from .nmpc_ipopt_base import NonlinearMPCIpopt
from .nlp_ipopt_base import NMPCProblemStructure


__all__ = [
    "NonlinearMPCIpopt",
    "NMPCProblemStructure",
]
