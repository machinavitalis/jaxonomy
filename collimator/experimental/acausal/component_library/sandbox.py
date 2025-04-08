# Copyright (C) 2025 Collimator, Inc
# SPDX-License-Identifier: MIT

from typing import TYPE_CHECKING
from collimator.lazy_loader import LazyLoader
from .base import SymKind
from .rotational import RotationalOnePort

if TYPE_CHECKING:
    import sympy as sp
else:
    sp = LazyLoader("sp", globals(), "sympy")

"""
components here are just sandbox development. they will either go away,
or be moved to a proper domain library.
"""


class TorqueSwitch(RotationalOnePort):
    """
    Dev component to develop conditionals in acausal components.
    """

    def __init__(
        self,
        ev,
        name=None,
        timeThr=2.123456789,
        onTrq=-10.0,
        offTrq=10.0,
    ):
        self.name = self.__class__.__name__ if name is None else name
        # name the port 'flange_a' so the component is interchangebale with a
        # normal torque source with enable_flange_b=False. this makes for easier
        # debugging.
        super().__init__(ev, self.name, p="flange_a")

        timeThr = self.declare_symbol(
            ev, "timeThr", self.name, kind=SymKind.param, val=timeThr
        )
        onTrq = self.declare_symbol(
            ev, "onTrq", self.name, kind=SymKind.param, val=onTrq
        )
        offTrq = self.declare_symbol(
            ev, "offTrq", self.name, kind=SymKind.param, val=offTrq
        )
        cond_sym = self.declare_conditional(
            ev,
            ev.t <= timeThr.s,
            onTrq.s,
            offTrq.s,
            cond_name="trqSwitch",
            non_bool_zc_expr=ev.t - timeThr.s,
        )
        self.add_eqs([sp.Eq(self.t.s, cond_sym.s)])
