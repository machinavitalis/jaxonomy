# Copyright (C) 2025 Collimator, Inc
# SPDX-License-Identifier: MIT

from typing import TYPE_CHECKING
from collimator.lazy_loader import LazyLoader

from index_reduction import IndexReduction

if TYPE_CHECKING:
    import sympy as sp
else:
    sp = LazyLoader("sp", globals(), "sympy")

# Define the symbol for time
t = sp.symbols("t")

# Define parameters
m = sp.symbols("m")
k = sp.symbols("k")

# Define functions of time
x = sp.Function("x")(t)
v = sp.Function("v")(t)
a = sp.Function("a")(t)
f = sp.Function("f")(t)
d = sp.Function("d")(t)
dd = sp.Function("dd")(t)

# knowns
knowns = [m, k]

# Define the derivatives of these functions with respect to time
xdot = x.diff(t)
vdot = v.diff(t)
ddot = d.diff(t)

# Define equations
eq0 = xdot - v
eq1 = vdot - a
eq2 = f - x * k
eq3 = m * a + f
eq4 = ddot - dd


# Equations list
eqs = [eq0, eq1, eq2, eq3, eq4]

ir = IndexReduction(t, eqs, knowns)
ir()
