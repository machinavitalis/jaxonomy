# Copyright (C) 2025 Collimator, Inc
# SPDX-License-Identifier: MIT

from collimator.experimental.acausal.index_reduction import IndexReduction
from collimator.experimental.acausal.index_reduction.equation_parsing import (
    parse_string_inputs,
)

str_eqs = [
    "dx(t)/dt - w(t)",
    "dy(t)/dt - z(t)",
    "dw(t)/dt - T(t) * x(t)",
    "dz(t)/dt - T(t) * y(t) + g",
    "x(t)**2 + y(t)**2 - L**2",
]

str_knowns = {
    "g": 9.8,
    "L": 2.0,
}

str_ics = {
    "x(t)": 1.73,
    "dy(t)/dt": 1.0,
}

str_ics_weak = None

# ic_weak_val = -0.5
# str_ics_weak = {
#     "y(t)": ic_weak_val,
#     "z(t)": ic_weak_val,
#     "w(t)": ic_weak_val,
#     "T(t)": ic_weak_val,
#     "dx(t)/dt": ic_weak_val,
#     "dz(t)/dt": ic_weak_val,
#     "dw(t)/dt": ic_weak_val,
# }

t, eqs, knowns, ics, ics_weak = parse_string_inputs(
    str_eqs, str_knowns, str_ics, str_ics_weak
)

ir = IndexReduction(t, eqs, knowns, ics, ics_weak, verbose=True)

ir()
