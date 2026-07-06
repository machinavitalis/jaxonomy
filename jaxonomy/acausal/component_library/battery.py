# SPDX-License-Identifier: MIT
"""Battery-domain standard-library acausal components.

T-121 phase 1 — Equivalent Circuit Model (ECM) battery cell.

This module provides the first standard-library battery cell for Jaxonomy's
acausal modelling framework.  The ECM cell follows the canonical Thevenin
topology used in industry (UPS, EV, drone, robotics, BMS test benches):

    Terminal V = OCV(SOC) + R0 * I + V_RC
    d V_RC / dt  = -V_RC / (R1 * C1) + I / C1
    d SOC  / dt  = -I / (3600 * capacity_Ah)         # Coulomb counting

with optional thermal port that exposes I^2 * (R0 + R1) joule heating.

Sign convention (Modelica passive convention, identical to the existing
``electrical.Battery`` block):

    ``Ip`` is the current flowing **into** the positive pin from the rest of
    the circuit.  When the battery *discharges* into an external load, the
    external load draws conventional current out of the positive pin, so
    ``Ip < 0`` and ``dSOC/dt = Ip / (3600 * capacity_Ah) < 0`` — SOC drops.
    When the battery is *charged* by an external source, ``Ip > 0`` and
    ``dSOC/dt > 0`` — SOC rises.

This is the same convention as the legacy ``electrical.Battery`` class.
The new ``BatteryCellECM`` block exposes a tidier, name-aligned API that
matches the conventional equivalent-circuit ``BatteryCell`` block used by
acausal modelling tools:

    BatteryCellECM(R0, R1, C1, capacity_Ah, ocv_soc, ocv_volts, ...)

A first-class table-based cell is available as ``BatteryCellTabular``
(T-121-followup-table-cell): the degenerate ``R1 = 0``, ``C1 = 0`` ECM with
no RC transient state — useful when transient dynamics aren't important.

Parameter-fitting / module / pack are tracked in later phases of T-121.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from jaxonomy.lazy_loader import LazyLoader
from .base import SymKind, EqnKind
from .electrical import ElecTwoPin
from jaxonomy.backend import numpy_api as npa

if TYPE_CHECKING:
    import sympy as sp
else:
    sp = LazyLoader("sp", globals(), "sympy")


__all__ = [
    "BatteryCellECM",
    "BatteryCellTabular",
    "BatteryModule",
    "BatteryPack",
    "battery_module",
    "battery_pack",
]


class BatteryCellECM(ElecTwoPin):
    """Equivalent-Circuit-Model (Thevenin, single-RC) battery cell.

    Governing equations (passive sign convention; see module docstring):

        1. V_terminal = OCV(SOC) + R0 * Ip + V_RC                 (Kirchhoff)
        2. d(V_RC)/dt = Ip / C1 - V_RC / (R1 * C1)                (RC pair)
        3. d(SOC)/dt  = Ip / (3600 * capacity_Ah)                 (Coulomb counter)

    Optional thermal port exposes joule heating ``I^2 * (R0 + R1)`` as a
    heat-flow source (matches the ``Resistor(enable_heat_port=True)`` and
    ``IdealMotor(enable_heat_port=True)`` patterns).

    Args:
        R0:
            Series ohmic resistance in Ohms.  Differentiable parameter.
        R1:
            RC-pair resistance in Ohms (transient/polarisation resistance).
        C1:
            RC-pair capacitance in Farads (transient/polarisation capacitance).
        capacity_Ah:
            Cell capacity in Amp-Hours.
        ocv_soc:
            SOC break-points (monotonically increasing array in [0, 1]) for
            the OCV-vs-SOC lookup table.
        ocv_volts:
            Open-circuit-voltage values (Volts) at each ``ocv_soc`` breakpoint.
        initial_soc:
            Initial state of charge (in [0, 1]).  Default 0.5.
        initial_soc_fixed:
            Whether ``initial_soc`` is a fixed initial condition.
        initial_v_rc:
            Initial transient voltage across the RC pair (Volts).
        initial_v_rc_fixed:
            Whether ``initial_v_rc`` is a fixed initial condition.
        enable_heat_port:
            When ``True``, declare a thermal port ``"heat"`` whose heat-flow
            equals ``I^2 * (R0 + R1)``.
        enable_soc_port:
            When ``True``, declare a causal output port ``"soc"`` carrying SOC.
        enable_v_rc_port:
            When ``True``, declare a causal output port ``"v_rc"`` carrying
            the transient voltage.
        enable_ocv_port:
            When ``True``, declare a causal output port ``"ocv"`` carrying
            the open-circuit voltage.
    """

    def __init__(
        self,
        ev,
        name: str | None = None,
        R0: float = 0.01,
        R1: float = 0.01,
        C1: float = 1000.0,
        capacity_Ah: float = 1.0,
        ocv_soc=(0.0, 1.0),
        ocv_volts=(3.0, 4.2),
        initial_soc: float = 0.5,
        initial_soc_fixed: bool = False,
        initial_v_rc: float = 0.0,
        initial_v_rc_fixed: bool = False,
        enable_heat_port: bool = False,
        enable_soc_port: bool = False,
        enable_v_rc_port: bool = False,
        enable_ocv_port: bool = False,
    ):
        self.name = self.__class__.__name__ if name is None else name

        # Initial terminal voltage = OCV(initial_soc).  Used for component
        # initial-condition guess.
        ocv_soc_arr = npa.array(ocv_soc)
        ocv_volts_arr = npa.array(ocv_volts)
        V_ic = npa.interp(initial_soc, ocv_soc_arr, ocv_volts_arr)

        super().__init__(ev, self.name, V_ic=V_ic, I_ic=0.0)

        # ------------------------------------------------------------------
        # State variables: SOC and V_RC (transient voltage across RC pair).
        # ------------------------------------------------------------------
        SOC = self.declare_symbol(
            ev,
            "SOC",
            self.name,
            kind=SymKind.var,
            ic=initial_soc,
            ic_fixed=initial_soc_fixed,
        )
        dSOC = self.declare_symbol(
            ev,
            "dSOC",
            self.name,
            kind=SymKind.var,
            int_sym=SOC,
            ic=0.0,
        )
        SOC.der_sym = dSOC

        V_RC = self.declare_symbol(
            ev,
            "V_RC",
            self.name,
            kind=SymKind.var,
            ic=initial_v_rc,
            ic_fixed=initial_v_rc_fixed,
        )
        dV_RC = self.declare_symbol(
            ev,
            "dV_RC",
            self.name,
            kind=SymKind.var,
            int_sym=V_RC,
            ic=0.0,
        )
        V_RC.der_sym = dV_RC

        # ------------------------------------------------------------------
        # Differentiable parameters.
        # ------------------------------------------------------------------
        cap = self.declare_symbol(
            ev,
            "capacity_Ah",
            self.name,
            kind=SymKind.param,
            val=capacity_Ah,
            validator=lambda x: x > 0.0,
            invalid_msg=(
                f"Component {self.__class__.__name__} {self.name} "
                "must have capacity_Ah>0"
            ),
        )
        R0_sym = self.declare_symbol(
            ev,
            "R0",
            self.name,
            kind=SymKind.param,
            val=R0,
            validator=lambda x: x > 0.0,
            invalid_msg=(
                f"Component {self.__class__.__name__} {self.name} must have R0>0"
            ),
        )
        R1_sym = self.declare_symbol(
            ev,
            "R1",
            self.name,
            kind=SymKind.param,
            val=R1,
            validator=lambda x: x > 0.0,
            invalid_msg=(
                f"Component {self.__class__.__name__} {self.name} must have R1>0"
            ),
        )
        C1_sym = self.declare_symbol(
            ev,
            "C1",
            self.name,
            kind=SymKind.param,
            val=C1,
            validator=lambda x: x > 0.0,
            invalid_msg=(
                f"Component {self.__class__.__name__} {self.name} must have C1>0"
            ),
        )

        # OCV-vs-SOC lookup table (1D interp).  Differentiable through the
        # tabulated voltage values via ``jax.numpy.interp``.
        OCV_lut = self.declare_1D_lookup_table(
            ev,
            SOC.s,
            "ocv_soc",
            ocv_soc_arr,
            "ocv_volts",
            ocv_volts_arr,
            "OCV_lut",
        )

        # ------------------------------------------------------------------
        # Component equations.
        # ------------------------------------------------------------------
        # Eqn 3: Coulomb counter.  Passive sign: Ip > 0 ==> charging ==> SOC up.
        # Eqn 1: Terminal voltage = OCV + R0*I + V_RC.
        # Eqn 2: RC dynamics.  d(V_RC)/dt = Ip/C1 - V_RC/(R1*C1).
        self.add_eqs(
            [
                sp.Eq(dSOC.s, self.Ip.s / (cap.s * 3600)),
                sp.Eq(self.V.s, OCV_lut.s + R0_sym.s * self.Ip.s + V_RC.s),
                sp.Eq(dV_RC.s, self.Ip.s / C1_sym.s - V_RC.s / (R1_sym.s * C1_sym.s)),
            ]
        )

        # ------------------------------------------------------------------
        # Optional thermal port: joule heating I^2 * (R0 + R1).
        # ------------------------------------------------------------------
        if enable_heat_port:
            port_name = "heat"
            T, Q = self.declare_thermal_port(ev, port_name)
            # Flow vars are negative for flow leaving the component, hence the
            # minus sign on Q (matches Resistor / IdealMotor heat ports).
            self.add_eqs(
                [
                    sp.Eq(
                        -Q.s,
                        self.Ip.s * self.Ip.s * (R0_sym.s + R1_sym.s),
                    )
                ]
            )
            self.port_idx_to_name[2] = port_name

        # ------------------------------------------------------------------
        # Optional causal output ports for control / debugging.
        # ------------------------------------------------------------------
        if enable_soc_port:
            soc_out = self.declare_symbol(
                ev, "soc", self.name, kind=SymKind.outp
            )
            self.declare_equation(sp.Eq(soc_out.s, SOC.s), kind=EqnKind.outp)
        if enable_v_rc_port:
            v_rc_out = self.declare_symbol(
                ev, "v_rc", self.name, kind=SymKind.outp
            )
            self.declare_equation(sp.Eq(V_RC.s, v_rc_out.s), kind=EqnKind.outp)
        if enable_ocv_port:
            ocv_out = self.declare_symbol(
                ev, "ocv", self.name, kind=SymKind.outp
            )
            self.declare_equation(
                sp.Eq(ocv_out.s, OCV_lut.s), kind=EqnKind.outp
            )


# ---------------------------------------------------------------------------
# T-121-followup-module-pack — Battery module + pack builders.
#
# Real battery packs are made of N cells in series (for voltage) × M cells in
# parallel (for current).  A "module" is one such N-cell series arrangement;
# a "pack" is M such modules in parallel.
#
# These are container builder functions, not new acausal components.  They use
# the existing ``AcausalDiagram.connect`` API to wire up ``BatteryCellECM`` /
# ``BatteryCellTabular`` cells (any ``ElecTwoPin`` cell, really) into the
# series/parallel topology.  The result is an acausal subsystem whose
# ``pos`` / ``neg`` electrical terminals are the module/pack terminals --
# callers can ``ad.connect`` further from them just like any other
# ``ElecTwoPin``.
#
# Cell factories take ``(ev, name)`` and return an ``ElecTwoPin`` cell.  This
# is the natural way to express "a fresh cell per slot" while preserving each
# slot's unique name (required for the acausal compiler).  Variability between
# cells is supported trivially by closing over a per-index lookup in the
# factory.
# ---------------------------------------------------------------------------


class BatteryModule:
    """N cells wired in series.  Exposes ``pos``/``neg`` electrical pins.

    The module's positive pin is the positive pin of cell 0; the module's
    negative pin is the negative pin of cell N-1.  Internal connections are
    ``cell[i].neg -> cell[i+1].pos`` for ``i in 0..N-2``.

    Instances are produced by :func:`battery_module`; this class is the
    handle the caller uses to wire the module's terminals to the rest of the
    diagram, e.g.::

        ad.connect(module.pos_cmp, module.pos_port, sensI, "p")
        ad.connect(module.neg_cmp, module.neg_port, gnd, "p")

    Attributes
    ----------
    cells:
        list of cell components in series order (cells[0] is the most
        positive cell, cells[-1] is the most negative).
    pos_cmp, pos_port:
        component + port name for the module's positive terminal.
    neg_cmp, neg_port:
        component + port name for the module's negative terminal.
    name:
        module name (also used as prefix for default-named cells).
    """

    __slots__ = ("cells", "pos_cmp", "pos_port", "neg_cmp", "neg_port", "name")

    def __init__(self, cells, name):
        if len(cells) == 0:
            raise ValueError("BatteryModule requires at least one cell")
        self.cells = list(cells)
        self.name = name
        self.pos_cmp = self.cells[0]
        self.pos_port = "p"
        self.neg_cmp = self.cells[-1]
        self.neg_port = "n"

    def connect_pos(self, ad, other_cmp, other_port):
        """Wire the module's positive pin to ``(other_cmp, other_port)``."""
        ad.connect(self.pos_cmp, self.pos_port, other_cmp, other_port)

    def connect_neg(self, ad, other_cmp, other_port):
        """Wire the module's negative pin to ``(other_cmp, other_port)``."""
        ad.connect(self.neg_cmp, self.neg_port, other_cmp, other_port)


class BatteryPack:
    """M modules wired in parallel.  Exposes ``pos``/``neg`` electrical pins.

    All modules share a common positive bus and a common negative bus.  The
    pack's positive pin is the positive pin of module 0; the pack's negative
    pin is the negative pin of module 0.  Modules 1..M-1 are wired in
    parallel by ``module[i].pos -> module[0].pos`` and
    ``module[i].neg -> module[0].neg``.

    Instances are produced by :func:`battery_pack`.

    Attributes
    ----------
    modules:
        list of :class:`BatteryModule` instances comprising the pack.
    pos_cmp, pos_port:
        component + port name for the pack's positive terminal.
    neg_cmp, neg_port:
        component + port name for the pack's negative terminal.
    name:
        pack name (also used as prefix for default-named modules/cells).
    """

    __slots__ = (
        "modules",
        "pos_cmp",
        "pos_port",
        "neg_cmp",
        "neg_port",
        "name",
    )

    def __init__(self, modules, name):
        if len(modules) == 0:
            raise ValueError("BatteryPack requires at least one module")
        self.modules = list(modules)
        self.name = name
        # Pack terminals are module 0's terminals (all modules share buses
        # via parallel-wiring inside ``battery_pack``).
        m0 = self.modules[0]
        self.pos_cmp = m0.pos_cmp
        self.pos_port = m0.pos_port
        self.neg_cmp = m0.neg_cmp
        self.neg_port = m0.neg_port

    def connect_pos(self, ad, other_cmp, other_port):
        """Wire the pack's positive pin to ``(other_cmp, other_port)``."""
        ad.connect(self.pos_cmp, self.pos_port, other_cmp, other_port)

    def connect_neg(self, ad, other_cmp, other_port):
        """Wire the pack's negative pin to ``(other_cmp, other_port)``."""
        ad.connect(self.neg_cmp, self.neg_port, other_cmp, other_port)

    @property
    def cells(self):
        """Flat list of all cells (in (module, series) order)."""
        out = []
        for m in self.modules:
            out.extend(m.cells)
        return out


def battery_module(ev, ad, n_cells, cell_factory, name="module"):
    """Build an ``n_cells``-in-series battery module on an acausal diagram.

    Parameters
    ----------
    ev:
        the :class:`EqnEnv` used to declare cell symbols.
    ad:
        the :class:`AcausalDiagram` to wire cells into.
    n_cells:
        number of cells in series.  Must be >= 1.
    cell_factory:
        callable ``(ev, name) -> ElecTwoPin``.  Called once per cell with a
        unique ``name`` of the form ``f"{module_name}_cell{i}"``.  Cells
        may be heterogeneous (per-index parameter variation) -- the factory
        has full control.  Typical use::

            lambda ev, n: BatteryCellECM(ev, name=n, R0=0.02, ...)

    name:
        module name; used as a prefix for cell names.

    Returns
    -------
    BatteryModule
        Handle exposing ``pos_cmp``/``pos_port`` and
        ``neg_cmp``/``neg_port`` for further wiring.

    Notes
    -----
    Voltage adds in series: ``V_module = sum(V_cell_i)``.
    Capacity does not change in series: ``capacity_module = capacity_cell``.

    The wiring is purely structural -- there are no new equations or symbols
    introduced beyond what the cell factory itself declares -- so the module
    is fully differentiable through every cell parameter (R0, R1, C1,
    capacity_Ah, internal_resistance, ocv table, ...) just like an
    individual cell.
    """
    if n_cells < 1:
        raise ValueError(
            f"battery_module: n_cells must be >= 1, got {n_cells}"
        )
    cells = []
    for i in range(n_cells):
        cell = cell_factory(ev, f"{name}_cell{i}")
        cells.append(cell)
    # Wire cells in series: cell[i].n -> cell[i+1].p.
    for i in range(n_cells - 1):
        ad.connect(cells[i], "n", cells[i + 1], "p")
    return BatteryModule(cells, name)


def battery_pack(
    ev,
    ad,
    n_modules,
    n_cells_per_module,
    cell_factory,
    name="pack",
):
    """Build an ``n_modules`` × ``n_cells_per_module`` battery pack.

    ``n_modules`` series-strings are placed in *parallel*; each string has
    ``n_cells_per_module`` cells in *series*.  Wiring:

    - within a module: cell[i].neg -> cell[i+1].pos  (series)
    - across modules: module[i].pos -> module[0].pos and
      module[i].neg -> module[0].neg  (parallel buses)

    Parameters
    ----------
    ev, ad:
        the equation environment and acausal diagram.
    n_modules:
        number of parallel series-strings (>= 1).
    n_cells_per_module:
        number of cells per series-string (>= 1).
    cell_factory:
        callable ``(ev, name) -> ElecTwoPin``.  Called once per cell with a
        unique ``name``.  The factory receives the per-cell name only; if
        you need to vary parameters per cell, close over a counter or
        switch on the name suffix.
    name:
        pack name; used as a prefix for module and cell names.

    Returns
    -------
    BatteryPack
        Handle exposing the pack's positive and negative terminals.

    Notes
    -----
    Voltage: ``V_pack = n_cells_per_module * V_cell``  (series within a module).
    Capacity: ``C_pack = n_modules * C_cell``  (parallel across modules).

    Series within a module doesn't change capacity; parallel across modules
    doesn't change voltage.  This is the canonical equivalent-circuit
    pack topology.

    The wiring is purely structural -- fully differentiable through every
    cell parameter just like an individual cell.
    """
    if n_modules < 1:
        raise ValueError(
            f"battery_pack: n_modules must be >= 1, got {n_modules}"
        )
    if n_cells_per_module < 1:
        raise ValueError(
            f"battery_pack: n_cells_per_module must be >= 1, got "
            f"{n_cells_per_module}"
        )
    modules = []
    for j in range(n_modules):
        mod = battery_module(
            ev,
            ad,
            n_cells_per_module,
            cell_factory,
            name=f"{name}_mod{j}",
        )
        modules.append(mod)
    # Wire modules in parallel: tie every module's positive bus to module
    # 0's positive bus, and similarly for the negative bus.
    m0 = modules[0]
    for j in range(1, n_modules):
        mj = modules[j]
        ad.connect(mj.pos_cmp, mj.pos_port, m0.pos_cmp, m0.pos_port)
        ad.connect(mj.neg_cmp, mj.neg_port, m0.neg_cmp, m0.neg_port)
    return BatteryPack(modules, name)

class BatteryCellTabular(ElecTwoPin):
    """Table-driven battery cell with no RC transient (degenerate ECM).

    A simpler alternative to :class:`BatteryCellECM`: the equivalent circuit
    collapses to OCV(SOC) in series with a single ohmic resistance.  There is
    no RC pair — i.e. no transient/polarisation state — so integration is
    cheaper (one continuous state instead of two) at the cost of losing the
    short-time-scale voltage dynamics.

    Governing equations (passive sign convention, identical to
    :class:`BatteryCellECM`):

        1. V_terminal = OCV(SOC) + R_internal * Ip               (Kirchhoff)
        2. d(SOC)/dt  = Ip / (3600 * capacity_Ah)                (Coulomb counter)

    This is a degenerate ECM with ``R1 = C1 = 0``, but is shipped as its own
    class for clarity (callers don't have to invent dummy RC values to skip
    the transient) and for speed (one fewer continuous state per cell, which
    matters for module/pack composition).

    Sign convention: ``Ip`` is the current flowing **into** the positive pin
    from the rest of the circuit.  Positive ``Ip`` charges the cell, negative
    ``Ip`` discharges it — same as :class:`BatteryCellECM`.

    Optional thermal port exposes joule heating ``I^2 * R_internal`` as a
    heat-flow source.

    Args:
        capacity_Ah:
            Cell capacity in Amp-Hours.  Differentiable parameter.
        ocv_soc:
            SOC break-points (monotonically increasing array in [0, 1]) for
            the OCV-vs-SOC lookup table.
        ocv_volts:
            Open-circuit-voltage values (Volts) at each ``ocv_soc`` breakpoint.
        internal_resistance:
            Series ohmic resistance in Ohms.  Differentiable parameter.
        initial_soc:
            Initial state of charge (in [0, 1]).  Default 1.0 (fully charged).
        initial_soc_fixed:
            Whether ``initial_soc`` is a fixed initial condition.
        enable_heat_port:
            When ``True``, declare a thermal port ``"heat"`` whose heat-flow
            equals ``I^2 * R_internal``.
        enable_soc_port:
            When ``True``, declare a causal output port ``"soc"`` carrying SOC.
        enable_ocv_port:
            When ``True``, declare a causal output port ``"ocv"`` carrying
            the open-circuit voltage.
    """

    def __init__(
        self,
        ev,
        name: str | None = None,
        capacity_Ah: float = 1.0,
        ocv_soc=(0.0, 1.0),
        ocv_volts=(3.0, 4.2),
        internal_resistance: float = 0.01,
        initial_soc: float = 1.0,
        initial_soc_fixed: bool = False,
        enable_heat_port: bool = False,
        enable_soc_port: bool = False,
        enable_ocv_port: bool = False,
    ):
        self.name = self.__class__.__name__ if name is None else name

        # Initial terminal voltage = OCV(initial_soc) (no RC transient).
        ocv_soc_arr = npa.array(ocv_soc)
        ocv_volts_arr = npa.array(ocv_volts)
        V_ic = npa.interp(initial_soc, ocv_soc_arr, ocv_volts_arr)

        super().__init__(ev, self.name, V_ic=V_ic, I_ic=0.0)

        # ------------------------------------------------------------------
        # State variable: SOC only (no V_RC in the tabular model).
        # ------------------------------------------------------------------
        SOC = self.declare_symbol(
            ev,
            "SOC",
            self.name,
            kind=SymKind.var,
            ic=initial_soc,
            ic_fixed=initial_soc_fixed,
        )
        dSOC = self.declare_symbol(
            ev,
            "dSOC",
            self.name,
            kind=SymKind.var,
            int_sym=SOC,
            ic=0.0,
        )
        SOC.der_sym = dSOC

        # ------------------------------------------------------------------
        # Differentiable parameters.
        # ------------------------------------------------------------------
        cap = self.declare_symbol(
            ev,
            "capacity_Ah",
            self.name,
            kind=SymKind.param,
            val=capacity_Ah,
            validator=lambda x: x > 0.0,
            invalid_msg=(
                f"Component {self.__class__.__name__} {self.name} "
                "must have capacity_Ah>0"
            ),
        )
        R_sym = self.declare_symbol(
            ev,
            "internal_resistance",
            self.name,
            kind=SymKind.param,
            val=internal_resistance,
            validator=lambda x: x > 0.0,
            invalid_msg=(
                f"Component {self.__class__.__name__} {self.name} "
                "must have internal_resistance>0"
            ),
        )

        # OCV-vs-SOC lookup table (1D interp).  Differentiable through the
        # tabulated voltage values via ``jax.numpy.interp``.
        OCV_lut = self.declare_1D_lookup_table(
            ev,
            SOC.s,
            "ocv_soc",
            ocv_soc_arr,
            "ocv_volts",
            ocv_volts_arr,
            "OCV_lut",
        )

        # ------------------------------------------------------------------
        # Component equations.
        # ------------------------------------------------------------------
        # Eqn 2: Coulomb counter.  Passive sign: Ip > 0 ==> charging ==> SOC up.
        # Eqn 1: Terminal voltage = OCV(SOC) + R_internal * Ip.
        self.add_eqs(
            [
                sp.Eq(dSOC.s, self.Ip.s / (cap.s * 3600)),
                sp.Eq(self.V.s, OCV_lut.s + R_sym.s * self.Ip.s),
            ]
        )

        # ------------------------------------------------------------------
        # Optional thermal port: joule heating I^2 * R_internal.
        # ------------------------------------------------------------------
        if enable_heat_port:
            port_name = "heat"
            T, Q = self.declare_thermal_port(ev, port_name)
            # Flow vars are negative for flow leaving the component.
            self.add_eqs(
                [
                    sp.Eq(-Q.s, self.Ip.s * self.Ip.s * R_sym.s),
                ]
            )
            self.port_idx_to_name[2] = port_name

        # ------------------------------------------------------------------
        # Optional causal output ports for control / debugging.
        # ------------------------------------------------------------------
        if enable_soc_port:
            soc_out = self.declare_symbol(
                ev, "soc", self.name, kind=SymKind.outp
            )
            self.declare_equation(sp.Eq(soc_out.s, SOC.s), kind=EqnKind.outp)
        if enable_ocv_port:
            ocv_out = self.declare_symbol(
                ev, "ocv", self.name, kind=SymKind.outp
            )
            self.declare_equation(
                sp.Eq(ocv_out.s, OCV_lut.s), kind=EqnKind.outp
            )
