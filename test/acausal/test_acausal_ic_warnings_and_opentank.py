# SPDX-License-Identifier: MIT
"""Acausal IC-warning wording, weak-IC override warning,
and the ``OpenTank`` ``enabble_h_sensor`` typo.

1. The ill-conditioned-Jacobian warnings must not recommend ``scale=True``
   when the caller already passed it (the check runs on the raw, pre-scaling
   Jacobian, so the old wording was contradictory advice).
2. When a declared (non-default) weak IC is overridden by the
   consistent-initialization solve -- e.g. a gravity-tank level state with
   ``ic=0.4`` silently flattening to 0.0 because the port pressure IC is not
   fixed -- a warning must name the value that was discarded.
3. ``OpenTank(..., enabble_h_sensor=...)`` is renamed ``enable_h_sensor``;
   the misspelling still works but emits a DeprecationWarning.
"""

from __future__ import annotations

import warnings

import jaxonomy as jx

import pytest
import sympy as sp

from jaxonomy.acausal import (
    AcausalCompiler,
    AcausalDiagram,
    EqnEnv,
    electrical as elec,
    fluid as fld,
    hydraulic as hd,
)
from jaxonomy.acausal.component_library.base import SymKind
from jaxonomy.acausal.index_reduction.index_reduction import IndexReduction

pytestmark = pytest.mark.minimal

G_N = 9.81


def _compile_borderline_rc(*, scale):
    """Borderline-conditioned RC circuit (condition number ~1.4e4 > 1e4).

    Same network as test_v003_condition_number_threshold.py; at the default
    threshold it always trips the determined-ICs conditioning warning.
    """
    IndexReduction.clear_cache()  # warnings are skipped on a SED cache hit
    ev = EqnEnv()
    ad = AcausalDiagram()
    v = elec.VoltageSource(ev, name="v", V=1.0)
    r = elec.Resistor(ev, name="r", R=10.0)
    c = elec.Capacitor(
        ev, name="c", C=1e-3, initial_voltage=0.0, initial_voltage_fixed=True
    )
    gnd = elec.Ground(ev, name="gnd")
    ad.connect(v, "p", r, "p")
    ad.connect(r, "n", c, "p")
    ad.connect(c, "n", v, "n")
    ad.connect(v, "n", gnd, "p")
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        AcausalCompiler(ev, ad, scale=scale)()
    return [x for x in w if "condition number" in str(x.message)]


class TestConditioningHintWording:
    def test_without_scale_recommends_scale(self):
        cond_warns = _compile_borderline_rc(scale=False)
        assert cond_warns
        assert any(
            "Consider passing scale=True" in str(x.message) for x in cond_warns
        )

    def test_with_scale_does_not_recommend_scale(self):
        cond_warns = _compile_borderline_rc(scale=True)
        assert cond_warns, "conditioning warning should still fire with scale=True"
        offending = [
            str(x.message)
            for x in cond_warns
            if "Consider passing scale=True" in str(x.message)
        ]
        assert not offending, (
            "warning must not recommend scale=True when it is already set: "
            f"{offending}"
        )
        assert any(
            "scale=True is already set" in str(x.message) for x in cond_warns
        )


class GravityTank(hd.HydraulicOnePort):
    """Open gravity-drained tank; port at the bottom (P = rho*g*h).

    Mirrors the custom component from
    docs/examples/media/pinn_tank_network_dae.py. With
    ``pin_port_pressure=False`` the level IC ``h(0)`` is only a weak IC and
    the consistent-initialization solve flattens the level to zero.
    """

    def __init__(
        self, ev, name=None, area=1.0, h_ic=0.5, rho_ref=1000.0,
        pin_port_pressure=False,
    ):
        self.name = self.__class__.__name__ if name is None else name
        if pin_port_pressure:
            super().__init__(
                ev, self.name, P_ic=rho_ref * G_N * h_ic, P_ic_fixed=True
            )
        else:
            super().__init__(ev, self.name)
        self.area = self.declare_symbol(
            ev, "area", self.name, kind=SymKind.param, val=area,
            validator=lambda a: a > 0.0,
            invalid_msg=f"GravityTank {self.name} must have area>0",
        )
        self.h = self.declare_symbol(ev, "h", self.name, kind=SymKind.var, ic=h_ic)
        self.dh = self.declare_symbol(
            ev, "dh", self.name, kind=SymKind.var, int_sym=self.h, ic=0.0
        )
        self.h.der_sym = self.dh

    def finalize(self, ev):
        fluid = self.ports["port"].fluid
        self.add_eqs(
            [
                sp.Eq(self.P.s, fluid.density * G_N * self.h.s),
                sp.Eq(self.dh.s, self.M.s / (fluid.density * self.area.s)),
            ]
        )


def _compile_tank(*, pin_port_pressure, h_ic=0.4):
    """Tiny tank -> pipe -> reservoir network; returns recorded warnings."""
    IndexReduction.clear_cache()
    ev = EqnEnv()
    ad = AcausalDiagram()
    props = hd.HydraulicProperties(ev, fluid_name="water")
    tank = GravityTank(
        ev, name="tank", area=0.02, h_ic=h_ic,
        pin_port_pressure=pin_port_pressure,
    )
    pipe = hd.Pipe(ev, name="pipe", R=1e5)
    res = hd.PressureSource(ev, name="res", pressure=0.0)
    ad.connect(tank, "port", pipe, "port_a")
    ad.connect(pipe, "port_b", res, "port")
    ad.connect(props, "prop", tank, "port")
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        system = AcausalCompiler(ev, ad)()
        # the override check runs at system creation, where the
        # authoritative IC solve happens with resolved input values
        # (doing it at compile time regressed the compile benchmarks)
        builder = jx.DiagramBuilder()
        builder.add(system)
        builder.build().create_context()
    return [
        x
        for x in w
        if "overrode declared (non-fixed) initial conditions" in str(x.message)
    ]


class TestDeclaredWeakIcOverrideWarning:
    def test_weak_level_ic_flattening_warns(self):
        override_warns = _compile_tank(pin_port_pressure=False)
        assert override_warns, (
            "flattening a declared level IC (h(0)=0.4 -> 0.0) must warn"
        )
        msg = str(override_warns[0].message)
        # names the declared value and the value it actually received
        # (the solved level is ~0; the exact float can be a denormal)
        assert "declared ic=0.4" in msg
        assert "initialized to" in msg
        # actionable advice
        assert "ic_fixed=True" in msg

    def test_strong_port_ic_does_not_warn(self):
        assert not _compile_tank(pin_port_pressure=True), (
            "with the port pressure IC fixed, the declared level is honored "
            "and no override warning should fire"
        )


class TestOpenTankSensorKwarg:
    def test_new_spelling_works_without_warning(self):
        ev = EqnEnv()
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            tank = fld.OpenTank(ev, name="ot", enable_h_sensor=True)
        assert tank.enable_h_sensor is True
        assert hasattr(tank, "height_output")
        assert not [x for x in w if issubclass(x.category, DeprecationWarning)]

    def test_old_misspelling_still_works_with_deprecation(self):
        ev = EqnEnv()
        with pytest.warns(DeprecationWarning, match="enabble_h_sensor"):
            tank = fld.OpenTank(ev, name="ot", enabble_h_sensor=True)
        assert tank.enable_h_sensor is True
        assert tank.enabble_h_sensor is True  # back-compat attribute alias
        assert hasattr(tank, "height_output")

    def test_old_misspelling_false_disables_sensor(self):
        ev = EqnEnv()
        with pytest.warns(DeprecationWarning, match="enabble_h_sensor"):
            tank = fld.OpenTank(ev, name="ot", enabble_h_sensor=False)
        assert tank.enable_h_sensor is False
        assert not hasattr(tank, "height_output")
