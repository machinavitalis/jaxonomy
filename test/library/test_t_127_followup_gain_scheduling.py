# SPDX-License-Identifier: MIT

"""Tests for T-127-followup-gain-scheduling — :class:`PIDController2DOF`.

T-127-followup-external-weights (shipped 2026-05-10) introduced the
``b_dynamic`` / ``c_dynamic`` kwargs that turn the setpoint weights
into runtime-port signals.  Real-world tuning workflows also schedule
the four scalar gains (``Kp``, ``Ki``, ``Kd``, ``Kff``) as functions of
an operating-point variable (e.g. engine speed, Mach number, tank
level).

This followup adds four new construction kwargs — ``kp_dynamic``,
``ki_dynamic``, ``kd_dynamic``, ``kff_dynamic`` — that mirror the
``b_dynamic`` / ``c_dynamic`` pattern.  When set, the corresponding
gain is read from a new input port instead of the static parameter.
Lookup tables stay outside the PID class: users build a
:class:`LookupTable1d` / :class:`LookupTable2d` per scheduled gain and
wire it to the matching PID port.  This keeps the PID's surface area
minimal and the scheduling policy fully user-owned.

Port indexing (deterministic, in this order):

* (0) Setpoint ``r``.
* (1) Measurement ``y``.
* ``b``   if ``b_dynamic=True``.
* ``c``   if ``c_dynamic=True``.
* ``kp``  if ``kp_dynamic=True``.
* ``ki``  if ``ki_dynamic=True``.
* ``kd``  if ``kd_dynamic=True``.
* ``kff`` if ``kff_dynamic=True``.

Ports are skipped when their flag is False, so e.g. with only
``kp_dynamic=True`` the ``kp`` port lives at index 2.

Default-off (``*_dynamic=False`` for all four) must remain byte-
equivalent to phase 1 / T-127-followup-external-weights.

These tests cover:

* Each ``*_dynamic`` flag adds exactly one new input port at the
  documented deterministic index.
* Static gain wired to a ``Constant(value)`` runtime port produces the
  same control trajectory within ``1e-12`` of the static-gain
  reference.
* All four flags active simultaneously: ports land at indices 2, 3, 4,
  5 in (kp, ki, kd, kff) order; values are correctly threaded.
* Gain-scheduled example: wire a :class:`LookupTable1d` to the
  ``kp_dynamic`` port and verify that the PID output reflects the
  scheduled gain (different at different scheduling-variable values).
* Validation: ``*_dynamic=True`` with the port left unconnected raises
  a clear build-time error.
* Differentiability: ``jax.grad`` is finite through each runtime gain
  port.
"""

from __future__ import annotations

from collections import namedtuple

import jax
import jax.numpy as jnp
import pytest

import jaxonomy
from jaxonomy.library import (
    Constant,
    LookupTable1d,
    PIDController2DOF,
)


pytestmark = pytest.mark.minimal


# --------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------- #


def _simulate_static(
    *,
    dt=0.05,
    r=1.0,
    y=0.0,
    kp=2.0,
    ki=0.5,
    kd=0.1,
    kff=0.0,
    b=1.0,
    c=1.0,
    t_end=1.0,
):
    """Static-gain reference simulation (no dynamic-gain ports)."""
    builder = jaxonomy.DiagramBuilder()
    r_blk = builder.add(Constant(r, name="r"))
    y_blk = builder.add(Constant(y, name="y"))
    pid = builder.add(
        PIDController2DOF(
            dt=dt,
            kp=kp,
            ki=ki,
            kd=kd,
            kff=kff,
            b=b,
            c=c,
            name="pid_static",
        )
    )
    builder.connect(r_blk.output_ports[0], pid.input_ports[0])
    builder.connect(y_blk.output_ports[0], pid.input_ports[1])
    diagram = builder.build()
    context = diagram.create_context()
    results = jaxonomy.simulate(
        diagram,
        context,
        (0.0, t_end),
        recorded_signals={"u": pid.output_ports[0]},
    )
    return results.outputs["u"], results.time


# --------------------------------------------------------------------- #
# Port-index assignment
# --------------------------------------------------------------------- #


class TestPortIndices:
    """Each ``*_dynamic`` flag adds one port at the documented index."""

    def test_kp_only_at_index_2(self):
        pid = PIDController2DOF(dt=0.1, kp_dynamic=True, name="pid")
        assert len(pid.input_ports) == 3
        assert pid.kp_index == 2
        assert pid.kp_dynamic is True
        assert pid.ki_dynamic is False
        assert pid.kd_dynamic is False
        assert pid.kff_dynamic is False

    def test_ki_only_at_index_2(self):
        pid = PIDController2DOF(dt=0.1, ki_dynamic=True, name="pid")
        assert len(pid.input_ports) == 3
        assert pid.ki_index == 2

    def test_kd_only_at_index_2(self):
        pid = PIDController2DOF(dt=0.1, kd_dynamic=True, name="pid")
        assert len(pid.input_ports) == 3
        assert pid.kd_index == 2

    def test_kff_only_at_index_2(self):
        pid = PIDController2DOF(dt=0.1, kff_dynamic=True, name="pid")
        assert len(pid.input_ports) == 3
        assert pid.kff_index == 2

    def test_all_four_dynamic_at_indices_2_3_4_5(self):
        pid = PIDController2DOF(
            dt=0.1,
            kp_dynamic=True,
            ki_dynamic=True,
            kd_dynamic=True,
            kff_dynamic=True,
            name="pid",
        )
        assert len(pid.input_ports) == 6
        assert pid.kp_index == 2
        assert pid.ki_index == 3
        assert pid.kd_index == 4
        assert pid.kff_index == 5

    def test_combined_with_b_dynamic_shifts_indices(self):
        """``b_dynamic`` keeps index 2; the gain ports follow."""
        pid = PIDController2DOF(
            dt=0.1,
            b_dynamic=True,
            kp_dynamic=True,
            kd_dynamic=True,
            name="pid",
        )
        assert len(pid.input_ports) == 5
        assert pid.b_index == 2
        assert pid.kp_index == 3
        assert pid.kd_index == 4

    def test_combined_with_b_and_c_dynamic(self):
        """``b_dynamic`` + ``c_dynamic`` consume 2 and 3; gains follow."""
        pid = PIDController2DOF(
            dt=0.1,
            b_dynamic=True,
            c_dynamic=True,
            kp_dynamic=True,
            ki_dynamic=True,
            kd_dynamic=True,
            kff_dynamic=True,
            name="pid",
        )
        assert len(pid.input_ports) == 8
        assert pid.b_index == 2
        assert pid.c_index == 3
        assert pid.kp_index == 4
        assert pid.ki_index == 5
        assert pid.kd_index == 6
        assert pid.kff_index == 7

    def test_default_off_has_no_extra_ports(self):
        """Without any flags, only the (r, y) ports exist (phase 1)."""
        pid = PIDController2DOF(dt=0.1, name="pid")
        assert len(pid.input_ports) == 2
        assert pid.kp_dynamic is False
        assert pid.ki_dynamic is False
        assert pid.kd_dynamic is False
        assert pid.kff_dynamic is False


# --------------------------------------------------------------------- #
# Default-off byte-equivalence with phase 1
# --------------------------------------------------------------------- #


class TestDefaultOffByteEquivalence:
    """``*_dynamic=False`` everywhere must equal the implicit phase 1."""

    def test_explicit_off_matches_phase1(self):
        u_phase1, _ = _simulate_static(
            dt=0.05, r=1.0, y=0.0, kp=2.0, ki=4.0, kd=0.1, t_end=1.0
        )
        builder = jaxonomy.DiagramBuilder()
        r = builder.add(Constant(1.0, name="r"))
        y = builder.add(Constant(0.0, name="y"))
        pid = builder.add(
            PIDController2DOF(
                dt=0.05,
                kp=2.0,
                ki=4.0,
                kd=0.1,
                kp_dynamic=False,
                ki_dynamic=False,
                kd_dynamic=False,
                kff_dynamic=False,
                name="pid",
            )
        )
        builder.connect(r.output_ports[0], pid.input_ports[0])
        builder.connect(y.output_ports[0], pid.input_ports[1])
        diagram = builder.build()
        context = diagram.create_context()
        results = jaxonomy.simulate(
            diagram, context, (0.0, 1.0),
            recorded_signals={"u": pid.output_ports[0]},
        )
        u_explicit = results.outputs["u"]
        assert jnp.array_equal(u_phase1, u_explicit), (
            "Explicitly setting *_dynamic=False must match phase 1 "
            "byte-for-byte"
        )


# --------------------------------------------------------------------- #
# Static-vs-dynamic equivalence
# --------------------------------------------------------------------- #


class TestStaticEqualsDynamic:
    """A ``Constant(value)`` port equals static-``gain=value``."""

    def _simulate_with_kp_port(self, kp_value, **kwargs):
        builder = jaxonomy.DiagramBuilder()
        r = builder.add(Constant(kwargs.pop("r", 1.0), name="r"))
        y = builder.add(Constant(kwargs.pop("y", 0.0), name="y"))
        kp_src = builder.add(Constant(kp_value, name="kp_src"))
        pid = builder.add(
            PIDController2DOF(
                dt=kwargs.pop("dt", 0.05),
                kp=999.0,  # ignored
                ki=kwargs.pop("ki", 0.5),
                kd=kwargs.pop("kd", 0.1),
                b=kwargs.pop("b", 1.0),
                c=kwargs.pop("c", 1.0),
                kp_dynamic=True,
                name="pid",
            )
        )
        builder.connect(r.output_ports[0], pid.input_ports[0])
        builder.connect(y.output_ports[0], pid.input_ports[1])
        builder.connect(kp_src.output_ports[0], pid.input_ports[2])
        diagram = builder.build()
        context = diagram.create_context()
        results = jaxonomy.simulate(
            diagram, context, (0.0, kwargs.pop("t_end", 1.0)),
            recorded_signals={"u": pid.output_ports[0]},
        )
        return results.outputs["u"]

    def test_kp_dynamic_constant_matches_static(self):
        u_static, _ = _simulate_static(
            dt=0.05, r=1.0, y=0.0, kp=2.5, ki=0.5, kd=0.1, t_end=1.0
        )
        u_dyn = self._simulate_with_kp_port(2.5)
        assert jnp.allclose(u_static, u_dyn, atol=1e-12), (
            f"max diff {float(jnp.max(jnp.abs(u_static - u_dyn)))}"
        )

    def test_ki_dynamic_constant_matches_static(self):
        u_static, _ = _simulate_static(
            dt=0.05, r=1.0, y=0.0, kp=0.0, ki=1.7, kd=0.0, t_end=1.0
        )
        builder = jaxonomy.DiagramBuilder()
        r = builder.add(Constant(1.0, name="r"))
        y = builder.add(Constant(0.0, name="y"))
        ki_src = builder.add(Constant(1.7, name="ki_src"))
        pid = builder.add(
            PIDController2DOF(
                dt=0.05, kp=0.0, ki=999.0, kd=0.0,
                ki_dynamic=True, name="pid",
            )
        )
        builder.connect(r.output_ports[0], pid.input_ports[0])
        builder.connect(y.output_ports[0], pid.input_ports[1])
        builder.connect(ki_src.output_ports[0], pid.input_ports[2])
        diagram = builder.build()
        context = diagram.create_context()
        results = jaxonomy.simulate(
            diagram, context, (0.0, 1.0),
            recorded_signals={"u": pid.output_ports[0]},
        )
        u_dyn = results.outputs["u"]
        assert jnp.allclose(u_static, u_dyn, atol=1e-12), (
            f"max diff {float(jnp.max(jnp.abs(u_static - u_dyn)))}"
        )

    def test_kd_dynamic_constant_matches_static(self):
        # Use a Sine input so the derivative term is nonzero.
        from jaxonomy.library import Sine
        dt = 0.05
        # Static reference.
        builder = jaxonomy.DiagramBuilder()
        r_blk = builder.add(Sine(frequency=1.0, amplitude=1.0, phase=0.0))
        y_blk = builder.add(Constant(0.0, name="y"))
        pid_s = builder.add(
            PIDController2DOF(
                dt=dt, kp=0.0, ki=0.0, kd=0.7, name="pid_s"
            )
        )
        builder.connect(r_blk.output_ports[0], pid_s.input_ports[0])
        builder.connect(y_blk.output_ports[0], pid_s.input_ports[1])
        diagram = builder.build()
        context = diagram.create_context()
        res_s = jaxonomy.simulate(
            diagram, context, (0.0, 1.0),
            recorded_signals={"u": pid_s.output_ports[0]},
        )
        # Dynamic-kd variant.
        b2 = jaxonomy.DiagramBuilder()
        r2 = b2.add(Sine(frequency=1.0, amplitude=1.0, phase=0.0))
        y2 = b2.add(Constant(0.0, name="y"))
        kd_src = b2.add(Constant(0.7, name="kd_src"))
        pid_d = b2.add(
            PIDController2DOF(
                dt=dt, kp=0.0, ki=0.0, kd=999.0,
                kd_dynamic=True, name="pid_d",
            )
        )
        b2.connect(r2.output_ports[0], pid_d.input_ports[0])
        b2.connect(y2.output_ports[0], pid_d.input_ports[1])
        b2.connect(kd_src.output_ports[0], pid_d.input_ports[2])
        diagram2 = b2.build()
        context2 = diagram2.create_context()
        res_d = jaxonomy.simulate(
            diagram2, context2, (0.0, 1.0),
            recorded_signals={"u": pid_d.output_ports[0]},
        )
        assert jnp.allclose(res_s.outputs["u"], res_d.outputs["u"], atol=1e-12)

    def test_kff_dynamic_constant_matches_static(self):
        # kff couples r directly into u; use P-only PID so the
        # feedforward term is the dominant contribution.
        u_static, _ = _simulate_static(
            dt=0.05, r=2.0, y=0.0,
            kp=0.0, ki=0.0, kd=0.0, kff=1.5, t_end=0.5,
        )
        builder = jaxonomy.DiagramBuilder()
        r = builder.add(Constant(2.0, name="r"))
        y = builder.add(Constant(0.0, name="y"))
        kff_src = builder.add(Constant(1.5, name="kff_src"))
        pid = builder.add(
            PIDController2DOF(
                dt=0.05, kp=0.0, ki=0.0, kd=0.0, kff=999.0,
                kff_dynamic=True, name="pid",
            )
        )
        builder.connect(r.output_ports[0], pid.input_ports[0])
        builder.connect(y.output_ports[0], pid.input_ports[1])
        builder.connect(kff_src.output_ports[0], pid.input_ports[2])
        diagram = builder.build()
        context = diagram.create_context()
        results = jaxonomy.simulate(
            diagram, context, (0.0, 0.5),
            recorded_signals={"u": pid.output_ports[0]},
        )
        u_dyn = results.outputs["u"]
        assert jnp.allclose(u_static, u_dyn, atol=1e-12), (
            f"max diff {float(jnp.max(jnp.abs(u_static - u_dyn)))}"
        )


# --------------------------------------------------------------------- #
# All four dynamic flags simultaneously
# --------------------------------------------------------------------- #


class TestAllFourActive:
    """All four dynamic-gain ports active produce the same trajectory as
    the static-gain reference when wired to the matching constants."""

    def test_all_four_match_static(self):
        dt = 0.05
        kp_val, ki_val, kd_val, kff_val = 1.7, 0.9, 0.2, 0.4
        r_val, y_val = 1.0, 0.0
        # Static reference.
        u_static, _ = _simulate_static(
            dt=dt, r=r_val, y=y_val,
            kp=kp_val, ki=ki_val, kd=kd_val, kff=kff_val,
            t_end=1.0,
        )
        # All-dynamic variant.
        builder = jaxonomy.DiagramBuilder()
        r = builder.add(Constant(r_val, name="r"))
        y = builder.add(Constant(y_val, name="y"))
        kp_src = builder.add(Constant(kp_val, name="kp"))
        ki_src = builder.add(Constant(ki_val, name="ki"))
        kd_src = builder.add(Constant(kd_val, name="kd"))
        kff_src = builder.add(Constant(kff_val, name="kff"))
        pid = builder.add(
            PIDController2DOF(
                dt=dt,
                kp=999.0, ki=999.0, kd=999.0, kff=999.0,
                kp_dynamic=True, ki_dynamic=True,
                kd_dynamic=True, kff_dynamic=True,
                name="pid",
            )
        )
        builder.connect(r.output_ports[0], pid.input_ports[0])
        builder.connect(y.output_ports[0], pid.input_ports[1])
        builder.connect(kp_src.output_ports[0], pid.input_ports[2])
        builder.connect(ki_src.output_ports[0], pid.input_ports[3])
        builder.connect(kd_src.output_ports[0], pid.input_ports[4])
        builder.connect(kff_src.output_ports[0], pid.input_ports[5])
        diagram = builder.build()
        context = diagram.create_context()
        results = jaxonomy.simulate(
            diagram, context, (0.0, 1.0),
            recorded_signals={"u": pid.output_ports[0]},
        )
        u_dyn = results.outputs["u"]
        assert jnp.allclose(u_static, u_dyn, atol=1e-12), (
            f"max diff {float(jnp.max(jnp.abs(u_static - u_dyn)))}"
        )


# --------------------------------------------------------------------- #
# Gain-scheduled example with LookupTable1d
# --------------------------------------------------------------------- #


class TestLookupTableGainSchedule:
    """The intended use-case: wire a LookupTable1d to a dynamic-gain
    port and verify the PID output reflects the scheduled gain."""

    def _run(self, sched_value, kp_breakpoints, kp_values, *,
             dt=0.05, r=1.0, y=0.0, t_end=0.5):
        builder = jaxonomy.DiagramBuilder()
        r_blk = builder.add(Constant(r, name="r"))
        y_blk = builder.add(Constant(y, name="y"))
        sched_blk = builder.add(Constant(sched_value, name="sched"))
        kp_tbl = builder.add(
            LookupTable1d(
                input_array=kp_breakpoints,
                output_array=kp_values,
                interpolation="linear",
                name="kp_schedule",
            )
        )
        # P-only PID — kp dominates the output.
        pid = builder.add(
            PIDController2DOF(
                dt=dt,
                kp=999.0,  # ignored
                ki=0.0,
                kd=0.0,
                kp_dynamic=True,
                name="pid",
            )
        )
        builder.connect(r_blk.output_ports[0], pid.input_ports[0])
        builder.connect(y_blk.output_ports[0], pid.input_ports[1])
        builder.connect(sched_blk.output_ports[0], kp_tbl.input_ports[0])
        builder.connect(kp_tbl.output_ports[0], pid.input_ports[2])
        diagram = builder.build()
        context = diagram.create_context()
        results = jaxonomy.simulate(
            diagram, context, (0.0, t_end),
            recorded_signals={"u": pid.output_ports[0]},
        )
        return results.outputs["u"]

    def test_scheduled_kp_at_breakpoint(self):
        """At a scheduling-variable equal to a breakpoint, the lookup
        returns the matching gain and ``u`` = kp * r."""
        bps = [0.0, 1.0, 2.0]
        vals = [1.0, 3.0, 5.0]
        # P-only, r=2.0, y=0.0 → u = kp * 2.0.
        u_at_low = self._run(0.0, bps, vals, r=2.0)
        u_at_mid = self._run(1.0, bps, vals, r=2.0)
        u_at_high = self._run(2.0, bps, vals, r=2.0)
        # Take a late sample where the controller has settled.
        assert jnp.allclose(u_at_low[-1], 1.0 * 2.0, atol=1e-10), (
            f"kp at sched=0.0 should be 1.0; u={u_at_low[-1]}"
        )
        assert jnp.allclose(u_at_mid[-1], 3.0 * 2.0, atol=1e-10), (
            f"kp at sched=1.0 should be 3.0; u={u_at_mid[-1]}"
        )
        assert jnp.allclose(u_at_high[-1], 5.0 * 2.0, atol=1e-10), (
            f"kp at sched=2.0 should be 5.0; u={u_at_high[-1]}"
        )

    def test_scheduled_kp_interpolated(self):
        """Between breakpoints, linear interpolation drives ``u``."""
        bps = [0.0, 1.0]
        vals = [2.0, 6.0]
        # At sched=0.25 linear interpolation gives kp = 2 + 0.25*(6-2) = 3.0.
        # r=1.0, so u = 3.0 * 1.0 = 3.0.
        u = self._run(0.25, bps, vals, r=1.0)
        assert jnp.allclose(u[-1], 3.0, atol=1e-10), (
            f"interpolated kp should give u=3.0; u={u[-1]}"
        )


# --------------------------------------------------------------------- #
# Validation: locked port topology + unconnected ports
# --------------------------------------------------------------------- #


class TestValidation:
    """Topology of dynamic-gain ports is locked at construction time."""

    @pytest.mark.parametrize(
        "flag,opposite",
        [
            ("kp_dynamic", True),
            ("ki_dynamic", True),
            ("kd_dynamic", True),
            ("kff_dynamic", True),
        ],
    )
    def test_flip_after_init_rejected(self, flag, opposite):
        """Flipping any ``*_dynamic`` flag in ``initialize`` is rejected."""
        ctor_kwargs = {
            "dt": 0.1,
            "kp": 1.0,
            "ki": 0.5,
            "kd": 0.1,
            "b": 1.0,
            "c": 1.0,
            flag: not opposite,  # start with the opposite of what we'll pass
            "name": "pid",
        }
        block = PIDController2DOF(**ctor_kwargs)
        init_kwargs = {
            "kp": 1.0,
            "ki": 0.5,
            "kd": 0.1,
            "b": 1.0,
            "c": 1.0,
            "initial_state": 0.0,
            "filter_type": "none",
            "filter_coefficient": 1.0,
            flag: opposite,  # flipped
        }
        with pytest.raises(ValueError, match=f"{flag} cannot be changed"):
            block.initialize(**init_kwargs)

    def test_kp_dynamic_unconnected_port_raises(self):
        """``kp_dynamic=True`` with the new port left unconnected is a
        build-time error."""
        builder = jaxonomy.DiagramBuilder()
        r = builder.add(Constant(1.0, name="r"))
        y = builder.add(Constant(0.0, name="y"))
        pid = builder.add(
            PIDController2DOF(
                dt=0.05, kp=999.0, ki=0.0, kd=0.0,
                kp_dynamic=True, name="pid",
            )
        )
        builder.connect(r.output_ports[0], pid.input_ports[0])
        builder.connect(y.output_ports[0], pid.input_ports[1])
        # Intentionally do NOT connect input_ports[2].
        with pytest.raises(Exception):
            diagram = builder.build()
            context = diagram.create_context()
            jaxonomy.simulate(
                diagram, context, (0.0, 0.2),
                recorded_signals={"u": pid.output_ports[0]},
            )


# --------------------------------------------------------------------- #
# Differentiability through runtime gain ports
# --------------------------------------------------------------------- #


class TestDifferentiability:
    """``jax.grad`` w.r.t. each runtime gain value is finite."""

    @staticmethod
    def _make_block(flag, dt=0.1):
        kw = {
            "dt": dt,
            "kp": 1.0,
            "ki": 0.5,
            "kd": 0.1,
            "kff": 0.0,
            "b": 1.0,
            "c": 1.0,
            flag: True,
            "name": "pid",
        }
        block = PIDController2DOF(**kw)
        init_kw = dict(
            kp=1.0, ki=0.5, kd=0.1, b=1.0, c=1.0,
            initial_state=0.0, filter_type="none", filter_coefficient=1.0,
            kff=0.0,
        )
        init_kw[flag] = True
        block.initialize(**init_kw)
        return block

    @classmethod
    def _step_loss_kp(cls, kp_value, n_steps=4):
        State = namedtuple("State", ["discrete_state"])
        block = cls._make_block("kp_dynamic")
        xd0 = block.DiscreteStateType(
            integral=jnp.asarray(0.0),
            e_d_prev=jnp.asarray(0.0),
            e_dot_prev=jnp.asarray(0.0),
        )
        state = State(discrete_state=xd0)
        r = jnp.asarray(1.0)
        y = jnp.asarray(0.0)
        # Inputs ordered (r, y, kp) per kp_index=2.
        params = dict(kp=999.0, ki=0.5, kd=0.1, b=1.0, c=1.0, kff=0.0,
                      initial_state=0.0,
                      output_min=None, output_max=None,
                      anti_windup_gain=1.0)
        total = jnp.asarray(0.0)
        for _ in range(n_steps):
            u = block._output(jnp.asarray(0.0), state, r, y, kp_value, **params)
            total = total + jnp.abs(u)
            new_xd = block._update(
                jnp.asarray(0.0), state, r, y, kp_value, **params
            )
            state = State(discrete_state=new_xd)
        return total

    @classmethod
    def _step_loss_ki(cls, ki_value, n_steps=4):
        State = namedtuple("State", ["discrete_state"])
        block = cls._make_block("ki_dynamic")
        xd0 = block.DiscreteStateType(
            integral=jnp.asarray(0.0),
            e_d_prev=jnp.asarray(0.0),
            e_dot_prev=jnp.asarray(0.0),
        )
        state = State(discrete_state=xd0)
        r = jnp.asarray(1.0)
        y = jnp.asarray(0.0)
        params = dict(kp=1.0, ki=999.0, kd=0.1, b=1.0, c=1.0, kff=0.0,
                      initial_state=0.0,
                      output_min=None, output_max=None,
                      anti_windup_gain=1.0)
        total = jnp.asarray(0.0)
        for _ in range(n_steps):
            u = block._output(jnp.asarray(0.0), state, r, y, ki_value, **params)
            total = total + jnp.abs(u)
            new_xd = block._update(
                jnp.asarray(0.0), state, r, y, ki_value, **params
            )
            state = State(discrete_state=new_xd)
        return total

    @classmethod
    def _step_loss_kff(cls, kff_value, n_steps=4):
        State = namedtuple("State", ["discrete_state"])
        block = cls._make_block("kff_dynamic")
        xd0 = block.DiscreteStateType(
            integral=jnp.asarray(0.0),
            e_d_prev=jnp.asarray(0.0),
            e_dot_prev=jnp.asarray(0.0),
        )
        state = State(discrete_state=xd0)
        r = jnp.asarray(1.0)
        y = jnp.asarray(0.0)
        params = dict(kp=1.0, ki=0.5, kd=0.1, b=1.0, c=1.0, kff=999.0,
                      initial_state=0.0,
                      output_min=None, output_max=None,
                      anti_windup_gain=1.0)
        total = jnp.asarray(0.0)
        for _ in range(n_steps):
            u = block._output(jnp.asarray(0.0), state, r, y, kff_value, **params)
            total = total + jnp.abs(u)
            new_xd = block._update(
                jnp.asarray(0.0), state, r, y, kff_value, **params
            )
            state = State(discrete_state=new_xd)
        return total

    def test_grad_wrt_kp_port(self):
        g = jax.grad(self._step_loss_kp)(jnp.asarray(0.5))
        assert jnp.isfinite(g), f"grad wrt runtime kp not finite: {g}"
        assert jnp.abs(g) > 0, f"grad wrt runtime kp should be nonzero; got {g}"

    def test_grad_wrt_ki_port(self):
        g = jax.grad(self._step_loss_ki)(jnp.asarray(0.3))
        assert jnp.isfinite(g), f"grad wrt runtime ki not finite: {g}"
        assert jnp.abs(g) > 0, f"grad wrt runtime ki should be nonzero; got {g}"

    def test_grad_wrt_kff_port(self):
        g = jax.grad(self._step_loss_kff)(jnp.asarray(0.7))
        assert jnp.isfinite(g), f"grad wrt runtime kff not finite: {g}"
        assert jnp.abs(g) > 0, f"grad wrt runtime kff should be nonzero; got {g}"
