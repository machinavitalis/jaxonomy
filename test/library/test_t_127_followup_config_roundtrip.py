# SPDX-License-Identifier: MIT

"""Tests for T-127-followup-config-roundtrip — :class:`PIDController2DOF`.

This followup adds ``to_dict()`` / ``from_dict()`` so a configured
:class:`PIDController2DOF` block can be serialized to a JSON-friendly
dict (and back) without losing any construction-time field.  The
target use cases:

* Config-driven test setups (build a block from a fixture dict).
* Saving controller designs to JSON for review.
* Loading pre-tuned controllers from a parameter store.

These tests cover:

* Default config: every documented field round-trips with its
  documented default value.
* Custom config (anti-windup + feedforward + deadband + gain
  scheduling): ``to_dict()`` -> ``from_dict()`` reconstructs a block
  whose open-loop step response is identical (bit-for-bit when
  possible, otherwise within floating-point tolerance) to the
  original.
* JSON safety: ``json.dumps`` on ``to_dict()`` succeeds and the
  resulting string round-trips through ``json.loads`` + ``from_dict``.
* Validation: missing required field (``dt``) raises a clear
  ``ValueError`` from ``from_dict``.
* Dynamic-port topology flags (``b_dynamic`` / ``kp_dynamic`` / ...)
  survive the round-trip and the reconstructed block still declares
  the right number of input ports.
"""

from __future__ import annotations

import json

import jax.numpy as jnp
import numpy as np
import pytest

import jaxonomy
from jaxonomy.library import (
    Constant,
    PIDController2DOF,
)


pytestmark = pytest.mark.minimal


# --------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------- #


def _simulate_open_loop(pid_factory, *, r=1.0, y=0.0, t_end=1.0, dt=0.05):
    """Drive a freshly-built ``PIDController2DOF`` with constant r/y.

    ``pid_factory`` is a zero-arg callable that returns a new
    ``PIDController2DOF`` (so the same factory can be used twice to
    rebuild the diagram for the "original" and "reconstructed"
    controllers in the round-trip test).
    """
    builder = jaxonomy.DiagramBuilder()
    r_b = builder.add(Constant(r, name="r"))
    y_b = builder.add(Constant(y, name="y"))
    pid = builder.add(pid_factory())
    builder.connect(r_b.output_ports[0], pid.input_ports[0])
    builder.connect(y_b.output_ports[0], pid.input_ports[1])
    diagram = builder.build()
    context = diagram.create_context()
    recorded = {"u": pid.output_ports[0]}
    results = jaxonomy.simulate(
        diagram, context, (0.0, t_end), recorded_signals=recorded
    )
    return np.asarray(results.outputs["u"])


# --------------------------------------------------------------------- #
# 1. Default config — every field appears with its documented default.
# --------------------------------------------------------------------- #


def test_default_config_to_dict():
    pid = PIDController2DOF(dt=0.1, name="pid")
    data = pid.to_dict()

    # Static fields (mode strings + topology flags).
    assert data["dt"] == pytest.approx(0.1)
    assert data["filter_type"] == "none"
    assert data["filter_coefficient"] == pytest.approx(1.0)
    assert data["anti_windup_method"] == "none"
    assert data["b_dynamic"] is False
    assert data["c_dynamic"] is False
    assert data["integrator_method"] == "forward_euler"
    assert data["derivative_method"] == "forward_diff"
    assert data["kp_dynamic"] is False
    assert data["ki_dynamic"] is False
    assert data["kd_dynamic"] is False
    assert data["kff_dynamic"] is False
    assert data["error_deadband_mode"] == "hard"

    # Dynamic-parameter scalars.
    assert data["kp"] == pytest.approx(1.0)
    assert data["ki"] == pytest.approx(1.0)
    assert data["kd"] == pytest.approx(1.0)
    assert data["b"] == pytest.approx(1.0)
    assert data["c"] == pytest.approx(1.0)
    assert data["initial_state"] == pytest.approx(0.0)
    assert data["output_min"] is None
    assert data["output_max"] is None
    assert data["anti_windup_gain"] == pytest.approx(1.0)
    assert data["kff"] == pytest.approx(0.0)
    assert data["error_deadband"] == pytest.approx(0.0)
    assert data["error_deadband_sharpness"] == pytest.approx(10.0)

    # Every documented field is present (and nothing else).
    expected_keys = set(
        PIDController2DOF._CONFIG_STATIC_FIELDS
        + PIDController2DOF._CONFIG_DYNAMIC_FIELDS
    )
    assert set(data.keys()) == expected_keys


# --------------------------------------------------------------------- #
# 2. Custom config — round-trip preserves behavior on a step input.
# --------------------------------------------------------------------- #


def _make_custom_pid():
    """Custom config exercising every T-127 follow-up surface."""
    return PIDController2DOF(
        dt=0.05,
        kp=2.5,
        ki=0.7,
        kd=0.3,
        b=0.8,
        c=0.5,
        initial_state=0.1,
        filter_type="none",
        filter_coefficient=1.0,
        output_min=-1.5,
        output_max=2.0,
        anti_windup_method="back_calc",
        anti_windup_gain=0.5,
        kff=0.2,
        integrator_method="trapezoidal",
        derivative_method="backward_diff",
        error_deadband=0.02,
        error_deadband_mode="smooth",
        error_deadband_sharpness=12.0,
        name="pid_custom",
    )


def test_custom_config_dict_round_trip_preserves_fields():
    original = _make_custom_pid()
    data = original.to_dict()
    reconstructed = PIDController2DOF.from_dict(data, name="pid_reload")
    # Re-serialize the reconstructed block; the dict must be identical
    # except for the (intentionally fresh) name kwarg, which doesn't
    # appear in the config dict.
    assert reconstructed.to_dict() == data


def test_custom_config_round_trip_preserves_step_response():
    """Open-loop step response of the reconstructed block matches the
    original bit-for-bit within float tolerance."""
    u_original = _simulate_open_loop(_make_custom_pid, r=1.0, y=0.0, t_end=0.5)

    cfg = _make_custom_pid().to_dict()

    def _rebuild():
        return PIDController2DOF.from_dict(cfg, name="pid_reload")

    u_reload = _simulate_open_loop(_rebuild, r=1.0, y=0.0, t_end=0.5)

    assert u_original.shape == u_reload.shape
    np.testing.assert_allclose(u_reload, u_original, rtol=1e-12, atol=1e-12)


# --------------------------------------------------------------------- #
# 3. JSON safety — json.dumps / json.loads round-trip is supported.
# --------------------------------------------------------------------- #


def test_json_dumps_loads_round_trip():
    original = _make_custom_pid()
    data = original.to_dict()

    blob = json.dumps(data)
    # All values are JSON primitives — no TypeError from json.dumps.
    assert isinstance(blob, str)
    assert len(blob) > 0

    reloaded = json.loads(blob)
    # Floats may round-trip with the same string repr; check
    # numerically via the rebuilt block's dict.
    rebuilt = PIDController2DOF.from_dict(reloaded, name="pid_json")
    assert rebuilt.to_dict() == data


# --------------------------------------------------------------------- #
# 4. Validation — missing required field raises a clear error.
# --------------------------------------------------------------------- #


def test_from_dict_missing_dt_raises():
    with pytest.raises(ValueError, match="missing required field 'dt'"):
        PIDController2DOF.from_dict({})


def test_from_dict_none_dt_raises():
    with pytest.raises(ValueError, match="missing required field 'dt'"):
        PIDController2DOF.from_dict({"dt": None, "kp": 1.0})


# --------------------------------------------------------------------- #
# 5. Dynamic-port topology — *_dynamic flags survive round-trip.
# --------------------------------------------------------------------- #


def test_dynamic_flag_round_trip_preserves_port_count():
    pid = PIDController2DOF(
        dt=0.05,
        b_dynamic=True,
        c_dynamic=True,
        kp_dynamic=True,
        ki_dynamic=False,
        kd_dynamic=False,
        kff_dynamic=True,
        name="pid_dyn",
    )
    data = pid.to_dict()
    assert data["b_dynamic"] is True
    assert data["c_dynamic"] is True
    assert data["kp_dynamic"] is True
    assert data["ki_dynamic"] is False
    assert data["kff_dynamic"] is True

    rebuilt = PIDController2DOF.from_dict(data, name="pid_dyn_reload")
    # Original block: r, y, b, c, kp, kff → 6 input ports.
    assert len(pid.input_ports) == 6
    assert len(rebuilt.input_ports) == 6
    # Port indices recovered on the rebuilt block.
    assert rebuilt.b_dynamic is True
    assert rebuilt.c_dynamic is True
    assert rebuilt.kp_dynamic is True
    assert rebuilt.kff_dynamic is True


# --------------------------------------------------------------------- #
# 6. Saturation-limit round-trip preserves None semantics.
# --------------------------------------------------------------------- #


def test_output_limits_none_round_trip():
    pid = PIDController2DOF(dt=0.1, output_min=None, output_max=None)
    data = pid.to_dict()
    assert data["output_min"] is None
    assert data["output_max"] is None
    rebuilt = PIDController2DOF.from_dict(data)
    assert rebuilt.to_dict()["output_min"] is None
    assert rebuilt.to_dict()["output_max"] is None


def test_output_limits_partial_round_trip():
    # Only one of the two limits set.
    pid = PIDController2DOF(dt=0.1, output_min=None, output_max=3.0)
    data = pid.to_dict()
    assert data["output_min"] is None
    assert data["output_max"] == pytest.approx(3.0)
    rebuilt = PIDController2DOF.from_dict(data)
    assert rebuilt.to_dict() == data
