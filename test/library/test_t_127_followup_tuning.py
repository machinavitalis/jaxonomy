# SPDX-License-Identifier: MIT
"""T-127-followup-pid-tuning-helpers smoke tests.

The agent shipped the implementation in primitives.py but ran out of context
before producing tests. These cover the standard tuning formulas and basic
validation.
"""

from __future__ import annotations

import math
import pytest

from jaxonomy.library import PIDController2DOF


class TestZieglerNichols:
    def test_pid_mode_canonical_values(self):
        pid = PIDController2DOF.ziegler_nichols(Ku=4.0, Tu=2.0, dt=0.1, mode="PID")
        # PID: Kp = 0.6*Ku, Ki = 1.2*Ku/Tu, Kd = 0.075*Ku*Tu
        assert math.isclose(float(pid.parameters["kp"].get()), 0.6 * 4.0)
        assert math.isclose(float(pid.parameters["ki"].get()), 1.2 * 4.0 / 2.0)
        assert math.isclose(float(pid.parameters["kd"].get()), 0.075 * 4.0 * 2.0)

    def test_pi_mode(self):
        pid = PIDController2DOF.ziegler_nichols(Ku=4.0, Tu=2.0, dt=0.1, mode="PI")
        assert math.isclose(float(pid.parameters["kp"].get()), 0.45 * 4.0)
        assert math.isclose(float(pid.parameters["ki"].get()), 0.54 * 4.0 / 2.0)
        assert math.isclose(float(pid.parameters["kd"].get()), 0.0)

    def test_p_mode(self):
        pid = PIDController2DOF.ziegler_nichols(Ku=4.0, Tu=2.0, dt=0.1, mode="P")
        assert math.isclose(float(pid.parameters["kp"].get()), 0.5 * 4.0)
        assert math.isclose(float(pid.parameters["ki"].get()), 0.0)
        assert math.isclose(float(pid.parameters["kd"].get()), 0.0)

    def test_invalid_mode_raises(self):
        with pytest.raises(ValueError, match="mode must be one of"):
            PIDController2DOF.ziegler_nichols(Ku=4.0, Tu=2.0, dt=0.1, mode="BAD")

    def test_non_positive_Ku_raises(self):
        with pytest.raises(ValueError, match="Ku must be positive"):
            PIDController2DOF.ziegler_nichols(Ku=0.0, Tu=2.0, dt=0.1)

    def test_non_positive_Tu_raises(self):
        with pytest.raises(ValueError, match="Tu must be positive"):
            PIDController2DOF.ziegler_nichols(Ku=4.0, Tu=-1.0, dt=0.1)


class TestCohenCoon:
    def test_pid_mode_returns_pid(self):
        # FOPDT plant with K=1, tau=1, theta=0.1 → r = 0.1
        pid = PIDController2DOF.cohen_coon(K=1.0, tau=1.0, theta=0.1, dt=0.1, mode="PID")
        # Just verify gains are non-zero and finite per the formulas
        kp = float(pid.parameters["kp"].get())
        ki = float(pid.parameters["ki"].get())
        kd = float(pid.parameters["kd"].get())
        assert kp > 0 and math.isfinite(kp)
        assert ki > 0 and math.isfinite(ki)
        assert kd > 0 and math.isfinite(kd)

    def test_pi_mode_has_zero_kd(self):
        pid = PIDController2DOF.cohen_coon(K=1.0, tau=1.0, theta=0.1, dt=0.1, mode="PI")
        assert math.isclose(float(pid.parameters["kd"].get()), 0.0)

    def test_p_mode_has_zero_ki_kd(self):
        pid = PIDController2DOF.cohen_coon(K=1.0, tau=1.0, theta=0.1, dt=0.1, mode="P")
        assert math.isclose(float(pid.parameters["ki"].get()), 0.0)
        assert math.isclose(float(pid.parameters["kd"].get()), 0.0)

    def test_invalid_mode_raises(self):
        with pytest.raises(ValueError, match="mode must be one of"):
            PIDController2DOF.cohen_coon(K=1.0, tau=1.0, theta=0.1, dt=0.1, mode="BAD")


class TestTyreusLuyben:
    def test_returns_pi_controller(self):
        pid = PIDController2DOF.tyreus_luyben(Ku=4.0, Tu=2.0, dt=0.1)
        # T-L is a PI rule — Kd = 0
        assert math.isclose(float(pid.parameters["kd"].get()), 0.0)
        assert float(pid.parameters["kp"].get()) > 0
        assert float(pid.parameters["ki"].get()) > 0
