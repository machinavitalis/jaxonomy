#!/bin/env pytest
# Copyright (C) 2025 Collimator, Inc
# SPDX-License-Identifier: MIT

import pytest
import sys
import collimator.testing as test

pytestmark = pytest.mark.app

"""
the intend here is just to ensure that wildcat can load/run an FMU
when this op is specified in the model.json
"""


@pytest.mark.skipif(sys.platform != "linux", reason="Only supports linux/x86_64")
def test_ModelicaFMU(request):
    test_paths = test.get_paths(request)
    test.copy_to_workdir(test_paths, "thermal_1.fmu")
    test.run(test_paths=test_paths, stop_time=0.1, check_only=True)


@pytest.mark.skipif(sys.platform != "linux", reason="Only supports linux/x86_64")
def test_fmu_clock(request):
    test_paths = test.get_paths(request)
    test.copy_to_workdir(test_paths, "fmu_clock.fmu")
    test.run(test_paths=test_paths, model_json="fmu_clock.json", check_only=True)
