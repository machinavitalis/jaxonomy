#!/bin/env pytest
# Copyright (C) 2025 Collimator, Inc
# SPDX-License-Identifier: MIT

import pytest
import collimator.testing as test

pytestmark = pytest.mark.app


def test_0092_InitScript(request):
    test_paths = test.get_paths(request)
    test.copy_to_workdir(test_paths, "init_script.py")
    test.run(test_paths=test_paths, stop_time=0.1)


def test_InitScript(request):
    test_paths = test.get_paths(request)
    test.copy_to_workdir(test_paths, "cartpole_init_lqg.py")
    test.run(test_paths=test_paths, stop_time=1.0, model_json="model_cartpole.json")
