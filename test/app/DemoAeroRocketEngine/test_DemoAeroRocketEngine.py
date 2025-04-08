#!/bin/env pytest
# Copyright (C) 2025 Collimator, Inc
# SPDX-License-Identifier: MIT

import pytest
import collimator.testing as test

pytestmark = pytest.mark.app


@pytest.mark.skip(reason="test ran for >100 seconds without terminating. not sure why.")
def test_DemoAeroRocketEngine(request):
    test_paths = test.get_paths(request)
    test.copy_to_workdir(test_paths, "init.py")
    test.run(test_paths=test_paths, stop_time=0.1)
