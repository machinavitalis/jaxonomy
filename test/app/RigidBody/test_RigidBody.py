#!/bin/env pytest
# Copyright (C) 2025 Collimator, Inc
# SPDX-License-Identifier: MIT

import pytest
import collimator.testing as test

pytestmark = pytest.mark.app


def test_RigidBody(request):
    # test ingesting the block with various configurations
    stop_time = 1.0
    test.run(pytest_request=request, stop_time=stop_time, check_only=True)
