#!/bin/env pytest
# Copyright (C) 2025 Collimator, Inc
# SPDX-License-Identifier: MIT

import pytest
import collimator.testing as test

pytestmark = pytest.mark.app


def test_Sindy_pretrained_from_ui(request):
    # test not failing is considered a "pass"
    test.run(
        pytest_request=request,
        stop_time=0.1,
        model_json="model_pretrained.json",
        check_only=True,
    )
