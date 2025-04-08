#!/bin/env pytest
# Copyright (C) 2025 Collimator, Inc
# SPDX-License-Identifier: MIT

import pytest
import collimator.testing as test

pytestmark = pytest.mark.app
"""
blocks in test_rotations.py are tested in:
test/app/CoordinateRotation/
test/app/CoordinateRotationConversion/
test/app/RigidBody/

FMU tested in:
test/app/ModelicaFMU/

SINDy tested in:
test/app/Sindy/

Predictor tested in:
test/app/test_predictor/

StateMachine tested in:
test/app/StateMachine/
"""


@pytest.mark.parametrize(
    "model_json",
    [
        "test_continuous.json",
        "test_custom.json",
        "test_discontinuities.json",
        "test_discrete.json",
        "test_logic.json",
        "test_lookup_tables.json",
        "test_math.json",
        "test_signal_routing.json",
        "test_source.json",
        "test_sink.json",
        "test_custom_leaf_system.json",
    ],
)
def test_json(request, model_json):
    test.run(pytest_request=request, check_only=True, model_json=model_json)
