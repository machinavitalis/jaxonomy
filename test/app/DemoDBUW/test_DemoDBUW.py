#!/bin/env pytest
# Copyright (C) 2025 Collimator, Inc
# SPDX-License-Identifier: MIT

import pytest
import collimator.testing as test

pytestmark = pytest.mark.app


# @pytest.mark.skip(reason="")
def test_CruiseControl(request):
    test.run(pytest_request=request, stop_time=0.1, model_json="cruise_control.json")


def test_KrsticESC(request):
    test.run(pytest_request=request, stop_time=0.1, model_json="KrsticESC.json")


def test_lorenz(request):
    test.run(pytest_request=request, stop_time=0.1, model_json="lorenz.json")


def test_lotka(request):
    test.run(pytest_request=request, stop_time=0.1, model_json="lotka.json")


def test_pendulum_step(request):
    test.run(pytest_request=request, stop_time=0.1, model_json="pendulum_step.json")


def test_simpleESC(request):
    test.run(pytest_request=request, stop_time=0.1, model_json="simpleESC.json")


@pytest.mark.skip(reason="compilation never seems to end")
def test_cartpoleLQG(request):
    test_paths = test.get_paths(request)
    test.copy_to_workdir(test_paths, "cartpole_init_lqg.py")
    test.run(test_paths=test_paths, stop_time=0.1, model_json="cartpoleLQG.json")


@pytest.mark.xfail(reason="see pallascat results")
def test_cartpoleKF(request):
    test_paths = test.get_paths(request)
    test.copy_to_workdir(test_paths, "cartpole_init.py")
    test.run(test_paths=test_paths, stop_time=0.1, model_json="cartpoleKF.json")
