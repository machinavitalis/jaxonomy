#!/bin/env pytest
# SPDX-License-Identifier: MIT

import pytest
import jaxonomy.testing as test

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


def test_cartpoleLQG(request):
    test_paths = test.get_paths(request)
    test.copy_to_workdir(test_paths, "cartpole_init_lqg.py")
    test.run(test_paths=test_paths, stop_time=0.1, model_json="cartpoleLQG.json")


def test_cartpoleKF(request):
    test_paths = test.get_paths(request)
    test.copy_to_workdir(test_paths, "cartpole_init.py")
    test.run(test_paths=test_paths, stop_time=0.1, model_json="cartpoleKF.json")
