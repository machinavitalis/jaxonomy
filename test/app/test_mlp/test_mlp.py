# Copyright (C) 2025 Collimator, Inc
# SPDX-License-Identifier: MIT

import os

import pytest

import collimator
import collimator.testing as test
from collimator.library import MLP

pytestmark = pytest.mark.minimal


@pytest.mark.skip("this test seems broken now 2025/03")
def test_mlp(request):
    # create workdir
    test_paths = test.get_paths(request)
    workdir = test_paths["workdir"]
    print(f"workdir: {workdir}")

    # create dummy NN to placehold for 'pretrtained model'
    nn_config = {
        "in_size": 2,
        "out_size": 2,
        "width_size": 2,
        "depth": 2,
        "seed": 0,
    }
    pretrained = MLP(**nn_config, name="pretrained")

    # save NN
    pretrained.serialize(f"{workdir}/pretrained.eqx")

    # copy model to workdir
    model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "model.json"))
    os.system(f"ln -sf {model_path} {workdir}/")

    # FIXME collimator.load_model should know to look for files in workdir
    curdir = os.getcwd()
    try:
        os.chdir(workdir)
        # load/simulate collimator model which references the above saved NN
        model = collimator.load_model(".")
        model.simulate()

    finally:
        os.chdir(curdir)
