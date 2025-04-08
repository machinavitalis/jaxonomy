# Copyright (C) 2025 Collimator, Inc
# SPDX-License-Identifier: MIT

"""
Test for autotuner with constraints in frequency domain. The autotuner tested
here is part of the collimator/optimization suite.
"""

import pytest
import numpy as np
from collimator.library import (
    TransferFunction,
)
from collimator.optimization import AutoTuner
from collimator.testing.markers import requires_jax


def get_plant():
    plant_tf_num = [1.0]
    plant_tf_den = [2.0, 6.0, 4.5, 1.0]
    plant = TransferFunction(plant_tf_num, plant_tf_den)
    return plant


@requires_jax()
@pytest.mark.parametrize(
    "Mt, Ms",
    [
        (100.0, 100.0),
        (1.2, 1.2),
    ],
)
def test_autotuner(Mt, Ms):
    plant = get_plant()
    params_0 = np.array([0.1, 0.1, 0.1])

    tuner = AutoTuner(
        plant,
        n=100,
        sim_time=10.0,
        metric="IAE",
        pid_gains_0=params_0,
        Ms=Ms,
        Mt=Mt,
        method="scipy-slsqp",  # choose scipy method test for all platforms
    )
    _, res = tuner.tune()
    assert res.success


if __name__ == "__main__":
    test_autotuner(100.0, 100.0)
    test_autotuner(1.2, 1.2)
