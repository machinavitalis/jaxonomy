#!env pytest
# Copyright (C) 2025 Collimator, Inc
# SPDX-License-Identifier: MIT

import pytest
import collimator.testing as test
import time

from collimator.testing.markers import skip_if_not_jax

skip_if_not_jax()
pytestmark = pytest.mark.app


@pytest.mark.timeout(30)
def test_DemoWindTurbine(request):
    test_paths = test.get_paths(request)
    tic = time.perf_counter()
    test.copy_to_workdir(test_paths, "full_load_windfield.csv")
    test.run(test_paths=test_paths, stop_time=1, check_only=True)
    toc = time.perf_counter()
    print(f"exe_time={toc - tic:0.4f}")
