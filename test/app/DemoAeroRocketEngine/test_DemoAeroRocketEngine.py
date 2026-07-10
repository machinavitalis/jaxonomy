#!/bin/env pytest
# SPDX-License-Identifier: MIT

import pytest
import jaxonomy.testing as test

pytestmark = pytest.mark.app


def test_DemoAeroRocketEngine(request):
    # Was skip-quarantined twice over: a stale-parameter load error (fixed
    # 2026-07-09) masked a genuine hang — eager port evaluation recomputed
    # shared upstream subgraphs once per consumer, exponentially in this
    # model's composition depth (13 submodel instances, 8 nested groups).
    # Fixed by the per-eval-tree memo in framework/cache.py.
    test_paths = test.get_paths(request)
    test.copy_to_workdir(test_paths, "init.py")
    test.run(test_paths=test_paths, stop_time=0.1)
