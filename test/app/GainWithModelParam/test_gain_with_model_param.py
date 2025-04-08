# Copyright (C) 2025 Collimator, Inc
# SPDX-License-Identifier: MIT

import pytest
import collimator.testing as test

pytestmark = pytest.mark.app


def test_GainWithModelParam(request):
    """This model contains a constant block linked to a gain block
    that has its gain value set to "[0.0, a]" where "a" is a model parameter.
    """
    test.run(
        pytest_request=request,
        stop_time=0.1,
        model_json="model.json",
        check_only=True,
    )
