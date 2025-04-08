# Copyright (C) 2025 Collimator, Inc
# SPDX-License-Identifier: MIT

import pytest
from collimator.backend import numpy_api, DEFAULT_BACKEND

import logging
from collimator import logging as collimator_logging


@pytest.fixture(autouse=True)
def configure_logging():
    logger = logging.getLogger()
    level = logger.getEffectiveLevel()
    collimator_logging.set_log_level(level)
    yield


# Make sure we end up with the default backend for the other tests,
# since that's what the rest of the test cases will expect.
@pytest.fixture(autouse=True)
def reset_backend():
    yield
    numpy_api.set_backend(DEFAULT_BACKEND)
