# SPDX-License-Identifier: MIT

import pytest
from jaxonomy.backend import set_backend, DEFAULT_BACKEND

import logging
from jaxonomy import logging as jaxonomy_logging


@pytest.fixture(autouse=True)
def configure_logging():
    logger = logging.getLogger()
    level = logger.getEffectiveLevel()
    jaxonomy_logging.set_log_level(level)
    yield


# Make sure we end up with the default backend for the other tests,
# since that's what the rest of the test cases will expect.
@pytest.fixture(autouse=True)
def reset_backend():
    yield
    set_backend(DEFAULT_BACKEND)
