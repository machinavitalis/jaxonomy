# SPDX-License-Identifier: MIT

import importlib.util

# TensorFlow's saved-model executor deadlocks (macOS arm64, TF 2.21 +
# scikit-learn 1.9) if sklearn — an OpenMP user, pulled in by pysindy — is
# imported before tensorflow: two OpenMP runtimes land in the process and
# TF's first kernel execution never returns. Importing tensorflow FIRST
# avoids it. pytest imports every collected test module up front
# (test_sindy.py imports pysindy → sklearn), so when both optional deps are
# installed, import tensorflow here, before collection begins.
# Minimal repro: `python -c "import sklearn, tensorflow as tf; ..."` hangs on
# the first executed TF signature; swap the imports and it works.
if (
    importlib.util.find_spec("tensorflow") is not None
    and importlib.util.find_spec("sklearn") is not None
):
    import tensorflow  # noqa: F401

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
