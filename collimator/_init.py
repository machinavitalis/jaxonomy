# Copyright (C) 2025 Collimator, Inc
# SPDX-License-Identifier: MIT

"""Internal module to properly initialize logging and JAX/x64"""

import os

# Enable x64 by default, see also backend.py
# Note: this enables floats to default to 64-bit but not integers, so there are
# still issues on Windows where int defaults to 32bit but some other calculations
# will yield int64.
# Setting np.int_ = np.int64 globally is a big hack and does not fix it.
os.environ.setdefault("JAX_ENABLE_X64", "true")

# pylint: disable=wrong-import-position
from . import logging  # noqa: E402

_log_level = os.environ.get("LOG_LEVEL", "INFO")
logging.set_log_level(_log_level)
logging.set_stream_handler()

_per_package_log_levels = os.environ.get("LOG_LEVELS", None)
if _per_package_log_levels is not None:
    _per_package_log_levels = _per_package_log_levels.split(",")
    _per_package_log_levels = [level.split(":") for level in _per_package_log_levels]
    for pkg, level in _per_package_log_levels:
        logging.set_log_level(level, pkg=pkg)
