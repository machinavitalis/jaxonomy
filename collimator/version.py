# Copyright (C) 2025 Collimator, Inc
# SPDX-License-Identifier: MIT

"""collimator version information."""

# Keep in sync with pyproject.toml
# Carefully bump version number based on changes:
# add .alphaN to prepublish alpha releases
# minor update for major new features
# major update shouldn't happen for now
# TODO: better respect semver (re: breaking api changes)
__version__ = "2.2.0"
