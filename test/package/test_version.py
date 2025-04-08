# Copyright (C) 2025 Collimator, Inc
# SPDX-License-Identifier: MIT

import os
import toml

from collimator import __version__


def _get_version() -> str:
    dirname = os.path.dirname(os.path.abspath(__file__))
    pyproject_path = os.path.join(dirname, "..", "..", "pyproject.toml")
    with open(pyproject_path, "r", encoding="utf-8") as file:
        pyproject = toml.load(file)
    return pyproject["project"]["version"]


def test_version():
    """Checks that version number matches between pyproject.toml and version.py.
    Also checks that version number is in the correct format.
    """
    version = __version__.split(".")
    assert len(version) == 3 or len(version) == 4

    # Check major and minor
    assert version[0] == "2"
    assert version[1] == "2"

    # Check micro
    assert version[2].isdigit()
    assert len(version) == 3 or version[3].startswith("alpha")

    # Check version in pyproject.toml
    assert (
        _get_version() == __version__
    ), "Version mismatch between pyproject.toml and version.py"
