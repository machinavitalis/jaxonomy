# Copyright (C) 2025 Collimator, Inc
# SPDX-License-Identifier: MIT

import fnmatch
import os


def add_py_init_file(dir: str):
    """add __init__.py file to dir"""
    init_file = os.path.join(dir, "__init__.py")
    if not os.path.exists(init_file):
        with open(init_file, "w") as f:
            f.write("")


def get_ignored_files(directory: str) -> set:
    """Get a set of files that are ignored by the gitignore or collimatorignore file

    Args:
        directory (str): Path to the target directory

    Returns:
        set: set of ignored files
    """

    gitignore_file = os.path.join(directory, ".gitignore")
    collimatorignore_file = os.path.join(directory, ".collimatorignore")

    ignore_patterns = []
    if os.path.exists(collimatorignore_file):
        with open(collimatorignore_file, "r") as f:
            ignore_patterns = [
                line.strip() for line in f if line.strip() and not line.startswith("#")
            ]

    if os.path.exists(gitignore_file):
        with open(gitignore_file, "r") as f:
            ignore_patterns += [
                line.strip() for line in f if line.strip() and not line.startswith("#")
            ]

    ignored_files = set()
    # Walk through the target directory
    for root, dirs, files in os.walk(directory):
        for file in files:
            relative_path = os.path.relpath(os.path.join(root, file), directory)
            # Check if the file matches any ignore pattern
            for pattern in ignore_patterns:
                if pattern.endswith("/"):
                    # It's a directory pattern, ignore if it's in the path
                    if fnmatch.fnmatch(relative_path, f"{pattern}*"):
                        ignored_files.add(relative_path)
                        break
                else:
                    # It's a file pattern
                    if fnmatch.fnmatch(file, pattern) or fnmatch.fnmatch(
                        relative_path, pattern
                    ):
                        ignored_files.add(relative_path)
                        break

    return ignored_files
