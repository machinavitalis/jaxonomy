# Copyright (C) 2025 Collimator, Inc
# SPDX-License-Identifier: MIT

"""Generates CSV files for testing purposes."""

import pandas as pd
import numpy as np


def create_test_csv(file_path, include_header=True, rows=5, cols=3):
    """Generates a CSV file for testing."""
    data = np.random.rand(rows, cols)
    if include_header:
        header = [f"col{i}" for i in range(cols)]
    else:
        header = None
    df = pd.DataFrame(data, columns=header)
    df.to_csv(file_path, index=False, header=include_header)


if __name__ == "__main__":
    create_test_csv("with_header.csv", include_header=True, rows=10, cols=5)
    create_test_csv("without_header.csv", include_header=False, rows=10, cols=5)
