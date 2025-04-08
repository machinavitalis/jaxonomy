# Copyright (C) 2025 Collimator, Inc
# SPDX-License-Identifier: MIT

from typing import Any

import numpy as np


class RandomDistribution:
    """Represents a random distribution to be used in monte-carlo simulations.
    Must be a valid numpy random distribution.
    """

    def __init__(self, distribution: str, **parameters):
        if distribution not in np.random.__dict__:
            raise ValueError(f"Unknown distribution: {distribution}")
        self.distribution = distribution
        self.parameters = parameters


class SweepValues:
    """Represents a list of values to sweep over."""

    def __init__(self, values: list[Any]):
        self.values = values
