# Copyright (C) 2025 Collimator, Inc
# SPDX-License-Identifier: MIT

from abc import ABC, abstractmethod


class Optimizer(ABC):
    """
    Base class that all optimizers should inherit from.
    """

    @abstractmethod
    def optimize(self):
        pass

    @property
    def metrics(self):
        return {}
